import json
from unittest import mock

import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _classify_text, _util
from snowflake.snowpark import functions, types


class ClassifyTextTest(absltest.TestCase):
    input = "|input|"
    categories = ["category1", "category2", "category3"]

    @staticmethod
    def classify_stub(_input: str, categories: list[str]) -> str:
        return f"prediction: {categories[-1]}"

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.classify_stub,
            name="classify_text",
            return_type=types.StringType(),
            input_types=[types.StringType(), types.ArrayType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function classify_text(string, array)").collect()
        self._session.close()

    def test_classify_str(self) -> None:
        res = _classify_text._classify_text_impl("classify_text", self.input, self.categories)
        self.assertEqual(self.classify_stub(self.input, self.categories), res)

    def test_classify_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(input=self.input, categories=self.categories)])
        df_out = df_in.select(
            _classify_text._classify_text_impl("classify_text", functions.col("input"), functions.col("categories"))
        )
        res = df_out.collect()[0][0]
        self.assertEqual(self.classify_stub(self.input, self.categories), res)


class ClassifyTextMigrationTest(absltest.TestCase):
    def setUp(self) -> None:
        # The deprecation warning is emitted once per process; reset so each test observes it.
        _util._emitted_deprecations.clear()

    @mock.patch("snowflake.cortex._classify_text.call_sql_function", autospec=True)
    def test_classify_text_retargets_and_adapts_ai_classify(self, mock_call_sql_function: mock.Mock) -> None:
        mock_call_sql_function.return_value = {"labels": ["travel"]}
        result = _classify_text.classify_text("go abroad", ["travel", "cooking"], session=None)
        # Single-label mode is requested explicitly to match legacy CLASSIFY_TEXT.
        mock_call_sql_function.assert_called_once_with(
            "AI_CLASSIFY", None, "go abroad", ["travel", "cooking"], {"output_mode": "single"}
        )
        # Legacy CLASSIFY_TEXT shape preserved: a {"label": ...} object (JSON string on the immediate path).
        assert isinstance(result, str)  # immediate path returns the JSON string; narrows str | Column for mypy
        self.assertEqual({"label": "travel"}, json.loads(result))

    @mock.patch("snowflake.cortex._classify_text.call_sql_function", autospec=True)
    def test_classify_text_impl_returns_raw_result(self, mock_call_sql_function: mock.Mock) -> None:
        raw_result = {"labels": ["travel"]}
        mock_call_sql_function.return_value = raw_result

        result = _classify_text._classify_text_impl("AI_CLASSIFY", "go abroad", ["travel", "cooking"])

        self.assertIs(raw_result, result)

    def test_adapt_ai_classify_result_from_json_string(self) -> None:
        self.assertEqual(
            {"label": "travel"}, json.loads(_classify_text._adapt_ai_classify_result('{"labels": ["travel"]}'))
        )


if __name__ == "__main__":
    absltest.main()
