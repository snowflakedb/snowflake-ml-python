from unittest import mock

import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _embed_text_768
from snowflake.snowpark import functions, types


class EmbedTest768Test(absltest.TestCase):
    model = "snowflake-arctic-embed-m"
    text = "|text|"

    @staticmethod
    def embed_text_768_for_test(model: str, text: str) -> list[float]:
        return [0.0] * 768

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.embed_text_768_for_test,
            name="embed_text_768",
            session=self._session,
            return_type=types.VectorType(float, 768),
            input_types=[types.StringType(), types.StringType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function embed_text_768(string,string)").collect()
        self._session.close()

    def test_embed_text_768_str(self) -> None:
        res = _embed_text_768._embed_text_768_impl(
            "embed_text_768",
            self.model,
            self.text,
            session=self._session,
        )
        out = self.embed_text_768_for_test(self.model, self.text)
        self.assertEqual(out, res, f"Expected ({type(out)}) {out}, got ({type(res)}) {res}")

    def test_embed_text_768_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model=self.model, text=self.text)])
        df_out = df_in.select(
            _embed_text_768._embed_text_768_impl(
                "embed_text_768",
                functions.col("model"),
                functions.col("text"),
                session=self._session,
            )
        )
        res = df_out.collect()[0][0]
        out = self.embed_text_768_for_test(self.model, self.text)

        self.assertEqual(out, res)

    @mock.patch("snowflake.cortex._embed_text_768.call_sql_function", autospec=True)
    def test_embed_text_768_retargets_to_ai_embed(self, mock_call_sql_function: mock.Mock) -> None:
        mock_call_sql_function.return_value = [0.0] * 768
        result = _embed_text_768.embed_text_768("e5-base-v2", "hello", session=None)
        mock_call_sql_function.assert_called_once_with("AI_EMBED", None, "e5-base-v2", "hello")
        self.assertEqual([0.0] * 768, result)


if __name__ == "__main__":
    absltest.main()
