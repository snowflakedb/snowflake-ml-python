import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _classify_text
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


if __name__ == "__main__":
    absltest.main()
