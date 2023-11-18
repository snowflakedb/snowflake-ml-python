import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _extract_answer
from snowflake.snowpark import functions, types


class ExtractAnswerTest(absltest.TestCase):
    from_text = "|from_text|"
    question = "|question|"

    @staticmethod
    def extract_answer_for_test(from_text: str, question: str) -> str:
        return f"answered: {from_text}, {question}"

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.extract_answer_for_test,
            name="extract_answer",
            return_type=types.StringType(),
            input_types=[types.StringType(), types.StringType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function extract_answer(string,string)").collect()
        self._session.close()

    def test_embed_text_str(self) -> None:
        res = _extract_answer._extract_answer_impl("extract_answer", self.from_text, self.question)

        # UDFs output non-integer / string values as JSON.
        assert isinstance(res, str)
        self.assertEqual(self.extract_answer_for_test(self.from_text, self.question), res)

    def test_embed_text_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(from_text=self.from_text, question=self.question)])
        df_out = df_in.select(
            _extract_answer._extract_answer_impl(
                "extract_answer", functions.col("from_text"), functions.col("question")
            )
        )
        res = df_out.collect()[0][0]

        # UDFs output non-integer / string values as JSON.
        assert isinstance(res, str)
        self.assertEqual(self.extract_answer_for_test(self.from_text, self.question), res)


if __name__ == "__main__":
    absltest.main()
