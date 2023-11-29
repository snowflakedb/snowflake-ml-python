import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _sentiment
from snowflake.snowpark import functions, types


class SentimentTest(absltest.TestCase):
    prompt = "|prompt|"

    @staticmethod
    def sentiment_for_test(prompt: str) -> str:
        return f"result: {prompt}"

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.sentiment_for_test,
            name="sentiment",
            return_type=types.StringType(),
            input_types=[types.StringType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function sentiment(string)").collect()
        self._session.close()

    def test_sentiment_str(self) -> None:
        res = _sentiment._sentiment_impl("sentiment", self.prompt)
        self.assertEqual(self.sentiment_for_test(self.prompt), res)

    def test_sentiment_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(prompt=self.prompt)])
        df_out = df_in.select(_sentiment._sentiment_impl("sentiment", functions.col("prompt")))
        res = df_out.collect()[0][0]
        self.assertEqual(self.sentiment_for_test(self.prompt), res)


if __name__ == "__main__":
    absltest.main()
