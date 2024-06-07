import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _sentiment
from snowflake.snowpark import functions, types


class SentimentTest(absltest.TestCase):
    sentiment = "0.53"

    @staticmethod
    def sentiment_for_test(sentiment: str) -> float:
        return float(sentiment)

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.sentiment_for_test,
            name="sentiment",
            session=self._session,
            return_type=types.FloatType(),
            input_types=[types.FloatType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function sentiment(float)").collect()
        self._session.close()

    def test_sentiment_str(self) -> None:
        res = _sentiment._sentiment_impl("sentiment", self.sentiment, session=self._session)
        self.assertEqual(self.sentiment_for_test(self.sentiment), res)

    def test_sentiment_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(prompt=self.sentiment)])
        df_out = df_in.select(_sentiment._sentiment_impl("sentiment", functions.col("prompt")))
        res = df_out.collect()[0][0]
        self.assertEqual(self.sentiment_for_test(self.sentiment), res)


if __name__ == "__main__":
    absltest.main()
