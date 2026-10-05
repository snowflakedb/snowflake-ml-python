import warnings

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _sentiment, _util
from snowflake.snowpark import functions


class SentimentMigrationTest(absltest.TestCase):
    """AI_SENTIMENT returns a categorical label, so sentiment() is retargeted to AI_COMPLETE to keep
    the legacy FLOAT-in-[-1, 1] contract. These checks don't require a live AI backend."""

    def setUp(self) -> None:
        # The deprecation warning is emitted once per process; reset so each test observes it.
        _util._emitted_deprecations.clear()

    def test_sentiment_column_builds_clamped_ai_complete_expression(self) -> None:
        result = _sentiment.sentiment(functions.col("review"))
        assert isinstance(result, snowpark.Column)  # narrows str | Column for mypy and asserts the path
        sql = result._expression.sql.upper()
        self.assertIn("AI_COMPLETE", sql)
        self.assertIn("PARSE_JSON", sql)
        # Clamped to the legacy [-1, 1] range.
        self.assertIn("LEAST", sql)
        self.assertIn("GREATEST", sql)

    def test_sentiment_emits_single_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _sentiment.sentiment(functions.col("review"))

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("snowflake.cortex.sentiment", str(deprecations[0].message))


if __name__ == "__main__":
    absltest.main()
