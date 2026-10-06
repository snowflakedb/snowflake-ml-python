import warnings

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _extract_answer, _util
from snowflake.snowpark import functions


class ExtractAnswerMigrationTest(absltest.TestCase):
    """extract_answer targets AI_EXTRACT but reshapes the result back to the legacy EXTRACT_ANSWER
    [{"answer": ..., "score": ...}] ARRAY. These checks don't require a live AI backend."""

    def setUp(self) -> None:
        # The deprecation warning is emitted once per process; reset so each test observes it.
        _util._emitted_deprecations.clear()

    def test_extract_answer_column_builds_legacy_shape(self) -> None:
        result = _extract_answer.extract_answer(functions.col("from_text"), "What is the capital?")
        assert isinstance(result, snowpark.Column)  # narrows str | Column for mypy and asserts the path
        sql = result._expression.sql.upper()
        self.assertIn("AI_EXTRACT", sql)
        # score comes only from the named `scores` argument
        self.assertIn("SCORES => TRUE", sql)
        # reshaped to an array of {"answer", "score"}
        self.assertIn("ARRAY_CONSTRUCT", sql)
        self.assertIn("OBJECT_CONSTRUCT", sql)
        # score lives 4 levels deep (scoring -> scores -> answer -> score); its subfield chain must be present
        self.assertIn("SUBFIELDSTRING(SUBFIELDSTRING(SUBFIELDSTRING(SUBFIELDSTRING(", sql)

    def test_extract_answer_column_question_builds_response_format(self) -> None:
        result = _extract_answer.extract_answer(functions.col("from_text"), functions.col("question"))
        assert isinstance(result, snowpark.Column)  # narrows str | Column for mypy and asserts the path
        self.assertIn("AI_EXTRACT", result._expression.sql.upper())

    def test_extract_answer_emits_single_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _extract_answer.extract_answer(functions.col("from_text"), "What is the capital?")

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("snowflake.cortex.extract_answer", str(deprecations[0].message))


if __name__ == "__main__":
    absltest.main()
