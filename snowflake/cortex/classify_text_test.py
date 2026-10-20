import json
import warnings

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _classify_text, _util
from snowflake.snowpark import functions


class ClassifyTextMigrationTest(absltest.TestCase):
    """classify_text targets AI_CLASSIFY (single-label) and re-wraps to the legacy CLASSIFY_TEXT
    {"label": ...} OBJECT. These checks don't require a live AI backend."""

    def setUp(self) -> None:
        # The deprecation warning is emitted once per process; reset so each test observes it.
        _util._emitted_deprecations.clear()

    def test_classify_text_column_builds_single_label_shape(self) -> None:
        result = _classify_text.classify_text(functions.col("review"), ["positive", "negative"])
        assert isinstance(result, snowpark.Column)  # narrows str | Column for mypy and asserts the path
        sql = result._expression.sql.upper()
        self.assertIn("AI_CLASSIFY", sql)
        # re-wrapped to the legacy single-label {"label": ...} object
        self.assertIn("OBJECT_CONSTRUCT", sql)

    def test_classify_text_config_is_a_constant_object_literal(self) -> None:
        # Regression: AI_CLASSIFY requires its config to be a constant. Passing a Python dict renders as
        # PARSE_JSON('{"output_mode": "single"}')::OBJECT, which the server rejects as non-constant. The
        # config must appear as a SQL object literal instead.
        result = _classify_text.classify_text(functions.col("review"), ["positive", "negative"])
        assert isinstance(result, snowpark.Column)  # narrows str | Column for mypy
        sql = result._expression.sql
        self.assertIn("{'output_mode': 'single'}", sql)
        self.assertNotIn("parse_json('{", sql.lower())

    def test_classify_text_emits_single_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _classify_text.classify_text(functions.col("review"), ["positive", "negative"])

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("snowflake.cortex.classify_text", str(deprecations[0].message))

    def test_adapt_ai_classify_result_from_json_string(self) -> None:
        self.assertEqual(
            {"label": "travel"}, json.loads(_classify_text._adapt_ai_classify_result('{"labels": ["travel"]}'))
        )


if __name__ == "__main__":
    absltest.main()
