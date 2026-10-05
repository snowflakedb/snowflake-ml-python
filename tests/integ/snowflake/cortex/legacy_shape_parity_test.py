import json
from typing import Any

from absl.testing import absltest

from snowflake.cortex import (
    classify_text,
    embed_text_768,
    extract_answer,
    sentiment,
    summarize,
    translate,
)
from snowflake.ml._internal.utils import connection_params, snowflake_env
from snowflake.snowpark import Session, functions
from tests.integ.snowflake.ml.test_utils import test_env_utils


@absltest.skipUnless(
    test_env_utils.get_current_snowflake_cloud_type() == snowflake_env.SnowflakeCloudType.AWS,
    "AI SQL functions only available in AWS",
)
class LegacyShapeParityTest(absltest.TestCase):
    """Assert the AI_*-backed wrappers return the SAME schema/shape customers received from the legacy
    snowflake.cortex.* SQL functions.

    Only shape is asserted (type + keys), not values: the AI_* functions use different underlying models,
    so generated/classified values legitimately differ. The expected shapes are the documented legacy
    contracts, so these checks stay valid after the legacy SQL functions are removed.
    """

    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def _collect(self, column: Any) -> Any:
        # Accepts the wrappers' str | float | list | Column return union; at runtime it is always a Column
        # because every caller passes a Column input.
        value = self._session.range(1).select(column.alias("out")).collect()[0]["OUT"]
        # OBJECT/ARRAY/VARIANT columns are collected as JSON strings; decode so we can inspect the shape.
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def test_classify_text_shape(self) -> None:
        # Legacy CLASSIFY_TEXT: single-label OBJECT {"label": <str>}. Use an input that matches both
        # categories to confirm the output stays single-label (not AI_CLASSIFY's {"labels": [...]}).
        result = self._collect(
            classify_text(functions.lit("I love this product but hate the price"), ["positive", "negative"])
        )
        self.assertIsInstance(result, dict)
        self.assertIn("label", result)
        self.assertIsInstance(result["label"], str)
        self.assertNotIn("labels", result)

    def test_extract_answer_shape(self) -> None:
        # Legacy EXTRACT_ANSWER: ARRAY of one {"answer": <str>, "score": <float>}.
        result = self._collect(extract_answer(functions.lit("The capital of France is Paris."), "What is the capital?"))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("answer", result[0])
        self.assertIsInstance(result[0]["answer"], str)
        self.assertIn("score", result[0])
        self.assertIsInstance(result[0]["score"], float)

    def test_sentiment_shape(self) -> None:
        # Legacy SENTIMENT: FLOAT in [-1, 1].
        result = self._collect(sentiment(functions.lit("I love this product")))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_translate_shape(self) -> None:
        # Legacy TRANSLATE: a string.
        result = self._collect(translate(functions.lit("Hello world"), "en", "fr"))
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_summarize_shape(self) -> None:
        # Legacy SUMMARIZE: a string.
        result = self._collect(
            summarize(functions.lit("The quick brown fox jumps over the lazy dog. It was a sunny day in the forest."))
        )
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_embed_text_768_shape(self) -> None:
        # Legacy EMBED_TEXT_768: VECTOR(FLOAT, 768), collected as a list of floats.
        result = self._collect(embed_text_768(functions.lit("snowflake-arctic-embed-m"), functions.lit("hello")))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 768)
        self.assertIsInstance(result[0], float)


if __name__ == "__main__":
    absltest.main()
