import dataclasses
import math
import pickle
from unittest import mock

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _evaluation_result


class EvaluationResultTest(absltest.TestCase):
    def test_stores_score_and_details(self) -> None:
        details = mock.create_autospec(snowpark.DataFrame, instance=True)

        result = _evaluation_result.EvaluationResult(
            score=0.75,
            details=details,
        )

        self.assertEqual(result.score, 0.75)
        self.assertIs(result.details, details)

    def test_details_default_to_none(self) -> None:
        result = _evaluation_result.EvaluationResult(score=0.75)

        self.assertIsNone(result.details)

    def test_is_immutable(self) -> None:
        result = _evaluation_result.EvaluationResult(score=0.75)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.details = None  # type: ignore[misc]

        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.score = 1.0  # type: ignore[misc]

    def test_is_explicitly_unhashable(self) -> None:
        result = _evaluation_result.EvaluationResult(score=0.75)

        with self.assertRaisesRegex(TypeError, "unhashable type"):
            hash(result)

    def test_equality_does_not_compare_dataframes(self) -> None:
        first_details = mock.create_autospec(snowpark.DataFrame, instance=True)
        second_details = mock.create_autospec(snowpark.DataFrame, instance=True)

        self.assertEqual(
            _evaluation_result.EvaluationResult(0.75, first_details),
            _evaluation_result.EvaluationResult(0.75, second_details),
        )

    def test_repr_does_not_include_dataframe(self) -> None:
        class ExplodingRepr:
            def __repr__(self) -> str:
                raise AssertionError("repr must not access the DataFrame")

        details = ExplodingRepr()

        result_repr = repr(_evaluation_result.EvaluationResult(0.75, details))  # type: ignore[arg-type]

        self.assertIn("score=0.75", result_repr)
        self.assertNotIn("details", result_repr)

    def test_preserves_nonfinite_score(self) -> None:
        nan_result = _evaluation_result.EvaluationResult(score=math.nan)
        infinity_result = _evaluation_result.EvaluationResult(score=math.inf)

        self.assertTrue(math.isnan(nan_result.score))
        self.assertEqual(infinity_result.score, math.inf)

    def test_is_picklable(self) -> None:
        result = _evaluation_result.EvaluationResult(score=0.75)

        restored = pickle.loads(pickle.dumps(result))

        self.assertEqual(restored, result)


if __name__ == "__main__":
    absltest.main()
