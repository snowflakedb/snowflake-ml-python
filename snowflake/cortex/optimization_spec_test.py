from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.cortex import _evaluation_spec, _optimization_spec
from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.dataset import dataset


class OptimizationSpecificationTest(absltest.TestCase):
    def test_builds_optimization_specification(self) -> None:
        specification = self._specification()

        self.assertEqual(
            specification.to_spec_dict(),
            {
                "function": {
                    "function_body": "AI_COMPLETE('claude-haiku-4-5', PROMPT('Classify: {0}', REVIEW))",
                },
                "metrics": [{"name": "exact_match"}],
                "dataset": {
                    "name": "DB.SCHEMA.TRAIN_DATA",
                    "version": "v1",
                    "column_mapping": {
                        "argument_mapping": {"$1": "REVIEW"},
                        "ground_truth": "EXPECTED_LABEL",
                    },
                },
                "optimization": {"models": ["claude-haiku-4-5"]},
            },
        )

    def test_uses_candidate_models(self) -> None:
        specification = self._specification(
            models=["claude-haiku-4-5", "openai-gpt-5-nano"],
            budget="medium",
        )

        optimization = specification.to_spec_dict()["optimization"]

        self.assertEqual(optimization["models"], ["claude-haiku-4-5", "openai-gpt-5-nano"])
        self.assertEqual(optimization["budget"], "medium")

    def test_includes_optional_optimization_controls(self) -> None:
        specification = self._specification(
            reflection_model="claude-sonnet-4-5",
            strategy="quality_first",
            algorithm="CoCoEvolve",
        )

        optimization = specification.to_spec_dict()["optimization"]

        self.assertEqual(optimization["reflection_model"], "claude-sonnet-4-5")
        self.assertEqual(optimization["strategy"], "quality_first")
        self.assertEqual(optimization["algorithm"], "CoCoEvolve")

    def test_maps_multiple_inputs_by_position(self) -> None:
        specification = self._specification(prompt="Classify: {0} {1}", inputs=["subject", "body"])

        argument_mapping = specification.to_spec_dict()["dataset"]["column_mapping"]["argument_mapping"]

        self.assertEqual(argument_mapping, {"$1": "SUBJECT", "$2": "BODY"})

    def test_argument_mapping_keys_are_contiguous_and_one_based(self) -> None:
        specification = self._specification(prompt="{0} {1} {2}", inputs=["first", "second", "third"])

        argument_mapping = specification.to_spec_dict()["dataset"]["column_mapping"]["argument_mapping"]

        self.assertEqual(list(argument_mapping), ["$1", "$2", "$3"])

    def test_requires_input_column(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires at least one input column"):
            self._specification(inputs=[]).to_spec_dict()

    def test_requires_ground_truth_column(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires a ground-truth column"):
            self._specification(label_col="").to_spec_dict()

    def test_rejects_invalid_input_column_with_public_exception(self) -> None:
        with self.assertRaisesRegex(
            snowml_exceptions.SnowflakeMLException,
            "AI function optimization received an invalid input column identifier",
        ):
            self._specification(inputs=["1column"]).to_spec_dict()

    def test_rejects_invalid_ground_truth_column(self) -> None:
        with self.assertRaisesRegex(
            snowml_exceptions.SnowflakeMLException,
            "AI function optimization received an invalid ground-truth column identifier",
        ):
            self._specification(label_col="label, 1 AS answer").to_spec_dict()

    def test_includes_holdout_dataset(self) -> None:
        specification = self._specification(holdout_data=self._dataset("DB.SCHEMA.HOLDOUT_DATA", "v2"))

        dataset_specification = specification.to_spec_dict()["dataset"]

        self.assertEqual(dataset_specification["holdout_data"], "DB.SCHEMA.HOLDOUT_DATA")
        self.assertEqual(dataset_specification["holdout_version"], "v2")

    def test_allows_trainset_as_holdout_dataset(self) -> None:
        trainset = self._dataset("DB.SCHEMA.TRAIN_DATA", "v1")
        specification = self._specification(trainset=trainset, holdout_data=trainset)

        dataset_specification = specification.to_spec_dict()["dataset"]

        self.assertEqual(dataset_specification["holdout_data"], "DB.SCHEMA.TRAIN_DATA")
        self.assertEqual(dataset_specification["holdout_version"], "v1")

    def test_preserves_duplicate_candidate_models(self) -> None:
        specification = self._specification(models=["claude-haiku-4-5", "claude-haiku-4-5"])

        self.assertEqual(
            specification.to_spec_dict()["optimization"]["models"],
            ["claude-haiku-4-5", "claude-haiku-4-5"],
        )

    def test_rejects_empty_candidate_models(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires at least one candidate model"):
            self._specification(models=[]).to_spec_dict()

    def test_rejects_unsupported_budget(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "Unsupported optimization budget"):
            self._specification(budget="unlimited").to_spec_dict()

    def test_accepts_auto_budget(self) -> None:
        self.assertEqual(self._specification(budget="auto").to_spec_dict()["optimization"]["budget"], "auto")

    def test_rejects_empty_reflection_model(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "reflection model must not be empty"):
            self._specification(reflection_model="").to_spec_dict()

    def test_rejects_unsupported_strategy(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "Unsupported optimization strategy"):
            self._specification(strategy="fastest").to_spec_dict()

    def test_rejects_unsupported_algorithm(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "Unsupported optimization algorithm"):
            self._specification(algorithm="random_search").to_spec_dict()

    def test_requires_version_selected_dataset(self) -> None:
        trainset = mock.create_autospec(dataset.Dataset, instance=True)
        trainset.selected_version = None

        with self.assertRaisesRegex(
            snowml_exceptions.SnowflakeMLException,
            "AI function optimization requires a Dataset with a selected version",
        ):
            self._specification(trainset=trainset).to_spec_dict()

    def _specification(self, **overrides: object) -> _optimization_spec.OptimizationSpecification:
        values: dict[str, object] = {
            "model": "claude-haiku-4-5",
            "prompt": "Classify: {0}",
            "inputs": ["review"],
            "trainset": self._dataset("DB.SCHEMA.TRAIN_DATA", "v1"),
            "label_col": "expected_label",
            "metric": _evaluation_spec.EvaluationMetric("exact_match"),
            "holdout_data": None,
            "models": None,
            "budget": None,
            "reflection_model": None,
            "strategy": None,
            "algorithm": None,
        }
        values.update(overrides)
        return _optimization_spec.OptimizationSpecification(**values)  # type: ignore[arg-type]

    @staticmethod
    def _dataset(name: str, version: str) -> dataset.Dataset:
        input_dataset = mock.create_autospec(dataset.Dataset, instance=True)
        input_dataset.fully_qualified_name = name
        selected_version = mock.create_autospec(dataset.DatasetVersion, instance=True)
        selected_version.name = version
        input_dataset.selected_version = selected_version
        return cast(dataset.Dataset, input_dataset)


if __name__ == "__main__":
    absltest.main()
