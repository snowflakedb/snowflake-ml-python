from typing import cast
from unittest import mock

import yaml
from absl.testing import absltest

from snowflake.cortex import _evaluation_spec
from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.dataset import dataset


class EvaluationMetricTest(absltest.TestCase):
    def test_builtin_metric(self) -> None:
        metric = _evaluation_spec.EvaluationMetric("exact_match")

        self.assertEqual(metric.to_spec_dict(), {"name": "exact_match"})

    def test_llm_judge_metric(self) -> None:
        metric = _evaluation_spec.EvaluationMetric("llm_judge", judge_model="claude-sonnet-4-5")

        self.assertEqual(
            metric.to_spec_dict(),
            {"name": "llm_judge", "judge_model": "claude-sonnet-4-5"},
        )

    def test_custom_metric(self) -> None:
        metric = _evaluation_spec.EvaluationMetric("custom", custom_udf="DB.SCHEMA.SCORER")

        self.assertEqual(
            metric.to_spec_dict(),
            {"name": "custom", "custom_udf": "DB.SCHEMA.SCORER"},
        )

    def test_custom_metric_requires_udf(self) -> None:
        for custom_udf in (None, ""):
            with self.subTest(custom_udf=custom_udf):
                with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires a custom UDF name"):
                    _evaluation_spec.EvaluationMetric("custom", custom_udf=custom_udf).to_spec_dict()

    def test_judge_model_requires_llm_judge(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "only be used with the llm_judge"):
            _evaluation_spec.EvaluationMetric("exact_match", judge_model="model").to_spec_dict()

    def test_optional_fields_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            _evaluation_spec.EvaluationMetric("custom", None, "DB.SCHEMA.SCORER")  # type: ignore[misc]


class EvaluationSpecificationTest(absltest.TestCase):
    def test_builds_evaluation_specification(self) -> None:
        input_dataset = self._dataset("DB.SCHEMA.EVAL_DATA", "v1")
        specification = _evaluation_spec.EvaluationSpecification(
            model="claude-sonnet-4-5",
            prompt="Classify: {0}",
            inputs=["review"],
            dataset=input_dataset,
            label_col="expected_label",
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        )

        self.assertEqual(
            specification.to_spec_dict(),
            {
                "query_text": "AI_COMPLETE('claude-sonnet-4-5', PROMPT('Classify: {0}', REVIEW))::VARCHAR",
                "metrics": [{"name": "exact_match"}],
                "dataset": {
                    "name": "DB.SCHEMA.EVAL_DATA",
                    "version": "v1",
                    "ground_truth": "EXPECTED_LABEL",
                },
            },
        )

    def test_requires_exactly_one_metric(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt="prompt",
            inputs=[],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col="label",
            metrics=[],
        )

        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires exactly one metric"):
            specification.to_spec_dict()

    def test_requires_ground_truth_column(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt="prompt",
            inputs=[],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col="",
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        )

        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires a ground-truth column"):
            specification.to_spec_dict()

    def test_requires_input_column(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt="prompt",
            inputs=[],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col="label",
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        )

        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires at least one input column"):
            specification.to_spec_dict()

    def test_normalizes_ground_truth_column(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt="prompt {0}",
            inputs=["text"],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col='"expectedLabel"',
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        )

        self.assertEqual(specification.to_spec_dict()["dataset"]["ground_truth"], '"expectedLabel"')

    def test_rejects_invalid_ground_truth_column(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt="prompt {0}",
            inputs=["text"],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col="label, 1 AS answer",
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        )

        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "invalid ground-truth column identifier"):
            specification.to_spec_dict()

    def test_spec_dict_survives_yaml_round_trip(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt=': value # literal\n-leading: {braces} "quoted": {0}',
            inputs=["input_col"],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col="label",
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        ).to_spec_dict()

        serialized = yaml.safe_dump(specification)

        self.assertEqual(yaml.safe_load(serialized), specification)
        self.assertFalse(serialized.lstrip().startswith(("{", "[", "@")))

    def test_label_column_can_also_be_an_input(self) -> None:
        specification = _evaluation_spec.EvaluationSpecification(
            model="model",
            prompt="Answer: {0}",
            inputs=["answer"],
            dataset=self._dataset("DB.SCHEMA.EVAL_DATA", "v1"),
            label_col="answer",
            metrics=[_evaluation_spec.EvaluationMetric("exact_match")],
        )

        self.assertEqual(
            specification.to_spec_dict()["query_text"],
            "AI_COMPLETE('model', PROMPT('Answer: {0}', ANSWER))::VARCHAR",
        )

    @staticmethod
    def _dataset(name: str, version: str) -> dataset.Dataset:
        input_dataset = mock.create_autospec(dataset.Dataset, instance=True)
        input_dataset.fully_qualified_name = name
        selected_version = mock.create_autospec(dataset.DatasetVersion, instance=True)
        selected_version.name = version
        input_dataset.selected_version = selected_version
        return cast(dataset.Dataset, input_dataset)


class ResolveDatasetTest(absltest.TestCase):
    def test_requires_dataset(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "requires a snowflake.ml.Dataset"):
            _evaluation_spec.resolve_dataset("DB.SCHEMA.TABLE")  # type: ignore[arg-type]

    def test_requires_selected_version(self) -> None:
        input_dataset = mock.create_autospec(dataset.Dataset, instance=True)
        input_dataset.selected_version = None

        with self.assertRaisesRegex(
            snowml_exceptions.SnowflakeMLException, "requires a Dataset with a selected version"
        ):
            _evaluation_spec.resolve_dataset(input_dataset)

    def test_uses_operation_context_in_error(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "AI function optimization requires"):
            _evaluation_spec.resolve_dataset("DB.SCHEMA.TABLE", context="optimization")  # type: ignore[arg-type]


class BuildAICompleteQueryTest(absltest.TestCase):
    def test_builds_literal_prompt_without_inputs(self) -> None:
        self.assertEqual(
            _evaluation_spec.build_ai_complete_query("model", "classify this", []),
            "AI_COMPLETE('model', 'classify this')",
        )

    def test_escapes_string_literals(self) -> None:
        self.assertEqual(
            _evaluation_spec.build_ai_complete_query("model's", "it's {0}", ["text"]),
            "AI_COMPLETE('model''s', PROMPT('it''s {0}', TEXT))",
        )

    def test_keeps_adversarial_values_inside_string_literals(self) -> None:
        payloads = ["x', 1) OR 1=1 --", "$$", "path\\to\\value", "line one\nline two"]
        for payload in payloads:
            with self.subTest(payload=payload):
                escaped_payload = payload.replace("\\", "\\\\").replace("'", "''")
                self.assertEqual(
                    _evaluation_spec.build_ai_complete_query(payload, payload, []),
                    f"AI_COMPLETE('{escaped_payload}', '{escaped_payload}')",
                )

    def test_rejects_prompt_placeholders_that_do_not_match_inputs(self) -> None:
        cases = [
            ("Redact PII.", ["text"]),
            ("Answer {0} using {1}", ["question"]),
            ("Answer {0}", ["question", "context"]),
            ("Answer {1}", ["question"]),
            ("Answer {0}", []),
        ]
        for prompt, inputs in cases:
            with self.subTest(prompt=prompt, inputs=inputs):
                with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "do not match"):
                    _evaluation_spec.build_ai_complete_query("model", prompt, inputs)

    def test_allows_repeated_placeholders(self) -> None:
        self.assertEqual(
            _evaluation_spec.build_ai_complete_query("model", "{0}, then {0} again", ["text"]),
            "AI_COMPLETE('model', PROMPT('{0}, then {0} again', TEXT))",
        )

    def test_keeps_duplicate_inputs(self) -> None:
        self.assertEqual(
            _evaluation_spec.build_ai_complete_query("model", "{0} {1}", ["review", "review"]),
            "AI_COMPLETE('model', PROMPT('{0} {1}', REVIEW, REVIEW))",
        )

    def test_normalizes_input_identifiers(self) -> None:
        self.assertEqual(
            _evaluation_spec.build_ai_complete_query("model", "{0} {1}", ['"myCol"', "MyCol"]),
            "AI_COMPLETE('model', PROMPT('{0} {1}', \"myCol\", MYCOL))",
        )

    def test_rejects_invalid_column_identifier(self) -> None:
        with self.assertRaisesRegex(snowml_exceptions.SnowflakeMLException, "invalid input column identifier"):
            _evaluation_spec.build_ai_complete_query("model", "prompt", ["text); DROP TABLE data"])

    def test_uses_operation_context_in_invalid_identifier_error(self) -> None:
        with self.assertRaisesRegex(
            snowml_exceptions.SnowflakeMLException,
            "AI function optimization received an invalid input column identifier",
        ):
            _evaluation_spec.build_ai_complete_query("model", "prompt", ["1column"], context="optimization")


if __name__ == "__main__":
    absltest.main()
