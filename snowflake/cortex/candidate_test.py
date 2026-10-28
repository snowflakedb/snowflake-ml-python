import dataclasses
from typing import Any, cast

from absl.testing import absltest

from snowflake.cortex import _candidate


class CandidateTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.candidate = _candidate.Candidate(
            model="llama3.1-70b",
            body="AI_COMPLETE('llama3.1-70b', PROMPT('Answer {0}', QUESTION))",
            run_name="ITER_3",
            val_score=0.88,
            test_score=0.91,
            relative_cost=0.72,
            run_type="accepted",
            parent_candidate="SEED",
        )

    def test_properties(self) -> None:
        self.assertEqual(self.candidate.model, "llama3.1-70b")
        self.assertEqual(
            self.candidate.body,
            "AI_COMPLETE('llama3.1-70b', PROMPT('Answer {0}', QUESTION))",
        )
        self.assertEqual(self.candidate.run_name, "ITER_3")
        self.assertEqual(self.candidate.val_score, 0.88)
        self.assertEqual(self.candidate.test_score, 0.91)
        self.assertEqual(self.candidate.relative_cost, 0.72)
        self.assertEqual(self.candidate.run_type, "accepted")
        self.assertEqual(self.candidate.parent_candidate, "SEED")

    def test_unrecorded_fields_default_to_none(self) -> None:
        candidate = _candidate.Candidate(
            model="llama3.1-70b",
            body="AI_COMPLETE('llama3.1-70b', 'Answer')",
            run_name="ITER_1",
        )

        self.assertIsNone(candidate.val_score)
        self.assertIsNone(candidate.test_score)
        self.assertIsNone(candidate.relative_cost)
        self.assertIsNone(candidate.run_type)
        self.assertIsNone(candidate.parent_candidate)

    def test_all_fields_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            cast(Any, _candidate.Candidate)(
                "llama3.1-70b",
                "AI_COMPLETE('llama3.1-70b', 'Answer')",
                "SEED",
            )

    def test_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.candidate.val_score = 1.0  # type: ignore[misc]

    def test_from_experiment_run_reads_recorded_metrics_and_parameters(self) -> None:
        candidate = _candidate.from_experiment_run(
            "ITER_1",
            {"test_score": "0.9", "val_score": "0.8", "cost_compared_to_seed": "0.6"},
            {
                "model": "model-a",
                "function_impl": "optimized-body",
                "run_type": "accepted",
                "parent_candidate": "SEED",
            },
            "EXPERIMENT",
        )

        self.assertEqual(
            candidate,
            _candidate.Candidate(
                model="model-a",
                body="optimized-body",
                run_name="ITER_1",
                val_score=0.8,
                test_score=0.9,
                relative_cost=0.6,
                run_type="accepted",
                parent_candidate="SEED",
            ),
        )

    def test_from_experiment_run_leaves_unreported_values_as_none(self) -> None:
        candidate = _candidate.from_experiment_run(
            "ITER_2", {}, {"model": "model-a", "function_impl": "rejected-body"}, "EXPERIMENT"
        )

        self.assertEqual((candidate.model, candidate.body), ("model-a", "rejected-body"))
        self.assertIsNone(candidate.val_score)
        self.assertIsNone(candidate.test_score)
        self.assertIsNone(candidate.relative_cost)
        self.assertIsNone(candidate.run_type)
        self.assertIsNone(candidate.parent_candidate)

    def test_from_experiment_run_rejects_invalid_metrics(self) -> None:
        parameters = {"model": "model-a", "function_impl": "optimized-body"}

        for metrics, expected_message in (
            ({"val_score": "nan", "cost_compared_to_seed": "0.6"}, "invalid val_score value 'nan'"),
            ({"val_score": "0.8", "cost_compared_to_seed": "inf"}, "invalid cost_compared_to_seed value 'inf'"),
            ({"val_score": "high", "cost_compared_to_seed": "0.6"}, "invalid val_score value 'high'"),
        ):
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(RuntimeError, f"{expected_message} for run 'ITER_1'"):
                    _candidate.from_experiment_run("ITER_1", metrics, parameters, "EXPERIMENT")

    def test_from_experiment_run_requires_model_and_body(self) -> None:
        for parameters in (
            {"function_impl": "optimized-body"},
            {"model": "model-a"},
            {"model": "", "function_impl": "b"},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(
                    _candidate.IncompleteCandidateError, "returned an incomplete candidate 'ITER_1'"
                ):
                    _candidate.from_experiment_run("ITER_1", {"val_score": "0.8"}, parameters, "EXPERIMENT")


if __name__ == "__main__":
    absltest.main()
