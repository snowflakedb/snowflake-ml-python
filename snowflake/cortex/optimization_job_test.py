import json
import pickle
from unittest import mock

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import (
    _candidate,
    _experiment_job,
    _optimization_job,
    _optimization_result,
)
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment._client import experiment_tracking_sql_client


def _attribute(run_name: str, name: str, value: object) -> snowpark.Row:
    return snowpark.Row(run_name=run_name, name=name, value=value)


class OptimizationJobTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = mock.create_autospec(snowpark.Session, instance=True)
        self.client = mock.create_autospec(
            experiment_tracking_sql_client.ExperimentTrackingSQLClient,
            instance=True,
        )
        self.experiment_name = sql_identifier.SqlIdentifier("EXPERIMENT")
        self.client.describe_experiment.return_value = {"execution_status": "COMPLETED", "summary": None}
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.EXPERIMENT"

    def _create_job(self, experiment: str = "DB.SCHEMA.EXPERIMENT") -> _optimization_job.OptimizationJob:
        with mock.patch.object(
            _experiment_job,
            "resolve_client",
            return_value=(self.client, self.experiment_name),
        ):
            return _optimization_job.OptimizationJob(self.session, experiment)

    def _set_complete_candidates(self) -> None:
        self.client.show_runs_in_experiment.return_value = [
            snowpark.Row(name="SEED"),
            snowpark.Row(name="ITER_1"),
            snowpark.Row(name="ITER_2"),
        ]
        self.client.show_run_metrics_in_experiment.return_value = [
            _attribute("SEED", "val_score", "0.5"),
            _attribute("SEED", "cost_compared_to_seed", "1.0"),
            _attribute("ITER_1", "val_score", "0.8"),
            _attribute("ITER_1", "cost_compared_to_seed", "0.6"),
            _attribute("ITER_1", "is_frontier", "1"),
            _attribute("ITER_2", "val_score", "0.7"),
            _attribute("ITER_2", "cost_compared_to_seed", "0.4"),
            _attribute("ITER_2", "is_frontier", "1"),
        ]
        self.client.show_run_parameters_in_experiment.return_value = [
            _attribute("SEED", "model", "seed-model"),
            _attribute("SEED", "function_impl", "seed-body"),
            _attribute("ITER_1", "model", "model-a"),
            _attribute("ITER_1", "function_impl", "body-a"),
            _attribute("ITER_2", "model", "model-b"),
            _attribute("ITER_2", "function_impl", "body-b"),
        ]

    def test_status_uses_shared_job_status(self) -> None:
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.COMPLETED)

    def test_result_loads_frontier_and_baseline(self) -> None:
        self._set_complete_candidates()
        job = self._create_job()

        result = job.result()

        self.assertEqual([candidate.run_name for candidate in result.frontier], ["ITER_2", "ITER_1"])
        baseline = result.baseline
        assert baseline is not None
        self.assertEqual(baseline.run_name, "SEED")
        self.assertEqual(result.frontier[0].val_score, 0.7)
        self.assertIsNone(result.frontier[0].test_score)
        self.assertEqual(result.frontier[0].relative_cost, 0.4)
        self.assertEqual([candidate.run_name for candidate in result.candidates], ["SEED", "ITER_1", "ITER_2"])

    def test_summary_frontier_takes_precedence_over_run_metric(self) -> None:
        self._set_complete_candidates()
        self.client.describe_experiment.side_effect = [
            {"execution_status": "COMPLETED"},
            {"summary": json.dumps({"pareto_frontier_runs": ["ITER_1"]})},
        ]
        job = self._create_job()

        result = job.result()

        self.assertEqual([candidate.run_name for candidate in result.frontier], ["ITER_1"])

    def test_result_surfaces_summary_error(self) -> None:
        self.client.describe_experiment.side_effect = [
            {"execution_status": "COMPLETED"},
            {"summary": json.dumps({"error_message": "backend failure"})},
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "failed: backend failure"):
            job.result()

    def test_frontier_uses_validation_scores_when_test_score_is_missing(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_metrics_in_experiment.return_value.extend(
            [
                _attribute("ITER_1", "test_score", "0.9"),
            ]
        )
        job = self._create_job()

        result = job.result()

        self.assertEqual(result.selection_metric, "val_score")
        self.assertEqual([candidate.val_score for candidate in result.frontier], [0.7, 0.8])
        self.assertEqual([candidate.test_score for candidate in result.frontier], [None, 0.9])

    def test_baseline_uses_frontier_selection_metric(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_metrics_in_experiment.return_value.extend(
            [
                _attribute("SEED", "test_score", "0.4"),
                _attribute("ITER_1", "test_score", "0.9"),
            ]
        )
        job = self._create_job()

        result = job.result()

        baseline = result.baseline
        assert baseline is not None
        self.assertEqual(result.selection_metric, "val_score")
        self.assertEqual((baseline.val_score, baseline.test_score), (0.5, 0.4))

    def test_uses_test_scores_when_frontier_and_baseline_all_have_them(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_metrics_in_experiment.return_value.extend(
            [
                _attribute("SEED", "test_score", "0.4"),
                _attribute("ITER_1", "test_score", "0.9"),
                _attribute("ITER_2", "test_score", "0.6"),
            ]
        )
        job = self._create_job()

        result = job.result()

        baseline = result.baseline
        assert baseline is not None
        self.assertEqual(result.selection_metric, "test_score")
        self.assertEqual(baseline.test_score, 0.4)
        self.assertEqual([candidate.test_score for candidate in result.frontier], [0.6, 0.9])
        self.assertEqual(result.best_quality().run_name, "ITER_1")

    def test_accepts_boolean_frontier_markers(self) -> None:
        self._set_complete_candidates()
        metrics = [
            row for row in self.client.show_run_metrics_in_experiment.return_value if row["name"] != "is_frontier"
        ]
        self.client.show_run_metrics_in_experiment.return_value = metrics + [
            _attribute("ITER_1", "is_frontier", "true"),
            _attribute("ITER_2", "is_frontier", 1.0),
            _attribute("SEED", "is_frontier", "FALSE"),
        ]
        job = self._create_job()

        result = job.result()

        self.assertEqual([candidate.run_name for candidate in result.frontier], ["ITER_2", "ITER_1"])

    def test_rejects_unrecognized_frontier_marker(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_metrics_in_experiment.return_value.append(_attribute("SEED", "is_frontier", "maybe"))
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "unrecognized is_frontier value 'maybe' for run 'SEED'"):
            job.result()

    def test_explicit_empty_summary_frontier_is_not_overridden(self) -> None:
        self._set_complete_candidates()
        self.client.describe_experiment.side_effect = [
            {"execution_status": "COMPLETED"},
            {"summary": json.dumps({"pareto_frontier_runs": []})},
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "did not identify any candidates"):
            job.result()

    def test_rejects_invalid_frontier_metric(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_metrics_in_experiment.return_value.append(_attribute("ITER_1", "val_score", "nan"))
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "invalid val_score value 'nan' for run 'ITER_1'"):
            job.result()

    def test_candidates_include_rejected_runs_without_scores(self) -> None:
        self._set_complete_candidates()
        self.client.show_runs_in_experiment.return_value.insert(0, snowpark.Row(name="ITER_3"))
        self.client.show_run_parameters_in_experiment.return_value.extend(
            [
                _attribute("ITER_3", "model", "model-c"),
                _attribute("ITER_3", "function_impl", "body-c"),
                _attribute("ITER_3", "run_type", "rejected"),
                _attribute("ITER_3", "parent_candidate", "ITER_1"),
            ]
        )
        job = self._create_job()

        result = job.result()

        self.assertEqual([candidate.run_name for candidate in result.frontier], ["ITER_2", "ITER_1"])
        self.assertEqual(
            [candidate.run_name for candidate in result.candidates], ["SEED", "ITER_1", "ITER_2", "ITER_3"]
        )
        rejected = result.candidates[-1]
        self.assertEqual((rejected.model, rejected.body), ("model-c", "body-c"))
        self.assertEqual((rejected.run_type, rejected.parent_candidate), ("rejected", "ITER_1"))
        self.assertIsNone(rejected.val_score)
        self.assertIsNone(rejected.test_score)
        self.assertIsNone(rejected.relative_cost)

    def test_candidates_order_iterations_numerically(self) -> None:
        self._set_complete_candidates()
        self.client.show_runs_in_experiment.return_value.append(snowpark.Row(name="ITER_10"))
        self.client.show_run_parameters_in_experiment.return_value.extend(
            [_attribute("ITER_10", "model", "model-d"), _attribute("ITER_10", "function_impl", "body-d")]
        )
        job = self._create_job()

        result = job.result()

        self.assertEqual(
            [candidate.run_name for candidate in result.candidates], ["SEED", "ITER_1", "ITER_2", "ITER_10"]
        )

    def test_leaves_out_non_frontier_run_without_model_or_body(self) -> None:
        self._set_complete_candidates()
        self.client.show_runs_in_experiment.return_value.append(snowpark.Row(name="ITER_REJECTED"))
        self.client.show_run_metrics_in_experiment.return_value.append(_attribute("ITER_REJECTED", "is_frontier", "0"))
        job = self._create_job()

        with self.assertLogs(_optimization_job.logger, level="WARNING") as logs:
            result = job.result()

        self.assertIn("run 'ITER_REJECTED' did not report its model or function body", logs.output[0])
        self.assertEqual([candidate.run_name for candidate in result.frontier], ["ITER_2", "ITER_1"])
        self.assertEqual([candidate.run_name for candidate in result.candidates], ["SEED", "ITER_1", "ITER_2"])

    def test_rejects_frontier_candidate_without_model_or_body(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_parameters_in_experiment.return_value = [
            row for row in self.client.show_run_parameters_in_experiment.return_value if row["run_name"] != "ITER_1"
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "returned an incomplete candidate 'ITER_1'"):
            job.result()

    def test_rejects_frontier_candidate_without_scores(self) -> None:
        self._set_complete_candidates()
        self.client.show_run_metrics_in_experiment.return_value = [
            row
            for row in self.client.show_run_metrics_in_experiment.return_value
            if (row["run_name"], row["name"]) != ("ITER_1", "val_score")
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "returned an incomplete candidate 'ITER_1'"):
            job.result()

    def test_rejects_missing_frontier_marker(self) -> None:
        self.client.show_runs_in_experiment.return_value = [snowpark.Row(name="ITER_1")]
        self.client.show_run_metrics_in_experiment.return_value = [
            _attribute("ITER_1", "val_score", "0.8"),
            _attribute("ITER_1", "cost_compared_to_seed", "0.6"),
        ]
        self.client.show_run_parameters_in_experiment.return_value = [
            _attribute("ITER_1", "model", "model-a"),
            _attribute("ITER_1", "function_impl", "body-a"),
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "did not identify any candidates"):
            job.result()

    def test_rejects_unknown_summary_frontier_candidate(self) -> None:
        self._set_complete_candidates()
        self.client.describe_experiment.side_effect = [
            {"execution_status": "COMPLETED"},
            {"summary": json.dumps({"pareto_frontier_runs": ["ITER_UNKNOWN"]})},
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "unknown frontier candidates"):
            job.result()

    def test_abort(self) -> None:
        job = self._create_job()

        job.abort()

        self.client.abort_experiment.assert_called_once_with(self.experiment_name)
        with self.assertRaisesRegex(RuntimeError, "was aborted"):
            job.result()

    def test_result_is_cached(self) -> None:
        expected = _optimization_result.OptimizationResult(
            [_candidate.Candidate(model="m", body="b", run_name="ITER_1", val_score=1, relative_cost=1)]
        )
        job = self._create_job()
        job._result = expected

        self.assertIs(job.result(), expected)

    def test_job_survives_pickle_round_trip(self) -> None:
        self.session.get_current_account.return_value = "ACCOUNT"
        self.session.get_current_role.return_value = "ROLE"
        self.session.get_current_database.return_value = "DB"
        self.session.get_current_schema.return_value = "SCHEMA"
        job = self._create_job()
        job._status = _experiment_job.JobStatus.COMPLETED
        job._result = _optimization_result.OptimizationResult(
            [_candidate.Candidate(model="m", body="b", run_name="ITER_1", val_score=1, relative_cost=1)]
        )

        restored = pickle.loads(pickle.dumps(job))

        with mock.patch("snowflake.snowpark.session._get_active_sessions", return_value={self.session}):
            self.assertEqual(restored.experiment, "DB.SCHEMA.EXPERIMENT")
            self.assertEqual(restored.status, _experiment_job.JobStatus.COMPLETED)
            self.assertEqual([candidate.run_name for candidate in restored.result().frontier], ["ITER_1"])


if __name__ == "__main__":
    absltest.main()
