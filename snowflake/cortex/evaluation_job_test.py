import json
import pickle
import time
from unittest import mock

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _evaluation_job, _evaluation_result, _experiment_job
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment._client import experiment_tracking_sql_client


class EvaluationJobTest(absltest.TestCase):
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
        self.client.show_runs_in_experiment.return_value = []

    @staticmethod
    def _run(name: str, status: str) -> snowpark.Row:
        return snowpark.Row(name=name, metadata=json.dumps({"status": status}))

    def _create_job(self, experiment: str = "DB.SCHEMA.EXPERIMENT") -> _evaluation_job.EvaluationJob:
        with mock.patch.object(
            _experiment_job,
            "resolve_client",
            return_value=(self.client, self.experiment_name),
        ):
            return _evaluation_job.EvaluationJob(self.session, experiment)

    def test_experiment_returns_backing_experiment_name(self) -> None:
        self.assertEqual(self._create_job().experiment, "DB.SCHEMA.EXPERIMENT")

    def test_experiment_is_resolved_to_fully_qualified_name(self) -> None:
        job = self._create_job("EXPERIMENT")

        self.assertEqual(job.experiment, "DB.SCHEMA.EXPERIMENT")
        self.client.fully_qualified_object_name.assert_called_once_with(None, None, self.experiment_name)

    def test_status_is_cached_after_completion(self) -> None:
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.COMPLETED)
        self.assertEqual(job.status, _experiment_job.JobStatus.COMPLETED)

        self.client.describe_experiment.assert_called_once_with(self.experiment_name)

    def test_running_status_is_not_cached(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": "RUNNING"}
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.RUNNING)
        self.assertEqual(job.status, _experiment_job.JobStatus.RUNNING)

        self.assertEqual(self.client.describe_experiment.call_count, 2)

    def test_status_treats_unrecognized_execution_status_as_in_progress(self) -> None:
        self.client.describe_experiment.side_effect = [
            {"execution_status": "PAUSED"},
            {"execution_status": "COMPLETED"},
        ]
        job = self._create_job()

        with self.assertLogs(_experiment_job.logger, level="WARNING") as logs:
            self.assertEqual(job.status, _experiment_job.JobStatus.UNKNOWN)
        self.assertIn("unrecognized execution status 'PAUSED'", logs.output[0])
        with mock.patch.object(time, "sleep") as sleep:
            self.assertEqual(job.wait(), _experiment_job.JobStatus.COMPLETED)
        sleep.assert_not_called()

    def test_status_falls_back_to_running_runs(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": None}
        self.client.show_runs_in_experiment.return_value = [snowpark.Row(metadata=json.dumps({"status": "RUNNING"}))]
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.RUNNING)

    def test_status_falls_back_to_completed_when_all_runs_are_terminal(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": None}
        self.client.show_runs_in_experiment.return_value = [
            snowpark.Row(metadata=json.dumps({"status": "FINISHED"})),
            snowpark.Row(metadata=json.dumps({"status": "FAILED"})),
            snowpark.Row(metadata=json.dumps({"status": "CANCELLED"})),
        ]
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.COMPLETED)

    def test_status_falls_back_to_running_when_no_runs_exist(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": None}
        self.client.show_runs_in_experiment.return_value = []
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.RUNNING)

    def test_status_treats_run_without_status_as_running(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": None}
        self.client.show_runs_in_experiment.return_value = [snowpark.Row(metadata=json.dumps({}))]
        job = self._create_job()

        self.assertEqual(job.status, _experiment_job.JobStatus.RUNNING)

    def test_status_rejects_malformed_run_metadata(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": None}
        for run in (
            snowpark.Row(metadata="{bad"),
            snowpark.Row(metadata=json.dumps(["FINISHED"])),
            snowpark.Row(metadata=None),
            snowpark.Row(name="EVAL_1"),
            snowpark.Row(),
        ):
            with self.subTest(run=run):
                self.client.show_runs_in_experiment.return_value = [run]
                job = self._create_job()

                with self.assertRaisesRegex(RuntimeError, "returned malformed run metadata"):
                    _ = job.status

    def test_status_treats_nonterminal_run_status_as_running(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": None}
        for status in ("PAUSED", "QUEUED"):
            with self.subTest(status=status):
                self.client.show_runs_in_experiment.return_value = [
                    snowpark.Row(metadata=json.dumps({"status": status}))
                ]
                job = self._create_job()

                self.assertEqual(job.status, _experiment_job.JobStatus.RUNNING)

    def test_wait_returns_terminal_status(self) -> None:
        self.client.describe_experiment.side_effect = [
            {"execution_status": "RUNNING"},
            {"execution_status": "COMPLETED"},
        ]
        job = self._create_job()

        with mock.patch.object(time, "sleep") as sleep:
            self.assertEqual(job.wait(), _experiment_job.JobStatus.COMPLETED)

        sleep.assert_called_once_with(10.0)

    def test_wait_rejects_experiment_that_was_not_executed(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": "NOT_EXECUTED"}
        job = self._create_job()

        with mock.patch.object(time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "'DB.SCHEMA.EXPERIMENT' has not been executed"):
                job.wait()
        sleep.assert_not_called()

    def test_wait_polls_queued_experiment(self) -> None:
        self.client.describe_experiment.side_effect = [
            {"execution_status": "QUEUED"},
            {"execution_status": "COMPLETED"},
        ]
        job = self._create_job()

        with mock.patch.object(time, "sleep") as sleep:
            self.assertEqual(job.wait(), _experiment_job.JobStatus.COMPLETED)
        sleep.assert_called_once_with(10.0)

    def test_wait_times_out_without_overshooting_timeout(self) -> None:
        self.client.describe_experiment.return_value = {"execution_status": "RUNNING"}
        job = self._create_job()

        with mock.patch.object(time, "monotonic", side_effect=[100.0, 100.0, 105.0]):
            with mock.patch.object(time, "sleep") as sleep:
                with self.assertRaises(TimeoutError):
                    job.wait(timeout=5)

        sleep.assert_called_once_with(5.0)

    def test_result_loads_and_caches_successful_result(self) -> None:
        self.client.show_run_metrics_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="score", value="0.5"),
            snowpark.Row(run_name="EVAL_2", name="score", value="1.0"),
        ]
        job = self._create_job()

        self.assertEqual(job.result().score, 0.75)
        self.assertEqual(job.result().score, 0.75)

        self.client.show_run_metrics_in_experiment.assert_called_once_with(experiment_name=self.experiment_name)

    def test_result_surfaces_summary_error(self) -> None:
        self.client.describe_experiment.side_effect = [
            {"execution_status": "COMPLETED"},
            {"summary": json.dumps({"error_message": "backend failure"})},
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "failed: backend failure"):
            job.result()

    def test_error_message_reads_summary(self) -> None:
        self.client.describe_experiment.return_value = {"summary": json.dumps({"error_message": "backend failure"})}
        job = self._create_job()

        self.assertEqual(job.error_message, "backend failure")

    def test_error_message_is_none_without_reported_error(self) -> None:
        for description in ({"summary": None}, {}, {"summary": json.dumps({"error_message": ""})}):
            with self.subTest(description=description):
                self.client.describe_experiment.return_value = description
                job = self._create_job()

                self.assertIsNone(job.error_message)

    def test_result_handles_sparse_and_duplicate_runs(self) -> None:
        self.client.show_run_metrics_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="score", value="0.5"),
            snowpark.Row(run_name="EVAL_1", name="score", value="1.0"),
            snowpark.Row(run_name="EVAL_2", name="rows_evaluated", value="10"),
            snowpark.Row(run_name="EVAL_3", name="score", value="0.9"),
        ]
        job = self._create_job()

        self.assertEqual(job.result().score, 0.7)

    def test_result_requires_score(self) -> None:
        self.client.show_run_metrics_in_experiment.return_value = []
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "did not return a score"):
            job.result()

    def test_result_rejects_invalid_score(self) -> None:
        self.client.show_run_metrics_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="score", value="invalid")
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "returned an invalid score"):
            job.result()

    def test_result_rejects_mixed_success_and_failure(self) -> None:
        self.client.show_runs_in_experiment.return_value = [
            self._run("EVAL_1", "FINISHED"),
            self._run("EVAL_2", "FAILED"),
        ]
        self.client.show_run_metrics_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="score", value="1.0"),
            snowpark.Row(run_name="EVAL_2", name="score", value="0.0"),
        ]
        self.client.show_run_parameters_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_2", name="error_message", value="model request failed")
        ]
        job = self._create_job()

        for _ in range(2):
            with self.assertRaisesRegex(
                RuntimeError, r"'DB.SCHEMA.EXPERIMENT' run 'EVAL_2' failed: model request failed$"
            ):
                job.result()

        self.client.show_run_parameters_in_experiment.assert_called_with(
            experiment_name=self.experiment_name,
            run_name=sql_identifier.SqlIdentifier("EVAL_2"),
            like="error_message",
        )
        self.client.show_run_metrics_in_experiment.assert_not_called()

    def test_result_rejects_failed_run_that_reported_a_score(self) -> None:
        self.client.show_runs_in_experiment.return_value = [self._run("EVAL_1", "FAILED")]
        self.client.show_run_metrics_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="score", value="0.75")
        ]
        self.client.show_run_parameters_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="error_message", value="eval_detail upload failed")
        ]
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "run 'EVAL_1' failed: eval_detail upload failed"):
            job.result()

    def test_result_reports_failed_run_without_error_message(self) -> None:
        self.client.show_runs_in_experiment.return_value = [self._run("EVAL_1", "FAILED")]
        self.client.show_run_parameters_in_experiment.return_value = []
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, r"run 'EVAL_1' failed$"):
            job.result()

    def test_result_reports_first_failed_run(self) -> None:
        self.client.show_runs_in_experiment.return_value = [
            self._run("EVAL_3", "FAILED"),
            self._run("EVAL_2", "FAILED"),
        ]
        self.client.show_run_parameters_in_experiment.return_value = []
        job = self._create_job()

        with self.assertRaisesRegex(RuntimeError, "run 'EVAL_2' failed"):
            job.result()

    def test_result_ignores_runs_outside_the_evaluation(self) -> None:
        self.client.show_runs_in_experiment.return_value = [
            self._run("EVAL_1", "FINISHED"),
            self._run("OTHER", "FAILED"),
        ]
        self.client.show_run_metrics_in_experiment.return_value = [
            snowpark.Row(run_name="EVAL_1", name="score", value="0.5")
        ]
        job = self._create_job()

        self.assertEqual(job.result().score, 0.5)
        self.client.show_run_parameters_in_experiment.assert_not_called()

    def test_abort(self) -> None:
        job = self._create_job()

        job.abort()

        self.client.abort_experiment.assert_called_once_with(self.experiment_name)
        with self.assertRaisesRegex(RuntimeError, "was aborted"):
            job.result()

    def test_job_survives_pickle_round_trip(self) -> None:
        self.session.get_current_account.return_value = "ACCOUNT"
        self.session.get_current_role.return_value = "ROLE"
        self.session.get_current_database.return_value = "DB"
        self.session.get_current_schema.return_value = "SCHEMA"
        job = self._create_job()
        job._status = _experiment_job.JobStatus.COMPLETED
        job._result = _evaluation_result.EvaluationResult(0.75)

        restored = pickle.loads(pickle.dumps(job))

        with mock.patch("snowflake.snowpark.session._get_active_sessions", return_value={self.session}):
            self.assertEqual(restored.experiment, "DB.SCHEMA.EXPERIMENT")
            self.assertEqual(restored.status, _experiment_job.JobStatus.COMPLETED)
            self.assertEqual(restored.result().score, 0.75)


class ExperimentJobClientTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = mock.create_autospec(snowpark.Session, instance=True)

    def test_resolve_client_requires_current_database(self) -> None:
        self.session.get_current_database.return_value = None

        with self.assertRaisesRegex(ValueError, "current database"):
            _experiment_job.resolve_client(self.session, "EXPERIMENT")

    @mock.patch.object(experiment_tracking_sql_client, "ExperimentTrackingSQLClient")
    def test_resolve_client_defaults_to_public_schema(self, client_class: mock.Mock) -> None:
        self.session.get_current_database.return_value = '"DB"'
        self.session.get_current_schema.return_value = None

        _, experiment_name = _experiment_job.resolve_client(self.session, "EXPERIMENT")

        client_class.assert_called_once_with(
            self.session,
            database_name=sql_identifier.SqlIdentifier('"DB"'),
            schema_name=sql_identifier.SqlIdentifier("PUBLIC"),
        )
        self.assertEqual(experiment_name, sql_identifier.SqlIdentifier("EXPERIMENT"))

    @mock.patch.object(experiment_tracking_sql_client, "ExperimentTrackingSQLClient")
    def test_resolve_client_uses_fully_qualified_name_without_session_context(self, client_class: mock.Mock) -> None:
        _, experiment_name = _experiment_job.resolve_client(self.session, "DB.SCHEMA.EXPERIMENT")

        self.session.get_current_database.assert_not_called()
        self.session.get_current_schema.assert_not_called()
        client_class.assert_called_once_with(
            self.session,
            database_name=sql_identifier.SqlIdentifier("DB"),
            schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
        )
        self.assertEqual(experiment_name, sql_identifier.SqlIdentifier("EXPERIMENT"))


if __name__ == "__main__":
    absltest.main()
