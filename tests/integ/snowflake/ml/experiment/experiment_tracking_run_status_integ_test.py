import json
import os
import tempfile

from absl.testing import absltest, parameterized

from snowflake.ml.experiment import ExperimentTracking
from snowflake.snowpark import exceptions
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentRunStatusIntegTest(ExperimentTrackingIntegTestBase):
    """End-to-end coverage of paused and cancelled experiment runs."""

    # Each status is enabled by its own session parameter and defaults to on once that parameter exists.
    _STATUS_PARAMETERS = (
        "ENABLE_PAUSED_EXPERIMENT_RUN_STATUS",
        "ENABLE_CANCELLED_EXPERIMENT_RUN_STATUS",
    )

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._statuses_supported = True
        for parameter in cls._STATUS_PARAMETERS:
            try:
                cls._session.sql(f"ALTER SESSION SET {parameter} = TRUE").collect()
            except exceptions.SnowparkSQLException as exc:
                # Deployments that predate these statuses reject the parameter name itself.
                if "invalid parameter" in str(exc).lower():
                    cls._statuses_supported = False
                    return
                raise

    def setUp(self) -> None:
        if not self._statuses_supported:
            self.skipTest("Paused and cancelled run statuses are not available on this Snowflake account.")
        super().setUp()

    def _experiment_fqn(self, experiment_name: str) -> str:
        return f"{self._db_name}.{self._schema_name}.{experiment_name}"

    def _run_status(self, experiment_name: str, run_name: str) -> str:
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {self._experiment_fqn(experiment_name)}").collect()
        matched = [run for run in runs if run["name"] == run_name]
        self.assertLen(matched, 1)
        return str(json.loads(matched[0]["metadata"])["status"])

    def _metric_values(self, experiment_name: str, run_name: str) -> dict[tuple[str, int], float]:
        metrics = self._session.sql(
            "SELECT * FROM TABLE(SYSTEM$GET_EXPERIMENT_RUN_METRICS"
            f"('{self._experiment_fqn(experiment_name)}', '{run_name}'))"
        ).collect()
        return {(str(metric["METRIC_NAME"]), int(metric["STEP"])): float(metric["VALUE"]) for metric in metrics}

    def _param_values(self, experiment_name: str, run_name: str) -> dict[str, str]:
        parameters = self._session.sql(
            f"SHOW RUN PARAMETERS IN EXPERIMENT {self._experiment_fqn(experiment_name)} RUN {run_name}"
        ).collect()
        return {str(parameter["name"]): str(parameter["value"]) for parameter in parameters}

    def test_pause_and_resume_round_trip(self) -> None:
        """A running run can be paused, rejects writes, then resumes and finishes."""
        experiment_name = "TEST_EXPERIMENT_PAUSE_RESUME"
        run_name = "PAUSE_RESUME_RUN"
        self.exp.set_experiment(experiment_name=experiment_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "notes.txt")
            with open(artifact_path, "w", encoding="utf-8") as artifact_file:
                artifact_file.write("before-pause")
            self.exp.start_run(run_name=run_name)
            self.exp.log_metric("accuracy", 0.4, step=1)
            self.exp.log_param("learning_rate", 0.1)
            self.exp.log_artifact(artifact_path)

        self.assertEqual(self._run_status(experiment_name, run_name), "RUNNING")
        self.exp.pause_run()
        self.assertIsNone(self.exp._run)
        self.assertEqual(self._run_status(experiment_name, run_name), "PAUSED")

        # Pausing a paused run is a no-op.
        self.exp.pause_run(run_name=run_name)
        self.assertEqual(self._run_status(experiment_name, run_name), "PAUSED")

        # Reads still work. Writes do not, because only a running run is mutable.
        self.assertEqual([artifact.name for artifact in self.exp.list_artifacts(run_name=run_name)], ["notes.txt"])
        self.assertEqual(self._metric_values(experiment_name, run_name), {("accuracy", 1): 0.4})
        self.assertEqual(self._param_values(experiment_name, run_name), {"learning_rate": "0.1"})

        with self.assertRaises(exceptions.SnowparkSQLException) as write_error:
            self._session.sql(
                f"ALTER EXPERIMENT {self._experiment_fqn(experiment_name)} MODIFY RUN {run_name} "
                'ADD METRICS=$$[{"name": "loss", "value": 0.1, "step": 1}]$$'
            ).collect()
        self.assertIn(
            f"Run {run_name} does not have status RUNNING: can only modify runs with status RUNNING",
            str(write_error.exception),
        )
        self.assertEqual(self._metric_values(experiment_name, run_name), {("accuracy", 1): 0.4})

        resumed = self.exp.start_run(run_name=run_name)
        self.assertEqual(resumed.name, run_name)
        self.assertIs(self.exp._run, resumed)
        self.assertEqual(self._run_status(experiment_name, run_name), "RUNNING")

        self.exp.log_metric("accuracy", 0.8, step=2)
        self.exp.log_param("learning_rate", 0.01)
        self.exp.end_run(status="FINISHED")

        self.assertIsNone(self.exp._run)
        self.assertEqual(self._run_status(experiment_name, run_name), "FINISHED")
        self.assertEqual(
            self._metric_values(experiment_name, run_name),
            {("accuracy", 1): 0.4, ("accuracy", 2): 0.8},
        )
        self.assertEqual(self._param_values(experiment_name, run_name), {"learning_rate": "0.01"})
        with self.assertRaises(RuntimeError):
            self.exp.start_run(run_name=run_name)

    def test_logging_after_pause_does_not_change_paused_run(self) -> None:
        """Logging with no active run starts a new run and leaves the paused run unchanged."""
        experiment_name = "TEST_EXPERIMENT_LOG_WHILE_PAUSED"
        paused_run_name = "PAUSED_RUN"
        self.exp.set_experiment(experiment_name=experiment_name)
        self.exp.start_run(run_name=paused_run_name)
        self.exp.log_metric("accuracy", 0.4, step=1)
        self.exp.log_param("learning_rate", 0.1)
        self.exp.pause_run()

        self.exp.log_metric("accuracy", 0.9, step=2)
        self.exp.log_param("learning_rate", 0.2)
        new_run = self.exp._run
        self.assertIsNotNone(new_run)
        assert new_run is not None  # for mypy
        new_run_name = new_run.name.resolved()
        self.assertNotEqual(new_run_name, paused_run_name)

        self.assertEqual(self._run_status(experiment_name, paused_run_name), "PAUSED")
        self.assertEqual(self._metric_values(experiment_name, paused_run_name), {("accuracy", 1): 0.4})
        self.assertEqual(self._param_values(experiment_name, paused_run_name), {"learning_rate": "0.1"})
        self.assertEqual(self._run_status(experiment_name, new_run_name), "RUNNING")
        self.assertEqual(self._metric_values(experiment_name, new_run_name), {("accuracy", 2): 0.9})
        self.assertEqual(self._param_values(experiment_name, new_run_name), {"learning_rate": "0.2"})

    def test_cancel_running_run(self) -> None:
        """end_run(status='CANCELLED') commits a running run, which cannot be resumed or paused."""
        experiment_name = "TEST_EXPERIMENT_CANCEL"
        run_name = "CANCELLED_RUN"
        self.exp.set_experiment(experiment_name=experiment_name)
        self.exp.start_run(run_name=run_name)
        self.exp.log_metric("accuracy", 0.2, step=1)
        self.exp.end_run(status="CANCELLED")

        self.assertIsNone(self.exp._run)
        self.assertEqual(self._run_status(experiment_name, run_name), "CANCELLED")
        self.assertEqual(self._metric_values(experiment_name, run_name), {("accuracy", 1): 0.2})
        with self.assertRaises(RuntimeError) as resume_error:
            self.exp.start_run(run_name=run_name)
        self.assertEqual(
            f"Run {run_name} exists but cannot be resumed because its status is CANCELLED.",
            str(resume_error.exception),
        )
        with self.assertRaises(RuntimeError) as pause_error:
            self.exp.pause_run(run_name=run_name)
        self.assertEqual(
            f"Run {run_name} cannot be paused because its status is CANCELLED.",
            str(pause_error.exception),
        )

    def test_paused_run_can_be_cancelled_after_resume(self) -> None:
        """A paused run resumed from a new client can then be cancelled."""
        experiment_name = "TEST_EXPERIMENT_RESUME_THEN_CANCEL"
        run_name = "RESUME_THEN_CANCEL"
        self.exp.set_experiment(experiment_name=experiment_name)
        self.exp.start_run(run_name=run_name)
        self.exp.log_metric("accuracy", 0.3, step=1)
        self.exp.pause_run()
        self.assertEqual(self._run_status(experiment_name, run_name), "PAUSED")

        ExperimentTracking._instance = None
        resumed_exp = ExperimentTracking(self._session, database_name=self._db_name, schema_name=self._schema_name)
        resumed_exp.set_experiment(experiment_name=experiment_name)
        resumed = resumed_exp.start_run(run_name=run_name)
        self.assertEqual(resumed.name, run_name)
        self.assertEqual(self._run_status(experiment_name, run_name), "RUNNING")

        resumed_exp.log_metric("accuracy", 0.6, step=2)
        resumed_exp.end_run(status="CANCELLED")
        self.assertEqual(self._run_status(experiment_name, run_name), "CANCELLED")
        self.assertEqual(
            self._metric_values(experiment_name, run_name),
            {("accuracy", 1): 0.3, ("accuracy", 2): 0.6},
        )

    def test_context_manager_leaves_paused_run_paused(self) -> None:
        """Pausing inside a start_run context leaves the run paused when the context exits."""
        experiment_name = "TEST_EXPERIMENT_PAUSE_CONTEXT"
        run_name = "PAUSED_IN_CONTEXT"
        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            self.exp.log_metric("accuracy", 0.5, step=1)
            self.exp.pause_run()
            self.assertIsNone(self.exp._run)

        self.assertEqual(self._run_status(experiment_name, run_name), "PAUSED")
        self.assertEqual(self._metric_values(experiment_name, run_name), {("accuracy", 1): 0.5})

    @parameterized.named_parameters(
        ("finished", "FINISHED"),
        ("failed", "FAILED"),
        ("cancelled", "CANCELLED"),
    )
    def test_terminal_run_cannot_be_resumed_or_paused(self, status: str) -> None:
        """Finished, failed, and cancelled runs stay terminal."""
        experiment_name = f"TEST_EXPERIMENT_TERMINAL_{status}"
        run_name = f"TERMINAL_{status}"
        self.exp.set_experiment(experiment_name=experiment_name)

        if status == "FAILED":
            with self.assertRaises(RuntimeError):
                with self.exp.start_run(run_name=run_name):
                    raise RuntimeError("simulated failure")
        elif status == "CANCELLED":
            self.exp.start_run(run_name=run_name)
            self.exp.end_run(status="CANCELLED")
        else:
            with self.exp.start_run(run_name=run_name):
                pass

        self.assertEqual(self._run_status(experiment_name, run_name), status)
        with self.assertRaises(RuntimeError) as resume_error:
            self.exp.start_run(run_name=run_name)
        self.assertEqual(
            f"Run {run_name} exists but cannot be resumed because its status is {status}.",
            str(resume_error.exception),
        )
        with self.assertRaises(RuntimeError) as pause_error:
            self.exp.pause_run(run_name=run_name)
        self.assertEqual(
            f"Run {run_name} cannot be paused because its status is {status}.",
            str(pause_error.exception),
        )


if __name__ == "__main__":
    absltest.main()
