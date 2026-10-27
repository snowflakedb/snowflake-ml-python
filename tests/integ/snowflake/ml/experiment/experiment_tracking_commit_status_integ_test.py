import json

from absl.testing import absltest, parameterized

from snowflake.snowpark import exceptions
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentCommitStatusIntegTest(ExperimentTrackingIntegTestBase):
    """End-to-end coverage of committing an experiment run with a terminal status."""

    # CANCELLED is gated by a session parameter and is unavailable where that parameter is not registered.
    _CANCELLED_PARAMETER = "ENABLE_CANCELLED_EXPERIMENT_RUN_STATUS"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._cancelled_supported = True
        try:
            cls._session.sql(f"ALTER SESSION SET {cls._CANCELLED_PARAMETER} = TRUE").collect()
        except exceptions.SnowparkSQLException as exc:
            if "invalid parameter" in str(exc).lower():
                cls._cancelled_supported = False
                return
            raise

    def _experiment_fqn(self, experiment_name: str) -> str:
        return f"{self._db_name}.{self._schema_name}.{experiment_name}"

    def _run_status(self, experiment_name: str, run_name: str) -> str:
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {self._experiment_fqn(experiment_name)}").collect()
        matched = [run for run in runs if run["name"] == run_name]
        self.assertLen(matched, 1)
        return str(json.loads(matched[0]["metadata"])["status"])

    @parameterized.named_parameters(
        ("default", None, "FINISHED"),
        ("finished", "FINISHED", "FINISHED"),
        ("failed", "FAILED", "FAILED"),
        ("cancelled", "CANCELLED", "CANCELLED"),
    )
    def test_end_run_commits_terminal_status(self, status: str | None, expected: str) -> None:
        """Committing with no status finishes the run. An explicit terminal status is stored."""
        if status == "CANCELLED" and not self._cancelled_supported:
            self.skipTest("Cancelled run status is not available on this Snowflake account.")

        experiment_name = f"TEST_EXPERIMENT_COMMIT_{expected}"
        run_name = f"COMMIT_{expected}"
        self.exp.set_experiment(experiment_name=experiment_name)
        self.exp.start_run(run_name=run_name)
        self.exp.log_metric("accuracy", 0.5, step=1)
        if status is None:
            self.exp.end_run()
        else:
            self.exp.end_run(status=status)

        self.assertIsNone(self.exp._run)
        self.assertEqual(self._run_status(experiment_name, run_name), expected)
        with self.assertRaises(RuntimeError) as resume_error:
            self.exp.start_run(run_name=run_name)
        self.assertIn(f"Run {run_name} exists but cannot be resumed", str(resume_error.exception))

    def test_end_run_rejects_non_terminal_status(self) -> None:
        """RUNNING and PAUSED are rejected before commit, so the run stays running."""
        experiment_name = "TEST_EXPERIMENT_REJECT_NON_TERMINAL"
        run_name = "STILL_RUNNING"
        self.exp.set_experiment(experiment_name=experiment_name)
        self.exp.start_run(run_name=run_name)

        for status in ("RUNNING", "PAUSED"):
            with self.assertRaises(ValueError) as status_error:
                self.exp.end_run(status=status)
            self.assertEqual(
                f"Unsupported status value: '{status}'. Supported values are: ['FAILED', 'FINISHED', 'CANCELLED'].",
                str(status_error.exception),
            )

        self.assertIsNotNone(self.exp._run)
        self.assertEqual(self._run_status(experiment_name, run_name), "RUNNING")


if __name__ == "__main__":
    absltest.main()
