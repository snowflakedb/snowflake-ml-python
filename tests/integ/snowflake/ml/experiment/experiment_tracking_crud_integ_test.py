import json

from absl.testing import absltest

from snowflake.ml.experiment import ExperimentTracking
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentCrudIntegTest(ExperimentTrackingIntegTestBase):
    def test_experiment_creation_and_deletion(self) -> None:
        # set_experiment with a new experiment name creates an experiment
        self.exp.set_experiment(experiment_name="test_experiment")
        res = self._session.sql("SHOW EXPERIMENTS").collect()
        self.assertEqual(len(res), 1)
        self.assertIn("TEST_EXPERIMENT", [experiment.name for experiment in res])

        # set_experiment with an existing experiment name does not create a new experiment
        self.exp.set_experiment(experiment_name="test_experiment")
        res = self._session.sql("SHOW EXPERIMENTS").collect()
        self.assertEqual(len(res), 1)

        # delete_experiment deletes the experiment
        self.exp.delete_experiment(experiment_name="test_experiment")
        res = self._session.sql("SHOW EXPERIMENTS").collect()
        self.assertEqual(len(res), 0)

    def test_experiment_start_multiple_runs(self) -> None:
        """Test starting multiple runs with different names."""
        experiment_name = "TEST_EXPERIMENT_MULTIPLE_RUNS"

        # Set up experiment
        self.exp.set_experiment(experiment_name=experiment_name)

        # Start multiple runs
        with self.exp.start_run(run_name="RUN_ONE"):
            pass
        with self.exp.start_run() as run:
            pass

        # Verify all runs exist in Snowflake
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        self.assertEqual(len(runs), 2)
        run_names = [row["name"] for row in runs]
        self.assertIn("RUN_ONE", run_names)
        self.assertIn(run.name, run_names)

    def test_experiment_delete_run(self) -> None:
        experiment_name = "TEST_EXPERIMENT_DELETE_RUNS"
        run_name_to_keep = "KEEP_THIS_RUN"
        run_name_to_delete = "DELETE_THIS_RUN"

        # Set up experiment and create runs
        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name_to_keep):
            pass
        with self.exp.start_run(run_name=run_name_to_delete):
            pass

        # Verify both runs exist
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        run_names = [row["name"] for row in runs]
        self.assertIn(run_name_to_keep, run_names)
        self.assertIn(run_name_to_delete, run_names)
        self.assertEqual(len(run_names), 2)

        # Delete one run
        self.exp.delete_run(run_name=run_name_to_delete)

        # Verify only the kept run remains
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        run_names = [row["name"] for row in runs]
        self.assertIn(run_name_to_keep, run_names)
        self.assertNotIn(run_name_to_delete, run_names)
        self.assertEqual(len(run_names), 1)

    def test_start_run_resumes_running_run_across_instances(self) -> None:
        """A RUNNING run should be resumable from a new ExperimentTracking instance."""
        experiment_name = "TEST_EXPERIMENT_RESUME"
        target_run_name = "RUN_TO_RESUME"

        # Set up experiment and start a run (leave it RUNNING by not closing the context)
        self.exp.set_experiment(experiment_name=experiment_name)
        run = self.exp.start_run(run_name=target_run_name)
        self.assertEqual(run.name, target_run_name)

        # New ExperimentTracking instance in the same DB/Schema resumes the same run
        ExperimentTracking._instance = None  # Reset singleton for test
        exp2 = ExperimentTracking(self._session, database_name=self._db_name, schema_name=self._schema_name)
        exp2.set_experiment(experiment_name=experiment_name)
        resumed = exp2.start_run(run_name=target_run_name)
        self.assertEqual(resumed.name, target_run_name)

        # Verify no duplicate runs were created
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["name"], target_run_name)

        # Cleanup: end the run
        self.exp.end_run()

    def test_start_run_existing_non_running_raises(self) -> None:
        """If a run exists but is not RUNNING, starting with same name should raise."""
        experiment_name = "TEST_EXPERIMENT_NON_RUNNING"
        run_name = "FINISHED_RUN"

        # Create and finish a run
        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            pass

        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        self.assertEqual(len(runs), 1)
        self.assertEqual(json.loads(runs[0]["metadata"])["status"], "FINISHED")

        # New instance attempting to start the same (non-running) run should error
        ExperimentTracking._instance = None  # Reset singleton for test
        exp2 = ExperimentTracking(self._session, database_name=self._db_name, schema_name=self._schema_name)
        exp2.set_experiment(experiment_name=experiment_name)
        with self.assertRaises(RuntimeError):
            exp2.start_run(run_name=run_name)


if __name__ == "__main__":
    absltest.main()
