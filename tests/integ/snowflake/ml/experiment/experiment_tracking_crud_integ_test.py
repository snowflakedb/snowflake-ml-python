import json

import yaml
from absl.testing import absltest

from snowflake import cortex
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.dataset import dataset
from snowflake.ml.experiment import ExperimentTracking, ExperimentType
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as snowpark_exceptions
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentCrudIntegTest(ExperimentTrackingIntegTestBase):
    def _create_evaluation_spec(self) -> str:
        dataset_name = "EVALUATION_DATASET"
        dataset_fqn = f"{self._db_name}.{self._schema_name}.{dataset_name}"
        self._session.sql(f"CREATE OR REPLACE DATASET {dataset_fqn}").collect()
        self._session.sql(
            f"ALTER DATASET {dataset_fqn} ADD VERSION 'v1' FROM "
            "(SELECT 'What is the capital of France?' AS question, 'Paris' AS label_col)"
        ).collect()
        return (
            "query_text: AI_COMPLETE('llama3.1-8b', question)\n"
            "metrics:\n"
            "  - name: exact_match\n"
            "dataset:\n"
            f"  name: {dataset_fqn}\n"
            "  version: v1\n"
            "  ground_truth: label_col\n"
            "evaluation:\n"
            "  num_eval_runs: 10\n"
            "# SQL literal escaping check: C:\\path\\"
        )

    def test_typed_experiment_creation_and_invalid_execution(self) -> None:
        typed_experiment_name = sql_identifier.SqlIdentifier("TYPED_EXPERIMENT")
        self.exp._sql_client.create_experiment(
            typed_experiment_name,
            creation_mode=sql_client.CreationMode(),
            experiment_type=ExperimentType.AI_FUNCTION_EVALUATION,
        )
        typed_experiments = self._session.sql(
            f"SHOW EXPERIMENTS LIKE 'TYPED_EXPERIMENT' IN SCHEMA {self._db_name}.{self._schema_name}"
        ).collect()
        self.assertLen(typed_experiments, 1)
        self.assertEqual(typed_experiments[0]["type"], "AI_FUNCTION_EVALUATION")

        self.exp.set_experiment(experiment_name="REGULAR_EXPERIMENT")
        with self.assertRaises(snowpark_exceptions.SnowparkSQLException):
            self.exp._sql_client.execute_experiment(sql_identifier.SqlIdentifier("REGULAR_EXPERIMENT"))

    def test_create_or_replace_experiment(self) -> None:
        experiment_name = sql_identifier.SqlIdentifier("OR_REPLACE_EXPERIMENT")
        self.exp._sql_client.create_experiment(experiment_name, creation_mode=sql_client.CreationMode())

        self.exp._sql_client.create_experiment(
            experiment_name,
            creation_mode=sql_client.CreationMode(or_replace=True),
        )

        experiments = self._session.sql(
            f"SHOW EXPERIMENTS LIKE 'OR_REPLACE_EXPERIMENT' IN SCHEMA {self._db_name}.{self._schema_name}"
        ).collect()
        self.assertLen(experiments, 1)

    def test_typed_experiment_specification_round_trip(self) -> None:
        experiment_name = sql_identifier.SqlIdentifier("SPECIFICATION_ROUND_TRIP")
        specification = self._create_evaluation_spec()
        self.exp._sql_client.create_experiment(
            experiment_name,
            creation_mode=sql_client.CreationMode(),
            experiment_type=ExperimentType.AI_FUNCTION_EVALUATION,
            specification=specification,
        )

        result = self._session.sql(f"DESC EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}").collect()

        self.assertLen(result, 1)
        self.assertEqual(yaml.safe_load(result[0]["spec"]), yaml.safe_load(specification))

    def test_public_evaluate_api(self) -> None:
        self._create_evaluation_spec()
        input_dataset = dataset.Dataset(
            self._session,
            self._db_name,
            self._schema_name,
            "EVALUATION_DATASET",
            selected_version="v1",
        )
        experiment_fqn = f"{self._db_name}.{self._schema_name}.PUBLIC_EVALUATE_API"

        job = cortex.evaluate(
            data=input_dataset,
            scorers=["exact_match"],
            model="llama3.1-8b",
            prompt="Answer {0}",
            inputs=["question"],
            label_col="label_col",
            session=self._session,
            experiment=experiment_fqn,
        )

        result = self._session.sql(f"DESC EXPERIMENT {experiment_fqn}").collect()

        self.assertEqual(job.experiment, experiment_fqn)
        self.assertLen(result, 1)
        self.assertEqual(result[0]["type"], "AI_FUNCTION_EVALUATION")
        self.assertEqual(
            yaml.safe_load(result[0]["spec"]),
            {
                "query_text": "AI_COMPLETE('llama3.1-8b', PROMPT('Answer {0}', QUESTION))::VARCHAR",
                "metrics": [{"name": "exact_match"}],
                "dataset": {
                    "name": f"{self._db_name}.{self._schema_name}.EVALUATION_DATASET",
                    "version": "v1",
                    "ground_truth": "LABEL_COL",
                },
            },
        )

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

    def test_end_run_with_status(self) -> None:
        """Test implicit and explicit run status for both FINISHED and FAILED."""
        experiment_name = "TEST_EXPERIMENT_END_RUN_STATUS"
        self.exp.set_experiment(experiment_name=experiment_name)

        # Implicit FINISHED: context manager exits cleanly
        with self.exp.start_run(run_name="IMPLICIT_FINISHED"):
            pass

        # Implicit FAILED: context manager exits with exception
        with self.assertRaises(ValueError):
            with self.exp.start_run(run_name="IMPLICIT_FAILED"):
                raise ValueError("simulated failure")

        # Explicit FINISHED: end_run(status="FINISHED")
        self.exp.start_run(run_name="EXPLICIT_FINISHED")
        self.exp.end_run(status="FINISHED")

        # Explicit FAILED: end_run(status="FAILED")
        self.exp.start_run(run_name="EXPLICIT_FAILED")
        self.exp.end_run(status="FAILED")

        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        status_by_name = {row["name"]: json.loads(row["metadata"])["status"] for row in runs}
        self.assertEqual(status_by_name["IMPLICIT_FINISHED"], "FINISHED")
        self.assertEqual(status_by_name["IMPLICIT_FAILED"], "FAILED")
        self.assertEqual(status_by_name["EXPLICIT_FINISHED"], "FINISHED")
        self.assertEqual(status_by_name["EXPLICIT_FAILED"], "FAILED")

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
