import json
import os
import tempfile
import uuid

import pandas as pd
import xgboost as xgb
from absl.testing import absltest

from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.experiment._entities.run_metadata import RunStatus
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class ExperimentTrackingIntegrationTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self._session)
        self._schema_name = "PUBLIC"
        self._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "TEST_EXPERIMENT_TRACKING"
        ).upper()
        self._db_manager.create_database(self._db_name, data_retention_time_in_days=1)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.exp = ExperimentTracking(
            self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
        )

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._db_name)
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

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

    def test_log_metrics_and_params_comprehensive(self) -> None:
        """Comprehensive test for logging metrics and parameters with various scenarios."""
        experiment_name = "TEST_EXPERIMENT_METRICS_PARAMS"
        run_name = "TEST_RUN_COMPREHENSIVE"

        # Set up experiment and run
        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name) as run:

            # Verify run is running
            self.assertEqual(run._get_metadata().status, RunStatus.RUNNING)

            # Log single metric
            self.exp.log_metric("accuracy", 0.95, step=1)

            # Log multiple metrics at once
            batch_metrics = {"precision": 0.92, "recall": 0.88, "f1_score": 0.90}
            self.exp.log_metrics(batch_metrics, step=5)

            # Log same metric at different steps
            self.exp.log_metric("loss", 0.8, step=1)
            self.exp.log_metric("loss", 0.6, step=2)
            self.exp.log_metric("loss", 0.4, step=3)

            # Update metric at same step (should overwrite)
            self.exp.log_metric("accuracy", 0.97, step=1)  # Should overwrite 0.95

            # Log single parameter
            self.exp.log_param("learning_rate", 0.001)

            # Log multiple parameters of different types
            batch_params = {"batch_size": 32, "model_type": "transformer", "dropout": 0.1, "use_attention": True}
            self.exp.log_params(batch_params)

            # Update parameter (should overwrite)
            self.exp.log_param("batch_size", 64)  # Should overwrite 32

            # Log additional metric and parameter
            self.exp.log_metric("train_accuracy", 0.85, step=10)
            self.exp.log_param("optimizer", "adam")

        # Verify all data was logged correctly
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        self.assertEqual(len(runs), 1)

        # Parse and verify metadata
        metadata = json.loads(runs[0]["metadata"])

        # Verify run is finished
        self.assertEqual(metadata["status"], RunStatus.FINISHED.value)

        # Check metrics
        self.assertIn("metrics", metadata)
        # Should have: accuracy(updated), precision, recall, f1_score, loss(3 steps), train_accuracy = 8 total
        self.assertEqual(len(metadata["metrics"]), 8)

        metric_dict = {}
        for m in metadata["metrics"]:
            key = f"{m['name']}_step_{m['step']}"
            metric_dict[key] = m

        # Verify single metric (updated value)
        self.assertEqual(metric_dict["accuracy_step_1"]["value"], 0.97)  # Updated value

        # Verify batch metrics
        self.assertEqual(metric_dict["precision_step_5"]["value"], 0.92)
        self.assertEqual(metric_dict["recall_step_5"]["value"], 0.88)
        self.assertEqual(metric_dict["f1_score_step_5"]["value"], 0.90)

        # Verify multi-step metrics
        self.assertEqual(metric_dict["loss_step_1"]["value"], 0.8)
        self.assertEqual(metric_dict["loss_step_2"]["value"], 0.6)
        self.assertEqual(metric_dict["loss_step_3"]["value"], 0.4)

        # Verify additional metric
        self.assertEqual(metric_dict["train_accuracy_step_10"]["value"], 0.85)

        # Check parameters
        self.assertEqual(len(metadata["parameters"]), 6)
        param_dict = {p["name"]: p["value"] for p in metadata["parameters"]}
        self.assertEqual(param_dict["learning_rate"], "0.001")
        self.assertEqual(param_dict["batch_size"], "64")  # Updated value
        self.assertEqual(param_dict["model_type"], "transformer")
        self.assertEqual(param_dict["dropout"], "0.1")
        self.assertEqual(param_dict["use_attention"], "True")
        self.assertEqual(param_dict["optimizer"], "adam")

    def test_log_metrics_params_auto_run_creation(self) -> None:
        """Test that logging metrics/params without an active run creates a new run automatically."""
        # Log metric - should create a new run automatically
        self.exp.log_metric("accuracy", 0.95)

        # Log parameter - should use the same auto-created run
        self.exp.log_param("model", "bert")

        # Log more data to the auto-created run
        self.exp.log_metrics({"precision": 0.92, "recall": 0.88}, step=2)
        self.exp.log_params({"epochs": 10, "learning_rate": 0.001})

        # Verify a run was created automatically in the default experiment
        self.assertEqual(self.exp._experiment.name, "DEFAULT")
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.DEFAULT").collect()
        self.assertEqual(len(runs), 1)

        # Parse and verify metadata contains all logged data
        metadata = json.loads(runs[0]["metadata"])
        self.assertEqual(metadata["status"], RunStatus.RUNNING.value)

        # Check metrics
        self.assertIn("metrics", metadata)
        self.assertEqual(len(metadata["metrics"]), 3)
        metric_dict = {m["name"]: m for m in metadata["metrics"]}
        self.assertEqual(metric_dict["accuracy"]["value"], 0.95)
        self.assertEqual(metric_dict["accuracy"]["step"], 0)  # Default step
        self.assertEqual(metric_dict["precision"]["value"], 0.92)
        self.assertEqual(metric_dict["precision"]["step"], 2)
        self.assertEqual(metric_dict["recall"]["value"], 0.88)
        self.assertEqual(metric_dict["recall"]["step"], 2)

        # Check parameters
        self.assertIn("parameters", metadata)
        self.assertEqual(len(metadata["parameters"]), 3)
        param_dict = {p["name"]: p["value"] for p in metadata["parameters"]}
        self.assertEqual(param_dict["model"], "bert")
        self.assertEqual(param_dict["epochs"], "10")
        self.assertEqual(param_dict["learning_rate"], "0.001")

        # End run
        self.exp.end_run()
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.DEFAULT").collect()
        self.assertEqual(len(runs), 1)
        self.assertEqual(json.loads(runs[0]["metadata"])["status"], RunStatus.FINISHED.value)

    def test_log_model(self) -> None:
        """Test that log_model works with experiment tracking"""
        experiment_name = "TEST_EXPERIMENT_LOG_MODEL"
        run_name = "TEST_RUN_LOG_MODEL"
        model_name = "TEST_MODEL"

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            model = xgb.XGBClassifier()
            model.fit(X, y)
            mv = self.exp.log_model(
                model,
                model_name=model_name,
                sample_input_data=X,
            )

        # Test that model exists
        models = self._session.sql(f"SHOW MODELS IN DATABASE {self._db_name}").collect()
        self.assertEqual(len(models), 1)
        self.assertEqual(model_name, models[0]["name"])
        self.assertEqual(self._schema_name, models[0]["schema_name"])
        self.assertEqual(self._db_name, models[0]["database_name"])
        self.assertIn(mv.version_name, models[0]["versions"])

        # Test that the model version can be run and the output is correct
        actual = mv.run(X, function_name="predict")
        expected = pd.DataFrame({"output_feature_0": model.predict(X)})
        pd.testing.assert_frame_equal(actual, expected)

    def test_log_artifact_file(self) -> None:
        experiment_name = "TEST_EXPERIMENT_LOG_ARTIFACT_FILE"
        run_name = "TEST_RUN_LOG_ARTIFACT_FILE"
        local_path = "tests/integ/snowflake/ml/experiment/test_artifact.json"

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            self.exp.log_artifact(local_path)

        # Test that the artifact is logged correctly
        artifacts = self.exp.list_artifacts(run_name=run_name)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].name, "test_artifact.json")

        # Test that the artifact can be retrieved and that the content is the same
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(run_name=run_name, target_path=temp_dir)
            with (
                open(os.path.join(temp_dir, "test_artifact.json")) as uploaded_file,
                open(local_path) as original_file,
            ):
                self.assertEqual(uploaded_file.read(), original_file.read())

        # Test downloading a specific file path
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(run_name=run_name, target_path=temp_dir, artifact_path="test_artifact.json")
            self.assertEqual(os.listdir(temp_dir), ["test_artifact.json"])
            with (
                open(os.path.join(temp_dir, "test_artifact.json")) as uploaded_file,
                open(local_path) as original_file,
            ):
                self.assertEqual(uploaded_file.read(), original_file.read())

    def test_log_artifact_directory(self) -> None:
        experiment_name = "TEST_EXPERIMENT_LOG_ARTIFACT_DIR"
        run_name = "TEST_RUN_LOG_ARTIFACT_DIR"
        local_path = "tests/integ/snowflake/ml/experiment/test_artifact_dir"

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            self.exp.log_artifact(local_path)

        # Test that the artifacts are logged correctly
        expected_artifacts = [
            "artifact1.txt",
            "artifact2.py",
            "nested_dir/artifact3.md",
        ]
        artifacts = self.exp.list_artifacts(run_name=run_name)
        self.assertListEqual(expected_artifacts, [a.name for a in artifacts])

        # Test that artifacts can be retrieved and that the content is the same
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(run_name=run_name, target_path=temp_dir)
            for expected_artifact in expected_artifacts:
                with (
                    open(os.path.join(temp_dir, expected_artifact)) as uploaded_file,
                    open(os.path.join(local_path, expected_artifact)) as original_file,
                ):
                    self.assertEqual(uploaded_file.read(), original_file.read())

        # Test downloading a specific path, should only download the artifact3.md file
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(
                run_name=run_name, target_path=temp_dir, artifact_path="nested_dir/artifact3.md"
            )
            self.assertEqual(os.listdir(temp_dir), ["nested_dir"])
            with (
                open(os.path.join(temp_dir, "nested_dir/artifact3.md")) as uploaded_file,
                open(os.path.join(local_path, "nested_dir/artifact3.md")) as original_file,
            ):
                self.assertEqual(uploaded_file.read(), original_file.read())


if __name__ == "__main__":
    absltest.main()
