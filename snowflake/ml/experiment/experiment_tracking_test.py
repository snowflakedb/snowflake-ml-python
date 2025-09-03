import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch
from urllib.parse import quote

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment import _entities as entities, experiment_tracking
from snowflake.ml.experiment._client import artifact
from snowflake.ml.experiment._entities import run_metadata
from snowflake.snowpark import session


class ExperimentTrackingTest(absltest.TestCase):
    def setUp(self) -> None:
        # Create a mock Snowpark session
        self.mock_session = MagicMock(spec=session.Session)

        # Set up mock return values for session methods
        self.mock_session.get_current_database.return_value = "TEST_DB"
        self.mock_session.get_current_schema.return_value = "TEST_SCHEMA"

        # Create a patcher for the ExperimentTrackingSQLClient
        self.sql_client_patcher = patch(
            "snowflake.ml.experiment.experiment_tracking.sql_client.ExperimentTrackingSQLClient"
        )
        self.mock_sql_client_class = self.sql_client_patcher.start()
        self.mock_sql_client = MagicMock()
        self.mock_sql_client_class.return_value = self.mock_sql_client

        # Create a patcher for the Registry
        self.registry_patcher = patch("snowflake.ml.experiment.experiment_tracking.registry.Registry")
        self.mock_registry_class = self.registry_patcher.start()
        self.mock_registry = MagicMock()
        self.mock_registry_class.return_value = self.mock_registry

    def tearDown(self) -> None:
        self.sql_client_patcher.stop()
        self.registry_patcher.stop()

    def test_init(self) -> None:
        """Test all initialization scenarios."""
        # Test with default parameters
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        self.assertEqual(exp._database_name.identifier(), "TEST_DB")
        self.assertEqual(exp._schema_name.identifier(), "TEST_SCHEMA")

        # Test with custom parameters
        exp = experiment_tracking.ExperimentTracking(
            session=self.mock_session, database_name="CUSTOM_DB", schema_name="CUSTOM_SCHEMA"
        )
        self.assertEqual(exp._database_name.identifier(), "CUSTOM_DB")
        self.assertEqual(exp._schema_name.identifier(), "CUSTOM_SCHEMA")

        # Test with no schema, should use PUBLIC schema
        self.mock_session.get_current_schema.return_value = None
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        self.assertEqual(exp._schema_name.identifier(), "PUBLIC")

        # Test with no database, should raise ValueError
        self.mock_session.get_current_database.return_value = None
        with self.assertRaises(ValueError):
            experiment_tracking.ExperimentTracking(session=self.mock_session)

    def test_set_experiment(self) -> None:
        """Test setting an experiment"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        experiment = exp.set_experiment("TEST_EXPERIMENT")

        # Verify SQL client was called to create the experiment
        self.mock_sql_client.create_experiment.assert_called_once()
        call_args = self.mock_sql_client.create_experiment.call_args[1]
        self.assertEqual(call_args["experiment_name"].identifier(), "TEST_EXPERIMENT")
        self.assertTrue(call_args["creation_mode"].if_not_exists)

        # Verify experiment object properties
        self.assertIsInstance(experiment, entities.Experiment)
        self.assertEqual(experiment.name.identifier(), "TEST_EXPERIMENT")

        # Verify the experiment is stored in the tracking context
        self.assertEqual(exp._experiment, experiment)

    def test_delete_experiment(self) -> None:
        """Test deleting an experiment."""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        experiment = exp.set_experiment("TEST_EXPERIMENT")
        self.assertIs(experiment, exp._experiment)
        exp.delete_experiment("TEST_EXPERIMENT")
        self.assertIsNone(exp._experiment)

        # Verify SQL client was called to delete the experiment
        self.mock_sql_client.drop_experiment.assert_called_once()
        call_args = self.mock_sql_client.drop_experiment.call_args[1]
        self.assertEqual(call_args["experiment_name"].identifier(), "TEST_EXPERIMENT")

    def test_start_run(self) -> None:
        """Test starting a run with a specific name"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Set up existing experiment
        experiment = exp.set_experiment("TEST_EXPERIMENT")

        # Start a run with specific name
        run = exp.start_run("TEST_RUN")

        # Verify SQL client was called to add the run
        self.mock_sql_client.add_run.assert_called_once()
        call_args = self.mock_sql_client.add_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], experiment.name)
        self.assertEqual(call_args["run_name"].identifier(), "TEST_RUN")

        # Verify Run object properties
        self.assertIsInstance(run, entities.Run)
        self.assertEqual(run.name.identifier(), "TEST_RUN")

        # Verify the run is stored in the tracking context
        self.assertEqual(exp._run, run)

        with self.assertRaises(RuntimeError) as context:
            exp.start_run("TEST_RUN_2")
        self.assertIn(
            "A run is already active. Please end the current run before starting a new one.", str(context.exception)
        )

    def test_end_run(self) -> None:
        """Test ending a run"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        exp.set_experiment("TEST_EXPERIMENT")
        run = exp.start_run("TEST_RUN")
        self.assertIs(run, exp._run)
        exp.end_run()
        self.assertIsNone(exp._run)

        # Verify SQL client was called to commit the run
        self.mock_sql_client.commit_run.assert_called_once()
        call_args = self.mock_sql_client.commit_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], "TEST_EXPERIMENT")
        self.assertEqual(call_args["run_name"], "TEST_RUN")

        with self.assertRaises(RuntimeError) as context:
            exp.end_run()
        self.assertIn("No run is active. Please start a run before ending it.", str(context.exception))

        # End a run with a specified name
        self.mock_sql_client.commit_run.reset_mock()
        exp.end_run("ANOTHER_RUN")
        self.mock_sql_client.commit_run.assert_called_once()
        call_args = self.mock_sql_client.commit_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], "TEST_EXPERIMENT")
        self.assertEqual(call_args["run_name"], "ANOTHER_RUN")

    def test_delete_run(self) -> None:
        """Test deleting a run"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Set up existing experiment
        experiment = exp.set_experiment("TEST_EXPERIMENT")
        run = exp.start_run("TEST_RUN")
        self.assertIs(run, exp._run)

        # Delete a run
        exp.delete_run("TEST_RUN")
        self.assertIsNone(exp._run)

        # Verify SQL client was called to drop the run
        self.mock_sql_client.drop_run.assert_called_once()
        call_args = self.mock_sql_client.drop_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], experiment.name)
        self.assertEqual(call_args["run_name"].identifier(), "TEST_RUN")
        self.assertIsNone(exp._run)

    def test_log_metrics_with_active_run(self) -> None:
        """Test logging metrics with an active run"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        exp.set_experiment("TEST_EXPERIMENT")
        run = exp.start_run("TEST_RUN")

        # Use actual run metadata
        metadata = run_metadata.RunMetadata(status=run_metadata.RunStatus.RUNNING, metrics=[], parameters=[])
        run._get_metadata = MagicMock(return_value=metadata)

        # Log metrics
        metrics = {"accuracy": 0.95, "loss": 0.05}
        exp.log_metrics(metrics, step=1)

        # Verify metrics were set in metadata
        self.assertEqual(len(metadata.metrics), 2)
        metric_dict = {(m.name, m.step): m.value for m in metadata.metrics}
        self.assertEqual(metric_dict[("accuracy", 1)], 0.95)
        self.assertEqual(metric_dict[("loss", 1)], 0.05)

        # Verify SQL client was called to modify the run
        self.mock_sql_client.modify_run.assert_called_once()
        call_args = self.mock_sql_client.modify_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], run.experiment_name)
        self.assertEqual(call_args["run_name"], run.name)
        # Verify JSON-serialized metadata was passed to SQL client
        self.assertEqual(call_args["run_metadata"], json.dumps(metadata.to_dict()))

    def test_log_metrics_without_active_run(self) -> None:
        """Test logging metrics creates a run when none is active"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Mock _get_or_start_run to return a new run
        mock_run = MagicMock(spec=entities.Run)
        mock_run.experiment_name = "TEST_EXPERIMENT"
        mock_run.name = "TEST_RUN"
        metadata = run_metadata.RunMetadata(status=run_metadata.RunStatus.RUNNING, metrics=[], parameters=[])
        mock_run._get_metadata.return_value = metadata
        exp._get_or_start_run = MagicMock(return_value=mock_run)

        # Log metrics
        metrics = {"precision": 0.85}
        exp.log_metrics(metrics, step=2)

        # Verify _get_or_start_run was called
        exp._get_or_start_run.assert_called_once()

        # Verify metric was set in metadata
        self.assertEqual(len(metadata.metrics), 1)
        self.assertEqual(metadata.metrics[0].name, "precision")
        self.assertEqual(metadata.metrics[0].value, 0.85)
        self.assertEqual(metadata.metrics[0].step, 2)

        # Verify SQL client was called to modify the run with correct args
        self.mock_sql_client.modify_run.assert_called_once()
        call_args = self.mock_sql_client.modify_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], "TEST_EXPERIMENT")
        self.assertEqual(call_args["run_name"], "TEST_RUN")
        # Verify JSON-serialized metadata was passed to SQL client
        self.assertEqual(call_args["run_metadata"], json.dumps(metadata.to_dict()))

    def test_log_params_with_active_run(self) -> None:
        """Test logging params with an active run"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        exp.set_experiment("TEST_EXPERIMENT")
        run = exp.start_run("TEST_RUN")

        # Use actual run metadata
        metadata = run_metadata.RunMetadata(status=run_metadata.RunStatus.RUNNING, metrics=[], parameters=[])
        run._get_metadata = MagicMock(return_value=metadata)

        # Log params
        params = {"learning_rate": 0.01, "batch_size": 32, "model_type": "RandomForest"}
        exp.log_params(params)

        # Verify params were set in metadata
        self.assertEqual(len(metadata.parameters), 3)
        param_dict = {p.name: p.value for p in metadata.parameters}
        self.assertEqual(param_dict["learning_rate"], "0.01")
        self.assertEqual(param_dict["batch_size"], "32")
        self.assertEqual(param_dict["model_type"], "RandomForest")

        # Verify SQL client was called to modify the run with correct args
        self.mock_sql_client.modify_run.assert_called_once()
        call_args = self.mock_sql_client.modify_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], run.experiment_name)
        self.assertEqual(call_args["run_name"], run.name)
        # Verify JSON-serialized metadata was passed to SQL client
        self.assertEqual(call_args["run_metadata"], json.dumps(metadata.to_dict()))

    def test_log_params_without_active_run(self) -> None:
        """Test logging params creates a run when none is active"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Mock _get_or_start_run to return a new run
        mock_run = MagicMock(spec=entities.Run)
        mock_run.experiment_name = "TEST_EXPERIMENT"
        mock_run.name = "TEST_RUN"
        metadata = run_metadata.RunMetadata(status=run_metadata.RunStatus.RUNNING, metrics=[], parameters=[])
        mock_run._get_metadata.return_value = metadata
        exp._get_or_start_run = MagicMock(return_value=mock_run)

        # Log params
        params = {"algorithm": "gradient_boosting"}
        exp.log_params(params)

        # Verify _get_or_start_run was called
        exp._get_or_start_run.assert_called_once()

        # Verify param was set in metadata
        self.assertEqual(len(metadata.parameters), 1)
        self.assertEqual(metadata.parameters[0].name, "algorithm")
        self.assertEqual(metadata.parameters[0].value, "gradient_boosting")

        # Verify SQL client was called to modify the run with correct args
        self.mock_sql_client.modify_run.assert_called_once()
        call_args = self.mock_sql_client.modify_run.call_args[1]
        self.assertEqual(call_args["experiment_name"], "TEST_EXPERIMENT")
        self.assertEqual(call_args["run_name"], "TEST_RUN")
        # Verify JSON-serialized metadata was passed to SQL client
        self.assertEqual(call_args["run_metadata"], json.dumps(metadata.to_dict()))

    def test_log_model(self) -> None:
        """Test that log_model uses ExperimentInfoPatcher with correct experiment info"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Mock _get_or_start_run to return a mock run
        mock_run = MagicMock()
        mock_experiment_info = MagicMock()
        exp._experiment = MagicMock()
        exp._run = mock_run
        mock_run._get_experiment_info.return_value = mock_experiment_info
        mock_model = MagicMock()

        # Mock ExperimentInfoPatcher
        with patch("snowflake.ml.experiment._experiment_info.ExperimentInfoPatcher") as mock_patcher_class:
            mock_patcher = MagicMock()
            mock_patcher_class.return_value = mock_patcher

            # Call log_model with test arguments
            exp.log_model(mock_model, model_name="test", version_name="v1")

            # Verify ExperimentInfoPatcher was created with the correct experiment info
            mock_patcher_class.assert_called_once_with(experiment_info=mock_experiment_info)

            # Verify the patcher was used as a context manager
            mock_patcher.__enter__.assert_called_once()
            mock_patcher.__exit__.assert_called_once()

            # Verify registry.log_model was called with original arguments
            self.mock_registry.log_model.assert_called_once_with(mock_model, model_name="test", version_name="v1")

    def test_log_artifact(self) -> None:
        """Test logging artifact with nested artifact path"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        exp.set_experiment("TEST_EXPERIMENT")
        run = exp.start_run("TEST_RUN")
        # Mock artifact expansion to simulate nested paths
        mock_pairs = [
            ("/tmp/file1.txt", "nested/path"),
            ("/tmp/subdir/file2.bin", "nested/path/subdir"),
        ]
        with patch(
            "snowflake.ml.experiment._client.artifact.get_put_path_pairs",
            return_value=mock_pairs,
        ) as mock_get_pairs:
            exp.log_artifact(local_path="/local/dir", artifact_path="nested/path")

            # Verify expansion was called with provided args
            mock_get_pairs.assert_called_once_with("/local/dir", "nested/path")

        # Verify put_artifact was called for each pair with correct kwargs
        self.assertEqual(self.mock_sql_client.put_artifact.call_count, len(mock_pairs))
        for idx, (file_path, dest_artifact_path) in enumerate(mock_pairs):
            call_kwargs = self.mock_sql_client.put_artifact.call_args_list[idx][1]
            self.assertEqual(call_kwargs["experiment_name"], run.experiment_name)
            self.assertEqual(call_kwargs["run_name"], run.name)
            self.assertEqual(call_kwargs["artifact_path"], dest_artifact_path)
            self.assertEqual(call_kwargs["file_path"], file_path)

    def test_delete_run_no_experiment_raises_error(self) -> None:
        """Test that deleting a run without an experiment raises RuntimeError"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Try to delete run without setting experiment
        with self.assertRaises(RuntimeError) as context:
            exp.delete_run("TEST_RUN")

        self.assertIn("No experiment set. Please set an experiment before deleting a run.", str(context.exception))

    def test_print_urls(self) -> None:
        """Test _print_urls method properly URL encodes names with special characters"""
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Create test data with special characters that need URL encoding
        experiment_name = sql_identifier.SqlIdentifier('"ExPerIment w $ymb0ls! #"')
        run_name = sql_identifier.SqlIdentifier('"RUN WITH SPACES"')

        # Mock stdout to capture output
        mock_stdout = StringIO()
        with patch.object(sys, "stdout", mock_stdout):
            exp._print_urls(experiment_name=experiment_name, run_name=run_name)

        # Get the captured output
        output = mock_stdout.getvalue()

        # Construct expected URLs with URL-encoded special characters
        expected_experiment_url = (
            f"https://app.snowflake.com/_deeplink/#/experiments"
            f"/databases/{quote('TEST_DB')}"
            f"/schemas/{quote('TEST_SCHEMA')}"
            f"""/experiments/{quote('"ExPerIment w $ymb0ls! #"')}"""
        )
        expected_run_url = expected_experiment_url + f"""/runs/{quote('"RUN WITH SPACES"')}"""

        # Verify both lines are in output with proper URL encoding
        expected_run_line = f'🏃 View run "RUN WITH SPACES" at: {expected_run_url}\n'
        expected_experiment_line = f"🧪 View experiment at: {expected_experiment_url}\n"

        self.assertIn(expected_run_line, output)
        self.assertIn(expected_experiment_line, output)

    def test_list_artifacts(self) -> None:
        # Prepare fake return from sql client
        expected = [
            artifact.ArtifactInfo(name="file1.txt", size=10, md5="aaa", last_modified="2024-01-01"),
            artifact.ArtifactInfo(name="dir/file2.bin", size=20, md5="bbb", last_modified="2024-01-02"),
        ]
        self.mock_sql_client.list_artifacts.return_value = expected

        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)

        # Test that listing artifacts without an experiment raises an error
        with self.assertRaises(RuntimeError) as ctx:
            exp.list_artifacts(run_name="RUN1")
        self.assertIn("No experiment set. Please set an experiment before listing artifacts.", str(ctx.exception))

        # Test that listing artifacts with an experiment works
        exp.set_experiment("EXP1")
        res = exp.list_artifacts(run_name="RUN1")

        # Verify delegation to SQL client with root path
        self.mock_sql_client.list_artifacts.assert_called_once()
        kwargs = self.mock_sql_client.list_artifacts.call_args[1]
        self.assertEqual(kwargs["experiment_name"].identifier(), "EXP1")
        self.assertEqual(kwargs["run_name"].identifier(), "RUN1")
        self.assertEqual(kwargs["artifact_path"], "")
        self.assertIs(res, expected)

    def test_download_artifacts(self) -> None:
        # Scenario 1: raises without experiment
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        with self.assertRaises(RuntimeError) as ctx:
            exp.download_artifacts(run_name="RUN1")
        self.assertIn("No experiment set. Please set an experiment before downloading artifacts.", str(ctx.exception))

        # Scenario 2: explicit artifact_path and target_path
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        exp.set_experiment("EXP1")
        expected_artifacts_1 = [
            artifact.ArtifactInfo(name="sub/dir/a.txt", size=1, md5="md5a", last_modified="2024-01-01"),
            artifact.ArtifactInfo(name="sub/dir/b.bin", size=2, md5="md5b", last_modified="2024-01-02"),
        ]
        self.mock_sql_client.list_artifacts.reset_mock()
        self.mock_sql_client.get_artifact.reset_mock()
        self.mock_sql_client.list_artifacts.return_value = expected_artifacts_1
        pairs_1 = [("sub/dir/a.txt", "/tmp/out"), ("sub/dir/b.bin", "/tmp/out/sub")]
        with patch(
            "snowflake.ml.experiment._client.artifact.get_download_path_pairs",
            return_value=pairs_1,
        ) as mock_get_pairs:
            exp.download_artifacts(run_name="RUN1", artifact_path="sub/dir", target_path="/tmp/out")
            self.mock_sql_client.list_artifacts.assert_called_once()
            list_kwargs = self.mock_sql_client.list_artifacts.call_args[1]
            self.assertEqual(list_kwargs["experiment_name"].identifier(), "EXP1")
            self.assertEqual(list_kwargs["run_name"].identifier(), "RUN1")
            self.assertEqual(list_kwargs["artifact_path"], "sub/dir")
            mock_get_pairs.assert_called_once_with(expected_artifacts_1, "/tmp/out")
        self.assertEqual(self.mock_sql_client.get_artifact.call_count, len(pairs_1))
        for idx, (rel_path, local_dir) in enumerate(pairs_1):
            call_kwargs = self.mock_sql_client.get_artifact.call_args_list[idx][1]
            self.assertEqual(call_kwargs["experiment_name"].identifier(), "EXP1")
            self.assertEqual(call_kwargs["run_name"].identifier(), "RUN1")
            self.assertEqual(call_kwargs["artifact_path"], rel_path)
            self.assertEqual(call_kwargs["target_path"], local_dir)

        # Scenario 3: defaults for artifact_path and target_path
        exp = experiment_tracking.ExperimentTracking(session=self.mock_session)
        exp.set_experiment("EXP2")
        expected_artifacts_2: list[artifact.ArtifactInfo] = []
        self.mock_sql_client.list_artifacts.reset_mock()
        self.mock_sql_client.get_artifact.reset_mock()
        self.mock_sql_client.list_artifacts.return_value = expected_artifacts_2
        pairs_2 = [("a.txt", ""), ("b.bin", "subdir")]
        with patch(
            "snowflake.ml.experiment._client.artifact.get_download_path_pairs",
            return_value=pairs_2,
        ) as mock_get_pairs:
            exp.download_artifacts(run_name="RUN2")
            list_kwargs = self.mock_sql_client.list_artifacts.call_args[1]
            self.assertEqual(list_kwargs["artifact_path"], "")
            mock_get_pairs.assert_called_once_with(expected_artifacts_2, "")
        self.assertEqual(self.mock_sql_client.get_artifact.call_count, len(pairs_2))
        for idx, (rel_path, local_dir) in enumerate(pairs_2):
            call_kwargs = self.mock_sql_client.get_artifact.call_args_list[idx][1]
            self.assertEqual(call_kwargs["experiment_name"].identifier(), "EXP2")
            self.assertEqual(call_kwargs["run_name"].identifier(), "RUN2")
            self.assertEqual(call_kwargs["artifact_path"], rel_path)
            self.assertEqual(call_kwargs["target_path"], local_dir)


if __name__ == "__main__":
    absltest.main()
