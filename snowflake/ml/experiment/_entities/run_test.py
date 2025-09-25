import unittest.mock as mock
from unittest.mock import MagicMock

from absl.testing import absltest

import snowflake.ml.experiment._entities as entities
import snowflake.ml.experiment._experiment_info as experiment_info
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment import experiment_tracking
from snowflake.ml.experiment._client import experiment_tracking_sql_client as sql_client
from snowflake.ml.registry._manager import model_manager


class RunTest(absltest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock experiment tracking
        self.mock_experiment_tracking = MagicMock(spec=experiment_tracking.ExperimentTracking)

        # Create mock SQL client
        self.mock_sql_client = MagicMock(spec=sql_client.ExperimentTrackingSQLClient)
        self.mock_experiment_tracking._sql_client = self.mock_sql_client

        # Set up mock database and schema names
        self.mock_experiment_tracking._database_name = sql_identifier.SqlIdentifier("TEST_DB")
        self.mock_experiment_tracking._schema_name = sql_identifier.SqlIdentifier("TEST_SCHEMA")

        # Set up test identifiers
        self.experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        self.run_name = sql_identifier.SqlIdentifier("TEST_RUN")

        # Mock the fully_qualified_object_name method
        expected_fqn = "TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT"
        self.mock_sql_client.fully_qualified_object_name.return_value = expected_fqn

        # Clear any existing experiment info stack state
        experiment_info.ExperimentInfoPatcher._experiment_info_stack.clear()

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clear experiment info stack
        experiment_info.ExperimentInfoPatcher._experiment_info_stack.clear()

    def test_init_creates_patcher_with_experiment_info(self) -> None:
        """Test that Run initialization creates a patcher with correct experiment info."""
        with mock.patch.object(
            experiment_info.ExperimentInfoPatcher, "__init__", return_value=None
        ) as mock_patcher_init:
            entities.Run(
                experiment_tracking=self.mock_experiment_tracking,
                experiment_name=self.experiment_name,
                run_name=self.run_name,
            )

            # Verify ExperimentInfoPatcher was initialized with correct experiment info
            mock_patcher_init.assert_called_once()
            _, kwargs = mock_patcher_init.call_args
            experiment_info_obj = kwargs["experiment_info"]

            self.assertIsInstance(experiment_info_obj, experiment_info.ExperimentInfo)
            self.assertEqual(experiment_info_obj.fully_qualified_name, "TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT")
            self.assertEqual(experiment_info_obj.run_name, self.run_name)

    def test_context_manager_patches_model_manager_log_model(self) -> None:
        """Test that using Run as context manager patches ModelManager.log_model correctly."""

        unpatched = model_manager.ModelManager.log_model

        run = entities.Run(
            experiment_tracking=self.mock_experiment_tracking,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
        )
        self.mock_experiment_tracking._run = run

        # Use as context manager and call log_model
        with run:
            # Verify that the original log_model method is not the same as the mocked log_model
            self.assertIsNot(model_manager.ModelManager.log_model, unpatched)

        # Verify that the original log_model method is restored
        self.assertIs(model_manager.ModelManager.log_model, unpatched)

    def test_context_manager_usage(self) -> None:
        """Test Run as a context manager in typical usage."""
        run = entities.Run(
            experiment_tracking=self.mock_experiment_tracking,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
        )

        # Set this run as the active run
        self.mock_experiment_tracking._run = run

        # Use as context manager
        with run as context_run:
            self.assertIs(context_run, run)

        # Verify end_run was called after exiting context
        self.mock_experiment_tracking.end_run.assert_called_once()

    def test_context_manager_with_exception(self) -> None:
        """Test Run as a context manager when exception occurs."""
        run = entities.Run(
            experiment_tracking=self.mock_experiment_tracking,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
        )

        # Set this run as the active run
        self.mock_experiment_tracking._run = run

        # Use as context manager with exception
        with self.assertRaises(ValueError):
            with run:
                raise ValueError("Test exception in context")

        # Verify end_run was still called even with exception
        self.mock_experiment_tracking.end_run.assert_called_once()


if __name__ == "__main__":
    absltest.main()
