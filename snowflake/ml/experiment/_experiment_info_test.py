import unittest.mock as mock

from absl.testing import absltest

from snowflake.ml.experiment import _experiment_info as experiment_info
from snowflake.ml.registry._manager.model_manager import ModelManager
from snowflake.ml.test_utils.mock_progress import create_mock_progress_status


class ExperimentInfoPatcherTest(absltest.TestCase):
    """Test cases for ExperimentInfoPatcher context manager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.experiment_info = experiment_info.ExperimentInfo(
            fully_qualified_name="test.experiment.name", run_name="test_run"
        )
        self.original_log_model = ModelManager.log_model

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clear the stack
        experiment_info.ExperimentInfoPatcher._experiment_info_stack.clear()
        # Restore the original method
        ModelManager.log_model = self.original_log_model  # type: ignore[method-assign]

    def test_context_manager_basic_usage(self) -> None:
        """Test basic context manager functionality."""
        patcher = experiment_info.ExperimentInfoPatcher(self.experiment_info)

        # Test __enter__ returns self
        with patcher as context_patcher:
            self.assertIs(context_patcher, patcher)

            # Verify experiment info is added to stack
            self.assertEqual(len(experiment_info.ExperimentInfoPatcher._experiment_info_stack), 1)
            self.assertEqual(experiment_info.ExperimentInfoPatcher._experiment_info_stack[0], self.experiment_info)

            # Verify method is patched (different from original)
            self.assertIsNot(ModelManager.log_model, self.original_log_model)

        # After exiting, stack should be empty and method restored
        self.assertEqual(len(experiment_info.ExperimentInfoPatcher._experiment_info_stack), 0)
        self.assertIs(ModelManager.log_model, self.original_log_model)

    def test_method_restoration_on_exception(self) -> None:
        """Test that the original method is restored even when an exception occurs."""
        patcher = experiment_info.ExperimentInfoPatcher(self.experiment_info)

        with self.assertRaises(ValueError):
            with patcher:
                # Verify method is patched
                self.assertIsNot(ModelManager.log_model, self.original_log_model)

                # Verify experiment info is in stack
                self.assertEqual(len(experiment_info.ExperimentInfoPatcher._experiment_info_stack), 1)

                raise ValueError("Test exception")

        # Verify method is restored and stack is cleared even after exception
        self.assertIs(ModelManager.log_model, self.original_log_model)
        self.assertEqual(len(experiment_info.ExperimentInfoPatcher._experiment_info_stack), 0)

    def test_patching_only_happens_on_first_context(self) -> None:
        """Test that patching only happens when entering the first context."""
        experiment_info_1 = experiment_info.ExperimentInfo(fully_qualified_name="experiment.1", run_name="run_1")
        experiment_info_2 = experiment_info.ExperimentInfo(fully_qualified_name="experiment.2", run_name="run_2")

        with experiment_info.ExperimentInfoPatcher(experiment_info_1):
            patched_method = ModelManager.log_model

            # Verify method was patched
            self.assertIsNot(patched_method, self.original_log_model)

            with experiment_info.ExperimentInfoPatcher(experiment_info_2):
                # Verify method is the same (not re-patched)
                self.assertIs(ModelManager.log_model, patched_method)

            # Verify method is still the same after exiting inner context
            self.assertIs(ModelManager.log_model, patched_method)

        # Verify method is restored only after exiting all contexts
        self.assertIs(ModelManager.log_model, self.original_log_model)

    def test_preserves_original_method_signature_and_return_value(self) -> None:
        """Test that the patched method preserves original method's behavior."""
        # Mock the original method with specific return value
        expected_result = mock.MagicMock()
        mock_original = mock.MagicMock(return_value=expected_result)
        experiment_info.ExperimentInfoPatcher._original_log_model = mock_original

        mock_manager = mock.MagicMock(spec=ModelManager)
        mock_progress_status = create_mock_progress_status()

        with experiment_info.ExperimentInfoPatcher(self.experiment_info):
            # Call with various arguments
            result = ModelManager.log_model(
                mock_manager,
                model="test_model",
                model_name="test_name",
                version_name="v1.0",
                comment="test comment",
                metrics={"accuracy": 0.95},
                progress_status=mock_progress_status,
            )

            # Verify all arguments are passed through with experiment_info added
            mock_original.assert_called_once_with(
                mock_manager,
                model="test_model",
                model_name="test_name",
                version_name="v1.0",
                comment="test comment",
                metrics={"accuracy": 0.95},
                progress_status=mock_progress_status,
                experiment_info=self.experiment_info,
            )

            # Verify return value is preserved
            self.assertIs(result, expected_result)


if __name__ == "__main__":
    absltest.main()
