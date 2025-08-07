import json
import os
import pickle
import sys
import tempfile
import time
from typing import Any, Optional
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import constants, interop_utils
from snowflake.ml.jobs._utils.scripts import mljob_launcher
from snowflake.ml.jobs._utils.test_file_helper import resolve_path


class MLJobLauncherTests(parameterized.TestCase):
    def setUp(self) -> None:
        # Set up test environment
        self.temp_dir = tempfile.TemporaryDirectory()
        self.result_path = os.path.join(self.temp_dir.name, "mljob_result.pkl")
        self.result_json_path = os.path.join(self.temp_dir.name, "mljob_result.json")

        # Override launcher's result path
        mljob_launcher.JOB_RESULT_PATH = self.result_path

        # Test script paths
        self.test_dir = resolve_path("mljob_launcher_tests")
        self.simple_script = os.path.join(self.test_dir, "simple_script.py")
        self.function_script = os.path.join(self.test_dir, "function_script.py")
        self.error_script = os.path.join(self.test_dir, "error_script.py")
        self.complex_script = os.path.join(self.test_dir, "complex_result_script.py")
        self.argument_script = os.path.join(self.test_dir, "argument_script.py")

    def tearDown(self) -> None:
        # Clean up
        self.temp_dir.cleanup()
        if constants.RESULT_PATH_ENV_VAR in os.environ:
            del os.environ[constants.RESULT_PATH_ENV_VAR]

    def test_run_script_simple(self) -> None:
        # Test running a simple script
        result = mljob_launcher.main(self.simple_script)
        self.assertTrue(result.success)
        self.assertEqual(result.result["status"], "success")
        self.assertEqual(result.result["value"], 42)

    def test_run_script_with_function(self) -> None:
        # Test running a script with a specified main function
        result = mljob_launcher.main(self.function_script, script_main_func="main_function")
        self.assertTrue(result.success)
        self.assertEqual(result.result["status"], "success from function")
        self.assertEqual(result.result["value"], 100)

    @parameterized.parameters(  # type: ignore[misc]
        (100, {"status": "success from another function", "value": 100}),
        (0, {"status": "success from another function", "value": 0}),
        (None, {"status": "success from another function", "value": 0}),
    )
    def test_run_script_with_function_and_args(self, arg_value: Optional[int], expected: dict[str, Any]) -> None:
        # Test running a script with a function that takes arguments
        args = [] if arg_value is None else [arg_value]
        result = mljob_launcher.main(self.function_script, *args, script_main_func="another_function")
        self.assertTrue(result.success)
        self.assertEqual(result.result, expected)

    def test_run_script_invalid_function(self) -> None:
        # Test error when function doesn't exist
        with self.assertRaises(RuntimeError):
            mljob_launcher.main(self.function_script, script_main_func="nonexistent_function")

    def test_run_script_with_args(self) -> None:
        # Test running a script with arguments
        result = mljob_launcher.main(self.argument_script, "arg1", "arg2", "--named_arg=value")
        self.assertTrue(result.success)
        self.assertListEqual(result.result["args"], ["arg1", "arg2", "--named_arg=value"])

    def test_main_success(self) -> None:
        # Test the main function with successful execution
        try:
            result_obj = mljob_launcher.main(self.simple_script)
            self.assertTrue(result_obj.success)
            self.assertEqual(result_obj.result["value"], 42)

            # Check serialized results
            with open(self.result_path, "rb") as f:
                pickled_result: dict[str, Any] = pickle.load(f)
            pickled_result_obj = interop_utils.ExecutionResult.from_dict(pickled_result)
            self.assertTrue(pickled_result_obj.success)
            assert isinstance(pickled_result_obj.result, dict)
            self.assertEqual(pickled_result_obj.result["value"], 42)

            with open(self.result_json_path) as f:
                json_result: dict[str, Any] = json.load(f)
            json_result_obj = interop_utils.ExecutionResult.from_dict(json_result)
            self.assertTrue(json_result_obj.success)
            assert isinstance(json_result_obj.result, dict)
            self.assertEqual(json_result_obj.result["value"], 42)
        except Exception as e:
            self.fail(f"main() raised exception unexpectedly: {e}")

    def test_main_error(self) -> None:
        # Test the main function with script that raises an error
        with self.assertRaises(RuntimeError):
            mljob_launcher.main(self.error_script)

        # Check serialized error results
        with open(self.result_path, "rb") as f:
            pickled_result: dict[str, Any] = pickle.load(f)
        pickled_result_obj = interop_utils.ExecutionResult.from_dict(pickled_result)
        self.assertFalse(pickled_result_obj.success)
        self.assertEqual(type(pickled_result_obj.exception), RuntimeError)
        self.assertIn("Test error from script", str(pickled_result_obj.exception))
        pickled_exc_tb = pickled_result.get("exc_tb")
        self.assertIsInstance(pickled_exc_tb, str)
        self.assertNotIn("mljob_launcher.py", pickled_exc_tb)
        self.assertNotIn("runpy", pickled_exc_tb)

        with open(self.result_json_path) as f:
            json_result: dict[str, Any] = json.load(f)
        json_result_obj = interop_utils.ExecutionResult.from_dict(json_result)
        self.assertFalse(json_result_obj.success)
        self.assertEqual(type(json_result_obj.exception), RuntimeError)
        self.assertIn("Test error from script", str(json_result_obj.exception))
        json_exc_tb = json_result.get("exc_tb")
        self.assertIsInstance(json_exc_tb, str)
        self.assertNotIn("mljob_launcher.py", json_exc_tb)
        self.assertNotIn("runpy", json_exc_tb)

    def test_function_error(self) -> None:
        # Test error in a function
        with self.assertRaises(ValueError):
            mljob_launcher.main(self.error_script, script_main_func="error_function")

        # Check serialized error results
        with open(self.result_path, "rb") as f:
            pickled_result = pickle.load(f)
        pickled_result_obj = interop_utils.ExecutionResult.from_dict(pickled_result)
        self.assertFalse(pickled_result_obj.success)
        self.assertEqual(type(pickled_result_obj.exception), ValueError)
        self.assertIn("Test error from function", str(pickled_result_obj.exception))

    def test_complex_result_serialization(self) -> None:
        # Import needed to test complex result serialization
        sys.path.append(str(resolve_path("mljob_launcher_tests")))
        from custom_object_type import CustomObject

        # Test handling of complex, non-JSON-serializable results
        try:
            result_obj = mljob_launcher.main(self.complex_script)
            self.assertTrue(result_obj.success)

            # Check serialized results - pickle should handle complex objects
            with open(self.result_path, "rb") as f:
                pickled_result = pickle.load(f)
            pickled_result_obj = interop_utils.ExecutionResult.from_dict(pickled_result)
            self.assertTrue(pickled_result_obj.success)
            assert isinstance(pickled_result_obj.result, dict)
            self.assertIsInstance(pickled_result_obj.result["custom"], CustomObject)

            # JSON should convert non-serializable objects to strings
            with open(self.result_json_path) as f:
                json_result = json.load(f)
            json_result_obj = interop_utils.ExecutionResult.from_dict(json_result)
            self.assertTrue(json_result_obj.success)
            assert isinstance(json_result_obj.result, dict)
            self.assertIsInstance(json_result_obj.result["custom"], str)
            self.assertIn("CustomObject", json_result_obj.result["custom"])
        except Exception as e:
            self.fail(f"main() raised exception unexpectedly: {e}")

    def test_invalid_script_path(self) -> None:
        # Test with non-existent script path
        nonexistent_path = os.path.join(self.test_dir, "nonexistent_script.py")
        with self.assertRaises(FileNotFoundError):
            mljob_launcher.main(nonexistent_path)

    @absltest.mock.patch("cloudpickle.dump")  # type: ignore[misc]
    def test_result_pickling_error(self, mock_dump: absltest.mock.MagicMock) -> None:
        # Test handling of pickling errors by creating an unpicklable result
        # (by monkeypatching cloudpickle.dump to raise an exception)
        mock_dump.side_effect = pickle.PicklingError("Mocked pickling error")
        with self.assertWarns(RuntimeWarning):
            mljob_launcher.main(self.simple_script)

    @mock.patch.dict("sys.modules", {"common_utils": mock.MagicMock()})
    @mock.patch("common_utils.common_util")
    def test_wait_for_instances_target_instances_one_or_less(self, mock_common_util: mock.MagicMock) -> None:
        """Test that wait_for_instances returns immediately when target_instances <= 1."""
        # Should return immediately without checking Ray nodes
        mljob_launcher.wait_for_instances(1, 1)
        mock_common_util.get_num_ray_nodes.assert_not_called()

        # Test with target_instances = 0, which should raise ValueError
        with self.assertRaises(ValueError):
            mljob_launcher.wait_for_instances(1, 0)
        mock_common_util.get_num_ray_nodes.assert_not_called()

    @mock.patch.dict("sys.modules", {"common_utils": mock.MagicMock()})
    @mock.patch("common_utils.common_util")
    def test_wait_for_instances_target_met_immediately(self, mock_common_util: mock.MagicMock) -> None:
        """Test that wait_for_instances returns immediately when target instances are already available."""
        mock_common_util.get_num_ray_nodes.return_value = 5

        start_time = time.time()
        mljob_launcher.wait_for_instances(2, 5)
        elapsed = time.time() - start_time

        # Should return very quickly
        self.assertLess(elapsed, 0.1)
        mock_common_util.get_num_ray_nodes.assert_called_once()

    @mock.patch.dict("sys.modules", {"common_utils": mock.MagicMock()})
    @mock.patch("common_utils.common_util")
    @mock.patch("time.sleep")
    def test_wait_for_instances_target_met_after_wait(
        self, mock_sleep: mock.MagicMock, mock_common_util: mock.MagicMock
    ) -> None:
        """Test that wait_for_instances waits and returns when target instances become available."""
        # Simulate nodes becoming available over time (not enough minimum instances initially)
        mock_common_util.get_num_ray_nodes.side_effect = [1, 2, 5]

        mljob_launcher.wait_for_instances(2, 5, timeout=60, check_interval=10, min_wait_time=5)

        # Should have checked 3 times
        self.assertEqual(mock_common_util.get_num_ray_nodes.call_count, 3)
        # Should have slept twice (between checks)
        self.assertEqual(mock_sleep.call_count, 2)
        # Check exponential backoff: first sleep is 1s, second is 2s
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @mock.patch.dict("sys.modules", {"common_utils": mock.MagicMock()})
    @mock.patch("common_utils.common_util")
    @mock.patch("time.sleep")
    def test_wait_for_instances_minimum_met_after_min_wait_time(
        self, mock_sleep: mock.MagicMock, mock_common_util: mock.MagicMock
    ) -> None:
        """Test that wait_for_instances returns when minimum instances are met after min_wait_time."""
        # Simulate having enough minimum instances but not target instances
        mock_common_util.get_num_ray_nodes.return_value = 3

        # Mock time.time to simulate elapsed time > min_wait_time
        with mock.patch("time.time") as mock_time:
            # First call: start_time, second call: elapsed > min_wait_time
            # Add more return values for potential logging calls
            mock_time.side_effect = [0, 10, 10, 10, 10, 10]  # 10 seconds elapsed > 5 seconds min_wait_time

            mljob_launcher.wait_for_instances(2, 5, min_wait_time=5, timeout=60, check_interval=10)

        # Should have checked once and returned
        self.assertEqual(mock_common_util.get_num_ray_nodes.call_count, 1)
        mock_sleep.assert_not_called()

    @mock.patch.dict("sys.modules", {"common_utils": mock.MagicMock()})
    @mock.patch("common_utils.common_util")
    @mock.patch("time.sleep")
    def test_wait_for_instances_exponential_backoff(
        self, mock_sleep: mock.MagicMock, mock_common_util: mock.MagicMock
    ) -> None:
        """Test that wait_for_instances uses exponential backoff up to max check_interval."""
        # Simulate nodes never becoming available until the end
        mock_common_util.get_num_ray_nodes.side_effect = [1, 1, 1, 1, 1, 5]

        mljob_launcher.wait_for_instances(2, 5, timeout=60, check_interval=8)

        # Should have checked 6 times
        self.assertEqual(mock_common_util.get_num_ray_nodes.call_count, 6)
        # Should have slept 5 times (between checks)
        self.assertEqual(mock_sleep.call_count, 5)

        # Check exponential backoff: 1, 2, 4, 8, 8 (capped at check_interval)
        expected_sleeps = [1, 2, 4, 8, 8]
        actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertEqual(actual_sleeps, expected_sleeps)

    def test_wait_for_instances_invalid_arguments(self) -> None:
        """Test that wait_for_instances raises ValueError for invalid arguments."""
        with self.assertRaises(ValueError) as cm:
            mljob_launcher.wait_for_instances(5, 3)  # min_instances > target_instances

        self.assertIn("Minimum instances (5) cannot be greater than target instances (3)", str(cm.exception))

    @mock.patch.dict("sys.modules", {"common_utils": mock.MagicMock()})
    @mock.patch("common_utils.common_util")
    @mock.patch("time.sleep")
    def test_wait_for_instances_timeout(self, mock_sleep: mock.MagicMock, mock_common_util: mock.MagicMock) -> None:
        """Test that wait_for_instances raises TimeoutError when timeout is reached."""
        # Simulate nodes never becoming available
        mock_common_util.get_num_ray_nodes.return_value = 1

        with mock.patch("time.time") as mock_time:
            # Simulate timeout being reached
            mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6]  # 6 seconds elapsed > 5 seconds timeout

            with self.assertRaises(TimeoutError) as cm:
                mljob_launcher.wait_for_instances(2, 5, timeout=5, check_interval=10)

            self.assertIn("Timed out after 6s waiting for 2 instances", str(cm.exception))
            self.assertIn("only 1 available", str(cm.exception))


if __name__ == "__main__":
    absltest.main()
