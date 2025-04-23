import json
import os
import pickle
import sys
import tempfile
from typing import Any, Optional

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


if __name__ == "__main__":
    absltest.main()
