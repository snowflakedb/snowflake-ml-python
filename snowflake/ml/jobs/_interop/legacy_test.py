import json
import os
import posixpath
from typing import Any
from unittest import mock

from absl.testing import absltest

from snowflake.connector import errorcode, errors as connector_errors
from snowflake.ml.jobs._interop import exception_utils, legacy
from snowflake.snowpark import exceptions, file_operation


class TestLegacy(absltest.TestCase):
    def test_load_exception_with_builtin_exception(self) -> None:
        """Test loading a built-in exception type."""
        exc = legacy.load_exception("ValueError", "test error message", "traceback info")
        self.assertIsInstance(exc, ValueError)
        self.assertEqual(str(exc), "test error message")

        remote_error = exception_utils.retrieve_remote_error_info(exc)
        assert remote_error is not None
        self.assertEqual(remote_error.exc_type, "ValueError")
        self.assertEqual(remote_error.exc_msg, "test error message")
        self.assertEqual(remote_error.exc_tb, "traceback info")

    def test_load_exception_with_custom_exception_name(self) -> None:
        """Test loading an exception with a custom exception name."""
        # Create a non-existent exception type name
        exc = legacy.load_exception("NonExistentError", "custom error", "traceback info")
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("NonExistentError", str(exc))

        remote_error = exception_utils.retrieve_remote_error_info(exc)
        assert remote_error is not None
        self.assertEqual(remote_error.exc_type, "NonExistentError")
        self.assertEqual(remote_error.exc_msg, "custom error")
        self.assertEqual(remote_error.exc_tb, "traceback info")

    def test_load_exception_with_qualified_name(self) -> None:
        """Test loading an exception with a qualified name."""
        # Use a common exception from a module
        exc_type = exceptions.SnowparkClientException
        exc = legacy.load_exception(f"{exc_type.__module__}.{exc_type.__name__}", "mock error", "traceback info")
        self.assertIsInstance(exc, exceptions.SnowparkClientException)
        self.assertEqual(str(exc), "mock error")

        remote_error = exception_utils.retrieve_remote_error_info(exc)
        assert remote_error is not None
        self.assertEqual(remote_error.exc_type, "SnowparkClientException")
        self.assertEqual(remote_error.exc_msg, "mock error")
        self.assertEqual(remote_error.exc_tb, "traceback info")

    def test_load_exception_with_exception_instance(self) -> None:
        """Test loading with an existing exception instance."""
        original_exc = ValueError("original error")
        exc = legacy.load_exception("ValueError", original_exc, "traceback info")
        self.assertIs(exc, original_exc)  # Should be the same object

        remote_error = exception_utils.retrieve_remote_error_info(exc)
        assert remote_error is not None
        self.assertEqual(remote_error.exc_type, "ValueError")
        self.assertEqual(remote_error.exc_msg, "original error")
        self.assertEqual(remote_error.exc_tb, "traceback info")

    def test_fetch_result_falls_back_to_json_when_pickle_is_missing(self) -> None:
        def fake_get(path: str, target_directory: str, **kwargs: Any) -> list[Any]:
            if path.endswith(".pkl"):
                raise connector_errors.OperationalError(
                    msg="the file does not exist.", errno=errorcode.ER_FILE_NOT_EXISTS
                )
            with open(os.path.join(target_directory, posixpath.basename(path)), "w") as f:
                json.dump({"success": True, "result_type": "int", "result": 7}, f)
            return []

        session = mock.MagicMock()
        session.file = mock.create_autospec(file_operation.FileOperation, instance=True)
        session.file.get.side_effect = fake_get

        result = legacy.fetch_result(session, "@stage/output/mljob_result.pkl")

        self.assertTrue(result.success)
        self.assertEqual(result.result, 7)


if __name__ == "__main__":
    absltest.main()
