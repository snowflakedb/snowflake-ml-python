import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
from unittest import mock

import cloudpickle as cp
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.jobs._interop import utils as u
from snowflake.ml.jobs._interop.dto_schema import (
    ExceptionMetadata,
    PayloadManifest,
    ResultMetadata,
)
from snowflake.ml.jobs._interop.exception_utils import RemoteError
from snowflake.ml.jobs._interop.results import ExecutionResult, LoadedExecutionResult
from snowflake.snowpark import exceptions as sp_exceptions


@dataclass(frozen=True)
class DataClass:
    int_value: int = 0
    str_value: str = "null"


class DummyClass:
    def __init__(self, value: int = 0, label: str = "null") -> None:
        self.value = value
        self.label = label

    def __eq__(self, value: object) -> bool:
        return isinstance(value, DummyClass) and self.value == value.value and self.label == value.label

    def __repr__(self) -> str:
        return f"DummyClass(value={self.value}, label={self.label})"


class DummyException(Exception):
    def __init__(self, message: str = "dummy error") -> None:
        super().__init__(message)
        self.message = message

    def __eq__(self, value: object) -> bool:
        return isinstance(value, DummyException) and self.message == value.message


class DummyNonserializableClass:
    def __init__(self) -> None:
        self._lock = threading.Lock()


class DummyNonserializableException(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()


def create_serialized_unavailable_class() -> bytes:
    """Creates a cloudpickle-serialized object from an unavailable module namespace."""
    code = """
import sys, cloudpickle, base64

# Create fake module and a class inside the fake module
module_name = "unavailable_test_module"
fake_mod = type(sys)(module_name)
sys.modules[module_name] = fake_mod

# Define class in fake module
class ExternalTestError(Exception):
    def __init__(self, message, value=42):
        super().__init__(message)
        self.value = value

ExternalTestError.__module__ = module_name
fake_mod.ExternalTestError = ExternalTestError

# Serialize and encode
serialized = cloudpickle.dumps(ExternalTestError("test exception", 123))
print(base64.b64encode(serialized).decode())
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    import base64

    return base64.b64decode(result.stdout.strip())


# Create a serialized unavailable class once and re-use it
external_exception_bytes = create_serialized_unavailable_class()


class TestInteropUtils(parameterized.TestCase):
    def setUp(self) -> None:
        self.mock_session = mock.MagicMock()

    @parameterized.named_parameters(  # type: ignore[misc]
        ("int_result", ExecutionResult(True, 1)),
        ("float_result", ExecutionResult(True, 3.14)),
        ("str_result", ExecutionResult(True, "test string")),
    )
    def test_save_result(self, result: ExecutionResult) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = f"{temp_dir}/result.json"
            u.save_result(result, temp_file_path)

            with open(temp_file_path) as f:
                result_json = json.load(f)
            self.assertEqual(result_json["success"], result.success)
            self.assertEqual(result_json["value"], result.value)
            self.assertIsNone(result_json["protocol"])
            self.assertIsNone(result_json["metadata"])

            # Assert that no other files exist in temp_dir
            files_in_temp_dir = os.listdir(temp_dir)
            self.assertEqual(len(files_in_temp_dir), 1)
            self.assertEqual(files_in_temp_dir[0], "result.json")

    @parameterized.named_parameters(  # type: ignore[misc]
        ("result_object", ExecutionResult(True, DummyClass()), "mljob_extra.pkl"),
        ("dataclass_object", ExecutionResult(True, DataClass()), "mljob_extra.pkl"),
        ("numpy_array", ExecutionResult(True, np.array([1, 2, 3])), "mljob_extra.npy"),
        (
            "pandas_dataframe",
            ExecutionResult(True, pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})),
            "mljob_extra_0.parquet",
            lambda x: x["paths"][0],
        ),
        ("exception_as_return_value", ExecutionResult(True, ValueError("error as result")), "mljob_extra.pkl"),
        ("value_error", ExecutionResult(False, ValueError("test error")), "mljob_extra.pkl"),
    )
    def test_save_result_complex(
        self,
        result: ExecutionResult,
        expected_path: str,
        path_getter: Callable[[PayloadManifest], str] = lambda x: x["path"],  # type: ignore[typeddict-item]
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = f"{temp_dir}/result.json"
            u.save_result(result, temp_file_path)

            with open(temp_file_path) as f:
                result_json = json.load(f)
            self.assertEqual(result_json["success"], result.success)
            self.assertIsNone(result_json["value"], result.value)

            actual_path = path_getter(result_json["protocol"]["manifest"])
            self.assertEndsWith(actual_path, expected_path)
            self.assertIsNotNone(result_json["protocol"])
            self.assertIsNotNone(result_json["metadata"])
            self.assertTrue(os.path.exists(actual_path))

    @parameterized.named_parameters(  # type: ignore[misc]
        ("result", ExecutionResult(True, DummyNonserializableClass())),
        ("exception", ExecutionResult(False, DummyNonserializableException())),
    )
    def test_save_result_nonserializable(self, result: ExecutionResult) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = f"{temp_dir}/result.json"
            u.save_result(result, temp_file_path)

            with open(temp_file_path) as f:
                result_json = json.load(f)
            self.assertEqual(result_json["success"], result.success)
            self.assertIsNone(result_json["value"], result.value)

            # In this case, protocol should not be set
            # but metadata should be set
            self.assertIsNone(result_json["protocol"])
            self.assertIsNotNone(result_json["metadata"])

            # Assert that no other files exist in temp_dir
            # FIXME: Protocols currently may write partial files on error
            # files_in_temp_dir = os.listdir(temp_dir)
            # self.assertEqual(len(files_in_temp_dir), 1)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("result_object", ExecutionResult(True, DummyClass()), 2),
        ("value_error", ExecutionResult(False, ValueError("test error")), 2),
        ("nonserializable_result", ExecutionResult(True, DummyNonserializableClass()), 1),
        ("nonserializable_exception", ExecutionResult(False, DummyNonserializableException()), 1),
    )
    def test_save_result_to_stage(
        self, result: ExecutionResult, expected_files: int, expected_path: str = "mljob_extra.pkl"
    ) -> None:
        temp_file_path = "@dummy_stage/result.json"
        u.save_result(result, temp_file_path, session=self.mock_session)

        self.assertEqual(self.mock_session.file.put_stream.call_count, expected_files)
        self.mock_session.file.put_stream.assert_any_call(mock.ANY, temp_file_path)
        if expected_files > 1:
            self.mock_session.file.put_stream.assert_any_call(mock.ANY, f"@dummy_stage/{expected_path}")

    @parameterized.named_parameters(  # type: ignore[misc]
        dict(
            testcase_name="no_result",
            data={"success": True, "value": None, "protocol": None},
            expected_value=None,
        ),
        dict(
            testcase_name="simple_result",
            data={"success": True, "value": "test string", "protocol": None},
            expected_value="test string",
        ),
        dict(
            testcase_name="cloudpickled_result",
            data={
                "success": True,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=cp.dumps(DummyClass(42, "loaded label")),
            expected_value=DummyClass(42, "loaded label"),
        ),
        dict(
            testcase_name="cloudpickled_result_dataclass",
            data={
                "success": True,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=cp.dumps(DataClass(int_value=42, str_value="loaded string")),
            expected_value=DataClass(int_value=42, str_value="loaded string"),
        ),
        # TODO: Add pandas DataFrame or numpy array test case
        dict(
            testcase_name="nonserializable_result",
            data={
                "success": True,
                "value": None,
                "protocol": None,
                "metadata": {"type": "__main__.DummyNonserializableClass", "repr": "..."},
                "serialize_error": "TypeError(\"cannot pickle '_thread.lock' object\")",
            },
            expected_error=ValueError("Job execution succeeded but result retrieval failed"),
            expected_cause=TypeError("Original result serialization failed"),
        ),
        dict(
            testcase_name="nonserializable_result_wrapped_exception",
            data={
                "success": True,
                "value": None,
                "protocol": None,
                "metadata": {"type": "__main__.DummyNonserializableClass", "repr": "..."},
                "serialize_error": "TypeError(\"cannot pickle '_thread.lock' object\")",
            },
            wrap_exceptions=True,  # Shouldn't make a difference for success case
            expected_error=ValueError("Job execution succeeded but result retrieval failed"),
            expected_cause=TypeError("Original result serialization failed"),
        ),
        dict(
            testcase_name="exception_as_result",
            data={
                "success": True,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=cp.dumps(DummyException("loaded dummy error")),
            expected_value=DummyException("loaded dummy error"),
        ),
        dict(
            testcase_name="simple_exception",
            data={
                "success": False,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=cp.dumps(RuntimeError("loaded runtime error")),
            expected_error=RuntimeError("loaded runtime error"),
            wrap_exceptions=False,
        ),
        dict(
            testcase_name="simple_exception_wrapped",
            data={
                "success": False,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=cp.dumps(RuntimeError("loaded runtime error")),
            expected_error=RuntimeError("Job execution failed"),
            expected_cause=RuntimeError("loaded runtime error"),
            wrap_exceptions=True,
        ),
        dict(
            testcase_name="nonserializable_exception",
            data={
                "success": False,
                "value": None,
                "protocol": None,
                "metadata": {
                    "type": "SomeNonserializableClass",
                    "repr": "SomeNonserializableClass('...')",
                    "message": "...",
                    "traceback": "...",
                },
                "serialize_error": "TypeError(\"cannot pickle '_thread.lock' object\")",
            },
            expected_error=RemoteError("SomeNonserializableClass('...')"),
            expected_cause=ModuleNotFoundError("Unrecognized exception"),
        ),
        dict(
            testcase_name="nonserializable_exception_reconstructed",
            data={
                "success": False,
                "value": None,
                "protocol": None,
                "metadata": {
                    "type": "NotImplementedError",
                    "repr": "NotImplementedError('test')",
                    "message": "test",
                    "traceback": "...",
                },
                "serialize_error": "TypeError(\"cannot pickle '_thread.lock' object\")",
            },
            expected_error=NotImplementedError("test"),
        ),
        dict(
            testcase_name="nonserializable_exception_no_metadata",
            data={
                "success": False,
                "value": None,
                "protocol": None,
                "metadata": None,
                "serialize_error": "TypeError(\"cannot pickle '_thread.lock' object\")",
            },
            expected_error=RuntimeError("Unknown remote error"),
            expected_cause=TypeError("Original result serialization failed"),
        ),
        dict(
            testcase_name="unknown_exception",
            data={
                "success": False,
                "value": None,
            },
            expected_error=RuntimeError("Unknown remote error"),
        ),
        # Negative deserialization cases
        dict(
            testcase_name="remote_only_result",
            data={
                "success": True,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=external_exception_bytes,
            expected_error=ValueError("Job execution succeeded but result retrieval failed"),
            expected_cause=ModuleNotFoundError("No module"),
        ),
        dict(
            testcase_name="remote_only_exception",
            data={
                "success": False,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
                "metadata": {
                    "type": "ExternalTestError",
                    "repr": "ExternalTestError('test exception', 123)",
                    "message": "test exception",
                    "traceback": "...",
                },
            },
            secondary_data=external_exception_bytes,
            expected_error=RemoteError("ExternalTestError('test exception', 123)"),
            expected_cause=ModuleNotFoundError("Unrecognized exception"),
        ),
        dict(
            testcase_name="remote_only_exception_no_metadata",
            data={
                "success": False,
                "value": None,
                "protocol": {
                    "name": "cloudpickle",
                    "version": cp.__version__,
                    "manifest": {"path": "@dummy/secondary"},
                },
            },
            secondary_data=external_exception_bytes,
            expected_error=RuntimeError("Unknown remote error"),
            expected_cause=ModuleNotFoundError("No module"),
        ),
        dict(
            testcase_name="incompatible_cloudpickle",
            data={
                "success": True,
                "value": None,
                "protocol": {"name": "cloudpickle", "version": "0.0.0", "manifest": {"path": "@dummy/secondary"}},
            },
            secondary_data=b"invalid_pickle_data",
            expected_error=ValueError("Job execution succeeded but result retrieval failed"),
            expected_cause=ValueError("cloudpickle version"),
        ),
        dict(
            testcase_name="incompatible_result_protocol",
            data={
                "success": True,
                "value": None,
                "protocol": {"name": "unknown_proto", "manifest": {"path": "@dummy/secondary"}},
            },
            secondary_data=b"fake_data",
            expected_error=ValueError("Job execution succeeded but result retrieval failed"),
            expected_cause=TypeError("No protocol matching"),
        ),
        dict(
            testcase_name="incompatible_exception_protocol",
            data={
                "success": False,
                "value": None,
                "protocol": {"name": "unknown_proto", "manifest": {"path": "@dummy/secondary"}},
            },
            secondary_data=b"fake_data",
            expected_error=RuntimeError("Unknown remote error"),
            expected_cause=TypeError("No protocol matching"),
        ),
        dict(
            testcase_name="incompatible_exception_protocol_reconstructed",
            data={
                "success": False,
                "value": None,
                "protocol": {"name": "unknown_proto", "manifest": {"path": "@dummy/secondary"}},
                "metadata": {
                    "type": "NotImplementedError",
                    "repr": "NotImplementedError('test')",
                    "message": "test",
                    "traceback": "...",
                },
            },
            secondary_data=b"fake_data",
            expected_error=NotImplementedError("test"),
        ),
        dict(
            testcase_name="legacy_result_simple",
            data={"success": True, "result_type": int.__qualname__, "result": 42},
            secondary_data=cp.dumps({"success": True, "result_type": int.__qualname__, "result": 42}),
            expected_value=42,
        ),
        dict(
            testcase_name="legacy_result_complex",
            data={"success": True, "result_type": int.__qualname__, "result": str(DummyClass(42, "loaded label"))},
            secondary_data=cp.dumps(
                {"success": True, "result_type": int.__qualname__, "result": DummyClass(42, "loaded label")}
            ),
            expected_value=DummyClass(42, "loaded label"),
        ),
        dict(
            testcase_name="legacy_result_nonserializable",
            data={
                "success": True,
                "result_type": DummyNonserializableClass.__qualname__,
                "result": str(DummyNonserializableClass()),
            },
            secondary_data=None,
            expected_value=str(DummyNonserializableClass()),
        ),
        dict(
            testcase_name="legacy_exception_simple",
            data={"success": False, "exc_type": "builtins.ValueError", "exc_value": "legacy error", "exc_tb": "..."},
            secondary_data=cp.dumps(
                {"success": False, "exc_type": "builtins.ValueError", "exc_value": "legacy error", "exc_tb": "..."}
            ),
            expected_error=ValueError("legacy error"),
        ),
        dict(
            testcase_name="legacy_exception_nonserializable",
            data={
                "success": False,
                "exc_type": "SomeNonserializableClass",
                "exc_value": "legacy error",
                "exc_tb": "...",
            },
            secondary_data=None,
            expected_error=RemoteError("SomeNonserializableClass('legacy error')"),
            expected_cause=ModuleNotFoundError("Unrecognized exception"),
        ),
        dict(
            testcase_name="legacy_exception_nonserializable_reconstructed",
            data={
                "success": False,
                "exc_type": "__main__.DummyNonserializableException",
                "exc_value": "legacy error",
                "exc_tb": "...",
            },
            secondary_data=None,
            expected_error=DummyNonserializableException("legacy error"),
        ),
    )
    def test_load_result(
        self,
        data: dict[str, Any],
        secondary_data: Optional[bytes] = None,
        expected_value: Any = None,
        expected_error: Optional[Exception] = None,
        expected_cause: Optional[Exception] = None,
        expected_context: Optional[Exception] = None,
        wrap_exceptions: bool = False,
    ) -> None:
        result_path = "@dummy_stage/result.json"
        data_str = json.dumps(data)  # NOTE: Need to do this outside mock_get_stream to make closure work correctly

        def mock_get_stream(path: str, *args: Any, **kwargs: Any) -> io.BytesIO:
            # Hacky behavior: If the input path is result_path, return the encoded JSON data
            # Else, return secondary_data
            # Note that path must be a stage path to trigger this mock, else it'll try to read from disk
            if path == result_path:
                return io.BytesIO(data_str.encode("utf-8"))
            if secondary_data is None:
                raise sp_exceptions.SnowparkSQLException(f"No secondary data, path: {path}")
            return io.BytesIO(secondary_data)

        self.mock_session.file.get_stream.side_effect = mock_get_stream

        result = u.load_result(result_path, session=self.mock_session)
        self.assertIsInstance(result, ExecutionResult)

        if expected_error is not None:
            with self.assertRaisesRegex(type(expected_error), re.escape(str(expected_error))) as cm:
                _ = result.get_value(wrap_exceptions=wrap_exceptions)
            if expected_cause:
                actual_cause = cm.exception.__cause__
                self.assertIsInstance(actual_cause, type(expected_cause))
                self.assertIn(str(expected_cause), str(actual_cause))
            else:
                self.assertIsNone(cm.exception.__cause__)
            if expected_context:
                actual_context = cm.exception.__context__
                self.assertIsInstance(actual_context, type(expected_context))
                self.assertIn(str(expected_context), str(actual_context))
            else:
                self.assertIsNone(cm.exception.__context__)
        else:
            value = result.get_value(wrap_exceptions=wrap_exceptions)
            load_error = result.load_error if isinstance(result, LoadedExecutionResult) else None
            self.assertEqual(value, expected_value, load_error)
            self.assertEqual(type(value), type(expected_value), load_error)

    @parameterized.named_parameters(  # type: ignore[misc]
        dict(
            testcase_name="not_json",
            data="this is not json",
            expected_error=json.JSONDecodeError("Expecting value", "this is not json", 0),
        ),
        dict(
            testcase_name="malformed_json",
            data="{'success': True, 'value': 1",  # Missing closing brace
            expected_error=json.JSONDecodeError(
                "Expecting property name enclosed in double quotes", "{'success': True, 'value': 1", 1
            ),
        ),
        dict(
            testcase_name="empty_dict",
            data={},
            expected_error=ValueError("Invalid result"),
        ),
        dict(
            testcase_name="missing_multiple_fields",
            data={"value": "test"},
            expected_error=ValueError("Invalid result schema"),
        ),
        dict(
            testcase_name="missing_success_field",
            data={"value": "test", "protocol": None},
            expected_error=ValueError("Invalid result schema"),
        ),
    )
    def test_load_result_negative(
        self,
        data: Union[dict[str, Any], str],
        expected_error: Exception,
        secondary_data: Optional[bytes] = None,
    ) -> None:
        result_path = "@dummy_stage/result.json"
        if isinstance(data, dict):
            data = json.dumps(data)  # NOTE: Need to do this outside mock_get_stream to make closure work correctly

        def mock_get_stream(path: str, *args: Any, **kwargs: Any) -> io.BytesIO:
            # Hacky behavior: If the input path is result_path, return the encoded JSON data
            # Else, return secondary_data
            # Note that path must be a stage path to trigger this mock, else it'll try to read from disk
            if path == result_path:
                return io.BytesIO(data.encode("utf-8"))
            if secondary_data is None:
                raise sp_exceptions.SnowparkSQLException(f"No secondary data, path: {path}")
            return io.BytesIO(secondary_data)

        self.mock_session.file.get_stream.side_effect = mock_get_stream

        with self.assertRaisesRegex(type(expected_error), re.escape(str(expected_error))):
            _ = u.load_result(result_path, session=self.mock_session)

    @parameterized.parameters(  # type: ignore[misc]
        (None, "builtins.NoneType"),
        (1, "builtins.int"),
        (1.0, "builtins.float"),
        (False, "builtins.bool"),
        ((1, 2, 3), "builtins.tuple"),
        (DummyClass(), "__main__.DummyClass"),
        (ValueError("test error"), "builtins.ValueError"),
        (RuntimeError("test error"), "builtins.RuntimeError"),
        (sp_exceptions.SnowparkClientException("test error"), "snowflake.snowpark.exceptions.SnowparkClientException"),
    )
    def test_get_metadata(self, obj: Any, expected_type: str) -> None:
        m = u._get_metadata(obj)
        assert isinstance(m, ResultMetadata)
        self.assertEqual(m.type, expected_type)
        self.assertEqual(m.repr, repr(obj))

    @parameterized.parameters(  # type: ignore[misc]
        (ValueError("test error"), "builtins.ValueError"),
        (RuntimeError("test error"), "builtins.RuntimeError"),
        (sp_exceptions.SnowparkClientException("test error"), "snowflake.snowpark.exceptions.SnowparkClientException"),
    )
    def test_get_metadata_exception(self, exception: Exception, expected_type: str) -> None:
        # Raise the exception to generate a traceback
        try:
            raise exception
        except Exception as e:
            err = e

        m = u._get_metadata(err)
        assert isinstance(m, ExceptionMetadata)
        self.assertEqual(m.type, expected_type)
        self.assertEqual(m.repr, repr(err))
        self.assertEqual(m.message, str(err))
        self.assertNotEmpty(m.traceback)


if __name__ == "__main__":
    absltest.main()
