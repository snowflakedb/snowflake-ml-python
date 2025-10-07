"""Legacy result serialization protocol support for ML Jobs.

This module provides backward compatibility with the result serialization protocol used by
mljob_launcher.py prior to snowflake-ml-python>=1.17.0

LEGACY PROTOCOL (v1):
---------------------
The old serialization protocol (save_mljob_result_v1 in mljob_launcher.py) worked as follows:

1. Results were stored in an ExecutionResult dataclass with two optional fields:
   - result: Any = None          # For successful executions
   - exception: BaseException = None  # For failed executions

2. The ExecutionResult was converted to a dictionary via to_dict():
   Success case:
     {"success": True, "result_type": <type qualname>, "result": <value>}

   Failure case:
     {"success": False, "exc_type": "<module>.<class>", "exc_value": <exception>,
      "exc_tb": <formatted traceback string>}

3. The dictionary was serialized TWICE for fault tolerance:
   - Primary: cloudpickle to .pkl file under output/mljob_result.pkl (supports complex Python objects)
   - Fallback: JSON to .json file under output/mljob_result.json (for cross-version compatibility)

WHY THIS MODULE EXISTS:
-----------------------
Jobs submitted with client versions using the v1 protocol will write v1-format result files.
This module ensures that newer clients can still retrieve results from:
- Jobs submitted before the protocol change
- Jobs running in environments where snowflake.ml.jobs._interop is not available
  (triggering the ImportError fallback to v1 in save_mljob_result)

RETRIEVAL FLOW:
---------------
fetch_result() implements the v1 retrieval logic:
1. Try to unpickle from .pkl file
2. On failure (version mismatch, missing imports, etc.), fall back to .json file
3. Convert the legacy dict format to ExecutionResult
4. Provide helpful error messages for common failure modes

REMOVAL IMPLICATIONS:
---------------------
Removing this module would break result retrieval for:
- Any jobs that were submitted with snowflake-ml-python<1.17.0 and are still running/completed
- Any jobs running in old runtime environments that fall back to v1 serialization

Safe to remove when:
- All ML Runtime images have been updated to include the new _interop modules
- Sufficient time has passed that no jobs using the old protocol are still retrievable
  (consider retention policies for job history/logs)
"""

import json
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Union

from snowflake import snowpark
from snowflake.ml.jobs._interop import exception_utils, results
from snowflake.snowpark import exceptions as sp_exceptions


@dataclass(frozen=True)
class ExecutionResult:
    result: Any = None
    exception: Optional[BaseException] = None

    @property
    def success(self) -> bool:
        return self.exception is None

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable dictionary."""
        if isinstance(self.exception, BaseException):
            exc_type = type(self.exception)
            return {
                "success": False,
                "exc_type": f"{exc_type.__module__}.{exc_type.__name__}",
                "exc_value": self.exception,
                "exc_tb": "".join(traceback.format_tb(self.exception.__traceback__)),
            }
        return {
            "success": True,
            "result_type": type(self.result).__qualname__,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, result_dict: dict[str, Any]) -> "ExecutionResult":
        if not isinstance(result_dict.get("success"), bool):
            raise ValueError("Invalid result dictionary")

        if result_dict["success"]:
            # Load successful result
            return cls(result=result_dict.get("result"))

        # Load exception
        exc_type = result_dict.get("exc_type", "RuntimeError")
        exc_value = result_dict.get("exc_value", "Unknown error")
        exc_tb = result_dict.get("exc_tb", "")
        return cls(exception=load_exception(exc_type, exc_value, exc_tb))


def fetch_result(
    session: snowpark.Session, result_path: str, result_json: Optional[dict[str, Any]] = None
) -> ExecutionResult:
    """
    Fetch the serialized result from the specified path.

    Args:
        session: Snowpark Session to use for file operations.
        result_path: The path to the serialized result file.
        result_json: Optional pre-loaded JSON result dictionary to use instead of fetching from file.

    Returns:
        A dictionary containing the execution result if available, None otherwise.

    Raises:
        RuntimeError: If both pickle and JSON result retrieval fail.
    """
    try:
        with session.file.get_stream(result_path) as result_stream:
            return ExecutionResult.from_dict(pickle.load(result_stream))
    except (
        sp_exceptions.SnowparkSQLException,
        pickle.UnpicklingError,
        TypeError,
        ImportError,
        AttributeError,
        MemoryError,
    ) as pickle_error:
        # Fall back to JSON result if loading pickled result fails for any reason
        try:
            if result_json is None:
                result_json_path = os.path.splitext(result_path)[0] + ".json"
                with session.file.get_stream(result_json_path) as result_stream:
                    result_json = json.load(result_stream)
            return ExecutionResult.from_dict(result_json)
        except Exception as json_error:
            # Both pickle and JSON failed - provide helpful error message
            raise RuntimeError(_fetch_result_error_message(pickle_error, result_path, json_error)) from pickle_error


def _fetch_result_error_message(error: Exception, result_path: str, json_error: Optional[Exception] = None) -> str:
    """Create helpful error messages for common result retrieval failures."""

    # Package import issues
    if isinstance(error, ImportError):
        return f"Failed to retrieve job result: Package not installed in your local environment. Error: {str(error)}"

    # Package versions differ between runtime and local environment
    if isinstance(error, AttributeError):
        return f"Failed to retrieve job result: Package version mismatch. Error: {str(error)}"

    # Serialization issues
    if isinstance(error, TypeError):
        return f"Failed to retrieve job result: Non-serializable objects were returned. Error: {str(error)}"

    # Python version pickling incompatibility
    if isinstance(error, pickle.UnpicklingError) and "protocol" in str(error).lower():
        client_version = f"Python {sys.version_info.major}.{sys.version_info.minor}"
        runtime_version = "Python 3.10"  # NOTE: This may be inaccurate, but this path isn't maintained anymore
        return (
            f"Failed to retrieve job result: Python version mismatch - job ran on {runtime_version}, "
            f"local environment using Python {client_version}. Error: {str(error)}"
        )

    # File access issues
    if isinstance(error, sp_exceptions.SnowparkSQLException):
        if "not found" in str(error).lower() or "does not exist" in str(error).lower():
            return (
                f"Failed to retrieve job result: No result file found. Check job.get_logs() for execution "
                f"errors. Error: {str(error)}"
            )
        else:
            return f"Failed to retrieve job result: Cannot access result file. Error: {str(error)}"

    if isinstance(error, MemoryError):
        return f"Failed to retrieve job result: Result too large for memory. Error: {str(error)}"

    # Generic fallback
    base_message = f"Failed to retrieve job result: {str(error)}"
    if json_error:
        base_message += f" (JSON fallback also failed: {str(json_error)})"
    return base_message


def load_exception(exc_type_name: str, exc_value: Union[Exception, str], exc_tb: str) -> BaseException:
    """
    Create an exception with a string-formatted traceback.

    When this exception is raised and not caught, it will display the original traceback.
    When caught, it behaves like a regular exception without showing the traceback.

    Args:
        exc_type_name: Name of the exception type (e.g., 'ValueError', 'RuntimeError')
        exc_value: The deserialized exception value or exception string (i.e. message)
        exc_tb: String representation of the traceback

    Returns:
        An exception object with the original traceback information

    # noqa: DAR401
    """
    if isinstance(exc_value, Exception):
        exception = exc_value
        return exception_utils.attach_remote_error_info(exception, exc_type_name, str(exc_value), exc_tb)
    return exception_utils.build_exception(exc_type_name, str(exc_value), exc_tb)


def load_legacy_result(
    session: snowpark.Session, result_path: str, result_json: Optional[dict[str, Any]] = None
) -> results.ExecutionResult:
    # Load result using legacy interop
    legacy_result = fetch_result(session, result_path, result_json=result_json)

    # Adapt legacy result to new result
    return results.ExecutionResult(
        success=legacy_result.success,
        value=legacy_result.exception or legacy_result.result,
    )
