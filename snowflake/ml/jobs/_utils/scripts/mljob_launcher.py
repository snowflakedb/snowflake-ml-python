import argparse
import copy
import importlib.util
import json
import os
import runpy
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

import cloudpickle

from snowflake.ml.jobs._utils import constants
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

# Fallbacks in case of SnowML version mismatch
RESULT_PATH_ENV_VAR = getattr(constants, "RESULT_PATH_ENV_VAR", "MLRS_RESULT_PATH")

JOB_RESULT_PATH = os.environ.get(RESULT_PATH_ENV_VAR, "mljob_result.pkl")


try:
    from snowflake.ml.jobs._utils.interop_utils import ExecutionResult
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ExecutionResult:  # type: ignore[no-redef]
        result: Optional[Any] = None
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


# Create a custom JSON encoder that converts non-serializable types to strings
class SimpleJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def run_script(script_path: str, *script_args: Any, main_func: Optional[str] = None) -> Any:
    """
    Execute a Python script and return its result.

    Args:
        script_path: Path to the Python script
        script_args: Arguments to pass to the script
        main_func: The name of the function to call in the script (if any)

    Returns:
        Result from script execution, either from the main function or the script's __return__ value

    Raises:
        RuntimeError: If the specified main_func is not found or not callable
    """
    # Save original sys.argv and modify it for the script (applies to runpy execution only)
    original_argv = sys.argv
    sys.argv = [script_path, *script_args]

    # Create a Snowpark session before running the script
    # Session can be retrieved from using snowflake.snowpark.context.get_active_session()
    session = Session.builder.configs(SnowflakeLoginOptions()).create()  # noqa: F841

    try:
        if main_func:
            # Use importlib for scripts with a main function defined
            module_name = Path(script_path).stem
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Validate main function
            if not (func := getattr(module, main_func, None)) or not callable(func):
                raise RuntimeError(f"Function '{main_func}' not a valid entrypoint for {script_path}")

            # Call main function
            result = func(*script_args)
            return result
        else:
            # Use runpy for other scripts
            globals_dict = runpy.run_path(script_path, run_name="__main__")
            result = globals_dict.get("__return__", None)
            return result
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def main(script_path: str, *script_args: Any, script_main_func: Optional[str] = None) -> ExecutionResult:
    """Executes a Python script and serializes the result to JOB_RESULT_PATH.

    Args:
        script_path (str): Path to the Python script to execute.
        script_args (Any): Arguments to pass to the script.
        script_main_func (str, optional): The name of the function to call in the script (if any).

    Returns:
        ExecutionResult: Object containing execution results.

    Raises:
        Exception: Re-raises any exception caught during script execution.
    """
    # Run the script with the specified arguments
    try:
        result = run_script(script_path, *script_args, main_func=script_main_func)
        result_obj = ExecutionResult(result=result)
        return result_obj
    except Exception as e:
        tb = e.__traceback__
        skip_files = {__file__, runpy.__file__}
        while tb and tb.tb_frame.f_code.co_filename in skip_files:
            # Skip any frames preceding user script execution
            tb = tb.tb_next
        cleaned_ex = copy.copy(e)  # Need to create a mutable copy of exception to set __traceback__
        cleaned_ex = cleaned_ex.with_traceback(tb)
        result_obj = ExecutionResult(exception=cleaned_ex)
        raise
    finally:
        result_dict = result_obj.to_dict()
        try:
            # Serialize result using cloudpickle
            result_pickle_path = JOB_RESULT_PATH
            with open(result_pickle_path, "wb") as f:
                cloudpickle.dump(result_dict, f)  # Pickle dictionary form for compatibility
        except Exception as pkl_exc:
            warnings.warn(f"Failed to pickle result to {result_pickle_path}: {pkl_exc}", RuntimeWarning, stacklevel=1)

        try:
            # Serialize result to JSON as fallback path in case of cross version incompatibility
            # TODO: Manually convert non-serializable types to strings
            result_json_path = os.path.splitext(JOB_RESULT_PATH)[0] + ".json"
            with open(result_json_path, "w") as f:
                json.dump(result_dict, f, indent=2, cls=SimpleJSONEncoder)
        except Exception as json_exc:
            warnings.warn(
                f"Failed to serialize JSON result to {result_json_path}: {json_exc}", RuntimeWarning, stacklevel=1
            )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch a Python script and save the result")
    parser.add_argument("script_path", help="Path to the Python script to execute")
    parser.add_argument("script_args", nargs="*", help="Arguments to pass to the script")
    parser.add_argument(
        "--script_main_func", required=False, help="The name of the main function to call in the script"
    )
    args, unknown_args = parser.parse_known_args()

    main(
        args.script_path,
        *args.script_args,
        *unknown_args,
        script_main_func=args.script_main_func,
    )
