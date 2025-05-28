import argparse
import copy
import importlib.util
import json
import logging
import os
import runpy
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

import cloudpickle
from constants import LOG_END_MSG, LOG_START_MSG

from snowflake.ml.jobs._utils import constants
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fallbacks in case of SnowML version mismatch
RESULT_PATH_ENV_VAR = getattr(constants, "RESULT_PATH_ENV_VAR", "MLRS_RESULT_PATH")
JOB_RESULT_PATH = os.environ.get(RESULT_PATH_ENV_VAR, "mljob_result.pkl")

# Constants for the wait_for_min_instances function
CHECK_INTERVAL = 10  # seconds
TIMEOUT = 720  # seconds


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
            return f"Unserializable object: {repr(obj)}"


def get_active_node_count() -> int:
    """
    Count the number of active nodes in the Ray cluster.

    Returns:
        int: Total count of active nodes
    """
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        total_active = len(nodes)

        logger.info(f"Active nodes: {total_active}")
        return total_active
    except Exception as e:
        logger.warning(f"Error getting active node count: {e}")
        return 0


def wait_for_min_instances(min_instances: int) -> None:
    """
    Wait until the specified minimum number of instances are available in the Ray cluster.

    Args:
        min_instances: Minimum number of instances required

    Raises:
        TimeoutError: If failed to connect to Ray or if minimum instances are not available within timeout
    """
    if min_instances <= 1:
        logger.debug("Minimum instances is 1 or less, no need to wait for additional instances")
        return

    start_time = time.time()
    timeout = os.getenv("JOB_MIN_INSTANCES_TIMEOUT", TIMEOUT)
    check_interval = os.getenv("JOB_MIN_INSTANCES_CHECK_INTERVAL", CHECK_INTERVAL)
    logger.debug(f"Waiting for at least {min_instances} instances to be ready (timeout: {timeout}s)")

    while time.time() - start_time < timeout:
        total_nodes = get_active_node_count()

        if total_nodes >= min_instances:
            elapsed = time.time() - start_time
            logger.info(f"Minimum instance requirement met: {total_nodes} instances available after {elapsed:.1f}s")
            return

        logger.debug(
            f"Waiting for instances: {total_nodes}/{min_instances} available "
            f"(elapsed: {time.time() - start_time:.1f}s)"
        )
        time.sleep(check_interval)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for {min_instances} instances, only {get_active_node_count()} available"
    )


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
    try:
        # Wait for minimum required instances if specified
        min_instances_str = os.environ.get("JOB_MIN_INSTANCES", 1)
        if min_instances_str and int(min_instances_str) > 1:
            wait_for_min_instances(int(min_instances_str))

        # Log start marker for user script execution
        print(LOG_START_MSG)  # noqa: T201

        # Run the script with the specified arguments
        result = run_script(script_path, *script_args, main_func=script_main_func)

        # Log end marker for user script execution
        print(LOG_END_MSG)  # noqa: T201

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
