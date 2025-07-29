import argparse
import copy
import importlib.util
import json
import logging
import math
import os
import runpy
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

import cloudpickle

from snowflake.ml.jobs._utils import constants
from snowflake.snowpark import Session

try:
    from snowflake.ml._internal.utils.connection_params import SnowflakeLoginOptions
except ImportError:
    from snowflake.ml.utils.connection_params import SnowflakeLoginOptions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# The followings are Inherited from snowflake.ml.jobs._utils.constants
# We need to copy them here since snowml package on the server side does
# not have the latest version of the code
# Log start and end messages
LOG_START_MSG = getattr(
    constants,
    "LOG_START_MSG",
    "--------------------------------\nML job started\n--------------------------------",
)
LOG_END_MSG = getattr(
    constants,
    "LOG_END_MSG",
    "--------------------------------\nML job finished\n--------------------------------",
)

# min_instances environment variable name
MIN_INSTANCES_ENV_VAR = getattr(constants, "MIN_INSTANCES_ENV_VAR", "MLRS_MIN_INSTANCES")
TARGET_INSTANCES_ENV_VAR = getattr(constants, "TARGET_INSTANCES_ENV_VAR", "SNOWFLAKE_JOBS_COUNT")

# Fallbacks in case of SnowML version mismatch
RESULT_PATH_ENV_VAR = getattr(constants, "RESULT_PATH_ENV_VAR", "MLRS_RESULT_PATH")
JOB_RESULT_PATH = os.environ.get(RESULT_PATH_ENV_VAR, "/mnt/job_stage/output/mljob_result.pkl")
PAYLOAD_DIR_ENV_VAR = getattr(constants, "PAYLOAD_DIR_ENV_VAR", "MLRS_PAYLOAD_DIR")

# Constants for the wait_for_instances function
MIN_WAIT_TIME = float(os.getenv("MLRS_INSTANCES_MIN_WAIT") or -1)  # seconds
TIMEOUT = float(os.getenv("MLRS_INSTANCES_TIMEOUT") or 720)  # seconds
CHECK_INTERVAL = float(os.getenv("MLRS_INSTANCES_CHECK_INTERVAL") or 10)  # seconds


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


def wait_for_instances(
    min_instances: int,
    target_instances: int,
    *,
    min_wait_time: float = -1,  # seconds
    timeout: float = 720,  # seconds
    check_interval: float = 10,  # seconds
) -> None:
    """
    Wait until the specified minimum number of instances are available in the Ray cluster.

    Args:
        min_instances: Minimum number of instances required
        target_instances: Target number of instances to wait for
        min_wait_time: Minimum time to wait for target_instances to be available.
            If less than 0, automatically set based on target_instances.
        timeout: Maximum time to wait for min_instances to be available before raising a TimeoutError.
        check_interval: Maximum time to wait between checks (uses exponential backoff).

    Examples:
        Scenario 1 - Ideal case (target met quickly):
            wait_for_instances(min_instances=2, target_instances=4, min_wait_time=5, timeout=60)
            If 4 instances are available after 1 second, the function returns without further waiting (target met).

        Scenario 2 - Min instances met, target not reached:
            wait_for_instances(min_instances=2, target_instances=4, min_wait_time=10, timeout=60)
            If only 3 instances are available after 10 seconds, the function returns (min requirement satisfied).

        Scenario 3 - Min instances met early, but min_wait_time not elapsed:
            wait_for_instances(min_instances=2, target_instances=4, min_wait_time=30, timeout=60)
            If 2 instances are available after 5 seconds, function continues waiting for target_instances
            until either 4 instances are found or 30 seconds have elapsed.

        Scenario 4 - Timeout scenario:
            wait_for_instances(min_instances=3, target_instances=5, min_wait_time=10, timeout=30)
            If only 2 instances are available after 30 seconds, TimeoutError is raised.

        Scenario 5 - Single instance job (early return):
            wait_for_instances(min_instances=1, target_instances=1, min_wait_time=5, timeout=60)
            The function returns without waiting because target_instances <= 1.

    Raises:
        ValueError: If arguments are invalid
        TimeoutError: If failed to connect to Ray or if minimum instances are not available within timeout
    """
    if min_instances > target_instances:
        raise ValueError(
            f"Minimum instances ({min_instances}) cannot be greater than target instances ({target_instances})"
        )
    if timeout < 0:
        raise ValueError("Timeout must be greater than 0")
    if check_interval < 0:
        raise ValueError("Check interval must be greater than 0")

    if target_instances <= 1:
        logger.debug("Target instances is 1 or less, no need to wait for additional instances")
        return

    if min_wait_time < 0:
        # Automatically set min_wait_time based on the number of target instances
        # Using min_wait_time = 3 * log2(target_instances) as a starting point:
        #   target_instances = 1    => min_wait_time = 0
        #   target_instances = 2    => min_wait_time = 3
        #   target_instances = 4    => min_wait_time = 6
        #   target_instances = 8    => min_wait_time = 9
        #   target_instances = 32   => min_wait_time = 15
        #   target_instances = 50   => min_wait_time = 16.9
        #   target_instances = 100  => min_wait_time = 19.9
        min_wait_time = min(3 * math.log2(target_instances), timeout / 10)  # Clamp to timeout / 10

    # mljob_launcher runs inside the CR where mlruntime libraries are available, so we can import common_util directly
    from common_utils import common_util as mlrs_util

    start_time = time.time()
    current_interval = max(min(1, check_interval), 0.1)  # Default 1s, minimum 0.1s
    logger.debug(
        "Waiting for instances to be ready "
        "(min_instances={}, target_instances={}, timeout={}s, max_check_interval={}s)".format(
            min_instances, target_instances, timeout, check_interval
        )
    )

    while (elapsed := time.time() - start_time) < timeout:
        total_nodes = mlrs_util.get_num_ray_nodes()
        if total_nodes >= target_instances:
            # Best case scenario: target_instances are already available
            logger.info(f"Target instance requirement met: {total_nodes} instances available after {elapsed:.1f}s")
            return
        elif total_nodes >= min_instances and elapsed >= min_wait_time:
            # Second best case scenario: target_instances not met within min_wait_time, but min_instances met
            logger.info(f"Minimum instance requirement met: {total_nodes} instances available after {elapsed:.1f}s")
            return

        logger.info(
            f"Waiting for instances: current_instances={total_nodes}, min_instances={min_instances}, "
            f"target_instances={target_instances}, elapsed={elapsed:.1f}s, next check in {current_interval:.1f}s"
        )
        time.sleep(current_interval)
        current_interval = min(current_interval * 2, check_interval)  # Exponential backoff

    raise TimeoutError(
        f"Timed out after {elapsed}s waiting for {min_instances} instances, only " f"{total_nodes} available"
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

    # Ensure payload directory is in sys.path for module imports
    # This is needed because mljob_launcher.py is now in /mnt/job_stage/system
    # but user scripts are in the payload directory and may import from each other
    payload_dir = os.environ.get(PAYLOAD_DIR_ENV_VAR)
    if payload_dir and payload_dir not in sys.path:
        sys.path.insert(0, payload_dir)

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
    # Ensure the output directory exists before trying to write result files.
    output_dir = os.path.dirname(JOB_RESULT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Wait for minimum required instances if specified
        min_instances_str = os.environ.get(MIN_INSTANCES_ENV_VAR) or "1"
        target_instances_str = os.environ.get(TARGET_INSTANCES_ENV_VAR) or min_instances_str
        if target_instances_str and int(target_instances_str) > 1:
            wait_for_instances(
                int(min_instances_str),
                int(target_instances_str),
                min_wait_time=MIN_WAIT_TIME,
                timeout=TIMEOUT,
                check_interval=CHECK_INTERVAL,
            )

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
