import functools
import inspect
import io
import itertools
import pickle
import sys
import textwrap
from pathlib import Path, PurePath
from typing import Any, Callable, Optional, Union, cast, get_args, get_origin

import cloudpickle as cp

from snowflake import snowpark
from snowflake.ml.jobs._utils import constants, types
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark._internal import code_generation

_SUPPORTED_ARG_TYPES = {str, int, float}
_SUPPORTED_ENTRYPOINT_EXTENSIONS = {".py"}
_ENTRYPOINT_FUNC_NAME = "func"
_STARTUP_SCRIPT_PATH = PurePath("startup.sh")
_STARTUP_SCRIPT_CODE = textwrap.dedent(
    f"""
    #!/bin/bash

    ##### Perform common set up steps #####
    set -e # exit if a command fails

    echo "Creating log directories..."
    mkdir -p /var/log/managedservices/user/mlrs
    mkdir -p /var/log/managedservices/system/mlrs
    mkdir -p /var/log/managedservices/system/ray

    echo "*/1 * * * * root /etc/ray_copy_cron.sh" >> /etc/cron.d/ray_copy_cron
    echo "" >> /etc/cron.d/ray_copy_cron
    chmod 744 /etc/cron.d/ray_copy_cron

    service cron start

    mkdir -p /tmp/prometheus-multi-dir

    # Change directory to user payload directory
    if [ -n "${constants.PAYLOAD_DIR_ENV_VAR}" ]; then
        cd ${constants.PAYLOAD_DIR_ENV_VAR}
    fi

    ##### Set up Python environment #####
    export PYTHONPATH=/opt/env/site-packages/
    MLRS_REQUIREMENTS_FILE=${{MLRS_REQUIREMENTS_FILE:-"requirements.txt"}}
    if [ -f "${{MLRS_REQUIREMENTS_FILE}}" ]; then
        # TODO: Prevent collisions with MLRS packages using virtualenvs
        echo "Installing packages from $MLRS_REQUIREMENTS_FILE"
        pip install -r $MLRS_REQUIREMENTS_FILE
    fi

    MLRS_CONDA_ENV_FILE=${{MLRS_CONDA_ENV_FILE:-"environment.yml"}}
    if [ -f "${{MLRS_CONDA_ENV_FILE}}" ]; then
        # TODO: Handle conda environment
        echo "Custom conda environments not currently supported"
        exit 1
    fi
    ##### End Python environment setup #####

    ##### Ray configuration #####
    shm_size=$(df --output=size --block-size=1 /dev/shm | tail -n 1)

    # Check if the local get_instance_ip.py script exists
    HELPER_EXISTS=$(
        [ -f "get_instance_ip.py" ] && echo "true" || echo "false"
    )

    # Configure IP address and logging directory
    if [ "$HELPER_EXISTS" = "true" ]; then
        eth0Ip=$(python3 get_instance_ip.py "$SNOWFLAKE_SERVICE_NAME" --instance-index=-1)
    else
        eth0Ip=$(ifconfig eth0 2>/dev/null | sed -En -e 's/.*inet ([0-9.]+).*/\1/p')
    fi
    log_dir="/tmp/ray"

    # Check if eth0Ip is a valid IP address and fall back to default if necessary
    if [[ ! $eth0Ip =~ ^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$ ]]; then
        eth0Ip="127.0.0.1"
    fi

    # Get the environment values of SNOWFLAKE_JOBS_COUNT and SNOWFLAKE_JOB_INDEX for batch jobs
    # These variables don't exist for non-batch jobs, so set defaults
    if [ -z "$SNOWFLAKE_JOBS_COUNT" ]; then
        SNOWFLAKE_JOBS_COUNT=1
    fi

    if [ -z "$SNOWFLAKE_JOB_INDEX" ]; then
        SNOWFLAKE_JOB_INDEX=0
    fi

    # Determine if it should be a worker or a head node for batch jobs
    if [[ "$SNOWFLAKE_JOBS_COUNT" -gt 1 && "$HELPER_EXISTS" = "true" ]]; then
        head_info=$(python3 get_instance_ip.py "$SNOWFLAKE_SERVICE_NAME" --head)
        if [ $? -eq 0 ]; then
            # Parse the output using read
            read head_index head_ip <<< "$head_info"

            # Use the parsed variables
            echo "Head Instance Index: $head_index"
            echo "Head Instance IP: $head_ip"

        else
            echo "Error: Failed to get head instance information."
            echo "$head_info" # Print the error message
            exit 1
        fi

        if [ "$SNOWFLAKE_JOB_INDEX" -ne "$head_index" ]; then
            NODE_TYPE="worker"
        fi
    fi

    # Common parameters for both head and worker nodes
    common_params=(
        "--node-ip-address=$eth0Ip"
        "--object-manager-port=${{RAY_OBJECT_MANAGER_PORT:-12011}}"
        "--node-manager-port=${{RAY_NODE_MANAGER_PORT:-12012}}"
        "--runtime-env-agent-port=${{RAY_RUNTIME_ENV_AGENT_PORT:-12013}}"
        "--dashboard-agent-grpc-port=${{RAY_DASHBOARD_AGENT_GRPC_PORT:-12014}}"
        "--dashboard-agent-listen-port=${{RAY_DASHBOARD_AGENT_LISTEN_PORT:-12015}}"
        "--min-worker-port=${{RAY_MIN_WORKER_PORT:-12031}}"
        "--max-worker-port=${{RAY_MAX_WORKER_PORT:-13000}}"
        "--metrics-export-port=11502"
        "--temp-dir=$log_dir"
        "--disable-usage-stats"
    )

    if [ "$NODE_TYPE" = "worker" ]; then
        # Use head_ip as head address if it exists
        if [ ! -z "$head_ip" ]; then
            RAY_HEAD_ADDRESS="$head_ip"
        fi

        # If RAY_HEAD_ADDRESS is still empty, exit with an error
        if [ -z "$RAY_HEAD_ADDRESS" ]; then
            echo "Error: Failed to determine head node address using default instance-index=0"
            exit 1
        fi

        if [ -z "$SERVICE_NAME" ]; then
            SERVICE_NAME="$SNOWFLAKE_SERVICE_NAME"
        fi

        if [ -z "$RAY_HEAD_ADDRESS" ] || [ -z "$SERVICE_NAME" ]; then
            echo "Error: RAY_HEAD_ADDRESS and SERVICE_NAME must be set."
            exit 1
        fi

        # Additional worker-specific parameters
        worker_params=(
            "--address=${{RAY_HEAD_ADDRESS}}:12001"       # Connect to head node
            "--resources={{\\"${{SERVICE_NAME}}\\":1, \\"node_tag:worker\\":1}}"  # Tag for node identification
            "--object-store-memory=${{shm_size}}"
        )

        # Start Ray on a worker node - run in background
        ray start "${{common_params[@]}}" "${{worker_params[@]}}" -v --block &

        # Start the worker shutdown listener in the background
        echo "Starting worker shutdown listener..."
        python worker_shutdown_listener.py
        WORKER_EXIT_CODE=$?

        echo "Worker shutdown listener exited with code $WORKER_EXIT_CODE"
        exit $WORKER_EXIT_CODE
    else
        # Additional head-specific parameters
        head_params=(
            "--head"
            "--port=${{RAY_HEAD_GCS_PORT:-12001}}"                                  # Port of Ray (GCS server)
            "--ray-client-server-port=${{RAY_HEAD_CLIENT_SERVER_PORT:-10001}}"      # Rort for Ray Client Server
            "--dashboard-host=${{NODE_IP_ADDRESS}}"                                 # Host to bind the dashboard server
            "--dashboard-grpc-port=${{RAY_HEAD_DASHBOARD_GRPC_PORT:-12002}}"        # Dashboard head to listen for grpc
            "--dashboard-port=${{DASHBOARD_PORT}}"                  # Port to bind the dashboard server for debugging
            "--resources={{\\"node_tag:head\\":1}}"                   # Resource tag for selecting head as coordinator
        )

        # Start Ray on the head node
        ray start "${{common_params[@]}}" "${{head_params[@]}}" -v
        ##### End Ray configuration #####

        # TODO: Monitor MLRS and handle process crashes
        python -m web.ml_runtime_grpc_server &

        # TODO: Launch worker service(s) using SQL if Ray and MLRS successfully started

        # Run user's Python entrypoint
        echo Running command: python "$@"
        python "$@"

        # After the user's job completes, signal workers to shut down
        echo "User job completed. Signaling workers to shut down..."
        python signal_workers.py --wait-time 15
        echo "Head node job completed. Exiting."
    fi
    """
).strip()


def resolve_source(source: Union[Path, Callable[..., Any]]) -> Union[Path, Callable[..., Any]]:
    if callable(source):
        return source
    elif isinstance(source, Path):
        # Validate source
        source = source
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist")
        return source.absolute()
    else:
        raise ValueError("Unsupported source type. Source must be a file, directory, or callable.")


def resolve_entrypoint(source: Union[Path, Callable[..., Any]], entrypoint: Optional[Path]) -> types.PayloadEntrypoint:
    if callable(source):
        # Entrypoint is generated for callable payloads
        return types.PayloadEntrypoint(
            file_path=entrypoint or Path(constants.DEFAULT_ENTRYPOINT_PATH),
            main_func=_ENTRYPOINT_FUNC_NAME,
        )

    # Resolve entrypoint path for file-based payloads
    parent = source.absolute()
    if entrypoint is None:
        if parent.is_file():
            # Infer entrypoint from source
            entrypoint = parent
        else:
            raise ValueError("entrypoint must be provided when source is a directory")
    elif entrypoint.is_absolute():
        # Absolute path - validate it's a subpath of source dir
        if not entrypoint.is_relative_to(parent):
            raise ValueError(f"Entrypoint must be a subpath of {parent}, got: {entrypoint})")
    else:
        # Relative path
        if (abs_entrypoint := entrypoint.absolute()).is_relative_to(parent) and abs_entrypoint.is_file():
            # Relative to working dir iff path is relative to source dir and exists
            entrypoint = abs_entrypoint
        else:
            # Relative to source dir
            entrypoint = parent.joinpath(entrypoint)

    # Validate resolved entrypoint file
    if not entrypoint.is_file():
        raise FileNotFoundError(
            "Entrypoint not found. Ensure the entrypoint is a valid file and is under"
            f" the source directory (source={parent}, entrypoint={entrypoint})"
        )
    if entrypoint.suffix not in _SUPPORTED_ENTRYPOINT_EXTENSIONS:
        raise ValueError(
            "Unsupported entrypoint type:"
            f" supported={','.join(_SUPPORTED_ENTRYPOINT_EXTENSIONS)} got={entrypoint.suffix}"
        )

    return types.PayloadEntrypoint(
        file_path=entrypoint,  # entrypoint is an absolute path at this point
        main_func=None,
    )


class JobPayload:
    def __init__(
        self,
        source: Union[str, Path, Callable[..., Any]],
        entrypoint: Optional[Union[str, Path]] = None,
        *,
        pip_requirements: Optional[list[str]] = None,
    ) -> None:
        self.source = Path(source) if isinstance(source, str) else source
        self.entrypoint = Path(entrypoint) if isinstance(entrypoint, str) else entrypoint
        self.pip_requirements = pip_requirements

    def upload(self, session: snowpark.Session, stage_path: Union[str, PurePath]) -> types.UploadedPayload:
        # Prepare local variables
        stage_path = PurePath(stage_path) if isinstance(stage_path, str) else stage_path
        source = resolve_source(self.source)
        entrypoint = resolve_entrypoint(source, self.entrypoint)

        # Create stage if necessary
        stage_name = stage_path.parts[0].lstrip("@")
        # Explicitly check if stage exists first since we may not have CREATE STAGE privilege
        try:
            session.sql(f"describe stage {stage_name}").collect()
        except sp_exceptions.SnowparkSQLException:
            session.sql(
                f"create stage if not exists {stage_name}"
                " encryption = ( type = 'SNOWFLAKE_SSE' )"
                " comment = 'Created by snowflake.ml.jobs Python API'"
            ).collect()

        # Upload payload to stage
        if not isinstance(source, Path):
            source_code = generate_python_code(source, source_code_display=True)
            _ = session.file.put_stream(
                io.BytesIO(source_code.encode()),
                stage_location=stage_path.joinpath(entrypoint.file_path).as_posix(),
                auto_compress=False,
                overwrite=True,
            )
            source = Path(entrypoint.file_path.parent)
        elif source.is_dir():
            # Manually traverse the directory and upload each file, since Snowflake PUT
            # can't handle directories. Reduce the number of PUT operations by using
            # wildcard patterns to batch upload files with the same extension.
            for path in {
                p.parent.joinpath(f"*{p.suffix}") if p.suffix else p for p in source.resolve().rglob("*") if p.is_file()
            }:
                session.file.put(
                    str(path),
                    stage_path.joinpath(path.parent.relative_to(source)).as_posix(),
                    overwrite=True,
                    auto_compress=False,
                )
        else:
            session.file.put(
                str(source.resolve()),
                stage_path.as_posix(),
                overwrite=True,
                auto_compress=False,
            )
            source = source.parent

        # Upload requirements
        # TODO: Check if payload includes both a requirements.txt file and pip_requirements
        if self.pip_requirements:
            # Upload requirements.txt to stage
            session.file.put_stream(
                io.BytesIO("\n".join(self.pip_requirements).encode()),
                stage_location=stage_path.joinpath("requirements.txt").as_posix(),
                auto_compress=False,
                overwrite=True,
            )

        # Upload startup script
        # TODO: Make sure payload does not include file with same name
        session.file.put_stream(
            io.BytesIO(_STARTUP_SCRIPT_CODE.encode()),
            stage_location=stage_path.joinpath(_STARTUP_SCRIPT_PATH).as_posix(),
            auto_compress=False,
            overwrite=False,  # FIXME
        )

        # Upload system scripts
        scripts_dir = Path(__file__).parent.joinpath("scripts")
        for script_file in scripts_dir.glob("*"):
            if script_file.is_file():
                session.file.put(
                    script_file.as_posix(),
                    stage_path.as_posix(),
                    overwrite=True,
                    auto_compress=False,
                )

        python_entrypoint: list[Union[str, PurePath]] = [
            PurePath("mljob_launcher.py"),
            entrypoint.file_path.relative_to(source),
        ]
        if entrypoint.main_func:
            python_entrypoint += ["--script_main_func", entrypoint.main_func]

        return types.UploadedPayload(
            stage_path=stage_path,
            entrypoint=[
                "bash",
                _STARTUP_SCRIPT_PATH,
                *python_entrypoint,
            ],
        )


def _get_parameter_type(param: inspect.Parameter) -> Optional[type[object]]:
    # Unwrap Optional type annotations
    param_type = param.annotation
    if get_origin(param_type) is Union and len(get_args(param_type)) == 2 and type(None) in get_args(param_type):
        param_type = next(t for t in get_args(param_type) if t is not type(None))

    # Return None for empty type annotations
    if param_type == inspect.Parameter.empty:
        return None
    return cast(type[object], param_type)


def _validate_parameter_type(param_type: type[object], param_name: str) -> None:
    # Validate param_type is a supported type
    if param_type not in _SUPPORTED_ARG_TYPES:
        raise ValueError(
            f"Unsupported argument type {param_type} for '{param_name}'."
            f" Supported types: {', '.join(t.__name__ for t in _SUPPORTED_ARG_TYPES)}"
        )


def _generate_source_code_comment(func: Callable[..., Any]) -> str:
    """Generate a comment string containing the source code of a function for readability."""
    try:
        if isinstance(func, functools.partial):
            # Unwrap functools.partial and generate source code comment from the original function
            comment = code_generation.generate_source_code(func.func)  # type: ignore[arg-type]
            args = itertools.chain((repr(a) for a in func.args), (f"{k}={v!r}" for k, v in func.keywords.items()))

            # Update invocation comment to show arguments passed via functools.partial
            comment = comment.replace(
                f"= {func.func.__name__}",
                "= functools.partial({}({}))".format(
                    func.func.__name__,
                    ", ".join(args),
                ),
            )
            return comment
        else:
            return code_generation.generate_source_code(func)  # type: ignore[arg-type]
    except Exception as exc:
        error_msg = f"Source code comment could not be generated for {func} due to error {exc}."
        return code_generation.comment_source_code(error_msg)


def _serialize_callable(func: Callable[..., Any]) -> bytes:
    try:
        func_bytes: bytes = cp.dumps(func)
        return func_bytes
    except pickle.PicklingError as e:
        if isinstance(func, functools.partial):
            # Try to find which part of the partial isn't serializable for better debuggability
            objects = [
                ("function", func.func),
                *((f"positional arg {i}", a) for i, a in enumerate(func.args)),
                *((f"keyword arg '{k}'", v) for k, v in func.keywords.items()),
            ]
            for name, obj in objects:
                try:
                    cp.dumps(obj)
                except pickle.PicklingError:
                    raise ValueError(f"Unable to serialize {name}: {obj}") from e
        raise ValueError(f"Unable to serialize function: {func}") from e


def _generate_param_handler_code(signature: inspect.Signature, output_name: str = "kwargs") -> str:
    # Generate argparse logic for argument handling (type coercion, default values, etc)
    argparse_code = ["import argparse", "", "parser = argparse.ArgumentParser()"]
    argparse_postproc = []
    for name, param in signature.parameters.items():
        opts = {}

        param_type = _get_parameter_type(param)
        if param_type is not None:
            _validate_parameter_type(param_type, name)
            opts["type"] = param_type.__name__

        if param.default != inspect.Parameter.empty:
            opts["default"] = f"'{param.default}'" if isinstance(param.default, str) else param.default

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # Keyword argument
            argparse_code.append(
                f"parser.add_argument('--{name}', required={'default' not in opts},"
                f" {', '.join(f'{k}={v}' for k, v in opts.items())})"
            )
        else:
            # Positional argument. Use `argparse.add_mutually_exclusive_group()`
            # to allow passing positional args by name as well
            group_name = f"{name}_group"
            argparse_code.append(
                f"{group_name} = parser.add_mutually_exclusive_group(required={'default' not in opts})"
            )
            argparse_code.append(
                f"{group_name}.add_argument('pos-{name}', metavar='{name}', nargs='?',"
                f" {', '.join(f'{k}={v}' for k, v in opts.items() if k != 'default')})"
            )
            argparse_code.append(
                f"{group_name}.add_argument('--{name}', {', '.join(f'{k}={v}' for k, v in opts.items())})"
            )
            argparse_code.append("")  # Add newline for readability
            argparse_postproc.append(
                f"args.{name} = {name} if ({name} := args.__dict__.pop('pos-{name}')) is not None else args.{name}"
            )
    argparse_code.append("args = parser.parse_args()")
    param_code = "\n".join(argparse_code + argparse_postproc)
    param_code += f"\n{output_name} = vars(args)"

    return param_code


def generate_python_code(func: Callable[..., Any], source_code_display: bool = False) -> str:
    """Generate an entrypoint script from a Python function."""
    signature = inspect.signature(func)
    if any(
        p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for p in signature.parameters.values()
    ):
        raise NotImplementedError("Function must not have unpacking arguments (* or **)")

    # Mirrored from Snowpark generate_python_code() function
    # https://github.com/snowflakedb/snowpark-python/blob/main/src/snowflake/snowpark/_internal/udf_utils.py
    source_code_comment = _generate_source_code_comment(func) if source_code_display else ""

    func_code = f"""
{source_code_comment}

import pickle
{_ENTRYPOINT_FUNC_NAME} = pickle.loads(bytes.fromhex('{_serialize_callable(func).hex()}'))
"""

    arg_dict_name = "kwargs"
    if getattr(func, constants.IS_MLJOB_REMOTE_ATTR, None):
        param_code = f"{arg_dict_name} = {{}}"
    else:
        param_code = _generate_param_handler_code(signature, arg_dict_name)

    return f"""
### Version guard to check compatibility across Python versions ###
import os
import sys
import warnings

if sys.version_info.major != {sys.version_info.major} or sys.version_info.minor != {sys.version_info.minor}:
    warnings.warn(
        "Python version mismatch: job was created using"
        " python{sys.version_info.major}.{sys.version_info.minor}"
        f" but runtime environment uses python{{sys.version_info.major}}.{{sys.version_info.minor}}."
        " Compatibility across Python versions is not guaranteed and may result in unexpected behavior."
        " This will be fixed in a future release; for now, please use Python version"
        f" {{sys.version_info.major}}.{{sys.version_info.minor}}.",
        RuntimeWarning,
        stacklevel=0,
    )
### End version guard ###

{func_code.strip()}

if __name__ == '__main__':
{textwrap.indent(param_code, '    ')}

    __return__ = {_ENTRYPOINT_FUNC_NAME}(**{arg_dict_name})
"""
