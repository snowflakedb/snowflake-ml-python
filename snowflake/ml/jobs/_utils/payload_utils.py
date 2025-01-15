import inspect
import io
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, get_args, get_origin

import cloudpickle as cp

from snowflake import snowpark
from snowflake.ml.jobs._utils import constants, types
from snowflake.snowpark._internal import code_generation

_STARTUP_SCRIPT_PATH = Path("startup.sh")
_STARTUP_SCRIPT_CODE = textwrap.dedent(
    f"""
    #!/bin/bash

    ##### Perform common set up steps #####
    set -x # all executed commands are printed to the terminal
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

    # Configure IP address and logging directory
    eth0Ip=$(ifconfig eth0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p')
    log_dir="/tmp/ray"

    # Check if eth0Ip is empty and set default if necessary
    if [ -z "$eth0Ip" ]; then
        # This should never happen, but just in case ethOIp is not set, we should default to localhost
        eth0Ip="127.0.0.1"
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

    # Additional head-specific parameters
    head_params=(
        "--head"
        "--port=${{RAY_HEAD_GCS_PORT:-12001}}"                                  # Port of Ray (GCS server)
        "--ray-client-server-port=${{RAY_HEAD_CLIENT_SERVER_PORT:-10001}}"      # Listening port for Ray Client Server
        "--dashboard-host=${{NODE_IP_ADDRESS}}"                                 # Host to bind the dashboard server
        "--dashboard-grpc-port=${{RAY_HEAD_DASHBOARD_GRPC_PORT:-12002}}"        # Dashboard head to listen for grpc on
        "--dashboard-port=${{DASHBOARD_PORT}}"                  # Port to bind the dashboard server for local debugging
        "--resources={{\\"node_tag:head\\":1}}"                   # Resource tag for selecting head as coordinator
    )

    # Start Ray on the head node
    ray start "${{common_params[@]}}" "${{head_params[@]}}" &
    ##### End Ray configuration #####

    # TODO: Monitor MLRS and handle process crashes
    python -m web.ml_runtime_grpc_server &

    # TODO: Launch worker service(s) using SQL if Ray and MLRS successfully started

    # Run user's Python entrypoint
    echo Running command: python "$@"
    python "$@"
    """
).strip()


class JobPayload:
    def __init__(
        self,
        source: Union[str, Path, Callable[..., Any]],
        entrypoint: Optional[Union[str, Path]] = None,
        *,
        pip_requirements: Optional[List[str]] = None,
    ) -> None:
        self.source = Path(source) if isinstance(source, str) else source
        self.entrypoint = Path(entrypoint) if isinstance(entrypoint, str) else entrypoint
        self.pip_requirements = pip_requirements

    def validate(self) -> None:
        if callable(self.source):
            # Any entrypoint value is OK for callable payloads (including None aka default)
            # since we will generate the file from the serialized callable
            pass
        elif isinstance(self.source, Path):
            # Validate self.source and self.entrypoint for files
            if not self.source.exists():
                raise FileNotFoundError(f"{self.source} does not exist")
            if self.entrypoint is None:
                if self.source.is_file():
                    self.entrypoint = self.source
                else:
                    raise ValueError("entrypoint must be provided when source is a directory")
            if not self.entrypoint.is_file():
                # Check if self.entrypoint is a valid relative path
                self.entrypoint = self.source.joinpath(self.entrypoint)
                if not self.entrypoint.is_file():
                    raise FileNotFoundError(f"File {self.entrypoint} does not exist")
            if not self.entrypoint.is_relative_to(self.source):
                raise ValueError(f"{self.entrypoint} must be a subpath of {self.source}")
            if self.entrypoint.suffix != ".py":
                raise NotImplementedError("Only Python entrypoints are supported currently")
        else:
            raise ValueError("Unsupported source type. Source must be a file, directory, or callable.")

    def upload(self, session: snowpark.Session, stage_path: Union[str, Path]) -> types.UploadedPayload:
        # Validate payload
        self.validate()

        # Prepare local variables
        if isinstance(stage_path, str):
            stage_path = Path(stage_path)
        source = self.source
        entrypoint = self.entrypoint or Path(constants.DEFAULT_ENTRYPOINT_PATH)

        # Create stage if necessary
        stage_name = stage_path.parts[0]
        session.sql(f"create stage if not exists {stage_name.lstrip('@')}").collect()

        # Upload payload to stage
        if not isinstance(source, Path):
            source_code = generate_python_code(source, source_code_display=True)
            _ = session.file.put_stream(
                io.BytesIO(source_code.encode()),
                stage_location=str(stage_path.joinpath(entrypoint)),
                auto_compress=False,
                overwrite=True,
            )
            source = entrypoint.parent
        elif source.is_dir():
            # Manually traverse the directory and upload each file, since Snowflake PUT
            # can't handle directories. Reduce the number of PUT operations by using
            # wildcard patterns to batch upload files with the same extension.
            for path in {
                p.parent.joinpath(f"*{p.suffix}") if p.suffix else p for p in source.rglob("*") if p.is_file()
            }:
                session.file.put(
                    str(path.resolve()),
                    str(stage_path.joinpath(path.parent.relative_to(source))),
                    overwrite=True,
                    auto_compress=False,
                )
        else:
            session.file.put(
                str(source.resolve()),
                str(stage_path),
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
                stage_location=f"{stage_path}/requirements.txt",
                auto_compress=False,
                overwrite=True,
            )

        # Upload startup script
        # TODO: Make sure payload does not include file with same name
        session.file.put_stream(
            io.BytesIO(_STARTUP_SCRIPT_CODE.encode()),
            stage_location=f"{stage_path}/{_STARTUP_SCRIPT_PATH}",
            auto_compress=False,
            overwrite=False,  # FIXME
        )

        return types.UploadedPayload(
            stage_path=stage_path,
            entrypoint=[
                "bash",
                _STARTUP_SCRIPT_PATH,
                entrypoint.relative_to(source),
            ],
        )


def generate_python_code(func: Callable[..., Any], source_code_display: bool = False) -> str:
    signature = inspect.signature(func)
    if any(
        p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for p in signature.parameters.values()
    ):
        raise NotImplementedError("Function must not have unpacking arguments (* or **)")

    # Mirrored from Snowpark generate_python_code() function
    # https://github.com/snowflakedb/snowpark-python/blob/main/src/snowflake/snowpark/_internal/udf_utils.py
    try:
        source_code_comment = (
            code_generation.generate_source_code(func) if source_code_display else ""  # type: ignore[arg-type]
        )
    except Exception as exc:
        error_msg = f"Source code comment could not be generated for {func} due to error {exc}."
        source_code_comment = code_generation.comment_source_code(error_msg)

    func_name = "func"
    func_code = f"""
{source_code_comment}

import pickle
{func_name} = pickle.loads(bytes.fromhex('{cp.dumps(func).hex()}'))
"""

    # Generate argparse logic for argument handling (type coercion, default values, etc)
    argparse_code = ["import argparse", "", "parser = argparse.ArgumentParser()"]
    argparse_postproc = []
    for name, param in signature.parameters.items():
        opts = {}

        # Unwrap Optional type annotations
        param_type = param.annotation
        if get_origin(param_type) is Union and len(get_args(param_type)) == 2 and type(None) in get_args(param_type):
            param_type = next(t for t in get_args(param_type) if t is not type(None))

        # Check if param_type is a supported type
        if param_type in {str, int, float}:
            opts["type"] = param_type.__name__
        elif param_type != inspect.Parameter.empty:
            raise NotImplementedError(
                f"Unsupported argument type {param_type}."
                " Supported types: {','.join(t.__name__ for t in SUPPORTED_ARG_TYPES)}"
            )

        if param.default != inspect.Parameter.empty:
            opts["default"] = f"'{param.default}'" if isinstance(param.default, str) else param.default

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # Keyword argument
            opts["required"] = "default" not in opts
            argparse_code.append(f"parser.add_argument('--{name}', {', '.join(f'{k}={v}' for k, v in opts.items())})")
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

    return f"""
### Version guard to check compatibility across Python versions ###
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

    {func_name}(**vars(args))
"""
