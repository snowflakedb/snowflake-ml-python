import dataclasses
import json
import logging
import os
import sys
from pathlib import PurePath, PurePosixPath
from typing import Any, Callable, Generic, Optional, TypeVar, Union
from uuid import uuid4

from typing_extensions import ParamSpec

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.mixins import SerializableSessionMixin
from snowflake.ml.jobs import job as jb
from snowflake.ml.jobs._interop import utils as interop_utils
from snowflake.ml.jobs._utils import (
    arg_protocol,
    constants,
    feature_flags,
    payload_utils,
    query_helper,
    types,
)
from snowflake.snowpark import context as sp_context
from snowflake.snowpark._internal import utils
from snowflake.snowpark.exceptions import SnowparkSQLException

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")
JOB_ID_PREFIX = "MLJOB_"
_PROJECT = "MLJob"
logger = logging.getLogger(__name__)


class MLJobDefinition(Generic[_Args, _ReturnValue], SerializableSessionMixin):
    def __init__(
        self,
        source: Union[str, Callable[_Args, _ReturnValue]],
        compute_pool: str,
        stage_name: str,
        session: Optional[snowpark.Session] = None,
        name: Optional[str] = None,
        target_instances: int = 1,
        min_instances: Optional[int] = None,
        generate_suffix: bool = True,
        external_access_integrations: Optional[list[str]] = None,
        env_vars: Optional[dict[str, str]] = None,
        spec_overrides: Optional[dict[str, Any]] = None,
        enable_metrics: bool = True,
        query_warehouse: Optional[str] = None,
        runtime_environment: Optional[str] = None,
        overwrite: bool = False,
        arg_protocol: arg_protocol.ArgProtocol = arg_protocol.ArgProtocol.NONE,
        default_args: Optional[list[Any]] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        payload: Optional[payload_utils.JobPayload] = None,
    ) -> None:
        self.source = source
        self.compute_pool = identifier.resolve_identifier(compute_pool)
        self.stage_name = stage_name
        self.session = session
        self.query_warehouse = identifier.resolve_identifier(query_warehouse) if query_warehouse is not None else None
        self.arg_protocol = arg_protocol
        self.default_args = default_args
        self.database = database
        self.schema = schema
        self.name = name
        self.target_instances = target_instances
        self.min_instances = min_instances or target_instances
        self.generate_suffix = generate_suffix
        self.external_access_integrations = external_access_integrations
        self.env_vars = env_vars
        self.spec_overrides = spec_overrides
        self.enable_metrics = enable_metrics
        self.runtime_environment = runtime_environment
        self.overwrite = overwrite
        self.payload = payload

        self._is_registered = False

    def _ensure_registered(self) -> None:
        if not self._is_registered:
            self._register()

    def _register(self) -> None:
        self.session = self.session or sp_context.get_active_session()

        if self.min_instances > 1:
            # Validate min_instances against compute pool max_nodes
            pool_info = jb._get_compute_pool_info(self.session, self.compute_pool)
            max_nodes = int(pool_info["max_nodes"])
            if self.min_instances > max_nodes:
                raise ValueError(
                    f"The requested min_instances ({self.min_instances}) exceeds the max_nodes ({max_nodes}) "
                    f"of compute pool '{self.compute_pool}'. Reduce min_instances or increase max_nodes."
                )

        database = self.database or self.session.get_current_database()
        schema = self.schema or self.session.get_current_schema()

        if database is None:
            raise ValueError("Database must be specified either in the session context or as a parameter.")
        if schema is None:
            raise ValueError("Schema must be specified either in the session context or as a parameter.")

        self.database = identifier.resolve_identifier(database)
        self.schema = identifier.resolve_identifier(schema)

        assert self.name is not None
        self.job_definition_id = identifier.get_schema_level_object_identifier(self.database, self.schema, self.name)
        stage_path_name = self.name if not self.generate_suffix else self.name + _generate_suffix()

        stage_path_parts = identifier.parse_snowflake_stage_path(self.stage_name.lstrip("@"))
        stage_name_prefix = f"@{'.'.join(filter(None, stage_path_parts[:3]))}"
        stage_path = PurePosixPath(f"{stage_name_prefix}{stage_path_parts[-1].rstrip('/')}/{stage_path_name}")
        self.stage_name = stage_path.as_posix()

        try:
            # Upload payload
            assert self.payload is not None
            uploaded_payload = self.payload.upload(self.session, stage_path, self.overwrite)
        except SnowparkSQLException as e:
            if e.sql_error_code == 90106:
                raise RuntimeError(
                    "Please specify a schema, either in the session context or as a parameter in the job submission"
                )
            raise

        if self.runtime_environment is None and feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS.is_enabled():
            # Pass a JSON object for runtime versions so it serializes as nested JSON in options
            self.runtime_environment = json.dumps(
                {"pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}"}
            )

        combined_env_vars = {**uploaded_payload.env_vars, **(self.env_vars or {})}
        self.entrypoint_args = [v.as_posix() if isinstance(v, PurePath) else v for v in uploaded_payload.entrypoint]
        self.spec_options = types.SpecOptions(
            stage_path=stage_path.as_posix(),
            # the args will be set at runtime
            args=None,
            env_vars=combined_env_vars,
            enable_metrics=self.enable_metrics,
            spec_overrides=self.spec_overrides,
            runtime=self.runtime_environment if self.runtime_environment else None,
            enable_stage_mount_v2=feature_flags.FeatureFlags.ENABLE_STAGE_MOUNT_V2.is_enabled(),
        )

        self.job_options = types.JobOptions(
            external_access_integrations=self.external_access_integrations,
            query_warehouse=self.query_warehouse or self.session.get_current_warehouse(),
            target_instances=self.target_instances,
            min_instances=self.min_instances,
            generate_suffix=self.generate_suffix,
        )

        self._is_registered = True

    def delete(self) -> None:
        if not self._is_registered:
            return
        if self.session is None:
            raise RuntimeError("Session is required to delete job definition")
        if self.stage_name:
            try:
                self.session.sql(f"REMOVE {self.stage_name}/").collect()
                logger.debug(f"Successfully cleaned up stage files for job definition {self.stage_name}")
            except Exception as e:
                logger.warning(f"Failed to clean up stage files for job definition {self.stage_name}: {e}")

    def _prepare_arguments(self, *args: _Args.args, **kwargs: _Args.kwargs) -> Optional[list[Any]]:
        if self.arg_protocol == arg_protocol.ArgProtocol.NONE:
            if len(kwargs) > 0:
                raise ValueError(f"Keyword arguments are not supported with {self.arg_protocol}")
            return list(args)
        elif self.arg_protocol == arg_protocol.ArgProtocol.CLI:
            return _combine_runtime_arguments(self.default_args, *args, **kwargs)
        elif self.arg_protocol == arg_protocol.ArgProtocol.PICKLE:
            if not args and not kwargs:
                return []
            uid = uuid4().hex[:8]
            rel_path = f"{uid}/function_args"
            file_path = f"{self.stage_name}/{constants.APP_STAGE_SUBPATH}/{rel_path}"
            payload = interop_utils.save_result(
                (args, kwargs), file_path, session=self.session, max_inline_size=interop_utils._MAX_INLINE_SIZE
            )
            if payload is not None:
                return [f"--function_args={payload.decode('utf-8')}"]
            return [f"--function_args={rel_path}"]
        else:
            raise ValueError(f"Invalid arg_protocol: {self.arg_protocol}")

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def __call__(self, *args: _Args.args, **kwargs: _Args.kwargs) -> jb.MLJob[_ReturnValue]:
        # we need session to upload the arguments to the stage if the arg_protocol is PICKLE
        self._ensure_registered()
        statement_params = telemetry.get_statement_params(_PROJECT)
        statement_params = telemetry.add_statement_params_custom_tags(
            statement_params,
            custom_tags={
                "job_definition_id": self.job_definition_id,
            },
        )
        args_list = self._prepare_arguments(*args, **kwargs)
        query = self.to_sql(job_args=args_list, use_async=True)
        job_id = query_helper.run_query(self.session, query, statement_params=statement_params)[0][0]
        return jb.MLJob[_ReturnValue](job_id, session=self.session)

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def to_sql(self, *, job_args: Optional[list[Any]] = None, use_async: bool = False) -> str:
        self._ensure_registered()
        # Combine the entrypoint_args and job_args for use in the query
        combined_args = (self.entrypoint_args or []) + (job_args or [])
        spec_options = dataclasses.replace(self.spec_options, args=combined_args)
        # Uppercase option keys to match the expected SYSTEM$EXECUTE_ML_JOB parameter format
        spec_options_dict = {k.upper(): v for k, v in dataclasses.asdict(spec_options).items()}
        job_options = dataclasses.replace(self.job_options, use_async=use_async)
        # Uppercase option keys to match the expected SYSTEM$EXECUTE_ML_JOB parameter format
        job_options_dict = {k.upper(): v for k, v in dataclasses.asdict(job_options).items()}
        job_options_dict["ASYNC"] = job_options_dict.pop("USE_ASYNC")
        params = [
            self.job_definition_id + ("_" if self.job_options.generate_suffix else ""),
            self.compute_pool,
            json.dumps(spec_options_dict),
            json.dumps(job_options_dict),
        ]
        query_template = "CALL SYSTEM$EXECUTE_ML_JOB(%s, %s, %s, %s)"
        assert self.session is not None, "Session is required to generate MLJob SQL query"
        sql = self.session._conn._cursor._preprocess_pyformat_query(query_template, params)
        return sql

    @classmethod
    def _create(
        cls,
        source: Union[str, Callable[_Args, _ReturnValue]],
        compute_pool: str,
        stage_name: str,
        session: Optional[snowpark.Session] = None,
        **kwargs: Any,
    ) -> "MLJobDefinition[_Args, _ReturnValue]":
        """
        Create and register a new MLJobDefinition instance for internal use, avoid warning logs.
        """
        # Extract and validate parameters on the client side
        entrypoint = kwargs.pop("entrypoint", None)
        target_instances = kwargs.pop("target_instances", 1)
        generate_suffix = kwargs.pop("generate_suffix", True)
        min_instances = kwargs.pop("min_instances", target_instances)
        pip_requirements = kwargs.pop("pip_requirements", None)
        external_access_integrations = kwargs.pop("external_access_integrations", None)
        env_vars = kwargs.pop("env_vars", None)
        spec_overrides = kwargs.pop("spec_overrides", None)
        enable_metrics = kwargs.pop("enable_metrics", True)
        query_warehouse = kwargs.pop("query_warehouse", None)
        imports = kwargs.pop("imports", None)
        runtime_environment = kwargs.pop(
            "runtime_environment", os.environ.get(constants.RUNTIME_IMAGE_TAG_ENV_VAR, None)
        )
        overwrite = kwargs.pop("overwrite", False)
        name = kwargs.pop("name", None)

        arg_protocol_val = kwargs.pop("arg_protocol", arg_protocol.ArgProtocol.NONE)
        default_args = kwargs.pop("default_args", None)

        database = kwargs.pop("database", None)
        schema = kwargs.pop("schema", None)

        if database and not schema:
            raise ValueError("Schema must be specified if database is specified.")

        if kwargs:
            logger.warning(f"Ignoring unknown kwargs: {kwargs.keys()}")

        if target_instances < 1:
            raise ValueError("target_instances must be greater than 0.")
        if not (0 < min_instances <= target_instances):
            raise ValueError("min_instances must be greater than 0 and less than or equal to target_instances.")

        if name:
            parsed_database, parsed_schema, parsed_name = identifier.parse_schema_level_object_identifier(name)
            database = parsed_database or database
            schema = parsed_schema or schema
            name = parsed_name
        else:
            name = payload_utils.get_payload_name(source, entrypoint)

        # Initialize JobPayload (validates source, entrypoint, and imports)
        payload = payload_utils.JobPayload(
            source, entrypoint=entrypoint, pip_requirements=pip_requirements, imports=imports
        )

        return MLJobDefinition[_Args, _ReturnValue](
            source=source,
            compute_pool=compute_pool,
            stage_name=stage_name,
            session=session,
            name=name,
            target_instances=target_instances,
            min_instances=min_instances,
            generate_suffix=generate_suffix,
            external_access_integrations=external_access_integrations,
            env_vars=env_vars,
            spec_overrides=spec_overrides,
            enable_metrics=enable_metrics,
            query_warehouse=query_warehouse,
            runtime_environment=runtime_environment,
            overwrite=overwrite,
            arg_protocol=arg_protocol_val,
            default_args=default_args,
            database=database,
            schema=schema,
            payload=payload,
        )

    @classmethod
    @utils.private_preview(version="1.26.0")
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        func_params_to_log=[
            "pip_requirements",
            "external_access_integrations",
            "target_instances",
            "min_instances",
            "enable_metrics",
            "query_warehouse",
            "runtime_environment",
        ],
    )
    def register(
        cls,
        source: Union[str, Callable[_Args, _ReturnValue]],
        compute_pool: str,
        stage_name: str,
        session: Optional[snowpark.Session] = None,
        **kwargs: Any,
    ) -> "MLJobDefinition[_Args, _ReturnValue]":
        """
        Create and register a new MLJobDefinition instance eagerly.
        """
        job_definition = cls._create(
            source=source, compute_pool=compute_pool, stage_name=stage_name, session=session, **kwargs
        )
        job_definition._register()
        return job_definition


def _generate_suffix() -> str:
    return str(uuid4().hex)[:8]


def _combine_runtime_arguments(
    default_runtime_args: Optional[list[Any]] = None, *args: Any, **kwargs: Any
) -> list[Any]:
    """Merge default CLI arguments with runtime overrides into a flat argument list.

    Parses `default_runtime_args` for flags (e.g., `--key value`) and merges them with
    `kwargs`. Keyword arguments override defaults unless their value is None. Positional
    arguments from both `default_args` and `*args` are preserved in order.

    Args:
        default_runtime_args: Optional list of default CLI arguments to parse for flags and positional args.
        *args: Additional positional arguments to include in the output.
        **kwargs: Keyword arguments that override default flags. Values of None are ignored.

    Returns:
        A list of CLI-style arguments: positional args followed by `--key value` pairs.
    """
    cli_args = list(args)
    flags: dict[str, Any] = {}
    if default_runtime_args:
        i = 0
        while i < len(default_runtime_args):
            arg = default_runtime_args[i]
            if isinstance(arg, str) and arg.startswith("--"):
                key = arg[2:]
                # Check if next arg is a value (not a flag)
                if i + 1 < len(default_runtime_args):
                    next_arg = default_runtime_args[i + 1]
                    if not (isinstance(next_arg, str) and next_arg.startswith("--")):
                        flags[key] = next_arg
                        i += 2
                        continue

                flags[key] = None
            else:
                cli_args.append(arg)
            i += 1
    # Prioritize kwargs over default_args. Explicit None values in kwargs
    # serve as overrides and are converted to the string "None" to match
    # CLI flag conventions (--key=value)
    # Downstream logic must handle the parsing of these string-based nulls.
    for k, v in kwargs.items():
        flags[k] = v
    for k, v in flags.items():
        cli_args.extend([f"--{k}", str(v)])
    return cli_args
