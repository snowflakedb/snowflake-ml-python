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
from snowflake.ml.jobs._utils import (
    constants,
    feature_flags,
    payload_utils,
    query_helper,
    types,
)
from snowflake.snowpark import context as sp_context
from snowflake.snowpark.exceptions import SnowparkSQLException

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")
JOB_ID_PREFIX = "MLJOB_"
_PROJECT = "MLJob"
logger = logging.getLogger(__name__)


class MLJobDefinition(Generic[_Args, _ReturnValue], SerializableSessionMixin):
    def __init__(
        self,
        job_options: types.JobOptions,
        spec_options: types.SpecOptions,
        stage_name: str,
        compute_pool: str,
        name: str,
        entrypoint_args: list[Any],
        database: Optional[str] = None,
        schema: Optional[str] = None,
        session: Optional[snowpark.Session] = None,
    ) -> None:
        self.stage_name = stage_name
        self.job_options = job_options
        self.spec_options = spec_options
        self.compute_pool = compute_pool
        self.session = session or sp_context.get_active_session()
        self.database = database or self.session.get_current_database()
        self.schema = schema or self.session.get_current_schema()
        self.job_definition_id = identifier.get_schema_level_object_identifier(self.database, self.schema, name)
        self.entrypoint_args = entrypoint_args

    def delete(self) -> None:
        if self.stage_name:
            try:
                self.session.sql(f"REMOVE {self.stage_name}/").collect()
                logger.debug(f"Successfully cleaned up stage files for job definition {self.stage_name}")
            except Exception as e:
                logger.warning(f"Failed to clean up stage files for job definition {self.stage_name}: {e}")

    def _prepare_arguments(self, *args: _Args.args, **kwargs: _Args.kwargs) -> list[Any]:
        # TODO: Add ArgProtocol and respective logics
        return [arg for arg in args]

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def __call__(self, *args: _Args.args, **kwargs: _Args.kwargs) -> jb.MLJob[_ReturnValue]:
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
        sql = self.session._conn._cursor._preprocess_pyformat_query(query_template, params)
        return sql

    @classmethod
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
        entrypoint: Optional[Union[str, list[str]]] = None,
        target_instances: int = 1,
        generate_suffix: bool = True,
        **kwargs: Any,
    ) -> "MLJobDefinition[_Args, _ReturnValue]":
        # Use kwargs for less common optional parameters
        database = kwargs.pop("database", None)
        schema = kwargs.pop("schema", None)
        min_instances = kwargs.pop("min_instances", target_instances)
        pip_requirements = kwargs.pop("pip_requirements", None)
        external_access_integrations = kwargs.pop("external_access_integrations", None)
        env_vars = kwargs.pop("env_vars", None)
        spec_overrides = kwargs.pop("spec_overrides", None)
        enable_metrics = kwargs.pop("enable_metrics", True)
        session = session or sp_context.get_active_session()
        query_warehouse = kwargs.pop("query_warehouse", session.get_current_warehouse())
        imports = kwargs.pop("imports", None)
        runtime_environment = kwargs.pop(
            "runtime_environment", os.environ.get(constants.RUNTIME_IMAGE_TAG_ENV_VAR, None)
        )
        overwrite = kwargs.pop("overwrite", False)
        name = kwargs.pop("name", None)
        # Warn if there are unknown kwargs
        if kwargs:
            logger.warning(f"Ignoring unknown kwargs: {kwargs.keys()}")

        # Validate parameters
        if database and not schema:
            raise ValueError("Schema must be specified if database is specified.")
        if target_instances < 1:
            raise ValueError("target_instances must be greater than 0.")
        if not (0 < min_instances <= target_instances):
            raise ValueError("min_instances must be greater than 0 and less than or equal to target_instances.")
        if min_instances > 1:
            # Validate min_instances against compute pool max_nodes
            pool_info = jb._get_compute_pool_info(session, compute_pool)
            max_nodes = int(pool_info["max_nodes"])
            if min_instances > max_nodes:
                raise ValueError(
                    f"The requested min_instances ({min_instances}) exceeds the max_nodes ({max_nodes}) "
                    f"of compute pool '{compute_pool}'. Reduce min_instances or increase max_nodes."
                )

        if name:
            parsed_database, parsed_schema, parsed_name = identifier.parse_schema_level_object_identifier(name)
            database = parsed_database or database
            schema = parsed_schema or schema
            name = parsed_name
        else:
            name = payload_utils.get_payload_name(source, entrypoint)

        # The logical identifier for this job definition (used in the stage path)
        # is the resolved object name, not the fully qualified identifier.
        job_definition_id = name if not generate_suffix else name + _generate_suffix()
        stage_path_parts = identifier.parse_snowflake_stage_path(stage_name.lstrip("@"))
        stage_name = f"@{'.'.join(filter(None, stage_path_parts[:3]))}"
        stage_path = PurePosixPath(f"{stage_name}{stage_path_parts[-1].rstrip('/')}/{job_definition_id}")

        try:
            # Upload payload
            uploaded_payload = payload_utils.JobPayload(
                source, entrypoint=entrypoint, pip_requirements=pip_requirements, imports=imports
            ).upload(session, stage_path, overwrite)
        except SnowparkSQLException as e:
            if e.sql_error_code == 90106:
                raise RuntimeError(
                    "Please specify a schema, either in the session context or as a parameter in the job submission"
                )
            raise

        if runtime_environment is None and feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS.is_enabled(default=True):
            # Pass a JSON object for runtime versions so it serializes as nested JSON in options
            runtime_environment = json.dumps({"pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}"})

        combined_env_vars = {**uploaded_payload.env_vars, **(env_vars or {})}
        entrypoint_args = [v.as_posix() if isinstance(v, PurePath) else v for v in uploaded_payload.entrypoint]
        spec_options = types.SpecOptions(
            stage_path=stage_path.as_posix(),
            # the args will be set at runtime
            args=None,
            env_vars=combined_env_vars,
            enable_metrics=enable_metrics,
            spec_overrides=spec_overrides,
            runtime=runtime_environment if runtime_environment else None,
            enable_stage_mount_v2=feature_flags.FeatureFlags.ENABLE_STAGE_MOUNT_V2.is_enabled(default=True),
        )

        job_options = types.JobOptions(
            external_access_integrations=external_access_integrations,
            query_warehouse=query_warehouse,
            target_instances=target_instances,
            min_instances=min_instances,
            generate_suffix=generate_suffix,
        )

        return cls(
            stage_name=stage_path.as_posix(),
            spec_options=spec_options,
            job_options=job_options,
            compute_pool=compute_pool,
            entrypoint_args=entrypoint_args,
            session=session,
            database=database,
            schema=schema,
            name=name,
        )


def _generate_suffix() -> str:
    return str(uuid4().hex)[:8]
