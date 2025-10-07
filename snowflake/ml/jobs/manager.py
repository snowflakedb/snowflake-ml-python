import json
import logging
import pathlib
import sys
import textwrap
from pathlib import PurePath
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload
from uuid import uuid4

import pandas as pd
import yaml

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import job as jb
from snowflake.ml.jobs._utils import (
    feature_flags,
    payload_utils,
    query_helper,
    spec_utils,
    types,
)
from snowflake.snowpark._internal import utils as sp_utils
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import coalesce, col, lit, when

logger = logging.getLogger(__name__)

_PROJECT = "MLJob"
JOB_ID_PREFIX = "MLJOB_"

T = TypeVar("T")


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["limit", "scope"])
def list_jobs(
    limit: int = 10,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame with the list of jobs in the current session.

    Args:
        limit: The maximum number of jobs to return. Non-positive values are treated as no limit.
        database: The database to use. If not specified, uses the current database.
        schema: The schema to use. If not specified, uses the current schema.
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        A DataFrame with the list of jobs.

    Raises:
        SnowparkSQLException: if there is an error retrieving the job history.

    Examples:
        >>> from snowflake.ml.jobs import list_jobs
        >>> list_jobs(limit=5)
    """

    session = _ensure_session(session)
    try:
        df = _get_job_history_spcs(
            session,
            limit=limit,
            database=database,
            schema=schema,
        )
        return df.to_pandas()
    except SnowparkSQLException as spcs_error:
        if spcs_error.sql_error_code == 2143:
            logger.debug("Job history is not enabled. Please enable it to use this feature.")
            df = _get_job_services(session, limit=limit, database=database, schema=schema)
            return df.to_pandas()
        raise


def _get_job_services(
    session: snowpark.Session, limit: int = 10, database: Optional[str] = None, schema: Optional[str] = None
) -> snowpark.DataFrame:
    query = "SHOW JOB SERVICES"
    query += f" LIKE '{JOB_ID_PREFIX}%'"
    database = database or session.get_current_database()
    schema = schema or session.get_current_schema()
    if database is None and schema is None:
        query += "IN account"
    elif not schema:
        query += f" IN DATABASE {database}"
    else:
        query += f" IN {database}.{schema}"
    if limit > 0:
        query += f" LIMIT {limit}"
    df = session.sql(query)
    df = df.select(
        df['"name"'],
        df['"status"'],
        lit(None).alias('"message"'),
        df['"database_name"'],
        df['"schema_name"'],
        df['"owner"'],
        df['"compute_pool"'],
        df['"target_instances"'],
        df['"created_on"'].alias('"created_time"'),
        when(col('"status"').isin(jb.TERMINAL_JOB_STATUSES), col('"updated_on"'))
        .otherwise(lit(None))
        .alias('"completed_time"'),
    ).order_by('"created_time"', ascending=False)
    return df


def _get_job_history_spcs(
    session: snowpark.Session,
    limit: int = 10,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    include_deleted: bool = False,
    created_time_start: Optional[str] = None,
    created_time_end: Optional[str] = None,
) -> snowpark.DataFrame:
    query = ["select * from table(snowflake.spcs.get_job_history("]
    query_params = []
    if created_time_start:
        query_params.append(f"created_time_start => TO_TIMESTAMP_LTZ('{created_time_start}')")
    if created_time_end:
        query_params.append(f"created_time_end => TO_TIMESTAMP_LTZ('{created_time_end}')")
    query.append(",".join(query_params))
    query.append("))")
    condition = []
    database = database or session.get_current_database()
    schema = schema or session.get_current_schema()

    # format database and schema identifiers
    if database:
        condition.append(f"DATABASE_NAME = '{identifier.resolve_identifier(database)}'")

    if schema:
        condition.append(f"SCHEMA_NAME = '{identifier.resolve_identifier(schema)}'")

    if not include_deleted:
        condition.append("DELETED_TIME IS NULL")

    if len(condition) > 0:
        query.append("WHERE " + " AND ".join(condition))
    if limit > 0:
        query.append(f"LIMIT {limit}")
    df = session.sql("\n".join(query))
    df = df.select(
        df["NAME"].alias('"name"'),
        df["STATUS"].alias('"status"'),
        df["MESSAGE"].alias('"message"'),
        df["DATABASE_NAME"].alias('"database_name"'),
        df["SCHEMA_NAME"].alias('"schema_name"'),
        df["OWNER"].alias('"owner"'),
        df["COMPUTE_POOL_NAME"].alias('"compute_pool"'),
        coalesce(df["PARAMETERS"]["REPLICAS"], lit(1)).alias('"target_instances"'),
        df["CREATED_TIME"].alias('"created_time"'),
        df["COMPLETED_TIME"].alias('"completed_time"'),
    )
    return df


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def get_job(job_id: str, session: Optional[snowpark.Session] = None) -> jb.MLJob[Any]:
    """Retrieve a job service from the backend."""
    session = _ensure_session(session)
    try:
        database, schema, job_name = identifier.parse_schema_level_object_identifier(job_id)
        database = identifier.resolve_identifier(cast(str, database or session.get_current_database()))
        schema = identifier.resolve_identifier(cast(str, schema or session.get_current_schema()))
    except ValueError as e:
        raise ValueError(f"Invalid job ID: {job_id}") from e

    job_id = f"{database}.{schema}.{job_name}"
    try:
        # Validate that job exists by doing a spec lookup
        job = jb.MLJob[Any](job_id, session=session)
        _ = job._service_spec
        return job
    except SnowparkSQLException as e:
        if e.sql_error_code == 2003:
            job = jb.MLJob[Any](job_id, session=session)
            _ = job.status
            return job
        raise


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def delete_job(job: Union[str, jb.MLJob[Any]], session: Optional[snowpark.Session] = None) -> None:
    """Delete a job service from the backend. Status and logs will be lost."""
    job = job if isinstance(job, jb.MLJob) else get_job(job, session=session)
    session = job._session
    try:
        stage_path = job._stage_path
        session.sql(f"REMOVE {stage_path}/").collect()
        logger.info(f"Successfully cleaned up stage files for job {job.id} at {stage_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up stage files for job {job.id}: {e}")
    query_helper.run_query(session, "DROP SERVICE IDENTIFIER(?)", params=(job.id,))


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def submit_file(
    file_path: str,
    compute_pool: str,
    *,
    stage_name: str,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> jb.MLJob[None]:
    """
    Submit a Python file as a job to the compute pool.

    Args:
        file_path: The path to the file containing the source code for the job.
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        args: A list of arguments to pass to the job.
        target_instances: The number of nodes in the job. If none specified, create a single node job.
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        session: The Snowpark session to use. If none specified, uses active session.
        kwargs: Additional keyword arguments. Supported arguments:
            database (str): The database to use for the job.
            schema (str): The schema to use for the job.
            min_instances (int): The minimum number of nodes required to start the job.
                If none specified, defaults to target_instances. If set, the job
                will not start until the minimum number of nodes is available.
            env_vars (dict): Environment variables to set in container.
            enable_metrics (bool): Whether to enable metrics publishing for the job.
            query_warehouse (str): The query warehouse to use. Defaults to session warehouse.
            spec_overrides (dict): A dictionary of overrides for the service spec.
            imports (list[Union[tuple[str, str], tuple[str]]]): A list of additional payloads used in the job.

    Returns:
        An object representing the submitted job.
    """
    return _submit_job(
        source=file_path,
        args=args,
        compute_pool=compute_pool,
        stage_name=stage_name,
        target_instances=target_instances,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        session=session,
        **kwargs,
    )


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def submit_directory(
    dir_path: str,
    compute_pool: str,
    *,
    entrypoint: str,
    stage_name: str,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> jb.MLJob[None]:
    """
    Submit a directory containing Python script(s) as a job to the compute pool.

    Args:
        dir_path: The path to the directory containing the job payload.
        compute_pool: The compute pool to use for the job.
        entrypoint: The relative path to the entry point script inside the source directory.
        stage_name: The name of the stage where the job payload will be uploaded.
        args: A list of arguments to pass to the job.
        target_instances: The number of nodes in the job. If none specified, create a single node job.
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        session: The Snowpark session to use. If none specified, uses active session.
        kwargs: Additional keyword arguments. Supported arguments:
            database (str): The database to use for the job.
            schema (str): The schema to use for the job.
            min_instances (int): The minimum number of nodes required to start the job.
                If none specified, defaults to target_instances. If set, the job
                will not start until the minimum number of nodes is available.
            env_vars (dict): Environment variables to set in container.
            enable_metrics (bool): Whether to enable metrics publishing for the job.
            query_warehouse (str): The query warehouse to use. Defaults to session warehouse.
            spec_overrides (dict): A dictionary of overrides for the service spec.
            imports (list[Union[tuple[str, str], tuple[str]]]): A list of additional payloads used in the job.

    Returns:
        An object representing the submitted job.
    """
    return _submit_job(
        source=dir_path,
        entrypoint=entrypoint,
        args=args,
        compute_pool=compute_pool,
        stage_name=stage_name,
        target_instances=target_instances,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        session=session,
        **kwargs,
    )


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def submit_from_stage(
    source: str,
    compute_pool: str,
    *,
    entrypoint: str,
    stage_name: str,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> jb.MLJob[None]:
    """
    Submit a directory containing Python script(s) as a job to the compute pool.

    Args:
        source: a stage path or a stage containing the job payload.
        compute_pool: The compute pool to use for the job.
        entrypoint: a stage path containing the entry point script inside the source directory.
        stage_name: The name of the stage where the job payload will be uploaded.
        args: A list of arguments to pass to the job.
        target_instances: The number of nodes in the job. If none specified, create a single node job.
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        session: The Snowpark session to use. If none specified, uses active session.
        kwargs: Additional keyword arguments. Supported arguments:
            database (str): The database to use for the job.
            schema (str): The schema to use for the job.
            min_instances (int): The minimum number of nodes required to start the job.
                If none specified, defaults to target_instances. If set, the job
                will not start until the minimum number of nodes is available.
            env_vars (dict): Environment variables to set in container.
            enable_metrics (bool): Whether to enable metrics publishing for the job.
            query_warehouse (str): The query warehouse to use. Defaults to session warehouse.
            spec_overrides (dict): A dictionary of overrides for the service spec.
            imports (list[Union[tuple[str, str], tuple[str]]]): A list of additional payloads used in the job.
            runtime_environment (str): The runtime image to use. Only support image tag or full image URL,
                e.g. "1.7.1" or "image_repo/image_name:image_tag". When it refers to a full image URL,
                it should contain image repository, image name and image tag.

    Returns:
        An object representing the submitted job.
    """
    return _submit_job(
        source=source,
        entrypoint=entrypoint,
        args=args,
        compute_pool=compute_pool,
        stage_name=stage_name,
        target_instances=target_instances,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        session=session,
        **kwargs,
    )


@overload
def _submit_job(
    source: str,
    compute_pool: str,
    *,
    stage_name: str,
    entrypoint: Optional[str] = None,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> jb.MLJob[None]:
    ...


@overload
def _submit_job(
    source: Callable[..., T],
    compute_pool: str,
    *,
    stage_name: str,
    entrypoint: Optional[str] = None,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> jb.MLJob[T]:
    ...


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    func_params_to_log=[
        # TODO: Log the source type (callable, file, directory, etc)
        # TODO: Log instance type of compute pool used
        # TODO: Log lengths of args, env_vars, and spec_overrides values
        "pip_requirements",
        "external_access_integrations",
        "num_instances",  # deprecated
        "target_instances",
        "min_instances",
        "enable_metrics",
        "query_warehouse",
        "runtime_environment",
    ],
)
def _submit_job(
    source: Union[str, Callable[..., T]],
    compute_pool: str,
    *,
    stage_name: str,
    entrypoint: Optional[str] = None,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> jb.MLJob[T]:
    """
    Submit a job to the compute pool.

    Args:
        source: The file/directory path containing payload source code or a serializable Python callable.
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        entrypoint: The entry point for the job execution. Required if source is a directory.
        args: A list of arguments to pass to the job.
        target_instances: The number of instances to use for the job. If none specified, single node job is created.
        session: The Snowpark session to use. If none specified, uses active session.
        kwargs: Additional keyword arguments.

    Returns:
        An object representing the submitted job.

    Raises:
        ValueError: If database or schema value(s) are invalid
        RuntimeError: If schema is not specified in session context or job submission
        SnowparkSQLException: if failed to upload payload
    """
    session = _ensure_session(session)

    # Check for deprecated args
    if "num_instances" in kwargs:
        logger.warning(
            "'num_instances' is deprecated and will be removed in a future release. Use 'target_instances' instead."
        )
        target_instances = max(target_instances, kwargs.pop("num_instances"))

    imports = None
    if "additional_payloads" in kwargs:
        logger.warning(
            "'additional_payloads' is deprecated and will be removed in a future release. Use 'imports' instead."
        )
        imports = kwargs.pop("additional_payloads")

    if "runtime_environment" in kwargs:
        logger.warning("'runtime_environment' is in private preview since 1.15.0, do not use it in production.")

    # Use kwargs for less common optional parameters
    database = kwargs.pop("database", None)
    schema = kwargs.pop("schema", None)
    min_instances = kwargs.pop("min_instances", target_instances)
    pip_requirements = kwargs.pop("pip_requirements", None)
    external_access_integrations = kwargs.pop("external_access_integrations", None)
    env_vars = kwargs.pop("env_vars", None)
    spec_overrides = kwargs.pop("spec_overrides", None)
    enable_metrics = kwargs.pop("enable_metrics", True)
    query_warehouse = kwargs.pop("query_warehouse", session.get_current_warehouse())
    imports = kwargs.pop("imports", None) or imports
    runtime_environment = kwargs.pop("runtime_environment", None)

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

    job_name = f"{JOB_ID_PREFIX}{str(uuid4()).replace('-', '_').upper()}"
    job_id = identifier.get_schema_level_object_identifier(database, schema, job_name)
    stage_path_parts = identifier.parse_snowflake_stage_path(stage_name.lstrip("@"))
    stage_name = f"@{'.'.join(filter(None, stage_path_parts[:3]))}"
    stage_path = pathlib.PurePosixPath(f"{stage_name}{stage_path_parts[-1].rstrip('/')}/{job_name}")

    try:
        # Upload payload
        uploaded_payload = payload_utils.JobPayload(
            source, entrypoint=entrypoint, pip_requirements=pip_requirements, additional_payloads=imports
        ).upload(session, stage_path)
    except SnowparkSQLException as e:
        if e.sql_error_code == 90106:
            raise RuntimeError(
                "Please specify a schema, either in the session context or as a parameter in the job submission"
            )
        raise

    if feature_flags.FeatureFlags.USE_SUBMIT_JOB_V2.is_enabled(default=True):
        # Add default env vars (extracted from spec_utils.generate_service_spec)
        combined_env_vars = {**uploaded_payload.env_vars, **(env_vars or {})}

        try:
            return _do_submit_job_v2(
                session=session,
                payload=uploaded_payload,
                args=args,
                env_vars=combined_env_vars,
                spec_overrides=spec_overrides,
                compute_pool=compute_pool,
                job_id=job_id,
                external_access_integrations=external_access_integrations,
                query_warehouse=query_warehouse,
                target_instances=target_instances,
                min_instances=min_instances,
                enable_metrics=enable_metrics,
                use_async=True,
                runtime_environment=runtime_environment,
            )
        except SnowparkSQLException as e:
            if not (e.sql_error_code == 90237 and sp_utils.is_in_stored_procedure()):  # type: ignore[no-untyped-call]
                raise
            # SNOW-2390287: SYSTEM$EXECUTE_ML_JOB() is erroneously blocked in owner's rights
            # stored procedures. This will be fixed in an upcoming release.
            logger.warning(
                "Job submission using V2 failed with error {}. Falling back to V1.".format(
                    str(e).split("\n", 1)[0],
                )
            )

    # Fall back to v1
    # Generate service spec
    spec = spec_utils.generate_service_spec(
        session,
        compute_pool=compute_pool,
        payload=uploaded_payload,
        args=args,
        target_instances=target_instances,
        min_instances=min_instances,
        enable_metrics=enable_metrics,
        runtime_environment=runtime_environment,
    )

    # Generate spec overrides
    spec_overrides = spec_utils.generate_spec_overrides(
        environment_vars=env_vars,
        custom_overrides=spec_overrides,
    )
    if spec_overrides:
        spec = spec_utils.merge_patch(spec, spec_overrides, display_name="spec_overrides")

    return _do_submit_job_v1(
        session, spec, external_access_integrations, query_warehouse, target_instances, compute_pool, job_id
    )


def _do_submit_job_v1(
    session: snowpark.Session,
    spec: dict[str, Any],
    external_access_integrations: list[str],
    query_warehouse: Optional[str],
    target_instances: int,
    compute_pool: str,
    job_id: str,
) -> jb.MLJob[Any]:
    """
    Generate the SQL query for job submission.

    Args:
        session: The Snowpark session to use.
        spec: The service spec for the job.
        external_access_integrations: The external access integrations for the job.
        query_warehouse: The query warehouse for the job.
        target_instances: The number of instances for the job.
        session: The Snowpark session to use.
        compute_pool: The compute pool to use for the job.
        job_id: The ID of the job.

    Returns:
        The job object.
    """
    query_template = textwrap.dedent(
        """\
        EXECUTE JOB SERVICE
        IN COMPUTE POOL IDENTIFIER(?)
        FROM SPECIFICATION $$
        {}
        $$
        NAME = IDENTIFIER(?)
        ASYNC = TRUE
        """
    )
    params: list[Any] = [compute_pool, job_id]
    query = query_template.format(yaml.dump(spec)).splitlines()
    if external_access_integrations:
        external_access_integration_list = ",".join(f"{e}" for e in external_access_integrations)
        query.append(f"EXTERNAL_ACCESS_INTEGRATIONS = ({external_access_integration_list})")
    if query_warehouse:
        query.append("QUERY_WAREHOUSE = IDENTIFIER(?)")
        params.append(query_warehouse)
    if target_instances > 1:
        query.append("REPLICAS = ?")
        params.append(target_instances)

    query_text = "\n".join(line for line in query if line)
    _ = query_helper.run_query(session, query_text, params=params)

    return get_job(job_id, session=session)


def _do_submit_job_v2(
    session: snowpark.Session,
    payload: types.UploadedPayload,
    args: Optional[list[str]],
    env_vars: dict[str, str],
    spec_overrides: dict[str, Any],
    compute_pool: str,
    job_id: Optional[str] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    target_instances: int = 1,
    min_instances: int = 1,
    enable_metrics: bool = True,
    use_async: bool = True,
    runtime_environment: Optional[str] = None,
) -> jb.MLJob[Any]:
    """
    Generate the SQL query for job submission.

    Args:
        session: The Snowpark session to use.
        payload: The uploaded job payload.
        args: Arguments to pass to the entrypoint script.
        env_vars: Environment variables to set in the job container.
        spec_overrides: Custom service specification overrides.
        compute_pool: The compute pool to use for job execution.
        job_id: The ID of the job.
        external_access_integrations: Optional list of external access integrations.
        query_warehouse: Optional query warehouse to use.
        target_instances: Number of instances for multi-node job.
        min_instances: Minimum number of instances required to start the job.
        enable_metrics: Whether to enable platform metrics for the job.
        use_async: Whether to run the job asynchronously.
        runtime_environment: image tag or full image URL to use for the job.

    Returns:
        The job object.
    """
    args = [
        (payload.stage_path.joinpath(v).as_posix() if isinstance(v, PurePath) else v) for v in payload.entrypoint
    ] + (args or [])
    spec_options = {
        "STAGE_PATH": payload.stage_path.as_posix(),
        "ENTRYPOINT": ["/usr/local/bin/_entrypoint.sh"],
        "ARGS": args,
        "ENV_VARS": env_vars,
        "ENABLE_METRICS": enable_metrics,
        "SPEC_OVERRIDES": spec_overrides,
    }
    # for the image tag or full image URL, we use that directly
    if runtime_environment:
        spec_options["RUNTIME"] = runtime_environment
    elif feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS.is_enabled():
        # when feature flag is enabled, we get the local python version and wrap it in a dict
        # in system function, we can know whether it is python version or image tag or full image URL through the format
        spec_options["RUNTIME"] = json.dumps({"pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}"})
    job_options = {
        "EXTERNAL_ACCESS_INTEGRATIONS": external_access_integrations,
        "QUERY_WAREHOUSE": query_warehouse,
        "TARGET_INSTANCES": target_instances,
        "MIN_INSTANCES": min_instances,
        "ASYNC": use_async,
    }
    job_options = {k: v for k, v in job_options.items() if v is not None}

    query_template = "CALL SYSTEM$EXECUTE_ML_JOB(?, ?, ?, ?)"
    params = [job_id, compute_pool, json.dumps(spec_options), json.dumps(job_options)]
    actual_job_id = query_helper.run_query(session, query_template, params=params)[0][0]

    return get_job(actual_job_id, session=session)


def _ensure_session(session: Optional[snowpark.Session]) -> snowpark.Session:
    try:
        session = session or get_active_session()
    except snowpark.exceptions.SnowparkSessionException as e:
        if "More than one active session" in e.message:
            raise RuntimeError(
                "More than one active session is found. Please specify the session explicitly as a parameter"
            ) from None
        if "No default Session is found" in e.message:
            raise RuntimeError("No active session is found. Please create a session") from None
        raise
    return session
