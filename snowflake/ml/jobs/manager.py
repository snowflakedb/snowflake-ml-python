import logging
import pathlib
import textwrap
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload
from uuid import uuid4

import pandas as pd
import yaml

from snowflake import snowpark
from snowflake.connector import errors
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import job as jb
from snowflake.ml.jobs._utils import payload_utils, query_helper, spec_utils
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
    session = session or get_active_session()
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
    session = session or get_active_session()
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
    except errors.ProgrammingError as e:
        if "does not exist" in str(e):
            raise ValueError(f"Job does not exist: {job_id}") from e
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
    session._conn.run_query("DROP SERVICE IDENTIFIER(?)", params=(job.id,), _force_qmark_paramstyle=True)


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
        RuntimeError: If required Snowflake features are not enabled.
        ValueError: If database or schema value(s) are invalid
        errors.ProgrammingError: if the SQL query or its parameters are invalid
    """
    session = session or get_active_session()

    # Use kwargs for less common optional parameters
    database = kwargs.pop("database", None)
    schema = kwargs.pop("schema", None)
    min_instances = kwargs.pop("min_instances", target_instances)
    pip_requirements = kwargs.pop("pip_requirements", None)
    external_access_integrations = kwargs.pop("external_access_integrations", None)
    env_vars = kwargs.pop("env_vars", None)
    spec_overrides = kwargs.pop("spec_overrides", None)
    enable_metrics = kwargs.pop("enable_metrics", True)
    query_warehouse = kwargs.pop("query_warehouse", None)

    # Check for deprecated args
    if "num_instances" in kwargs:
        logger.warning(
            "'num_instances' is deprecated and will be removed in a future release. Use 'target_instances' instead."
        )
        target_instances = max(target_instances, kwargs.pop("num_instances"))

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
        requested_attributes = query_helper.get_attribute_map(session, {"max_nodes": 3})
        max_nodes = int(pool_info[requested_attributes["max_nodes"]])
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

    # Upload payload
    uploaded_payload = payload_utils.JobPayload(
        source,
        entrypoint=entrypoint,
        pip_requirements=pip_requirements,
    ).upload(session, stage_path)

    # Generate service spec
    spec = spec_utils.generate_service_spec(
        session,
        compute_pool=compute_pool,
        payload=uploaded_payload,
        args=args,
        target_instances=target_instances,
        min_instances=min_instances,
        enable_metrics=enable_metrics,
    )
    spec_overrides = spec_utils.generate_spec_overrides(
        environment_vars=env_vars,
        custom_overrides=spec_overrides,
    )
    if spec_overrides:
        spec = spec_utils.merge_patch(spec, spec_overrides, display_name="spec_overrides")

    # Generate SQL command for job submission
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
    query_warehouse = query_warehouse or session.get_current_warehouse()
    if query_warehouse:
        query.append("QUERY_WAREHOUSE = IDENTIFIER(?)")
        params.append(query_warehouse)
    if target_instances > 1:
        query.append("REPLICAS = ?")
        params.append(target_instances)

    # Submit job
    query_text = "\n".join(line for line in query if line)

    try:
        _ = session._conn.run_query(query_text, params=params, _force_qmark_paramstyle=True)
    except errors.ProgrammingError as e:
        if "invalid property 'ASYNC'" in str(e):
            raise RuntimeError(
                "SPCS Async Jobs not enabled. Set parameter `ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE` to enable."
            ) from e
        raise

    return get_job(job_id, session=session)
