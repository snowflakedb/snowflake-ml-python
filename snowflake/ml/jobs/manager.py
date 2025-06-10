import logging
import pathlib
import textwrap
from typing import Any, Callable, Literal, Optional, TypeVar, Union, cast, overload
from uuid import uuid4

import yaml

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import job as jb
from snowflake.ml.jobs._utils import payload_utils, spec_utils
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

logger = logging.getLogger(__name__)

_PROJECT = "MLJob"
JOB_ID_PREFIX = "MLJOB_"

T = TypeVar("T")


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["limit", "scope"])
def list_jobs(
    limit: int = 10,
    scope: Union[Literal["account", "database", "schema"], str, None] = None,
    session: Optional[snowpark.Session] = None,
) -> snowpark.DataFrame:
    """
    Returns a Snowpark DataFrame with the list of jobs in the current session.

    Args:
        limit: The maximum number of jobs to return. Non-positive values are treated as no limit.
        scope: The scope to list jobs from, such as "schema" or "compute pool <pool_name>".
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        A DataFrame with the list of jobs.

    Examples:
        >>> from snowflake.ml.jobs import list_jobs
        >>> list_jobs(limit=5).show()
    """
    session = session or get_active_session()
    query = "SHOW JOB SERVICES"
    query += f" LIKE '{JOB_ID_PREFIX}%'"
    if scope:
        query += f" IN {scope}"
    if limit > 0:
        query += f" LIMIT {limit}"
    df = session.sql(query)
    df = df.select(
        df['"name"'],
        df['"owner"'],
        df['"status"'],
        df['"created_on"'],
        df['"compute_pool"'],
    ).order_by('"created_on"', ascending=False)
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
        # Validate that job exists by doing a status check
        job = jb.MLJob[Any](job_id, session=session)
        _ = job.status
        return job
    except SnowparkSQLException as e:
        if "does not exist" in e.message:
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
    session.sql("DROP SERVICE IDENTIFIER(?)", params=(job.id,)).collect()


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def submit_file(
    file_path: str,
    compute_pool: str,
    *,
    stage_name: str,
    args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    spec_overrides: Optional[dict[str, Any]] = None,
    target_instances: int = 1,
    min_instances: Optional[int] = None,
    enable_metrics: bool = False,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
) -> jb.MLJob[None]:
    """
    Submit a Python file as a job to the compute pool.

    Args:
        file_path: The path to the file containing the source code for the job.
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        args: A list of arguments to pass to the job.
        env_vars: Environment variables to set in container
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        query_warehouse: The query warehouse to use. Defaults to session warehouse.
        spec_overrides: Custom service specification overrides to apply.
        target_instances: The number of instances to use for the job. If none specified, single node job is created.
        min_instances: The minimum number of nodes required to start the job. If none specified,
            defaults to target_instances. If set, the job will not start until the minimum number of nodes is available.
        enable_metrics: Whether to enable metrics publishing for the job.
        database: The database to use.
        schema: The schema to use.
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        An object representing the submitted job.
    """
    return _submit_job(
        source=file_path,
        args=args,
        compute_pool=compute_pool,
        stage_name=stage_name,
        env_vars=env_vars,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        query_warehouse=query_warehouse,
        spec_overrides=spec_overrides,
        target_instances=target_instances,
        min_instances=min_instances,
        enable_metrics=enable_metrics,
        database=database,
        schema=schema,
        session=session,
    )


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def submit_directory(
    dir_path: str,
    compute_pool: str,
    *,
    entrypoint: str,
    stage_name: str,
    args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    spec_overrides: Optional[dict[str, Any]] = None,
    target_instances: int = 1,
    min_instances: Optional[int] = None,
    enable_metrics: bool = False,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
) -> jb.MLJob[None]:
    """
    Submit a directory containing Python script(s) as a job to the compute pool.

    Args:
        dir_path: The path to the directory containing the job payload.
        compute_pool: The compute pool to use for the job.
        entrypoint: The relative path to the entry point script inside the source directory.
        stage_name: The name of the stage where the job payload will be uploaded.
        args: A list of arguments to pass to the job.
        env_vars: Environment variables to set in container
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        query_warehouse: The query warehouse to use. Defaults to session warehouse.
        spec_overrides: Custom service specification overrides to apply.
        target_instances: The number of instances to use for the job. If none specified, single node job is created.
        min_instances: The minimum number of nodes required to start the job. If none specified,
            defaults to target_instances. If set, the job will not start until the minimum number of nodes is available.
        enable_metrics: Whether to enable metrics publishing for the job.
        database: The database to use.
        schema: The schema to use.
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        An object representing the submitted job.
    """
    return _submit_job(
        source=dir_path,
        entrypoint=entrypoint,
        args=args,
        compute_pool=compute_pool,
        stage_name=stage_name,
        env_vars=env_vars,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        query_warehouse=query_warehouse,
        spec_overrides=spec_overrides,
        target_instances=target_instances,
        min_instances=min_instances,
        enable_metrics=enable_metrics,
        database=database,
        schema=schema,
        session=session,
    )


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def submit_from_stage(
    source: str,
    compute_pool: str,
    *,
    entrypoint: str,
    stage_name: str,
    args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    spec_overrides: Optional[dict[str, Any]] = None,
    target_instances: int = 1,
    min_instances: Optional[int] = None,
    enable_metrics: bool = False,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
) -> jb.MLJob[None]:
    """
    Submit a directory containing Python script(s) as a job to the compute pool.

    Args:
        source: a stage path or a stage containing the job payload.
        compute_pool: The compute pool to use for the job.
        entrypoint: a stage path containing the entry point script inside the source directory.
        stage_name: The name of the stage where the job payload will be uploaded.
        args: A list of arguments to pass to the job.
        env_vars: Environment variables to set in container
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        query_warehouse: The query warehouse to use. Defaults to session warehouse.
        spec_overrides: Custom service specification overrides to apply.
        target_instances: The number of instances to use for the job. If none specified, single node job is created.
        min_instances: The minimum number of nodes required to start the job. If none specified,
            defaults to target_instances. If set, the job will not start until the minimum number of nodes is available.
        enable_metrics: Whether to enable metrics publishing for the job.
        database: The database to use.
        schema: The schema to use.
        session: The Snowpark session to use. If none specified, uses active session.


    Returns:
        An object representing the submitted job.
    """
    return _submit_job(
        source=source,
        entrypoint=entrypoint,
        args=args,
        compute_pool=compute_pool,
        stage_name=stage_name,
        env_vars=env_vars,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        query_warehouse=query_warehouse,
        spec_overrides=spec_overrides,
        target_instances=target_instances,
        min_instances=min_instances,
        enable_metrics=enable_metrics,
        database=database,
        schema=schema,
        session=session,
    )


@overload
def _submit_job(
    source: str,
    compute_pool: str,
    *,
    stage_name: str,
    entrypoint: Optional[str] = None,
    args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    spec_overrides: Optional[dict[str, Any]] = None,
    target_instances: int = 1,
    min_instances: Optional[int] = None,
    enable_metrics: bool = False,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
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
    env_vars: Optional[dict[str, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    spec_overrides: Optional[dict[str, Any]] = None,
    target_instances: int = 1,
    min_instances: Optional[int] = None,
    enable_metrics: bool = False,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
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
        "target_instances",
        "enable_metrics",
    ],
)
def _submit_job(
    source: Union[str, Callable[..., T]],
    compute_pool: str,
    *,
    stage_name: str,
    entrypoint: Optional[str] = None,
    args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    spec_overrides: Optional[dict[str, Any]] = None,
    target_instances: int = 1,
    min_instances: Optional[int] = None,
    enable_metrics: bool = False,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    session: Optional[snowpark.Session] = None,
) -> jb.MLJob[T]:
    """
    Submit a job to the compute pool.

    Args:
        source: The file/directory path containing payload source code or a serializable Python callable.
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        entrypoint: The entry point for the job execution. Required if source is a directory.
        args: A list of arguments to pass to the job.
        env_vars: Environment variables to set in container
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        query_warehouse: The query warehouse to use. Defaults to session warehouse.
        spec_overrides: Custom service specification overrides to apply.
        target_instances: The number of instances to use for the job. If none specified, single node job is created.
        min_instances: The minimum number of nodes required to start the job. If none specified,
            defaults to target_instances. If set, the job will not start until the minimum number of nodes is available.
        enable_metrics: Whether to enable metrics publishing for the job.
        database: The database to use.
        schema: The schema to use.
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        An object representing the submitted job.

    Raises:
        RuntimeError: If required Snowflake features are not enabled.
        ValueError: If database or schema value(s) are invalid
    """
    if database and not schema:
        raise ValueError("Schema must be specified if database is specified.")
    if target_instances < 1:
        raise ValueError("target_instances must be greater than 0.")

    min_instances = target_instances if min_instances is None else min_instances
    if not (0 < min_instances <= target_instances):
        raise ValueError("min_instances must be greater than 0 and less than or equal to target_instances.")

    session = session or get_active_session()

    if min_instances > 1:
        # Validate min_instances against compute pool max_nodes
        pool_info = jb._get_compute_pool_info(session, compute_pool)
        max_nodes = int(pool_info["max_nodes"])
        if min_instances > max_nodes:
            raise ValueError(
                f"The requested min_instances ({min_instances}) exceeds the max_nodes ({max_nodes}) "
                f"of compute pool '{compute_pool}'. Reduce min_instances or increase max_nodes."
            )

    # Validate database and schema identifiers on client side since
    # SQL parser for EXECUTE JOB SERVICE seems to struggle with this
    database = identifier.resolve_identifier(cast(str, database or session.get_current_database()))
    schema = identifier.resolve_identifier(cast(str, schema or session.get_current_schema()))

    job_name = f"{JOB_ID_PREFIX}{str(uuid4()).replace('-', '_').upper()}"
    job_id = f"{database}.{schema}.{job_name}"
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
        _ = session.sql(query_text, params=params).collect()
    except SnowparkSQLException as e:
        if "invalid property 'ASYNC'" in e.message:
            raise RuntimeError(
                "SPCS Async Jobs not enabled. Set parameter `ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE` to enable."
            ) from e
        raise

    return jb.MLJob(job_id, service_spec=spec, session=session)
