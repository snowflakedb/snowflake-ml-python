import json
import logging
import os
import time
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeVar, Union, cast, overload

import yaml

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.mixins import SerializableSessionMixin
from snowflake.ml.jobs._interop import results as interop_result, utils as interop_utils
from snowflake.ml.jobs._utils import (
    constants,
    payload_utils,
    query_helper,
    stage_utils,
    types,
)
from snowflake.snowpark import Row, context as sp_context
from snowflake.snowpark.exceptions import SnowparkSQLException

_PROJECT = "MLJob"
TERMINAL_JOB_STATUSES = {"FAILED", "DONE", "CANCELLED", "INTERNAL_ERROR", "DELETED"}

T = TypeVar("T")

logger = logging.getLogger(__name__)


class MLJob(Generic[T], SerializableSessionMixin):
    def __init__(
        self,
        id: str,
        service_spec: Optional[dict[str, Any]] = None,
        session: Optional[snowpark.Session] = None,
    ) -> None:
        self._id = id
        self._service_spec_cached: Optional[dict[str, Any]] = service_spec
        self._session = session or sp_context.get_active_session()

        self._status: types.JOB_STATUS = "PENDING"
        self._result: Optional[interop_result.ExecutionResult] = None

    @cached_property
    def _service_info(self) -> types.ServiceInfo:
        """Get the job's service info."""
        return _resolve_service_info(self.id, self._session)

    @cached_property
    def name(self) -> str:
        return identifier.parse_schema_level_object_identifier(self.id)[-1]

    @cached_property
    def target_instances(self) -> int:
        return self._service_info.target_instances

    @cached_property
    def min_instances(self) -> int:
        try:
            return int(self._container_spec["env"].get(constants.MIN_INSTANCES_ENV_VAR, 1))
        except (TypeError, ValueError):
            return 1

    @property
    def id(self) -> str:
        """Get the unique job ID"""
        return self._id

    @property
    def status(self) -> types.JOB_STATUS:
        """Get the job's execution status."""
        if self._status not in TERMINAL_JOB_STATUSES:
            # Query backend for job status if not in terminal state
            self._status = _get_status(self._session, self.id)
        return self._status

    @cached_property
    def _compute_pool(self) -> str:
        """Get the job's compute pool name."""
        return self._service_info.compute_pool

    @property
    def _service_spec(self) -> dict[str, Any]:
        """Get the job's service spec."""
        if not self._service_spec_cached:
            self._service_spec_cached = _get_service_spec(self._session, self.id)
        return self._service_spec_cached

    @property
    def _container_spec(self) -> dict[str, Any]:
        """Get the job's main container spec."""
        try:
            containers = self._service_spec["spec"]["containers"]
        except SnowparkSQLException as e:
            if e.sql_error_code == 2003:
                # If the job is deleted, the service spec is not available
                return {}
            raise
        if len(containers) == 1:
            return cast(dict[str, Any], containers[0])
        try:
            container_spec = next(c for c in containers if c["name"] == constants.DEFAULT_CONTAINER_NAME)
        except StopIteration:
            raise ValueError(f"Container '{constants.DEFAULT_CONTAINER_NAME}' not found in job {self.name}")
        return cast(dict[str, Any], container_spec)

    @property
    def _stage_path(self) -> str:
        """Get the job's artifact storage stage location."""
        volumes = self._service_spec["spec"]["volumes"]
        stage_path = next(v for v in volumes if v["name"] == constants.STAGE_VOLUME_NAME)["source"]
        return cast(str, stage_path)

    @property
    def _result_path(self) -> str:
        """Get the job's result file location."""
        result_path_str = self._container_spec["env"].get(constants.RESULT_PATH_ENV_VAR)
        if result_path_str is None:
            raise RuntimeError(f"Job {self.name} doesn't have a result path configured")

        return self._transform_path(result_path_str)

    def _transform_path(self, path_str: str) -> str:
        """Transform a local path within the container to a stage path."""
        path = payload_utils.resolve_path(path_str)
        if isinstance(path, stage_utils.StagePath):
            # Stage paths need no transformation
            return path.as_posix()
        if not path.is_absolute():
            # Assume relative paths are relative to stage mount path
            return f"{self._stage_path}/{path.as_posix()}"

        # If result path is absolute, rebase it onto the stage mount path
        # TODO: Rather than matching by name, use the longest mount path which matches
        volume_mounts = self._container_spec["volumeMounts"]
        stage_mount_str = next(v for v in volume_mounts if v.get("name") == constants.STAGE_VOLUME_NAME)["mountPath"]
        stage_mount = Path(stage_mount_str)
        try:
            relative_path = path.relative_to(stage_mount)
            return f"{self._stage_path}/{relative_path.as_posix()}"
        except ValueError:
            raise ValueError(f"Result path {path} is absolute, but should be relative to stage mount {stage_mount}")

    @overload
    def get_logs(
        self,
        limit: int = -1,
        instance_id: Optional[int] = None,
        *,
        as_list: Literal[True],
        verbose: bool = constants.DEFAULT_VERBOSE_LOG,
    ) -> list[str]:
        ...

    @overload
    def get_logs(
        self,
        limit: int = -1,
        instance_id: Optional[int] = None,
        *,
        as_list: Literal[False] = False,
        verbose: bool = constants.DEFAULT_VERBOSE_LOG,
    ) -> str:
        ...

    def get_logs(
        self,
        limit: int = -1,
        instance_id: Optional[int] = None,
        *,
        as_list: bool = False,
        verbose: bool = constants.DEFAULT_VERBOSE_LOG,
    ) -> Union[str, list[str]]:
        """
        Return the job's execution logs.

        Args:
            limit: The maximum number of lines to return. Negative values are treated as no limit.
            instance_id: Optional instance ID to get logs from a specific instance.
                         If not provided, returns logs from the head node.
            as_list: If True, returns logs as a list of lines. Otherwise, returns logs as a single string.
            verbose: Whether to return the full log or just the user log.

        Returns:
            The job's execution logs.
        """
        logs = _get_logs(
            self._session,
            self.id,
            limit,
            instance_id,
            self._container_spec["name"] if "name" in self._container_spec else constants.DEFAULT_CONTAINER_NAME,
            verbose,
        )
        assert isinstance(logs, str)  # mypy
        if as_list:
            return logs.splitlines()
        return logs

    def show_logs(
        self, limit: int = -1, instance_id: Optional[int] = None, verbose: bool = constants.DEFAULT_VERBOSE_LOG
    ) -> None:
        """
        Display the job's execution logs.

        Args:
            limit: The maximum number of lines to display. Negative values are treated as no limit.
            instance_id: Optional instance ID to get logs from a specific instance.
                         If not provided, displays logs from the head node.
            verbose: Whether to return the full log or just the user log.
        """
        print(self.get_logs(limit, instance_id, as_list=False, verbose=verbose))  # noqa: T201: we need to print here.

    @telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["timeout"])
    def wait(self, timeout: float = -1) -> types.JOB_STATUS:
        """
        Block until completion. Returns completion status.

        Args:
            timeout: The maximum time to wait in seconds. Negative values are treated as no timeout.

        Returns:
            The job's completion status.

        Raises:
            TimeoutError: If the job does not complete within the specified timeout.
        """
        delay = constants.JOB_POLL_INITIAL_DELAY_SECONDS  # Start with 100ms delay
        start_time = time.monotonic()
        warning_shown = False
        while (status := self.status) not in TERMINAL_JOB_STATUSES:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout >= 0:
                raise TimeoutError(f"Job {self.name} did not complete within {timeout} seconds")
            elif status == "PENDING" and not warning_shown and elapsed >= 5:  # Only show warning after 5s
                pool_info = _get_compute_pool_info(self._session, self._compute_pool)
                if (pool_info.max_nodes - pool_info.active_nodes) < self.min_instances:
                    logger.warning(
                        f"Compute pool busy ({pool_info.active_nodes}/{pool_info.max_nodes} nodes in use, "
                        f"{self.min_instances} nodes required). Job execution may be delayed."
                    )
                    warning_shown = True
            time.sleep(delay)
            delay = min(delay * 1.2, constants.JOB_POLL_MAX_DELAY_SECONDS)  # Exponential backoff
        return self.status

    @telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["timeout"])
    def result(self, timeout: float = -1) -> T:
        """
        Block until completion. Returns job execution result.

        Args:
            timeout: The maximum time to wait in seconds. Negative values are treated as no timeout.

        Returns:
            T: The deserialized job result.  # noqa: DAR401

        Raises:
            RuntimeError: If the job failed or if the job doesn't have a result to retrieve.
            TimeoutError: If the job does not complete within the specified timeout.  # noqa: DAR402
        """
        if self._result is None:
            self.wait(timeout)
            try:
                self._result = interop_utils.load_result(
                    self._result_path, session=self._session, path_transform=self._transform_path
                )
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve result for job, error: {e!r}") from e

        return cast(T, self._result.get_value())

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def cancel(self) -> None:
        """
        Cancel the running job.
        Raises:
            RuntimeError: If cancellation fails.  # noqa: DAR401
        """
        try:
            self._session.sql(f"CALL {self.id}!spcs_cancel_job()").collect()
            logger.debug(f"Cancellation requested for job {self.id}")
        except SnowparkSQLException as e:
            raise RuntimeError(f"Failed to cancel job, error: {e!r}") from e


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "instance_id"])
def _get_status(session: snowpark.Session, job_id: str, instance_id: Optional[int] = None) -> types.JOB_STATUS:
    """Retrieve job or job instance execution status."""
    try:
        if instance_id is not None:
            # Get specific instance status
            rows = query_helper.run_query(session, "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=(job_id,))
            for row in rows:
                if row["instance_id"] == str(instance_id):
                    return cast(types.JOB_STATUS, row["status"])
            raise ValueError(f"Instance {instance_id} not found in job {job_id}")
        else:
            row = _get_service_info(session, job_id)
            return cast(types.JOB_STATUS, row["status"])
    except SnowparkSQLException as e:
        if e.sql_error_code == 2003:
            row = _get_service_info_spcs(session, job_id)
            return cast(types.JOB_STATUS, row["STATUS"])
        raise


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_service_spec(session: snowpark.Session, job_id: str) -> dict[str, Any]:
    """Retrieve job execution service spec."""
    row = _get_service_info(session, job_id)
    return cast(dict[str, Any], yaml.safe_load(row["spec"]))


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "limit", "instance_id"])
def _get_logs(
    session: snowpark.Session,
    job_id: str,
    limit: int = -1,
    instance_id: Optional[int] = None,
    container_name: str = constants.DEFAULT_CONTAINER_NAME,
    verbose: bool = True,
) -> str:
    """
    Retrieve the job's execution logs.

    Args:
        job_id: The job ID.
        limit: The maximum number of lines to return. Negative values are treated as no limit.
        session: The Snowpark session to use. If none specified, uses active session.
        instance_id: Optional instance ID to get logs from a specific instance.
        container_name: The container name to get logs from a specific container.
        verbose: Whether to return the full log or just the portion between START and END messages.

    Returns:
        The job's execution logs.

    Raises:
        RuntimeError: if failed to get head instance_id
        SnowparkSQLException: if there is an error retrieving logs from SPCS interface.
    """
    # If instance_id is not specified, try to get the head instance ID
    if instance_id is None:
        try:
            instance_id = _get_head_instance_id(session, job_id)
        except RuntimeError:
            instance_id = None

    # Assemble params: [job_id, instance_id, container_name, (optional) limit]
    params: list[Any] = [
        job_id,
        0 if instance_id is None else instance_id,
        container_name,
    ]
    if limit > 0:
        params.append(limit)
    try:
        (row,) = query_helper.run_query(
            session,
            f"SELECT SYSTEM$GET_SERVICE_LOGS(?, ?, ?{f', ?' if limit > 0 else ''})",
            params=params,
        )
        full_log = str(row[0])
    except SnowparkSQLException as e:
        if "Container Status: PENDING" in e.message:
            logger.warning("Waiting for container to start. Logs will be shown when available.")
            return ""
        else:
            # Fallback plan:
            # 1. Try SPCS Interface (doesn't require event table permission)
            # 2. If the interface call fails, query Event Table (requires permission)
            logger.debug("falling back to SPCS Interface for logs")
            try:
                logs = _get_logs_spcs(
                    session,
                    job_id,
                    limit=limit,
                    instance_id=instance_id if instance_id else 0,
                    container_name=container_name,
                )
                full_log = os.linesep.join(row[0] for row in logs)

            except SnowparkSQLException as spcs_error:
                if spcs_error.sql_error_code == 2143:
                    logger.debug("persistent logs may not be enabled, falling back to event table")
                else:
                    # If SPCS Interface fails for any other reason,
                    # for example, incorrect argument format,raise the error directly
                    raise
                # event table accepts job name, not fully qualified name
                db, schema, name = identifier.parse_schema_level_object_identifier(job_id)
                db = db or session.get_current_database()
                schema = schema or session.get_current_schema()
                event_table_logs = _get_service_log_from_event_table(
                    session,
                    name,
                    database=db,
                    schema=schema,
                    instance_id=instance_id if instance_id else 0,
                    limit=limit,
                )
                if len(event_table_logs) == 0:
                    raise RuntimeError(
                        "No logs were found. Please verify that the database, schema, and job ID are correct."
                    )
                full_log = os.linesep.join(json.loads(row[0]) for row in event_table_logs)

    # If verbose is True, return the complete log
    if verbose:
        return full_log

    # Otherwise, extract only the portion between LOG_START_MSG and LOG_END_MSG
    start_idx = full_log.find(constants.LOG_START_MSG)
    if start_idx != -1:
        start_idx += len(constants.LOG_START_MSG)
    else:
        # If start message not found, start from the beginning
        start_idx = 0

    end_idx = full_log.find(constants.LOG_END_MSG, start_idx)
    if end_idx == -1:
        # If end message not found, return everything after start
        end_idx = len(full_log)

    return full_log[start_idx:end_idx].strip()


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_head_instance_id(session: snowpark.Session, job_id: str) -> Optional[int]:
    """
    Retrieve the head instance ID of a job.

    Args:
        session (Session): The Snowpark session to use.
        job_id (str): The job ID.

    Returns:
        Optional[int]: The head instance ID of the job, or None if the head instance has not started yet.

     Raises:
        RuntimeError: If the instances died or if some instances disappeared.
    """

    try:
        target_instances = _get_target_instances(session, job_id)
    except SnowparkSQLException:
        # service may be deleted
        raise RuntimeError("Couldn’t retrieve service information")

    if target_instances == 1:
        return 0

    try:
        rows = query_helper.run_query(
            session,
            "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)",
            params=(job_id,),
        )
    except SnowparkSQLException:
        # service may be deleted
        raise RuntimeError("Couldn’t retrieve instances")

    if not rows:
        return None

    # we have already integrated with first_instance startup policy,
    # the instance 0 is guaranteed to be the head instance
    head_instance = next(
        (
            row
            for row in rows
            if "instance_id" in row and row["instance_id"] is not None and int(row["instance_id"]) == 0
        ),
        None,
    )
    # fallback to find the first instance if the instance 0 is not found
    if not head_instance:
        if target_instances > len(rows):
            raise RuntimeError(
                f"Couldn’t retrieve head instance due to missing instances. {target_instances} > {len(rows)}"
            )
        # Sort by start_time first, then by instance_id
        try:
            sorted_instances = sorted(rows, key=lambda x: (x["start_time"], int(x["instance_id"])))
        except TypeError:
            raise RuntimeError("Job instance information unavailable.")
        head_instance = sorted_instances[0]

    if not head_instance["start_time"]:
        # If head instance hasn't started yet, return None
        return None
    try:
        return int(head_instance["instance_id"])
    except (ValueError, TypeError):
        return 0


def _get_service_log_from_event_table(
    session: snowpark.Session,
    name: str,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    instance_id: Optional[int] = None,
    limit: int = -1,
) -> list[Row]:
    event_table_name = session.sql("SHOW PARAMETERS LIKE 'event_table' IN ACCOUNT").collect()[0]["value"]
    query = [
        "SELECT VALUE FROM IDENTIFIER(?)",
        'WHERE RESOURCE_ATTRIBUTES:"snow.service.name" = ?',
    ]
    params: list[Any] = [
        event_table_name,
        name,
    ]
    if database:
        query.append('AND RESOURCE_ATTRIBUTES:"snow.database.name" = ?')
        params.append(database)

    if schema:
        query.append('AND RESOURCE_ATTRIBUTES:"snow.schema.name" = ?')
        params.append(schema)

    if instance_id:
        query.append('AND RESOURCE_ATTRIBUTES:"snow.service.container.instance" = ?')
        params.append(instance_id)

    query.append("AND RECORD_TYPE = 'LOG'")
    # sort by TIMESTAMP; although OBSERVED_TIMESTAMP is for log, it is NONE currently when record_type is log
    query.append("ORDER BY TIMESTAMP")

    if limit > 0:
        query.append("LIMIT ?")
        params.append(limit)
    # the wrap used in query_helper does not have return type.
    # sticking a # type: ignore[no-any-return] is to pass type check
    rows = query_helper.run_query(
        session,
        "\n".join(line for line in query if line),
        params=params,
    )
    return rows  # type: ignore[no-any-return]


def _get_service_info(session: snowpark.Session, job_id: str) -> Any:
    (row,) = query_helper.run_query(session, "DESCRIBE SERVICE IDENTIFIER(?)", params=(job_id,))
    return row


def _get_compute_pool_info(session: snowpark.Session, compute_pool: str) -> Row:
    """
    Check if the compute pool has enough available instances.

    Args:
        session (Session): The Snowpark session to use.
        compute_pool (str): The name of the compute pool.

    Returns:
        Row: The compute pool information.

    Raises:
        ValueError: If the compute pool is not found.
    """
    try:
        # the wrap used in query_helper does not have return type.
        # sticking a # type: ignore[no-any-return] is to pass type check
        (pool_info,) = query_helper.run_query(session, "SHOW COMPUTE POOLS LIKE ?", params=(compute_pool,))
        return pool_info  # type: ignore[no-any-return]
    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise ValueError(f"Compute pool '{compute_pool}' not found")
        raise


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_target_instances(session: snowpark.Session, job_id: str) -> int:
    try:
        row = _get_service_info(session, job_id)
        return int(row["target_instances"])
    except SnowparkSQLException as e:
        if e.sql_error_code == 2003:
            row = _get_service_info_spcs(session, job_id)
            try:
                params = json.loads(row["PARAMETERS"])
                if isinstance(params, dict):
                    return int(params.get("REPLICAS", 1))
                else:
                    return 1
            except (json.JSONDecodeError, ValueError):
                return 1
        raise


def _get_logs_spcs(
    session: snowpark.Session,
    fully_qualified_name: str,
    limit: int = -1,
    instance_id: Optional[int] = None,
    container_name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> list[Row]:
    query = [
        f"SELECT LOG FROM table({fully_qualified_name}!spcs_get_logs(",
    ]
    conditions_params = []
    if start_time:
        conditions_params.append(f"start_time => TO_TIMESTAMP_LTZ('{start_time}')")
    if end_time:
        conditions_params.append(f"end_time => TO_TIMESTAMP_LTZ('{end_time}')")
    if len(conditions_params) > 0:
        query.append(", ".join(conditions_params))

    query.append("))")

    query_params = []
    if instance_id is not None:
        query_params.append(f"INSTANCE_ID = {instance_id}")
    if container_name:
        query_params.append(f"CONTAINER_NAME = '{container_name}'")
    if len(query_params) > 0:
        query.append("WHERE " + " AND ".join(query_params))

    query.append("ORDER BY TIMESTAMP ASC")
    if limit > 0:
        query.append(f" LIMIT {limit};")
    rows = session.sql("\n".join(query)).collect()
    return rows


def _get_service_info_spcs(session: snowpark.Session, job_id: str) -> Any:
    """
    Retrieve the service info from the SPCS interface.

    Args:
        session (Session): The Snowpark session to use.
        job_id (str): The job ID.

    Returns:
        Any: The service info.

    Raises:
        SnowparkSQLException: If the job does not exist or is too old to retrieve.
    """
    db, schema, name = identifier.parse_schema_level_object_identifier(job_id)
    db = db or session.get_current_database()
    schema = schema or session.get_current_schema()
    rows = query_helper.run_query(
        session,
        """
        select DATABASE_NAME, SCHEMA_NAME, NAME, STATUS, COMPUTE_POOL_NAME, PARAMETERS
        from table(snowflake.spcs.get_job_history())
        where database_name = ? and schema_name = ? and name = ?
        """,
        params=(db, schema, name),
    )
    if rows:
        return rows[0]
    else:
        raise SnowparkSQLException(f"Job {job_id} does not exist or could not be retrieved", sql_error_code=2003)


def _resolve_service_info(id: str, session: snowpark.Session) -> types.ServiceInfo:
    try:
        row = _get_service_info(session, id)
    except SnowparkSQLException as e:
        if e.sql_error_code == 2003:
            row = _get_service_info_spcs(session, id)
        else:
            raise
    if not row:
        raise SnowparkSQLException(f"Job {id} does not exist or could not be retrieved", sql_error_code=2003)

    if "compute_pool" in row:
        compute_pool = row["compute_pool"]
    elif "COMPUTE_POOL_NAME" in row:
        compute_pool = row["COMPUTE_POOL_NAME"]
    else:
        raise ValueError(f"compute_pool not found in row: {row}")

    if "status" in row:
        status = row["status"]
    elif "STATUS" in row:
        status = row["STATUS"]
    else:
        raise ValueError(f"status not found in row: {row}")
    # Normalize target_instances
    target_instances: int
    if "target_instances" in row and row["target_instances"] is not None:
        try:
            target_instances = int(row["target_instances"])
        except (ValueError, TypeError):
            target_instances = 1
    elif "PARAMETERS" in row and row["PARAMETERS"]:
        try:
            params = json.loads(row["PARAMETERS"])
            target_instances = int(params.get("REPLICAS", 1)) if isinstance(params, dict) else 1
        except (json.JSONDecodeError, ValueError, TypeError):
            target_instances = 1
    else:
        target_instances = 1

    database_name = row["database_name"] if "database_name" in row else row["DATABASE_NAME"]
    schema_name = row["schema_name"] if "schema_name" in row else row["SCHEMA_NAME"]

    return types.ServiceInfo(
        database_name=database_name,
        schema_name=schema_name,
        status=cast(types.JOB_STATUS, status),
        compute_pool=cast(str, compute_pool),
        target_instances=target_instances,
    )
