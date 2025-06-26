import json
import logging
import os
import time
from functools import cached_property
from typing import Any, Generic, Literal, Optional, TypeVar, Union, cast, overload

import yaml

from snowflake import snowpark
from snowflake.connector import errors
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.mixins import SerializableSessionMixin
from snowflake.ml.jobs._utils import constants, interop_utils, query_helper, types
from snowflake.snowpark import Row, context as sp_context
from snowflake.snowpark.exceptions import SnowparkSQLException

_PROJECT = "MLJob"
TERMINAL_JOB_STATUSES = {"FAILED", "DONE", "CANCELLED", "INTERNAL_ERROR"}

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
        self._result: Optional[interop_utils.ExecutionResult] = None

    @cached_property
    def name(self) -> str:
        return identifier.parse_schema_level_object_identifier(self.id)[-1]

    @cached_property
    def target_instances(self) -> int:
        return _get_target_instances(self._session, self.id)

    @cached_property
    def min_instances(self) -> int:
        try:
            return int(self._container_spec["env"].get(constants.MIN_INSTANCES_ENV_VAR, 1))
        except TypeError:
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
        row = _get_service_info(self._session, self.id)
        compute_pool = row[query_helper.get_attribute_map(self._session, {"compute_pool": 5})["compute_pool"]]
        return cast(str, compute_pool)

    @property
    def _service_spec(self) -> dict[str, Any]:
        """Get the job's service spec."""
        if not self._service_spec_cached:
            self._service_spec_cached = _get_service_spec(self._session, self.id)
        return self._service_spec_cached

    @property
    def _container_spec(self) -> dict[str, Any]:
        """Get the job's main container spec."""
        containers = self._service_spec["spec"]["containers"]
        container_spec = next(c for c in containers if c["name"] == constants.DEFAULT_CONTAINER_NAME)
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
        result_path = self._container_spec["env"].get(constants.RESULT_PATH_ENV_VAR)
        if result_path is None:
            raise RuntimeError(f"Job {self.name} doesn't have a result path configured")
        return f"{self._stage_path}/{result_path}"

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
        logs = _get_logs(self._session, self.id, limit, instance_id, verbose)
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
            if status == "PENDING" and not warning_shown:
                pool_info = _get_compute_pool_info(self._session, self._compute_pool)
                requested_attributes = {"max_nodes": 3, "active_nodes": 9}
                if (
                    pool_info[requested_attributes["max_nodes"]] - pool_info[requested_attributes["active_nodes"]]
                ) < self.min_instances:
                    logger.warning(
                        f'Compute pool busy ({pool_info[requested_attributes["active_nodes"]]}'
                        f'/{pool_info[requested_attributes["max_nodes"]]} nodes in use, '
                        f"{self.min_instances} nodes required). Job execution may be delayed."
                    )
                    warning_shown = True
            if timeout >= 0 and (elapsed := time.monotonic() - start_time) >= timeout:
                raise TimeoutError(f"Job {self.name} did not complete within {elapsed} seconds")
            time.sleep(delay)
            delay = min(delay * 1.2, constants.JOB_POLL_MAX_DELAY_SECONDS)  # Exponential backoff
        return self.status

    @snowpark._internal.utils.private_preview(version="1.8.2")
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
                self._result = interop_utils.fetch_result(self._session, self._result_path)
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve result for job (id={self.name})") from e

        if self._result.success:
            return cast(T, self._result.result)
        raise RuntimeError(f"Job execution failed (id={self.name})") from self._result.exception

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
            raise RuntimeError(f"Failed to cancel job {self.id}: {e.message}") from e


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "instance_id"])
def _get_status(session: snowpark.Session, job_id: str, instance_id: Optional[int] = None) -> types.JOB_STATUS:
    """Retrieve job or job instance execution status."""
    if instance_id is not None:
        # Get specific instance status
        rows = session._conn.run_query(
            "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=[job_id], _force_qmark_paramstyle=True
        )
        request_attributes = query_helper.get_attribute_map(session, {"status": 5, "instance_id": 4})
        if isinstance(rows, dict) and "data" in rows:
            for row in rows["data"]:
                if row[request_attributes["instance_id"]] == str(instance_id):
                    return cast(types.JOB_STATUS, row[request_attributes["status"]])
        raise ValueError(f"Instance {instance_id} not found in job {job_id}")
    else:
        row = _get_service_info(session, job_id)
        request_attributes = query_helper.get_attribute_map(session, {"status": 1})
        return cast(types.JOB_STATUS, row[request_attributes["status"]])


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_service_spec(session: snowpark.Session, job_id: str) -> dict[str, Any]:
    """Retrieve job execution service spec."""
    row = _get_service_info(session, job_id)
    requested_attributes = query_helper.get_attribute_map(session, {"spec": 6})
    return cast(dict[str, Any], yaml.safe_load(row[requested_attributes["spec"]]))


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "limit", "instance_id"])
def _get_logs(
    session: snowpark.Session, job_id: str, limit: int = -1, instance_id: Optional[int] = None, verbose: bool = True
) -> str:
    """
    Retrieve the job's execution logs.

    Args:
        job_id: The job ID.
        limit: The maximum number of lines to return. Negative values are treated as no limit.
        session: The Snowpark session to use. If none specified, uses active session.
        instance_id: Optional instance ID to get logs from a specific instance.
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
        constants.DEFAULT_CONTAINER_NAME,
    ]
    if limit > 0:
        params.append(limit)
    try:
        data = session._conn.run_query(
            f"SELECT SYSTEM$GET_SERVICE_LOGS(?, ?, ?{f', ?' if limit > 0 else ''})",
            params=params,
            _force_qmark_paramstyle=True,
        )
        if isinstance(data, dict) and "data" in data:
            full_log = str(data["data"][0][0])
        # pass type check
        else:
            full_log = ""
    except errors.ProgrammingError as e:
        if "Container Status: PENDING" in str(e):
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
                    container_name=constants.DEFAULT_CONTAINER_NAME,
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
    except errors.ProgrammingError:
        # service may be deleted
        raise RuntimeError("Couldn’t retrieve service information")

    if target_instances == 1:
        return 0

    try:
        rows = session._conn.run_query(
            "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=(job_id,), _force_qmark_paramstyle=True
        )
    except errors.ProgrammingError:
        # service may be deleted
        raise RuntimeError("Couldn’t retrieve instances")

    if not rows or not isinstance(rows, dict) or not rows.get("data"):
        return None

    if target_instances > len(rows["data"]):
        raise RuntimeError("Couldn’t retrieve head instance due to missing instances.")

    requested_attributes = query_helper.get_attribute_map(session, {"start_time": 8, "instance_id": 4})
    # Sort by start_time first, then by instance_id
    try:
        sorted_instances = sorted(
            rows["data"],
            key=lambda x: (x[requested_attributes["start_time"]], int(x[requested_attributes["instance_id"]])),
        )
    except TypeError:
        raise RuntimeError("Job instance information unavailable.")
    head_instance = sorted_instances[0]
    if not head_instance[requested_attributes["start_time"]]:
        # If head instance hasn't started yet, return None
        return None
    try:
        return int(head_instance[requested_attributes["instance_id"]])
    except (ValueError, TypeError):
        return 0


def _get_service_log_from_event_table(
    session: snowpark.Session,
    name: str,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    instance_id: Optional[int] = None,
    limit: int = -1,
) -> Any:
    params: list[Any] = [
        name,
    ]
    query = [
        "SELECT VALUE FROM snowflake.telemetry.events_view",
        'WHERE RESOURCE_ATTRIBUTES:"snow.service.name" = ?',
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
    rows = session._conn.run_query(
        "\n".join(line for line in query if line), params=params, _force_qmark_paramstyle=True
    )
    if not rows or not isinstance(rows, dict) or not rows.get("data"):
        return []
    return rows["data"]


def _get_service_info(session: snowpark.Session, job_id: str) -> Any:
    row = session._conn.run_query("DESCRIBE SERVICE IDENTIFIER(?)", params=(job_id,), _force_qmark_paramstyle=True)
    # pass the type check
    if not row or not isinstance(row, dict) or not row.get("data"):
        raise errors.ProgrammingError("failed to retrieve service information")
    return row["data"][0]


def _get_compute_pool_info(session: snowpark.Session, compute_pool: str) -> Any:
    """
    Check if the compute pool has enough available instances.

    Args:
        session (Session): The Snowpark session to use.
        compute_pool (str): The name of the compute pool.

    Returns:
        Any: The compute pool information.

    Raises:
        ValueError: If the compute pool is not found.
    """
    try:
        compute_pool_info = session._conn.run_query(
            "SHOW COMPUTE POOLS LIKE ?", params=(compute_pool,), _force_qmark_paramstyle=True
        )
        # pass the type check
        if not compute_pool_info or not isinstance(compute_pool_info, dict) or not compute_pool_info.get("data"):
            raise ValueError(f"Compute pool '{compute_pool}' not found")
        return compute_pool_info["data"][0]
    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise ValueError(f"Compute pool '{compute_pool}' not found")
        raise


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_target_instances(session: snowpark.Session, job_id: str) -> int:
    row = _get_service_info(session, job_id)
    requested_attributes = query_helper.get_attribute_map(session, {"target_instances": 9})
    return int(row[requested_attributes["target_instances"]])


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
