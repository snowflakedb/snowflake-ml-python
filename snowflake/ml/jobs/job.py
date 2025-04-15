import time
from typing import Any, Generic, Literal, Optional, TypeVar, Union, cast, overload

import yaml

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.jobs._utils import constants, interop_utils, types
from snowflake.snowpark import context as sp_context

_PROJECT = "MLJob"
TERMINAL_JOB_STATUSES = {"FAILED", "DONE", "INTERNAL_ERROR"}

T = TypeVar("T")


class MLJob(Generic[T]):
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
            raise RuntimeError(f"Job {self.id} doesn't have a result path configured")
        return f"{self._stage_path}/{result_path}"

    @overload
    def get_logs(self, limit: int = -1, instance_id: Optional[int] = None, *, as_list: Literal[True]) -> list[str]:
        ...

    @overload
    def get_logs(self, limit: int = -1, instance_id: Optional[int] = None, *, as_list: Literal[False] = False) -> str:
        ...

    def get_logs(
        self, limit: int = -1, instance_id: Optional[int] = None, *, as_list: bool = False
    ) -> Union[str, list[str]]:
        """
        Return the job's execution logs.

        Args:
            limit: The maximum number of lines to return. Negative values are treated as no limit.
            instance_id: Optional instance ID to get logs from a specific instance.
                         If not provided, returns logs from the head node.
            as_list: If True, returns logs as a list of lines. Otherwise, returns logs as a single string.

        Returns:
            The job's execution logs.
        """
        logs = _get_logs(self._session, self.id, limit, instance_id)
        assert isinstance(logs, str)  # mypy
        if as_list:
            return logs.splitlines()
        return logs

    def show_logs(self, limit: int = -1, instance_id: Optional[int] = None) -> None:
        """
        Display the job's execution logs.

        Args:
            limit: The maximum number of lines to display. Negative values are treated as no limit.
            instance_id: Optional instance ID to get logs from a specific instance.
                         If not provided, displays logs from the head node.
        """
        print(self.get_logs(limit, instance_id, as_list=False))  # noqa: T201: we need to print here.

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
        while self.status not in TERMINAL_JOB_STATUSES:
            if timeout >= 0 and (elapsed := time.monotonic() - start_time) >= timeout:
                raise TimeoutError(f"Job {self.id} did not complete within {elapsed} seconds")
            time.sleep(delay)
            delay = min(delay * 2, constants.JOB_POLL_MAX_DELAY_SECONDS)  # Exponential backoff
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
                raise RuntimeError(f"Failed to retrieve result for job (id={self.id})") from e

        if self._result.success:
            return cast(T, self._result.result)
        raise RuntimeError(f"Job execution failed (id={self.id})") from self._result.exception


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "instance_id"])
def _get_status(session: snowpark.Session, job_id: str, instance_id: Optional[int] = None) -> types.JOB_STATUS:
    """Retrieve job or job instance execution status."""
    if instance_id is not None:
        # Get specific instance status
        rows = session.sql("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=(job_id,)).collect()
        for row in rows:
            if row["instance_id"] == str(instance_id):
                return cast(types.JOB_STATUS, row["status"])
        raise ValueError(f"Instance {instance_id} not found in job {job_id}")
    else:
        (row,) = session.sql("DESCRIBE SERVICE IDENTIFIER(?)", params=(job_id,)).collect()
        return cast(types.JOB_STATUS, row["status"])


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_service_spec(session: snowpark.Session, job_id: str) -> dict[str, Any]:
    """Retrieve job execution service spec."""
    (row,) = session.sql("DESCRIBE SERVICE IDENTIFIER(?)", params=[job_id]).collect()
    return cast(dict[str, Any], yaml.safe_load(row["spec"]))


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "limit", "instance_id"])
def _get_logs(session: snowpark.Session, job_id: str, limit: int = -1, instance_id: Optional[int] = None) -> str:
    """
    Retrieve the job's execution logs.

    Args:
        job_id: The job ID.
        limit: The maximum number of lines to return. Negative values are treated as no limit.
        session: The Snowpark session to use. If none specified, uses active session.
        instance_id: Optional instance ID to get logs from a specific instance.

    Returns:
        The job's execution logs.
    """
    # If instance_id is not specified, try to get the head instance ID
    if instance_id is None:
        instance_id = _get_head_instance_id(session, job_id)

    # Assemble params: [job_id, instance_id, container_name, (optional) limit]
    params: list[Any] = [
        job_id,
        0 if instance_id is None else instance_id,
        constants.DEFAULT_CONTAINER_NAME,
    ]
    if limit > 0:
        params.append(limit)

    (row,) = session.sql(
        f"SELECT SYSTEM$GET_SERVICE_LOGS(?, ?, ?{f', ?' if limit > 0 else ''})",
        params=params,
    ).collect()
    return str(row[0])


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_head_instance_id(session: snowpark.Session, job_id: str) -> Optional[int]:
    """
    Retrieve the head instance ID of a job.

    Args:
        session: The Snowpark session to use.
        job_id: The job ID.

    Returns:
        The head instance ID of the job. Returns None if the head instance has not started yet.
    """
    rows = session.sql("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=(job_id,)).collect()
    if not rows:
        return None

    # Sort by start_time first, then by instance_id
    sorted_instances = sorted(rows, key=lambda x: (x["start_time"], int(x["instance_id"])))
    head_instance = sorted_instances[0]
    if not head_instance["start_time"]:
        # If head instance hasn't started yet, return None
        return None
    try:
        return int(head_instance["instance_id"])
    except (ValueError, TypeError):
        return 0
