import time
from typing import Any, List, Optional, cast

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.jobs._utils import constants, types
from snowflake.snowpark import context as sp_context

_PROJECT = "MLJob"
TERMINAL_JOB_STATUSES = {"FAILED", "DONE", "INTERNAL_ERROR"}


class MLJob:
    def __init__(self, id: str, session: Optional[snowpark.Session] = None) -> None:
        self._id = id
        self._session = session or sp_context.get_active_session()
        self._status: types.JOB_STATUS = "PENDING"

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

    @snowpark._internal.utils.private_preview(version="1.7.4")
    def get_logs(self, limit: int = -1) -> str:
        """
        Return the job's execution logs.

        Args:
            limit: The maximum number of lines to return. Negative values are treated as no limit.

        Returns:
            The job's execution logs.
        """
        logs = _get_logs(self._session, self.id, limit)
        assert isinstance(logs, str)  # mypy
        return logs

    @snowpark._internal.utils.private_preview(version="1.7.4")
    def show_logs(self, limit: int = -1) -> None:
        """
        Display the job's execution logs.

        Args:
            limit: The maximum number of lines to display. Negative values are treated as no limit.
        """
        print(self.get_logs(limit))  # noqa: T201: we need to print here.

    @snowpark._internal.utils.private_preview(version="1.7.4")
    @telemetry.send_api_usage_telemetry(project=_PROJECT)
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


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id"])
def _get_status(session: snowpark.Session, job_id: str) -> types.JOB_STATUS:
    """Retrieve job execution status."""
    # TODO: snowflake-snowpark-python<1.24.0 shows spurious error messages on
    #       `DESCRIBE` queries with bind variables
    #       Switch to use bind variables instead of client side formatting after
    #       updating to snowflake-snowpark-python>=1.24.0
    (row,) = session.sql(f"DESCRIBE SERVICE {job_id}").collect()
    return cast(types.JOB_STATUS, row["status"])


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["job_id", "limit"])
def _get_logs(session: snowpark.Session, job_id: str, limit: int = -1) -> str:
    """
    Retrieve the job's execution logs.

    Args:
        job_id: The job ID.
        limit: The maximum number of lines to return. Negative values are treated as no limit.
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        The job's execution logs.
    """
    params: List[Any] = [job_id]
    if limit > 0:
        params.append(limit)
    (row,) = session.sql(
        f"SELECT SYSTEM$GET_SERVICE_LOGS(?, 0, '{constants.DEFAULT_CONTAINER_NAME}'{f', ?' if limit > 0 else ''})",
        params=params,
    ).collect()
    return str(row[0])
