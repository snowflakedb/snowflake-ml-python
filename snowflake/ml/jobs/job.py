import json
import logging
import os
import time
from functools import cached_property
from pathlib import PurePosixPath
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import yaml

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.mixins import SerializableSessionMixin
from snowflake.ml.jobs._interop import (
    data_utils,
    exception_utils,
    results as interop_result,
    utils as interop_utils,
)
from snowflake.ml.jobs._utils import constants, query_helper, stage_utils, type_utils
from snowflake.snowpark import Row, context as sp_context
from snowflake.snowpark.exceptions import SnowparkSQLException

_PROJECT = "MLJob"
TERMINAL_JOB_STATUSES = {"FAILED", "DONE", "CANCELLED", "INTERNAL_ERROR", "DELETED"}
RAY_DASHBOARD_ENDPOINT_NAME = "ray-dashboard-endpoint"
JOB_RUNNING_STATUS = "RUNNING"
# Launch backends whose jobs have no single result-bearing instance: result() reduces
# per-instance records into a DistributedResult instead of reading one head result.
_DISTRIBUTED_RESULT_BACKENDS = {constants.LAUNCH_BACKEND_PASSTHROUGH}

# Per-instance records upload to the stage asynchronously (SPCS visibility lag), so a written
# record may not be readable immediately. reduce retries missing records up to this timeout
# before treating an instance as lost.
_INSTANCE_RECORD_TIMEOUT_SECONDS = 30
_INSTANCE_RECORD_POLL_INTERVAL_SECONDS = 2

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

        self._status: type_utils.JOB_STATUS = "PENDING"
        self._result: Optional[interop_result.ExecutionResult] = None
        self._distributed_result: Optional[interop_result.DistributedResult] = None

    @cached_property
    def _service_info(self) -> type_utils.ServiceInfo:
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

    @cached_property
    def _has_distributed_result(self) -> bool:
        """Whether this job's result is per-instance records to reduce, not a single head result."""
        return self._container_spec.get("env", {}).get(constants.LAUNCH_BACKEND_ENV_VAR) in _DISTRIBUTED_RESULT_BACKENDS

    @property
    def id(self) -> str:
        """Get the unique job ID"""
        return self._id

    @property
    def status(self) -> type_utils.JOB_STATUS:
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
    def _stage_path(self) -> Optional[str]:
        """Get the job's artifact storage stage location."""
        volumes = self._service_spec["spec"]["volumes"]
        stage_volume = next((v for v in volumes if v["name"] == constants.STAGE_VOLUME_NAME), None)
        if stage_volume is None:
            return None
        elif "stageConfig" in stage_volume:
            return cast(str, stage_volume["stageConfig"]["name"])
        else:
            return cast(str, stage_volume["source"])

    @property
    def _result_path(self) -> str:
        """Get the job's result file location."""
        result_path_str = self._container_spec["env"].get(constants.RESULT_PATH_ENV_VAR)
        if result_path_str is None:
            raise NotImplementedError(f"Job {self.name} doesn't have a result path configured")

        return self._transform_path(result_path_str)

    # After introducing ML Job definitions, we have additional stage mount for result path
    # the result path is like @payload_stage/{job_definition_name}/{job_name}/mljob_result
    @property
    def _result_stage_path(self) -> Optional[str]:
        volumes = self._service_spec["spec"]["volumes"]
        stage_volume = next((v for v in volumes if v["name"] == constants.RESULT_VOLUME_NAME), None)
        if stage_volume is None:
            return self._stage_path
        elif "stageConfig" in stage_volume:
            return cast(str, stage_volume["stageConfig"]["name"])
        else:
            return cast(str, stage_volume["source"])

    def _transform_path(
        self,
        path_str: str,
    ) -> str:
        """Transform a local path within the container to a stage path.

        Container paths (from Linux SPCS) are always POSIX-style and must be handled
        with PurePosixPath to ensure consistent behavior across platforms (e.g., when
        Windows clients retrieve results from Linux containers).

        Args:
            path_str: The path string to transform.

        Returns:
            A stage path string in the format @stage_name/path.

        Raises:
            ValueError: If the path is absolute but not relative to any known mount point.
        """
        path = stage_utils.resolve_path(path_str)
        if isinstance(path, stage_utils.StagePath):
            return path.as_posix()

        # Use PurePosixPath for all path operations to ensure cross-platform compatibility.
        # Container paths are always POSIX-style (from Linux SPCS), regardless of client OS.
        # On Windows, Path("/mnt/...").is_absolute() returns False (the bug), but
        # PurePosixPath("/mnt/...").is_absolute() correctly returns True on all platforms.
        posix_path = PurePosixPath(path_str)

        if not posix_path.is_absolute():
            return f"{self._result_stage_path}/{posix_path.as_posix()}"

        volume_mounts = self._container_spec["volumeMounts"]
        stage_volume = next((v for v in volume_mounts if v["name"] == constants.RESULT_VOLUME_NAME), None)
        if stage_volume is None:
            stage_volume = next(v for v in volume_mounts if v["name"] == constants.STAGE_VOLUME_NAME)
        stage_mount_str = stage_volume["mountPath"]
        stage_mount = PurePosixPath(stage_mount_str)
        try:
            relative_path = posix_path.relative_to(stage_mount)
            return f"{self._result_stage_path}/{relative_path.as_posix()}"
        except ValueError:
            raise ValueError(
                f"Result Path {posix_path} is absolute, but should be relative to stage mount {stage_mount}"
            )

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
    def wait(self, timeout: float = -1) -> type_utils.JOB_STATUS:
        """
        Block until completion. Returns completion status.

        Args:
            timeout: The maximum time to wait in seconds. Negative values are treated as no timeout.

        Returns:
            The job's completion status.

        Raises:
            TimeoutError: If the job does not complete within the specified timeout.
        """
        start_time = time.monotonic()
        try:
            # spcs_wait_for() is a synchronous query, it’s more effective to do polling with exponential
            # backoff. If the job is running for a long time. We want a hybrid option: use spcs_wait_for()
            # for the first 30 seconds, then switch to polling for long running jobs
            min_timeout = (
                int(min(timeout, constants.JOB_SPCS_TIMEOUT_SECONDS))
                if timeout >= 0
                else constants.JOB_SPCS_TIMEOUT_SECONDS
            )
            query_helper.run_query(self._session, f"call {self.id}!spcs_wait_for('DONE', {min_timeout})")
            return self.status
        except SnowparkSQLException:
            # if the function does not support for this environment
            pass
        delay: float = float(constants.JOB_POLL_INITIAL_DELAY_SECONDS)  # Start with 5s delay
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

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def get_ray_dashboard_url(self) -> Optional[str]:
        """
        Get the Ray dashboard URL for the job.

        Returns:
            Optional[str]: The Ray dashboard URL if the job is running and has a Ray dashboard endpoint,
                None otherwise.
        """
        pool_info = _get_compute_pool_info(self._session, self._compute_pool)
        if pool_info["instance_family"] == "CPU_X64_XS":
            logger.warning(
                "Ray dashboard is not supported on XS compute pools."
                " Please use a larger compute pool if you want to use Ray dashboard."
            )
            return None
        if self.status != JOB_RUNNING_STATUS:
            logger.warning("Ray dashboard is not available for non-running jobs")
            return None
        rows = query_helper.run_query(self._session, "SHOW ENDPOINTS IN SERVICE IDENTIFIER(?)", params=(self.id,))
        for row in rows:
            if row["name"] == RAY_DASHBOARD_ENDPOINT_NAME:
                ingress_url = row["ingress_url"]
                return str(ingress_url) if ingress_url is not None else None
        return None

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
                self._result = interop_utils.load(
                    self._result_path, session=self._session, path_transform=self._transform_path
                )
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve result for job, error: {e!r}") from e

        return cast(T, self._result.get_value())

    @telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["timeout"])
    def distributed_result(self, timeout: float = -1) -> interop_result.DistributedResult:
        """
        Block until completion, then return the structured per-instance summary of a distributed run.

        This is the result API for distributed jobs (e.g. submitted with ``parallel=True``), which
        have no single head result. On full success it returns a :class:`DistributedResult`
        (``success``, per-instance ``exit_codes``, ``failed_instance``, instance 0's ``return_value``).
        On any instance failure it raises :class:`DistributedJobError`, which carries that same
        :class:`DistributedResult` as ``.result`` and the earliest-failing instance's reconstructed
        exception as its cause. ``result()`` remains the single-head API.

        Args:
            timeout: The maximum time to wait in seconds. Negative values are treated as no timeout.

        Returns:
            DistributedResult: The reduced per-instance result, on full success.  # noqa: DAR401

        Raises:
            NotImplementedError: If the job was not launched with a distributed backend (e.g.
                ``parallel=True``); single-head jobs have no per-instance result — use :meth:`result`.
            DistributedJobError: If any instance did not exit 0. Carries the DistributedResult
                (``.result``) with per-instance ``exit_codes`` / ``failed_instance``.  # noqa: DAR402
            RuntimeError: If the job's per-instance records could not be retrieved.
            TimeoutError: If the job does not complete within the specified timeout.  # noqa: DAR402
        """
        if not self._has_distributed_result:
            raise NotImplementedError(
                "distributed_result() is only available for distributed jobs "
                "(e.g. submitted with parallel=True); use result() for single-head jobs."
            )
        if self._distributed_result is None:
            self.wait(timeout)
            try:
                self._distributed_result = _reduce_distributed_result(
                    self._session, self.id, self._result_path, self._transform_path
                )
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve result for job, error: {e!r}") from e
        distributed_result = self._distributed_result
        if not distributed_result.success:
            raise interop_result.DistributedJobError.from_result(distributed_result) from _rebuild_failure_exception(
                self._session, self._result_path, distributed_result.failed_instance
            )
        return distributed_result

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
def _get_status(session: snowpark.Session, job_id: str, instance_id: Optional[int] = None) -> type_utils.JOB_STATUS:
    """Retrieve job or job instance execution status."""
    try:
        if instance_id is not None:
            # Get specific instance status
            rows = query_helper.run_query(session, "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=(job_id,))
            for row in rows:
                if row["instance_id"] == str(instance_id):
                    return cast(type_utils.JOB_STATUS, row["status"])
            raise ValueError(f"Instance {instance_id} not found in job {job_id}")
        else:
            row = _get_service_info(session, job_id)
            return cast(type_utils.JOB_STATUS, row["status"])
    except SnowparkSQLException as e:
        if e.sql_error_code == 2003:
            row = _get_service_info_spcs(session, job_id)
            return cast(type_utils.JOB_STATUS, row["STATUS"])
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


def _get_service_instances(session: snowpark.Session, job_id: str) -> list[dict[str, Any]]:
    """Return the control-plane instance set ``[{instance_id:int, start_time}]`` via SHOW SERVICE INSTANCES.

    This is the authoritative set of instances the job has; the reduce left-joins per-instance records onto it.

    Args:
        session: The Snowpark session to use.
        job_id: The fully-qualified job (service) ID.

    Returns:
        One dict per instance: ``{"instance_id": int, "start_time": <value>}``.
    """
    rows = query_helper.run_query(session, "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=(job_id,))
    instances = []
    for row in rows:
        # snowpark Row has no dict-style .get(); go through as_dict() for safe key access.
        row_dict = row.as_dict()
        if row_dict.get("instance_id") is None:
            continue
        instances.append({"instance_id": int(row_dict["instance_id"]), "start_time": row_dict.get("start_time")})
    return instances


def _read_instance_record(session: snowpark.Session, result_path: str, instance_id: int) -> Optional[dict[str, Any]]:
    """Read one per-instance record from the stage; ``None`` if absent/unreadable.

    Absent means the instance never wrote its finally-block record (killed/OOM) → lost.

    Args:
        session: The Snowpark session to use.
        result_path: The job's result file stage path; records live in an ``instances/`` dir beside it.
        instance_id: The instance whose record to read.

    Returns:
        The parsed record dict, or ``None`` if the record is absent or unreadable.
    """
    # Per-instance records live in an "instances/" directory alongside the result file.
    path = (PurePosixPath(result_path).parent / "instances" / f"{instance_id}.json").as_posix()
    try:
        with data_utils.open_stream(path, "r", session=session) as stream:
            return cast(dict[str, Any], json.load(stream))
    except Exception as e:
        logger.debug(f"could not read record for instance {instance_id}: {e!r}")
        return None


def _read_all_records_with_retry(
    session: snowpark.Session, result_path: str, instances: list[dict[str, Any]]
) -> dict[int, Optional[dict[str, Any]]]:
    """Read every instance's record, retrying missing ones until visible or timeout.

    Absorbs SPCS stage visibility lag: a written record may not be readable immediately. Returns
    as soon as all records are present (successful jobs pay no extra latency); a record still
    missing after the timeout is treated as lost (``None``) — best-effort, since we cannot tell
    "not yet visible" from "never written" (killed).

    Args:
        session: The Snowpark session to use.
        result_path: The job's result file stage path.
        instances: The control-plane instance set to read records for.

    Returns:
        instance_id -> record dict, or ``None`` for any instance still missing after the timeout (lost).
    """
    deadline = time.monotonic() + _INSTANCE_RECORD_TIMEOUT_SECONDS
    records: dict[int, Optional[dict[str, Any]]] = {}
    missing = {inst["instance_id"] for inst in instances}
    while missing:
        for instance_id in list(missing):
            rec = _read_instance_record(session, result_path, instance_id)
            if rec is not None:
                records[instance_id] = rec
                missing.discard(instance_id)
        if not missing or time.monotonic() >= deadline:
            break
        logger.debug(f"waiting for {len(missing)} instance record(s) {sorted(missing)}; retrying")
        time.sleep(_INSTANCE_RECORD_POLL_INTERVAL_SECONDS)
    if missing:
        logger.info(
            f"{len(missing)} instance record(s) not found after {_INSTANCE_RECORD_TIMEOUT_SECONDS}s; "
            f"treating as lost: {sorted(missing)}"
        )
    for instance_id in missing:
        records[instance_id] = None
    return records


def _earliest_failed_instance(
    instances: list[dict[str, Any]],
    records: dict[int, Optional[dict[str, Any]]],
    exit_codes: dict[int, Optional[int]],
) -> Optional[int]:
    """Return the failed instance that ended earliest, or None if none failed.

    Best-effort hint, not a guarantee: ordering uses each instance's record ended_at (subject to
    clock skew across instances), falling back to the control-plane start_time when a failure has
    no record.

    Args:
        instances: The control-plane instance set.
        records: instance_id -> record dict, or ``None`` if the instance is lost.
        exit_codes: instance_id -> exit code, or ``None`` if the instance is lost.

    Returns:
        The earliest-failing instance id, or ``None`` if every instance succeeded.
    """
    failed = [inst for inst in instances if exit_codes[inst["instance_id"]] != 0]
    if not failed:
        return None
    # Bind the record to a local so it narrows past the None check (mypy won't narrow a repeated
    # subscript expression). Cast the instance id to int so the return type stays int, not Any.
    with_ended: list[tuple[Any, int]] = []
    for inst in failed:
        rec = records[inst["instance_id"]]
        if rec is not None and rec.get("ended_at") is not None:
            with_ended.append((rec["ended_at"], int(inst["instance_id"])))
    if with_ended:
        return min(with_ended)[1]
    return int(
        min(failed, key=lambda inst: (inst["start_time"] is None, inst["start_time"], inst["instance_id"]))[
            "instance_id"
        ]
    )


def _load_instance0_value_or_none(
    session: snowpark.Session, result_path: str, path_transform: Callable[[str], str]
) -> Any:
    """Instance 0's return value on success; ``None`` if no value file (e.g. subprocess entrypoint)."""
    try:
        loaded = interop_utils.load(result_path, session=session, path_transform=path_transform)
        return loaded.get_value()
    except Exception as e:
        # No value file is expected (subprocess entrypoints don't produce one). Log so a real
        # read error isn't silently swallowed into a None return value.
        logger.debug(f"could not load instance 0 return value: {e!r}")
        return None


def _rebuild_failure_exception(
    session: snowpark.Session, result_path: str, failed_instance: Optional[int]
) -> BaseException:
    """Rebuild the earliest-failing instance's exception, for use as ``distributed_result()``'s raised cause.

    Reads that instance's per-instance record and reconstructs from its string metadata (the same
    build_exception path the classic loader uses). Falls back to a generic ``RemoteError`` when the
    instance recorded no exception metadata — either it wrote no record (killed/lost) or it exited
    non-zero without a reconstructable exception (e.g. a result-save failure).

    Args:
        session: The Snowpark session to use.
        result_path: The job's result file stage path.
        failed_instance: The instance to rebuild the exception for; ``None`` yields a generic error.

    Returns:
        The reconstructed exception, or a generic ``RemoteError`` when no metadata is available.
    """
    rec = _read_instance_record(session, result_path, failed_instance) if failed_instance is not None else None
    exc = rec.get("exc") if rec else None
    if isinstance(exc, dict):
        return exception_utils.build_exception(
            type_str=exc.get("type", ""),
            message=exc.get("message", ""),
            traceback=exc.get("traceback", ""),
            original_repr=exc.get("repr"),
        )
    # exc is absent for two distinct cases; give each an accurate hint rather than guessing a reason.
    if rec is None:
        detail = "wrote no result record (likely killed or OOM before it could report)"
    else:
        detail = f"exited with code {rec.get('exit_code')} but recorded no reconstructable exception"
    return exception_utils.RemoteError(
        f"Instance {failed_instance} {detail}; see job.get_logs(instance_id={failed_instance})."
    )


def _reduce_distributed_result(
    session: snowpark.Session, job_id: str, result_path: str, path_transform: Callable[[str], str]
) -> interop_result.DistributedResult:
    """Reduce per-instance stage records + control-plane state into a DistributedResult.

    LEFT-JOINs each instance's record (``instances/<id>.json``) onto the authoritative
    control-plane instance set, retrying missing records to absorb stage visibility lag. A record
    still missing after retry is treated as lost (exit code ``None``) — best-effort.

    Args:
        session: The Snowpark session to use.
        job_id: The fully-qualified job (service) ID.
        result_path: The job's result file stage path.
        path_transform: Maps a container path to its stage path (used to load instance 0's value).

    Returns:
        The aggregated :class:`DistributedResult`.

    Raises:
        RuntimeError: If the control plane returns no usable instances (couldn't read job state).
    """
    instances = _get_service_instances(session, job_id)
    if not instances:
        raise RuntimeError(f"Couldn't retrieve instance state for job {job_id}")
    records = _read_all_records_with_retry(session, result_path, instances)

    exit_codes: dict[int, Optional[int]] = {}
    for inst in instances:
        rec = records[inst["instance_id"]]
        exit_codes[inst["instance_id"]] = rec.get("exit_code") if rec is not None else None

    success = all(code == 0 for code in exit_codes.values())
    failed_instance = None if success else _earliest_failed_instance(instances, records, exit_codes)
    # return_value is the run's Python return value (instance 0), which only exists on success.
    return_value = _load_instance0_value_or_none(session, result_path, path_transform) if success else None
    return interop_result.DistributedResult(
        success=success,
        exit_codes=exit_codes,
        failed_instance=failed_instance,
        return_value=return_value,
    )


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


def _resolve_service_info(id: str, session: snowpark.Session) -> type_utils.ServiceInfo:
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

    return type_utils.ServiceInfo(
        database_name=database_name,
        schema_name=schema_name,
        status=cast(type_utils.JOB_STATUS, status),
        compute_pool=cast(str, compute_pool),
        target_instances=target_instances,
    )
