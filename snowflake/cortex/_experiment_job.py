import enum
import json
import logging
import time
from typing import Any, cast

from snowflake import snowpark
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml._internal.utils.mixins import SerializableSessionMixin
from snowflake.ml.experiment._client import experiment_tracking_sql_client
from snowflake.ml.experiment._entities import (
    experiment as experiment_entity,
    run_metadata,
)
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as snowpark_exceptions

logger = logging.getLogger(__name__)

_TERMINAL_RUN_STATUSES = frozenset(status.value for status in run_metadata.RunStatus if status.is_terminal())
_POLL_INTERVAL_SECONDS = 10.0


class JobStatus(str, enum.Enum):
    NOT_EXECUTED = "NOT_EXECUTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    # Reported when the server returns a status this client does not recognize. It is treated as still in progress.
    UNKNOWN = "UNKNOWN"


class ExperimentJob(SerializableSessionMixin):
    """Shared lifecycle for asynchronous Cortex experiment jobs."""

    def __init__(self, session: snowpark.Session, experiment: str, *, job_type: str) -> None:
        self._session = session
        self._job_type = job_type
        self._client, self._experiment_name = resolve_client(session, experiment)
        self._experiment = self._client.fully_qualified_object_name(None, None, self._experiment_name)
        self._status = JobStatus.NOT_EXECUTED
        self._aborted = False

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_client", None)
        return state

    @property
    def experiment(self) -> str:
        """Fully qualified name of the experiment backing this job."""
        return self._experiment

    @property
    def status(self) -> JobStatus:
        if self._status is not JobStatus.COMPLETED:
            description = self._get_client().describe_experiment(self._experiment_name)
            execution_status = description.get("execution_status")
            if execution_status is None:
                self._status = self._get_status_from_runs()
            else:
                try:
                    self._status = JobStatus(execution_status)
                except ValueError:
                    logger.warning(
                        f"{self._job_type} experiment {self._experiment!r} reported an unrecognized execution status "
                        f"{execution_status!r}; treating it as still in progress."
                    )
                    self._status = JobStatus.UNKNOWN
        return self._status

    @property
    def error_message(self) -> str | None:
        """Error reported by the backend for this experiment, or ``None`` if it has not reported one."""
        error_message = self._get_summary().get("error_message")
        return str(error_message) if error_message else None

    def wait(self, *, timeout: float = -1) -> JobStatus:
        start_time = time.monotonic()
        while (status := self.status) is not JobStatus.COMPLETED:
            if status is JobStatus.NOT_EXECUTED:
                raise RuntimeError(f"{self._job_type} experiment {self._experiment!r} has not been executed")
            elapsed = time.monotonic() - start_time
            if timeout >= 0 and elapsed >= timeout:
                raise TimeoutError(
                    f"{self._job_type} experiment {self._experiment!r} did not complete within {timeout} seconds"
                )
            sleep_seconds = _POLL_INTERVAL_SECONDS if timeout < 0 else min(_POLL_INTERVAL_SECONDS, timeout - elapsed)
            time.sleep(sleep_seconds)
        return status

    def abort(self) -> None:
        self._get_client().abort_experiment(self._experiment_name)
        self._aborted = True

    def _get_summary(self) -> dict[str, Any]:
        summary = self._get_client().describe_experiment(self._experiment_name).get("summary")
        return cast(dict[str, Any], json.loads(summary)) if summary is not None else {}

    def _get_status_from_runs(self) -> JobStatus:
        runs = self._get_client().show_runs_in_experiment(experiment_name=self._experiment_name)
        # An executed experiment may not have created its runs yet.
        if not runs or any(self._run_is_running(run) for run in runs):
            return JobStatus.RUNNING
        return JobStatus.COMPLETED

    def _run_is_running(self, run: snowpark.Row) -> bool:
        return self._get_run_status(run) not in _TERMINAL_RUN_STATUSES

    def _get_run_status(self, run: snowpark.Row) -> Any:
        try:
            metadata = json.loads(run.as_dict()[experiment_tracking_sql_client.RUN_METADATA_COL_NAME])
        except (KeyError, TypeError, ValueError):
            metadata = None
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{self._job_type} experiment {self._experiment!r} returned malformed run metadata")
        return metadata.get("status")

    def _get_failed_runs(self, *, run_prefix: str) -> list[str]:
        """Return the sorted names of runs starting with ``run_prefix`` that the backend committed as failed."""
        failed_runs = []
        for run in self._get_client().show_runs_in_experiment(experiment_name=self._experiment_name):
            run_name = run.as_dict().get(experiment_tracking_sql_client.RUN_NAME_COL_NAME)
            if (
                isinstance(run_name, str)
                and run_name.startswith(run_prefix)
                and self._get_run_status(run) == run_metadata.RunStatus.FAILED.value
            ):
                failed_runs.append(run_name)
        return sorted(failed_runs)

    def _get_run_error_message(self, run_name: str) -> str | None:
        """Return the ``error_message`` parameter the backend recorded on a failed run, if any."""
        rows = self._get_client().show_run_parameters_in_experiment(
            experiment_name=self._experiment_name,
            run_name=sql_identifier.SqlIdentifier(run_name),
            like="error_message",
        )
        for row in rows:
            attributes = row.as_dict()
            if attributes.get("name") == "error_message" and attributes.get("value"):
                return str(attributes["value"])
        return None

    def _get_client(self) -> experiment_tracking_sql_client.ExperimentTrackingSQLClient:
        if "_client" not in self.__dict__:
            session = self.session
            if session is None:
                raise RuntimeError("No active Snowpark session available. Please create a session.")
            self._client, self._experiment_name = resolve_client(session, self._experiment)
        return self._client


def create_and_execute_experiment(
    session: snowpark.Session,
    experiment: str | None,
    *,
    name_prefix: str,
    experiment_type: experiment_entity.ExperimentType,
    specification: str,
) -> str:
    """Create a server-executed experiment from a specification and start it.

    Args:
        session: Snowpark session used to create and execute the experiment.
        experiment: Experiment name. A name starting with ``name_prefix`` is generated when omitted.
        name_prefix: Prefix of the generated experiment name.
        experiment_type: Type of the experiment.
        specification: Specification of the experiment.

    Returns:
        Fully qualified name of the created experiment.

    Raises:
        SnowflakeMLException: If the experiment is created but execution does not start. The experiment is left in
            place for inspection or cleanup.
    """
    experiment_name = experiment or experiment_tracking_sql_client.ExperimentTrackingSQLClient.get_tmp_name_with_prefix(
        name_prefix
    )
    client, experiment_identifier = resolve_client(session, experiment_name)
    client.create_experiment(
        experiment_identifier,
        sql_client.CreationMode(),
        experiment_type=experiment_type,
        specification=specification,
    )
    experiment_fqn = client.fully_qualified_object_name(None, None, experiment_identifier)
    try:
        client.execute_experiment(experiment_identifier)
    except snowpark_exceptions.SnowparkSQLException as error:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"Experiment {experiment_fqn!r} was created but execution did not start: {error!r}. "
                "It was left in place for inspection or cleanup."
            ),
        ) from error
    return experiment_fqn


def resolve_client(
    session: snowpark.Session,
    experiment: str,
) -> tuple[experiment_tracking_sql_client.ExperimentTrackingSQLClient, sql_identifier.SqlIdentifier]:
    database, schema, experiment_name = sql_identifier.parse_fully_qualified_name(experiment)
    if database is None:
        current_database = session.get_current_database()
        if current_database is None:
            raise ValueError("A current database or fully qualified experiment name is required")
        database = sql_identifier.SqlIdentifier(current_database)
    if schema is None:
        schema = sql_identifier.SqlIdentifier(session.get_current_schema() or "PUBLIC")
    return (
        experiment_tracking_sql_client.ExperimentTrackingSQLClient(
            session,
            database_name=database,
            schema_name=schema,
        ),
        experiment_name,
    )
