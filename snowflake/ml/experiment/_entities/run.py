import types
import warnings
from typing import TYPE_CHECKING

from snowflake import snowpark
from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.experiment import _experiment_info as experiment_info
from snowflake.ml.experiment._entities import run_metadata
from snowflake.ml.lineage import lineage_node

if TYPE_CHECKING:
    from snowflake.ml.experiment import experiment_tracking

METADATA_SIZE_WARNING_MESSAGE = "It is likely that no further metrics or parameters will be logged for this run."


class Run(lineage_node.LineageNode):
    """An experiment run.

    When created by ``ExperimentTracking.start_run``, the run is an active context manager that
    patches model logging and ends the run on exit. When returned by the lineage system
    (``model_version.lineage(direction="upstream", domain_filter={"experiment"})``), it is a
    read-only handle exposing the identifiers needed to look up metrics and parameters.
    """

    def __init__(
        self,
        experiment_tracking: "experiment_tracking.ExperimentTracking | None" = None,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        session: snowpark.Session | None = None,
        database_name: str | None = None,
        schema_name: str | None = None,
    ) -> None:
        self._experiment_tracking = experiment_tracking
        self.experiment_name = experiment_name
        self.name = run_name

        # Whether we've already shown the user a warning about exceeding the run metadata size limit.
        self._warned_about_metadata_size = False

        if experiment_tracking is not None:
            _session = experiment_tracking._session
            self.database_name = str(experiment_tracking._database_name)
            self.schema_name = str(experiment_tracking._schema_name)
        else:
            if session is None or database_name is None or schema_name is None:
                raise ValueError(
                    "session, database_name, and schema_name are required when experiment_tracking is None"
                )
            _session = session
            self.database_name = database_name
            self.schema_name = schema_name

        self._patcher = experiment_info.ExperimentInfoPatcher(
            experiment_info=self._get_experiment_info(),
        )

        # A run is modelled in lineage as a version of its experiment, so the node name is the experiment's.
        super().__init__(
            session=_session,
            name=self._fully_qualified_experiment_name(),
            domain="experiment",
            version=str(run_name),
        )

    def __enter__(self) -> "Run":
        self._patcher.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self._patcher.__exit__(exc_type, exc_value, tb)
        if self._experiment_tracking is not None and self._experiment_tracking._run is self:
            if exc_type is not None:
                self._experiment_tracking._try_log_exception(exc_type, exc_value, tb)
                self._experiment_tracking.end_run(status=run_metadata.RunStatus.FAILED.value)
            else:
                self._experiment_tracking.end_run()

    def _fully_qualified_experiment_name(self) -> str:
        return sql_identifier.get_fully_qualified_name(self.database_name, self.schema_name, self.experiment_name)

    def _get_experiment_info(self) -> experiment_info.ExperimentInfo:
        return experiment_info.ExperimentInfo(
            fully_qualified_name=self._fully_qualified_experiment_name(),
            run_name=self.name.identifier(),
        )

    def _handle_run_metadata_size_limit(self, sql_error_msg: str) -> None:
        """Report that run metadata exceeded the size limit.

        Args:
            sql_error_msg: The error message returned by Snowflake.

        Raises:
            RuntimeWarning: If the experiment tracking instance was created with warnings_as_errors=True.
        """
        message = f"{sql_error_msg}. {METADATA_SIZE_WARNING_MESSAGE}"
        if self._experiment_tracking is not None and self._experiment_tracking._warnings_as_errors:
            raise RuntimeWarning(message)
        # Once the limit is hit every subsequent log attempt fails the same way, so warn only once.
        if not self._warned_about_metadata_size:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._warned_about_metadata_size = True

    @staticmethod
    def _load_from_lineage_node(
        session: snowpark.Session,
        name: str,
        version: str,
    ) -> "Run":
        database_name, schema_name, experiment_name = identifier.parse_schema_level_object_identifier(name)
        return Run(
            experiment_name=sql_identifier.SqlIdentifier(experiment_name),
            run_name=sql_identifier.SqlIdentifier(version),
            session=session,
            database_name=database_name,
            schema_name=schema_name,
        )


lineage_node.DOMAIN_LINEAGE_REGISTRY["experiment"] = Run
