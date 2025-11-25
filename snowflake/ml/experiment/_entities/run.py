import types
import warnings
from typing import TYPE_CHECKING, Optional

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment import _experiment_info as experiment_info

if TYPE_CHECKING:
    from snowflake.ml.experiment import experiment_tracking

METADATA_SIZE_WARNING_MESSAGE = "It is likely that no further metrics or parameters will be logged for this run."


class Run:
    def __init__(
        self,
        experiment_tracking: "experiment_tracking.ExperimentTracking",
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._experiment_tracking = experiment_tracking
        self.experiment_name = experiment_name
        self.name = run_name

        # Whether we've already shown the user a warning about exceeding the run metadata size limit.
        self._warned_about_metadata_size = False

        self._patcher = experiment_info.ExperimentInfoPatcher(
            experiment_info=self._get_experiment_info(),
        )

    def __enter__(self) -> "Run":
        self._patcher.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        self._patcher.__exit__(exc_type, exc_value, traceback)
        if self._experiment_tracking._run is self:
            self._experiment_tracking.end_run()

    def _get_experiment_info(self) -> experiment_info.ExperimentInfo:
        return experiment_info.ExperimentInfo(
            fully_qualified_name=self._experiment_tracking._sql_client.fully_qualified_object_name(
                self._experiment_tracking._database_name, self._experiment_tracking._schema_name, self.experiment_name
            ),
            run_name=self.name.identifier(),
        )

    def _warn_about_run_metadata_size(self, sql_error_msg: str) -> None:
        if not self._warned_about_metadata_size:
            warnings.warn(
                f"{sql_error_msg}. {METADATA_SIZE_WARNING_MESSAGE}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_about_metadata_size = True
