import types
from typing import TYPE_CHECKING, Optional

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment import _experiment_info as experiment_info

if TYPE_CHECKING:
    from snowflake.ml.experiment import experiment_tracking


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
