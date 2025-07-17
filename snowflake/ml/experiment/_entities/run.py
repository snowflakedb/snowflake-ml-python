import json
import types
from typing import TYPE_CHECKING, Optional

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment import _experiment_info as experiment_info
from snowflake.ml.experiment._client import experiment_tracking_sql_client
from snowflake.ml.experiment._entities import run_metadata

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

    def _get_metadata(
        self,
    ) -> run_metadata.RunMetadata:
        runs = self._experiment_tracking._sql_client.show_runs_in_experiment(
            experiment_name=self.experiment_name, like=str(self.name)
        )
        if not runs:
            raise RuntimeError(f"Run {self.name} not found in experiment {self.experiment_name}.")
        return run_metadata.RunMetadata.from_dict(
            json.loads(runs[0][experiment_tracking_sql_client.ExperimentTrackingSQLClient.RUN_METADATA_COL_NAME])
        )

    def _get_experiment_info(self) -> experiment_info.ExperimentInfo:
        return experiment_info.ExperimentInfo(
            fully_qualified_name=self._experiment_tracking._sql_client.fully_qualified_object_name(
                self._experiment_tracking._database_name, self._experiment_tracking._schema_name, self.experiment_name
            ),
            run_name=self.name.identifier(),
        )
