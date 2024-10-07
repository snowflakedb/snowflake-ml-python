from typing import List, Union

import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring._client import monitor_sql_client


class ModelMonitor:
    """Class to manage instrumentation of Model Monitoring and Observability"""

    name: sql_identifier.SqlIdentifier
    _model_monitor_client: monitor_sql_client._ModelMonitorSQLClient
    _fully_qualified_model_name: str
    _version_name: sql_identifier.SqlIdentifier
    _function_name: sql_identifier.SqlIdentifier
    _prediction_columns: List[sql_identifier.SqlIdentifier]
    _label_columns: List[sql_identifier.SqlIdentifier]

    def __init__(self) -> None:
        raise RuntimeError("ModelMonitor's initializer is not meant to be used.")

    @classmethod
    def _ref(
        cls,
        model_monitor_client: monitor_sql_client._ModelMonitorSQLClient,
        name: sql_identifier.SqlIdentifier,
        *,
        fully_qualified_model_name: str,
        version_name: sql_identifier.SqlIdentifier,
        function_name: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
    ) -> "ModelMonitor":
        self: "ModelMonitor" = object.__new__(cls)
        self.name = name
        self._model_monitor_client = model_monitor_client
        self._fully_qualified_model_name = fully_qualified_model_name
        self._version_name = version_name
        self._function_name = function_name
        self._prediction_columns = prediction_columns
        self._label_columns = label_columns
        return self

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def set_baseline(self, baseline_df: Union[pd.DataFrame, snowpark.DataFrame]) -> None:
        """
        The baseline dataframe is compared with the monitored data once monitoring is enabled.
        The columns of the dataframe should match the columns of the source table that the
        ModelMonitor was configured with. Calling this method overwrites any existing baseline split data.

        Args:
            baseline_df: Snowpark dataframe containing baseline data.

        Raises:
            ValueError: baseline_df does not contain prediction or label columns
        """
        statement_params = telemetry.get_statement_params(
            project=telemetry.TelemetryProject.MLOPS.value,
            subproject=telemetry.TelemetrySubProject.MONITORING.value,
        )

        if isinstance(baseline_df, pd.DataFrame):
            baseline_df = self._model_monitor_client._sql_client._session.create_dataframe(baseline_df)

        column_names_identifiers: List[sql_identifier.SqlIdentifier] = [
            sql_identifier.SqlIdentifier(column_name) for column_name in baseline_df.columns
        ]
        prediction_cols_not_found = any(
            [prediction_col not in column_names_identifiers for prediction_col in self._prediction_columns]
        )
        label_cols_not_found = any(
            [label_col.identifier() not in column_names_identifiers for label_col in self._label_columns]
        )

        if prediction_cols_not_found:
            raise ValueError(
                "Specified prediction columns were not found in the baseline dataframe. "
                f"Columns provided were: {column_names_identifiers}. "
                f"Configured prediction columns were: {self._prediction_columns}."
            )
        if label_cols_not_found:
            raise ValueError(
                "Specified label columns were not found in the baseline dataframe."
                f"Columns provided in the baseline dataframe were: {column_names_identifiers}."
                f"Configured label columns were: {self._label_columns}."
            )

        # Create the table by materializing the df
        self._model_monitor_client.materialize_baseline_dataframe(
            baseline_df,
            self._fully_qualified_model_name,
            self._version_name,
            statement_params=statement_params,
        )

    def suspend(self) -> None:
        """Suspend pipeline for ModelMonitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        _, _, model_name = sql_identifier.parse_fully_qualified_name(self._fully_qualified_model_name)
        self._model_monitor_client.suspend_monitor_dynamic_tables(
            model_name=model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    def resume(self) -> None:
        """Resume pipeline for ModelMonitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        _, _, model_name = sql_identifier.parse_fully_qualified_name(self._fully_qualified_model_name)
        self._model_monitor_client.resume_monitor_dynamic_tables(
            model_name=model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )
