from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring._client import model_monitor_sql_client


class ModelMonitor:
    """Class to manage instrumentation of Model Monitoring and Observability"""

    name: sql_identifier.SqlIdentifier
    _model_monitor_client: model_monitor_sql_client.ModelMonitorSQLClient

    def __init__(self) -> None:
        raise RuntimeError("Model Monitor's initializer is not meant to be used.")

    @classmethod
    def _ref(
        cls,
        model_monitor_client: model_monitor_sql_client.ModelMonitorSQLClient,
        name: sql_identifier.SqlIdentifier,
    ) -> "ModelMonitor":
        self: "ModelMonitor" = object.__new__(cls)
        self.name = name
        self._model_monitor_client = model_monitor_client
        return self

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def suspend(self) -> None:
        """Suspend the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        self._model_monitor_client.suspend_monitor(self.name, statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def resume(self) -> None:
        """Resume the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        self._model_monitor_client.resume_monitor(self.name, statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def add_segment_column(self, segment_column: str) -> None:
        """Add a segment column to the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        segment_column_id = sql_identifier.SqlIdentifier(segment_column)
        self._model_monitor_client.add_segment_column(self.name, segment_column_id, statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def drop_segment_column(self, segment_column: str) -> None:
        """Drop a segment column from the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        segment_column_id = sql_identifier.SqlIdentifier(segment_column)
        self._model_monitor_client.drop_segment_column(self.name, segment_column_id, statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def add_custom_metric_column(self, custom_metric_column: str) -> None:
        """Add a custom metric column to the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        custom_metric_column_identifier = sql_identifier.SqlIdentifier(custom_metric_column)
        self._model_monitor_client.add_custom_metric_column(
            self.name, custom_metric_column_identifier, statement_params=statement_params
        )

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    def drop_custom_metric_column(self, custom_metric_column: str) -> None:
        """Drop a custom metric column from the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        custom_metric_column_identifier = sql_identifier.SqlIdentifier(custom_metric_column)
        self._model_monitor_client.drop_custom_metric_column(
            self.name, custom_metric_column_identifier, statement_params=statement_params
        )
