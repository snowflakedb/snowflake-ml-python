from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring import model_monitor_version
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
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
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
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
    def resume(self) -> None:
        """Resume the Model Monitor"""
        statement_params = telemetry.get_statement_params(
            telemetry.TelemetryProject.MLOPS.value,
            telemetry.TelemetrySubProject.MONITORING.value,
        )
        self._model_monitor_client.resume_monitor(self.name, statement_params=statement_params)
