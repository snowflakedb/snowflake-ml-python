from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring._client import model_monitor_sql_client


class ModelMonitor:
    """Class to manage instrumentation of Model Monitoring and Observability"""

    name: sql_identifier.SqlIdentifier
    _model_monitor_client: model_monitor_sql_client.ModelMonitorSQLClient

    statement_params = telemetry.get_statement_params(
        telemetry.TelemetryProject.MLOPS.value,
        telemetry.TelemetrySubProject.MONITORING.value,
    )

    def __init__(self) -> None:
        raise RuntimeError("ModelMonitor's initializer is not meant to be used.")

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

    def suspend(self) -> None:
        """Suspend pipeline for ModelMonitor"""
        self._model_monitor_client.suspend_monitor(self.name, statement_params=self.statement_params)

    def resume(self) -> None:
        """Resume pipeline for ModelMonitor"""
        self._model_monitor_client.resume_monitor(self.name, statement_params=self.statement_params)
