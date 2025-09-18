import json
from typing import Any, Optional

from snowflake import snowpark
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.monitoring._client import model_monitor_sql_client
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.snowpark import session


class ModelMonitorManager:
    """Class to manage internal operations for Model Monitor workflows."""

    def __init__(
        self,
        session: session.Session,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        *,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Opens a ModelMonitorManager for a given database and schema.
        Optionally sets up the schema for Model Monitoring.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The name of the database.
            schema_name: The name of the schema.
            statement_params: Optional set of statement params.
        """
        self._database_name = database_name
        self._schema_name = schema_name
        self.statement_params = statement_params

        self._model_monitor_client = model_monitor_sql_client.ModelMonitorSQLClient(
            session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )

    def _validate_task_from_model_version(
        self,
        model_version: model_version_impl.ModelVersion,
    ) -> type_hints.Task:
        task = model_version.get_model_task()
        if task == type_hints.Task.UNKNOWN:
            raise ValueError("Registry model must be logged with task in order to be monitored.")
        return task

    def _validate_model_function_from_model_version(
        self, function: str, model_version: model_version_impl.ModelVersion
    ) -> None:
        functions = model_version.show_functions()
        for f in functions:
            if f["target_method"] == function:
                return
        existing_target_methods = {f["target_method"] for f in functions}
        raise ValueError(
            f"Function with name {function} does not exist in the given model version. "
            f"Found: {existing_target_methods}."
        )

    def _build_column_list_from_input(self, columns: Optional[list[str]]) -> list[sql_identifier.SqlIdentifier]:
        return [sql_identifier.SqlIdentifier(column_name) for column_name in columns] if columns else []

    def add_monitor(
        self,
        name: str,
        source_config: model_monitor_config.ModelMonitorSourceConfig,
        model_monitor_config: model_monitor_config.ModelMonitorConfig,
    ) -> model_monitor.ModelMonitor:
        """Add a new Model Monitor.

        Args:
            name: Name of Model Monitor to create.
            source_config: Configuration options for the source table used in ModelMonitor.
            model_monitor_config: Configuration options of ModelMonitor.

        Returns:
            The newly added ModelMonitor object.
        """
        warehouse_name_id = sql_identifier.SqlIdentifier(model_monitor_config.background_compute_warehouse_name)
        self._model_monitor_client.validate_monitor_warehouse(warehouse_name_id, statement_params=self.statement_params)
        self._validate_model_function_from_model_version(
            model_monitor_config.model_function_name, model_monitor_config.model_version
        )
        self._validate_task_from_model_version(model_monitor_config.model_version)
        monitor_database_name_id, monitor_schema_name_id, monitor_name_id = sql_identifier.parse_fully_qualified_name(
            name
        )
        source_database_name_id, source_schema_name_id, source_name_id = sql_identifier.parse_fully_qualified_name(
            source_config.source
        )
        baseline_database_name_id, baseline_schema_name_id, baseline_name_id = (
            sql_identifier.parse_fully_qualified_name(source_config.baseline)
            if source_config.baseline
            else (None, None, None)
        )
        model_database_name_id, model_schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(
            model_monitor_config.model_version.fully_qualified_model_name
        )

        prediction_score_columns = self._build_column_list_from_input(source_config.prediction_score_columns)
        prediction_class_columns = self._build_column_list_from_input(source_config.prediction_class_columns)
        actual_score_columns = self._build_column_list_from_input(source_config.actual_score_columns)
        actual_class_columns = self._build_column_list_from_input(source_config.actual_class_columns)
        segment_columns = self._build_column_list_from_input(source_config.segment_columns)
        custom_metric_columns = self._build_column_list_from_input(source_config.custom_metric_columns)

        id_columns = [sql_identifier.SqlIdentifier(column_name) for column_name in source_config.id_columns]
        ts_column = sql_identifier.SqlIdentifier(source_config.timestamp_column)

        # Validate source table
        self._model_monitor_client.validate_source(
            source_database=source_database_name_id,
            source_schema=source_schema_name_id,
            source=source_name_id,
            timestamp_column=ts_column,
            prediction_score_columns=prediction_score_columns,
            prediction_class_columns=prediction_class_columns,
            actual_score_columns=actual_score_columns,
            actual_class_columns=actual_class_columns,
            id_columns=id_columns,
            segment_columns=segment_columns,
            custom_metric_columns=custom_metric_columns,
        )

        self._model_monitor_client.create_model_monitor(
            monitor_database=monitor_database_name_id,
            monitor_schema=monitor_schema_name_id,
            monitor_name=monitor_name_id,
            source_database=source_database_name_id,
            source_schema=source_schema_name_id,
            source=source_name_id,
            model_database=model_database_name_id,
            model_schema=model_schema_name_id,
            model_name=model_name_id,
            version_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.version_name),
            function_name=model_monitor_config.model_function_name,
            warehouse_name=warehouse_name_id,
            timestamp_column=ts_column,
            id_columns=id_columns,
            prediction_score_columns=prediction_score_columns,
            prediction_class_columns=prediction_class_columns,
            actual_score_columns=actual_score_columns,
            actual_class_columns=actual_class_columns,
            segment_columns=segment_columns,
            custom_metric_columns=custom_metric_columns,
            refresh_interval=model_monitor_config.refresh_interval,
            aggregation_window=model_monitor_config.aggregation_window,
            baseline_database=baseline_database_name_id,
            baseline_schema=baseline_schema_name_id,
            baseline=baseline_name_id,
            statement_params=self.statement_params,
        )
        return model_monitor.ModelMonitor._ref(
            model_monitor_client=self._model_monitor_client,
            name=monitor_name_id,
        )

    def get_monitor_by_model_version(
        self, model_version: model_version_impl.ModelVersion
    ) -> model_monitor.ModelMonitor:
        """Get a Model Monitor by Model Version.

        Args:
            model_version: ModelVersion to retrieve Model Monitor for.

        Returns:
            The fetched ModelMonitor.

        Raises:
            ValueError: If model monitor is not found.
        """
        rows = self._model_monitor_client.show_model_monitors(statement_params=self.statement_params)

        def model_match_fn(model_details: dict[str, str]) -> bool:
            return (
                model_details[model_monitor_sql_client.MODEL_JSON_MODEL_NAME_FIELD] == model_version.model_name
                and model_details[model_monitor_sql_client.MODEL_JSON_VERSION_NAME_FIELD] == model_version.version_name
            )

        rows = [row for row in rows if model_match_fn(json.loads(row[model_monitor_sql_client.MODEL_JSON_COL_NAME]))]
        if len(rows) == 0:
            raise ValueError("Unable to find model monitor for the given model version.")
        if len(rows) > 1:
            raise ValueError("Found multiple model monitors for the given model version.")

        return model_monitor.ModelMonitor._ref(
            model_monitor_client=self._model_monitor_client,
            name=sql_identifier.SqlIdentifier(rows[0]["name"]),
        )

    def get_monitor(self, name: str) -> model_monitor.ModelMonitor:
        """Get a Model Monitor from the Registry

        Args:
            name: Name of Model Monitor to retrieve.

        Raises:
            ValueError: If model monitor is not found.

        Returns:
            The fetched ModelMonitor.
        """
        database_name_id, schema_name_id, monitor_name_id = sql_identifier.parse_fully_qualified_name(name)

        if not self._model_monitor_client.validate_existence_by_name(
            database_name=database_name_id,
            schema_name=schema_name_id,
            monitor_name=monitor_name_id,
            statement_params=self.statement_params,
        ):
            raise ValueError(f"Unable to find model monitor '{name}'")
        return model_monitor.ModelMonitor._ref(
            model_monitor_client=self._model_monitor_client,
            name=monitor_name_id,
        )

    def show_model_monitors(self) -> list[snowpark.Row]:
        """Show all model monitors in the registry.

        Returns:
            List of snowpark.Row containing metadata for each model monitor.
        """
        return self._model_monitor_client.show_model_monitors(statement_params=self.statement_params)

    def delete_monitor(self, name: str) -> None:
        """Delete a Model Monitor from the Registry

        Args:
            name: Name of the Model Monitor to delete.
        """
        database_name_id, schema_name_id, monitor_name_id = sql_identifier.parse_fully_qualified_name(name)
        self._model_monitor_client.drop_model_monitor(
            database_name=database_name_id,
            schema_name=schema_name_id,
            monitor_name=monitor_name_id,
            statement_params=self.statement_params,
        )
