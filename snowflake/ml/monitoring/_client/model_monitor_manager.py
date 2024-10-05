from typing import Any, Dict, List, Optional

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import db_utils, sql_identifier
from snowflake.ml.model import type_hints
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.monitoring._client import model_monitor, monitor_sql_client
from snowflake.ml.monitoring.entities import (
    model_monitor_config,
    model_monitor_interval,
)
from snowflake.snowpark import session


def _validate_name_constraints(model_version: model_version_impl.ModelVersion) -> None:
    system_table_prefixes = [
        monitor_sql_client._SNOWML_MONITORING_TABLE_NAME_PREFIX,
        monitor_sql_client._SNOWML_MONITORING_ACCURACY_TABLE_NAME_PREFIX,
    ]

    max_allowed_model_name_and_version_length = (
        db_utils.MAX_IDENTIFIER_LENGTH - max(len(prefix) for prefix in system_table_prefixes) - 1
    )  # -1 includes '_' between model_name + model_version
    if len(model_version.model_name) + len(model_version.version_name) > max_allowed_model_name_and_version_length:
        error_msg = f"Model name and version name exceeds maximum length of {max_allowed_model_name_and_version_length}"
        raise ValueError(error_msg)


class ModelMonitorManager:
    """Class to manage internal operations for Model Monitor workflows."""  # TODO: Move to Registry.

    @staticmethod
    def setup(session: session.Session, database_name: str, schema_name: str) -> None:
        """Static method to set up schema for Model Monitoring resources.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The name of the database. If None, the current database of the session
                will be used. Defaults to None.
            schema_name: The name of the schema. If None, the current schema of the session
                will be used. If there is no active schema, the PUBLIC schema will be used. Defaults to None.
        """
        statement_params = telemetry.get_statement_params(
            project=telemetry.TelemetryProject.MLOPS.value,
            subproject=telemetry.TelemetrySubProject.MONITORING.value,
        )
        database_name_id = sql_identifier.SqlIdentifier(database_name)
        schema_name_id = sql_identifier.SqlIdentifier(schema_name)
        monitor_sql_client._ModelMonitorSQLClient.initialize_monitoring_schema(
            session, database_name_id, schema_name_id, statement_params=statement_params
        )

    def _fetch_task_from_model_version(
        self,
        model_version: model_version_impl.ModelVersion,
    ) -> type_hints.Task:
        task = model_version.get_model_task()
        if task == type_hints.Task.UNKNOWN:
            raise ValueError("Registry model must be logged with task in order to be monitored.")
        return task

    def __init__(
        self,
        session: session.Session,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        *,
        create_if_not_exists: bool = False,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Opens a ModelMonitorManager for a given database and schema.
        Optionally sets up the schema for Model Monitoring.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The name of the database.
            schema_name: The name of the schema.
            create_if_not_exists: Flag whether to initialize resources in the schema needed for Model Monitoring.
            statement_params: Optional set of statement params.

        Raises:
            ValueError: When there is no specified or active database in the session.
        """
        self._database_name = database_name
        self._schema_name = schema_name
        self.statement_params = statement_params
        self._model_monitor_client = monitor_sql_client._ModelMonitorSQLClient(
            session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )
        if create_if_not_exists:
            monitor_sql_client._ModelMonitorSQLClient.initialize_monitoring_schema(
                session, self._database_name, self._schema_name, self.statement_params
            )
        elif not self._model_monitor_client._validate_is_initialized():
            raise ValueError(
                "Monitoring has not been setup. Set create_if_not_exists or call ModelMonitorManager.setup"
            )

    def _get_and_validate_model_function_from_model_version(
        self, function: str, model_version: model_version_impl.ModelVersion
    ) -> model_manifest_schema.ModelFunctionInfo:
        functions = model_version.show_functions()
        for f in functions:
            if f["target_method"] == function:
                return f
        existing_target_methods = {f["target_method"] for f in functions}
        raise ValueError(
            f"Function with name {function} does not exist in the given model version. "
            f"Found: {existing_target_methods}."
        )

    def _validate_monitor_config_or_raise(
        self,
        table_config: model_monitor_config.ModelMonitorTableConfig,
        model_monitor_config: model_monitor_config.ModelMonitorConfig,
    ) -> None:
        """Validate provided config for model monitor.

        Args:
            table_config: Config for model monitor tables.
            model_monitor_config: Config for ModelMonitor.

        Raises:
            ValueError: If warehouse provided does not exist.
        """

        # Validate naming will not exceed 255 chars
        _validate_name_constraints(model_monitor_config.model_version)

        if len(table_config.prediction_columns) != len(table_config.label_columns):
            raise ValueError("Prediction and Label column names must be of the same length.")
        # output and ground cols are list to keep interface extensible.
        # for prpr only one label and one output col will be supported
        if len(table_config.prediction_columns) != 1 or len(table_config.label_columns) != 1:
            raise ValueError("Multiple Output columns are not supported in monitoring")

        # Validate warehouse exists.
        warehouse_name_id = sql_identifier.SqlIdentifier(model_monitor_config.background_compute_warehouse_name)
        self._model_monitor_client.validate_monitor_warehouse(warehouse_name_id, statement_params=self.statement_params)

        # Validate refresh interval.
        try:
            num_units, time_units = model_monitor_config.refresh_interval.strip().split(" ")
            int(num_units)  # try to cast
            if time_units.lower() not in {"seconds", "minutes", "hours", "days"}:
                raise ValueError(
                    """Invalid time unit in refresh interval. Provide '<num> <seconds | minutes | hours | days>'.
See https://docs.snowflake.com/en/sql-reference/sql/create-dynamic-table#required-parameters for more info."""
                )
        except Exception as e:  # TODO: Link to DT page.
            raise ValueError(
                f"""Failed to parse refresh interval with exception {e}.
                Provide '<num> <seconds | minutes | hours | days>'.
See https://docs.snowflake.com/en/sql-reference/sql/create-dynamic-table#required-parameters for more info."""
            )

    def add_monitor(
        self,
        name: str,
        table_config: model_monitor_config.ModelMonitorTableConfig,
        model_monitor_config: model_monitor_config.ModelMonitorConfig,
        *,
        add_dashboard_udtfs: bool = False,
    ) -> model_monitor.ModelMonitor:
        """Add a new Model Monitor.

        Args:
            name: Name of Model Monitor to create.
            table_config: Configuration options for the source table used in ModelMonitor.
            model_monitor_config: Configuration options of ModelMonitor.
            add_dashboard_udtfs: Add UDTFs useful for creating a dashboard.

        Returns:
            The newly added ModelMonitor object.
        """
        # Validates configuration or raise.
        self._validate_monitor_config_or_raise(table_config, model_monitor_config)
        model_function = self._get_and_validate_model_function_from_model_version(
            model_monitor_config.model_function_name, model_monitor_config.model_version
        )
        monitor_refresh_interval = model_monitor_interval.ModelMonitorRefreshInterval(
            model_monitor_config.refresh_interval
        )
        name_id = sql_identifier.SqlIdentifier(name)
        source_table_name_id = sql_identifier.SqlIdentifier(table_config.source_table)
        prediction_columns = [
            sql_identifier.SqlIdentifier(column_name) for column_name in table_config.prediction_columns
        ]
        label_columns = [sql_identifier.SqlIdentifier(column_name) for column_name in table_config.label_columns]
        id_columns = [sql_identifier.SqlIdentifier(column_name) for column_name in table_config.id_columns]
        ts_column = sql_identifier.SqlIdentifier(table_config.timestamp_column)

        # Validate source table
        self._model_monitor_client.validate_source_table(
            source_table_name=source_table_name_id,
            timestamp_column=ts_column,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
            id_columns=id_columns,
            model_function=model_function,
        )

        task = self._fetch_task_from_model_version(model_version=model_monitor_config.model_version)
        score_type = self._model_monitor_client.get_score_type(task, source_table_name_id, prediction_columns)

        # Insert monitoring metadata for new model version.
        self._model_monitor_client.create_monitor_on_model_version(
            monitor_name=name_id,
            source_table_name=source_table_name_id,
            fully_qualified_model_name=model_monitor_config.model_version.fully_qualified_model_name,
            version_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.version_name),
            function_name=model_monitor_config.model_function_name,
            timestamp_column=ts_column,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
            id_columns=id_columns,
            task=task,
            statement_params=self.statement_params,
        )

        # Create Dynamic tables for model monitor.
        self._model_monitor_client.create_dynamic_tables_for_monitor(
            model_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.model_name),
            model_version_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.version_name),
            task=task,
            source_table_name=source_table_name_id,
            refresh_interval=monitor_refresh_interval,
            aggregation_window=model_monitor_config.aggregation_window,
            warehouse_name=sql_identifier.SqlIdentifier(model_monitor_config.background_compute_warehouse_name),
            timestamp_column=sql_identifier.SqlIdentifier(table_config.timestamp_column),
            id_columns=id_columns,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
            score_type=score_type,
        )

        # Initialize baseline table.
        self._model_monitor_client.initialize_baseline_table(
            model_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.model_name),
            version_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.version_name),
            source_table_name=table_config.source_table,
            columns_to_drop=[ts_column, *id_columns],
            statement_params=self.statement_params,
        )

        # Add udtfs helpful for dashboard queries.
        # TODO(apgupta) Make this true by default.
        if add_dashboard_udtfs:
            self._model_monitor_client.add_dashboard_udtfs(
                name_id,
                model_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.model_name),
                model_version_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.version_name),
                task=task,
                score_type=score_type,
                output_columns=prediction_columns,
                ground_truth_columns=label_columns,
            )

        return model_monitor.ModelMonitor._ref(
            model_monitor_client=self._model_monitor_client,
            name=name_id,
            fully_qualified_model_name=model_monitor_config.model_version.fully_qualified_model_name,
            version_name=sql_identifier.SqlIdentifier(model_monitor_config.model_version.version_name),
            function_name=sql_identifier.SqlIdentifier(model_monitor_config.model_function_name),
            prediction_columns=prediction_columns,
            label_columns=label_columns,
        )

    def get_monitor_by_model_version(
        self, model_version: model_version_impl.ModelVersion
    ) -> model_monitor.ModelMonitor:
        fq_model_name = model_version.fully_qualified_model_name
        version_name = sql_identifier.SqlIdentifier(model_version.version_name)
        if self._model_monitor_client.validate_existence(fq_model_name, version_name, self.statement_params):
            model_db, model_schema, model_name = sql_identifier.parse_fully_qualified_name(fq_model_name)
            if model_db is None or model_schema is None:
                raise ValueError("Failed to parse model name")

            model_monitor_params: monitor_sql_client._ModelMonitorParams = (
                self._model_monitor_client.get_model_monitor_by_model_version(
                    model_db=model_db,
                    model_schema=model_schema,
                    model_name=model_name,
                    version_name=version_name,
                    statement_params=self.statement_params,
                )
            )
            return model_monitor.ModelMonitor._ref(
                model_monitor_client=self._model_monitor_client,
                name=sql_identifier.SqlIdentifier(model_monitor_params["monitor_name"]),
                fully_qualified_model_name=fq_model_name,
                version_name=version_name,
                function_name=sql_identifier.SqlIdentifier(model_monitor_params["function_name"]),
                prediction_columns=model_monitor_params["prediction_columns"],
                label_columns=model_monitor_params["label_columns"],
            )

        else:
            raise ValueError(
                f"ModelMonitor not found for model version {model_version.model_name} - {model_version.version_name}"
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
        name_id = sql_identifier.SqlIdentifier(name)

        if not self._model_monitor_client.validate_existence_by_name(
            monitor_name=name_id,
            statement_params=self.statement_params,
        ):
            raise ValueError(f"Unable to find model monitor '{name}'")
        model_monitor_params: monitor_sql_client._ModelMonitorParams = (
            self._model_monitor_client.get_model_monitor_by_name(name_id, statement_params=self.statement_params)
        )

        return model_monitor.ModelMonitor._ref(
            model_monitor_client=self._model_monitor_client,
            name=name_id,
            fully_qualified_model_name=model_monitor_params["fully_qualified_model_name"],
            version_name=sql_identifier.SqlIdentifier(model_monitor_params["version_name"]),
            function_name=sql_identifier.SqlIdentifier(model_monitor_params["function_name"]),
            prediction_columns=model_monitor_params["prediction_columns"],
            label_columns=model_monitor_params["label_columns"],
        )

    def show_model_monitors(self) -> List[snowpark.Row]:
        """Show all model monitors in the registry.

        Returns:
            List of snowpark.Row containing metadata for each model monitor.
        """
        return self._model_monitor_client.get_all_model_monitor_metadata()

    def delete_monitor(self, name: str) -> None:
        """Delete a Model Monitor from the Registry

        Args:
            name: Name of the Model Monitor to delete.
        """
        name_id = sql_identifier.SqlIdentifier(name)
        monitor_params = self._model_monitor_client.get_model_monitor_by_name(name_id)
        _, _, model = sql_identifier.parse_fully_qualified_name(monitor_params["fully_qualified_model_name"])
        version = sql_identifier.SqlIdentifier(monitor_params["version_name"])
        self._model_monitor_client.delete_monitor_metadata(name_id)
        self._model_monitor_client.delete_baseline_table(model, version)
        self._model_monitor_client.delete_dynamic_tables(model, version)
