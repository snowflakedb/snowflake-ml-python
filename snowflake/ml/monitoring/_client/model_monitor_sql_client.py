from typing import Any, Dict, List, Mapping, Optional

from snowflake import snowpark
from snowflake.ml._internal.utils import (
    db_utils,
    query_result_checker,
    sql_identifier,
    table_manager,
)
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import session, types

MODEL_JSON_COL_NAME = "model"
MODEL_JSON_MODEL_NAME_FIELD = "model_name"
MODEL_JSON_VERSION_NAME_FIELD = "version_name"


def _build_sql_list_from_columns(columns: List[sql_identifier.SqlIdentifier]) -> str:
    sql_list = ", ".join([f"'{column}'" for column in columns])
    return f"({sql_list})"


class ModelMonitorSQLClient:
    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        """Client to manage monitoring metadata persisted in SNOWML_OBSERVABILITY.METADATA schema.

        Args:
            session: Active snowpark session.
            database_name: Name of the Database where monitoring resources are provisioned.
            schema_name: Name of the Schema where monitoring resources are provisioned.
        """
        self._sql_client = _base._BaseSQLClient(session, database_name=database_name, schema_name=schema_name)
        self._database_name = database_name
        self._schema_name = schema_name

    def _infer_qualified_schema(
        self, database_name: Optional[sql_identifier.SqlIdentifier], schema_name: Optional[sql_identifier.SqlIdentifier]
    ) -> str:
        return f"""{database_name or self._database_name}.{schema_name or self._schema_name}"""

    def create_model_monitor(
        self,
        *,
        monitor_database: Optional[sql_identifier.SqlIdentifier],
        monitor_schema: Optional[sql_identifier.SqlIdentifier],
        monitor_name: sql_identifier.SqlIdentifier,
        source_database: Optional[sql_identifier.SqlIdentifier],
        source_schema: Optional[sql_identifier.SqlIdentifier],
        source: sql_identifier.SqlIdentifier,
        model_database: Optional[sql_identifier.SqlIdentifier],
        model_schema: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        function_name: str,
        warehouse_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        id_columns: List[sql_identifier.SqlIdentifier],
        prediction_score_columns: List[sql_identifier.SqlIdentifier],
        prediction_class_columns: List[sql_identifier.SqlIdentifier],
        actual_score_columns: List[sql_identifier.SqlIdentifier],
        actual_class_columns: List[sql_identifier.SqlIdentifier],
        refresh_interval: str,
        aggregation_window: str,
        baseline_database: Optional[sql_identifier.SqlIdentifier] = None,
        baseline_schema: Optional[sql_identifier.SqlIdentifier] = None,
        baseline: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        baseline_sql = ""
        if baseline:
            baseline_sql = f"""BASELINE={self._infer_qualified_schema(baseline_database, baseline_schema)}.{baseline}"""
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"""
            CREATE MODEL MONITOR {self._infer_qualified_schema(monitor_database, monitor_schema)}.{monitor_name}
                WITH
                    MODEL={self._infer_qualified_schema(model_database, model_schema)}.{model_name}
                    VERSION='{version_name}'
                    FUNCTION='{function_name}'
                    WAREHOUSE='{warehouse_name}'
                    SOURCE={self._infer_qualified_schema(source_database, source_schema)}.{source}
                    ID_COLUMNS={_build_sql_list_from_columns(id_columns)}
                    PREDICTION_SCORE_COLUMNS={_build_sql_list_from_columns(prediction_score_columns)}
                    PREDICTION_CLASS_COLUMNS={_build_sql_list_from_columns(prediction_class_columns)}
                    ACTUAL_SCORE_COLUMNS={_build_sql_list_from_columns(actual_score_columns)}
                    ACTUAL_CLASS_COLUMNS={_build_sql_list_from_columns(actual_class_columns)}
                    TIMESTAMP_COLUMN='{timestamp_column}'
                    REFRESH_INTERVAL='{refresh_interval}'
                    AGGREGATION_WINDOW='{aggregation_window}'
                    {baseline_sql}""",
            statement_params=statement_params,
        ).has_column("status").has_dimensions(1, 1).validate()

    def drop_model_monitor(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier] = None,
        schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        search_database_name = database_name or self._database_name
        search_schema_name = schema_name or self._schema_name
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"DROP MODEL MONITOR {search_database_name}.{search_schema_name}.{monitor_name}",
            statement_params=statement_params,
        ).validate()

    def show_model_monitors(
        self,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[snowpark.Row]:
        fully_qualified_schema_name = ".".join([self._database_name.identifier(), self._schema_name.identifier()])
        return (
            query_result_checker.SqlResultValidator(
                self._sql_client._session,
                f"SHOW MODEL MONITORS IN {fully_qualified_schema_name}",
                statement_params=statement_params,
            )
            .has_column("name", allow_empty=True)
            .validate()
        )

    def validate_existence_by_name(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier] = None,
        schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        search_database_name = database_name or self._database_name
        search_schema_name = schema_name or self._schema_name
        res = (
            query_result_checker.SqlResultValidator(
                self._sql_client._session,
                f"SHOW MODEL MONITORS LIKE '{monitor_name.resolved()}' IN {search_database_name}.{search_schema_name}",
                statement_params=statement_params,
            )
            .has_column("name", allow_empty=True)
            .validate()
        )
        return len(res) == 1

    def validate_monitor_warehouse(
        self,
        warehouse_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate warehouse provided for monitoring exists.

        Args:
            warehouse_name: Warehouse name
            statement_params: Optional set of statement params to include in queries.

        Raises:
            ValueError: If warehouse does not exist.
        """
        if not db_utils.db_object_exists(
            session=self._sql_client._session,
            object_type=db_utils.SnowflakeDbObjectType.WAREHOUSE,
            object_name=warehouse_name,
            statement_params=statement_params,
        ):
            raise ValueError(f"Warehouse '{warehouse_name}' not found.")

    def _validate_columns_exist_in_source(
        self,
        *,
        source_column_schema: Mapping[str, types.DataType],
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_score_columns: List[sql_identifier.SqlIdentifier],
        prediction_class_columns: List[sql_identifier.SqlIdentifier],
        actual_score_columns: List[sql_identifier.SqlIdentifier],
        actual_class_columns: List[sql_identifier.SqlIdentifier],
        id_columns: List[sql_identifier.SqlIdentifier],
    ) -> None:
        """Ensures all columns exist in the source table.

        Args:
            source_column_schema: Dictionary of column names and types in the source.
            timestamp_column: Name of the timestamp column.
            prediction_score_columns: List of prediction score column names.
            prediction_class_columns: List of prediction class names.
            actual_score_columns: List of actual score column names.
            actual_class_columns: List of actual class column names.
            id_columns: List of id column names.

        Raises:
            ValueError: If any of the columns do not exist in the source.
        """

        if timestamp_column not in source_column_schema:
            raise ValueError(f"Timestamp column {timestamp_column} does not exist in source.")

        if not all([column_name in source_column_schema for column_name in prediction_score_columns]):
            raise ValueError(f"Prediction Score column(s): {prediction_score_columns} do not exist in source.")
        if not all([column_name in source_column_schema for column_name in prediction_class_columns]):
            raise ValueError(f"Prediction Class column(s): {prediction_class_columns} do not exist in source.")
        if not all([column_name in source_column_schema for column_name in actual_score_columns]):
            raise ValueError(f"Actual Score column(s): {actual_score_columns} do not exist in source.")

        if not all([column_name in source_column_schema for column_name in actual_class_columns]):
            raise ValueError(f"Actual Class column(s): {actual_class_columns} do not exist in source.")

        if not all([column_name in source_column_schema for column_name in id_columns]):
            raise ValueError(f"ID column(s): {id_columns} do not exist in source.")

    def validate_source(
        self,
        *,
        source_database: Optional[sql_identifier.SqlIdentifier],
        source_schema: Optional[sql_identifier.SqlIdentifier],
        source: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_score_columns: List[sql_identifier.SqlIdentifier],
        prediction_class_columns: List[sql_identifier.SqlIdentifier],
        actual_score_columns: List[sql_identifier.SqlIdentifier],
        actual_class_columns: List[sql_identifier.SqlIdentifier],
        id_columns: List[sql_identifier.SqlIdentifier],
    ) -> None:
        source_database = source_database or self._database_name
        source_schema = source_schema or self._schema_name
        # Get Schema of the source. Implicitly validates that the source exists.
        source_column_schema: Mapping[str, types.DataType] = table_manager.get_table_schema_types(
            self._sql_client._session,
            source_database,
            source_schema,
            source,
        )
        self._validate_columns_exist_in_source(
            source_column_schema=source_column_schema,
            timestamp_column=timestamp_column,
            prediction_score_columns=prediction_score_columns,
            prediction_class_columns=prediction_class_columns,
            actual_score_columns=actual_score_columns,
            actual_class_columns=actual_class_columns,
            id_columns=id_columns,
        )

    def _alter_monitor(
        self,
        operation: str,
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if operation not in {"SUSPEND", "RESUME"}:
            raise ValueError(f"Operation {operation} not supported for altering Dynamic Tables")
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"""ALTER MODEL MONITOR {self._database_name}.{self._schema_name}.{monitor_name} {operation}""",
            statement_params=statement_params,
        ).has_column("status").has_dimensions(1, 1).validate()

    def suspend_monitor(
        self,
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._alter_monitor(
            operation="SUSPEND",
            monitor_name=monitor_name,
            statement_params=statement_params,
        )

    def resume_monitor(
        self,
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._alter_monitor(
            operation="RESUME",
            monitor_name=monitor_name,
            statement_params=statement_params,
        )
