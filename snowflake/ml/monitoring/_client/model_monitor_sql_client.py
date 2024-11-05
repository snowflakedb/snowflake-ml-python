import typing
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Set

from snowflake import snowpark
from snowflake.ml._internal.utils import (
    db_utils,
    query_result_checker,
    sql_identifier,
    table_manager,
)
from snowflake.ml.model._client.sql import _base
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.snowpark import session, types

SNOWML_MONITORING_METADATA_TABLE_NAME = "_SYSTEM_MONITORING_METADATA"

MODEL_JSON_COL_NAME = "model"
MODEL_JSON_MODEL_NAME_FIELD = "model_name"
MODEL_JSON_VERSION_NAME_FIELD = "version_name"

MONITOR_NAME_COL_NAME = "MONITOR_NAME"
SOURCE_TABLE_NAME_COL_NAME = "SOURCE_TABLE_NAME"
FQ_MODEL_NAME_COL_NAME = "FULLY_QUALIFIED_MODEL_NAME"
VERSION_NAME_COL_NAME = "MODEL_VERSION_NAME"
FUNCTION_NAME_COL_NAME = "FUNCTION_NAME"
TASK_COL_NAME = "TASK"
MONITORING_ENABLED_COL_NAME = "IS_ENABLED"
TIMESTAMP_COL_NAME_COL_NAME = "TIMESTAMP_COLUMN_NAME"
PREDICTION_COL_NAMES_COL_NAME = "PREDICTION_COLUMN_NAMES"
LABEL_COL_NAMES_COL_NAME = "LABEL_COLUMN_NAMES"
ID_COL_NAMES_COL_NAME = "ID_COLUMN_NAMES"


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
        return f"{database_name or self._database_name}.{schema_name or self._schema_name}"

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
            baseline_sql = f"BASELINE='{self._infer_qualified_schema(baseline_database, baseline_schema)}.{baseline}'"
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"""
            CREATE MODEL MONITOR {self._infer_qualified_schema(monitor_database, monitor_schema)}.{monitor_name}
                WITH
                    MODEL='{self._infer_qualified_schema(model_database, model_schema)}.{model_name}'
                    VERSION='{version_name}'
                    FUNCTION='{function_name}'
                    WAREHOUSE='{warehouse_name}'
                    SOURCE='{self._infer_qualified_schema(source_database, source_schema)}.{source}'
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

    def _validate_unique_columns(
        self,
        timestamp_column: sql_identifier.SqlIdentifier,
        id_columns: List[sql_identifier.SqlIdentifier],
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
    ) -> None:
        all_columns = [*id_columns, *prediction_columns, *label_columns, timestamp_column]
        num_all_columns = len(all_columns)
        num_unique_columns = len(set(all_columns))
        if num_all_columns != num_unique_columns:
            raise ValueError("Column names must be unique across id, timestamp, prediction, and label columns.")

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

    def _validate_timestamp_column_type(
        self, table_schema: Mapping[str, types.DataType], timestamp_column: sql_identifier.SqlIdentifier
    ) -> None:
        """Ensures columns have the same type.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            timestamp_column: Name of the timestamp column.

        Raises:
            ValueError: If the timestamp column is not of type TimestampType.
        """
        if not isinstance(table_schema[timestamp_column], types.TimestampType):
            raise ValueError(
                f"Timestamp column: {timestamp_column} must be TimestampType. "
                f"Found: {table_schema[timestamp_column]}"
            )

    def _validate_id_columns_types(
        self, table_schema: Mapping[str, types.DataType], id_columns: List[sql_identifier.SqlIdentifier]
    ) -> None:
        """Ensures id columns have the correct type.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            id_columns: List of id column names.

        Raises:
            ValueError: If the id column is not of type StringType.
        """
        id_column_types = list({table_schema[column_name] for column_name in id_columns})
        all_id_columns_string = all([isinstance(column_type, types.StringType) for column_type in id_column_types])
        if not all_id_columns_string:
            raise ValueError(f"Id columns must all be StringType. Found: {id_column_types}")

    def _validate_prediction_columns_types(
        self, table_schema: Mapping[str, types.DataType], prediction_columns: List[sql_identifier.SqlIdentifier]
    ) -> None:
        """Ensures prediction columns have the same type.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            prediction_columns: List of prediction column names.

        Raises:
            ValueError: If the prediction columns do not share the same type.
        """

        prediction_column_types = {table_schema[column_name] for column_name in prediction_columns}
        if len(prediction_column_types) > 1:
            raise ValueError(f"Prediction column types must be the same. Found: {prediction_column_types}")

    def _validate_label_columns_types(
        self,
        table_schema: Mapping[str, types.DataType],
        label_columns: List[sql_identifier.SqlIdentifier],
    ) -> None:
        """Ensures label columns have the same type, and the correct type for the score type.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            label_columns: List of label column names.

        Raises:
            ValueError: If the label columns do not share the same type.
        """
        label_column_types = {table_schema[column_name] for column_name in label_columns}
        if len(label_column_types) > 1:
            raise ValueError(f"Label column types must be the same. Found: {label_column_types}")

    def _validate_column_types(
        self,
        *,
        table_schema: Mapping[str, types.DataType],
        timestamp_column: sql_identifier.SqlIdentifier,
        id_columns: List[sql_identifier.SqlIdentifier],
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
    ) -> None:
        """Ensures columns have the expected type.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            timestamp_column: Name of the timestamp column.
            id_columns: List of id column names.
            prediction_columns: List of prediction column names.
            label_columns: List of label column names.
        """
        self._validate_timestamp_column_type(table_schema, timestamp_column)
        self._validate_id_columns_types(table_schema, id_columns)
        self._validate_prediction_columns_types(table_schema, prediction_columns)
        self._validate_label_columns_types(table_schema, label_columns)
        # TODO(SNOW-1646693): Validate label makes sense with model task

    def _validate_source_table_features_shape(
        self,
        table_schema: Mapping[str, types.DataType],
        special_columns: Set[sql_identifier.SqlIdentifier],
        model_function: model_manifest_schema.ModelFunctionInfo,
    ) -> None:
        table_schema_without_special_columns = {
            k: v for k, v in table_schema.items() if sql_identifier.SqlIdentifier(k) not in special_columns
        }
        schema_column_types_to_count: typing.Counter[types.DataType] = Counter()
        for column_type in table_schema_without_special_columns.values():
            schema_column_types_to_count[column_type] += 1

        inputs = model_function["signature"].inputs
        function_input_types = [input.as_snowpark_type() for input in inputs]
        function_input_types_to_count: typing.Counter[types.DataType] = Counter()
        for function_input_type in function_input_types:
            function_input_types_to_count[function_input_type] += 1

        if function_input_types_to_count != schema_column_types_to_count:
            raise ValueError(
                "Model function input types do not match the source table input columns types. "
                f"Model function expected: {inputs} but got {table_schema_without_special_columns}"
            )

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

    def delete_monitor_metadata(
        self,
        name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete the row in the metadata table corresponding to the given monitor name.

        Args:
            name: Name of the model monitor whose metadata should be deleted.
            statement_params: Optional set of statement_params to include with query.
        """
        self._sql_client._session.sql(
            f"""DELETE FROM {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}
            WHERE {MONITOR_NAME_COL_NAME} = '{name}'""",
        ).collect(statement_params=statement_params)

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
