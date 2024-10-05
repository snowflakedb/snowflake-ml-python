import json
import string
import textwrap
import typing
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, TypedDict

from importlib_resources import files
from typing_extensions import Required

from snowflake import snowpark
from snowflake.connector import errors
from snowflake.ml._internal.utils import (
    db_utils,
    formatting,
    query_result_checker,
    sql_identifier,
    table_manager,
)
from snowflake.ml.model import type_hints
from snowflake.ml.model._client.sql import _base
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.monitoring.entities import model_monitor_interval, output_score_type
from snowflake.ml.monitoring.entities.model_monitor_interval import (
    ModelMonitorAggregationWindow,
    ModelMonitorRefreshInterval,
)
from snowflake.snowpark import DataFrame, exceptions, session, types
from snowflake.snowpark._internal import type_utils

SNOWML_MONITORING_METADATA_TABLE_NAME = "_SYSTEM_MONITORING_METADATA"
_SNOWML_MONITORING_TABLE_NAME_PREFIX = "_SNOWML_OBS_MONITORING_"
_SNOWML_MONITORING_ACCURACY_TABLE_NAME_PREFIX = "_SNOWML_OBS_ACCURACY_"

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

_DASHBOARD_UDTFS_COMMON_LIST = ["record_count"]
_DASHBOARD_UDTFS_REGRESSION_LIST = ["rmse"]


def _initialize_monitoring_metadata_tables(
    session: session.Session,
    database_name: sql_identifier.SqlIdentifier,
    schema_name: sql_identifier.SqlIdentifier,
    statement_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Create tables necessary for Model Monitoring in provided schema.

    Args:
        session: Active Snowpark session.
        database_name: The database in which to setup resources for Model Monitoring.
        schema_name: The schema in which to setup resources for Model Monitoring.
        statement_params: Optional statement params for queries.
    """
    table_manager.create_single_table(
        session,
        database_name,
        schema_name,
        SNOWML_MONITORING_METADATA_TABLE_NAME,
        [
            (MONITOR_NAME_COL_NAME, "VARCHAR"),
            (SOURCE_TABLE_NAME_COL_NAME, "VARCHAR"),
            (FQ_MODEL_NAME_COL_NAME, "VARCHAR"),
            (VERSION_NAME_COL_NAME, "VARCHAR"),
            (FUNCTION_NAME_COL_NAME, "VARCHAR"),
            (TASK_COL_NAME, "VARCHAR"),
            (MONITORING_ENABLED_COL_NAME, "BOOLEAN"),
            (TIMESTAMP_COL_NAME_COL_NAME, "VARCHAR"),
            (PREDICTION_COL_NAMES_COL_NAME, "ARRAY"),
            (LABEL_COL_NAMES_COL_NAME, "ARRAY"),
            (ID_COL_NAMES_COL_NAME, "ARRAY"),
        ],
        statement_params=statement_params,
    )


def _create_baseline_table_name(model_name: str, version_name: str) -> str:
    return f"_SNOWML_OBS_BASELINE_{model_name}_{version_name}"


def _infer_numeric_categoric_feature_column_names(
    *,
    source_table_schema: Mapping[str, types.DataType],
    timestamp_column: sql_identifier.SqlIdentifier,
    id_columns: List[sql_identifier.SqlIdentifier],
    prediction_columns: List[sql_identifier.SqlIdentifier],
    label_columns: List[sql_identifier.SqlIdentifier],
) -> Tuple[List[sql_identifier.SqlIdentifier], List[sql_identifier.SqlIdentifier]]:
    cols_to_remove = {timestamp_column, *id_columns, *prediction_columns, *label_columns}
    cols_to_consider = [
        (col_name, source_table_schema[col_name]) for col_name in source_table_schema if col_name not in cols_to_remove
    ]
    numeric_cols = [
        sql_identifier.SqlIdentifier(column[0])
        for column in cols_to_consider
        if isinstance(column[1], types._NumericType)
    ]
    categorical_cols = [
        sql_identifier.SqlIdentifier(column[0])
        for column in cols_to_consider
        if isinstance(column[1], types.StringType) or isinstance(column[1], types.BooleanType)
    ]
    return (numeric_cols, categorical_cols)


class _ModelMonitorParams(TypedDict):
    """Class to transfer model monitor parameters to the ModelMonitor class."""

    monitor_name: Required[str]
    fully_qualified_model_name: Required[str]
    version_name: Required[str]
    function_name: Required[str]
    prediction_columns: Required[List[sql_identifier.SqlIdentifier]]
    label_columns: Required[List[sql_identifier.SqlIdentifier]]


class _ModelMonitorSQLClient:
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

    @staticmethod
    def initialize_monitoring_schema(
        session: session.Session,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize tables for tracking metadata associated with model monitoring.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The database in which to setup resources for Model Monitoring.
            schema_name: The schema in which to setup resources for Model Monitoring.
            statement_params: Optional set of statement_params to include with query.
        """
        # Create metadata management tables
        _initialize_monitoring_metadata_tables(session, database_name, schema_name, statement_params)

    def _validate_is_initialized(self) -> bool:
        """Validates whether monitoring metadata has been initialized.

        Returns:
            boolean to indicate whether tables have been initialized.
        """
        try:
            return table_manager.validate_table_exist(
                self._sql_client._session,
                SNOWML_MONITORING_METADATA_TABLE_NAME,
                f"{self._database_name}.{self._schema_name}",
            )
        except exceptions.SnowparkSQLException:
            return False

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
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        res = (
            query_result_checker.SqlResultValidator(
                self._sql_client._session,
                f"""SELECT {FQ_MODEL_NAME_COL_NAME}, {VERSION_NAME_COL_NAME}
                FROM {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}
                WHERE {MONITOR_NAME_COL_NAME} = '{monitor_name}'""",
                statement_params=statement_params,
            )
            .has_column(FQ_MODEL_NAME_COL_NAME, allow_empty=True)
            .has_column(VERSION_NAME_COL_NAME, allow_empty=True)
            .validate()
        )
        return len(res) >= 1

    def validate_existence(
        self,
        fully_qualified_model_name: str,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate existence of a ModelMonitor on a Model Version.

        Args:
            fully_qualified_model_name: Fully qualified name of model.
            version_name: Name of model version.
            statement_params: Optional set of statement_params to include with query.

        Returns:
            Boolean indicating whether monitor exists on model version.
        """
        res = (
            query_result_checker.SqlResultValidator(
                self._sql_client._session,
                f"""SELECT {FQ_MODEL_NAME_COL_NAME}, {VERSION_NAME_COL_NAME}
                FROM {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}
                WHERE {FQ_MODEL_NAME_COL_NAME} = '{fully_qualified_model_name}'
                AND {VERSION_NAME_COL_NAME} = '{version_name}'""",
                statement_params=statement_params,
            )
            .has_column(FQ_MODEL_NAME_COL_NAME, allow_empty=True)
            .has_column(VERSION_NAME_COL_NAME, allow_empty=True)
            .validate()
        )
        return len(res) >= 1

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

    def add_dashboard_udtfs(
        self,
        monitor_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        task: type_hints.Task,
        score_type: output_score_type.OutputScoreType,
        output_columns: List[sql_identifier.SqlIdentifier],
        ground_truth_columns: List[sql_identifier.SqlIdentifier],
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        udtf_name_query_map = self._create_dashboard_udtf_queries(
            monitor_name,
            model_name,
            model_version_name,
            task,
            score_type,
            output_columns,
            ground_truth_columns,
        )
        for udtf_query in udtf_name_query_map.values():
            query_result_checker.SqlResultValidator(
                self._sql_client._session,
                f"""{udtf_query}""",
                statement_params=statement_params,
            ).validate()

    def get_monitoring_table_fully_qualified_name(
        self,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
    ) -> str:
        table_name = f"{_SNOWML_MONITORING_TABLE_NAME_PREFIX}_{model_name}_{model_version_name}"
        return table_manager.get_fully_qualified_table_name(self._database_name, self._schema_name, table_name)

    def get_accuracy_monitoring_table_fully_qualified_name(
        self,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
    ) -> str:
        table_name = f"{_SNOWML_MONITORING_ACCURACY_TABLE_NAME_PREFIX}_{model_name}_{model_version_name}"
        return table_manager.get_fully_qualified_table_name(self._database_name, self._schema_name, table_name)

    def _create_dashboard_udtf_queries(
        self,
        monitor_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        task: type_hints.Task,
        score_type: output_score_type.OutputScoreType,
        output_columns: List[sql_identifier.SqlIdentifier],
        ground_truth_columns: List[sql_identifier.SqlIdentifier],
    ) -> Mapping[str, str]:
        query_files = files("snowflake.ml.monitoring._client")
        # TODO(apgupta): Expand list of queries based on model objective and score type.
        queries_list = []
        queries_list.extend(_DASHBOARD_UDTFS_COMMON_LIST)
        if task == type_hints.Task.TABULAR_REGRESSION:
            queries_list.extend(_DASHBOARD_UDTFS_REGRESSION_LIST)
        var_map = {
            "MODEL_MONITOR_NAME": monitor_name,
            "MONITORING_TABLE": self.get_monitoring_table_fully_qualified_name(model_name, model_version_name),
            "MONITORING_PRED_LABEL_JOINED_TABLE": self.get_accuracy_monitoring_table_fully_qualified_name(
                model_name, model_version_name
            ),
            "OUTPUT_COLUMN_NAME": output_columns[0],
            "GROUND_TRUTH_COLUMN_NAME": ground_truth_columns[0],
        }

        udf_name_query_map = {}
        for q in queries_list:
            q_template = query_files.joinpath(f"queries/{q}.ssql").read_text()
            q_actual = string.Template(q_template).substitute(var_map)
            udf_name_query_map[q] = q_actual
        return udf_name_query_map

    def _validate_columns_exist_in_source_table(
        self,
        *,
        table_schema: Mapping[str, types.DataType],
        source_table_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
        id_columns: List[sql_identifier.SqlIdentifier],
    ) -> None:
        """Ensures all columns exist in the source table.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            source_table_name: Name of the table with model data to monitor.
            timestamp_column: Name of the timestamp column.
            prediction_columns: List of prediction column names.
            label_columns: List of label column names.
            id_columns: List of id column names.

        Raises:
            ValueError: If any of the columns do not exist in the source table.
        """

        if timestamp_column not in table_schema:
            raise ValueError(f"Timestamp column {timestamp_column} does not exist in table {source_table_name}.")

        if not all([column_name in table_schema for column_name in prediction_columns]):
            raise ValueError(f"Prediction column(s): {prediction_columns} do not exist in table {source_table_name}.")
        if not all([column_name in table_schema for column_name in label_columns]):
            raise ValueError(f"Label column(s): {label_columns} do not exist in table {source_table_name}.")
        if not all([column_name in table_schema for column_name in id_columns]):
            raise ValueError(f"ID column(s): {id_columns} do not exist in table {source_table_name}.")

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

    def get_model_monitor_by_name(
        self,
        monitor_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> _ModelMonitorParams:
        """Fetch metadata for a Model Monitor by name.

        Args:
            monitor_name: Name of ModelMonitor to fetch.
            statement_params: Optional set of statement_params to include with query.

        Returns:
            _ModelMonitorParams dict with Name of monitor, fully qualified model name,
            model version name, model function name, prediction_col, label_col.

        Raises:
            ValueError: If multiple ModelMonitors exist with the same name.
        """
        try:
            res = (
                query_result_checker.SqlResultValidator(
                    self._sql_client._session,
                    f"""SELECT {FQ_MODEL_NAME_COL_NAME}, {VERSION_NAME_COL_NAME}, {FUNCTION_NAME_COL_NAME},
                    {PREDICTION_COL_NAMES_COL_NAME}, {LABEL_COL_NAMES_COL_NAME}
                    FROM {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}
                    WHERE {MONITOR_NAME_COL_NAME} = '{monitor_name}'""",
                    statement_params=statement_params,
                )
                .has_column(FQ_MODEL_NAME_COL_NAME)
                .has_column(VERSION_NAME_COL_NAME)
                .has_column(FUNCTION_NAME_COL_NAME)
                .has_column(PREDICTION_COL_NAMES_COL_NAME)
                .has_column(LABEL_COL_NAMES_COL_NAME)
                .validate()
            )
        except errors.DataError:
            raise ValueError(f"Failed to find any monitor with name '{monitor_name}'")

        if len(res) > 1:
            raise ValueError(f"Invalid state. Multiple Monitors exist with name '{monitor_name}'")

        return _ModelMonitorParams(
            monitor_name=str(monitor_name),
            fully_qualified_model_name=res[0][FQ_MODEL_NAME_COL_NAME],
            version_name=res[0][VERSION_NAME_COL_NAME],
            function_name=res[0][FUNCTION_NAME_COL_NAME],
            prediction_columns=[
                sql_identifier.SqlIdentifier(prediction_column)
                for prediction_column in json.loads(res[0][PREDICTION_COL_NAMES_COL_NAME])
            ],
            label_columns=[
                sql_identifier.SqlIdentifier(label_column)
                for label_column in json.loads(res[0][LABEL_COL_NAMES_COL_NAME])
            ],
        )

    def get_model_monitor_by_model_version(
        self,
        *,
        model_db: sql_identifier.SqlIdentifier,
        model_schema: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> _ModelMonitorParams:
        """Fetch metadata for a Model Monitor by model version.

        Args:
            model_db: Database of model.
            model_schema: Schema of model.
            model_name: Model name.
            version_name: Model version name
            statement_params: Optional set of statement_params to include with queries.

        Returns:
            _ModelMonitorParams dict with Name of monitor, fully qualified model name,
            model version name, model function name, prediction_col, label_col.

        Raises:
            ValueError: If multiple ModelMonitors exist with the same name.
        """
        res = (
            query_result_checker.SqlResultValidator(
                self._sql_client._session,
                f"""SELECT {MONITOR_NAME_COL_NAME}, {FQ_MODEL_NAME_COL_NAME},
                {VERSION_NAME_COL_NAME}, {FUNCTION_NAME_COL_NAME}
                FROM {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}
                WHERE {FQ_MODEL_NAME_COL_NAME} = '{model_db}.{model_schema}.{model_name}'
                AND {VERSION_NAME_COL_NAME} = '{version_name}'""",
                statement_params=statement_params,
            )
            .has_column(MONITOR_NAME_COL_NAME)
            .has_column(FQ_MODEL_NAME_COL_NAME)
            .has_column(VERSION_NAME_COL_NAME)
            .has_column(FUNCTION_NAME_COL_NAME)
            .validate()
        )
        if len(res) > 1:
            raise ValueError(
                f"Invalid state. Multiple Monitors exist for model: '{model_name}' and version: '{version_name}'"
            )
        return _ModelMonitorParams(
            monitor_name=res[0][MONITOR_NAME_COL_NAME],
            fully_qualified_model_name=res[0][FQ_MODEL_NAME_COL_NAME],
            version_name=res[0][VERSION_NAME_COL_NAME],
            function_name=res[0][FUNCTION_NAME_COL_NAME],
            prediction_columns=[
                sql_identifier.SqlIdentifier(prediction_column)
                for prediction_column in json.loads(res[0][PREDICTION_COL_NAMES_COL_NAME])
            ],
            label_columns=[
                sql_identifier.SqlIdentifier(label_column)
                for label_column in json.loads(res[0][LABEL_COL_NAMES_COL_NAME])
            ],
        )

    def get_score_type(
        self,
        task: type_hints.Task,
        source_table_name: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
    ) -> output_score_type.OutputScoreType:
        """Infer score type given model task and prediction table columns.

        Args:
            task: Model task
            source_table_name: Source data table containing model outputs.
            prediction_columns: columns in source data table corresponding to model outputs.

        Returns:
            OutputScoreType for model.
        """
        table_schema: Mapping[str, types.DataType] = table_manager.get_table_schema_types(
            self._sql_client._session,
            self._database_name,
            self._schema_name,
            source_table_name,
        )
        return output_score_type.OutputScoreType.deduce_score_type(table_schema, prediction_columns, task)

    def validate_source_table(
        self,
        source_table_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
        id_columns: List[sql_identifier.SqlIdentifier],
        model_function: model_manifest_schema.ModelFunctionInfo,
    ) -> None:
        # Validate source table exists
        if not table_manager.validate_table_exist(
            self._sql_client._session,
            source_table_name,
            f"{self._database_name}.{self._schema_name}",
        ):
            raise ValueError(
                f"Table {source_table_name} does not exist in schema {self._database_name}.{self._schema_name}."
            )
        table_schema: Mapping[str, types.DataType] = table_manager.get_table_schema_types(
            self._sql_client._session,
            self._database_name,
            self._schema_name,
            source_table_name,
        )
        self._validate_columns_exist_in_source_table(
            table_schema=table_schema,
            source_table_name=source_table_name,
            timestamp_column=timestamp_column,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
            id_columns=id_columns,
        )
        self._validate_column_types(
            table_schema=table_schema,
            timestamp_column=timestamp_column,
            id_columns=id_columns,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
        )
        self._validate_source_table_features_shape(
            table_schema=table_schema,
            special_columns={timestamp_column, *id_columns, *prediction_columns, *label_columns},
            model_function=model_function,
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

    def delete_baseline_table(
        self,
        fully_qualified_model_name: str,
        version_name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete the baseline table corresponding to a particular model and version.

        Args:
            fully_qualified_model_name: Fully qualified name of the model.
            version_name: Name of the model version.
            statement_params: Optional set of statement_params to include with query.
        """
        table_name = _create_baseline_table_name(fully_qualified_model_name, version_name)
        self._sql_client._session.sql(
            f"""DROP TABLE IF EXISTS {self._database_name}.{self._schema_name}.{table_name}"""
        ).collect(statement_params=statement_params)

    def delete_dynamic_tables(
        self,
        fully_qualified_model_name: str,
        version_name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete the dynamic tables corresponding to a particular model and version.

        Args:
            fully_qualified_model_name: Fully qualified name of the model.
            version_name: Name of the model version.
            statement_params: Optional set of statement_params to include with query.
        """
        _, _, model_name = sql_identifier.parse_fully_qualified_name(fully_qualified_model_name)
        model_id = sql_identifier.SqlIdentifier(model_name)
        version_id = sql_identifier.SqlIdentifier(version_name)
        monitoring_table_name = self.get_monitoring_table_fully_qualified_name(model_id, version_id)
        self._sql_client._session.sql(f"""DROP DYNAMIC TABLE IF EXISTS {monitoring_table_name}""").collect(
            statement_params=statement_params
        )
        accuracy_table_name = self.get_accuracy_monitoring_table_fully_qualified_name(model_id, version_id)
        self._sql_client._session.sql(f"""DROP DYNAMIC TABLE IF EXISTS {accuracy_table_name}""").collect(
            statement_params=statement_params
        )

    def create_monitor_on_model_version(
        self,
        monitor_name: sql_identifier.SqlIdentifier,
        source_table_name: sql_identifier.SqlIdentifier,
        fully_qualified_model_name: str,
        version_name: sql_identifier.SqlIdentifier,
        function_name: str,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
        id_columns: List[sql_identifier.SqlIdentifier],
        task: type_hints.Task,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates a ModelMonitor on a Model Version from the Snowflake Model Registry. Creates public schema for metadata.

        Args:
            monitor_name: Name of monitor object to create.
            source_table_name: Name of source data table to monitor.
            fully_qualified_model_name: fully qualified name of model to monitor '<db>.<schema>.<model_name>'.
            version_name: model version name to monitor.
            function_name: function_name to monitor in model version.
            timestamp_column: timestamp column name.
            prediction_columns: list of prediction column names.
            label_columns: list of label column names.
            id_columns: list of id column names.
            task: Task of the model, e.g. TABULAR_REGRESSION.
            statement_params: Optional dict of statement_params to include with queries.

        Raises:
            ValueError: If model version is already monitored.
        """
        # Validate monitor does not already exist on model version.
        if self.validate_existence(fully_qualified_model_name, version_name, statement_params):
            raise ValueError(f"Model {fully_qualified_model_name} Version {version_name} is already monitored!")

        if self.validate_existence_by_name(monitor_name, statement_params):
            raise ValueError(f"Model Monitor with name '{monitor_name}' already exists!")

        prediction_columns_for_select = formatting.format_value_for_select(prediction_columns)
        label_columns_for_select = formatting.format_value_for_select(label_columns)
        id_columns_for_select = formatting.format_value_for_select(id_columns)
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            textwrap.dedent(
                f"""INSERT INTO {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}
                ({MONITOR_NAME_COL_NAME}, {SOURCE_TABLE_NAME_COL_NAME}, {FQ_MODEL_NAME_COL_NAME},
                {VERSION_NAME_COL_NAME}, {FUNCTION_NAME_COL_NAME}, {TASK_COL_NAME},
                {MONITORING_ENABLED_COL_NAME}, {TIMESTAMP_COL_NAME_COL_NAME},
                {PREDICTION_COL_NAMES_COL_NAME}, {LABEL_COL_NAMES_COL_NAME},
                {ID_COL_NAMES_COL_NAME})
                SELECT '{monitor_name}', '{source_table_name}', '{fully_qualified_model_name}',
                '{version_name}', '{function_name}', '{task.value}', TRUE, '{timestamp_column}',
                {prediction_columns_for_select}, {label_columns_for_select}, {id_columns_for_select}"""
            ),
            statement_params=statement_params,
        ).insertion_success(expected_num_rows=1).validate()

    def initialize_baseline_table(
        self,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        source_table_name: str,
        columns_to_drop: Optional[List[sql_identifier.SqlIdentifier]] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the baseline table for a Model Version. Creates schema for baseline data using the source table.

        Args:
            model_name: name of model to monitor.
            version_name: model version name to monitor.
            source_table_name: name of the user's table containing their model data.
            columns_to_drop: special columns in the source table to be excluded from baseline tables.
            statement_params: Optional dict of statement_params to include with queries.
        """
        table_schema = table_manager.get_table_schema_types(
            self._sql_client._session,
            database=self._database_name,
            schema=self._schema_name,
            table_name=source_table_name,
        )

        if columns_to_drop is None:
            columns_to_drop = []

        table_manager.create_single_table(
            self._sql_client._session,
            self._database_name,
            self._schema_name,
            _create_baseline_table_name(model_name, version_name),
            [
                (k, type_utils.convert_sp_to_sf_type(v))
                for k, v in table_schema.items()
                if sql_identifier.SqlIdentifier(k) not in columns_to_drop
            ],
            statement_params=statement_params,
        )

    def get_all_model_monitor_metadata(
        self,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[snowpark.Row]:
        """Get the metadata for all model monitors in the given schema.

        Args:
            statement_params: Optional dict of statement_params to include with queries.

        Returns:
            List of snowpark.Row containing metadata for each model monitor.
        """
        return query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"""SELECT *
                FROM {self._database_name}.{self._schema_name}.{SNOWML_MONITORING_METADATA_TABLE_NAME}""",
            statement_params=statement_params,
        ).validate()

    def materialize_baseline_dataframe(
        self,
        baseline_df: DataFrame,
        fully_qualified_model_name: str,
        model_version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Materialize baseline dataframe to a permanent snowflake table. This method
        truncates (overwrite without dropping) any existing data in the baseline table.

        Args:
            baseline_df: dataframe containing baseline data that monitored data will be compared against.
            fully_qualified_model_name: name of the model.
            model_version_name: model version name to monitor.
            statement_params: Optional dict of statement_params to include with queries.

        Raises:
            ValueError: If no baseline table was initialized.
        """

        _, _, model_name = sql_identifier.parse_fully_qualified_name(fully_qualified_model_name)
        baseline_table_name = _create_baseline_table_name(model_name, model_version_name)

        baseline_table_exists = db_utils.db_object_exists(
            self._sql_client._session,
            db_utils.SnowflakeDbObjectType.TABLE,
            sql_identifier.SqlIdentifier(baseline_table_name),
            database_name=self._database_name,
            schema_name=self._schema_name,
            statement_params=statement_params,
        )
        if not baseline_table_exists:
            raise ValueError(
                f"Baseline table '{baseline_table_name}' does not exist for model: "
                f"'{model_name}' and model_version: '{model_version_name}'"
            )

        fully_qualified_baseline_table_name = [self._database_name, self._schema_name, baseline_table_name]

        try:
            # Truncate overwrites by clearing the rows in the table, instead of dropping the table.
            # This lets us keep the schema to validate the baseline_df against.
            baseline_df.write.mode("truncate").save_as_table(
                fully_qualified_baseline_table_name, statement_params=statement_params
            )
        except exceptions.SnowparkSQLException as e:
            raise ValueError(
                f"""Failed to save baseline dataframe.
                Ensure that the baseline dataframe columns match those provided in your monitored table: {e}"""
            )

    def _alter_monitor_dynamic_tables(
        self,
        operation: str,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if operation not in {"SUSPEND", "RESUME"}:
            raise ValueError(f"Operation {operation} not supported for altering Dynamic Tables")
        fq_monitor_dt_name = self.get_monitoring_table_fully_qualified_name(model_name, version_name)
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"""ALTER DYNAMIC TABLE {fq_monitor_dt_name} {operation}""",
            statement_params=statement_params,
        ).has_column("status").has_dimensions(1, 1).validate()

        fq_accuracy_dt_name = self.get_accuracy_monitoring_table_fully_qualified_name(model_name, version_name)
        query_result_checker.SqlResultValidator(
            self._sql_client._session,
            f"""ALTER DYNAMIC TABLE {fq_accuracy_dt_name} {operation}""",
            statement_params=statement_params,
        ).has_column("status").has_dimensions(1, 1).validate()

    def suspend_monitor_dynamic_tables(
        self,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._alter_monitor_dynamic_tables(
            operation="SUSPEND",
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

    def resume_monitor_dynamic_tables(
        self,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._alter_monitor_dynamic_tables(
            operation="RESUME",
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

    def create_dynamic_tables_for_monitor(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        task: type_hints.Task,
        source_table_name: sql_identifier.SqlIdentifier,
        refresh_interval: model_monitor_interval.ModelMonitorRefreshInterval,
        aggregation_window: model_monitor_interval.ModelMonitorAggregationWindow,
        warehouse_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        id_columns: List[sql_identifier.SqlIdentifier],
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
        score_type: output_score_type.OutputScoreType,
    ) -> None:
        table_schema: Mapping[str, types.DataType] = table_manager.get_table_schema_types(
            self._sql_client._session,
            self._database_name,
            self._schema_name,
            source_table_name,
        )
        (numeric_features_names, categorical_feature_names) = _infer_numeric_categoric_feature_column_names(
            source_table_schema=table_schema,
            timestamp_column=timestamp_column,
            id_columns=id_columns,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
        )
        features_dynamic_table_query = self._monitoring_dynamic_table_query(
            model_name=model_name,
            model_version_name=model_version_name,
            source_table_name=source_table_name,
            refresh_interval=refresh_interval,
            aggregate_window=aggregation_window,
            warehouse_name=warehouse_name,
            timestamp_column=timestamp_column,
            numeric_features=numeric_features_names,
            categoric_features=categorical_feature_names,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
        )
        query_result_checker.SqlResultValidator(self._sql_client._session, features_dynamic_table_query).has_column(
            "status"
        ).has_dimensions(1, 1).validate()

        label_pred_join_table_query = self._monitoring_accuracy_table_query(
            model_name=model_name,
            model_version_name=model_version_name,
            task=task,
            source_table_name=source_table_name,
            refresh_interval=refresh_interval,
            aggregate_window=aggregation_window,
            warehouse_name=warehouse_name,
            timestamp_column=timestamp_column,
            prediction_columns=prediction_columns,
            label_columns=label_columns,
            score_type=score_type,
        )
        query_result_checker.SqlResultValidator(self._sql_client._session, label_pred_join_table_query).has_column(
            "status"
        ).has_dimensions(1, 1).validate()

    def _monitoring_dynamic_table_query(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        source_table_name: sql_identifier.SqlIdentifier,
        refresh_interval: ModelMonitorRefreshInterval,
        aggregate_window: ModelMonitorAggregationWindow,
        warehouse_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        numeric_features: List[sql_identifier.SqlIdentifier],
        categoric_features: List[sql_identifier.SqlIdentifier],
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
    ) -> str:
        """
        Generates a dynamic table query for Observability - Monitoring.

        Args:
            model_name: Model name to monitor.
            model_version_name: Model version name to monitor.
            source_table_name: Name of source data table to monitor.
            refresh_interval: Refresh interval in minutes.
            aggregate_window: Aggregate window minutes.
            warehouse_name: Warehouse name to use for dynamic table.
            timestamp_column: Timestamp column name.
            numeric_features: List of numeric features to capture.
            categoric_features: List of categoric features to capture.
            prediction_columns: List of columns that contain model inference outputs.
            label_columns: List of columns that contain ground truth values.

        Raises:
            ValueError: If multiple output/ground truth columns are specified. MultiClass models are not yet supported.

        Returns:
            Dynamic table query.
        """
        # output and ground cols are list to keep interface extensible.
        # for prpr only one label and one output col will be supported
        if len(prediction_columns) != 1 or len(label_columns) != 1:
            raise ValueError("Multiple Output columns are not supported in monitoring")

        monitoring_dt_name = self.get_monitoring_table_fully_qualified_name(model_name, model_version_name)

        feature_cols_query_list = []
        for feature in numeric_features + prediction_columns + label_columns:
            feature_cols_query_list.append(
                """
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE({col}),
                'count', count_if({col} is not null),
                'count_null', count_if({col} is null),
                'min', min({col}),
                'max', max({col}),
                'sum', sum({col})
            ) AS {col}""".format(
                    col=feature
                )
            )

        for col in categoric_features:
            feature_cols_query_list.append(
                f"""
            {self._database_name}.{self._schema_name}.OBJECT_SUM(to_varchar({col})) AS {col}"""
            )
        feature_cols_query = ",".join(feature_cols_query_list)

        return f"""
        CREATE DYNAMIC TABLE IF NOT EXISTS {monitoring_dt_name}
            TARGET_LAG = '{refresh_interval.minutes} minutes'
            WAREHOUSE = {warehouse_name}
            REFRESH_MODE = AUTO
            INITIALIZE = ON_CREATE
        AS
        SELECT
            TIME_SLICE({timestamp_column}, {aggregate_window.minutes}, 'MINUTE') timestamp,{feature_cols_query}
        FROM
            {source_table_name}
        GROUP BY
            1
        """

    def _monitoring_accuracy_table_query(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        task: type_hints.Task,
        source_table_name: sql_identifier.SqlIdentifier,
        refresh_interval: ModelMonitorRefreshInterval,
        aggregate_window: ModelMonitorAggregationWindow,
        warehouse_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
        score_type: output_score_type.OutputScoreType,
    ) -> str:
        # output and ground cols are list to keep interface extensible.
        # for prpr only one label and one output col will be supported
        if len(prediction_columns) != 1 or len(label_columns) != 1:
            raise ValueError("Multiple Output columns are not supported in monitoring")
        if task == type_hints.Task.TABULAR_BINARY_CLASSIFICATION:
            return self._monitoring_classification_accuracy_table_query(
                model_name=model_name,
                model_version_name=model_version_name,
                source_table_name=source_table_name,
                refresh_interval=refresh_interval,
                aggregate_window=aggregate_window,
                warehouse_name=warehouse_name,
                timestamp_column=timestamp_column,
                prediction_columns=prediction_columns,
                label_columns=label_columns,
                score_type=score_type,
            )
        else:
            return self._monitoring_regression_accuracy_table_query(
                model_name=model_name,
                model_version_name=model_version_name,
                source_table_name=source_table_name,
                refresh_interval=refresh_interval,
                aggregate_window=aggregate_window,
                warehouse_name=warehouse_name,
                timestamp_column=timestamp_column,
                prediction_columns=prediction_columns,
                label_columns=label_columns,
            )

    def _monitoring_regression_accuracy_table_query(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        source_table_name: sql_identifier.SqlIdentifier,
        refresh_interval: ModelMonitorRefreshInterval,
        aggregate_window: ModelMonitorAggregationWindow,
        warehouse_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
    ) -> str:
        """
        Generates a dynamic table query for Monitoring - regression model accuracy.

        Args:
            model_name: Model name to monitor.
            model_version_name: Model version name to monitor.
            source_table_name: Name of source data table to monitor.
            refresh_interval: Refresh interval in minutes.
            aggregate_window: Aggregate window minutes.
            warehouse_name: Warehouse name to use for dynamic table.
            timestamp_column: Timestamp column name.
            prediction_columns: List of output columns.
            label_columns: List of ground truth columns.

        Returns:
            Dynamic table query.

        Raises:
            ValueError: If output columns are not same as ground truth columns.

        """

        if len(prediction_columns) != len(label_columns):
            raise ValueError(f"Mismatch in output & ground truth columns: {prediction_columns} != {label_columns}")

        monitoring_dt_name = self.get_accuracy_monitoring_table_fully_qualified_name(model_name, model_version_name)

        output_cols_query_list = []

        output_cols_query_list.append(
            f"""
            OBJECT_CONSTRUCT(
                'sum_difference_label_pred', sum({prediction_columns[0]} - {label_columns[0]}),
                'sum_log_difference_square_label_pred',
                sum(
                    case
                        when {prediction_columns[0]} > -1 and {label_columns[0]} > -1
                        then pow(ln({prediction_columns[0]} + 1) - ln({label_columns[0]} + 1),2)
                        else null
                    END
                ),
                'sum_difference_squares_label_pred',
                sum(
                    pow(
                        {prediction_columns[0]} - {label_columns[0]},
                        2
                    )
                ),
                'sum_absolute_regression_labels', sum(abs({label_columns[0]})),
                'sum_absolute_percentage_error',
                sum(
                    abs(
                        div0null(
                            ({prediction_columns[0]} - {label_columns[0]}),
                            {label_columns[0]}
                        )
                    )
                ),
                'sum_absolute_difference_label_pred',
                sum(
                    abs({prediction_columns[0]} - {label_columns[0]})
                ),
                'sum_prediction', sum({prediction_columns[0]}),
                'sum_label', sum({label_columns[0]}),
                'count', count(*)
            ) AS AGGREGATE_METRICS,
            APPROX_PERCENTILE_ACCUMULATE({prediction_columns[0]}) prediction_sketch,
            APPROX_PERCENTILE_ACCUMULATE({label_columns[0]}) label_sketch"""
        )
        output_cols_query = ", ".join(output_cols_query_list)

        return f"""
        CREATE DYNAMIC TABLE IF NOT EXISTS {monitoring_dt_name}
            TARGET_LAG = '{refresh_interval.minutes} minutes'
            WAREHOUSE = {warehouse_name}
            REFRESH_MODE = AUTO
            INITIALIZE = ON_CREATE
        AS
        SELECT
            TIME_SLICE({timestamp_column}, {aggregate_window.minutes}, 'MINUTE') timestamp,
            'class_regression' label_class,{output_cols_query}
        FROM
            {source_table_name}
        GROUP BY
            1
        """

    def _monitoring_classification_accuracy_table_query(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        model_version_name: sql_identifier.SqlIdentifier,
        source_table_name: sql_identifier.SqlIdentifier,
        refresh_interval: ModelMonitorRefreshInterval,
        aggregate_window: ModelMonitorAggregationWindow,
        warehouse_name: sql_identifier.SqlIdentifier,
        timestamp_column: sql_identifier.SqlIdentifier,
        prediction_columns: List[sql_identifier.SqlIdentifier],
        label_columns: List[sql_identifier.SqlIdentifier],
        score_type: output_score_type.OutputScoreType,
    ) -> str:
        monitoring_dt_name = self.get_accuracy_monitoring_table_fully_qualified_name(model_name, model_version_name)

        # Initialize the select clause components
        select_clauses = []

        select_clauses.append(
            f"""
            {prediction_columns[0]},
            {label_columns[0]},
            CASE
                WHEN {label_columns[0]} = 1 THEN 'class_positive'
                ELSE 'class_negative'
            END AS label_class"""
        )

        # Join all the select clauses into a single string
        select_clause = f"{timestamp_column} AS timestamp," + ",".join(select_clauses)

        # Create the final CTE query
        cte_query = f"""
        WITH filtered_data AS (
            SELECT
                {select_clause}
            FROM
                {source_table_name}
        )"""

        # Initialize the select clause components
        select_clauses = []

        score_type_agg_clause = ""
        if score_type == output_score_type.OutputScoreType.PROBITS:
            score_type_agg_clause = f"""
                'sum_log_loss',
                CASE
                    WHEN label_class = 'class_positive' THEN sum(-ln({prediction_columns[0]}))
                    ELSE sum(-ln(1 - {prediction_columns[0]}))
                END,"""
        else:
            score_type_agg_clause = f"""
                'tp', count_if({label_columns[0]} = 1 AND {prediction_columns[0]} = 1),
                'tn', count_if({label_columns[0]} = 0 AND {prediction_columns[0]} = 0),
                'fp', count_if({label_columns[0]} = 0 AND {prediction_columns[0]} = 1),
                'fn', count_if({label_columns[0]} = 1 AND {prediction_columns[0]} = 0),"""

        select_clauses.append(
            f"""
            label_class,
            OBJECT_CONSTRUCT(
                'sum_prediction', sum({prediction_columns[0]}),
                'sum_label', sum({label_columns[0]}),{score_type_agg_clause}
                'count', count(*)
            ) AS AGGREGATE_METRICS,
            APPROX_PERCENTILE_ACCUMULATE({prediction_columns[0]}) prediction_sketch,
            APPROX_PERCENTILE_ACCUMULATE({label_columns[0]}) label_sketch"""
        )

        # Join all the select clauses into a single string
        select_clause = ",\n".join(select_clauses)

        return f"""
            CREATE DYNAMIC TABLE IF NOT EXISTS {monitoring_dt_name}
                TARGET_LAG = '{refresh_interval.minutes} minutes'
                WAREHOUSE = {warehouse_name}
                REFRESH_MODE = AUTO
                INITIALIZE = ON_CREATE
            AS{cte_query}
            select
                time_slice(timestamp, {aggregate_window.minutes}, 'MINUTE') timestamp,{select_clause}
            FROM
                filtered_data
            group by
                1,
                2
        """
