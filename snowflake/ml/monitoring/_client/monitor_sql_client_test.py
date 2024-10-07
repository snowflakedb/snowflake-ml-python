from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.monitoring._client import monitor_sql_client
from snowflake.ml.monitoring.entities import output_score_type
from snowflake.ml.monitoring.entities.model_monitor_interval import (
    ModelMonitorAggregationWindow,
    ModelMonitorRefreshInterval,
)
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import DataFrame, Row, Session, types


class ModelMonitorSqlClientTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_db_name = sql_identifier.SqlIdentifier("SNOWML_OBSERVABILITY")
        self.test_schema_name = sql_identifier.SqlIdentifier("DATA")

        self.test_monitor_name = sql_identifier.SqlIdentifier("TEST")
        self.test_source_table_name = sql_identifier.SqlIdentifier("MODEL_OUTPUTS")
        self.test_model_version_name = sql_identifier.SqlIdentifier("TEST_MODEL_VERSION")
        self.test_model_name = sql_identifier.SqlIdentifier("TEST_MODEL")
        self.test_fq_model_name = f"{self.test_db_name}.{self.test_schema_name}.{self.test_model_name}"
        self.test_function_name = sql_identifier.SqlIdentifier("PREDICT")
        self.test_timestamp_column = sql_identifier.SqlIdentifier("TIMESTAMP")
        self.test_prediction_column_name = sql_identifier.SqlIdentifier("PREDICTION")
        self.test_label_column_name = sql_identifier.SqlIdentifier("LABEL")
        self.test_id_column_name = sql_identifier.SqlIdentifier("ID")
        self.test_baseline_table_name_sql = "_SNOWML_OBS_BASELINE_TEST_MODEL_TEST_MODEL_VERSION"
        self.test_wh_name = sql_identifier.SqlIdentifier("ML_OBS_WAREHOUSE")

        session = cast(Session, self.m_session)
        self.monitor_sql_client = monitor_sql_client._ModelMonitorSQLClient(
            session, database_name=self.test_db_name, schema_name=self.test_schema_name
        )

        self.mon_table_name = (
            f"{monitor_sql_client._SNOWML_MONITORING_TABLE_NAME_PREFIX}_"
            + self.test_model_name
            + f"_{self.test_model_version_name}"
        )
        self.acc_table_name = (
            f"{monitor_sql_client._SNOWML_MONITORING_ACCURACY_TABLE_NAME_PREFIX}_"
            + self.test_model_name
            + f"_{self.test_model_version_name}"
        )

    def test_validate_source_table(self) -> None:
        mocked_table_out = mock.MagicMock(name="schema")
        self.m_session.table = mock.MagicMock(name="table", return_value=mocked_table_out)
        mocked_table_out.schema = mock.MagicMock(name="schema")
        mocked_table_out.schema.fields = [
            types.StructField(self.test_timestamp_column, types.TimestampType()),
            types.StructField(self.test_prediction_column_name, types.DoubleType()),
            types.StructField(self.test_label_column_name, types.DoubleType()),
            types.StructField(self.test_id_column_name, types.StringType()),
        ]

        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '{self.test_source_table_name}' IN SNOWML_OBSERVABILITY.DATA""",
            result=mock_data_frame.MockDataFrame([Row(name=self.test_source_table_name)]),
        )
        self.monitor_sql_client.validate_source_table(
            source_table_name=self.test_source_table_name,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            model_function=model_manifest_schema.ModelFunctionInfo(
                name="PREDICT",
                target_method="predict",
                target_method_function_type="FUNCTION",
                signature=model_signature.ModelSignature(inputs=[], outputs=[]),
                is_partitioned=False,
            ),
        )
        self.m_session.table.assert_called_once_with(
            f"{self.test_db_name}.{self.test_schema_name}.{self.test_source_table_name}"
        )
        self.m_session.finalize()

    def test_validate_source_table_shape(self) -> None:
        mocked_table_out = mock.MagicMock(name="schema")
        self.m_session.table = mock.MagicMock(name="table", return_value=mocked_table_out)
        mocked_table_out.schema = mock.MagicMock(name="schema")
        mocked_table_out.schema.fields = [
            types.StructField(self.test_timestamp_column, types.TimestampType()),
            types.StructField(self.test_prediction_column_name, types.DoubleType()),
            types.StructField(self.test_label_column_name, types.DoubleType()),
            types.StructField(self.test_id_column_name, types.StringType()),
            types.StructField("feature1", types.StringType()),
        ]

        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '{self.test_source_table_name}' IN SNOWML_OBSERVABILITY.DATA""",
            result=mock_data_frame.MockDataFrame([Row(name=self.test_source_table_name)]),
        )
        self.monitor_sql_client.validate_source_table(
            source_table_name=self.test_source_table_name,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            model_function=model_manifest_schema.ModelFunctionInfo(
                name="PREDICT",
                target_method="predict",
                target_method_function_type="FUNCTION",
                signature=model_signature.ModelSignature(
                    inputs=[
                        model_signature.FeatureSpec("input_feature_0", model_signature.DataType.STRING),
                    ],
                    outputs=[],
                ),
                is_partitioned=False,
            ),
        )
        self.m_session.table.assert_called_once_with(
            f"{self.test_db_name}.{self.test_schema_name}.{self.test_source_table_name}"
        )
        self.m_session.finalize()

    def test_validate_source_table_shape_does_not_match_function_signature(self) -> None:
        mocked_table_out = mock.MagicMock(name="schema")
        self.m_session.table = mock.MagicMock(name="table", return_value=mocked_table_out)
        mocked_table_out.schema = mock.MagicMock(name="schema")
        mocked_table_out.schema.fields = [
            types.StructField(self.test_timestamp_column, types.TimestampType()),
            types.StructField(self.test_prediction_column_name, types.DoubleType()),
            types.StructField(self.test_label_column_name, types.DoubleType()),
            types.StructField(self.test_id_column_name, types.StringType()),
            types.StructField("feature1", types.StringType()),
        ]

        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '{self.test_source_table_name}' IN SNOWML_OBSERVABILITY.DATA""",
            result=mock_data_frame.MockDataFrame([Row(name=self.test_source_table_name)]),
        )

        expected_msg = (
            r"Model function input types do not match the source table input columns types\. Model function expected: "
            r"\[FeatureSpec\(dtype=DataType\.STRING, name='input_feature_0'\), FeatureSpec\(dtype=DataType\.STRING, "
            r"name='unexpected_feature'\)\] but got \{'FEATURE1': StringType\(\)\}"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.monitor_sql_client.validate_source_table(
                source_table_name=self.test_source_table_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                id_columns=[sql_identifier.SqlIdentifier("ID")],
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                model_function=model_manifest_schema.ModelFunctionInfo(
                    name="PREDICT",
                    target_method="predict",
                    target_method_function_type="FUNCTION",
                    signature=model_signature.ModelSignature(
                        inputs=[
                            model_signature.FeatureSpec("input_feature_0", model_signature.DataType.STRING),
                            model_signature.FeatureSpec("unexpected_feature", model_signature.DataType.STRING),
                        ],
                        outputs=[],
                    ),
                    is_partitioned=False,
                ),
            )
        self.m_session.finalize()

    def test_validate_monitor_warehouse(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SHOW WAREHOUSES LIKE '{self.test_wh_name}'""",
            result=mock_data_frame.MockDataFrame([]),
        )
        with self.assertRaisesRegex(ValueError, f"Warehouse '{self.test_wh_name}' not found"):
            self.monitor_sql_client.validate_monitor_warehouse(self.test_wh_name)

    def test_validate_source_table_not_exists(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '{self.test_source_table_name}' IN SNOWML_OBSERVABILITY.DATA""",
            result=mock_data_frame.MockDataFrame([]),
        )
        expected_msg = (
            f"Table {self.test_source_table_name} does not exist in schema {self.test_db_name}.{self.test_schema_name}."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.monitor_sql_client.validate_source_table(
                source_table_name=self.test_source_table_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                id_columns=[sql_identifier.SqlIdentifier("ID")],
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                model_function=model_manifest_schema.ModelFunctionInfo(
                    name="PREDICT",
                    target_method="predict",
                    target_method_function_type="FUNCTION",
                    signature=model_signature.ModelSignature(inputs=[], outputs=[]),
                    is_partitioned=False,
                ),
            )
        self.m_session.finalize()

    def test_validate_columns_exist_in_source_table(self) -> None:
        source_table_name = self.test_source_table_name

        table_schema = {
            "feature1": types.StringType(),
            "feature2": types.StringType(),
            "feature3": types.StringType(),
            "TIMESTAMP": types.TimestampType(),
            "PREDICTION": types.DoubleType(),
            "LABEL": types.DoubleType(),
            "ID": types.StringType(),
        }
        self.monitor_sql_client._validate_columns_exist_in_source_table(
            table_schema=table_schema,
            source_table_name=source_table_name,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            id_columns=[sql_identifier.SqlIdentifier("ID")],
        )

        table_schema = {
            "feature1": types.StringType(),
            "feature2": types.StringType(),
            "feature3": types.StringType(),
            "PREDICTION": types.DoubleType(),
            "LABEL": types.DoubleType(),
            "ID": types.StringType(),
        }
        with self.assertRaisesRegex(ValueError, "Timestamp column TIMESTAMP does not exist in table MODEL_OUTPUTS"):
            self.monitor_sql_client._validate_columns_exist_in_source_table(
                table_schema=table_schema,
                source_table_name=source_table_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                id_columns=[sql_identifier.SqlIdentifier("ID")],
            )

        table_schema = {
            "feature1": types.StringType(),
            "feature2": types.StringType(),
            "feature3": types.StringType(),
            "TIMESTAMP": types.TimestampType(),
            "LABEL": types.DoubleType(),
            "ID": types.StringType(),
        }

        with self.assertRaisesRegex(
            ValueError, r"Prediction column\(s\): \['PREDICTION'\] do not exist in table MODEL_OUTPUTS."
        ):
            self.monitor_sql_client._validate_columns_exist_in_source_table(
                table_schema=table_schema,
                source_table_name=source_table_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                id_columns=[sql_identifier.SqlIdentifier("ID")],
            )

        table_schema = {
            "feature1": types.StringType(),
            "feature2": types.StringType(),
            "feature3": types.StringType(),
            "TIMESTAMP": types.TimestampType(),
            "PREDICTION": types.DoubleType(),
            "ID": types.StringType(),
        }
        with self.assertRaisesRegex(ValueError, r"Label column\(s\): \['LABEL'\] do not exist in table MODEL_OUTPUTS."):
            self.monitor_sql_client._validate_columns_exist_in_source_table(
                table_schema=table_schema,
                source_table_name=source_table_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                id_columns=[sql_identifier.SqlIdentifier("ID")],
            )

        table_schema = {
            "feature1": types.StringType(),
            "feature2": types.StringType(),
            "feature3": types.StringType(),
            "TIMESTAMP": types.TimestampType(),
            "PREDICTION": types.DoubleType(),
            "LABEL": types.DoubleType(),
        }
        with self.assertRaisesRegex(ValueError, r"ID column\(s\): \['ID'\] do not exist in table MODEL_OUTPUTS"):
            self.monitor_sql_client._validate_columns_exist_in_source_table(
                table_schema=table_schema,
                source_table_name=source_table_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                id_columns=[sql_identifier.SqlIdentifier("ID")],
            )

    def test_validate_column_types(self) -> None:
        self.monitor_sql_client._validate_column_types(
            table_schema={
                "PREDICTION1": types.DoubleType(),
                "PREDICTION2": types.DoubleType(),
                "LABEL1": types.DoubleType(),
                "LABEL2": types.DoubleType(),
                "ID": types.StringType(),
                "TIMESTAMP": types.TimestampType(types.TimestampTimeZone("ltz")),
            },
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            prediction_columns=[
                sql_identifier.SqlIdentifier("PREDICTION1"),
                sql_identifier.SqlIdentifier("PREDICTION2"),
            ],
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL1"), sql_identifier.SqlIdentifier("LABEL2")],
        )

    def test_validate_prediction_column_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prediction column types must be the same. Found: .*"):
            self.monitor_sql_client._validate_prediction_columns_types(
                table_schema={
                    "PREDICTION1": types.DoubleType(),
                    "PREDICTION2": types.StringType(),
                },
                prediction_columns=[
                    sql_identifier.SqlIdentifier("PREDICTION1"),
                    sql_identifier.SqlIdentifier("PREDICTION2"),
                ],
            )

    def test_validate_label_column_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "Label column types must be the same. Found:"):
            self.monitor_sql_client._validate_label_columns_types(
                table_schema={
                    "LABEL1": types.DoubleType(),
                    "LABEL2": types.StringType(),
                },
                label_columns=[sql_identifier.SqlIdentifier("LABEL1"), sql_identifier.SqlIdentifier("LABEL2")],
            )

    def test_validate_timestamp_column_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Timestamp column: TIMESTAMP must be TimestampType"):
            self.monitor_sql_client._validate_timestamp_column_type(
                table_schema={
                    "TIMESTAMP": types.StringType(),
                },
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            )

    def test_validate_id_columns_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "Id columns must all be StringType"):
            self.monitor_sql_client._validate_id_columns_types(
                table_schema={
                    "ID": types.DoubleType(),
                },
                id_columns=[
                    sql_identifier.SqlIdentifier("ID"),
                ],
            )

    def test_validate_multiple_id_columns_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "Id columns must all be StringType. Found"):
            self.monitor_sql_client._validate_id_columns_types(
                table_schema={
                    "ID1": types.StringType(),
                    "ID2": types.DecimalType(),
                },
                id_columns=[
                    sql_identifier.SqlIdentifier("ID1"),
                    sql_identifier.SqlIdentifier("ID2"),
                ],
            )

    def test_validate_id_columns_types_all_string(self) -> None:
        self.monitor_sql_client._validate_id_columns_types(
            table_schema={
                "ID1": types.StringType(36),
                "ID2": types.StringType(64),
                "ID3": types.StringType(),
            },
            id_columns=[
                sql_identifier.SqlIdentifier("ID1"),
                sql_identifier.SqlIdentifier("ID2"),
                sql_identifier.SqlIdentifier("ID3"),
            ],
        )

    def test_monitoring_dynamic_table_query_single_numeric_single_categoric(self) -> None:
        query = self.monitor_sql_client._monitoring_dynamic_table_query(
            model_name=self.test_model_name,
            model_version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            refresh_interval=ModelMonitorRefreshInterval("15 minutes"),
            aggregate_window=ModelMonitorAggregationWindow.WINDOW_1_HOUR,
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            numeric_features=[sql_identifier.SqlIdentifier("NUM_0")],
            categoric_features=[sql_identifier.SqlIdentifier("STR_COL_0")],
            prediction_columns=[sql_identifier.SqlIdentifier("OUTPUT")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
        )

        expected = f"""
        CREATE DYNAMIC TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA.{self.mon_table_name}
            TARGET_LAG = '15 minutes'
            WAREHOUSE = ML_OBS_WAREHOUSE
            REFRESH_MODE = AUTO
            INITIALIZE = ON_CREATE
        AS
        SELECT
            TIME_SLICE(TIMESTAMP, 60, 'MINUTE') timestamp,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(NUM_0),
                'count', count_if(NUM_0 is not null),
                'count_null', count_if(NUM_0 is null),
                'min', min(NUM_0),
                'max', max(NUM_0),
                'sum', sum(NUM_0)
            ) AS NUM_0,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(OUTPUT),
                'count', count_if(OUTPUT is not null),
                'count_null', count_if(OUTPUT is null),
                'min', min(OUTPUT),
                'max', max(OUTPUT),
                'sum', sum(OUTPUT)
            ) AS OUTPUT,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(LABEL),
                'count', count_if(LABEL is not null),
                'count_null', count_if(LABEL is null),
                'min', min(LABEL),
                'max', max(LABEL),
                'sum', sum(LABEL)
            ) AS LABEL,
            SNOWML_OBSERVABILITY.DATA.OBJECT_SUM(to_varchar(STR_COL_0)) AS STR_COL_0
        FROM
            MODEL_OUTPUTS
        GROUP BY
            1
        """
        self.assertEqual(query, expected)

    def test_monitoring_dynamic_table_query_multi_feature(self) -> None:
        query = self.monitor_sql_client._monitoring_dynamic_table_query(
            model_name=self.test_model_name,
            model_version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            refresh_interval=ModelMonitorRefreshInterval("15 minutes"),
            aggregate_window=ModelMonitorAggregationWindow.WINDOW_1_HOUR,
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            numeric_features=[
                sql_identifier.SqlIdentifier("NUM_0"),
                sql_identifier.SqlIdentifier("NUM_1"),
                sql_identifier.SqlIdentifier("NUM_2"),
            ],
            categoric_features=[sql_identifier.SqlIdentifier("STR_COL_0"), sql_identifier.SqlIdentifier("STR_COL_1")],
            prediction_columns=[sql_identifier.SqlIdentifier("OUTPUT")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
        )
        self.assertEqual(
            query,
            f"""
        CREATE DYNAMIC TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA.{self.mon_table_name}
            TARGET_LAG = '15 minutes'
            WAREHOUSE = ML_OBS_WAREHOUSE
            REFRESH_MODE = AUTO
            INITIALIZE = ON_CREATE
        AS
        SELECT
            TIME_SLICE(TIMESTAMP, 60, 'MINUTE') timestamp,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(NUM_0),
                'count', count_if(NUM_0 is not null),
                'count_null', count_if(NUM_0 is null),
                'min', min(NUM_0),
                'max', max(NUM_0),
                'sum', sum(NUM_0)
            ) AS NUM_0,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(NUM_1),
                'count', count_if(NUM_1 is not null),
                'count_null', count_if(NUM_1 is null),
                'min', min(NUM_1),
                'max', max(NUM_1),
                'sum', sum(NUM_1)
            ) AS NUM_1,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(NUM_2),
                'count', count_if(NUM_2 is not null),
                'count_null', count_if(NUM_2 is null),
                'min', min(NUM_2),
                'max', max(NUM_2),
                'sum', sum(NUM_2)
            ) AS NUM_2,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(OUTPUT),
                'count', count_if(OUTPUT is not null),
                'count_null', count_if(OUTPUT is null),
                'min', min(OUTPUT),
                'max', max(OUTPUT),
                'sum', sum(OUTPUT)
            ) AS OUTPUT,
            OBJECT_CONSTRUCT(
                'sketch', APPROX_PERCENTILE_ACCUMULATE(LABEL),
                'count', count_if(LABEL is not null),
                'count_null', count_if(LABEL is null),
                'min', min(LABEL),
                'max', max(LABEL),
                'sum', sum(LABEL)
            ) AS LABEL,
            SNOWML_OBSERVABILITY.DATA.OBJECT_SUM(to_varchar(STR_COL_0)) AS STR_COL_0,
            SNOWML_OBSERVABILITY.DATA.OBJECT_SUM(to_varchar(STR_COL_1)) AS STR_COL_1
        FROM
            MODEL_OUTPUTS
        GROUP BY
            1
        """,
        )

    def test_monitoring_accuracy_regression_dynamic_table_query_single_prediction(self) -> None:
        query = self.monitor_sql_client._monitoring_regression_accuracy_table_query(
            model_name=self.test_model_name,
            model_version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            refresh_interval=ModelMonitorRefreshInterval("15 minutes"),
            aggregate_window=ModelMonitorAggregationWindow.WINDOW_1_HOUR,
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            prediction_columns=[sql_identifier.SqlIdentifier("OUTPUT")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
        )
        self.assertEqual(
            query,
            f"""
        CREATE DYNAMIC TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA.{self.acc_table_name}
            TARGET_LAG = '15 minutes'
            WAREHOUSE = ML_OBS_WAREHOUSE
            REFRESH_MODE = AUTO
            INITIALIZE = ON_CREATE
        AS
        SELECT
            TIME_SLICE(TIMESTAMP, 60, 'MINUTE') timestamp,
            'class_regression' label_class,
            OBJECT_CONSTRUCT(
                'sum_difference_label_pred', sum(OUTPUT - LABEL),
                'sum_log_difference_square_label_pred',
                sum(
                    case
                        when OUTPUT > -1 and LABEL > -1
                        then pow(ln(OUTPUT + 1) - ln(LABEL + 1),2)
                        else null
                    END
                ),
                'sum_difference_squares_label_pred',
                sum(
                    pow(
                        OUTPUT - LABEL,
                        2
                    )
                ),
                'sum_absolute_regression_labels', sum(abs(LABEL)),
                'sum_absolute_percentage_error',
                sum(
                    abs(
                        div0null(
                            (OUTPUT - LABEL),
                            LABEL
                        )
                    )
                ),
                'sum_absolute_difference_label_pred',
                sum(
                    abs(OUTPUT - LABEL)
                ),
                'sum_prediction', sum(OUTPUT),
                'sum_label', sum(LABEL),
                'count', count(*)
            ) AS AGGREGATE_METRICS,
            APPROX_PERCENTILE_ACCUMULATE(OUTPUT) prediction_sketch,
            APPROX_PERCENTILE_ACCUMULATE(LABEL) label_sketch
        FROM
            MODEL_OUTPUTS
        GROUP BY
            1
        """,
        )

    def test_monitoring_accuracy_classification_probit_dynamic_table_query_single_prediction(self) -> None:
        query = self.monitor_sql_client._monitoring_classification_accuracy_table_query(
            model_name=self.test_model_name,
            model_version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            refresh_interval=ModelMonitorRefreshInterval("15 minutes"),
            aggregate_window=ModelMonitorAggregationWindow.WINDOW_1_HOUR,
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            prediction_columns=[sql_identifier.SqlIdentifier("OUTPUT")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            score_type=output_score_type.OutputScoreType.PROBITS,
        )
        self.assertEqual(
            query,
            f"""
            CREATE DYNAMIC TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA.{self.acc_table_name}
                TARGET_LAG = '15 minutes'
                WAREHOUSE = ML_OBS_WAREHOUSE
                REFRESH_MODE = AUTO
                INITIALIZE = ON_CREATE
            AS
        WITH filtered_data AS (
            SELECT
                TIMESTAMP AS timestamp,
            OUTPUT,
            LABEL,
            CASE
                WHEN LABEL = 1 THEN 'class_positive'
                ELSE 'class_negative'
            END AS label_class
            FROM
                MODEL_OUTPUTS
        )
            select
                time_slice(timestamp, 60, 'MINUTE') timestamp,
            label_class,
            OBJECT_CONSTRUCT(
                'sum_prediction', sum(OUTPUT),
                'sum_label', sum(LABEL),
                'sum_log_loss',
                CASE
                    WHEN label_class = 'class_positive' THEN sum(-ln(OUTPUT))
                    ELSE sum(-ln(1 - OUTPUT))
                END,
                'count', count(*)
            ) AS AGGREGATE_METRICS,
            APPROX_PERCENTILE_ACCUMULATE(OUTPUT) prediction_sketch,
            APPROX_PERCENTILE_ACCUMULATE(LABEL) label_sketch
            FROM
                filtered_data
            group by
                1,
                2
        """,
        )

    def test_monitoring_accuracy_classification_class_dynamic_table_query_single_prediction(self) -> None:
        query = self.monitor_sql_client._monitoring_classification_accuracy_table_query(
            model_name=self.test_model_name,
            model_version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            refresh_interval=ModelMonitorRefreshInterval("15 minutes"),
            aggregate_window=ModelMonitorAggregationWindow.WINDOW_1_HOUR,
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            prediction_columns=[sql_identifier.SqlIdentifier("OUTPUT")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            score_type=output_score_type.OutputScoreType.CLASSIFICATION,
        )
        self.assertEqual(
            query,
            f"""
            CREATE DYNAMIC TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA.{self.acc_table_name}
                TARGET_LAG = '15 minutes'
                WAREHOUSE = ML_OBS_WAREHOUSE
                REFRESH_MODE = AUTO
                INITIALIZE = ON_CREATE
            AS
        WITH filtered_data AS (
            SELECT
                TIMESTAMP AS timestamp,
            OUTPUT,
            LABEL,
            CASE
                WHEN LABEL = 1 THEN 'class_positive'
                ELSE 'class_negative'
            END AS label_class
            FROM
                MODEL_OUTPUTS
        )
            select
                time_slice(timestamp, 60, 'MINUTE') timestamp,
            label_class,
            OBJECT_CONSTRUCT(
                'sum_prediction', sum(OUTPUT),
                'sum_label', sum(LABEL),
                'tp', count_if(LABEL = 1 AND OUTPUT = 1),
                'tn', count_if(LABEL = 0 AND OUTPUT = 0),
                'fp', count_if(LABEL = 0 AND OUTPUT = 1),
                'fn', count_if(LABEL = 1 AND OUTPUT = 0),
                'count', count(*)
            ) AS AGGREGATE_METRICS,
            APPROX_PERCENTILE_ACCUMULATE(OUTPUT) prediction_sketch,
            APPROX_PERCENTILE_ACCUMULATE(LABEL) label_sketch
            FROM
                filtered_data
            group by
                1,
                2
        """,
        )

    def test_monitoring_accuracy_dynamic_table_query_multi_prediction(self) -> None:
        with self.assertRaises(ValueError):
            _ = self.monitor_sql_client._monitoring_accuracy_table_query(
                model_name=self.test_model_name,
                model_version_name=self.test_model_version_name,
                task=type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
                source_table_name=self.test_source_table_name,
                refresh_interval=ModelMonitorRefreshInterval("15 minutes"),
                aggregate_window=ModelMonitorAggregationWindow.WINDOW_1_HOUR,
                warehouse_name=self.test_wh_name,
                timestamp_column=self.test_timestamp_column,
                prediction_columns=[sql_identifier.SqlIdentifier("LABEL"), sql_identifier.SqlIdentifier("output_1")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL"), sql_identifier.SqlIdentifier("label_1")],
                score_type=output_score_type.OutputScoreType.REGRESSION,
            )

    def test_validate_existence_by_name(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE MONITOR_NAME = '{self.test_monitor_name}'
            """,
            result=mock_data_frame.MockDataFrame([]),
        )
        res = self.monitor_sql_client.validate_existence_by_name(self.test_monitor_name)
        self.assertFalse(res)

        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE MONITOR_NAME = '{self.test_monitor_name}'
            """,
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        FULLY_QUALIFIED_MODEL_NAME=self.test_fq_model_name,
                        MODEL_VERSION_NAME=self.test_model_version_name,
                    )
                ]
            ),
        )
        res = self.monitor_sql_client.validate_existence_by_name(self.test_monitor_name)
        self.assertTrue(res)
        self.m_session.finalize()

    def test_validate_existence(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE FULLY_QUALIFIED_MODEL_NAME = '{self.test_fq_model_name}'
            AND MODEL_VERSION_NAME = '{self.test_model_version_name}'
            """,
            result=mock_data_frame.MockDataFrame([]),
        )
        res = self.monitor_sql_client.validate_existence(self.test_fq_model_name, self.test_model_version_name)
        self.assertFalse(res)

        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE FULLY_QUALIFIED_MODEL_NAME = '{self.test_fq_model_name}'
            AND MODEL_VERSION_NAME = '{self.test_model_version_name}'
            """,
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        FULLY_QUALIFIED_MODEL_NAME=self.test_fq_model_name,
                        MODEL_VERSION_NAME=self.test_model_version_name,
                    )
                ]
            ),
        )
        res = self.monitor_sql_client.validate_existence(self.test_fq_model_name, self.test_model_version_name)
        self.assertTrue(res)

        self.m_session.finalize()

    def test_create_monitor_on_model_version(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE FULLY_QUALIFIED_MODEL_NAME = '{self.test_fq_model_name}'
            AND MODEL_VERSION_NAME = '{self.test_model_version_name}'
            """,
            result=mock_data_frame.MockDataFrame([]),
        )
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE MONITOR_NAME = '{self.test_monitor_name}'
            """,
            result=mock_data_frame.MockDataFrame([]),
        )

        self.m_session.add_mock_sql(
            query=f"""INSERT INTO SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            (MONITOR_NAME, SOURCE_TABLE_NAME, FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME,
            FUNCTION_NAME, TASK, IS_ENABLED,
            TIMESTAMP_COLUMN_NAME, PREDICTION_COLUMN_NAMES, LABEL_COLUMN_NAMES, ID_COLUMN_NAMES)
            SELECT '{self.test_monitor_name}', '{self.test_source_table_name}',
            '{self.test_fq_model_name}', '{self.test_model_version_name}', '{self.test_function_name}',
            'TABULAR_BINARY_CLASSIFICATION', TRUE,
            '{self.test_timestamp_column}', ARRAY_CONSTRUCT('{self.test_prediction_column_name}'),
            ARRAY_CONSTRUCT('{self.test_label_column_name}'), ARRAY_CONSTRUCT('{self.test_id_column_name}')""",
            result=mock_data_frame.MockDataFrame([Row(**{"number of rows inserted": 1})]),
        )

        self.monitor_sql_client.create_monitor_on_model_version(
            monitor_name=self.test_monitor_name,
            source_table_name=self.test_source_table_name,
            fully_qualified_model_name=self.test_fq_model_name,
            version_name=self.test_model_version_name,
            function_name=self.test_function_name,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            task=type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
            statement_params=None,
        )
        self.m_session.finalize()

    def test_create_monitor_on_model_version_fails_if_model_exists(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE FULLY_QUALIFIED_MODEL_NAME = '{self.test_fq_model_name}'
            AND MODEL_VERSION_NAME = '{self.test_model_version_name}'
            """,
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        FULLY_QUALIFIED_MODEL_NAME=self.test_fq_model_name,
                        MODEL_VERSION_NAME=self.test_model_version_name,
                    )
                ]
            ),
        )
        expected_msg = f"Model {self.test_fq_model_name} Version {self.test_model_version_name} is already monitored!"
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.monitor_sql_client.create_monitor_on_model_version(
                monitor_name=self.test_monitor_name,
                source_table_name=self.test_source_table_name,
                fully_qualified_model_name=self.test_fq_model_name,
                version_name=self.test_model_version_name,
                function_name=self.test_function_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                id_columns=[sql_identifier.SqlIdentifier("ID")],
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                task=type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
                statement_params=None,
            )

        self.m_session.finalize()

    def test_create_monitor_on_model_version_fails_if_monitor_name_exists(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE FULLY_QUALIFIED_MODEL_NAME = '{self.test_fq_model_name}'
            AND MODEL_VERSION_NAME = '{self.test_model_version_name}'
            """,
            result=mock_data_frame.MockDataFrame([]),
        )
        self.m_session.add_mock_sql(
            query=f"""SELECT FULLY_QUALIFIED_MODEL_NAME, MODEL_VERSION_NAME
            FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA
            WHERE MONITOR_NAME = '{self.test_monitor_name}'
            """,
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        FULLY_QUALIFIED_MODEL_NAME=self.test_fq_model_name,
                        MODEL_VERSION_NAME=self.test_model_version_name,
                    )
                ]
            ),
        )

        expected_msg = f"Model Monitor with name '{self.test_monitor_name}' already exists!"
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.monitor_sql_client.create_monitor_on_model_version(
                monitor_name=self.test_monitor_name,
                source_table_name=self.test_source_table_name,
                fully_qualified_model_name=self.test_fq_model_name,
                version_name=self.test_model_version_name,
                function_name=self.test_function_name,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                id_columns=[sql_identifier.SqlIdentifier("ID")],
                prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
                task=type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
                statement_params=None,
            )

        self.m_session.finalize()

    def test_validate_unique_columns(self) -> None:
        self.monitor_sql_client._validate_unique_columns(
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
        )

    def test_validate_unique_columns_column_used_twice(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Column names must be unique across id, timestamp, prediction, and label columns."
        ):
            self.monitor_sql_client._validate_unique_columns(
                id_columns=[sql_identifier.SqlIdentifier("ID")],
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_columns=[
                    sql_identifier.SqlIdentifier("PREDICTION"),
                    # This is a duplicate with the id column
                    sql_identifier.SqlIdentifier("ID"),
                ],
                label_columns=[sql_identifier.SqlIdentifier("LABEL")],
            )

    def test_infer_numeric_categoric_column_names(self) -> None:
        from snowflake.snowpark import types

        timestamp_col = sql_identifier.SqlIdentifier("TS_COL")
        id_col = sql_identifier.SqlIdentifier("ID_COL")
        output_column = sql_identifier.SqlIdentifier("OUTPUT")
        label_column = sql_identifier.SqlIdentifier("LABEL")
        test_schema = {
            timestamp_col: types.TimeType(),
            id_col: types.FloatType(),
            output_column: types.FloatType(),
            label_column: types.FloatType(),
            "STR_COL": types.StringType(16777216),
            "LONG_COL": types.LongType(),
            "FLOAT_COL": types.FloatType(),
            "DOUBLE_COL": types.DoubleType(),
            "BINARY_COL": types.BinaryType(),
            "ARRAY_COL": types.ArrayType(),
            "NULL_COL": types.NullType(),
        }

        expected_numeric = [
            sql_identifier.SqlIdentifier("LONG_COL"),
            sql_identifier.SqlIdentifier("FLOAT_COL"),
            sql_identifier.SqlIdentifier("DOUBLE_COL"),
        ]
        expected_categoric = [
            sql_identifier.SqlIdentifier("STR_COL"),
        ]

        numeric, categoric = monitor_sql_client._infer_numeric_categoric_feature_column_names(
            source_table_schema=test_schema,
            timestamp_column=timestamp_col,
            id_columns=[id_col],
            prediction_columns=[output_column],
            label_columns=[label_column],
        )
        self.assertListEqual(expected_numeric, numeric)
        self.assertListEqual(expected_categoric, categoric)

    def test_initialize_baseline_table(self) -> None:
        mocked_table_out = mock.MagicMock(name="schema")
        self.m_session.table = mock.MagicMock(name="table", return_value=mocked_table_out)
        mocked_table_out.schema = mock.MagicMock(name="schema")
        mocked_table_out.schema.fields = [
            types.StructField(self.test_timestamp_column, types.TimestampType()),
            types.StructField(self.test_prediction_column_name, types.DoubleType()),
            types.StructField(self.test_label_column_name, types.DoubleType()),
            types.StructField(self.test_id_column_name, types.StringType()),
        ]

        self.m_session.add_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA._SNOWML_OBS_BASELINE_"""
            f"""{self.test_model_name}_{self.test_model_version_name}"""
            f"""(PREDICTION DOUBLE, LABEL DOUBLE)""",
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        name="PREDICTION",
                        type="DOUBLE",
                    ),
                    Row(
                        name="LABEL",
                        type="DOUBLE",
                    ),
                ]
            ),
        )

        self.monitor_sql_client.initialize_baseline_table(
            model_name=self.test_model_name,
            version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            columns_to_drop=[self.test_id_column_name, self.test_timestamp_column],
        )

    def test_materialize_baseline_dataframe(self) -> None:
        mocked_dataframe = mock_data_frame.MockDataFrame(
            [
                Row(TIMESTAMP="2022-01-01 00:00:00", PREDICTION=0.8, LABEL=1.0, ID="12345"),
                Row(TIMESTAMP="2022-01-02 00:00:00", PREDICTION=0.6, LABEL=0.0, ID="67890"),
            ]
        )
        self.m_session.add_mock_sql(
            f"SHOW TABLES LIKE '{self.test_baseline_table_name_sql}' IN SNOWML_OBSERVABILITY.DATA",
            mock_data_frame.MockDataFrame([Row(name=self.test_baseline_table_name_sql)]),
        )

        mocked_dataframe.write = mock.MagicMock(name="write")
        save_as_table = mock.MagicMock(name="save_as_table")
        mocked_dataframe.write.mode = mock.MagicMock(name="mode", return_value=save_as_table)

        self.monitor_sql_client.materialize_baseline_dataframe(
            baseline_df=cast(DataFrame, mocked_dataframe),
            fully_qualified_model_name=self.test_model_name,
            model_version_name=self.test_model_version_name,
        )

        mocked_dataframe.write.mode.assert_called_once_with("truncate")
        save_as_table.save_as_table.assert_called_once_with(
            [self.test_db_name, self.test_schema_name, self.test_baseline_table_name_sql],
            statement_params=mock.ANY,
        )

    def test_materialize_baseline_dataframe_table_not_exists(self) -> None:
        mocked_dataframe = mock_data_frame.MockDataFrame(
            [
                Row(TIMESTAMP="2022-01-01 00:00:00", PREDICTION=0.8, LABEL=1.0, ID="12345"),
                Row(TIMESTAMP="2022-01-02 00:00:00", PREDICTION=0.6, LABEL=0.0, ID="67890"),
            ]
        )
        self.m_session.add_mock_sql(
            f"SHOW TABLES LIKE '{self.test_baseline_table_name_sql}' IN SNOWML_OBSERVABILITY.DATA",
            mock_data_frame.MockDataFrame([]),
        )

        expected_msg = (
            f"Baseline table '{self.test_baseline_table_name_sql}' does not exist for model: "
            "'TEST_MODEL' and model_version: 'TEST_MODEL_VERSION'"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.monitor_sql_client.materialize_baseline_dataframe(
                baseline_df=cast(DataFrame, mocked_dataframe),
                fully_qualified_model_name=self.test_model_name,
                model_version_name=self.test_model_version_name,
            )

    def test_initialize_baseline_table_different_data_kinds(self) -> None:
        mocked_table_out = mock.MagicMock(name="schema")
        self.m_session.table = mock.MagicMock(name="table", return_value=mocked_table_out)
        mocked_table_out.schema = mock.MagicMock(name="schema")
        mocked_table_out.schema.fields = [
            types.StructField(self.test_timestamp_column, types.TimestampType()),
            types.StructField(self.test_prediction_column_name, types.DoubleType()),
            types.StructField(self.test_label_column_name, types.DoubleType()),
            types.StructField(self.test_id_column_name, types.StringType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE1"), types.StringType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE2"), types.DoubleType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE3"), types.FloatType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE4"), types.DecimalType(38, 9)),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE5"), types.IntegerType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE6"), types.LongType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE7"), types.ShortType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE8"), types.BinaryType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE9"), types.BooleanType()),
            types.StructField(sql_identifier.SqlIdentifier("FEATURE10"), types.TimestampType()),
            types.StructField(
                sql_identifier.SqlIdentifier("FEATURE11"), types.TimestampType(types.TimestampTimeZone("ltz"))
            ),
            types.StructField(
                sql_identifier.SqlIdentifier("FEATURE12"), types.TimestampType(types.TimestampTimeZone("ntz"))
            ),
            types.StructField(
                sql_identifier.SqlIdentifier("FEATURE13"), types.TimestampType(types.TimestampTimeZone("tz"))
            ),
        ]

        self.m_session.add_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.DATA._SNOWML_OBS_BASELINE_"""
            f"""{self.test_model_name}_{self.test_model_version_name}"""
            f"""(PREDICTION DOUBLE, LABEL DOUBLE,
            FEATURE1 STRING, FEATURE2 DOUBLE, FEATURE3 FLOAT, FEATURE4 NUMBER(38, 9), FEATURE5 INT,
            FEATURE6 BIGINT, FEATURE7 SMALLINT, FEATURE8 BINARY, FEATURE9 BOOLEAN, FEATURE10 TIMESTAMP,
            FEATURE11 TIMESTAMP_LTZ, FEATURE12 TIMESTAMP_NTZ, FEATURE13 TIMESTAMP_TZ)""",
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        name="PREDICTION",
                        type="DOUBLE",
                    ),
                    Row(
                        name="LABEL",
                        type="DOUBLE",
                    ),
                    Row(
                        name="FEATURE1",
                        type="STRING",
                    ),
                    Row(
                        name="FEATURE2",
                        type="DOUBLE",
                    ),
                    Row(
                        name="FEATURE3",
                        type="FLOAT",
                    ),
                    Row(
                        name="FEATURE4",
                        type="NUMBER",
                    ),
                    Row(
                        name="FEATURE5",
                        type="INTEGER",
                    ),
                    Row(
                        name="FEATURE6",
                        type="INTEGER",
                    ),
                    Row(
                        name="FEATURE7",
                        type="INTEGER",
                    ),
                    Row(
                        name="FEATURE8",
                        type="BINARY",
                    ),
                    Row(
                        name="FEATURE9",
                        type="BOOLEAN",
                    ),
                    Row(
                        name="FEATURE10",
                        type="TIMESTAMP",
                    ),
                    Row(
                        name="FEATURE11",
                        type="TIMESTAMP_LTZ",
                    ),
                    Row(
                        name="FEATURE12",
                        type="TIMESTAMP_NTZ",
                    ),
                    Row(
                        name="FEATURE13",
                        type="TIMESTAMP_TZ",
                    ),
                ]
            ),
        )

        self.monitor_sql_client.initialize_baseline_table(
            model_name=self.test_model_name,
            version_name=self.test_model_version_name,
            source_table_name=self.test_source_table_name,
            columns_to_drop=[self.test_timestamp_column, self.test_id_column_name],
        )

    def test_get_model_monitor_by_model_version(self) -> None:
        model_db = sql_identifier.SqlIdentifier("MODEL_DB")
        model_schema = sql_identifier.SqlIdentifier("MODEL_SCHEMA")
        self.m_session.add_mock_sql(
            f"""SELECT {monitor_sql_client.MONITOR_NAME_COL_NAME}, {monitor_sql_client.FQ_MODEL_NAME_COL_NAME},
            {monitor_sql_client.VERSION_NAME_COL_NAME}, {monitor_sql_client.FUNCTION_NAME_COL_NAME}
            FROM {self.test_db_name}.{self.test_schema_name}.{monitor_sql_client.SNOWML_MONITORING_METADATA_TABLE_NAME}
            WHERE {monitor_sql_client.FQ_MODEL_NAME_COL_NAME} = '{model_db}.{model_schema}.{self.test_model_name}'
            AND {monitor_sql_client.VERSION_NAME_COL_NAME} = '{self.test_model_version_name}'""",
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        MONITOR_NAME=self.test_monitor_name,
                        FULLY_QUALIFIED_MODEL_NAME=f"{model_db}.{model_schema}.{self.test_model_name}",
                        MODEL_VERSION_NAME=self.test_model_version_name,
                        FUNCTION_NAME=self.test_function_name,
                        PREDICTION_COLUMN_NAMES="[]",
                        LABEL_COLUMN_NAMES="[]",
                    )
                ]
            ),
        )
        # name, fq_model_name, version_name, function_name
        monitor_params = self.monitor_sql_client.get_model_monitor_by_model_version(
            model_db=model_db,
            model_schema=model_schema,
            model_name=self.test_model_name,
            version_name=self.test_model_version_name,
        )
        self.assertEqual(monitor_params["monitor_name"], str(self.test_monitor_name))
        self.assertEqual(
            monitor_params["fully_qualified_model_name"], f"{model_db}.{model_schema}.{self.test_model_name}"
        )
        self.assertEqual(monitor_params["version_name"], str(self.test_model_version_name))
        self.assertEqual(monitor_params["function_name"], str(self.test_function_name))

        self.m_session.finalize()  # TODO: Move to tearDown() for all tests.

    def test_get_model_monitor_by_model_version_fails_if_multiple(self) -> None:
        model_db = sql_identifier.SqlIdentifier("MODEL_DB")
        model_schema = sql_identifier.SqlIdentifier("MODEL_SCHEMA")
        self.m_session.add_mock_sql(
            f"""SELECT {monitor_sql_client.MONITOR_NAME_COL_NAME}, {monitor_sql_client.FQ_MODEL_NAME_COL_NAME},
            {monitor_sql_client.VERSION_NAME_COL_NAME}, {monitor_sql_client.FUNCTION_NAME_COL_NAME}
            FROM {self.test_db_name}.{self.test_schema_name}.{monitor_sql_client.SNOWML_MONITORING_METADATA_TABLE_NAME}
            WHERE {monitor_sql_client.FQ_MODEL_NAME_COL_NAME} = '{model_db}.{model_schema}.{self.test_model_name}'
            AND {monitor_sql_client.VERSION_NAME_COL_NAME} = '{self.test_model_version_name}'""",
            result=mock_data_frame.MockDataFrame(
                [
                    Row(
                        MONITOR_NAME=self.test_monitor_name,
                        FULLY_QUALIFIED_MODEL_NAME=f"{model_db}.{model_schema}.{self.test_model_name}",
                        MODEL_VERSION_NAME=self.test_model_version_name,
                        FUNCTION_NAME=self.test_function_name,
                    ),
                    Row(
                        MONITOR_NAME=self.test_monitor_name,
                        FULLY_QUALIFIED_MODEL_NAME=f"{model_db}.{model_schema}.{self.test_model_name}",
                        MODEL_VERSION_NAME=self.test_model_version_name,
                        FUNCTION_NAME=self.test_function_name,
                    ),
                ]
            ),
        )
        with self.assertRaisesRegex(ValueError, "Invalid state. Multiple Monitors exist for model:"):
            self.monitor_sql_client.get_model_monitor_by_model_version(
                model_db=model_db,
                model_schema=model_schema,
                model_name=self.test_model_name,
                version_name=self.test_model_version_name,
            )

        self.m_session.finalize()  # TODO: Move to tearDown() for all tests.

    def test_dashboard_udtf_queries(self) -> None:
        queries_map = self.monitor_sql_client._create_dashboard_udtf_queries(
            self.test_monitor_name,
            self.test_model_version_name,
            self.test_model_name,
            type_hints.Task.TABULAR_REGRESSION,
            output_score_type.OutputScoreType.REGRESSION,
            output_columns=[self.test_prediction_column_name],
            ground_truth_columns=[self.test_label_column_name],
        )
        self.assertIn("rmse", queries_map)
        EXPECTED_RMSE = """CREATE OR REPLACE FUNCTION TEST_RMSE()
    RETURNS TABLE(event_timestamp TIMESTAMP_NTZ, value FLOAT)
    AS
$$
WITH metric_of_interest as (
    select
    time_slice(timestamp, 1, 'hour') as event_timestamp,
    AGGREGATE_METRICS:"sum_difference_squares_label_pred" as aggregate_field,
    AGGREGATE_METRICS:"count" as "count"
    from
        SNOWML_OBSERVABILITY.DATA._SNOWML_OBS_ACCURACY__TEST_MODEL_VERSION_TEST_MODEL
), metric_combine as (
    select
        event_timestamp,
        CAST(SUM(NVL(aggregate_field, 0)) as DOUBLE) as metric_sum,
        SUM("count") as metric_count
    from
        metric_of_interest
    where
        cast(aggregate_field as varchar) not in ('inf','-inf','NaN')
    group by
        1
) select
    event_timestamp,
    SQRT(DIV0(metric_sum,metric_count)) as VALUE
from metric_combine
order by 1 desc
$$;
"""
        self.assertEqual(queries_map["rmse"], EXPECTED_RMSE)

        self.assertIn("record_count", queries_map)
        EXPECTED_RECORD_COUNT = """CREATE OR REPLACE FUNCTION TEST_PREDICTION_COUNT()
    RETURNS TABLE(event_timestamp TIMESTAMP_NTZ, count FLOAT)
    AS
    $$
SELECT
    time_slice(timestamp, 1, 'hour') as "event_timestamp",
    sum(get(PREDICTION,'count')) as count
from
    SNOWML_OBSERVABILITY.DATA._SNOWML_OBS_MONITORING__TEST_MODEL_VERSION_TEST_MODEL
group by
    1
order by
    1 desc
    $$;
"""
        self.assertEqual(queries_map["record_count"], EXPECTED_RECORD_COUNT)

    def test_get_all_model_monitor_metadata(self) -> None:
        expected_result = [Row(MONITOR_NAME="monitor")]
        self.m_session.add_mock_sql(
            query="SELECT * FROM SNOWML_OBSERVABILITY.DATA._SYSTEM_MONITORING_METADATA",
            result=mock_data_frame.MockDataFrame(expected_result),
        )
        res = self.monitor_sql_client.get_all_model_monitor_metadata()
        self.assertEqual(res, expected_result)

    def test_suspend_monitor_dynamic_tables(self) -> None:
        self.m_session.add_mock_sql(
            f"""ALTER DYNAMIC TABLE {self.test_db_name}.{self.test_schema_name}.{self.mon_table_name} SUSPEND""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.m_session.add_mock_sql(
            f"""ALTER DYNAMIC TABLE {self.test_db_name}.{self.test_schema_name}.{self.acc_table_name} SUSPEND""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.suspend_monitor_dynamic_tables(self.test_model_name, self.test_model_version_name)
        self.m_session.finalize()

    def test_resume_monitor_dynamic_tables(self) -> None:
        self.m_session.add_mock_sql(
            f"""ALTER DYNAMIC TABLE {self.test_db_name}.{self.test_schema_name}.{self.mon_table_name} RESUME""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.m_session.add_mock_sql(
            f"""ALTER DYNAMIC TABLE {self.test_db_name}.{self.test_schema_name}.{self.acc_table_name} RESUME""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.resume_monitor_dynamic_tables(self.test_model_name, self.test_model_version_name)
        self.m_session.finalize()

    def test_delete_monitor_metadata(self) -> None:
        monitor = "TEST_MONITOR"
        self.m_session.add_mock_sql(
            query=f"DELETE FROM {self.test_db_name}.{self.test_schema_name}."
            f"{monitor_sql_client.SNOWML_MONITORING_METADATA_TABLE_NAME} WHERE "
            f"{monitor_sql_client.MONITOR_NAME_COL_NAME} = '{monitor}'",
            result=mock_data_frame.MockDataFrame([]),
        )
        self.monitor_sql_client.delete_monitor_metadata(monitor)

    def test_delete_baseline_table(self) -> None:
        model = "TEST_MODEL"
        version = "TEST_VERSION"
        table = monitor_sql_client._create_baseline_table_name(model, version)
        self.m_session.add_mock_sql(
            query=f"DROP TABLE IF EXISTS {self.test_db_name}.{self.test_schema_name}.{table}",
            result=mock_data_frame.MockDataFrame([]),
        )
        self.monitor_sql_client.delete_baseline_table(model, version)

    def test_delete_dynamic_tables(self) -> None:
        model = "TEST_MODEL"
        model_id = sql_identifier.SqlIdentifier(model)
        fully_qualified_model = f"{self.test_db_name}.{self.test_schema_name}.{model}"
        version = "TEST_VERSION"
        version_id = sql_identifier.SqlIdentifier(version)
        monitoring_table = self.monitor_sql_client.get_monitoring_table_fully_qualified_name(model_id, version_id)
        accuracy_table = self.monitor_sql_client.get_accuracy_monitoring_table_fully_qualified_name(
            model_id, version_id
        )
        self.m_session.add_mock_sql(
            query=f"DROP DYNAMIC TABLE IF EXISTS {monitoring_table}",
            result=mock_data_frame.MockDataFrame([]),
        )
        self.m_session.add_mock_sql(
            query=f"DROP DYNAMIC TABLE IF EXISTS {accuracy_table}",
            result=mock_data_frame.MockDataFrame([]),
        )
        self.monitor_sql_client.delete_dynamic_tables(fully_qualified_model, version)


if __name__ == "__main__":
    absltest.main()
