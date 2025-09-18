from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring._client import model_monitor_sql_client
from snowflake.ml.monitoring._client.model_monitor_sql_client import MonitorOperation
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session, types


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
        self.monitor_sql_client = model_monitor_sql_client.ModelMonitorSQLClient(
            session, database_name=self.test_db_name, schema_name=self.test_schema_name
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
        self.monitor_sql_client.validate_source(
            source_database=None,
            source_schema=None,
            source=self.test_source_table_name,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            prediction_score_columns=[],
            prediction_class_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            actual_score_columns=[],
            actual_class_columns=[sql_identifier.SqlIdentifier("LABEL")],
        )
        self.m_session.table.assert_called_once_with(
            f"{self.test_db_name}.{self.test_schema_name}.{self.test_source_table_name}"
        )
        self.m_session.finalize()

    def test_validate_monitor_warehouse(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SHOW WAREHOUSES LIKE '{self.test_wh_name}'""",
            result=mock_data_frame.MockDataFrame([]),
        )
        with self.assertRaisesRegex(ValueError, f"Warehouse '{self.test_wh_name}' not found"):
            self.monitor_sql_client.validate_monitor_warehouse(self.test_wh_name)

    def test_validate_columns_exist_in_source_table(self) -> None:
        table_schema = {
            "feature1": types.StringType(),
            "feature2": types.StringType(),
            "feature3": types.StringType(),
            "TIMESTAMP": types.TimestampType(),
            "PREDICTION": types.DoubleType(),
            "LABEL": types.DoubleType(),
            "ID": types.StringType(),
        }
        self.monitor_sql_client._validate_columns_exist_in_source(
            source_column_schema=table_schema,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            prediction_score_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            prediction_class_columns=[],
            actual_score_columns=[sql_identifier.SqlIdentifier("LABEL")],
            actual_class_columns=[],
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
        with self.assertRaisesRegex(ValueError, "Timestamp column TIMESTAMP does not exist in source"):
            self.monitor_sql_client._validate_columns_exist_in_source(
                source_column_schema=table_schema,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_score_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                prediction_class_columns=[],
                actual_class_columns=[sql_identifier.SqlIdentifier("LABEL")],
                actual_score_columns=[],
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
            ValueError, r"Prediction Class column\(s\): \['PREDICTION'\] do not exist in source."
        ):
            self.monitor_sql_client._validate_columns_exist_in_source(
                source_column_schema=table_schema,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_class_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                prediction_score_columns=[],
                actual_class_columns=[],
                actual_score_columns=[sql_identifier.SqlIdentifier("LABEL")],
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
        with self.assertRaisesRegex(ValueError, r"Actual Class column\(s\): \['LABEL'\] do not exist in source."):
            self.monitor_sql_client._validate_columns_exist_in_source(
                source_column_schema=table_schema,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_score_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                prediction_class_columns=[],
                actual_class_columns=[sql_identifier.SqlIdentifier("LABEL")],
                actual_score_columns=[],
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
        with self.assertRaisesRegex(ValueError, r"ID column\(s\): \['ID'\] do not exist in source."):
            self.monitor_sql_client._validate_columns_exist_in_source(
                source_column_schema=table_schema,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_score_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                prediction_class_columns=[],
                actual_class_columns=[sql_identifier.SqlIdentifier("LABEL")],
                actual_score_columns=[],
                id_columns=[sql_identifier.SqlIdentifier("ID")],
            )

    def test_validate_existence_by_name(self) -> None:
        self.m_session.add_mock_sql(
            query=f"SHOW MODEL MONITORS LIKE '{self.test_monitor_name}' IN {self.test_db_name}.{self.test_schema_name}",
            result=mock_data_frame.MockDataFrame([]),
        )
        res = self.monitor_sql_client.validate_existence_by_name(
            database_name=None, schema_name=None, monitor_name=self.test_monitor_name
        )
        self.assertFalse(res)

        self.m_session.add_mock_sql(
            query=f"SHOW MODEL MONITORS LIKE '{self.test_monitor_name}' IN {self.test_db_name}.{self.test_schema_name}",
            result=mock_data_frame.MockDataFrame([Row(name=self.test_monitor_name)]),
        )
        res = self.monitor_sql_client.validate_existence_by_name(
            database_name=None, schema_name=None, monitor_name=self.test_monitor_name
        )
        self.assertTrue(res)

        self.m_session.add_mock_sql(
            query=f"SHOW MODEL MONITORS LIKE '{self.test_monitor_name}' IN NEW_DB.NEW_SCHEMA",
            result=mock_data_frame.MockDataFrame([Row(name=self.test_monitor_name)]),
        )
        res = self.monitor_sql_client.validate_existence_by_name(
            database_name=sql_identifier.SqlIdentifier("NEW_DB"),
            schema_name=sql_identifier.SqlIdentifier("NEW_SCHEMA"),
            monitor_name=self.test_monitor_name,
        )
        self.assertTrue(res)
        self.m_session.finalize()

    def test_suspend_monitor(self) -> None:
        self.m_session.add_mock_sql(
            f"""ALTER MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name} SUSPEND""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.suspend_monitor(self.test_monitor_name)
        self.m_session.finalize()

    def test_resume_monitor(self) -> None:
        self.m_session.add_mock_sql(
            f"""ALTER MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name} RESUME""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.resume_monitor(self.test_monitor_name)
        self.m_session.finalize()

    def test_drop_model_monitor(self) -> None:
        self.m_session.add_mock_sql(
            f"""DROP MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name}""",
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.drop_model_monitor(monitor_name=self.test_monitor_name)
        self.m_session.finalize()

    def test_add_segment_column(self) -> None:
        test_segment_column = sql_identifier.SqlIdentifier("CUSTOMER_SEGMENT")
        self.m_session.add_mock_sql(
            (
                f"""ALTER MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name} """
                f"""ADD SEGMENT_COLUMN={test_segment_column}"""
            ),
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.add_segment_column(
            monitor_name=self.test_monitor_name, segment_column=test_segment_column
        )
        self.m_session.finalize()

    def test_drop_segment_column(self) -> None:
        test_segment_column = sql_identifier.SqlIdentifier("CUSTOMER_SEGMENT")
        self.m_session.add_mock_sql(
            (
                f"""ALTER MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name} """
                f"""DROP SEGMENT_COLUMN={test_segment_column}"""
            ),
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.drop_segment_column(
            monitor_name=self.test_monitor_name, segment_column=test_segment_column
        )
        self.m_session.finalize()

    def test_add_custom_metric_column(self) -> None:
        test_custom_metric_column = sql_identifier.SqlIdentifier("CUSTOM_METRIC")
        self.m_session.add_mock_sql(
            (
                f"""ALTER MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name} """
                f"""ADD CUSTOM_METRIC_COLUMN={test_custom_metric_column}"""
            ),
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.add_custom_metric_column(
            monitor_name=self.test_monitor_name, custom_metric_column=test_custom_metric_column
        )
        self.m_session.finalize()

    def test_drop_custom_metric_column(self) -> None:
        test_custom_metric_column = sql_identifier.SqlIdentifier("CUSTOM_METRIC")
        self.m_session.add_mock_sql(
            (
                f"""ALTER MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name} """
                f"""DROP CUSTOM_METRIC_COLUMN={test_custom_metric_column}"""
            ),
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )
        self.monitor_sql_client.drop_custom_metric_column(
            monitor_name=self.test_monitor_name, custom_metric_column=test_custom_metric_column
        )
        self.m_session.finalize()

    def test_alter_monitor_validation(self) -> None:
        """Test validation logic in _alter_monitor method"""
        # Test missing target property and value for ADD operation
        with self.assertRaisesRegex(ValueError, "Target property and value must be provided for ADD operation"):
            self.monitor_sql_client._alter_monitor(
                operation=MonitorOperation.ADD,
                monitor_name=self.test_monitor_name,
                target_property=None,
                target_value=None,
            )

        # Test missing target property and value for DROP operation
        with self.assertRaisesRegex(ValueError, "Target property and value must be provided for DROP operation"):
            self.monitor_sql_client._alter_monitor(
                operation=MonitorOperation.DROP,
                monitor_name=self.test_monitor_name,
                target_property=None,
                target_value=None,
            )

        # Test invalid target property for ADD operation
        with self.assertRaisesRegex(
            ValueError, "Only CUSTOM_METRIC_COLUMN, SEGMENT_COLUMN supported as target property for ADD operation"
        ):
            self.monitor_sql_client._alter_monitor(
                operation=MonitorOperation.ADD,
                monitor_name=self.test_monitor_name,
                target_property="INVALID_PROPERTY",
                target_value=sql_identifier.SqlIdentifier("test_column"),
            )

        # Test invalid target property for DROP operation
        with self.assertRaisesRegex(
            ValueError, "Only CUSTOM_METRIC_COLUMN, SEGMENT_COLUMN supported as target property for DROP operation"
        ):
            self.monitor_sql_client._alter_monitor(
                operation=MonitorOperation.DROP,
                monitor_name=self.test_monitor_name,
                target_property="INVALID_PROPERTY",
                target_value=sql_identifier.SqlIdentifier("test_column"),
            )

    def test_validate_segment_columns_exist_in_source(self) -> None:
        """Test validation of segment columns in source table"""
        # Test successful validation with segment columns
        table_schema = {
            "TIMESTAMP": types.TimestampType(),
            "PREDICTION": types.DoubleType(),
            "LABEL": types.DoubleType(),
            "ID": types.StringType(),
            "CUSTOMER_SEGMENT": types.StringType(),
            "REGION": types.StringType(),
        }
        self.monitor_sql_client._validate_columns_exist_in_source(
            source_column_schema=table_schema,
            timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
            prediction_score_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            prediction_class_columns=[],
            actual_score_columns=[sql_identifier.SqlIdentifier("LABEL")],
            actual_class_columns=[],
            id_columns=[sql_identifier.SqlIdentifier("ID")],
            segment_columns=[sql_identifier.SqlIdentifier("CUSTOMER_SEGMENT"), sql_identifier.SqlIdentifier("REGION")],
        )

        # Test missing segment column validation
        table_schema_missing_segment = {
            "TIMESTAMP": types.TimestampType(),
            "PREDICTION": types.DoubleType(),
            "LABEL": types.DoubleType(),
            "ID": types.StringType(),
        }
        with self.assertRaisesRegex(ValueError, r"Segment column\(s\): \['CUSTOMER_SEGMENT'\] do not exist in source."):
            self.monitor_sql_client._validate_columns_exist_in_source(
                source_column_schema=table_schema_missing_segment,
                timestamp_column=sql_identifier.SqlIdentifier("TIMESTAMP"),
                prediction_score_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
                prediction_class_columns=[],
                actual_score_columns=[sql_identifier.SqlIdentifier("LABEL")],
                actual_class_columns=[],
                id_columns=[sql_identifier.SqlIdentifier("ID")],
                segment_columns=[sql_identifier.SqlIdentifier("CUSTOMER_SEGMENT")],
            )

    def test_create_model_monitor_with_segment_columns(self) -> None:
        """Test creating model monitor with segment columns"""
        self.m_session.add_mock_sql(
            f"""
            CREATE MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name}
                WITH
                    MODEL={self.test_db_name}.{self.test_schema_name}.{self.test_model_name}
                    VERSION='{self.test_model_version_name}'
                    FUNCTION='predict'
                    WAREHOUSE='{self.test_wh_name}'
                    SOURCE={self.test_db_name}.{self.test_schema_name}.{self.test_source_table_name}
                    ID_COLUMNS=('ID')
                    PREDICTION_SCORE_COLUMNS=('PREDICTION')
                    PREDICTION_CLASS_COLUMNS=()
                    ACTUAL_SCORE_COLUMNS=('LABEL')
                    ACTUAL_CLASS_COLUMNS=()
                    TIMESTAMP_COLUMN='TIMESTAMP'
                    REFRESH_INTERVAL='1 hour'
                    AGGREGATION_WINDOW='1 day'
                    SEGMENT_COLUMNS=('CUSTOMER_SEGMENT', 'REGION')
                    """,
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )

        self.monitor_sql_client.create_model_monitor(
            monitor_database=self.test_db_name,
            monitor_schema=self.test_schema_name,
            monitor_name=self.test_monitor_name,
            source_database=self.test_db_name,
            source_schema=self.test_schema_name,
            source=self.test_source_table_name,
            model_database=self.test_db_name,
            model_schema=self.test_schema_name,
            model_name=self.test_model_name,
            version_name=self.test_model_version_name,
            function_name="predict",
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            id_columns=[self.test_id_column_name],
            prediction_score_columns=[self.test_prediction_column_name],
            prediction_class_columns=[],
            actual_score_columns=[self.test_label_column_name],
            actual_class_columns=[],
            refresh_interval="1 hour",
            aggregation_window="1 day",
            segment_columns=[sql_identifier.SqlIdentifier("CUSTOMER_SEGMENT"), sql_identifier.SqlIdentifier("REGION")],
        )
        self.m_session.finalize()

    def test_create_model_monitor_with_custom_metric_columns(self) -> None:
        """Test creating model monitor with custom metric columns"""
        self.m_session.add_mock_sql(
            f"""
            CREATE MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name}
                WITH
                    MODEL={self.test_db_name}.{self.test_schema_name}.{self.test_model_name}
                    VERSION='{self.test_model_version_name}'
                    FUNCTION='predict'
                    WAREHOUSE='{self.test_wh_name}'
                    SOURCE={self.test_db_name}.{self.test_schema_name}.{self.test_source_table_name}
                    ID_COLUMNS=('ID')
                    PREDICTION_SCORE_COLUMNS=('PREDICTION')
                    PREDICTION_CLASS_COLUMNS=()
                    ACTUAL_SCORE_COLUMNS=('LABEL')
                    ACTUAL_CLASS_COLUMNS=()
                    TIMESTAMP_COLUMN='TIMESTAMP'
                    REFRESH_INTERVAL='1 hour'
                    AGGREGATION_WINDOW='1 day'
                    CUSTOM_METRIC_COLUMNS=('CUSTOM_METRIC')
                    """,
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )

        self.monitor_sql_client.create_model_monitor(
            monitor_database=self.test_db_name,
            monitor_schema=self.test_schema_name,
            monitor_name=self.test_monitor_name,
            source_database=self.test_db_name,
            source_schema=self.test_schema_name,
            source=self.test_source_table_name,
            model_database=self.test_db_name,
            model_schema=self.test_schema_name,
            model_name=self.test_model_name,
            version_name=self.test_model_version_name,
            function_name="predict",
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            id_columns=[self.test_id_column_name],
            prediction_score_columns=[self.test_prediction_column_name],
            prediction_class_columns=[],
            actual_score_columns=[self.test_label_column_name],
            actual_class_columns=[],
            refresh_interval="1 hour",
            aggregation_window="1 day",
            custom_metric_columns=[sql_identifier.SqlIdentifier("CUSTOM_METRIC")],
        )
        self.m_session.finalize()

    def test_create_model_monitor_without_segment_columns(self) -> None:
        """Test creating model monitor without segment columns (empty list)"""
        self.m_session.add_mock_sql(
            f"""
            CREATE MODEL MONITOR {self.test_db_name}.{self.test_schema_name}.{self.test_monitor_name}
                WITH
                    MODEL={self.test_db_name}.{self.test_schema_name}.{self.test_model_name}
                    VERSION='{self.test_model_version_name}'
                    FUNCTION='predict'
                    WAREHOUSE='{self.test_wh_name}'
                    SOURCE={self.test_db_name}.{self.test_schema_name}.{self.test_source_table_name}
                    ID_COLUMNS=('ID')
                    PREDICTION_SCORE_COLUMNS=('PREDICTION')
                    PREDICTION_CLASS_COLUMNS=()
                    ACTUAL_SCORE_COLUMNS=('LABEL')
                    ACTUAL_CLASS_COLUMNS=()
                    TIMESTAMP_COLUMN='TIMESTAMP'
                    REFRESH_INTERVAL='1 hour'
                    AGGREGATION_WINDOW='1 day'
                    """,
            result=mock_data_frame.MockDataFrame([Row(status="Success")]),
        )

        # Test with empty segment_columns list (should not include SEGMENT_COLUMNS in SQL)
        self.monitor_sql_client.create_model_monitor(
            monitor_database=self.test_db_name,
            monitor_schema=self.test_schema_name,
            monitor_name=self.test_monitor_name,
            source_database=self.test_db_name,
            source_schema=self.test_schema_name,
            source=self.test_source_table_name,
            model_database=self.test_db_name,
            model_schema=self.test_schema_name,
            model_name=self.test_model_name,
            version_name=self.test_model_version_name,
            function_name="predict",
            warehouse_name=self.test_wh_name,
            timestamp_column=self.test_timestamp_column,
            id_columns=[self.test_id_column_name],
            prediction_score_columns=[self.test_prediction_column_name],
            prediction_class_columns=[],
            actual_score_columns=[self.test_label_column_name],
            actual_class_columns=[],
            refresh_interval="1 hour",
            aggregation_window="1 day",
            segment_columns=[],  # Empty list
        )
        self.m_session.finalize()


if __name__ == "__main__":
    absltest.main()
