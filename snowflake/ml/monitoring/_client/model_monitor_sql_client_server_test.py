from typing import Optional, cast

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring._client import model_monitor_sql_client
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


class ModelMonitorSqlClientServerTest(absltest.TestCase):
    """Test the ModelMonitorSqlClientServer class when calling server side Model Monitor SQL."""

    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_schema = sql_identifier.SqlIdentifier("TEST_SCHEMA")
        self.test_db = sql_identifier.SqlIdentifier("TEST_DB")
        session = cast(Session, self.m_session)
        self.monitor_sql_client = model_monitor_sql_client.ModelMonitorSQLClient(
            session, database_name=self.test_db, schema_name=self.test_schema
        )

        self.model_name = sql_identifier.SqlIdentifier("MODEL")
        self.model_version = sql_identifier.SqlIdentifier("VERSION")
        self.model_function = sql_identifier.SqlIdentifier("FUNCTION")
        self.warehouse_name = sql_identifier.SqlIdentifier("WAREHOUSE")
        self.source = sql_identifier.SqlIdentifier("SOURCE")
        self.id_columns = [sql_identifier.SqlIdentifier("ID")]
        self.timestamp_column = sql_identifier.SqlIdentifier("TIMESTAMP_COLUMN")
        self.refresh_interval = "1 day"
        self.aggregation_window = "1 day"

        self.prediction_score_columns = [sql_identifier.SqlIdentifier("PRED_SCORE")]
        self.prediction_class_columns = [sql_identifier.SqlIdentifier("PRED_CLASS")]
        self.actual_score_columns = [sql_identifier.SqlIdentifier("ACTUAL_SCORE")]
        self.actual_class_columns = [sql_identifier.SqlIdentifier("ACTUAL_CLASS")]

    def tearDown(self) -> None:
        self.m_session.finalize()

    def _build_expected_create_model_monitor_sql(
        self,
        id_cols_sql: str,
        baseline: Optional[str] = None,
        db_override: Optional[str] = None,
        schema_override: Optional[str] = None,
    ) -> str:
        fq_schema = (
            f"{db_override}.{schema_override}"
            if db_override and schema_override
            else f"{self.test_db}.{self.test_schema}"
        )
        baseline_sql = f"BASELINE='{fq_schema}.{baseline}'" if baseline else ""
        return f"""
            CREATE MODEL MONITOR {fq_schema}.M
            WITH
                MODEL='{fq_schema}.{self.model_name}'
                VERSION='{self.model_version}'
                FUNCTION='{self.model_function}'
                WAREHOUSE='{self.warehouse_name}'
                SOURCE='{fq_schema}.{self.source}'
                ID_COLUMNS={id_cols_sql}
                PREDICTION_SCORE_COLUMNS=('PRED_SCORE')
                PREDICTION_CLASS_COLUMNS=('PRED_CLASS')
                ACTUAL_SCORE_COLUMNS=('ACTUAL_SCORE')
                ACTUAL_CLASS_COLUMNS=('ACTUAL_CLASS')
                TIMESTAMP_COLUMN='{self.timestamp_column}'
                REFRESH_INTERVAL='{self.refresh_interval}'
                AGGREGATION_WINDOW='{self.aggregation_window}'
                {baseline_sql}
        """

    def test_build_sql_list_from_columns(self) -> None:
        columns = [sql_identifier.SqlIdentifier("col1")]
        res = model_monitor_sql_client._build_sql_list_from_columns(columns)
        self.assertEqual(res, "('COL1')")

        columns = [sql_identifier.SqlIdentifier("col1"), sql_identifier.SqlIdentifier("col2")]
        res = model_monitor_sql_client._build_sql_list_from_columns(columns)
        self.assertEqual(res, "('COL1', 'COL2')")

        columns = []
        res = model_monitor_sql_client._build_sql_list_from_columns(columns)
        self.assertEqual(res, "()")

    def test_show_model_monitors(self) -> None:
        self.m_session.add_mock_sql(
            f"SHOW MODEL MONITORS IN {self.test_db}.{self.test_schema}",
            result=mock_data_frame.MockDataFrame([Row(name="TEST")]),
        )
        res = self.monitor_sql_client.show_model_monitors()
        self.assertEqual(res[0]["name"], "TEST")

    def test_create_model_monitor(self) -> None:
        self.m_session.add_mock_sql(
            self._build_expected_create_model_monitor_sql(id_cols_sql="('ID')"),
            result=mock_data_frame.MockDataFrame([Row(status="success")]),
        )
        self.monitor_sql_client.create_model_monitor(
            monitor_database=None,
            monitor_schema=None,
            monitor_name=sql_identifier.SqlIdentifier("m"),
            source_database=None,
            source_schema=None,
            source=self.source,
            model_database=None,
            model_schema=None,
            model_name=self.model_name,
            version_name=self.model_version,
            function_name=self.model_function,
            warehouse_name=self.warehouse_name,
            timestamp_column=self.timestamp_column,
            id_columns=self.id_columns,
            prediction_score_columns=self.prediction_score_columns,
            prediction_class_columns=self.prediction_class_columns,
            actual_score_columns=self.actual_score_columns,
            actual_class_columns=self.actual_class_columns,
            refresh_interval=self.refresh_interval,
            aggregation_window=self.aggregation_window,
        )

    def test_create_model_monitor_multiple_id_cols(self) -> None:
        self.m_session.add_mock_sql(
            self._build_expected_create_model_monitor_sql(id_cols_sql="('ID1', 'ID2')"),
            result=mock_data_frame.MockDataFrame([Row(status="success")]),
        )
        self.monitor_sql_client.create_model_monitor(
            monitor_database=None,
            monitor_schema=None,
            monitor_name=sql_identifier.SqlIdentifier("m"),
            source_database=None,
            source_schema=None,
            source=self.source,
            model_database=None,
            model_schema=None,
            model_name=self.model_name,
            version_name=self.model_version,
            function_name=self.model_function,
            warehouse_name=self.warehouse_name,
            timestamp_column=self.timestamp_column,
            id_columns=[sql_identifier.SqlIdentifier("ID1"), sql_identifier.SqlIdentifier("ID2")],
            prediction_score_columns=self.prediction_score_columns,
            prediction_class_columns=self.prediction_class_columns,
            actual_score_columns=self.actual_score_columns,
            actual_class_columns=self.actual_class_columns,
            refresh_interval=self.refresh_interval,
            aggregation_window=self.aggregation_window,
        )

    def test_create_model_monitor_empty_id_cols(self) -> None:
        self.m_session.add_mock_sql(
            self._build_expected_create_model_monitor_sql(id_cols_sql="()"),
            result=mock_data_frame.MockDataFrame([Row(status="success")]),
        )
        self.monitor_sql_client.create_model_monitor(
            monitor_database=None,
            monitor_schema=None,
            monitor_name=sql_identifier.SqlIdentifier("m"),
            source_database=None,
            source_schema=None,
            source=self.source,
            model_database=None,
            model_schema=None,
            model_name=self.model_name,
            version_name=self.model_version,
            function_name=self.model_function,
            warehouse_name=self.warehouse_name,
            timestamp_column=self.timestamp_column,
            id_columns=[],
            prediction_score_columns=self.prediction_score_columns,
            prediction_class_columns=self.prediction_class_columns,
            actual_score_columns=self.actual_score_columns,
            actual_class_columns=self.actual_class_columns,
            refresh_interval=self.refresh_interval,
            aggregation_window=self.aggregation_window,
        )

    def test_create_model_monitor_objects_in_different_schemas(self) -> None:
        override_db = sql_identifier.SqlIdentifier("OVERRIDE_DB")
        override_schema = sql_identifier.SqlIdentifier("OVERRIDE_SCHEMA")
        self.m_session.add_mock_sql(
            self._build_expected_create_model_monitor_sql(
                id_cols_sql="()", baseline="BASELINE", db_override=override_db, schema_override=override_schema
            ),
            result=mock_data_frame.MockDataFrame([Row(status="success")]),
        )
        self.monitor_sql_client.create_model_monitor(
            monitor_database=override_db,
            monitor_schema=override_schema,
            monitor_name=sql_identifier.SqlIdentifier("m"),
            source_database=override_db,
            source_schema=override_schema,
            source=self.source,
            model_database=override_db,
            model_schema=override_schema,
            model_name=self.model_name,
            version_name=self.model_version,
            function_name=self.model_function,
            warehouse_name=self.warehouse_name,
            timestamp_column=self.timestamp_column,
            id_columns=[],
            prediction_score_columns=self.prediction_score_columns,
            prediction_class_columns=self.prediction_class_columns,
            actual_score_columns=self.actual_score_columns,
            actual_class_columns=self.actual_class_columns,
            refresh_interval=self.refresh_interval,
            aggregation_window=self.aggregation_window,
            baseline_database=override_db,
            baseline_schema=override_schema,
            baseline=sql_identifier.SqlIdentifier("BASELINE"),
        )


if __name__ == "__main__":
    absltest.main()
