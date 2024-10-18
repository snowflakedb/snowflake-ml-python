import re
from typing import cast
from unittest import mock
from unittest.mock import patch

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.monitoring._client import model_monitor_sql_client
from snowflake.ml.monitoring._manager import model_monitor_manager
from snowflake.ml.monitoring.entities import (
    model_monitor_config,
    model_monitor_interval,
    output_score_type,
)
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


def _build_mock_model_version(
    fq_model_name: str,
    model_version_name: str,
    task: type_hints.Task = type_hints.Task.TABULAR_REGRESSION,
) -> mock.MagicMock:
    model_version = mock.MagicMock()
    model_version.fully_qualified_model_name = fq_model_name
    model_version.version_name = model_version_name

    _, _, model_name = sql_identifier.parse_fully_qualified_name(fq_model_name)
    model_version.model_name = model_name
    model_version.get_model_task.return_value = task
    model_version.show_functions.return_value = [
        model_manifest_schema.ModelFunctionInfo(
            name="PREDICT",
            target_method="predict",
            target_method_function_type="FUNCTION",
            signature=model_signature.ModelSignature(inputs=[], outputs=[]),
            is_partitioned=False,
        )
    ]
    return model_version


class ModelMonitorManagerHelpersTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_db = sql_identifier.SqlIdentifier("SNOWML_OBSERVABILITY")
        self.test_schema = sql_identifier.SqlIdentifier("METADATA")
        self.test_warehouse = "WH_TEST"
        self.test_model_name = "TEST_MODEL"
        self.test_version_name = "TEST_VERSION"
        self.test_fq_model_name = f"{self.test_db}.{self.test_schema}.{self.test_model_name}"
        self.test_source_table_name = "TEST_TABLE"

        self.test_model_version = "TEST_VERSION"
        self.test_model = "TEST_MODEL"
        self.test_fq_model_name = f"{self.test_db}.{self.test_schema}.{self.test_model}"

        m_model_version = mock.MagicMock()
        m_model_version.version_name = self.test_model_version
        m_model_version.model_name = self.test_model
        m_model_version.fully_qualified_model_name = self.test_fq_model_name
        m_model_version.get_model_task.return_value = type_hints.Task.TABULAR_REGRESSION
        self.mv = m_model_version

        self.test_monitor_config = model_monitor_config.ModelMonitorConfig(
            model_version=self.mv,
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
        )
        self.test_table_config = model_monitor_config.ModelMonitorTableConfig(
            prediction_columns=["A"],
            label_columns=["B"],
            id_columns=["C"],
            timestamp_column="D",
            source_table=self.test_source_table_name,
        )
        self._init_mm_with_patch()

    def tearDown(self) -> None:
        self.m_session.finalize()

    def test_validate_monitor_config(self) -> None:
        malformed_refresh = "BAD BAD"
        mm_config = model_monitor_config.ModelMonitorConfig(
            model_version=_build_mock_model_version(self.test_fq_model_name, self.test_version_name),
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
            refresh_interval=malformed_refresh,
        )
        with self.assertRaisesRegex(ValueError, "Failed to parse refresh interval with exception"):
            self.mm._validate_monitor_config_or_raise(self.test_table_config, mm_config)

    def test_validate_name_constraints(self) -> None:
        model_name, version_name = "M" * 231, "V"
        m_model_version = _build_mock_model_version(model_name, version_name)
        with self.assertRaisesRegex(
            ValueError,
            "Model name and version name exceeds maximum length of 231",
        ):
            model_monitor_manager._validate_name_constraints(m_model_version)

        good_model_name = "M" * 230
        m_model_version = _build_mock_model_version(good_model_name, version_name)
        model_monitor_manager._validate_name_constraints(m_model_version)

    def test_fetch_task(self) -> None:
        model_version = _build_mock_model_version(
            self.test_fq_model_name, self.test_version_name, task=type_hints.Task.UNKNOWN
        )
        expected_msg = "Registry model must be logged with task in order to be monitored."
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.mm._fetch_task_from_model_version(model_version)

    def test_validate_function_name(self) -> None:
        model_version = _build_mock_model_version(self.test_fq_model_name, self.test_version_name)
        bad_function_name = "not_predict"
        expected_message = (
            f"Function with name {bad_function_name} does not exist in the given model version. Found: {{'predict'}}."
        )
        with self.assertRaisesRegex(ValueError, re.escape(expected_message)):
            self.mm._get_and_validate_model_function_from_model_version(bad_function_name, model_version)

    def test_get_monitor_by_model_version(self) -> None:
        self.mock_model_monitor_sql_client.validate_existence.return_value = True
        self.mock_model_monitor_sql_client.get_model_monitor_by_model_version.return_value = (
            model_monitor_sql_client._ModelMonitorParams(
                monitor_name="TEST_MONITOR_NAME",
                fully_qualified_model_name=self.test_fq_model_name,
                version_name=self.test_model_version,
                function_name="PREDICT",
                prediction_columns=[],
                label_columns=[],
            )
        )
        model_monitor = self.mm.get_monitor_by_model_version(self.mv)

        self.mock_model_monitor_sql_client.validate_existence.assert_called_once_with(
            self.test_fq_model_name, self.test_model_version, None
        )
        self.mock_model_monitor_sql_client.get_model_monitor_by_model_version.assert_called_once_with(
            model_db=self.test_db,
            model_schema=self.test_schema,
            model_name=self.test_model,
            version_name=self.test_model_version,
            statement_params=None,
        )
        self.assertEqual(model_monitor.name, "TEST_MONITOR_NAME")
        self.assertEqual(model_monitor._function_name, "PREDICT")

    def test_get_monitor_by_model_version_not_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "ModelMonitor not found for model version"):
            with mock.patch.object(
                self.mm._model_monitor_client, "validate_existence", return_value=False
            ) as mock_validate_existence:
                self.mm.get_monitor_by_model_version(self.mv)

        mock_validate_existence.assert_called_once_with(self.test_fq_model_name, self.test_model_version, None)

    def _init_mm_with_patch(self) -> None:
        patcher = patch("snowflake.ml.monitoring._client.model_monitor_sql_client.ModelMonitorSQLClient", autospec=True)
        self.addCleanup(patcher.stop)
        self.mock_model_monitor_sql_client_class = patcher.start()
        self.mock_model_monitor_sql_client = self.mock_model_monitor_sql_client_class.return_value
        self.mm = model_monitor_manager.ModelMonitorManager(
            cast(Session, self.m_session), database_name=self.test_db, schema_name=self.test_schema
        )


class ModelMonitorManagerTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_warehouse = "TEST_WAREHOUSE"
        self.test_db = sql_identifier.SqlIdentifier("TEST_DB")
        self.test_schema = sql_identifier.SqlIdentifier("TEST_SCHEMA")

        self.test_model_version = "TEST_VERSION"
        self.test_model = "TEST_MODEL"
        self.test_fq_model_name = f"db1.schema1.{self.test_model}"
        self.test_source_table_name = "TEST_TABLE"

        self.mv = _build_mock_model_version(self.test_fq_model_name, self.test_model_version)

        self.test_table_config = model_monitor_config.ModelMonitorTableConfig(
            prediction_columns=["PREDICTION"],
            label_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source_table=self.test_source_table_name,
        )
        self.test_monitor_config = model_monitor_config.ModelMonitorConfig(
            model_version=self.mv,
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
        )
        session = cast(Session, self.m_session)
        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '_SYSTEM_MONITORING_METADATA' IN {self.test_db}.{self.test_schema}""",
            result=mock_data_frame.MockDataFrame([Row(name="_SYSTEM_MONITORING_METADATA")]),
        )
        self.mm = model_monitor_manager.ModelMonitorManager(
            session, database_name=self.test_db, schema_name=self.test_schema
        )
        self.mm._model_monitor_client = mock.MagicMock()

    def tearDown(self) -> None:
        self.m_session.finalize()

    def test_manual_init(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {self.test_db}.{self.test_schema}._SYSTEM_MONITORING_METADATA
            (MONITOR_NAME VARCHAR, SOURCE_TABLE_NAME VARCHAR, FULLY_QUALIFIED_MODEL_NAME VARCHAR,
            MODEL_VERSION_NAME VARCHAR, FUNCTION_NAME VARCHAR, TASK VARCHAR, IS_ENABLED BOOLEAN,
            TIMESTAMP_COLUMN_NAME VARCHAR, PREDICTION_COLUMN_NAMES ARRAY,
            LABEL_COLUMN_NAMES ARRAY, ID_COLUMN_NAMES ARRAY)
            """,
            result=mock_data_frame.MockDataFrame([Row(status="Table successfully created.")]),
        )
        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '_SYSTEM_MONITORING_METADATA' IN {self.test_db}.{self.test_schema}""",
            result=mock_data_frame.MockDataFrame([Row(name="_SYSTEM_MONITORING_METADATA")]),
        )
        session = cast(Session, self.m_session)
        model_monitor_manager.ModelMonitorManager.setup(session, self.test_db, self.test_schema)
        model_monitor_manager.ModelMonitorManager(
            session, database_name=self.test_db, schema_name=self.test_schema, create_if_not_exists=False
        )

    def test_init_fails_not_initialized(self) -> None:
        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '_SYSTEM_MONITORING_METADATA' IN {self.test_db}.{self.test_schema}""",
            result=mock_data_frame.MockDataFrame([]),
        )
        session = cast(Session, self.m_session)
        expected_msg = "Monitoring has not been setup. Set create_if_not_exists or call ModelMonitorManager.setup"

        with self.assertRaisesRegex(ValueError, expected_msg):
            model_monitor_manager.ModelMonitorManager(
                session, database_name=self.test_db, schema_name=self.test_schema, create_if_not_exists=False
            )

    def test_add_monitor(self) -> None:
        with mock.patch.object(
            self.mm._model_monitor_client, "validate_source_table"
        ) as mock_validate_source_table, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ) as mock_get_model_task, mock.patch.object(
            self.mm._model_monitor_client,
            "get_score_type",
            return_value=output_score_type.OutputScoreType.REGRESSION,
        ) as mock_get_score_type, mock.patch.object(
            self.mm._model_monitor_client, "create_monitor_on_model_version", return_value=None
        ) as mock_create_monitor_on_model_version, mock.patch.object(
            self.mm._model_monitor_client, "create_dynamic_tables_for_monitor", return_value=None
        ) as mock_create_dynamic_tables_for_monitor, mock.patch.object(
            self.mm._model_monitor_client,
            "initialize_baseline_table",
            return_value=None,
        ) as mock_initialize_baseline_table:
            self.mm.add_monitor("TEST", self.test_table_config, self.test_monitor_config)
            mock_validate_source_table.assert_called_once_with(
                source_table_name=self.test_source_table_name,
                timestamp_column="TS",
                prediction_columns=["PREDICTION"],
                label_columns=["LABEL"],
                id_columns=["ID"],
                model_function=self.mv.show_functions()[0],
            )
            mock_get_model_task.assert_called_once()
            mock_get_score_type.assert_called_once()
            mock_create_monitor_on_model_version.assert_called_once_with(
                monitor_name=sql_identifier.SqlIdentifier("TEST"),
                source_table_name=sql_identifier.SqlIdentifier(self.test_source_table_name),
                fully_qualified_model_name=self.test_fq_model_name,
                version_name=sql_identifier.SqlIdentifier(self.test_model_version),
                function_name="predict",
                timestamp_column="TS",
                prediction_columns=["PREDICTION"],
                label_columns=["LABEL"],
                id_columns=["ID"],
                task=type_hints.Task.TABULAR_REGRESSION,
                statement_params=None,
            )
            mock_create_dynamic_tables_for_monitor.assert_called_once_with(
                model_name="TEST_MODEL",
                model_version_name="TEST_VERSION",
                task=type_hints.Task.TABULAR_REGRESSION,
                source_table_name=self.test_source_table_name,
                refresh_interval=model_monitor_interval.ModelMonitorRefreshInterval("1 days"),
                aggregation_window=model_monitor_interval.ModelMonitorAggregationWindow.WINDOW_1_DAY,
                warehouse_name="TEST_WAREHOUSE",
                timestamp_column="TS",
                id_columns=["ID"],
                prediction_columns=["PREDICTION"],
                label_columns=["LABEL"],
                score_type=output_score_type.OutputScoreType.REGRESSION,
            )
            mock_initialize_baseline_table.assert_called_once_with(
                model_name="TEST_MODEL",
                version_name="TEST_VERSION",
                source_table_name=self.test_source_table_name,
                columns_to_drop=[self.test_table_config.timestamp_column, *self.test_table_config.id_columns],
                statement_params=None,
            )

    def test_add_monitor_fails_no_task(self) -> None:
        with mock.patch.object(
            self.mm._model_monitor_client, "validate_source_table"
        ) as mock_validate_source_table, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.UNKNOWN
        ):
            with self.assertRaisesRegex(
                ValueError, "Registry model must be logged with task in order to be monitored."
            ):
                self.mm.add_monitor("TEST", self.test_table_config, self.test_monitor_config)
                mock_validate_source_table.assert_called_once()

    def test_add_monitor_fails_multiple_predictions(self) -> None:
        bad_table_config = model_monitor_config.ModelMonitorTableConfig(
            source_table=self.test_source_table_name,
            prediction_columns=["PREDICTION1", "PREDICTION2"],
            label_columns=["LABEL1", "LABEL2"],
            id_columns=["ID"],
            timestamp_column="TIMESTAMP",
        )
        expected_error = "Multiple Output columns are not supported in monitoring"
        with self.assertRaisesRegex(ValueError, expected_error):
            self.mm.add_monitor("test", bad_table_config, self.test_monitor_config)
        self.m_session.finalize()

    def test_add_monitor_fails_column_lengths_do_not_match(self) -> None:
        bad_table_config = model_monitor_config.ModelMonitorTableConfig(
            source_table=self.test_source_table_name,
            prediction_columns=["PREDICTION"],
            label_columns=["LABEL1", "LABEL2"],
            id_columns=["ID"],
            timestamp_column="TIMESTAMP",
        )
        expected_msg = "Prediction and Label column names must be of the same length."
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.mm.add_monitor(
                "test",
                bad_table_config,
                self.test_monitor_config,
            )

        self.m_session.finalize()

    def test_delete_monitor(self) -> None:
        monitor = "TEST"
        model = "TEST"
        version = "V1"
        monitor_params = model_monitor_sql_client._ModelMonitorParams(
            monitor_name=monitor,
            fully_qualified_model_name=f"TEST_DB.TEST_SCHEMA.{model}",
            version_name=version,
            function_name="predict",
            prediction_columns=[sql_identifier.SqlIdentifier("PREDICTION")],
            label_columns=[sql_identifier.SqlIdentifier("LABEL")],
        )
        with mock.patch.object(
            self.mm._model_monitor_client, "get_model_monitor_by_name", return_value=monitor_params
        ) as mock_get_model_monitor_by_name, mock.patch.object(
            self.mm._model_monitor_client, "delete_monitor_metadata"
        ) as mock_delete_monitor_metadata, mock.patch.object(
            self.mm._model_monitor_client, "delete_baseline_table"
        ) as mock_delete_baseline_table, mock.patch.object(
            self.mm._model_monitor_client, "delete_dynamic_tables"
        ) as mock_delete_dynamic_tables:
            self.mm.delete_monitor(monitor)
            mock_get_model_monitor_by_name.assert_called_once_with(monitor)
            mock_delete_monitor_metadata.assert_called_once_with(sql_identifier.SqlIdentifier(monitor))
            mock_delete_baseline_table.assert_called_once_with(model, version)
            mock_delete_dynamic_tables.assert_called_once_with(model, version)


if __name__ == "__main__":
    absltest.main()
