import re
from typing import cast
from unittest import mock
from unittest.mock import patch

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.monitoring._manager import model_monitor_manager
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.test_utils import mock_session
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
        self.test_source_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["A"],
            actual_score_columns=["B"],
            id_columns=["C"],
            timestamp_column="D",
            source=self.test_source_table_name,
        )
        self._init_mm_with_patch()

    def tearDown(self) -> None:
        self.m_session.finalize()

    def test_validate_task_from_model_version(self) -> None:
        model_version = _build_mock_model_version(
            self.test_fq_model_name, self.test_version_name, task=type_hints.Task.UNKNOWN
        )
        expected_msg = "Registry model must be logged with task in order to be monitored."
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.mm._validate_task_from_model_version(model_version)

    def test_validate_function_name(self) -> None:
        model_version = _build_mock_model_version(self.test_fq_model_name, self.test_version_name)
        bad_function_name = "not_predict"
        expected_message = (
            f"Function with name {bad_function_name} does not exist in the given model version. Found: {{'predict'}}."
        )
        with self.assertRaisesRegex(ValueError, re.escape(expected_message)):
            self.mm._validate_model_function_from_model_version(bad_function_name, model_version)

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
        self.test_fq_model_name = f"model_db.model_schema.{self.test_model}"
        self.test_source_table_name = "TEST_TABLE"

        self.mv = _build_mock_model_version(self.test_fq_model_name, self.test_model_version)

        self.test_table_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["PREDICTION"],
            actual_score_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source=self.test_source_table_name,
        )
        self.test_monitor_config = model_monitor_config.ModelMonitorConfig(
            model_version=self.mv,
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
        )
        session = cast(Session, self.m_session)
        self.mm = model_monitor_manager.ModelMonitorManager(
            session, database_name=self.test_db, schema_name=self.test_schema
        )
        self.mm._model_monitor_client = mock.MagicMock()

    def tearDown(self) -> None:
        self.m_session.finalize()

    def test_show_monitors(self) -> None:
        with mock.patch.object(
            self.mm._model_monitor_client, "show_model_monitors", return_value=[]
        ) as mock_show_model_monitors:
            self.mm.show_model_monitors()
            mock_show_model_monitors.assert_called_once_with(statement_params=None)

    def test_get_monitor_by_model_version(self) -> None:
        with mock.patch.object(
            self.mm._model_monitor_client, "show_model_monitors", return_value=[]
        ) as mock_show_model_monitors:
            with self.assertRaisesRegex(ValueError, "Unable to find model monitor for the given model version."):
                self.mm.get_monitor_by_model_version(self.mv)
            mock_show_model_monitors.assert_called_once_with(statement_params=None)

        mock_return = [Row(name="TEST", model='{"model_name": "TEST_MODEL", "version_name": "TEST_VERSION"}')]
        with mock.patch.object(
            self.mm._model_monitor_client, "show_model_monitors", return_value=mock_return
        ) as mock_show_model_monitors:
            m = self.mm.get_monitor_by_model_version(self.mv)
            mock_show_model_monitors.assert_called_once_with(statement_params=None)
            self.assertEqual(m.name, "TEST")

    def test_add_monitor(self) -> None:
        with mock.patch.object(
            self.mm._model_monitor_client, "validate_source"
        ) as mock_validate_source, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ) as mock_get_model_task, mock.patch.object(
            self.mm._model_monitor_client, "create_model_monitor", return_value=None
        ) as mock_create_model_monitor:
            self.mm.add_monitor("TEST", self.test_table_config, self.test_monitor_config)
            mock_validate_source.assert_called_once_with(
                source_database=None,
                source_schema=None,
                source=self.test_source_table_name,
                timestamp_column="TS",
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                id_columns=["ID"],
                segment_columns=[],
                custom_metric_columns=[],
            )
            mock_get_model_task.assert_called_once()
            mock_create_model_monitor.assert_called_once_with(
                monitor_database=None,
                monitor_schema=None,
                monitor_name=sql_identifier.SqlIdentifier("TEST"),
                source_database=None,
                source_schema=None,
                source=sql_identifier.SqlIdentifier(self.test_source_table_name),
                model_database=sql_identifier.SqlIdentifier("MODEL_DB"),
                model_schema=sql_identifier.SqlIdentifier("MODEL_SCHEMA"),
                model_name=self.test_model,
                version_name=sql_identifier.SqlIdentifier(self.test_model_version),
                function_name="predict",
                warehouse_name=sql_identifier.SqlIdentifier(self.test_warehouse),
                timestamp_column="TS",
                id_columns=["ID"],
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                segment_columns=[],
                custom_metric_columns=[],
                refresh_interval="1 hour",
                aggregation_window="1 day",
                baseline_database=None,
                baseline_schema=None,
                baseline=None,
                timestamp_custom_metric_database=None,
                timestamp_custom_metric_schema=None,
                timestamp_custom_metric_table=None,
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

    def test_add_monitor_fully_qualified_monitor_name(self) -> None:
        with mock.patch.object(self.mm._model_monitor_client, "validate_source_table"), mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ), mock.patch.object(self.mm._model_monitor_client, "create_model_monitor") as mock_create_model_monitor:
            self.mm.add_monitor("TEST_DB.TEST_SCHEMA.TEST", self.test_table_config, self.test_monitor_config)
            mock_create_model_monitor.assert_called_once_with(
                monitor_database=sql_identifier.SqlIdentifier("TEST_DB"),
                monitor_schema=sql_identifier.SqlIdentifier("TEST_SCHEMA"),
                monitor_name=sql_identifier.SqlIdentifier("TEST"),
                source_database=None,
                source_schema=None,
                source=sql_identifier.SqlIdentifier(self.test_source_table_name),
                model_database=sql_identifier.SqlIdentifier("MODEL_DB"),
                model_schema=sql_identifier.SqlIdentifier("MODEL_SCHEMA"),
                model_name=self.test_model,
                version_name=sql_identifier.SqlIdentifier(self.test_model_version),
                function_name="predict",
                warehouse_name=sql_identifier.SqlIdentifier(self.test_warehouse),
                timestamp_column="TS",
                id_columns=["ID"],
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                segment_columns=[],
                custom_metric_columns=[],
                refresh_interval="1 hour",
                aggregation_window="1 day",
                baseline_database=None,
                baseline_schema=None,
                baseline=None,
                timestamp_custom_metric_database=None,
                timestamp_custom_metric_schema=None,
                timestamp_custom_metric_table=None,
                statement_params=None,
            )

    def test_delete_monitor(self) -> None:
        monitor = "TEST"
        with mock.patch.object(self.mm._model_monitor_client, "drop_model_monitor") as mock_drop_model_monitor:
            self.mm.delete_monitor(monitor)
            mock_drop_model_monitor.assert_called_once_with(
                database_name=None, schema_name=None, monitor_name="TEST", statement_params=mock.ANY
            )

        monitor = "TEST_DB.TEST_SCHEMA.TEST"
        with mock.patch.object(self.mm._model_monitor_client, "drop_model_monitor") as mock_drop_model_monitor:
            self.mm.delete_monitor(monitor)
            mock_drop_model_monitor.assert_called_once_with(
                database_name="TEST_DB", schema_name="TEST_SCHEMA", monitor_name="TEST", statement_params=mock.ANY
            )

    def test_add_monitor_objects_in_different_schemas(self) -> None:
        source_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["PREDICTION"],
            actual_score_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source="SOURCE_DB.SOURCE_SCHEMA.SOURCE",
            baseline="BASELINE_DB.BASELINE_SCHEMA.BASELINE",
        )
        monitor_config = model_monitor_config.ModelMonitorConfig(
            model_version=_build_mock_model_version("MODEL_DB.MODEL_SCHEMA.MODEL", self.test_model_version),
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
        )
        with mock.patch.object(self.mm._model_monitor_client, "validate_source_table"), mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ), mock.patch.object(
            self.mm._model_monitor_client, "create_model_monitor", return_value=None
        ) as mock_create_model_monitor:
            self.mm.add_monitor("MONITOR_DB.MONITOR_SCHEMA.MONITOR", source_config, monitor_config)
            mock_create_model_monitor.assert_called_once_with(
                monitor_database=sql_identifier.SqlIdentifier("MONITOR_DB"),
                monitor_schema=sql_identifier.SqlIdentifier("MONITOR_SCHEMA"),
                monitor_name=sql_identifier.SqlIdentifier("MONITOR"),
                source_database=sql_identifier.SqlIdentifier("SOURCE_DB"),
                source_schema=sql_identifier.SqlIdentifier("SOURCE_SCHEMA"),
                source=sql_identifier.SqlIdentifier("SOURCE"),
                model_database=sql_identifier.SqlIdentifier("MODEL_DB"),
                model_schema=sql_identifier.SqlIdentifier("MODEL_SCHEMA"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier(self.test_model_version),
                function_name="predict",
                warehouse_name=sql_identifier.SqlIdentifier(self.test_warehouse),
                timestamp_column="TS",
                id_columns=["ID"],
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                segment_columns=[],
                custom_metric_columns=[],
                refresh_interval="1 hour",
                aggregation_window="1 day",
                baseline_database=sql_identifier.SqlIdentifier("BASELINE_DB"),
                baseline_schema=sql_identifier.SqlIdentifier("BASELINE_SCHEMA"),
                baseline=sql_identifier.SqlIdentifier("BASELINE"),
                timestamp_custom_metric_database=None,
                timestamp_custom_metric_schema=None,
                timestamp_custom_metric_table=None,
                statement_params=None,
            )

    def test_add_monitor_with_segment_columns_happy_path(self) -> None:
        """Test that segment_columns are correctly passed through when provided."""
        source_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["PREDICTION"],
            actual_score_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source=self.test_source_table_name,
            segment_columns=["CUSTOMER_SEGMENT", "REGION"],
        )
        with mock.patch.object(
            self.mm._model_monitor_client, "validate_source"
        ) as mock_validate_source, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ) as mock_get_model_task, mock.patch.object(
            self.mm._model_monitor_client, "create_model_monitor", return_value=None
        ) as mock_create_model_monitor:
            self.mm.add_monitor("TEST", source_config, self.test_monitor_config)

            # Verify validate_source was called with segment_columns
            mock_validate_source.assert_called_once_with(
                source_database=None,
                source_schema=None,
                source=self.test_source_table_name,
                timestamp_column="TS",
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                id_columns=["ID"],
                segment_columns=["CUSTOMER_SEGMENT", "REGION"],
                custom_metric_columns=[],
            )
            mock_get_model_task.assert_called_once()

            # Verify create_model_monitor was called with segment_columns
            mock_create_model_monitor.assert_called_once_with(
                monitor_database=None,
                monitor_schema=None,
                monitor_name=sql_identifier.SqlIdentifier("TEST"),
                source_database=None,
                source_schema=None,
                source=sql_identifier.SqlIdentifier(self.test_source_table_name),
                model_database=sql_identifier.SqlIdentifier("MODEL_DB"),
                model_schema=sql_identifier.SqlIdentifier("MODEL_SCHEMA"),
                model_name=self.test_model,
                version_name=sql_identifier.SqlIdentifier(self.test_model_version),
                function_name="predict",
                warehouse_name=sql_identifier.SqlIdentifier(self.test_warehouse),
                timestamp_column="TS",
                id_columns=["ID"],
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                segment_columns=["CUSTOMER_SEGMENT", "REGION"],
                custom_metric_columns=[],
                refresh_interval="1 hour",
                aggregation_window="1 day",
                baseline_database=None,
                baseline_schema=None,
                baseline=None,
                timestamp_custom_metric_database=None,
                timestamp_custom_metric_schema=None,
                timestamp_custom_metric_table=None,
                statement_params=None,
            )

    def test_add_monitor_with_custom_metric_columns(self) -> None:
        """Test that custom_metric_columns are correctly passed through when provided."""
        source_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["PREDICTION"],
            actual_score_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source=self.test_source_table_name,
            custom_metric_columns=["CUSTOM_METRIC_1", "CUSTOM_METRIC_2"],
        )
        with mock.patch.object(
            self.mm._model_monitor_client, "validate_source"
        ) as mock_validate_source, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ) as mock_get_model_task, mock.patch.object(
            self.mm._model_monitor_client, "create_model_monitor", return_value=None
        ) as mock_create_model_monitor:
            self.mm.add_monitor("TEST", source_config, self.test_monitor_config)

            # Verify validate_source was called with custom_metric_columns
            mock_validate_source.assert_called_once_with(
                source_database=None,
                source_schema=None,
                source=self.test_source_table_name,
                timestamp_column="TS",
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                id_columns=["ID"],
                segment_columns=[],
                custom_metric_columns=["CUSTOM_METRIC_1", "CUSTOM_METRIC_2"],
            )
            mock_get_model_task.assert_called_once()

            # Verify create_model_monitor was called with custom_metric_columns
            mock_create_model_monitor.assert_called_once_with(
                monitor_database=None,
                monitor_schema=None,
                monitor_name=sql_identifier.SqlIdentifier("TEST"),
                source_database=None,
                source_schema=None,
                source=sql_identifier.SqlIdentifier(self.test_source_table_name),
                model_database=sql_identifier.SqlIdentifier("MODEL_DB"),
                model_schema=sql_identifier.SqlIdentifier("MODEL_SCHEMA"),
                model_name=self.test_model,
                version_name=sql_identifier.SqlIdentifier(self.test_model_version),
                function_name="predict",
                warehouse_name=sql_identifier.SqlIdentifier(self.test_warehouse),
                timestamp_column="TS",
                id_columns=["ID"],
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                segment_columns=[],
                custom_metric_columns=["CUSTOM_METRIC_1", "CUSTOM_METRIC_2"],
                refresh_interval="1 hour",
                aggregation_window="1 day",
                baseline_database=None,
                baseline_schema=None,
                baseline=None,
                timestamp_custom_metric_database=None,
                timestamp_custom_metric_schema=None,
                timestamp_custom_metric_table=None,
                statement_params=None,
            )

    def test_add_monitor_with_segment_columns_validation_failure(self) -> None:
        """Test that add_monitor fails when segment_columns don't exist in source."""
        source_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["PREDICTION"],
            actual_score_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source=self.test_source_table_name,
            segment_columns=["NONEXISTENT_COLUMN"],
        )

        with mock.patch.object(
            self.mm._model_monitor_client,
            "validate_source",
            side_effect=ValueError("Segment column(s): ['NONEXISTENT_COLUMN'] do not exist in source."),
        ) as mock_validate_source, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ):
            with self.assertRaisesRegex(
                ValueError, "Segment column\\(s\\): \\['NONEXISTENT_COLUMN'\\] do not exist in source\\."
            ):
                self.mm.add_monitor("TEST", source_config, self.test_monitor_config)

            mock_validate_source.assert_called_once_with(
                source_database=None,
                source_schema=None,
                source=self.test_source_table_name,
                timestamp_column="TS",
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                id_columns=["ID"],
                segment_columns=["NONEXISTENT_COLUMN"],
                custom_metric_columns=[],
            )

    def test_add_monitor_with_custom_metric_columns_validation_failure(self) -> None:
        """Test that add_monitor fails when custom_metric_columns don't exist in source."""
        source_config = model_monitor_config.ModelMonitorSourceConfig(
            prediction_score_columns=["PREDICTION"],
            actual_score_columns=["LABEL"],
            id_columns=["ID"],
            timestamp_column="TS",
            source=self.test_source_table_name,
            custom_metric_columns=["NONEXISTENT_COLUMN"],
        )

        with mock.patch.object(
            self.mm._model_monitor_client,
            "validate_source",
            side_effect=ValueError("Custom metric column(s): ['NONEXISTENT_COLUMN'] do not exist in source."),
        ) as mock_validate_source, mock.patch.object(
            self.mv, "get_model_task", return_value=type_hints.Task.TABULAR_REGRESSION
        ):
            with self.assertRaisesRegex(
                ValueError, "Custom metric column\\(s\\): \\['NONEXISTENT_COLUMN'\\] do not exist in source\\."
            ):
                self.mm.add_monitor("TEST", source_config, self.test_monitor_config)

            mock_validate_source.assert_called_once_with(
                source_database=None,
                source_schema=None,
                source=self.test_source_table_name,
                timestamp_column="TS",
                prediction_score_columns=["PREDICTION"],
                prediction_class_columns=[],
                actual_score_columns=["LABEL"],
                actual_class_columns=[],
                id_columns=["ID"],
                segment_columns=[],
                custom_metric_columns=["NONEXISTENT_COLUMN"],
            )


if __name__ == "__main__":
    absltest.main()
