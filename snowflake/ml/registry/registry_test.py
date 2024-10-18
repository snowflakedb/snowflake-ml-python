from typing import cast
from unittest import mock
from unittest.mock import patch

from absl.testing import absltest

from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry import registry
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session, types


class RegistryNameTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_location(self) -> None:
        c_session = cast(Session, self.m_session)
        r = registry.Registry(c_session, database_name="TEMP", schema_name="TEST")
        self.assertEqual(r.location, "TEMP.TEST")
        r = registry.Registry(c_session, database_name="TEMP", schema_name="test")
        self.assertEqual(r.location, "TEMP.TEST")
        r = registry.Registry(c_session, database_name="TEMP", schema_name='"test"')
        self.assertEqual(r.location, 'TEMP."test"')

        with mock.patch.object(c_session, "get_current_schema", return_value='"CURRENT_TEMP"', create=True):
            r = registry.Registry(c_session, database_name="TEMP")
            self.assertEqual(r.location, "TEMP.PUBLIC")
            r = registry.Registry(c_session, database_name="temp")
            self.assertEqual(r.location, "TEMP.PUBLIC")
            r = registry.Registry(c_session, database_name='"temp"')
            self.assertEqual(r.location, '"temp".PUBLIC')

        with mock.patch.object(c_session, "get_current_schema", return_value=None, create=True):
            r = registry.Registry(c_session, database_name="TEMP")
            self.assertEqual(r.location, "TEMP.PUBLIC")
            r = registry.Registry(c_session, database_name="temp")
            self.assertEqual(r.location, "TEMP.PUBLIC")
            r = registry.Registry(c_session, database_name='"temp"')
            self.assertEqual(r.location, '"temp".PUBLIC')

        with mock.patch.object(c_session, "get_current_database", return_value='"CURRENT_TEMP"', create=True):
            r = registry.Registry(c_session, schema_name="TEMP")
            self.assertEqual(r.location, "CURRENT_TEMP.TEMP")
            r = registry.Registry(c_session, schema_name="temp")
            self.assertEqual(r.location, "CURRENT_TEMP.TEMP")
            r = registry.Registry(c_session, schema_name='"temp"')
            self.assertEqual(r.location, 'CURRENT_TEMP."temp"')

        with mock.patch.object(c_session, "get_current_database", return_value='"current_temp"', create=True):
            r = registry.Registry(c_session, schema_name="TEMP")
            self.assertEqual(r.location, '"current_temp".TEMP')
            r = registry.Registry(c_session, schema_name="temp")
            self.assertEqual(r.location, '"current_temp".TEMP')
            r = registry.Registry(c_session, schema_name='"temp"')
            self.assertEqual(r.location, '"current_temp"."temp"')

        with mock.patch.object(c_session, "get_current_database", return_value=None, create=True):
            with self.assertRaisesRegex(ValueError, "You need to provide a database to use registry."):
                r = registry.Registry(c_session, schema_name="TEMP")

        with mock.patch.object(
            c_session, "get_current_database", return_value='"CURRENT_TEMP"', create=True
        ), mock.patch.object(c_session, "get_current_schema", return_value='"CURRENT_TEMP"', create=True):
            r = registry.Registry(c_session)
            self.assertEqual(r.location, "CURRENT_TEMP.CURRENT_TEMP")

        with mock.patch.object(
            c_session, "get_current_database", return_value='"CURRENT_TEMP"', create=True
        ), mock.patch.object(c_session, "get_current_schema", return_value='"current_temp"', create=True):
            r = registry.Registry(c_session)
            self.assertEqual(r.location, 'CURRENT_TEMP."current_temp"')

        with mock.patch.object(
            c_session, "get_current_database", return_value='"CURRENT_TEMP"', create=True
        ), mock.patch.object(c_session, "get_current_schema", return_value=None, create=True):
            r = registry.Registry(c_session)
            self.assertEqual(r.location, "CURRENT_TEMP.PUBLIC")


class RegistryTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)
        self.m_r = registry.Registry(self.c_session, database_name="TEMP", schema_name="TEST")

    def test_get_model(self) -> None:
        with mock.patch.object(self.m_r._model_manager, "get_model", return_value=True) as mock_get_model:
            self.m_r.get_model("MODEL")
            mock_get_model.assert_called_once_with(
                model_name="MODEL",
                statement_params=mock.ANY,
            )

    def test_models(self) -> None:
        with mock.patch.object(
            self.m_r._model_manager,
            "models",
        ) as mock_show_models:
            self.m_r.models()
            mock_show_models.assert_called_once_with(
                statement_params=mock.ANY,
            )

    def test_show_models(self) -> None:
        with mock.patch.object(
            self.m_r._model_manager,
            "show_models",
        ) as mock_show_models:
            self.m_r.show_models()
            mock_show_models.assert_called_once_with(
                statement_params=mock.ANY,
            )

    def test_log_model(self) -> None:
        m_model = mock.MagicMock()
        m_conda_dependency = mock.MagicMock()
        m_sample_input_data = mock.MagicMock()
        m_signatures = mock.MagicMock()
        m_options = mock.MagicMock()
        m_python_version = mock.MagicMock()
        m_code_paths = mock.MagicMock()
        m_ext_modules = mock.MagicMock()
        m_comment = mock.MagicMock()
        m_metrics = mock.MagicMock()
        with mock.patch.object(self.m_r._model_manager, "log_model") as mock_log_model:
            self.m_r.log_model(
                model=m_model,
                model_name="MODEL",
                version_name="v1",
                comment=m_comment,
                metrics=m_metrics,
                conda_dependencies=m_conda_dependency,
                pip_requirements=None,
                target_platforms=None,
                python_version=m_python_version,
                signatures=m_signatures,
                sample_input_data=m_sample_input_data,
                code_paths=m_code_paths,
                ext_modules=m_ext_modules,
                options=m_options,
            )
            mock_log_model.assert_called_once_with(
                model=m_model,
                model_name="MODEL",
                version_name="v1",
                comment=m_comment,
                metrics=m_metrics,
                conda_dependencies=m_conda_dependency,
                pip_requirements=None,
                target_platforms=None,
                python_version=m_python_version,
                signatures=m_signatures,
                sample_input_data=m_sample_input_data,
                code_paths=m_code_paths,
                ext_modules=m_ext_modules,
                options=m_options,
                statement_params=mock.ANY,
                task=type_hints.Task.UNKNOWN,
            )

    def test_log_model_from_model_version(self) -> None:
        m_model_version = mock.MagicMock()
        with mock.patch.object(self.m_r._model_manager, "log_model") as mock_log_model:
            self.m_r.log_model(
                model=m_model_version,
                model_name="MODEL",
                version_name="v1",
            )
            mock_log_model.assert_called_once_with(
                model=m_model_version,
                model_name="MODEL",
                version_name="v1",
                comment=None,
                metrics=None,
                conda_dependencies=None,
                pip_requirements=None,
                target_platforms=None,
                python_version=None,
                signatures=None,
                sample_input_data=None,
                code_paths=None,
                ext_modules=None,
                options=None,
                statement_params=mock.ANY,
                task=type_hints.Task.UNKNOWN,
            )

    def test_delete_model(self) -> None:
        with mock.patch.object(self.m_r._model_manager, "delete_model") as mock_delete_model:
            self.m_r.delete_model(
                model_name="MODEL",
            )
            mock_delete_model.assert_called_once_with(
                model_name="MODEL",
                statement_params=mock.ANY,
            )


class MonitorRegistryTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_monitor_name = "TEST"
        self.test_source_table_name = "MODEL_OUTPUTS"
        self.test_db_name = "SNOWML_OBSERVABILITY"
        self.test_schema_name = "METADATA"
        self.test_model_name = "test_model"
        self.test_model_name_sql = "TEST_MODEL"
        self.test_model_version_name = "test_model_version"
        self.test_model_version_name_sql = "TEST_MODEL_VERSION"
        self.test_fq_model_name = f"{self.test_db_name}.{self.test_schema_name}.{self.test_model_name}"
        self.test_warehouse = "TEST_WAREHOUSE"
        self.test_timestamp_column = "TIMESTAMP"
        self.test_prediction_column_name = "PREDICTION"
        self.test_label_column_name = "LABEL"
        self.test_id_column_name = "ID"
        self.test_baseline_table_name_sql = "_SNOWML_OBS_BASELINE_TEST_MODEL_TEST_MODEL_VERSION"

        model_version = mock.MagicMock()
        model_version.version_name = self.test_model_version_name
        model_version.model_name = self.test_model_name
        model_version.fully_qualified_model_name = self.test_fq_model_name
        model_version.show_functions.return_value = [
            model_manifest_schema.ModelFunctionInfo(
                name="PREDICT",
                target_method="predict",
                target_method_function_type="FUNCTION",
                signature=model_signature.ModelSignature(inputs=[], outputs=[]),
                is_partitioned=False,
            )
        ]
        model_version.get_model_task.return_value = type_hints.Task.TABULAR_REGRESSION
        self.m_model_version: model_version_impl.ModelVersion = model_version
        self.test_monitor_config = model_monitor_config.ModelMonitorConfig(
            model_version=self.m_model_version,
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
        )
        self.test_table_config = model_monitor_config.ModelMonitorTableConfig(
            prediction_columns=[self.test_prediction_column_name],
            label_columns=[self.test_label_column_name],
            id_columns=[self.test_id_column_name],
            timestamp_column=self.test_timestamp_column,
            source_table=self.test_source_table_name,
        )

        mock_struct_fields = []
        for col in ["NUM_0"]:
            mock_struct_fields.append(types.StructField(col, types.FloatType(), True))
        for col in ["CAT_0"]:
            mock_struct_fields.append(types.StructField(col, types.StringType(), True))
        self.mock_schema = types.StructType._from_attributes(mock_struct_fields)

        mock_struct_fields = []
        for col in ["NUM_0"]:
            mock_struct_fields.append(types.StructField(col, types.FloatType(), True))
        for col in ["CAT_0"]:
            mock_struct_fields.append(types.StructField(col, types.StringType(), True))
        self.mock_schema = types.StructType._from_attributes(mock_struct_fields)

    def _add_expected_monitoring_init_calls(self, model_monitor_create_if_not_exists: bool = False) -> None:
        self.m_session.add_mock_sql(
            query="""CREATE TABLE IF NOT EXISTS SNOWML_OBSERVABILITY.METADATA._SYSTEM_MONITORING_METADATA
            (MONITOR_NAME VARCHAR, SOURCE_TABLE_NAME VARCHAR, FULLY_QUALIFIED_MODEL_NAME VARCHAR,
            MODEL_VERSION_NAME VARCHAR, FUNCTION_NAME VARCHAR, TASK VARCHAR, IS_ENABLED BOOLEAN,
            TIMESTAMP_COLUMN_NAME VARCHAR, PREDICTION_COLUMN_NAMES ARRAY,
            LABEL_COLUMN_NAMES ARRAY, ID_COLUMN_NAMES ARRAY)
            """,
            result=mock_data_frame.MockDataFrame([Row(status="Table successfully created.")]),
        )

        if not model_monitor_create_if_not_exists:  # this code path does validation on whether tables exist.
            self.m_session.add_mock_sql(
                query="""SHOW TABLES LIKE '_SYSTEM_MONITORING_METADATA' IN SNOWML_OBSERVABILITY.METADATA""",
                result=mock_data_frame.MockDataFrame([Row(name="_SYSTEM_MONITORING_METADATA")]),
            )

    def test_init(self) -> None:
        self._add_expected_monitoring_init_calls(model_monitor_create_if_not_exists=True)
        session = cast(Session, self.m_session)
        r1 = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": True},
        )
        self.assertEqual(r1.enable_monitoring, True)

        r2 = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
        )
        self.assertEqual(r2.enable_monitoring, False)
        self.m_session.finalize()

    def test_add_monitor(self) -> None:
        self._add_expected_monitoring_init_calls(model_monitor_create_if_not_exists=True)

        session = cast(Session, self.m_session)
        m_r = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": True},
        )
        m_monitor = mock.Mock()
        m_monitor.name = self.test_monitor_name

        with mock.patch.object(m_r._model_monitor_manager, "add_monitor", return_value=m_monitor) as mock_add_monitor:
            monitor: model_monitor.ModelMonitor = m_r.add_monitor(
                self.test_monitor_name,
                self.test_table_config,
                self.test_monitor_config,
            )
            mock_add_monitor.assert_called_once_with(
                self.test_monitor_name, self.test_table_config, self.test_monitor_config, add_dashboard_udtfs=False
            )
        self.assertEqual(monitor.name, self.test_monitor_name)
        self.m_session.finalize()

    def test_get_monitor(self) -> None:
        self._add_expected_monitoring_init_calls(model_monitor_create_if_not_exists=True)

        session = cast(Session, self.m_session)
        m_r = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": True},
        )
        m_model_monitor: model_monitor.ModelMonitor = mock.MagicMock()
        with mock.patch.object(
            m_r._model_monitor_manager, "get_monitor", return_value=m_model_monitor
        ) as mock_get_monitor:
            m_r.get_monitor(name=self.test_monitor_name)
            mock_get_monitor.assert_called_once_with(name=self.test_monitor_name)
        self.m_session.finalize()

    def test_get_monitor_by_model_version(self) -> None:
        self._add_expected_monitoring_init_calls(model_monitor_create_if_not_exists=True)
        session = cast(Session, self.m_session)
        m_r = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": True},
        )
        m_model_monitor: model_monitor.ModelMonitor = mock.MagicMock()
        with mock.patch.object(
            m_r._model_monitor_manager, "get_monitor_by_model_version", return_value=m_model_monitor
        ) as mock_get_monitor:
            m_r.get_monitor(model_version=self.m_model_version)
            mock_get_monitor.assert_called_once_with(model_version=self.m_model_version)
        self.m_session.finalize()

    @patch("snowflake.ml.monitoring._manager.model_monitor_manager.ModelMonitorManager", autospec=True)
    def test_show_model_monitors(self, m_model_monitor_manager_class: mock.MagicMock) -> None:
        # Dont need to call self._add_expected_monitoring_init_calls since ModelMonitorManager.__init__ is
        # auto mocked.
        m_model_monitor_manager = m_model_monitor_manager_class.return_value
        sql_result = [
            Row(
                col1="val1",
                col2="val2",
            )
        ]
        m_model_monitor_manager.show_model_monitors.return_value = sql_result
        session = cast(Session, self.m_session)
        m_r = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": True},
        )
        self.assertEqual(m_r.show_model_monitors(), sql_result)


if __name__ == "__main__":
    absltest.main()
