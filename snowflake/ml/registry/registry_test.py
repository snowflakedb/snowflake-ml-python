from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry import registry
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Row, Session


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
                user_files=None,
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
                user_files=None,
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
        self.test_db_name = "TEST_DB"
        self.test_schema_name = "TEST_SCHEMA"
        self.test_monitor_name = "TEST"
        self.test_source = "MODEL_OUTPUTS"

        self.test_warehouse = "TEST_WAREHOUSE"
        self.test_timestamp_column = "TIMESTAMP"
        self.test_pred_score_column_name = "PREDICTION"
        self.test_label_score_column_name = "LABEL"
        self.test_id_column_name = "ID"
        self.m_model_version: model_version_impl.ModelVersion = mock.MagicMock()

        self.test_monitor_config = model_monitor_config.ModelMonitorConfig(
            model_version=self.m_model_version,
            model_function_name="predict",
            background_compute_warehouse_name=self.test_warehouse,
        )
        self.test_table_config = model_monitor_config.ModelMonitorSourceConfig(
            source=self.test_source,
            id_columns=[self.test_id_column_name],
            timestamp_column=self.test_timestamp_column,
            prediction_score_columns=[self.test_pred_score_column_name],
            actual_score_columns=[self.test_label_score_column_name],
        )

        session = cast(Session, self.m_session)
        self.m_r = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": True},
        )

    def test_registry_monitoring_disabled_properly(self) -> None:
        session = cast(Session, self.m_session)
        m_r = registry.Registry(
            session,
            database_name=self.test_db_name,
            schema_name=self.test_schema_name,
            options={"enable_monitoring": False},
        )

        with self.assertRaisesRegex(ValueError, registry._MODEL_MONITORING_DISABLED_ERROR):
            m_r.add_monitor(
                self.test_monitor_name,
                self.test_table_config,
                self.test_monitor_config,
            )

        with self.assertRaisesRegex(ValueError, registry._MODEL_MONITORING_DISABLED_ERROR):
            m_r.show_model_monitors()

        with self.assertRaisesRegex(ValueError, registry._MODEL_MONITORING_DISABLED_ERROR):
            m_r.delete_monitor(self.test_monitor_name)

        with self.assertRaisesRegex(ValueError, registry._MODEL_MONITORING_DISABLED_ERROR):
            m_r.get_monitor(name=self.test_monitor_name)

        with self.assertRaisesRegex(ValueError, registry._MODEL_MONITORING_DISABLED_ERROR):
            m_r.get_monitor(model_version=self.m_model_version)

    def test_add_monitor(self) -> None:
        m_monitor = mock.Mock()
        m_monitor.name = self.test_monitor_name

        with mock.patch.object(
            self.m_r._model_monitor_manager, "add_monitor", return_value=m_monitor
        ) as mock_add_monitor:
            self.m_r.add_monitor(
                self.test_monitor_name,
                self.test_table_config,
                self.test_monitor_config,
            )
            mock_add_monitor.assert_called_once_with(
                self.test_monitor_name,
                self.test_table_config,
                self.test_monitor_config,
            )
        self.m_session.finalize()

    def test_get_monitor(self) -> None:
        m_model_monitor: model_monitor.ModelMonitor = mock.MagicMock()
        m_model_monitor.name = sql_identifier.SqlIdentifier(self.test_monitor_name)
        with mock.patch.object(self.m_r._model_monitor_manager, "get_monitor", return_value=m_model_monitor):
            monitor = self.m_r.get_monitor(name=self.test_monitor_name)
            self.assertEqual(f"{monitor.name}", self.test_monitor_name)
        self.m_session.finalize()

    def test_get_monitor_by_model_version(self) -> None:
        m_model_monitor: model_monitor.ModelMonitor = mock.MagicMock()
        m_model_monitor.name = sql_identifier.SqlIdentifier(self.test_monitor_name)
        with mock.patch.object(
            self.m_r._model_monitor_manager, "get_monitor_by_model_version", return_value=m_model_monitor
        ):
            monitor = self.m_r.get_monitor(model_version=self.m_model_version)
            self.assertEqual(f"{monitor.name}", self.test_monitor_name)

        self.m_session.finalize()

    def test_show_model_monitors(self) -> None:
        sql_result = [Row(name="monitor")]
        with mock.patch.object(
            self.m_r._model_monitor_manager, "show_model_monitors", return_value=sql_result
        ) as mock_show_model_monitors:
            self.assertEqual(self.m_r.show_model_monitors(), sql_result)
            mock_show_model_monitors.assert_called_once_with()


if __name__ == "__main__":
    absltest.main()
