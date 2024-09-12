from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml.registry import registry
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Session


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
                python_version=m_python_version,
                signatures=m_signatures,
                sample_input_data=m_sample_input_data,
                code_paths=m_code_paths,
                ext_modules=m_ext_modules,
                options=m_options,
                statement_params=mock.ANY,
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
                python_version=None,
                signatures=None,
                sample_input_data=None,
                code_paths=None,
                ext_modules=None,
                options=None,
                statement_params=mock.ANY,
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


if __name__ == "__main__":
    absltest.main()
