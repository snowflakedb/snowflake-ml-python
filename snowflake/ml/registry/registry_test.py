from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._model_composer import model_composer
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

    def test_log_model_1(self) -> None:
        m_model = mock.MagicMock()
        m_conda_dependency = mock.MagicMock()
        m_sample_input_data = mock.MagicMock()
        m_stage_path = "@TEMP.TEST.MODEL/V1"
        with mock.patch.object(
            self.m_r._model_ops, "prepare_model_stage_path", return_value=m_stage_path
        ) as mock_prepare_model_stage_path, mock.patch.object(
            model_composer.ModelComposer, "save"
        ) as mock_save, mock.patch.object(
            self.m_r._model_ops, "create_from_stage"
        ) as mock_create_from_stage:
            mv = self.m_r.log_model(
                model=m_model,
                model_name="MODEL",
                version_name="v1",
                conda_dependencies=m_conda_dependency,
                sample_input_data=m_sample_input_data,
            )
            mock_prepare_model_stage_path.assert_called_once_with(
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                name="MODEL",
                model=m_model,
                signatures=None,
                sample_input=m_sample_input_data,
                conda_dependencies=m_conda_dependency,
                pip_requirements=None,
                python_version=None,
                code_paths=None,
                ext_modules=None,
                options=None,
            )
            mock_create_from_stage.assert_called_once_with(
                composed_model=mock.ANY,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1"),
                statement_params=mock.ANY,
            )
            self.assertDictEqual(
                mv.__dict__,
                model_version_impl.ModelVersion(
                    self.m_r._model_ops,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1"),
                ).__dict__,
            )

    def test_log_model_2(self) -> None:
        m_model = mock.MagicMock()
        m_pip_requirements = mock.MagicMock()
        m_signatures = mock.MagicMock()
        m_options = mock.MagicMock()
        m_stage_path = "@TEMP.TEST.MODEL/V1"
        with mock.patch.object(
            self.m_r._model_ops, "prepare_model_stage_path", return_value=m_stage_path
        ) as mock_prepare_model_stage_path, mock.patch.object(
            model_composer.ModelComposer, "save"
        ) as mock_save, mock.patch.object(
            self.m_r._model_ops, "create_from_stage"
        ) as mock_create_from_stage:
            mv = self.m_r.log_model(
                model=m_model,
                model_name="MODEL",
                version_name="V1",
                pip_requirements=m_pip_requirements,
                signatures=m_signatures,
                options=m_options,
            )
            mock_prepare_model_stage_path.assert_called_once_with(
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                name="MODEL",
                model=m_model,
                signatures=m_signatures,
                sample_input=None,
                conda_dependencies=None,
                pip_requirements=m_pip_requirements,
                python_version=None,
                code_paths=None,
                ext_modules=None,
                options=m_options,
            )
            mock_create_from_stage.assert_called_once_with(
                composed_model=mock.ANY,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=mock.ANY,
            )
            self.assertDictEqual(
                mv.__dict__,
                model_version_impl.ModelVersion(
                    self.m_r._model_ops,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                ).__dict__,
            )

    def test_log_model_3(self) -> None:
        m_model = mock.MagicMock()
        m_python_version = mock.MagicMock()
        m_code_paths = mock.MagicMock()
        m_ext_modules = mock.MagicMock()
        m_stage_path = "@TEMP.TEST.MODEL/V1"
        with mock.patch.object(
            self.m_r._model_ops, "prepare_model_stage_path", return_value=m_stage_path
        ) as mock_prepare_model_stage_path, mock.patch.object(
            model_composer.ModelComposer, "save"
        ) as mock_save, mock.patch.object(
            self.m_r._model_ops, "create_from_stage"
        ) as mock_create_from_stage:
            mv = self.m_r.log_model(
                model=m_model,
                model_name="MODEL",
                version_name="V1",
                python_version=m_python_version,
                code_paths=m_code_paths,
                ext_modules=m_ext_modules,
            )
            mock_prepare_model_stage_path.assert_called_once_with(
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                name="MODEL",
                model=m_model,
                signatures=None,
                sample_input=None,
                conda_dependencies=None,
                pip_requirements=None,
                python_version=m_python_version,
                code_paths=m_code_paths,
                ext_modules=m_ext_modules,
                options=None,
            )
            mock_create_from_stage.assert_called_once_with(
                composed_model=mock.ANY,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=mock.ANY,
            )
            self.assertDictEqual(
                mv.__dict__,
                model_version_impl.ModelVersion(
                    self.m_r._model_ops,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                ).__dict__,
            )


if __name__ == "__main__":
    absltest.main()
