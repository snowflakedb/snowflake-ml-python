import os
import pathlib
import tempfile
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Session

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    ),
    "predict_table": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    ),
}


class ModelVersionImplTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)
        with mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]):
            self.m_mv = model_version_impl.ModelVersion._ref(
                model_ops.ModelOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            )

    def test_ref(self) -> None:
        with mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]) as mock_list_methods:
            model_version_impl.ModelVersion._ref(
                model_ops.ModelOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            )
            mock_list_methods.assert_called_once_with()

    def test_property(self) -> None:
        self.assertEqual(self.m_mv.model_name, "MODEL")
        self.assertEqual(self.m_mv.fully_qualified_model_name, 'TEMP."test".MODEL')
        self.assertEqual(self.m_mv.version_name, '"v1"')

    def test_show_metrics(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={})
        with mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load:
            self.assertDictEqual({}, self.m_mv.show_metrics())
            mock_load.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_get_metric_1(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load:
            self.assertEqual(1, self.m_mv.get_metric("a"))
            mock_load.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_get_metric_2(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load:
            with self.assertRaisesRegex(KeyError, "Cannot find metric with name b"):
                self.assertEqual(1, self.m_mv.get_metric("b"))
                mock_load.assert_called_once_with(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=mock.ANY,
                )

    def test_set_metric_1(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(
            self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata
        ) as mock_load, mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save:
            self.m_mv.set_metric("a", 2)
            mock_load.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                metadata_ops.ModelVersionMetadataSchema(metrics={"a": 2}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_set_metric_2(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(
            self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata
        ) as mock_load, mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save:
            self.m_mv.set_metric("b", 2)
            mock_load.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1, "b": 2}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_delete_metric_1(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(
            self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata
        ) as mock_load, mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save:
            self.m_mv.delete_metric("a")
            mock_load.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_delete_metric_2(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(
            self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata
        ) as mock_load, mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save:
            with self.assertRaisesRegex(KeyError, "Cannot find metric with name b"):
                self.m_mv.delete_metric("b")
                mock_load.assert_called_once_with(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=mock.ANY,
                )
                mock_save.assert_not_called()

    def test_show_functions(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, attribute="get_functions", return_value=[123]
        ) as mock_get_functions:
            self.assertListEqual([], self.m_mv.show_functions())
            mock_get_functions.assert_not_called()

    def test_get_functions(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, attribute="get_functions", return_value=[123]
        ) as mock_get_functions:
            self.assertListEqual([123], self.m_mv._get_functions())
            mock_get_functions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_run(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": "__CALL__",
                    "target_method": "__call__",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                }
            ),
        ]
        self.m_mv._functions = m_methods
        with self.assertRaisesRegex(ValueError, "There is no method with name PREDICT available in the model"):
            self.m_mv.run(m_df, function_name="PREDICT")

        with self.assertRaisesRegex(ValueError, "There are more than 1 target methods available in the model"):
            self.m_mv.run(m_df)

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name='"predict"')
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                method_function_type="FUNCTION",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
            )

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name="__call__")
            mock_invoke_method.assert_called_once_with(
                method_name="__CALL__",
                method_function_type="FUNCTION",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
            )

    def test_run_without_method_name(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                }
            ),
        ]

        self.m_mv._functions = m_methods

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df)
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                method_function_type="FUNCTION",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
            )

    def test_run_strict(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                }
            ),
        ]

        self.m_mv._functions = m_methods

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, strict_input_validation=True)
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                method_function_type="FUNCTION",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                strict_input_validation=True,
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                partition_column=None,
                statement_params=mock.ANY,
            )

    def test_run_table_function_method(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict_table"',
                    "target_method": "predict_table",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": _DUMMY_SIG["predict_table"],
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": "__CALL__",
                    "target_method": "__call__",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": _DUMMY_SIG["predict_table"],
                }
            ),
        ]
        self.m_mv._functions = m_methods

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name='"predict_table"')
            mock_invoke_method.assert_called_once_with(
                method_name='"predict_table"',
                method_function_type="TABLE_FUNCTION",
                signature=_DUMMY_SIG["predict_table"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
            )

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name='"predict_table"', partition_column="PARTITION_COLUMN")
            mock_invoke_method.assert_called_once_with(
                method_name='"predict_table"',
                method_function_type="TABLE_FUNCTION",
                signature=_DUMMY_SIG["predict_table"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column="PARTITION_COLUMN",
                statement_params=mock.ANY,
            )

    def test_description_getter(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, "get_comment", return_value="this is a comment"
        ) as mock_get_comment:
            self.assertEqual("this is a comment", self.m_mv.description)
            mock_get_comment.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_description_setter(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, "set_comment") as mock_set_comment:
            self.m_mv.description = "this is a comment"
            mock_set_comment.assert_called_once_with(
                comment="this is a comment",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_comment_getter(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, "get_comment", return_value="this is a comment"
        ) as mock_get_comment:
            self.assertEqual("this is a comment", self.m_mv.comment)
            mock_get_comment.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_comment_setter(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, "set_comment") as mock_set_comment:
            self.m_mv.comment = "this is a comment"
            mock_set_comment.assert_called_once_with(
                comment="this is a comment",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_export_invalid_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "dummy"), mode="w") as f:
                f.write("hello")
            with self.assertRaisesRegex(ValueError, "is a file or an non-empty folder"):
                self.m_mv.export(tmpdir)

    def test_export_model(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, "download_files"
        ) as mock_download_files, tempfile.TemporaryDirectory() as tmpdir:
            self.m_mv.export(tmpdir)
            mock_download_files.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                target_path=pathlib.Path(tmpdir),
                mode="model",
                statement_params=mock.ANY,
            )

    def test_export_full(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, "download_files"
        ) as mock_download_files, tempfile.TemporaryDirectory() as tmpdir:
            self.m_mv.export(tmpdir, export_mode=model_version_impl.ExportMode.FULL)
            mock_download_files.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                target_path=pathlib.Path(tmpdir),
                mode="full",
                statement_params=mock.ANY,
            )

    def test_load(self) -> None:
        m_pk_for_validation = mock.MagicMock()
        m_pk_for_validation.meta = mock.MagicMock()
        m_pk_for_validation.meta.model_type = "foo"
        m_pk_for_validation.meta.env = mock.MagicMock()

        m_model = mock.MagicMock()
        m_pk = mock.MagicMock()
        m_pk.meta = mock.MagicMock()
        m_pk.model = m_model

        m_options = model_types.ModelLoadOption(use_gpu=False)
        with mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files, mock.patch.object(
            model_composer.ModelComposer, "load", side_effect=[m_pk_for_validation, m_pk]
        ) as mock_load, mock.patch.object(
            m_pk_for_validation.meta.env, "validate_with_local_env", return_value=[]
        ) as mock_validate_with_local_env:
            self.assertEqual(self.m_mv.load(options=m_options), m_model)
            mock_download_files.assert_has_calls(
                [
                    mock.call(
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                        target_path=mock.ANY,
                        mode="minimal",
                        statement_params=mock.ANY,
                    ),
                    mock.call(
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                        target_path=mock.ANY,
                        mode="model",
                        statement_params=mock.ANY,
                    ),
                ]
            )
            mock_load.assert_has_calls(
                [
                    mock.call(mock.ANY, meta_only=True, options=m_options),
                    mock.call(mock.ANY, meta_only=False, options=m_options),
                ]
            )
            mock_validate_with_local_env.assert_called_once_with(check_snowpark_ml_version=False)

    def test_load_error(self) -> None:
        m_pk_for_validation = mock.MagicMock()
        m_pk_for_validation.meta = mock.MagicMock()
        m_pk_for_validation.meta.model_type = "snowml"
        m_pk_for_validation.meta.env = mock.MagicMock()

        m_model = mock.MagicMock()
        m_pk = mock.MagicMock()
        m_pk.meta = mock.MagicMock()
        m_pk.model = m_model

        m_options = model_types.ModelLoadOption(use_gpu=False)
        with mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files, mock.patch.object(
            model_composer.ModelComposer, "load", side_effect=[m_pk_for_validation, m_pk]
        ) as mock_load, mock.patch.object(
            m_pk_for_validation.meta.env, "validate_with_local_env", return_value=["error"]
        ) as mock_validate_with_local_env:
            with self.assertRaisesRegex(ValueError, "Unable to load this model due to following validation errors"):
                self.assertEqual(self.m_mv.load(options=m_options), m_model)
            mock_download_files.assert_has_calls(
                [
                    mock.call(
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                        target_path=mock.ANY,
                        mode="minimal",
                        statement_params=mock.ANY,
                    ),
                ]
            )
            mock_load.assert_has_calls(
                [
                    mock.call(mock.ANY, meta_only=True, options=m_options),
                ]
            )
            mock_validate_with_local_env.assert_called_once_with(check_snowpark_ml_version=True)

    def test_load_force(self) -> None:
        m_model = mock.MagicMock()
        m_pk = mock.MagicMock()
        m_pk.meta = mock.MagicMock()
        m_pk.model = m_model

        m_options = model_types.ModelLoadOption(use_gpu=False)
        with mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files, mock.patch.object(
            model_composer.ModelComposer, "load", side_effect=[m_pk]
        ) as mock_load:
            self.assertEqual(self.m_mv.load(force=True, options=m_options), m_model)
            mock_download_files.assert_has_calls(
                [
                    mock.call(
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                        target_path=mock.ANY,
                        mode="model",
                        statement_params=mock.ANY,
                    ),
                ]
            )
            mock_load.assert_has_calls(
                [
                    mock.call(mock.ANY, meta_only=False, options=m_options),
                ]
            )


if __name__ == "__main__":
    absltest.main()
