import os
import pathlib
import tempfile
from typing import cast
from unittest import mock

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities as pc
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
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
    "explain_table": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input1"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input2"),
        ],
        outputs=[
            model_signature.FeatureSpec(name="output1", dtype=model_signature.DataType.FLOAT),
            model_signature.FeatureSpec(name="output2", dtype=model_signature.DataType.FLOAT),
        ],
    ),
}


class ModelVersionImplTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)
        with (
            mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]),
            pc.PlatformCapabilities.mock_features({"ENABLE_INLINE_DEPLOYMENT_SPEC_FROM_CLIENT_VERSION": "1.8.6"}),
        ):
            self.m_mv = model_version_impl.ModelVersion._ref(
                model_ops.ModelOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                service_ops=service_ops.ServiceOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            )

    def test_ref(self) -> None:
        with (
            mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]) as mock_list_methods,
            pc.PlatformCapabilities.mock_features({"ENABLE_INLINE_DEPLOYMENT_SPEC_FROM_CLIENT_VERSION": "1.8.6"}),
        ):
            model_version_impl.ModelVersion._ref(
                model_ops.ModelOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                service_ops=service_ops.ServiceOperator(
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_get_metric_1(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load:
            self.assertEqual(1, self.m_mv.get_metric("a"))
            mock_load.assert_called_once_with(
                database_name=None,
                schema_name=None,
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
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=mock.ANY,
                )

    def test_set_metric_1(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with (
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load,
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save,
        ):
            self.m_mv.set_metric("a", 2)
            mock_load.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                metadata_ops.ModelVersionMetadataSchema(metrics={"a": 2}),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_set_metric_2(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with (
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load,
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save,
        ):
            self.m_mv.set_metric("b", 2)
            mock_load.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1, "b": 2}),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_delete_metric_1(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with (
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load,
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save,
        ):
            self.m_mv.delete_metric("a")
            mock_load.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_save.assert_called_once_with(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_delete_metric_2(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={"a": 1})
        with (
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load,
            mock.patch.object(self.m_mv._model_ops._metadata_ops, "save") as mock_save,
        ):
            with self.assertRaisesRegex(KeyError, "Cannot find metric with name b"):
                self.m_mv.delete_metric("b")
                mock_load.assert_called_once_with(
                    database_name=None,
                    schema_name=None,
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_get_model_task(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops,
            attribute="get_model_task",
            return_value=model_types.Task.TABULAR_REGRESSION,
        ) as mock_get_model_task:
            self.assertEqual(model_types.Task.TABULAR_REGRESSION, self.m_mv.get_model_task())
            mock_get_model_task.assert_called_once_with(
                database_name=None,
                schema_name=None,
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
                    "is_partitioned": False,
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": "__CALL__",
                    "target_method": "__call__",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                    "is_partitioned": False,
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
                is_partitioned=False,
            )

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name="__call__")
            mock_invoke_method.assert_called_once_with(
                method_name="__CALL__",
                method_function_type="FUNCTION",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
                is_partitioned=False,
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
                    "is_partitioned": False,
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
                is_partitioned=False,
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
                    "is_partitioned": False,
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                strict_input_validation=True,
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                partition_column=None,
                statement_params=mock.ANY,
                is_partitioned=False,
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
                    "is_partitioned": True,
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": "__CALL__",
                    "target_method": "__call__",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": _DUMMY_SIG["predict_table"],
                    "is_partitioned": True,
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
                is_partitioned=True,
            )

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name='"predict_table"', partition_column="PARTITION_COLUMN")
            mock_invoke_method.assert_called_once_with(
                method_name='"predict_table"',
                method_function_type="TABLE_FUNCTION",
                signature=_DUMMY_SIG["predict_table"],
                X=m_df,
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column="PARTITION_COLUMN",
                statement_params=mock.ANY,
                is_partitioned=True,
            )

    def test_run_table_function_method_no_partition(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict_table"',
                    "target_method": "predict_table",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": _DUMMY_SIG["predict_table"],
                    "is_partitioned": True,
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"explain_table"',
                    "target_method": "explain_table",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": _DUMMY_SIG["explain_table"],
                    "is_partitioned": False,
                }
            ),
        ]
        self.m_mv._functions = m_methods

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, function_name='"explain_table"')
            mock_invoke_method.assert_called_once_with(
                method_name='"explain_table"',
                method_function_type="TABLE_FUNCTION",
                signature=_DUMMY_SIG["explain_table"],
                X=m_df,
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                strict_input_validation=False,
                partition_column=None,
                statement_params=mock.ANY,
                is_partitioned=False,
            )

    def test_run_service(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                    "is_partitioned": False,
                }
            ),
        ]

        self.m_mv._functions = m_methods

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, service_name="SERVICE", function_name='"predict"')
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                database_name=None,
                schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                strict_input_validation=False,
                statement_params=mock.ANY,
            )

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self.m_mv.run(m_df, service_name="DB.SCHEMA.SERVICE", function_name='"predict"')
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                strict_input_validation=False,
                statement_params=mock.ANY,
            )

    def test_description_getter(self) -> None:
        with mock.patch.object(
            self.m_mv._model_ops, "get_comment", return_value="this is a comment"
        ) as mock_get_comment:
            self.assertEqual("this is a comment", self.m_mv.description)
            mock_get_comment.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_description_setter(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, "set_comment") as mock_set_comment:
            self.m_mv.description = "this is a comment"
            mock_set_comment.assert_called_once_with(
                comment="this is a comment",
                database_name=None,
                schema_name=None,
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_comment_setter(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, "set_comment") as mock_set_comment:
            self.m_mv.comment = "this is a comment"
            mock_set_comment.assert_called_once_with(
                comment="this is a comment",
                database_name=None,
                schema_name=None,
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
        with (
            mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            self.m_mv.export(tmpdir)
            mock_download_files.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                target_path=pathlib.Path(tmpdir),
                mode="model",
                statement_params=mock.ANY,
            )

    def test_export_full(self) -> None:
        with (
            mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            self.m_mv.export(tmpdir, export_mode=model_version_impl.ExportMode.FULL)
            mock_download_files.assert_called_once_with(
                database_name=None,
                schema_name=None,
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

        m_options = model_types.SKLModelLoadOptions()
        with (
            mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files,
            mock.patch.object(
                model_composer.ModelComposer, "load", side_effect=[m_pk_for_validation, m_pk]
            ) as mock_load,
            mock.patch.object(
                m_pk_for_validation.meta.env, "validate_with_local_env", return_value=[]
            ) as mock_validate_with_local_env,
        ):
            self.assertEqual(self.m_mv.load(options=m_options), m_model)
            mock_download_files.assert_has_calls(
                [
                    mock.call(
                        database_name=None,
                        schema_name=None,
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                        target_path=mock.ANY,
                        mode="minimal",
                        statement_params=mock.ANY,
                    ),
                    mock.call(
                        database_name=None,
                        schema_name=None,
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

        m_options = model_types.SKLModelLoadOptions()
        with (
            mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files,
            mock.patch.object(
                model_composer.ModelComposer, "load", side_effect=[m_pk_for_validation, m_pk]
            ) as mock_load,
            mock.patch.object(
                m_pk_for_validation.meta.env, "validate_with_local_env", return_value=["error"]
            ) as mock_validate_with_local_env,
        ):
            with self.assertRaisesRegex(ValueError, "Unable to load this model due to following validation errors"):
                self.assertEqual(self.m_mv.load(options=m_options), m_model)
            mock_download_files.assert_has_calls(
                [
                    mock.call(
                        database_name=None,
                        schema_name=None,
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

        m_options = model_types.SKLModelLoadOptions()
        with (
            mock.patch.object(self.m_mv._model_ops, "download_files") as mock_download_files,
            mock.patch.object(model_composer.ModelComposer, "load", side_effect=[m_pk]) as mock_load,
        ):
            self.assertEqual(self.m_mv.load(force=True, options=m_options), m_model)
            mock_download_files.assert_has_calls(
                [
                    mock.call(
                        database_name=None,
                        schema_name=None,
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

    def test_set_alias(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, "set_alias") as mock_set_alias:
            self.m_mv.set_alias("ally")
            mock_set_alias.assert_called_once_with(
                alias_name=sql_identifier.SqlIdentifier("ally"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_unset_alias(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, "unset_alias") as mock_unset_alias:
            self.m_mv.unset_alias("ally")
            mock_unset_alias.assert_called_once_with(
                version_or_alias_name=sql_identifier.SqlIdentifier("ally"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_create_service(self) -> None:
        with mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service:
            self.m_mv.create_service(
                service_name="SERVICE",
                image_build_compute_pool="IMAGE_BUILD_COMPUTE_POOL",
                service_compute_pool="SERVICE_COMPUTE_POOL",
                image_repo="IMAGE_REPO",
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["EAI"],
                block=True,
            )
            mock_create_service.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier(self.m_mv.model_name),
                version_name=sql_identifier.SqlIdentifier(self.m_mv.version_name),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=False,
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EAI")],
                block=True,
                statement_params=mock.ANY,
            )

    def test_create_service_same_pool(self) -> None:
        with mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service:
            self.m_mv.create_service(
                service_name="SERVICE",
                service_compute_pool="SERVICE_COMPUTE_POOL",
                image_repo="IMAGE_REPO",
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["EAI"],
                block=True,
            )
            mock_create_service.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier(self.m_mv.model_name),
                version_name=sql_identifier.SqlIdentifier(self.m_mv.version_name),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=False,
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EAI")],
                block=True,
                statement_params=mock.ANY,
            )

    def test_create_service_no_eai(self) -> None:
        with mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service:
            self.m_mv.create_service(
                service_name="SERVICE",
                image_build_compute_pool="IMAGE_BUILD_COMPUTE_POOL",
                service_compute_pool="SERVICE_COMPUTE_POOL",
                image_repo="IMAGE_REPO",
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                block=True,
            )
            mock_create_service.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier(self.m_mv.model_name),
                version_name=sql_identifier.SqlIdentifier(self.m_mv.version_name),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=False,
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=None,
                block=True,
                statement_params=mock.ANY,
            )

    def test_create_service_async_job(self) -> None:
        with mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service:
            self.m_mv.create_service(
                service_name="SERVICE",
                image_build_compute_pool="IMAGE_BUILD_COMPUTE_POOL",
                service_compute_pool="SERVICE_COMPUTE_POOL",
                image_repo="IMAGE_REPO",
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["EAI"],
                block=False,
            )
            mock_create_service.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier(self.m_mv.model_name),
                version_name=sql_identifier.SqlIdentifier(self.m_mv.version_name),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=False,
                max_instances=3,
                cpu_requests="CPU",
                memory_requests="MEMORY",
                gpu_requests="GPU",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EAI")],
                block=False,
                statement_params=mock.ANY,
            )

    def test_list_services(self) -> None:
        data = [
            {"name": "a.b.c", "inference_endpoint": "fooendpoint"},
            {"name": "d.e.f", "inference_endpoint": "bazendpoint"},
        ]
        m_df = pd.DataFrame(data)
        with mock.patch.object(
            self.m_mv._model_ops,
            attribute="show_services",
            return_value=data,
        ) as mock_get_functions:
            pd.testing.assert_frame_equal(m_df, self.m_mv.list_services())
            mock_get_functions.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_delete_service_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "service_name cannot be empty."):
            self.m_mv.delete_service("")

    def test_delete_service(self) -> None:
        with mock.patch.object(self.m_mv._model_ops, attribute="delete_service") as mock_delete_service:
            self.m_mv.delete_service("c")

            mock_delete_service.assert_called_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=mock.ANY,
            )

        with mock.patch.object(self.m_mv._model_ops, attribute="delete_service") as mock_delete_service:
            self.m_mv.delete_service("a.b.c")

            mock_delete_service.assert_called_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_database_name=sql_identifier.SqlIdentifier("a"),
                service_schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=mock.ANY,
            )

    def test_run_job(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                    "is_partitioned": False,
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": "__CALL__",
                    "target_method": "__call__",
                    "target_method_function_type": "FUNCTION",
                    "signature": _DUMMY_SIG["predict"],
                    "is_partitioned": False,
                }
            ),
        ]
        self.m_mv._functions = m_methods

        with self.assertRaisesRegex(ValueError, "There is no method with name PREDICT available in the model"):
            self.m_mv._run_job(
                X=m_df,
                job_name="TEST_JOB",
                compute_pool="TEST_COMPUTE_POOL",
                image_repo="TEST_IMAGE_REPO",
                output_table_name="TEST_OUTPUT_TABLE",
                function_name="PREDICT",
                cpu_requests="1",
                memory_requests="6Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["TEST_EAI"],
            )

        with self.assertRaisesRegex(ValueError, "There are more than 1 target methods available in the model"):
            self.m_mv._run_job(
                X=m_df,
                job_name="TEST_JOB",
                compute_pool="TEST_COMPUTE_POOL",
                image_repo="TEST_IMAGE_REPO",
                output_table_name="TEST_OUTPUT_TABLE",
                cpu_requests="1",
                memory_requests="6Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["TEST_EAI"],
            )

        with (
            mock.patch.object(self.m_mv._service_ops, "invoke_job_method", return_value=m_df) as mock_invoke_job_method,
            mock.patch.object(self.m_session, "get_current_warehouse", return_value="TEST_WAREHOUSE"),
        ):
            self.m_mv._run_job(
                X=m_df,
                job_name="TEST_JOB",
                compute_pool="TEST_COMPUTE_POOL",
                image_repo="TEST_IMAGE_REPO",
                output_table_name="TEST_OUTPUT_TABLE",
                function_name='"predict"',
                cpu_requests="1",
                memory_requests="6Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["TEST_EAI"],
            )
            mock_invoke_job_method.assert_called_once_with(
                target_method="predict",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                job_database_name=None,
                job_schema_name=None,
                job_name=sql_identifier.SqlIdentifier("TEST_JOB"),
                compute_pool_name=sql_identifier.SqlIdentifier("TEST_COMPUTE_POOL"),
                warehouse_name="TEST_WAREHOUSE",
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("TEST_IMAGE_REPO"),
                output_table_database_name=None,
                output_table_schema_name=None,
                output_table_name=sql_identifier.SqlIdentifier("TEST_OUTPUT_TABLE"),
                cpu_requests="1",
                memory_requests="6Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("TEST_EAI")],
                statement_params=mock.ANY,
            )

        with (
            mock.patch.object(self.m_mv._service_ops, "invoke_job_method", return_value=m_df) as mock_invoke_job_method,
            mock.patch.object(self.m_session, "get_current_warehouse", return_value="TEST_WAREHOUSE"),
        ):
            # fully qualified names
            self.m_mv._run_job(
                X=m_df,
                job_name="DB.SCHEMA.TEST_JOB",
                compute_pool="TEST_COMPUTE_POOL",
                image_repo="DB.SCHEMA.TEST_IMAGE_REPO",
                output_table_name="DB.SCHEMA.TEST_OUTPUT_TABLE",
                function_name='"predict"',
                cpu_requests="1",
                memory_requests="6Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=["TEST_EAI"],
            )
            mock_invoke_job_method.assert_called_once_with(
                target_method="predict",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                job_database_name=sql_identifier.SqlIdentifier("DB"),
                job_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                job_name=sql_identifier.SqlIdentifier("TEST_JOB"),
                compute_pool_name=sql_identifier.SqlIdentifier("TEST_COMPUTE_POOL"),
                warehouse_name=sql_identifier.SqlIdentifier("TEST_WAREHOUSE"),
                image_repo_database_name=sql_identifier.SqlIdentifier("DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("TEST_IMAGE_REPO"),
                output_table_database_name=sql_identifier.SqlIdentifier("DB"),
                output_table_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                output_table_name=sql_identifier.SqlIdentifier("TEST_OUTPUT_TABLE"),
                cpu_requests="1",
                memory_requests="6Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("TEST_EAI")],
                statement_params=mock.ANY,
            )

    def test_repr_html_happy_path_with_functions_and_metrics(self) -> None:
        """Test _repr_html_ method with functions and metrics present."""
        # Mock model signature with _repr_html_ method
        mock_signature = mock.MagicMock()
        mock_signature._repr_html_.return_value = "<div>Mock Signature HTML</div>"

        m_functions = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": mock_signature,
                    "is_partitioned": False,
                }
            ),
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict_table"',
                    "target_method": "predict_table",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": mock_signature,
                    "is_partitioned": True,
                }
            ),
        ]

        self.m_mv._functions = m_functions

        # Mock the various methods used in _repr_html_
        with (
            mock.patch.object(
                self.m_mv, "get_model_task", return_value=model_types.Task.TABULAR_REGRESSION
            ) as mock_get_task,
            mock.patch.object(self.m_mv, "show_functions", return_value=m_functions) as mock_show_functions,
            mock.patch.object(
                self.m_mv, "show_metrics", return_value={"accuracy": 0.95, "precision": 0.87, "recall": None}
            ) as mock_show_metrics,
            mock.patch.object(
                type(self.m_mv), "description", new_callable=mock.PropertyMock, return_value="Test model version"
            ),
        ):
            html = self.m_mv._repr_html_()

            # Verify basic structure and content
            self.assertIn("Model Version Details", html)
            self.assertIn("MODEL", html)  # model name
            self.assertIn('"v1"', html)  # version name
            self.assertIn('TEMP."test".MODEL', html)  # fully qualified name
            self.assertIn("Test model version", html)  # description
            self.assertIn("TABULAR_REGRESSION", html)  # task

            # Verify functions section
            self.assertIn("Functions", html)
            self.assertIn('"predict"', html)
            self.assertIn('"predict_table"', html)
            self.assertIn("FUNCTION", html)
            self.assertIn("TABLE_FUNCTION", html)
            self.assertIn("False", html)  # is_partitioned for predict
            self.assertIn("True", html)  # is_partitioned for predict_table
            self.assertIn("Mock Signature HTML", html)  # signature HTML

            # Verify metrics section
            self.assertIn("Metrics", html)
            self.assertIn("accuracy", html)
            self.assertIn("0.95", html)
            self.assertIn("precision", html)
            self.assertIn("0.87", html)
            self.assertIn("recall", html)
            self.assertIn("N/A", html)  # None value displayed as N/A

            # Verify method calls
            mock_get_task.assert_called_once()
            mock_show_functions.assert_called_once()
            mock_show_metrics.assert_called_once()

    def test_repr_html_happy_path_no_functions_no_metrics(self) -> None:
        """Test _repr_html_ method with no functions and no metrics."""
        with (
            mock.patch.object(self.m_mv, "get_model_task", return_value=model_types.Task.TABULAR_BINARY_CLASSIFICATION),
            mock.patch.object(self.m_mv, "show_functions", return_value=[]),
            mock.patch.object(self.m_mv, "show_metrics", return_value={}),
            mock.patch.object(type(self.m_mv), "description", new_callable=mock.PropertyMock, return_value=""),
        ):
            html = self.m_mv._repr_html_()

            # Verify basic structure
            self.assertIn("Model Version Details", html)
            self.assertIn("MODEL", html)
            self.assertIn('"v1"', html)
            self.assertIn("TABULAR_BINARY_CLASSIFICATION", html)

            # Verify empty states
            self.assertIn("No functions available", html)
            self.assertIn("No metrics available", html)

    def test_repr_html_happy_path_signature_fallback(self) -> None:
        """Test _repr_html_ method when signature._repr_html_() fails and falls back to string representation."""
        # Mock signature that raises exception in _repr_html_
        mock_signature = mock.MagicMock()
        mock_signature._repr_html_.side_effect = Exception("Signature HTML error")
        # Mock the string representation by setting configure_mock
        mock_signature.configure_mock(**{"__str__.return_value": "ModelSignature(inputs=[...], outputs=[...])"})

        m_functions = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": mock_signature,
                    "is_partitioned": False,
                }
            ),
        ]

        with (
            mock.patch.object(self.m_mv, "get_model_task", return_value=model_types.Task.TABULAR_REGRESSION),
            mock.patch.object(self.m_mv, "show_functions", return_value=m_functions),
            mock.patch.object(self.m_mv, "show_metrics", return_value={"f1_score": 0.89}),
            mock.patch.object(
                type(self.m_mv), "description", new_callable=mock.PropertyMock, return_value="Fallback test"
            ),
        ):
            html = self.m_mv._repr_html_()

            # Verify fallback to string representation in <pre> tag
            self.assertIn("<pre style='margin: 5px 0;'>", html)
            self.assertIn("ModelSignature(inputs=[...], outputs=[...])", html)
            self.assertIn("f1_score", html)
            self.assertIn("0.89", html)

    def test_repr_html_happy_path_mixed_metric_values(self) -> None:
        """Test _repr_html_ method with various metric value types."""
        with (
            mock.patch.object(self.m_mv, "get_model_task", return_value=model_types.Task.TABULAR_MULTI_CLASSIFICATION),
            mock.patch.object(self.m_mv, "show_functions", return_value=[]),
            mock.patch.object(
                self.m_mv,
                "show_metrics",
                return_value={
                    "string_metric": "excellent",
                    "int_metric": 42,
                    "float_metric": 3.14159,
                    "none_metric": None,
                    "bool_metric": True,
                    "list_metric": [1, 2, 3],
                },
            ),
            mock.patch.object(
                type(self.m_mv), "description", new_callable=mock.PropertyMock, return_value="Mixed metrics test"
            ),
        ):
            html = self.m_mv._repr_html_()

            # Verify all metric types are displayed correctly
            self.assertIn("string_metric", html)
            self.assertIn("excellent", html)
            self.assertIn("int_metric", html)
            self.assertIn("42", html)
            self.assertIn("float_metric", html)
            self.assertIn("3.14159", html)
            self.assertIn("none_metric", html)
            self.assertIn("N/A", html)  # None displays as N/A
            self.assertIn("bool_metric", html)
            self.assertIn("True", html)
            self.assertIn("list_metric", html)
            self.assertIn("[1, 2, 3]", html)

    def test_repr_html_happy_path_function_details(self) -> None:
        """Test _repr_html_ method with detailed function information."""
        mock_signature = mock.MagicMock()
        mock_signature._repr_html_.return_value = "<div class='signature'>Detailed Signature</div>"

        m_functions = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"custom_predict"',
                    "target_method": "custom_predict_method",
                    "target_method_function_type": "TABLE_FUNCTION",
                    "signature": mock_signature,
                    "is_partitioned": True,
                }
            ),
        ]

        with (
            mock.patch.object(self.m_mv, "get_model_task", return_value=model_types.Task.TABULAR_RANKING),
            mock.patch.object(self.m_mv, "show_functions", return_value=m_functions),
            mock.patch.object(self.m_mv, "show_metrics", return_value={"auc": 0.92}),
            mock.patch.object(
                type(self.m_mv), "description", new_callable=mock.PropertyMock, return_value="Custom function test"
            ),
        ):
            html = self.m_mv._repr_html_()

            # Verify function details are displayed correctly
            self.assertIn('"custom_predict"', html)
            self.assertIn("custom_predict_method", html)
            self.assertIn("TABLE_FUNCTION", html)
            self.assertIn("True", html)  # is_partitioned
            self.assertIn("Detailed Signature", html)

            # Verify details structure
            self.assertIn("Target Method:", html)
            self.assertIn("Function Type:", html)
            self.assertIn("Partitioned:", html)
            self.assertIn("Signature:", html)

            # Verify collapsible structure
            self.assertIn("<details", html)
            self.assertIn("<summary", html)


if __name__ == "__main__":
    absltest.main()
