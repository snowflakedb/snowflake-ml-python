import os
import pathlib
import tempfile
from typing import Any, cast
from unittest import mock

import pandas as pd
from absl.testing import absltest

from snowflake.ml import jobs
from snowflake.ml._internal import platform_capabilities as pc
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import (
    inference_engine,
    model_signature,
    openai_signatures,
    task,
    type_hints,
)
from snowflake.ml.model._client.model import (
    batch_inference_specs,
    inference_engine_utils,
    model_version_impl,
)
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.test_utils.mock_progress import create_mock_progress_status
from snowflake.snowpark import Session, row

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
            return_value=task.Task.TABULAR_REGRESSION,
        ) as mock_get_model_task:
            self.assertEqual(task.Task.TABULAR_REGRESSION, self.m_mv.get_model_task())
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
            self._add_show_versions_mock()
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
                explain_case_sensitive=False,
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
                explain_case_sensitive=False,
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
            self._add_show_versions_mock()
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
                explain_case_sensitive=False,
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
            self._add_show_versions_mock()
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
                explain_case_sensitive=False,
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
            self._add_show_versions_mock()
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
                explain_case_sensitive=False,
            )

        with mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method:
            self._add_show_versions_mock()
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
                explain_case_sensitive=False,
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

        with (
            mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=m_df) as mock_invoke_method,
            mock.patch.object(
                self.m_mv._model_ops,
                "_fetch_model_spec",
                return_value={
                    "model_type": "huggingface_pipeline",
                    "models": {
                        "model1": {
                            "model_type": "huggingface_pipeline",
                            "options": {"task": "text-generation"},
                        }
                    },
                },
            ),
        ):
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
                explain_case_sensitive=False,
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

        m_options = type_hints.SKLModelLoadOptions()
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

        m_options = type_hints.SKLModelLoadOptions()
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

        m_options = type_hints.SKLModelLoadOptions()
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
        mock_progress_status = create_mock_progress_status()
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service,
            mock.patch("snowflake.ml.model.event_handler.ModelEventHandler") as mock_event_handler_cls,
            mock.patch.object(self.m_mv, "_can_run_on_gpu", return_value=True),
        ):
            mock_event_handler = mock_event_handler_cls.return_value
            mock_event_handler.status.return_value.__enter__.return_value = mock_progress_status

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
                image_repo_name="IMAGE_REPO",
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
                progress_status=mock_progress_status,
                inference_engine_args=None,
                autocapture=None,
            )

    def test_create_service_same_pool(self) -> None:
        mock_progress_status = create_mock_progress_status()
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service,
            mock.patch("snowflake.ml.model.event_handler.ModelEventHandler") as mock_event_handler_cls,
            mock.patch.object(self.m_mv, "_can_run_on_gpu", return_value=True),
        ):
            mock_event_handler = mock_event_handler_cls.return_value
            mock_event_handler.status.return_value.__enter__.return_value = mock_progress_status

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
                image_repo_name="IMAGE_REPO",
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
                progress_status=mock_progress_status,
                inference_engine_args=None,
                autocapture=None,
            )

    def test_create_service_no_eai(self) -> None:
        mock_progress_status = create_mock_progress_status()
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service,
            mock.patch("snowflake.ml.model.event_handler.ModelEventHandler") as mock_event_handler_cls,
            mock.patch.object(self.m_mv, "_can_run_on_gpu", return_value=True),
        ):
            mock_event_handler = mock_event_handler_cls.return_value
            mock_event_handler.status.return_value.__enter__.return_value = mock_progress_status

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
                image_repo_name="IMAGE_REPO",
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
                progress_status=mock_progress_status,
                inference_engine_args=None,
                autocapture=None,
            )

    def test_create_service_async_job(self) -> None:
        mock_progress_status = create_mock_progress_status()
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service,
            mock.patch("snowflake.ml.model.event_handler.ModelEventHandler") as mock_event_handler_cls,
            mock.patch.object(self.m_mv, "_can_run_on_gpu", return_value=True),
        ):
            mock_event_handler = mock_event_handler_cls.return_value
            mock_event_handler.status.return_value.__enter__.return_value = mock_progress_status

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
                image_repo_name="IMAGE_REPO",
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
                progress_status=mock_progress_status,
                inference_engine_args=None,
                autocapture=None,
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

    def test_create_service_with_inference_engine_options(self) -> None:
        """Test create_service with inference engine options."""
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service,
            mock.patch.object(
                self.m_mv, "_check_huggingface_text_generation_model"
            ) as mock_check_huggingface_text_generation_model,
            mock.patch.object(
                self.m_mv._model_ops,
                "_fetch_model_spec",
                return_value={
                    "model_type": "huggingface_pipeline",
                    "models": {
                        "model1": {
                            "model_type": "huggingface_pipeline",
                            "options": {"task": "text-generation"},
                        }
                    },
                    "runtimes": {
                        "gpu": {
                            "cuda_version": "11.8",
                        }
                    },
                },
            ),
        ):
            # Test with inference engine and GPU
            self.m_mv.create_service(
                service_name="SERVICE",
                service_compute_pool="SERVICE_COMPUTE_POOL",
                image_repo="IMAGE_REPO",
                gpu_requests="4",
                inference_engine_options={
                    "engine": inference_engine.InferenceEngine.VLLM,
                    "engine_args_override": ["--max_tokens=1000", "--temperature=0.8"],
                },
            )

            # This check should happen when inference_engine_options is provided
            mock_check_huggingface_text_generation_model.assert_called_once()

            # Verify that the enriched kwargs were passed to create_service
            mock_create_service.assert_called_once()
            call_args = mock_create_service.call_args

            # Check that inference_engine is passed correctly
            self.assertEqual(
                call_args.kwargs["inference_engine_args"].inference_engine, inference_engine.InferenceEngine.VLLM
            )

            # Check that inference_engine_args is enriched correctly
            expected_args = [
                "--max_tokens=1000",
                "--temperature=0.8",
                "--tensor-parallel-size=4",
            ]

            self.assertEqual(
                call_args.kwargs["inference_engine_args"].inference_engine_args_override,
                expected_args,
            )

    def test_create_service_without_inference_engine_options(self) -> None:
        """Test create_service without inference engine options to ensure existing behavior is preserved."""
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service") as mock_create_service,
            mock.patch.object(self.m_mv, "_can_run_on_gpu", return_value=True),
        ):
            # Test without inference_engine_options
            self.m_mv.create_service(
                service_name="SERVICE",
                service_compute_pool="SERVICE_COMPUTE_POOL",
                image_repo="IMAGE_REPO",
                gpu_requests="2",
            )

            # Verify that None is passed for inference engine parameters
            mock_create_service.assert_called_once()
            call_args = mock_create_service.call_args

            self.assertIsNone(call_args.kwargs["inference_engine_args"])

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
            mock.patch.object(self.m_mv, "get_model_task", return_value=task.Task.TABULAR_REGRESSION) as mock_get_task,
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
            mock.patch.object(self.m_mv, "get_model_task", return_value=task.Task.TABULAR_BINARY_CLASSIFICATION),
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
            mock.patch.object(self.m_mv, "get_model_task", return_value=task.Task.TABULAR_REGRESSION),
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
            mock.patch.object(self.m_mv, "get_model_task", return_value=task.Task.TABULAR_MULTI_CLASSIFICATION),
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
            mock.patch.object(self.m_mv, "get_model_task", return_value=task.Task.TABULAR_RANKING),
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

    def test_get_inference_engine_args(self) -> None:
        # Test with None
        inference_engine_args = inference_engine_utils._get_inference_engine_args(None)
        self.assertIsNone(inference_engine_args)

        # Test with empty inference_engine_options
        inference_engine_args = inference_engine_utils._get_inference_engine_args({})
        self.assertIsNone(inference_engine_args)

        # Test with inference_engine_options missing inference_engine key
        with self.assertRaises(ValueError) as cm:
            inference_engine_utils._get_inference_engine_args({"other_key": "value"})
        self.assertEqual(str(cm.exception), "'engine' field is required in inference_engine_options")

        # Test with only engine (no args_override)
        inference_engine_options: dict[str, Any] = {"engine": inference_engine.InferenceEngine.VLLM}
        inference_engine_args = inference_engine_utils._get_inference_engine_args(inference_engine_options)
        assert inference_engine_args is not None
        self.assertEqual(inference_engine_args.inference_engine, inference_engine.InferenceEngine.VLLM)
        self.assertIsNone(inference_engine_args.inference_engine_args_override)

        # Test with inference_engine and args_override
        inference_engine_options = {
            "engine": inference_engine.InferenceEngine.VLLM,
            "engine_args_override": ["--max_tokens=100", "--temperature=0.7"],
        }
        inference_engine_args = inference_engine_utils._get_inference_engine_args(inference_engine_options)
        assert inference_engine_args is not None
        self.assertEqual(inference_engine_args.inference_engine, inference_engine.InferenceEngine.VLLM)
        self.assertEqual(
            inference_engine_args.inference_engine_args_override, ["--max_tokens=100", "--temperature=0.7"]
        )

        # Test with inference_engine and empty args_override
        inference_engine_options = {
            "engine": inference_engine.InferenceEngine.VLLM,
            "engine_args_override": [],
        }
        inference_engine_args = inference_engine_utils._get_inference_engine_args(inference_engine_options)
        assert inference_engine_args is not None
        self.assertEqual(inference_engine_args.inference_engine, inference_engine.InferenceEngine.VLLM)
        self.assertEqual(inference_engine_args.inference_engine_args_override, [])

    def test_enrich_inference_engine_args(self) -> None:
        """Test _enrich_inference_engine_args method with various inputs."""
        # Test with args=None and no GPU
        enriched = inference_engine_utils._enrich_inference_engine_args(
            service_ops.InferenceEngineArgs(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args_override=None,
            ),
            gpu_requests=None,
        )
        assert enriched is not None
        self.assertEqual(enriched.inference_engine, inference_engine.InferenceEngine.VLLM)
        self.assertEqual(enriched.inference_engine_args_override, [])

        # Test with empty args and GPU count
        enriched = inference_engine_utils._enrich_inference_engine_args(
            service_ops.InferenceEngineArgs(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args_override=None,
            ),
            gpu_requests=2,
        )
        assert enriched is not None
        self.assertEqual(enriched.inference_engine, inference_engine.InferenceEngine.VLLM)
        self.assertEqual(enriched.inference_engine_args_override, ["--tensor-parallel-size=2"])

        # Test with args and string GPU count
        original_args = ["--max_tokens=100", "--temperature=0.7"]
        enriched = inference_engine_utils._enrich_inference_engine_args(
            service_ops.InferenceEngineArgs(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args_override=original_args,
            ),
            gpu_requests="4",
        )
        assert enriched is not None
        self.assertEqual(
            enriched,
            service_ops.InferenceEngineArgs(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args_override=[
                    "--max_tokens=100",
                    "--temperature=0.7",
                    "--tensor-parallel-size=4",
                ],
            ),
        )

        # Test overwriting existing model key with new model key by appending to the list
        enriched = inference_engine_utils._enrich_inference_engine_args(
            service_ops.InferenceEngineArgs(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args_override=[
                    "--max_tokens=100",
                    "--temperature=0.7",
                ],
            ),
        )
        self.assertEqual(
            enriched,
            service_ops.InferenceEngineArgs(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args_override=[
                    "--max_tokens=100",
                    "--temperature=0.7",
                ],
            ),
        )

        # Test with invalid GPU string
        with self.assertRaises(ValueError):
            inference_engine_utils._enrich_inference_engine_args(
                service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.VLLM,
                    inference_engine_args_override=None,
                ),
                gpu_requests="invalid",
            )

        # Test with zero GPU (should not set tensor-parallel-size)
        with self.assertRaises(ValueError):
            inference_engine_utils._enrich_inference_engine_args(
                service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.VLLM,
                    inference_engine_args_override=None,
                ),
                gpu_requests=0,
            )

        # Test with negative GPU (should not set tensor-parallel-size)
        with self.assertRaises(ValueError):
            inference_engine_utils._enrich_inference_engine_args(
                service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.VLLM,
                    inference_engine_args_override=None,
                ),
                gpu_requests=-1,
            )

    def test_can_run_on_gpu(self) -> None:
        """Test _can_run_on_gpu method."""
        # Test successful case - model spec with gpu runtime
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "huggingface_pipeline",
                "runtimes": {
                    "gpu": {
                        "version": "1.0",
                    }
                },
            },
        ) as mock_fetch:
            result = self.m_mv._can_run_on_gpu()
            self.assertTrue(result)
            mock_fetch.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=self.m_mv._model_name,
                version_name=self.m_mv._version_name,
                statement_params=None,
            )

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test case - model spec without runtimes section
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "sklearn",
            },
        ):
            result = self.m_mv._can_run_on_gpu()
            self.assertFalse(result)

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test case - model spec with runtimes but no gpu
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "tensorflow",
                "runtimes": {
                    "cpu": {
                        "version": "1.0",
                    }
                },
            },
        ):
            result = self.m_mv._can_run_on_gpu()
            self.assertFalse(result)

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test case - model spec with empty runtimes
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "pytorch",
                "runtimes": {},
            },
        ):
            result = self.m_mv._can_run_on_gpu()
            self.assertFalse(result)

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test with statement_params
        test_statement_params = {"test_key": "test_value"}
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "huggingface_pipeline",
                "runtimes": {
                    "gpu": {
                        "version": "1.0",
                    }
                },
            },
        ) as mock_fetch:
            result = self.m_mv._can_run_on_gpu(statement_params=test_statement_params)
            self.assertTrue(result)
            mock_fetch.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=self.m_mv._model_name,
                version_name=self.m_mv._version_name,
                statement_params=test_statement_params,
            )

    def test_check_huggingface_text_generation_model(self) -> None:
        """Test _check_huggingface_text_generation_model method."""
        # Test successful case - HuggingFace text-generation model
        # Serialize the OPENAI_CHAT_SIGNATURE to match what the model spec returns
        serialized_signatures = {
            func_name: sig_obj.to_dict() for func_name, sig_obj in openai_signatures.OPENAI_CHAT_SIGNATURE.items()
        }

        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "huggingface_pipeline",
                "signatures": serialized_signatures,
                "models": {
                    "model1": {
                        "model_type": "huggingface_pipeline",
                        "options": {"task": "text-generation"},
                    }
                },
            },
        ) as mock_fetch:
            # Should not raise any exception
            self.m_mv._check_huggingface_text_generation_model()
            mock_fetch.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=self.m_mv._model_name,
                version_name=self.m_mv._version_name,
                statement_params=None,
            )

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test failure case - not a HuggingFace model
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "sklearn",
                "models": {"model1": {"model_type": "sklearn"}},
            },
        ):
            with self.assertRaises(ValueError) as cm:
                self.m_mv._check_huggingface_text_generation_model()
            self.assertIn(
                "Inference engine is only supported for HuggingFace text-generation models", str(cm.exception)
            )
            self.assertIn("Found model_type: sklearn", str(cm.exception))

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test failure case - HuggingFace model but wrong task
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "huggingface_pipeline",
                "models": {
                    "model1": {
                        "model_type": "huggingface_pipeline",
                        "options": {"task": "image-classification"},
                    }
                },
            },
        ):
            with self.assertRaises(ValueError) as cm:
                self.m_mv._check_huggingface_text_generation_model()
            self.assertIn("Inference engine is only supported for task 'text-generation'", str(cm.exception))
            self.assertIn("Found task(s): image-classification", str(cm.exception))

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test failure case - HuggingFace model with no task
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "huggingface_pipeline",
                "models": {
                    "model1": {
                        "model_type": "huggingface_pipeline",
                        "options": {},
                    }
                },
            },
        ):
            with self.assertRaises(ValueError) as cm:
                self.m_mv._check_huggingface_text_generation_model()
            self.assertIn("Inference engine is only supported for task 'text-generation'", str(cm.exception))
            self.assertIn("No task found in model spec.", str(cm.exception))

        # Reset the cached model spec
        self.m_mv._model_spec = None

        # Test failure case - HuggingFace text-generation model but wrong signature
        wrong_signature = {
            "__call__": {
                "inputs": [{"name": "input", "type": "STRING", "nullable": True}],
                "outputs": [{"name": "output", "type": "STRING", "nullable": True}],
            }
        }
        with mock.patch.object(
            self.m_mv._model_ops,
            "_fetch_model_spec",
            return_value={
                "model_type": "huggingface_pipeline",
                "signatures": wrong_signature,
                "models": {
                    "model1": {
                        "model_type": "huggingface_pipeline",
                        "options": {"task": "text-generation"},
                    }
                },
            },
        ):
            with self.assertRaises(ValueError) as cm:
                self.m_mv._check_huggingface_text_generation_model()
            self.assertIn(
                "Inference engine requires the model to be logged with OPENAI_CHAT_SIGNATURE", str(cm.exception)
            )

    def test_run_batch_all_parameters(self) -> None:
        """Test _run_batch with all possible parameters to ensure they're passed correctly."""
        input_df = mock.MagicMock()
        input_df.write.copy_into_location = mock.MagicMock()

        output_spec = batch_inference_specs.OutputSpec(
            stage_location="@output_stage",
        )
        job_spec = batch_inference_specs.JobSpec(
            function_name="predict",
            job_name="CUSTOM_JOB_NAME",
            warehouse="CUSTOM_WAREHOUSE",
            force_rebuild=True,
            image_repo="custom_repo",
            num_workers=4,
            max_batch_rows=2000,
            cpu_requests="4",
            memory_requests="8Gi",
            replicas=10,
        )

        mock_job = mock.MagicMock(spec=jobs.MLJob)

        with (
            mock.patch.object(self.m_mv, "_get_function_info", return_value={"target_method": "predict"}),
            mock.patch.object(self.m_mv._service_ops, "_enforce_save_mode"),
            mock.patch.object(
                self.m_mv._service_ops, "invoke_batch_job_method", return_value=mock_job
            ) as mock_invoke_batch_job,
        ):
            result = self.m_mv.run_batch(
                compute_pool="CUSTOM_POOL", input_spec=input_df, output_spec=output_spec, job_spec=job_spec
            )

            # Verify all parameters are passed correctly
            mock_invoke_batch_job.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                function_name="predict",
                compute_pool_name=sql_identifier.SqlIdentifier("CUSTOM_POOL"),
                force_rebuild=True,
                image_repo_name="custom_repo",
                num_workers=4,
                max_batch_rows=2000,
                warehouse=sql_identifier.SqlIdentifier("CUSTOM_WAREHOUSE"),
                cpu_requests="4",
                memory_requests="8Gi",
                gpu_requests=None,
                job_name="CUSTOM_JOB_NAME",
                replicas=10,
                input_stage_location="@output_stage/_temporary/",
                input_file_pattern="*",
                output_stage_location="@output_stage/",
                completion_filename="_SUCCESS",
                statement_params=mock.ANY,
            )

            self.assertEqual(result, mock_job)

    def test_run_batch_with_generated_job_name(self) -> None:
        """Test _run_batch with job_name generated when None."""
        input_df = mock.MagicMock()
        input_df.write.copy_into_location = mock.MagicMock()

        output_spec = batch_inference_specs.OutputSpec(stage_location="@output_stage")
        job_spec = batch_inference_specs.JobSpec(
            function_name="predict",
            job_name=None,  # This will trigger job name generation
            warehouse="TEST_WAREHOUSE",
            force_rebuild=False,
            image_repo="test_repo",
            num_workers=1,
            max_batch_rows=500,
            cpu_requests="1",
            memory_requests="2Gi",
        )

        mock_job = mock.MagicMock(spec=jobs.MLJob)

        with (
            mock.patch.object(self.m_mv, "_get_function_info", return_value={"target_method": "predict"}),
            mock.patch.object(self.m_mv._service_ops, "_enforce_save_mode"),
            mock.patch.object(
                self.m_mv._service_ops, "invoke_batch_job_method", return_value=mock_job
            ) as mock_invoke_batch_job,
            mock.patch("uuid.uuid4") as mock_uuid,
        ):
            # Setup the UUID mock to return a predictable value
            mock_uuid.return_value.configure_mock(__str__=lambda self: "12345678-1234-5678-9abc-123456789012")

            result = self.m_mv.run_batch(
                compute_pool="TEST_POOL", input_spec=input_df, output_spec=output_spec, job_spec=job_spec
            )

            # Verify result
            self.assertEqual(result, mock_job)

            # Verify the generated job name uses the mocked UUID
            call_args = mock_invoke_batch_job.call_args
            job_name = call_args.kwargs["job_name"]
            self.assertEqual(job_name, "BATCH_INFERENCE_12345678_1234_5678_9ABC_123456789012")

    def test_run_batch_with_warehouse_from_session(self) -> None:
        """Test _run_batch with warehouse from session when job_spec.warehouse is None."""
        input_df = mock.MagicMock()
        input_df.write.copy_into_location = mock.MagicMock()

        output_spec = batch_inference_specs.OutputSpec(stage_location="@output_stage")
        job_spec = batch_inference_specs.JobSpec(
            function_name="predict",
            job_name="TEST_JOB",
            warehouse=None,  # Will use session warehouse
            force_rebuild=False,
            image_repo="test_repo",
            num_workers=1,
            max_batch_rows=500,
            cpu_requests="1",
            memory_requests="2Gi",
        )

        mock_job = mock.MagicMock(spec=jobs.MLJob)

        with (
            mock.patch.object(self.m_mv, "_get_function_info", return_value={"target_method": "predict"}),
            mock.patch.object(self.m_mv._service_ops, "_enforce_save_mode"),
            mock.patch.object(
                self.m_mv._service_ops, "invoke_batch_job_method", return_value=mock_job
            ) as mock_invoke_batch_job,
            mock.patch.object(
                self.m_mv._service_ops._session, "get_current_warehouse", return_value="SESSION_WAREHOUSE"
            ) as mock_get_warehouse,
        ):
            result = self.m_mv.run_batch(
                compute_pool="TEST_POOL", input_spec=input_df, output_spec=output_spec, job_spec=job_spec
            )

            # Verify result
            self.assertEqual(result, mock_job)

            # Verify session warehouse was called
            mock_get_warehouse.assert_called_once()

            # Verify the session warehouse is used
            call_args = mock_invoke_batch_job.call_args
            self.assertEqual(call_args.kwargs["warehouse"], sql_identifier.SqlIdentifier("SESSION_WAREHOUSE"))

    def test_run_batch_no_warehouse_error(self) -> None:
        """Test _run_batch raises ValueError when no warehouse is available."""
        input_df = mock.MagicMock()
        input_df.write.copy_into_location = mock.MagicMock()

        output_spec = batch_inference_specs.OutputSpec(stage_location="@output_stage")
        job_spec = batch_inference_specs.JobSpec(
            function_name="predict",
            job_name="TEST_JOB",
            warehouse=None,  # No warehouse in job_spec
            force_rebuild=False,
            image_repo="test_repo",
            num_workers=1,
            max_batch_rows=500,
            cpu_requests="1",
            memory_requests="2Gi",
        )

        with (
            mock.patch.object(self.m_mv._service_ops, "_enforce_save_mode"),
            mock.patch.object(
                self.m_mv._service_ops._session, "get_current_warehouse", return_value=None
            ) as mock_get_warehouse,
        ):
            with self.assertRaisesRegex(
                ValueError, "Warehouse is not set. Please set the warehouse field in the JobSpec."
            ):
                self.m_mv.run_batch(
                    compute_pool="TEST_POOL", input_spec=input_df, output_spec=output_spec, job_spec=job_spec
                )

            # Verify session warehouse was called
            mock_get_warehouse.assert_called_once()

    def test_run_batch_with_none_job_spec(self) -> None:
        """Test _run_batch with job_spec=None uses default JobSpec values."""
        input_df = mock.MagicMock()
        input_df.write.copy_into_location = mock.MagicMock()

        output_spec = batch_inference_specs.OutputSpec(stage_location="@output_stage")

        mock_job = mock.MagicMock(spec=jobs.MLJob)

        with (
            mock.patch.object(self.m_mv, "_get_function_info", return_value={"target_method": "predict"}),
            mock.patch.object(self.m_mv._service_ops, "_enforce_save_mode"),
            mock.patch.object(
                self.m_mv._service_ops, "invoke_batch_job_method", return_value=mock_job
            ) as mock_invoke_batch_job,
            mock.patch.object(
                self.m_mv._service_ops._session, "get_current_warehouse", return_value="SESSION_WAREHOUSE"
            ) as mock_get_warehouse,
            mock.patch("uuid.uuid4") as mock_uuid,
        ):
            # Setup the UUID mock to return a predictable value
            mock_uuid.return_value.configure_mock(__str__=lambda self: "default-uuid-1234-5678-abcd")

            result = self.m_mv.run_batch(
                compute_pool="TEST_POOL", input_spec=input_df, output_spec=output_spec, job_spec=None
            )

            # Verify result
            self.assertEqual(result, mock_job)

            # Verify session warehouse was called since job_spec.warehouse defaults to None
            mock_get_warehouse.assert_called_once()

            # Verify the method was called with default JobSpec values
            mock_invoke_batch_job.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                function_name="predict",  # from _get_function_info with default function_name=None
                compute_pool_name=sql_identifier.SqlIdentifier("TEST_POOL"),
                force_rebuild=False,  # JobSpec default
                image_repo_name=None,  # JobSpec default
                num_workers=None,  # JobSpec default
                max_batch_rows=1024,  # JobSpec default
                warehouse=sql_identifier.SqlIdentifier("SESSION_WAREHOUSE"),  # from session since warehouse=None
                cpu_requests=None,  # JobSpec default
                memory_requests=None,  # JobSpec default
                gpu_requests=None,
                job_name="BATCH_INFERENCE_DEFAULT_UUID_1234_5678_ABCD",  # generated since job_name=None
                replicas=None,  # JobSpec default
                input_stage_location="@output_stage/_temporary/",
                input_file_pattern="*",  # InputSpec default
                output_stage_location="@output_stage/",
                completion_filename="_SUCCESS",  # OutputSpec default
                statement_params=mock.ANY,
            )

    def _add_show_versions_mock(self) -> None:
        current_dir = os.path.dirname(__file__)
        data_file_path = os.path.join(current_dir, "sample_model_spec.yaml")
        with open(data_file_path) as f:
            model_spec = f.read()
        model_attributes = """{
            "framework":"sklearn",
            "task":"TABULAR_BINARY_CLASSIFICATION",
            "client":"snowflake-ml-python 1.7.5"}"""
        sql_result = [
            row.Row(
                name='"v1"',
                comment=None,
                metadata={},
                model_spec=model_spec,
                model_attributes=model_attributes,
            ),
        ]
        self.m_session.add_mock_sql(
            "SHOW VERSIONS LIKE 'v1' IN MODEL TEMP.\"test\".MODEL", result=mock_data_frame.MockDataFrame(sql_result)
        )


if __name__ == "__main__":
    absltest.main()
