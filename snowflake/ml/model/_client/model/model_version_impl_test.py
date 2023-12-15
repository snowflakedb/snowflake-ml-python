import textwrap
from typing import cast
from unittest import mock

import yaml
from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Session

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}


class ModelVersionImplTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)
        self.m_mv = model_version_impl.ModelVersion._ref(
            model_ops.ModelOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
        )

    def test_property(self) -> None:
        self.assertEqual(self.m_mv.model_name, "MODEL")
        self.assertEqual(self.m_mv.fully_qualified_model_name, 'TEMP."test".MODEL')
        self.assertEqual(self.m_mv.version_name, '"v1"')

    def test_list_metrics(self) -> None:
        m_metadata = metadata_ops.ModelVersionMetadataSchema(metrics={})
        with mock.patch.object(self.m_mv._model_ops._metadata_ops, "load", return_value=m_metadata) as mock_load:
            self.assertDictEqual({}, self.m_mv.list_metrics())
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

    def test_list_methods(self) -> None:
        m_manifest = {
            "manifest_version": "1.0",
            "runtimes": {
                "python_runtime": {
                    "language": "PYTHON",
                    "version": "3.8",
                    "imports": ["model.zip", "runtimes/python_runtime/snowflake-ml-python.zip"],
                    "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                }
            },
            "methods": [
                {
                    "name": "predict",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "input", "type": "FLOAT"}],
                    "outputs": [{"type": "OBJECT"}],
                },
                {
                    "name": "__CALL__",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.__call__.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}],
                    "outputs": [{"type": "OBJECT"}],
                },
            ],
        }
        m_meta_yaml = yaml.safe_load(
            textwrap.dedent(
                """
                creation_timestamp: '2023-11-20 18:14:06.357187'
                env:
                    conda: env/conda.yml
                    cuda_version: null
                    pip: env/requirements.txt
                    python_version: '3.8'
                    snowpark_ml_version: 1.0.13+ca79e1b0720d35abd021c33707de789dc63918cc
                metadata: null
                min_snowpark_ml_version: 1.0.12
                model_type: sklearn
                models:
                    SKLEARN_MODEL:
                        artifacts: {}
                        handler_version: '2023-12-01'
                        model_type: sklearn
                        name: SKLEARN_MODEL
                        options: {}
                        path: model.pkl
                name: SKLEARN_MODEL
                signatures:
                    predict:
                        inputs:
                        - name: input
                          type: FLOAT
                        outputs:
                        - name: output
                          type: FLOAT
                    __call__:
                        inputs:
                        - name: input
                          type: FLOAT
                        outputs:
                        - name: output
                          type: FLOAT
                version: '2023-12-01'
                """
            )
        )
        with mock.patch.object(
            self.m_mv._model_ops, "get_model_version_manifest", return_value=m_manifest
        ) as mock_get_model_version_manifest, mock.patch.object(
            self.m_mv._model_ops, "get_model_version_native_packing_meta", return_value=m_meta_yaml
        ) as mock_get_model_version_native_packing_meta:
            methods = self.m_mv.list_methods()
            mock_get_model_version_manifest.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_get_model_version_native_packing_meta.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            self.assertEqual(
                methods,
                [
                    {
                        "name": '"predict"',
                        "target_method": "predict",
                        "signature": _DUMMY_SIG["predict"],
                    },
                    {
                        "name": "__CALL__",
                        "target_method": "__call__",
                        "signature": _DUMMY_SIG["predict"],
                    },
                ],
            )

    def test_run(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            {
                "name": '"predict"',
                "target_method": "predict",
                "signature": _DUMMY_SIG["predict"],
            },
            {
                "name": "__CALL__",
                "target_method": "__call__",
                "signature": _DUMMY_SIG["predict"],
            },
        ]
        with mock.patch.object(self.m_mv, "list_methods", return_value=m_methods) as mock_list_methods:
            with self.assertRaisesRegex(ValueError, "There is no method with name PREDICT available in the model"):
                self.m_mv.run(m_df, method_name="PREDICT")
            mock_list_methods.assert_called_once_with()

        with mock.patch.object(self.m_mv, "list_methods", return_value=m_methods) as mock_list_methods:
            with self.assertRaisesRegex(ValueError, "There are more than 1 target methods available in the model"):
                self.m_mv.run(m_df)
            mock_list_methods.assert_called_once_with()

        with mock.patch.object(
            self.m_mv, "list_methods", return_value=m_methods
        ) as mock_list_methods, mock.patch.object(
            self.m_mv._model_ops, "invoke_method", return_value=m_df
        ) as mock_invoke_method:
            self.m_mv.run(m_df, method_name='"predict"')
            mock_list_methods.assert_called_once_with()
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

        with mock.patch.object(
            self.m_mv, "list_methods", return_value=m_methods
        ) as mock_list_methods, mock.patch.object(
            self.m_mv._model_ops, "invoke_method", return_value=m_df
        ) as mock_invoke_method:
            self.m_mv.run(m_df, method_name="__call__")
            mock_list_methods.assert_called_once_with()
            mock_invoke_method.assert_called_once_with(
                method_name="__CALL__",
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_run_without_method_name(self) -> None:
        m_df = mock_data_frame.MockDataFrame()
        m_methods = [
            {
                "name": '"predict"',
                "target_method": "predict",
                "signature": _DUMMY_SIG["predict"],
            },
        ]

        with mock.patch.object(
            self.m_mv, "list_methods", return_value=m_methods
        ) as mock_list_methods, mock.patch.object(
            self.m_mv._model_ops, "invoke_method", return_value=m_df
        ) as mock_invoke_method:
            self.m_mv.run(m_df)
            mock_list_methods.assert_called_once_with()
            mock_invoke_method.assert_called_once_with(
                method_name='"predict"',
                signature=_DUMMY_SIG["predict"],
                X=m_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
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


if __name__ == "__main__":
    absltest.main()
