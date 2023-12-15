import os
import pathlib
import tempfile
import textwrap
from typing import List, cast
from unittest import mock

import numpy as np
import pandas as pd
import yaml
from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._client.ops import model_ops
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import DataFrame, Row, Session, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}


class ModelOpsTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.m_statement_params = {"test": "1"}
        self.c_session = cast(Session, self.m_session)
        self.m_ops = model_ops.ModelOperator(
            self.c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )

    def test_prepare_model_stage_path(self) -> None:
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",) as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ) as mock_random_name_for_temp_object:
            stage_path = self.m_ops.prepare_model_stage_path(
                statement_params=self.m_statement_params,
            )
            self.assertEqual(stage_path, '@TEMP."test".SNOWPARK_TEMP_STAGE_ABCDEF0123/model')
            mock_random_name_for_temp_object.assert_called_once_with(snowpark_utils.TempObjectType.STAGE)
            mock_create_stage.assert_called_once_with(
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )

    def test_list_models_or_versions_1(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="MODEL",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
            Row(
                create_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.list_models_or_versions(
                statement_params=self.m_statement_params,
            )
            self.assertListEqual(
                res,
                [
                    sql_identifier.SqlIdentifier("MODEL", case_sensitive=True),
                    sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                ],
            )
            mock_show_models.assert_called_once_with(
                statement_params=self.m_statement_params,
            )

    def test_list_models_or_versions_2(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
            Row(
                create_on="06/01",
                name="V1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=False,
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.list_models_or_versions(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            self.assertListEqual(
                res,
                [
                    sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    sql_identifier.SqlIdentifier("V1", case_sensitive=True),
                ],
            )
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_validate_existence_1(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.validate_existence(
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertTrue(res)
            mock_show_models.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_validate_existence_2(self) -> None:
        m_list_res: List[Row] = []
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.validate_existence(
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertFalse(res)
            mock_show_models.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_validate_existence_3(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.validate_existence(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertTrue(res)
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_validate_existence_4(self) -> None:
        m_list_res: List[Row] = []
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.validate_existence(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertFalse(res)
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_get_model_version_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
            m_manifest_path = os.path.join(tmpdir, "MANIFEST.yml")
            with open(m_manifest_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(m_manifest, f)
            with mock.patch.object(tempfile.TemporaryDirectory, "__enter__", return_value=tmpdir), mock.patch.object(
                self.m_ops._model_version_client, "get_file", return_value=m_manifest_path
            ) as mock_get_file:
                manifest_res = self.m_ops.get_model_version_manifest(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
                mock_get_file.assert_called_once_with(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    file_path=pathlib.PurePosixPath("MANIFEST.yml"),
                    target_path=mock.ANY,
                    statement_params=self.m_statement_params,
                )
                self.assertDictEqual(manifest_res, m_manifest)

    def test_get_model_version_native_packing_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            m_meta_yaml = textwrap.dedent(
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
                        - name: input_feature_0
                          type: DOUBLE
                        outputs:
                        - name: output_feature_0
                          type: BOOL
                version: '2023-12-01'
                """
            )
            m_meta_path = os.path.join(tmpdir, "model.yaml")
            with open(m_meta_path, "w", encoding="utf-8") as f:
                f.write(m_meta_yaml)
            with mock.patch.object(
                self.m_ops._model_version_client, "get_file", return_value=m_meta_path
            ) as mock_get_file:
                manifest_res = self.m_ops.get_model_version_native_packing_meta(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
                mock_get_file.assert_called_once_with(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    file_path=pathlib.PurePosixPath("model/model.yaml"),
                    target_path=mock.ANY,
                    statement_params=self.m_statement_params,
                )
                self.assertDictEqual(manifest_res, yaml.safe_load(m_meta_yaml))

    def test_create_from_stage_1(self) -> None:
        mock_composer = mock.MagicMock()
        mock_composer.stage_path = '@TEMP."test".MODEL/V1'

        with mock.patch.object(
            self.m_ops._model_version_client, "create_from_stage", return_value='TEMP."test".MODEL'
        ) as mock_create_from_stage, mock.patch.object(
            self.m_ops._model_version_client, "add_version_from_stage", return_value='TEMP."test".MODEL'
        ) as mock_add_version_from_stage, mock.patch.object(
            self.m_ops._model_client, "show_models", return_value=[]
        ):
            self.m_ops.create_from_stage(
                composed_model=mock_composer,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_create_from_stage.assert_called_once_with(
                stage_path='@TEMP."test".MODEL/V1',
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_add_version_from_stage.assert_not_called()

    def test_create_from_stage_2(self) -> None:
        mock_composer = mock.MagicMock()
        mock_composer.stage_path = '@TEMP."test".MODEL/V1'
        m_list_res = [
            Row(
                create_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_version_client, "create_from_stage", return_value='TEMP."test".MODEL'
        ) as mock_create_from_stage, mock.patch.object(
            self.m_ops._model_version_client, "add_version_from_stage", return_value='TEMP."test".MODEL'
        ) as mock_add_version_from_stage, mock.patch.object(
            self.m_ops._model_client, "show_models", return_value=m_list_res
        ), mock.patch.object(
            self.m_ops._model_client, attribute="show_versions", return_value=[]
        ):
            self.m_ops.create_from_stage(
                composed_model=mock_composer,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_create_from_stage.assert_not_called()
            mock_add_version_from_stage.assert_called_once_with(
                stage_path='@TEMP."test".MODEL/V1',
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_create_from_stage_3(self) -> None:
        mock_composer = mock.MagicMock()
        mock_composer.stage_path = '@TEMP."test".MODEL/V1'
        m_list_res_models = (
            Row(
                create_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        )
        m_list_res_versions = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_version_client, "create_from_stage", return_value='TEMP."test".MODEL'
        ) as mock_create_from_stage, mock.patch.object(
            self.m_ops._model_version_client, "add_version_from_stage", return_value='TEMP."test".MODEL'
        ) as mock_add_version_from_stagel, mock.patch.object(
            self.m_ops._model_client, "show_models", return_value=m_list_res_models
        ), mock.patch.object(
            self.m_ops._model_client, attribute="show_versions", return_value=m_list_res_versions
        ):
            with self.assertRaisesRegex(ValueError, 'Model TEMP."test".MODEL version V1 already existed.'):
                self.m_ops.create_from_stage(
                    composed_model=mock_composer,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_create_from_stage.assert_not_called()
            mock_add_version_from_stagel.assert_not_called()

    def test_invoke_method_1(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2"])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1", "COL2")
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
        ) as mock_convert_from_df, mock.patch.object(
            self.m_ops._model_version_client, "invoke_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                signature=m_sig,
                X=pd_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_called_once_with(self.c_session, mock.ANY, keep_order=True)
            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=['"input"'],
                returns=[("output", spt.FloatType(), '"output"')],
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_to_df.assert_called_once_with(m_df, features=m_sig.outputs)

    def test_invoke_method_1_no_drop(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", '"output"'])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1")
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
        ) as mock_convert_from_df, mock.patch.object(
            self.m_ops._model_version_client, "invoke_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                signature=m_sig,
                X=pd_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_called_once_with(self.c_session, mock.ANY, keep_order=True)
            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=['"input"'],
                returns=[("output", spt.FloatType(), '"output"')],
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_to_df.assert_called_once_with(m_df, features=m_sig.outputs)

    def test_invoke_method_2(self) -> None:
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df"
        ) as mock_convert_from_df, mock.patch.object(
            model_signature, "_validate_snowpark_data", return_value=model_signature.SnowparkIdentifierRule.NORMALIZED
        ) as mock_validate_snowpark_data, mock.patch.object(
            self.m_ops._model_version_client, "invoke_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df"
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                signature=m_sig,
                X=cast(DataFrame, m_df),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_not_called()
            mock_validate_snowpark_data.assert_called_once_with(m_df, m_sig.inputs)

            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=["INPUT"],
                returns=[("output", spt.FloatType(), "OUTPUT")],
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_to_df.assert_not_called()

    def test_get_comment_1(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.get_comment(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, "This is a comment")
            mock_show_models.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_get_comment_2(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="V1",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.get_comment(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, "This is a comment")
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_set_comment_1(self) -> None:
        with mock.patch.object(self.m_ops._model_client, "set_comment") as mock_set_comment:
            self.m_ops.set_comment(
                comment="This is a comment",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            mock_set_comment.assert_called_once_with(
                comment="This is a comment",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_set_comment_2(self) -> None:
        with mock.patch.object(self.m_ops._model_version_client, "set_comment") as mock_set_comment:
            self.m_ops.set_comment(
                comment="This is a comment",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_comment.assert_called_once_with(
                comment="This is a comment",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_delete_model_or_version(self) -> None:
        with mock.patch.object(
            self.m_ops._model_client,
            "drop_model",
        ) as mock_drop_model:
            self.m_ops.delete_model_or_version(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            mock_drop_model.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )


if __name__ == "__main__":
    absltest.main()
