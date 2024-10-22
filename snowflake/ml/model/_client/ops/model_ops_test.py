import pathlib
from typing import List, cast
from unittest import mock

import numpy as np
import pandas as pd
import yaml
from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.ops import model_ops
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema
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
    ),
    "predict_table": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    ),
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

    def _add_id_check_mock_operations(
        self, m_df: mock_data_frame.MockDataFrame, collect_result: List[Row]
    ) -> mock_data_frame.MockDataFrame:
        m_df.add_operation(operation="select", args=("_ID",))
        m_df.add_operation(operation="limit", args=(1,))
        m_df.add_collect_result(collect_result)
        return m_df

    def test_prepare_model_stage_path(self) -> None:
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage") as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ) as mock_random_name_for_temp_object:
            stage_path = self.m_ops.prepare_model_stage_path(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(stage_path, '@TEMP."test".SNOWPARK_TEMP_STAGE_ABCDEF0123/model')
            mock_random_name_for_temp_object.assert_called_once_with(snowpark_utils.TempObjectType.STAGE)
            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )

    def test_show_models_or_versions_1(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="MODEL",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="V1",
            ),
            Row(
                create_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="v1",
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.show_models_or_versions(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertListEqual(
                res,
                m_list_res,
            )
            mock_show_models.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                validate_result=False,
                statement_params=self.m_statement_params,
            )

    def test_show_models_or_versions_2(self) -> None:
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
            res = self.m_ops.show_models_or_versions(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            self.assertListEqual(
                res,
                m_list_res,
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                validate_result=False,
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
                default_version_name="V1",
            ),
            Row(
                create_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="v1",
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.list_models_or_versions(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                validate_result=False,
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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                validate_result=False,
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
                default_version_name="V1",
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.validate_existence(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertTrue(res)
            mock_show_models.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                validate_result=False,
                statement_params=self.m_statement_params,
            )

    def test_validate_existence_2(self) -> None:
        m_list_res: List[Row] = []
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.validate_existence(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertFalse(res)
            mock_show_models.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
                validate_result=False,
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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertTrue(res)
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                validate_result=False,
                statement_params=self.m_statement_params,
            )

    def test_validate_existence_4(self) -> None:
        m_list_res: List[Row] = []
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.validate_existence(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertFalse(res)
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                validate_result=False,
                statement_params=self.m_statement_params,
            )

    def test_get_tag_value_1(self) -> None:
        m_list_res: Row = Row(TAG_VALUE="a")
        with mock.patch.object(self.m_ops._tag_client, "get_tag_value", return_value=m_list_res) as mock_get_tag_value:
            res = self.m_ops.get_tag_value(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, "a")
            mock_get_tag_value.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )

    def test_get_tag_value_2(self) -> None:
        m_list_res: Row = Row(TAG_VALUE=1)
        with mock.patch.object(self.m_ops._tag_client, "get_tag_value", return_value=m_list_res) as mock_get_tag_value:
            res = self.m_ops.get_tag_value(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, "1")
            mock_get_tag_value.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )

    def test_get_tag_value_3(self) -> None:
        m_list_res: Row = Row(TAG_VALUE=None)
        with mock.patch.object(self.m_ops._tag_client, "get_tag_value", return_value=m_list_res) as mock_get_tag_value:
            res = self.m_ops.get_tag_value(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )
            self.assertIsNone(res)
            mock_get_tag_value.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )

    def test_show_tags(self) -> None:
        m_list_res: List[Row] = [
            Row(TAG_DATABASE="DB", TAG_SCHEMA="schema", TAG_NAME="MYTAG", TAG_VALUE="tag content"),
            Row(TAG_DATABASE="MYDB", TAG_SCHEMA="SCHEMA", TAG_NAME="my_another_tag", TAG_VALUE=1),
        ]
        with mock.patch.object(self.m_ops._tag_client, "get_tag_list", return_value=m_list_res) as mock_get_tag_list:
            res = self.m_ops.show_tags(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(res, {'DB."schema".MYTAG': "tag content", 'MYDB.SCHEMA."my_another_tag"': "1"})
            mock_get_tag_list.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_set_tag(self) -> None:
        with mock.patch.object(self.m_ops._tag_client, "set_tag_on_model") as mock_set_tag:
            self.m_ops.set_tag(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                tag_value="tag content",
                statement_params=self.m_statement_params,
            )
            mock_set_tag.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                tag_value="tag content",
                statement_params=self.m_statement_params,
            )

    def test_unset_tag(self) -> None:
        with mock.patch.object(self.m_ops._tag_client, "unset_tag_on_model") as mock_unset_tag:
            self.m_ops.unset_tag(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )
            mock_unset_tag.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=self.m_statement_params,
            )

    def test_list_inference_services(self) -> None:
        m_services_list_res = [Row(inference_services='["a.b.c", "d.e.f"]')]
        m_endpoints_list_res_0 = [Row(name="fooendpoint"), Row(name="barendpoint")]
        m_endpoints_list_res_1 = [Row(name="bazendpoint")]

        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_services_list_res
        ) as mock_show_versions, mock.patch.object(
            self.m_ops._model_client, "show_endpoints", side_effect=[m_endpoints_list_res_0, m_endpoints_list_res_1]
        ):
            res = self.m_ops.list_inference_services(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(
                res,
                {
                    "service_name": ["a.b.c", "a.b.c", "d.e.f"],
                    "endpoints": ["fooendpoint", "barendpoint", "bazendpoint"],
                },
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_list_inference_services_pre_bcr(self) -> None:
        m_list_res = [Row(comment="mycomment")]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            with self.assertRaises(exceptions.SnowflakeMLException) as context:
                self.m_ops.list_inference_services(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
            self.assertEqual(
                str(context.exception),
                "RuntimeError('(2104) Please opt in to BCR Bundle 2024_08 "
                "(https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_08_bundle).')",
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_list_inference_services_skip_build(self) -> None:
        m_list_res = [Row(inference_services='["A.B.MODEL_BUILD_34d35ew", "A.B.SERVICE"]')]
        m_endpoints_list_res = [Row(name="fooendpoint"), Row(name="barendpoint")]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions, mock.patch.object(
            self.m_ops._model_client, "show_endpoints", side_effect=[m_endpoints_list_res]
        ):
            res = self.m_ops.list_inference_services(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(
                res,
                {
                    "service_name": ["A.B.SERVICE", "A.B.SERVICE"],
                    "endpoints": ["fooendpoint", "barendpoint"],
                },
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )

    def test_delete_service_non_existent(self) -> None:
        m_list_res = [Row(inference_services='["A.B.C", "D.E.F"]')]
        m_endpoints_list_res = [Row(name="fooendpoint"), Row(name="barendpoint")]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions, mock.patch.object(
            self.m_session, attribute="get_current_database", return_value="a"
        ) as mock_get_database, mock.patch.object(
            self.m_session, attribute="get_current_schema", return_value="b"
        ) as mock_get_schema, mock_show_versions, mock.patch.object(
            self.m_ops._model_client, "show_endpoints", return_value=m_endpoints_list_res
        ):
            with self.assertRaisesRegex(
                ValueError, "Service 'A' does not exist or unauthorized or not associated with this model version."
            ):
                self.m_ops.delete_service(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    service_name="a",
                )
            with self.assertRaisesRegex(
                ValueError, "Service 'B' does not exist or unauthorized or not associated with this model version."
            ):
                self.m_ops.delete_service(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    service_name="a.b",
                )
            with self.assertRaisesRegex(
                ValueError, "Service 'D' does not exist or unauthorized or not associated with this model version."
            ):
                self.m_ops.delete_service(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    service_name="b.c.d",
                )

            mock_show_versions.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_get_database.assert_called()
            mock_get_schema.assert_called()

    def test_delete_service_exists(self) -> None:
        m_list_res = [Row(inference_services='["A.B.C", "D.E.F"]')]
        m_endpoints_list_res = [Row(name="fooendpoint"), Row(name="barendpoint")]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions, mock.patch.object(
            self.m_ops._service_client, "drop_service"
        ) as mock_drop_service, mock.patch.object(
            self.m_session, attribute="get_current_database", return_value="a"
        ) as mock_get_database, mock.patch.object(
            self.m_session, attribute="get_current_schema", return_value="b"
        ) as mock_get_schema, mock_show_versions, mock.patch.object(
            self.m_ops._model_client, "show_endpoints", return_value=m_endpoints_list_res
        ):
            self.m_ops.delete_service(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_name="c",
            )
            self.m_ops.delete_service(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_name="b.c",
            )
            self.m_ops.delete_service(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_name="a.b.c",
            )

            mock_show_versions.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_get_database.assert_called()
            mock_get_schema.assert_called()
            mock_drop_service.assert_called_with(
                database_name="A",
                schema_name="B",
                service_name="C",
                statement_params=mock.ANY,
            )

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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_create_from_stage.assert_called_once_with(
                stage_path='@TEMP."test".MODEL/V1',
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
                default_version_name="V1",
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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_create_from_stage.assert_not_called()
            mock_add_version_from_stage.assert_called_once_with(
                stage_path='@TEMP."test".MODEL/V1',
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
                default_version_name="V1",
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
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_create_from_stage.assert_not_called()
            mock_add_version_from_stagel.assert_not_called()

    def test_create_from_model_version_create(self) -> None:
        with mock.patch.object(
            self.m_ops._model_version_client, "create_from_model_version"
        ) as mock_create_from_model_version, mock.patch.object(self.m_ops, "validate_existence", return_value=False):
            self.m_ops.create_from_model_version(
                source_database_name=sql_identifier.SqlIdentifier("TEMP"),
                source_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
                source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_create_from_model_version.assert_called_once_with(
                source_database_name=sql_identifier.SqlIdentifier("TEMP"),
                source_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
                source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_create_from_model_version_add(self) -> None:
        with mock.patch.object(
            self.m_ops._model_version_client, "add_version_from_model_version"
        ) as mock_add_version_from_model_version, mock.patch.object(
            self.m_ops, "validate_existence", return_value=True
        ):
            self.m_ops.create_from_model_version(
                source_database_name=sql_identifier.SqlIdentifier("TEMP"),
                source_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
                source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_add_version_from_model_version.assert_called_once_with(
                source_database_name=sql_identifier.SqlIdentifier("TEMP"),
                source_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
                source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_invoke_method_1(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2"])
        self._add_id_check_mock_operations(m_df, [Row(1)])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1", "COL2")
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
        ) as mock_convert_from_df, mock.patch.object(
            self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                method_function_type=model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                signature=m_sig,
                X=pd_df,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session, mock.ANY, keep_order=True, features=m_sig.inputs
            )
            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=['"input"'],
                returns=[("output", spt.FloatType(), '"output"')],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_to_df.assert_called_once_with(m_df, features=m_sig.outputs)

    def test_invoke_method_1_no_sort(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2"])
        self._add_id_check_mock_operations(m_df, [Row(None)])
        m_df.add_mock_drop("COL1", "COL2")
        with self.assertWarns(Warning):
            with mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
            ) as mock_convert_from_df, mock.patch.object(
                self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method, mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
            ) as mock_convert_to_df:
                self.m_ops.invoke_method(
                    method_name=sql_identifier.SqlIdentifier("PREDICT"),
                    method_function_type=model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                    signature=m_sig,
                    X=pd_df,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
                mock_convert_from_df.assert_called_once_with(
                    self.c_session, mock.ANY, keep_order=True, features=m_sig.inputs
                )
                mock_invoke_method.assert_called_once_with(
                    method_name=sql_identifier.SqlIdentifier("PREDICT"),
                    input_df=m_df,
                    input_args=['"input"'],
                    returns=[("output", spt.FloatType(), '"output"')],
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
        self._add_id_check_mock_operations(m_df, [Row(1)])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1")
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
        ) as mock_convert_from_df, mock.patch.object(
            self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                method_function_type=model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                signature=m_sig,
                X=pd_df,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session, mock.ANY, keep_order=True, features=m_sig.inputs
            )
            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=['"input"'],
                returns=[("output", spt.FloatType(), '"output"')],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
            self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df"
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                method_function_type=model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                signature=m_sig,
                X=cast(DataFrame, m_df),
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_not_called()
            mock_validate_snowpark_data.assert_called_once_with(m_df, m_sig.inputs, strict=False)

            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=["INPUT"],
                returns=[("output", spt.FloatType(), "OUTPUT")],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_to_df.assert_not_called()

    def test_invoke_method_3(self) -> None:
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df"
        ) as mock_convert_from_df, mock.patch.object(
            model_signature, "_validate_snowpark_data", return_value=model_signature.SnowparkIdentifierRule.NORMALIZED
        ) as mock_validate_snowpark_data, mock.patch.object(
            self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df"
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                method_function_type=model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                signature=m_sig,
                X=cast(DataFrame, m_df),
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                strict_input_validation=True,
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_not_called()
            mock_validate_snowpark_data.assert_called_once_with(m_df, m_sig.inputs, strict=True)

            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=["INPUT"],
                returns=[("output", spt.FloatType(), "OUTPUT")],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_convert_to_df.assert_not_called()

    def test_invoke_method_table_function(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict_table"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2"])
        self._add_id_check_mock_operations(m_df, [Row(1)])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1", "COL2")
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
        ) as mock_convert_from_df, mock.patch.object(
            self.m_ops._model_version_client, "invoke_table_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                method_function_type=model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
                signature=m_sig,
                X=pd_df,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
                is_partitioned=True,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session, mock.ANY, keep_order=True, features=m_sig.inputs
            )
            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                input_df=m_df,
                input_args=['"input"'],
                partition_column=None,
                returns=[("output", spt.FloatType(), '"output"')],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
                is_partitioned=True,
            )
            mock_convert_to_df.assert_called_once_with(m_df, features=m_sig.outputs)

    def test_invoke_method_table_function_partition_column(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict_table"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2", "PARTITION_COLUMN"])
        self._add_id_check_mock_operations(m_df, [Row(1)])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1", "COL2")
        partition_column = sql_identifier.SqlIdentifier("PARTITION_COLUMN")
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
        ) as mock_convert_from_df, mock.patch.object(
            self.m_ops._model_version_client, "invoke_table_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                method_function_type=model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
                signature=m_sig,
                X=pd_df,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                partition_column=partition_column,
                statement_params=self.m_statement_params,
                is_partitioned=True,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session, mock.ANY, keep_order=True, features=m_sig.inputs
            )
            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                input_df=m_df,
                input_args=['"input"'],
                partition_column=partition_column,
                returns=[("output", spt.FloatType(), '"output"')],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
                is_partitioned=True,
            )
            mock_convert_to_df.assert_called_once_with(m_df, features=m_sig.outputs)

    def test_invoke_method_service(self) -> None:
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])
        with mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df"
        ) as mock_convert_from_df, mock.patch.object(
            model_signature, "_validate_snowpark_data", return_value=model_signature.SnowparkIdentifierRule.NORMALIZED
        ) as mock_validate_snowpark_data, mock.patch.object(
            self.m_ops._service_client, "invoke_function_method", return_value=m_df
        ) as mock_invoke_method, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df"
        ) as mock_convert_to_df:
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                signature=m_sig,
                X=cast(DataFrame, m_df),
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_not_called()
            mock_validate_snowpark_data.assert_called_once_with(m_df, m_sig.inputs, strict=False)

            mock_invoke_method.assert_called_once_with(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=m_df,
                input_args=["INPUT"],
                returns=[("output", spt.FloatType(), "OUTPUT")],
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
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
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, "This is a comment")
            mock_show_models.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
                default_version_name="V1",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.get_comment(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, "This is a comment")
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_set_comment_1(self) -> None:
        with mock.patch.object(self.m_ops._model_client, "set_comment") as mock_set_comment:
            self.m_ops.set_comment(
                comment="This is a comment",
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            mock_set_comment.assert_called_once_with(
                comment="This is a comment",
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_set_comment_2(self) -> None:
        with mock.patch.object(self.m_ops._model_version_client, "set_comment") as mock_set_comment:
            self.m_ops.set_comment(
                comment="This is a comment",
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_comment.assert_called_once_with(
                comment="This is a comment",
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_get_default_version(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="MODEL",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="v1",
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res) as mock_show_models:
            res = self.m_ops.get_default_version(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, sql_identifier.SqlIdentifier("v1", case_sensitive=True))
            mock_show_models.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_system_aliases(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                aliases=["FIRST", "DEFAULT"],
            ),
            Row(
                create_on="06/01",
                name="v2",
                comment="This is a comment",
                model_name="MODEL",
                aliases=["LAST"],
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("FIRST"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, sql_identifier.SqlIdentifier("v1", case_sensitive=True))

            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("LAST"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, sql_identifier.SqlIdentifier("v2", case_sensitive=True))
            self.assertEqual(mock_show_versions.call_count, 2)

    def test_set_default_version_1(self) -> None:
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
        ) as mock_show_versions, mock.patch.object(
            self.m_ops._model_version_client, "set_default_version"
        ) as mock_set_default_version:
            self.m_ops.set_default_version(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                validate_result=False,
                statement_params=self.m_statement_params,
            )
            mock_set_default_version.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )

    def test_set_default_version_2(self) -> None:
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=[]
        ) as mock_show_versions, mock.patch.object(
            self.m_ops._model_version_client, "set_default_version"
        ) as mock_set_default_version:
            with self.assertRaisesRegex(
                ValueError, "You cannot set version V1 as default version as it does not exist."
            ):
                self.m_ops.set_default_version(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                validate_result=False,
                statement_params=self.m_statement_params,
            )
            mock_set_default_version.assert_not_called()

    def test_delete_model_or_version_1(self) -> None:
        with mock.patch.object(
            self.m_ops._model_client,
            "drop_model",
        ) as mock_drop_model:
            self.m_ops.delete_model_or_version(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            mock_drop_model.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )

    def test_delete_model_or_version_2(self) -> None:
        with mock.patch.object(
            self.m_ops._model_version_client,
            "drop_version",
        ) as mock_drop_version:
            self.m_ops.delete_model_or_version(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V2"),
                statement_params=self.m_statement_params,
            )
            mock_drop_version.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V2"),
                statement_params=self.m_statement_params,
            )

    def test_rename(self) -> None:
        with mock.patch.object(
            self.m_ops._model_client,
            "rename",
        ) as mock_rename:
            self.m_ops.rename(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                new_model_db=None,
                new_model_schema=None,
                new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
                statement_params=self.m_statement_params,
            )
            mock_rename.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                new_model_db=None,
                new_model_schema=None,
                new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
                statement_params=self.m_statement_params,
            )

    def test_rename_fully_qualified_name(self) -> None:
        with mock.patch.object(
            self.m_ops._model_client,
            "rename",
        ) as mock_rename:
            self.m_ops.rename(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                new_model_db=sql_identifier.SqlIdentifier("TEMP"),
                new_model_schema=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
                statement_params=self.m_statement_params,
            )
            mock_rename.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                new_model_db=sql_identifier.SqlIdentifier("TEMP"),
                new_model_schema=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
                statement_params=self.m_statement_params,
            )

    def test_match_model_spec_with_sql_functions(self) -> None:
        with self.assertRaises(AssertionError):
            model_ops.ModelOperator._match_model_spec_with_sql_functions(
                [sql_identifier.SqlIdentifier("ABC")], ["predict"]
            )

        with self.assertRaises(AssertionError):
            self.assertDictEqual(
                {},
                model_ops.ModelOperator._match_model_spec_with_sql_functions([], ["predict"]),
            )

        self.assertDictEqual(
            {sql_identifier.SqlIdentifier("PREDICT"): "predict"},
            model_ops.ModelOperator._match_model_spec_with_sql_functions(
                [sql_identifier.SqlIdentifier("PREDICT")], ["predict"]
            ),
        )

        self.assertDictEqual(
            {sql_identifier.SqlIdentifier('"predict"'): "predict"},
            model_ops.ModelOperator._match_model_spec_with_sql_functions(
                [sql_identifier.SqlIdentifier('"predict"')], ["predict"]
            ),
        )

        self.assertDictEqual(
            {sql_identifier.SqlIdentifier('"predict"'): "predict", sql_identifier.SqlIdentifier("PREDICT"): "PREDICT"},
            model_ops.ModelOperator._match_model_spec_with_sql_functions(
                [sql_identifier.SqlIdentifier("PREDICT"), sql_identifier.SqlIdentifier('"predict"')],
                ["predict", "PREDICT"],
            ),
        )

    def test_get_functions(self) -> None:
        m_spec = {
            "signatures": {
                "predict": _DUMMY_SIG["predict"].to_dict(),
                "predict_table": _DUMMY_SIG["predict_table"].to_dict(),
            }
        }
        m_show_versions_result = [Row(model_spec=yaml.safe_dump(m_spec))]
        m_show_functions_result = [
            Row(name="predict", return_type="NUMBER"),
            Row(name="predict_table", return_type="TABLE (RESULTS VARCHAR)"),
        ]
        with mock.patch.object(
            self.m_ops._model_client,
            "show_versions",
            return_value=m_show_versions_result,
        ) as mock_show_versions, mock.patch.object(
            self.m_ops._model_version_client, "show_functions", return_value=m_show_functions_result
        ) as mock_show_functions, mock.patch.object(
            model_meta.ModelMetadata,
            "_validate_model_metadata",
            return_value=cast(model_meta_schema.ModelMetadataDict, m_spec),
        ) as mock_validate_model_metadata:
            self.m_ops.get_functions(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                check_model_details=True,
                statement_params={**self.m_statement_params, "SHOW_MODEL_DETAILS_IN_SHOW_VERSIONS_IN_MODEL": True},
            )
            mock_show_functions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            mock_validate_model_metadata.assert_called_once_with(m_spec)

    def test_get_model_task(self) -> None:
        m_spec = {
            "signatures": {
                "predict": _DUMMY_SIG["predict"].to_dict(),
                "predict_table": _DUMMY_SIG["predict_table"].to_dict(),
            },
            "task": "TABULAR_BINARY_CLASSIFICATION",
        }
        m_show_versions_result = [Row(model_spec=yaml.safe_dump(m_spec))]
        with mock.patch.object(
            self.m_ops._model_client,
            "show_versions",
            return_value=m_show_versions_result,
        ) as mock_show_versions, mock.patch.object(
            model_meta.ModelMetadata,
            "_validate_model_metadata",
            return_value=cast(model_meta_schema.ModelMetadataDict, m_spec),
        ) as mock_validate_model_metadata:
            res = self.m_ops.get_model_task(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                check_model_details=True,
                statement_params={**self.m_statement_params, "SHOW_MODEL_DETAILS_IN_SHOW_VERSIONS_IN_MODEL": True},
            )
            mock_validate_model_metadata.assert_called_once_with(m_spec)
            self.assertEqual(res, type_hints.Task.TABULAR_BINARY_CLASSIFICATION)

    def test_get_model_task_empty(self) -> None:
        m_spec = {
            "signatures": {
                "predict": _DUMMY_SIG["predict"].to_dict(),
                "predict_table": _DUMMY_SIG["predict_table"].to_dict(),
            }
        }
        m_show_versions_result = [Row(model_spec=yaml.safe_dump(m_spec))]
        with mock.patch.object(
            self.m_ops._model_client,
            "show_versions",
            return_value=m_show_versions_result,
        ) as mock_show_versions, mock.patch.object(
            model_meta.ModelMetadata,
            "_validate_model_metadata",
            return_value=cast(model_meta_schema.ModelMetadataDict, m_spec),
        ) as mock_validate_model_metadata:
            res = self.m_ops.get_model_task(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                check_model_details=True,
                statement_params={**self.m_statement_params, "SHOW_MODEL_DETAILS_IN_SHOW_VERSIONS_IN_MODEL": True},
            )
            mock_validate_model_metadata.assert_called_once_with(m_spec)
            self.assertEqual(res, type_hints.Task.UNKNOWN)

    def test_download_files_minimal(self) -> None:
        m_list_files_res = [
            [Row(name="versions/v1/model/model.yaml", size=419, md5="1234", last_modified="")],
            [
                Row(name="versions/v1/model/env/conda.yml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/env/requirements.txt", size=419, md5="1234", last_modified=""),
            ],
            [
                Row(name="versions/v1/model/runtimes/cpu/env/conda.yml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/runtimes/cpu/env/requirements.txt", size=419, md5="1234", last_modified=""),
            ],
        ]
        m_local_path = pathlib.Path("/tmp")
        with mock.patch.object(
            self.m_ops._model_version_client,
            "list_file",
            side_effect=m_list_files_res,
        ) as mock_list_file, mock.patch.object(
            self.m_ops._model_version_client, "get_file"
        ) as mock_get_file, mock.patch.object(
            pathlib.Path, "mkdir"
        ):
            self.m_ops.download_files(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                target_path=m_local_path,
                mode="minimal",
                statement_params=self.m_statement_params,
            )
            mock_list_file.assert_has_calls(
                [
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/model.yaml"),
                        is_dir=False,
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env"),
                        is_dir=True,
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/runtimes"),
                        is_dir=True,
                        statement_params=self.m_statement_params,
                    ),
                ]
            )
            mock_get_file.assert_has_calls(
                [
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/model.yaml"),
                        target_path=m_local_path / "model",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env/conda.yml"),
                        target_path=m_local_path / "model" / "env",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env/requirements.txt"),
                        target_path=m_local_path / "model" / "env",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/runtimes/cpu/env/conda.yml"),
                        target_path=m_local_path / "model" / "runtimes" / "cpu" / "env",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/runtimes/cpu/env/requirements.txt"),
                        target_path=m_local_path / "model" / "runtimes" / "cpu" / "env",
                        statement_params=self.m_statement_params,
                    ),
                ]
            )

    def test_download_files_model(self) -> None:
        m_list_files_res = [
            [
                Row(name="versions/v1/model/model.yaml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/env/conda.yml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/env/requirements.txt", size=419, md5="1234", last_modified=""),
            ],
        ]
        m_local_path = pathlib.Path("/tmp")
        with mock.patch.object(
            self.m_ops._model_version_client,
            "list_file",
            side_effect=m_list_files_res,
        ) as mock_list_file, mock.patch.object(
            self.m_ops._model_version_client, "get_file"
        ) as mock_get_file, mock.patch.object(
            pathlib.Path, "mkdir"
        ):
            self.m_ops.download_files(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                target_path=m_local_path,
                mode="model",
                statement_params=self.m_statement_params,
            )
            mock_list_file.assert_has_calls(
                [
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model"),
                        is_dir=True,
                        statement_params=self.m_statement_params,
                    ),
                ]
            )
            mock_get_file.assert_has_calls(
                [
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/model.yaml"),
                        target_path=m_local_path / "model",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env/conda.yml"),
                        target_path=m_local_path / "model" / "env",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env/requirements.txt"),
                        target_path=m_local_path / "model" / "env",
                        statement_params=self.m_statement_params,
                    ),
                ]
            )

    def test_download_files_full(self) -> None:
        m_list_files_res = [
            [
                Row(name="versions/v1/MANIFEST.yml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/model.yaml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/env/conda.yml", size=419, md5="1234", last_modified=""),
                Row(name="versions/v1/model/env/requirements.txt", size=419, md5="1234", last_modified=""),
            ],
        ]
        m_local_path = pathlib.Path("/tmp")
        with mock.patch.object(
            self.m_ops._model_version_client,
            "list_file",
            side_effect=m_list_files_res,
        ) as mock_list_file, mock.patch.object(
            self.m_ops._model_version_client, "get_file"
        ) as mock_get_file, mock.patch.object(
            pathlib.Path, "mkdir"
        ):
            self.m_ops.download_files(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                target_path=m_local_path,
                mode="full",
                statement_params=self.m_statement_params,
            )
            mock_list_file.assert_has_calls(
                [
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("."),
                        is_dir=True,
                        statement_params=self.m_statement_params,
                    ),
                ]
            )
            mock_get_file.assert_has_calls(
                [
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("MANIFEST.yml"),
                        target_path=m_local_path,
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/model.yaml"),
                        target_path=m_local_path / "model",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env/conda.yml"),
                        target_path=m_local_path / "model" / "env",
                        statement_params=self.m_statement_params,
                    ),
                    mock.call(
                        database_name=sql_identifier.SqlIdentifier("TEMP"),
                        schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        model_name=sql_identifier.SqlIdentifier("MODEL"),
                        version_name=sql_identifier.SqlIdentifier('"v1"'),
                        file_path=pathlib.PurePosixPath("model/env/requirements.txt"),
                        target_path=m_local_path / "model" / "env",
                        statement_params=self.m_statement_params,
                    ),
                ]
            )

    def test_set_alias(self) -> None:
        with mock.patch.object(self.m_ops._model_version_client, "set_alias") as mock_set_alias:
            self.m_ops.set_alias(
                alias_name=sql_identifier.SqlIdentifier("ally"),
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_alias.assert_called_once_with(
                alias_name="ALLY",
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_unset_alias(self) -> None:
        with mock.patch.object(self.m_ops._model_version_client, "unset_alias") as mock_unset_alias:
            self.m_ops.unset_alias(
                version_or_alias_name=sql_identifier.SqlIdentifier("ally"),
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )
            mock_unset_alias.assert_called_once_with(
                version_or_alias_name="ALLY",
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=self.m_statement_params,
            )


if __name__ == "__main__":
    absltest.main()
