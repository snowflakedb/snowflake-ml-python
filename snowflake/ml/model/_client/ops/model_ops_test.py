import json
import pathlib
from typing import Optional, cast
from unittest import mock

import numpy as np
import pandas as pd
import yaml
from absl.testing import absltest, parameterized

from snowflake.ml._internal import platform_capabilities
from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.ops import model_ops
from snowflake.ml.model._client.sql import service as service_sql
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

_SERVICE_SPEC_AUTOCAPTURE_ENABLED = """
spec:
  containers:
    - name: "main"
      image: "test-image"
    - name: "proxy"
      env:
        SPCS_MODEL_INFERENCE_SERVER__AUTOCAPTURE_ENABLED: "true"
"""

_SERVICE_SPEC_AUTOCAPTURE_DISABLED = """
spec:
  containers:
    - name: "proxy"
      env:
        SPCS_MODEL_INFERENCE_SERVER__AUTOCAPTURE_ENABLED: "false"
"""

_SERVICE_SPEC_NO_PROXY = """
spec:
  containers:
    - name: "main"
      image: "test-image"
"""

_SERVICE_SPEC_NO_AUTOCAPTURE_ENV_VAR = """
spec:
  containers:
    - name: "proxy"
      env:
        SOME_OTHER_VAR: "value"
"""


class ModelOpsTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.m_statement_params = {"test": "1"}
        self.c_session = cast(Session, self.m_session)

        mock_connection = mock.Mock()
        mock_connection.host = "account.snowflakecomputing.com"
        self.m_session.connection = mock_connection

        self.m_ops = model_ops.ModelOperator(
            self.c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )

    def _add_id_check_mock_operations(
        self, m_df: mock_data_frame.MockDataFrame, collect_result: list[Row]
    ) -> mock_data_frame.MockDataFrame:
        m_df.add_operation(operation="select", args=("_ID",))
        m_df.add_operation(operation="limit", args=(1,))
        m_df.add_collect_result(collect_result)
        return m_df

    def test_prepare_model_temp_stage_path(self) -> None:
        with (
            mock.patch.object(self.m_ops._stage_client, "create_tmp_stage") as mock_create_stage,
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ) as mock_random_name_for_temp_object,
        ):
            stage_path = self.m_ops.prepare_model_temp_stage_path(
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

    def _make_mock_describe_service_row(
        self,
        dns_name: str,
        spec: Optional[str] = None,
    ) -> Row:
        """Helper to create a mock Row for describe_service results."""
        row_data = {
            "name": "TEST_SERVICE",
            "dns_name": dns_name,
            "status": "RUNNING",
        }
        if spec is not None:
            row_data["spec"] = spec

        return Row(**row_data)

    def test_show_models_or_versions_1(self) -> None:
        m_list_res = [
            Row(
                created_on="06/01",
                name="MODEL",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="V1",
            ),
            Row(
                created_on="06/01",
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
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
            Row(
                created_on="06/01",
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
                created_on="06/01",
                name="MODEL",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="V1",
            ),
            Row(
                created_on="06/01",
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
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
            Row(
                created_on="06/01",
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
                created_on="06/01",
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
        m_list_res: list[Row] = []
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
                created_on="06/01",
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
        m_list_res: list[Row] = []
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
        m_list_res: list[Row] = [
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

    def test_show_services_1(self) -> None:
        m_services_list_res = [Row(inference_services='["a.b.c", "d.e.f"]')]
        # Row objects with privatelink_ingress_url field for Business Critical accounts
        m_endpoints_list_res_0 = [Row(name="inference", ingress_url="Waiting", privatelink_ingress_url=None, port=8080)]
        m_endpoints_list_res_1 = [
            Row(
                name="inference",
                ingress_url="foo.snowflakecomputing.app",
                privatelink_ingress_url="bar.privatelink.snowflakecomputing.com",
                port=9090,
            )
        ]
        m_statuses_0 = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]
        m_statuses_1 = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.RUNNING,
                instance_id=1,
                instance_status="READY",
                container_status="READY",
                message=None,
            )
        ]

        with (
            mock.patch.object(
                self.m_ops._model_client, "show_versions", return_value=m_services_list_res
            ) as mock_show_versions,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                side_effect=[m_statuses_0, m_statuses_1, m_statuses_0, m_statuses_1],
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "show_endpoints",
                side_effect=[
                    m_endpoints_list_res_0,
                    m_endpoints_list_res_1,
                    m_endpoints_list_res_0,
                    m_endpoints_list_res_1,
                ],
            ) as mock_show_endpoints,
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(
                        dns_name="abc.internal", spec=_SERVICE_SPEC_AUTOCAPTURE_ENABLED
                    ),
                    self._make_mock_describe_service_row(
                        dns_name="def.internal", spec=_SERVICE_SPEC_AUTOCAPTURE_DISABLED
                    ),
                    self._make_mock_describe_service_row(
                        dns_name="abc.internal", spec=_SERVICE_SPEC_AUTOCAPTURE_ENABLED
                    ),
                    self._make_mock_describe_service_row(
                        dns_name="def.internal", spec=_SERVICE_SPEC_AUTOCAPTURE_DISABLED
                    ),
                ],
            ) as mock_describe_service,
        ):

            # Test with regular connection - should display the ingress_url
            with mock.patch.object(self.m_ops._session.connection, "host", "account.snowflakecomputing.com"):
                res = self.m_ops.show_services(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
                self.assertListEqual(
                    res,
                    [
                        {
                            "name": "a.b.c",
                            "status": "PENDING",
                            "inference_endpoint": None,
                            "internal_endpoint": "http://abc.internal:8080",
                            "autocapture_enabled": True,
                        },
                        {
                            "name": "d.e.f",
                            "status": "RUNNING",
                            "inference_endpoint": "foo.snowflakecomputing.app",
                            "internal_endpoint": "http://def.internal:9090",
                            "autocapture_enabled": False,
                        },
                    ],
                )

            # Test with privatelink connection - should display the privatelink_ingress_url
            with mock.patch.object(
                self.m_ops._session.connection, "host", "account.privatelink.snowflakecomputing.com"
            ):
                res = self.m_ops.show_services(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
                self.assertListEqual(
                    res,
                    [
                        {
                            "name": "a.b.c",
                            "status": "PENDING",
                            "inference_endpoint": None,
                            "internal_endpoint": "http://abc.internal:8080",
                            "autocapture_enabled": True,
                        },
                        {
                            "name": "d.e.f",
                            "status": "RUNNING",
                            "inference_endpoint": "bar.privatelink.snowflakecomputing.com",
                            "internal_endpoint": "http://def.internal:9090",
                            "autocapture_enabled": False,
                        },
                    ],
                )
            expected_show_versions_call = mock.call(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            mock_show_versions.assert_has_calls([expected_show_versions_call, expected_show_versions_call])

            expected_status_calls = [
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("a"),
                    schema_name=sql_identifier.SqlIdentifier("b"),
                    service_name=sql_identifier.SqlIdentifier("c"),
                    statement_params=self.m_statement_params,
                ),
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("d"),
                    schema_name=sql_identifier.SqlIdentifier("e"),
                    service_name=sql_identifier.SqlIdentifier("f"),
                    statement_params=self.m_statement_params,
                ),
            ]
            mock_get_service_container_statuses.assert_has_calls(expected_status_calls * 2)

            expected_endpoint_calls = [
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("a"),
                    schema_name=sql_identifier.SqlIdentifier("b"),
                    service_name=sql_identifier.SqlIdentifier("c"),
                    statement_params=self.m_statement_params,
                ),
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("d"),
                    schema_name=sql_identifier.SqlIdentifier("e"),
                    service_name=sql_identifier.SqlIdentifier("f"),
                    statement_params=self.m_statement_params,
                ),
            ]
            mock_show_endpoints.assert_has_calls(expected_endpoint_calls * 2)

            expected_describe_service_calls = [
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("a"),
                    schema_name=sql_identifier.SqlIdentifier("b"),
                    service_name=sql_identifier.SqlIdentifier("c"),
                    statement_params=self.m_statement_params,
                ),
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("d"),
                    schema_name=sql_identifier.SqlIdentifier("e"),
                    service_name=sql_identifier.SqlIdentifier("f"),
                    statement_params=self.m_statement_params,
                ),
            ]
            mock_describe_service.assert_has_calls(expected_describe_service_calls * 2)

    def test_show_services_2(self) -> None:
        m_services_list_res = [Row(inference_services='["a.b.c"]')]
        m_endpoints_list_res = [Row(name="inference", ingress_url=None, privatelink_ingress_url=None, port=None)]
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        with (
            mock.patch.object(
                self.m_ops._model_client, "show_versions", return_value=m_services_list_res
            ) as mock_show_versions,
            mock.patch.object(
                self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res
            ) as mock_show_endpoints,
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                return_value=self._make_mock_describe_service_row(
                    dns_name="abc.internal", spec=_SERVICE_SPEC_NO_AUTOCAPTURE_ENV_VAR
                ),
            ) as mock_describe_service,
        ):
            res = self.m_ops.show_services(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            # Test with regular connection
            # Inference endpoint will be None as both ingress_url and privatelink_ingress_url are None
            self.assertListEqual(
                res,
                [
                    {
                        "name": "a.b.c",
                        "status": "PENDING",
                        "inference_endpoint": None,
                        "internal_endpoint": None,
                        "autocapture_enabled": False,
                    },
                ],
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )
            mock_show_endpoints.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )
            mock_describe_service.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )

    def test_show_services_3(self) -> None:
        m_services_list_res = [Row(inference_services='["a.b.c"]')]
        m_endpoints_list_res = [
            Row(
                name="inference",
                port=8000,
                ingress_url="foo.snowflakecomputing.app",
                privatelink_ingress_url="foo.privatelink.snowflakecomputing.com",
            ),
            Row(
                name="another",
                port=9000,
                ingress_url="bar.snowflakecomputing.app",
                privatelink_ingress_url="bar.privatelink.snowflakecomputing.com",
            ),
        ]
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        with (
            mock.patch.object(
                self.m_ops._model_client, "show_versions", return_value=m_services_list_res
            ) as mock_show_versions,
            mock.patch.object(
                self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res
            ) as mock_show_endpoints,
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                return_value=self._make_mock_describe_service_row(dns_name="abc.internal", spec=_SERVICE_SPEC_NO_PROXY),
            ) as mock_describe_service,
        ):
            res = self.m_ops.show_services(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            # Test with regular connection
            # Inference endpoint will take the ingress_url value only if the name field contains "inference"
            self.assertListEqual(
                res,
                [
                    {
                        "name": "a.b.c",
                        "status": "PENDING",
                        "inference_endpoint": "foo.snowflakecomputing.app",
                        "internal_endpoint": "http://abc.internal:8000",
                        "autocapture_enabled": False,
                    },
                ],
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )
            mock_show_endpoints.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )
            mock_describe_service.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )

    def test_show_services_4(self) -> None:
        m_services_list_res = [Row(inference_services='["a.b.c"]')]
        m_endpoints_list_res = [
            Row(
                name="custom",
                port=7000,
                ingress_url="foo.snowflakecomputing.app",
                privatelink_ingress_url="foo.privatelink.snowflakecomputing.com",
            )
        ]
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        with (
            mock.patch.object(
                self.m_ops._model_client, "show_versions", return_value=m_services_list_res
            ) as mock_show_versions,
            mock.patch.object(
                self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res
            ) as mock_show_endpoints,
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                return_value=self._make_mock_describe_service_row(dns_name="abc.internal"),
            ) as mock_describe_service,
        ):
            res = self.m_ops.show_services(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            # Test with regular connection
            # Inference endpoint will be None as the name field does not contain "inference"
            self.assertListEqual(
                res,
                [
                    {
                        "name": "a.b.c",
                        "status": "PENDING",
                        "inference_endpoint": None,
                        "internal_endpoint": None,
                        "autocapture_enabled": False,
                    },
                ],
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )
            mock_show_endpoints.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )
            mock_describe_service.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("c"),
                statement_params=self.m_statement_params,
            )

    def test_show_services_5(self) -> None:
        """Test show_services for non-Business Critical accounts where privatelink_ingress_url column doesn't exist."""
        m_services_list_res = [Row(inference_services='["a.b.c", "d.e.f"]')]
        # Row objects without privatelink_ingress_url field
        m_endpoints_list_res_0 = [Row(name="inference", port=8080, ingress_url="bar.snowflakecomputing.app")]
        m_endpoints_list_res_1 = [Row(name="inference", port=9090, ingress_url="foo.snowflakecomputing.app")]
        m_statuses_0 = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]
        m_statuses_1 = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.RUNNING,
                instance_id=1,
                instance_status="READY",
                container_status="READY",
                message=None,
            )
        ]

        with (
            mock.patch.object(
                self.m_ops._model_client, "show_versions", return_value=m_services_list_res
            ) as mock_show_versions,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                side_effect=[m_statuses_0, m_statuses_1, m_statuses_0, m_statuses_1],
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "show_endpoints",
                side_effect=[
                    m_endpoints_list_res_0,
                    m_endpoints_list_res_1,
                    m_endpoints_list_res_0,
                    m_endpoints_list_res_1,
                ],
            ) as mock_show_endpoints,
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ) as mock_describe_service,
        ):

            # Test with regular connection
            with mock.patch.object(self.m_ops._session.connection, "host", "account.snowflakecomputing.com"):
                res = self.m_ops.show_services(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
                self.assertListEqual(
                    res,
                    [
                        {
                            "name": "a.b.c",
                            "status": "PENDING",
                            "inference_endpoint": "bar.snowflakecomputing.app",
                            "internal_endpoint": "http://abc.internal:8080",
                            "autocapture_enabled": False,
                        },
                        {
                            "name": "d.e.f",
                            "status": "RUNNING",
                            "inference_endpoint": "foo.snowflakecomputing.app",
                            "internal_endpoint": "http://def.internal:9090",
                            "autocapture_enabled": False,
                        },
                    ],
                )

            # Test with privatelink connection - should still use ingress_url since privatelink column doesn't exist
            with mock.patch.object(
                self.m_ops._session.connection, "host", "account.privatelink.snowflakecomputing.com"
            ):
                res = self.m_ops.show_services(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=self.m_statement_params,
                )
                self.assertListEqual(
                    res,
                    [
                        {
                            "name": "a.b.c",
                            "status": "PENDING",
                            "inference_endpoint": "bar.snowflakecomputing.app",
                            "internal_endpoint": "http://abc.internal:8080",
                            "autocapture_enabled": False,
                        },
                        {
                            "name": "d.e.f",
                            "status": "RUNNING",
                            "inference_endpoint": "foo.snowflakecomputing.app",
                            "internal_endpoint": "http://def.internal:9090",
                            "autocapture_enabled": False,
                        },
                    ],
                )

            expected_show_versions_call = mock.call(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            mock_show_versions.assert_has_calls([expected_show_versions_call, expected_show_versions_call])

            expected_status_calls = [
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("a"),
                    schema_name=sql_identifier.SqlIdentifier("b"),
                    service_name=sql_identifier.SqlIdentifier("c"),
                    statement_params=self.m_statement_params,
                ),
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("d"),
                    schema_name=sql_identifier.SqlIdentifier("e"),
                    service_name=sql_identifier.SqlIdentifier("f"),
                    statement_params=self.m_statement_params,
                ),
            ]
            mock_get_service_container_statuses.assert_has_calls(expected_status_calls * 2)

            expected_endpoint_calls = [
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("a"),
                    schema_name=sql_identifier.SqlIdentifier("b"),
                    service_name=sql_identifier.SqlIdentifier("c"),
                    statement_params=self.m_statement_params,
                ),
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("d"),
                    schema_name=sql_identifier.SqlIdentifier("e"),
                    service_name=sql_identifier.SqlIdentifier("f"),
                    statement_params=self.m_statement_params,
                ),
            ]
            mock_show_endpoints.assert_has_calls(expected_endpoint_calls * 2)

            expected_describe_service_calls = [
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("a"),
                    schema_name=sql_identifier.SqlIdentifier("b"),
                    service_name=sql_identifier.SqlIdentifier("c"),
                    statement_params=self.m_statement_params,
                ),
                mock.call(
                    database_name=sql_identifier.SqlIdentifier("d"),
                    schema_name=sql_identifier.SqlIdentifier("e"),
                    service_name=sql_identifier.SqlIdentifier("f"),
                    statement_params=self.m_statement_params,
                ),
            ]
            mock_describe_service.assert_has_calls(expected_describe_service_calls * 2)

    def test_show_services_pre_bcr(self) -> None:
        m_list_res = [Row(comment="mycomment")]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            with self.assertRaises(exceptions.SnowflakeMLException) as context:
                self.m_ops.show_services(
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

    def test_show_services_skip_build(self) -> None:
        m_list_res = [Row(inference_services='["A.B.MODEL_BUILD_34d35ew", "A.B.SERVICE"]')]
        m_endpoints_list_res = [Row(name="fooendpoint"), Row(name="barendpoint")]
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(
                self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client, "show_endpoints", side_effect=[m_endpoints_list_res]
            ) as mock_show_endpoints,
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                return_value=self._make_mock_describe_service_row(dns_name="service.test.e6nv.svc.spcs.internal"),
            ) as mock_describe_service,
        ):
            res = self.m_ops.show_services(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            self.assertListEqual(
                res,
                [
                    {
                        "name": "A.B.SERVICE",
                        "status": "PENDING",
                        "inference_endpoint": None,
                        "internal_endpoint": None,
                        "autocapture_enabled": False,
                    },
                ],
            )
            mock_show_versions.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                statement_params=self.m_statement_params,
            )
            mock_show_endpoints.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                statement_params=self.m_statement_params,
            )
            mock_describe_service.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("a"),
                schema_name=sql_identifier.SqlIdentifier("b"),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                statement_params=self.m_statement_params,
            )

    def test_delete_service_non_existent(self) -> None:
        m_list_res = [Row(inference_services='["A.B.C", "D.E.F"]')]
        m_endpoints_list_res = [Row(name="fooendpoint"), Row(name="barendpoint")]
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses),
            mock.patch.object(self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res),
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ),
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with self.assertRaisesRegex(
                ValueError, "Service 'A.B.A' does not exist or unauthorized or not associated with this model version."
            ):
                self.m_ops.delete_service(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    service_database_name=sql_identifier.SqlIdentifier("A"),
                    service_schema_name=sql_identifier.SqlIdentifier("B"),
                    service_name=sql_identifier.SqlIdentifier("A"),
                )
            mock_show_versions.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses),
            mock.patch.object(self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res),
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ),
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Service 'FOO.\"bar\".B' does not exist or unauthorized or not associated with this model version.",
            ):
                self.m_ops.delete_service(
                    database_name=sql_identifier.SqlIdentifier("foo"),
                    schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    service_database_name=None,
                    service_schema_name=None,
                    service_name=sql_identifier.SqlIdentifier("B"),
                )
            mock_show_versions.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("foo"),
                schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses),
            mock.patch.object(self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res),
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ),
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Service 'TEMP.\"test\".D' does not exist or unauthorized or not associated with this model version.",
            ):
                self.m_ops.delete_service(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    service_database_name=None,
                    service_schema_name=None,
                    service_name=sql_identifier.SqlIdentifier("D"),
                )
            mock_show_versions.assert_called_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )

    def test_delete_service_exists(self) -> None:
        m_list_res = [Row(inference_services='["A.B.C", "D.E.F"]')]
        m_endpoints_list_res = [Row(name="fooendpoint"), Row(name="barendpoint")]
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(self.m_ops._service_client, "drop_service") as mock_drop_service,
            mock.patch.object(self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses),
            mock.patch.object(self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res),
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ),
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            self.m_ops.delete_service(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_database_name=sql_identifier.SqlIdentifier("A"),
                service_schema_name=sql_identifier.SqlIdentifier("B"),
                service_name=sql_identifier.SqlIdentifier("C"),
            )
            mock_show_versions.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_drop_service.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("A"),
                schema_name=sql_identifier.SqlIdentifier("B"),
                service_name=sql_identifier.SqlIdentifier("C"),
                statement_params=mock.ANY,
            )

        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(self.m_ops._service_client, "drop_service") as mock_drop_service,
            mock.patch.object(self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses),
            mock.patch.object(self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res),
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="abc.internal"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ),
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            self.m_ops.delete_service(
                database_name=sql_identifier.SqlIdentifier("A"),
                schema_name=sql_identifier.SqlIdentifier("B"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("C"),
            )
            mock_show_versions.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("A"),
                schema_name=sql_identifier.SqlIdentifier("B"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_drop_service.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("A"),
                schema_name=sql_identifier.SqlIdentifier("B"),
                service_name=sql_identifier.SqlIdentifier("C"),
                statement_params=mock.ANY,
            )
        with (
            mock.patch.object(
                self.m_ops._model_client,
                "show_versions",
                return_value=[Row(inference_services='["TEMP.\\"test\\".C", "D.E.F"]')],
            ) as mock_show_versions,
            mock.patch.object(self.m_ops._service_client, "drop_service") as mock_drop_service,
            mock.patch.object(self.m_ops._service_client, "get_service_container_statuses", return_value=m_statuses),
            mock.patch.object(self.m_ops._service_client, "show_endpoints", return_value=m_endpoints_list_res),
            mock.patch.object(
                self.m_ops._service_client,
                "describe_service",
                side_effect=[
                    self._make_mock_describe_service_row(dns_name="test_internal.dns"),
                    self._make_mock_describe_service_row(dns_name="def.internal"),
                ],
            ),
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            self.m_ops.delete_service(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("C"),
            )
            mock_show_versions.assert_called_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=mock.ANY,
            )
            mock_drop_service.assert_called_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("C"),
                statement_params=mock.ANY,
            )

    def test_create_from_stage_1(self) -> None:
        mock_composer = mock.MagicMock()
        mock_composer.stage_path = '@TEMP."test".MODEL/V1'

        with (
            mock.patch.object(
                self.m_ops._model_version_client, "create_from_stage", return_value='TEMP."test".MODEL'
            ) as mock_create_from_stage,
            mock.patch.object(
                self.m_ops._model_version_client, "add_version_from_stage", return_value='TEMP."test".MODEL'
            ) as mock_add_version_from_stage,
            mock.patch.object(self.m_ops._model_client, "show_models", return_value=[]),
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
                created_on="06/01",
                name="Model",
                comment="This is a comment",
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
                default_version_name="V1",
            ),
        ]
        with (
            mock.patch.object(
                self.m_ops._model_version_client, "create_from_stage", return_value='TEMP."test".MODEL'
            ) as mock_create_from_stage,
            mock.patch.object(
                self.m_ops._model_version_client, "add_version_from_stage", return_value='TEMP."test".MODEL'
            ) as mock_add_version_from_stage,
            mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res),
            mock.patch.object(self.m_ops._model_client, attribute="show_versions", return_value=[]),
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
                created_on="06/01",
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
        with (
            mock.patch.object(
                self.m_ops._model_version_client, "create_from_stage", return_value='TEMP."test".MODEL'
            ) as mock_create_from_stage,
            mock.patch.object(
                self.m_ops._model_version_client, "add_version_from_stage", return_value='TEMP."test".MODEL'
            ) as mock_add_version_from_stagel,
            mock.patch.object(self.m_ops._model_client, "show_models", return_value=m_list_res_models),
            mock.patch.object(self.m_ops._model_client, attribute="show_versions", return_value=m_list_res_versions),
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
        ) as mock_create_from_model_version:
            self.m_ops.create_from_model_version(
                source_database_name=sql_identifier.SqlIdentifier("TEMP"),
                source_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
                source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                model_exists=False,
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
        ) as mock_add_version_from_model_version:
            self.m_ops.create_from_model_version(
                source_database_name=sql_identifier.SqlIdentifier("TEMP"),
                source_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
                source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                model_exists=True,
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
        with (
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
            ) as mock_convert_from_df,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
            ) as mock_convert_to_df,
        ):
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
                self.c_session,
                mock.ANY,
                keep_order=True,
                features=m_sig.inputs,
                statement_params=self.m_statement_params,
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
                params=None,
            )
            mock_convert_to_df.assert_called_once_with(
                m_df, features=m_sig.outputs, statement_params=self.m_statement_params
            )

    def test_invoke_method_1_no_sort(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2"])
        self._add_id_check_mock_operations(m_df, [Row(None)])
        m_df.add_mock_drop("COL1", "COL2")
        with self.assertWarns(Warning):
            with (
                mock.patch.object(
                    snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
                ) as mock_convert_from_df,
                mock.patch.object(
                    self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
                ) as mock_invoke_method,
                mock.patch.object(
                    snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
                ) as mock_convert_to_df,
            ):
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
                    self.c_session,
                    mock.ANY,
                    keep_order=True,
                    features=m_sig.inputs,
                    statement_params=self.m_statement_params,
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
                    params=None,
                )
                mock_convert_to_df.assert_called_once_with(
                    m_df, features=m_sig.outputs, statement_params=self.m_statement_params
                )

    def test_invoke_method_1_no_drop(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", '"output"'])
        self._add_id_check_mock_operations(m_df, [Row(1)])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1")
        with (
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
            ) as mock_convert_from_df,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
            ) as mock_convert_to_df,
        ):
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
                self.c_session,
                mock.ANY,
                keep_order=True,
                features=m_sig.inputs,
                statement_params=self.m_statement_params,
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
                params=None,
            )
            mock_convert_to_df.assert_called_once_with(
                m_df, features=m_sig.outputs, statement_params=self.m_statement_params
            )

    def test_invoke_method_2(self) -> None:
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])
        with (
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_from_df") as mock_convert_from_df,
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ) as mock_validate_snowpark_data,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_to_df") as mock_convert_to_df,
        ):
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
                params=None,
            )
            mock_convert_to_df.assert_not_called()

    def test_invoke_method_3(self) -> None:
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])
        with (
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_from_df") as mock_convert_from_df,
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ) as mock_validate_snowpark_data,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_to_df") as mock_convert_to_df,
        ):
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
                params=None,
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
        with (
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
            ) as mock_convert_from_df,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_table_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
            ) as mock_convert_to_df,
        ):
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
                explain_case_sensitive=False,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session,
                mock.ANY,
                keep_order=True,
                features=m_sig.inputs,
                statement_params=self.m_statement_params,
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
                explain_case_sensitive=False,
                params=None,
            )
            mock_convert_to_df.assert_called_once_with(
                m_df, features=m_sig.outputs, statement_params=self.m_statement_params
            )

    def test_invoke_method_table_function_partition_column(self) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict_table"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("_statement_params", None)
        m_df.__setattr__("columns", ["COL1", "COL2", "PARTITION_COLUMN"])
        self._add_id_check_mock_operations(m_df, [Row(1)])
        m_df.add_mock_sort("_ID", ascending=True).add_mock_drop("COL1", "COL2")
        partition_column = sql_identifier.SqlIdentifier("PARTITION_COLUMN")
        with (
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_df
            ) as mock_convert_from_df,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_table_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
            ) as mock_convert_to_df,
        ):
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
                explain_case_sensitive=False,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session,
                mock.ANY,
                keep_order=True,
                features=m_sig.inputs,
                statement_params=self.m_statement_params,
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
                explain_case_sensitive=False,
                params=None,
            )
            mock_convert_to_df.assert_called_once_with(
                m_df, features=m_sig.outputs, statement_params=self.m_statement_params
            )

    def test_invoke_method_service(self) -> None:
        m_sig = _DUMMY_SIG["predict"]
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])
        with (
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_from_df") as mock_convert_from_df,
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ) as mock_validate_snowpark_data,
            mock.patch.object(
                self.m_ops._service_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_to_df") as mock_convert_to_df,
        ):
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
                params=None,
            )
            mock_convert_to_df.assert_not_called()

    def test_invoke_method_with_params(self) -> None:
        """Test that method parameters are correctly passed through invoke_method."""
        # Create a signature with params
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0),
                model_signature.ParamSpec(name="top_k", dtype=model_signature.DataType.INT64, default_value=10),
                model_signature.ParamSpec(
                    name="message", dtype=model_signature.DataType.STRING, default_value="default"
                ),
            ],
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])

        with (
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_from_df") as mock_convert_from_df,
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ) as mock_validate_snowpark_data,
            mock.patch.object(
                self.m_ops._model_version_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_to_df") as mock_convert_to_df,
        ):
            # Pass runtime params that override defaults
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
                params={"temperature": 0.7, "message": "it's a test"},  # Override some defaults
            )
            mock_convert_from_df.assert_not_called()
            mock_validate_snowpark_data.assert_called_once_with(m_df, m_sig.inputs, strict=False)

            # Verify parameters were passed correctly
            mock_invoke_method.assert_called_once()
            call_kwargs = mock_invoke_method.call_args.kwargs
            self.assertEqual(call_kwargs["method_name"], sql_identifier.SqlIdentifier("PREDICT"))
            self.assertEqual(call_kwargs["input_df"], m_df)
            self.assertEqual(call_kwargs["input_args"], ["INPUT"])

            # Check parameters: should have all three params with overrides applied
            parameters = call_kwargs["params"]
            self.assertIsNotNone(parameters)
            self.assertEqual(len(parameters), 3)

            # Convert to dict for easier assertion
            params_dict = {param_name.identifier(): param_value for param_name, param_value in parameters}
            self.assertEqual(params_dict["TEMPERATURE"], 0.7)  # Overridden
            self.assertEqual(params_dict["TOP_K"], 10)  # Default
            self.assertEqual(params_dict["MESSAGE"], "it's a test")  # Overridden with single quote

            mock_convert_to_df.assert_not_called()

    def test_invoke_method_service_with_params(self) -> None:
        """Test that method parameters are correctly passed through invoke_method for service invocation."""
        # Create a signature with params
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0),
                model_signature.ParamSpec(name="top_k", dtype=model_signature.DataType.INT64, default_value=10),
                model_signature.ParamSpec(
                    name="message", dtype=model_signature.DataType.STRING, default_value="default"
                ),
            ],
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])

        with (
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_from_df") as mock_convert_from_df,
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ) as mock_validate_snowpark_data,
            mock.patch.object(
                self.m_ops._service_client, "invoke_function_method", return_value=m_df
            ) as mock_invoke_method,
            mock.patch.object(snowpark_handler.SnowparkDataFrameHandler, "convert_to_df") as mock_convert_to_df,
        ):
            self.m_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                signature=m_sig,
                X=cast(DataFrame, m_df),
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                statement_params=self.m_statement_params,
                params={"temperature": 0.5},
            )
            mock_convert_from_df.assert_not_called()
            mock_validate_snowpark_data.assert_called_once_with(m_df, m_sig.inputs, strict=False)

            # Verify parameters were passed correctly to service client
            mock_invoke_method.assert_called_once()
            call_kwargs = mock_invoke_method.call_args.kwargs
            parameters = call_kwargs["params"]
            self.assertIsNotNone(parameters)
            self.assertEqual(len(parameters), 3)

            param_name, param_value = parameters[0]
            self.assertEqual(param_name.identifier(), "TEMPERATURE")
            self.assertEqual(param_value, 0.5)

            param_name, param_value = parameters[1]
            self.assertEqual(param_name.identifier(), "TOP_K")
            self.assertEqual(param_value, 10)

            param_name, param_value = parameters[2]
            self.assertEqual(param_name.identifier(), "MESSAGE")
            self.assertEqual(param_value, "default")

            mock_convert_to_df.assert_not_called()

    def test_invoke_method_with_unknown_param(self) -> None:
        """Test that invoke_method raises an error when unknown params are provided."""
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0),
            ],
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])

        with (
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ),
        ):
            with self.assertRaisesRegex(
                exceptions.SnowflakeMLException,
                r"Unknown parameter\(s\): \['unknown_param'\].*Valid parameters are: \['temperature'\]",
            ):
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
                    params={"unknown_param": 0.5},
                )

    def test_invoke_method_with_invalid_param_type(self) -> None:
        """Test that invoke_method raises an error when param value has incompatible type."""
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0),
            ],
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])

        with (
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ),
        ):
            with self.assertRaisesRegex(
                exceptions.SnowflakeMLException,
                r"not compatible with dtype",
            ):
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
                    params={"temperature": "not_a_float"},  # String cannot be converted to float
                )

    def test_invoke_method_with_params_but_no_signature_params(self) -> None:
        """Test that invoke_method raises an error when params provided but signature has none."""
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            # No params in signature
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])

        with (
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ),
        ):
            with self.assertRaisesRegex(
                exceptions.SnowflakeMLException,
                r"Parameters were provided.*but this method does not accept any parameters",
            ):
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
                    params={"temperature": 0.5},
                )

    def test_invoke_method_with_case_insensitive_param_matching(self) -> None:
        """Test that param names are matched case-insensitively."""
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0),
            ],
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["INPUT"])

        with (
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ),
            mock.patch.object(
                snowpark_handler.SnowparkDataFrameHandler,
                "convert_to_df",
            ) as mock_convert_to_df,
            mock.patch.object(
                self.m_ops._model_version_client,
                "invoke_function_method",
                return_value=cast(DataFrame, m_df),
            ) as mock_invoke_method,
        ):
            # User provides "TEMPERATURE" (upper case), signature has "temperature" (lower case)
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
                params={"TEMPERATURE": 0.5},  # Upper case should match "temperature"
            )
            mock_invoke_method.assert_called_once()
            call_kwargs = mock_invoke_method.call_args.kwargs
            parameters = call_kwargs["params"]
            self.assertIsNotNone(parameters)
            self.assertEqual(len(parameters), 1)
            # Parameter name should use the signature's original case
            param_name, param_value = parameters[0]
            self.assertEqual(param_name.identifier(), "TEMPERATURE")
            self.assertEqual(param_value, 0.5)
            mock_convert_to_df.assert_not_called()

    def test_invoke_method_with_duplicate_params_different_cases(self) -> None:
        """Test that duplicate params with different cases raises an error."""
        m_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0),
            ],
        )
        m_df = mock_data_frame.MockDataFrame()
        m_df.__setattr__("columns", ["COL1", "COL2"])

        with (
            mock.patch.object(
                model_signature,
                "_validate_snowpark_data",
                return_value=model_signature.SnowparkIdentifierRule.NORMALIZED,
            ),
        ):
            with self.assertRaisesRegex(
                exceptions.SnowflakeMLException,
                r"Duplicate parameter\(s\) provided with different cases.*case-insensitive",
            ):
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
                    params={"temperature": 0.5, "TEMPERATURE": 0.7},  # Duplicate with different cases
                )

    def test_get_comment_1(self) -> None:
        m_list_res = [
            Row(
                created_on="06/01",
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
                created_on="06/01",
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
                created_on="06/01",
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
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                aliases=json.dumps(["FIRST", "DEFAULT"]),
            ),
            Row(
                created_on="06/01",
                name="v2",
                comment="This is a comment",
                model_name="MODEL",
                aliases=json.dumps(["LAST"]),
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

    def test_get_version_by_alias_exact_match_only(self) -> None:
        """Test that partial matches are rejected and only exact matches work."""
        m_list_res = [
            Row(
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                aliases=json.dumps(["Production", "DEFAULT"]),
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            # Exact match should work
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("Production"),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, sql_identifier.SqlIdentifier("v1", case_sensitive=True))

            # Partial match should NOT work
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("P"),
                statement_params=self.m_statement_params,
            )
            self.assertIsNone(res)

            self.assertEqual(mock_show_versions.call_count, 2)

    def test_get_version_by_alias_case_insensitive(self) -> None:
        """Test that alias matching follows Snowflake identifier semantics."""
        m_list_res = [
            Row(
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                aliases=json.dumps(["Production", "DEFAULT", "Staging"]),
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res):
            # Test case-insensitive matching for unquoted identifiers
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("production"),  # lowercase should match "Production"
                statement_params=self.m_statement_params,
            )
            self.assertIsNotNone(res, "Unquoted identifiers should be case-insensitive")
            self.assertEqual(res, sql_identifier.SqlIdentifier("v1", case_sensitive=True))

    def test_get_version_by_alias_quoted_identifiers(self) -> None:
        """Test that quoted identifiers are case-sensitive and exact match only."""
        m_list_res = [
            Row(
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                aliases=json.dumps(['"production"', '"Staging"', "DEFAULT"]),
            ),
            Row(
                created_on="06/02",
                name="v2",
                comment="This is a comment",
                model_name="MODEL",
                aliases=json.dumps(['"Production"']),
            ),
        ]
        with mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res):
            # Quoted identifier - exact case match should work
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier('"production"'),
                statement_params=self.m_statement_params,
            )
            self.assertIsNotNone(res, "Quoted identifier with exact case should match")
            self.assertEqual(res, sql_identifier.SqlIdentifier("v1", case_sensitive=True))

            # Quoted identifier - case-sensitive match with different case matches different version
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier('"Production"'),
                statement_params=self.m_statement_params,
            )
            self.assertIsNotNone(res, "Quoted identifier 'Production' should match v2")
            self.assertEqual(res, sql_identifier.SqlIdentifier("v2", case_sensitive=True))

            # Unquoted identifier should NOT match quoted lowercase identifier
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("production"),  # Unquoted normalizes to PRODUCTION
                statement_params=self.m_statement_params,
            )
            self.assertIsNone(res, "Unquoted 'production' should not match quoted identifier")

            # Unquoted identifier should match unquoted identifier (case-insensitive)
            res = self.m_ops.get_version_by_alias(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                alias_name=sql_identifier.SqlIdentifier("default"),
                statement_params=self.m_statement_params,
            )
            self.assertIsNotNone(res, "Unquoted 'default' should match 'DEFAULT'")
            self.assertEqual(res, sql_identifier.SqlIdentifier("v1", case_sensitive=True))

    def test_set_default_version_1(self) -> None:
        m_list_res = [
            Row(
                created_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
        ]
        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=m_list_res) as mock_show_versions,
            mock.patch.object(self.m_ops._model_version_client, "set_default_version") as mock_set_default_version,
        ):
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
        with (
            mock.patch.object(self.m_ops._model_client, "show_versions", return_value=[]) as mock_show_versions,
            mock.patch.object(self.m_ops._model_version_client, "set_default_version") as mock_set_default_version,
        ):
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
        with (
            mock.patch.object(
                self.m_ops._model_client,
                "show_versions",
                return_value=m_show_versions_result,
            ) as mock_show_versions,
            mock.patch.object(
                self.m_ops._model_version_client, "show_functions", return_value=m_show_functions_result
            ) as mock_show_functions,
            mock.patch.object(
                model_meta.ModelMetadata,
                "_validate_model_metadata",
                return_value=cast(model_meta_schema.ModelMetadataDict, m_spec),
            ) as mock_validate_model_metadata,
        ):
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

    @parameterized.parameters(  # type: ignore[misc]
        {
            "runnable_in": '["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]',
            "expected": ["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        },
        {"runnable_in": None, "expected": None},
        {"runnable_in": '["SNOWPARK_CONTAINER_SERVICES"]', "expected": ["SNOWPARK_CONTAINER_SERVICES"]},
    )
    def test_fetch_model_spec_and_target_platforms(
        self, runnable_in: Optional[str], expected: Optional[list[str]]
    ) -> None:
        m_spec = {
            "signatures": {
                "predict": _DUMMY_SIG["predict"].to_dict(),
            }
        }
        row_kwargs: dict[str, str] = {"model_spec": yaml.safe_dump(m_spec)}
        if runnable_in is not None:
            row_kwargs["runnable_in"] = runnable_in
        m_show_versions_result = [Row(**row_kwargs)]
        with (
            mock.patch.object(
                self.m_ops._model_client,
                "show_versions",
                return_value=m_show_versions_result,
            ) as mock_show_versions,
            mock.patch.object(
                model_meta.ModelMetadata,
                "_validate_model_metadata",
                return_value=cast(model_meta_schema.ModelMetadataDict, m_spec),
            ),
        ):
            model_spec, target_platforms = self.m_ops._fetch_model_spec_and_target_platforms(
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
            self.assertEqual(model_spec, m_spec)
            self.assertEqual(target_platforms, expected)

    def test_fetch_model_spec(self) -> None:
        m_spec = {
            "signatures": {
                "predict": _DUMMY_SIG["predict"].to_dict(),
            }
        }
        with mock.patch.object(
            self.m_ops,
            "_fetch_model_spec_and_target_platforms",
            return_value=(cast(model_meta_schema.ModelMetadataDict, m_spec), ["WAREHOUSE"]),
        ) as mock_combined:
            result = self.m_ops._fetch_model_spec(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            mock_combined.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier('"v1"'),
                statement_params=self.m_statement_params,
            )
            self.assertEqual(result, m_spec)

    def test_get_model_task(self) -> None:
        m_show_versions_result = [Row(name='"v1"', model_attributes='{"task": "TABULAR_BINARY_CLASSIFICATION"}')]
        with mock.patch.object(
            self.m_ops._model_client,
            "show_versions",
            return_value=m_show_versions_result,
        ) as mock_show_versions:
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
                validate_result=True,
                statement_params=self.m_statement_params,
            )
            self.assertEqual(res, type_hints.Task.TABULAR_BINARY_CLASSIFICATION)

    def test_get_model_task_empty(self) -> None:
        m_show_versions_result = [Row(name='"v1"', model_attributes="{}")]
        with mock.patch.object(
            self.m_ops._model_client,
            "show_versions",
            return_value=m_show_versions_result,
        ) as mock_show_versions:
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
                validate_result=True,
                statement_params=self.m_statement_params,
            )
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
        with (
            mock.patch.object(
                self.m_ops._model_version_client,
                "list_file",
                side_effect=m_list_files_res,
            ) as mock_list_file,
            mock.patch.object(self.m_ops._model_version_client, "get_file") as mock_get_file,
            mock.patch.object(pathlib.Path, "mkdir"),
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
        with (
            mock.patch.object(
                self.m_ops._model_version_client,
                "list_file",
                side_effect=m_list_files_res,
            ) as mock_list_file,
            mock.patch.object(self.m_ops._model_version_client, "get_file") as mock_get_file,
            mock.patch.object(pathlib.Path, "mkdir"),
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
        with (
            mock.patch.object(
                self.m_ops._model_version_client,
                "list_file",
                side_effect=m_list_files_res,
            ) as mock_list_file,
            mock.patch.object(self.m_ops._model_version_client, "get_file") as mock_get_file,
            mock.patch.object(pathlib.Path, "mkdir"),
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
