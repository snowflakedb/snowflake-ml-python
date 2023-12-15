import json
from typing import Any, Dict, cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.ops import metadata_ops
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Row, Session


class metadataOpsTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.m_statement_params = {"test": "1"}
        self.c_session = cast(Session, self.m_session)
        self.m_ops = metadata_ops.MetadataOperator(
            self.c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )

    def test_get_metadata_dict_1(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="Model",
                metadata=None,
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            metadata_dict = self.m_ops._get_current_metadata_dict(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(metadata_dict, {})
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_get_metadata_dict_2(self) -> None:
        m_meta: Dict[str, Any] = {}
        m_list_res = [
            Row(
                create_on="06/01",
                name="Model",
                metadata=json.dumps(m_meta),
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            metadata_dict = self.m_ops._get_current_metadata_dict(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(metadata_dict, m_meta)
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_get_metadata_dict_3(self) -> None:
        m_meta = {"metrics": 1}
        m_list_res = [
            Row(
                create_on="06/01",
                name="Model",
                metadata=json.dumps(m_meta),
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            metadata_dict = self.m_ops._get_current_metadata_dict(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(metadata_dict, m_meta)
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_get_metadata_dict_4(self) -> None:
        m_meta = "metrics"
        m_list_res = [
            Row(
                create_on="06/01",
                name="Model",
                metadata=json.dumps(m_meta),
                model_name="MODEL",
                database_name="TEMP",
                schema_name="test",
            ),
        ]
        with mock.patch.object(
            self.m_ops._model_client, "show_versions", return_value=m_list_res
        ) as mock_show_versions:
            with self.assertRaisesRegex(ValueError, "Metadata is expected to be a dictionary"):
                self.m_ops._get_current_metadata_dict(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_show_versions.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_1(self) -> None:
        m_meta: Dict[str, Any] = {}
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            loaded_meta = self.m_ops.load(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(
                loaded_meta,
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
            )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_2(self) -> None:
        m_meta: Dict[str, Any] = {"metrics": {"a": 1}}
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            loaded_meta = self.m_ops.load(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(
                loaded_meta,
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
            )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_3(self) -> None:
        m_meta: Dict[str, Any] = {"snowpark_ml_schema_version": 1}
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            with self.assertRaisesRegex(ValueError, "Unsupported model metadata schema version"):
                self.m_ops.load(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_4(self) -> None:
        m_meta: Dict[str, Any] = {"snowpark_ml_schema_version": "2023-12-01"}
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            with self.assertRaisesRegex(ValueError, "Unsupported model metadata schema version"):
                self.m_ops.load(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_5(self) -> None:
        m_meta: Dict[str, Any] = {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION}
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            loaded_meta = self.m_ops.load(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(
                loaded_meta,
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
            )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_6(self) -> None:
        m_meta: Dict[str, Any] = {
            "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
            "metrics": 1,
        }
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            with self.assertRaisesRegex(ValueError, "Metrics in the metadata is expected to be a dictionary"):
                self.m_ops.load(
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=self.m_statement_params,
                )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_7(self) -> None:
        m_meta: Dict[str, Any] = {
            "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
            "metrics": {"a": 1},
        }
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            loaded_meta = self.m_ops.load(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(
                loaded_meta,
                metadata_ops.ModelVersionMetadataSchema(
                    metrics={"a": 1},
                ),
            )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_load_8(self) -> None:
        m_meta: Dict[str, Any] = {
            "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
            "metrics": {"a": 1},
            "metrics_2": 2,
        }
        with mock.patch.object(
            self.m_ops, "_get_current_metadata_dict", return_value=m_meta
        ) as mock_get_current_metadata_dict:
            loaded_meta = self.m_ops.load(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            self.assertDictEqual(
                loaded_meta,
                metadata_ops.ModelVersionMetadataSchema(
                    metrics={"a": 1},
                ),
            )
            mock_get_current_metadata_dict.assert_called_once_with(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_save_1(self) -> None:
        m_meta: Dict[str, Any] = {}
        with mock.patch.object(self.m_ops, "_get_current_metadata_dict", return_value=m_meta), mock.patch.object(
            self.m_ops._model_version_client, "set_metadata"
        ) as mock_set_metadata:
            self.m_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_metadata.assert_called_once_with(
                {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION, "metrics": {}},
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_save_2(self) -> None:
        m_meta: Dict[str, Any] = {"metrics": 1}
        with mock.patch.object(self.m_ops, "_get_current_metadata_dict", return_value=m_meta), mock.patch.object(
            self.m_ops._model_version_client, "set_metadata"
        ) as mock_set_metadata:
            self.m_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_metadata.assert_called_once_with(
                {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION, "metrics": {}},
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_save_3(self) -> None:
        m_meta: Dict[str, Any] = {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION}
        with mock.patch.object(self.m_ops, "_get_current_metadata_dict", return_value=m_meta), mock.patch.object(
            self.m_ops._model_version_client, "set_metadata"
        ) as mock_set_metadata:
            self.m_ops.save(
                metadata_ops.ModelVersionMetadataSchema(),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_metadata.assert_called_once_with(
                {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION},
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_save_4(self) -> None:
        m_meta: Dict[str, Any] = {
            "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
        }
        with mock.patch.object(self.m_ops, "_get_current_metadata_dict", return_value=m_meta), mock.patch.object(
            self.m_ops._model_version_client, "set_metadata"
        ) as mock_set_metadata:
            self.m_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_metadata.assert_called_once_with(
                {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION, "metrics": {}},
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_save_5(self) -> None:
        m_meta: Dict[str, Any] = {
            "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
            "metrics_2": {},
        }
        with mock.patch.object(self.m_ops, "_get_current_metadata_dict", return_value=m_meta), mock.patch.object(
            self.m_ops._model_version_client, "set_metadata"
        ) as mock_set_metadata:
            self.m_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_metadata.assert_called_once_with(
                {
                    "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
                    "metrics": {},
                    "metrics_2": {},
                },
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )

    def test_save_6(self) -> None:
        m_meta: Dict[str, Any] = {
            "snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION,
            "metrics": {"a": 1},
        }
        with mock.patch.object(self.m_ops, "_get_current_metadata_dict", return_value=m_meta), mock.patch.object(
            self.m_ops._model_version_client, "set_metadata"
        ) as mock_set_metadata:
            self.m_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics={}),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )
            mock_set_metadata.assert_called_once_with(
                {"snowpark_ml_schema_version": metadata_ops.MODEL_VERSION_METADATA_SCHEMA_VERSION, "metrics": {}},
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=self.m_statement_params,
            )


if __name__ == "__main__":
    absltest.main()
