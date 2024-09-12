import hashlib
import pathlib
import uuid
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model._client.sql import service as service_sql
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils


class ModelOpsTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.m_statement_params = {"test": "1"}
        self.c_session = cast(Session, self.m_session)
        self.m_ops = service_ops.ServiceOperator(
            self.c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )

    def test_create_service(self) -> None:
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",) as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ), mock.patch.object(self.m_ops._model_deployment_spec, "save",) as mock_save, mock.patch.object(
            file_utils, "upload_directory_to_stage", return_value=None
        ) as mock_upload_directory_to_stage, mock.patch.object(
            self.m_ops._service_client,
            "deploy_model",
            return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
        ) as mock_deploy_model, mock.patch.object(
            self.m_ops._service_client,
            "get_service_status",
            return_value=(service_sql.ServiceStatus.PENDING, None),
        ) as mock_get_service_status:
            self.m_ops.create_service(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
                service_database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                service_schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                gpu_requests="1",
                num_workers=1,
                force_rebuild=True,
                build_external_access_integration=sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION"),
                statement_params=self.m_statement_params,
            )
            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )
            mock_save.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
                service_database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                service_schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                gpu="1",
                num_workers=1,
                force_rebuild=True,
                external_access_integration=sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION"),
            )
            mock_upload_directory_to_stage.assert_called_once_with(
                self.c_session,
                local_path=self.m_ops._model_deployment_spec.workspace_path,
                stage_path=pathlib.PurePosixPath(
                    self.m_ops._stage_client.fully_qualified_object_name(
                        sql_identifier.SqlIdentifier("DB"),
                        sql_identifier.SqlIdentifier("SCHEMA"),
                        sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                    )
                ),
                statement_params=self.m_statement_params,
            )
            mock_deploy_model.assert_called_once_with(
                stage_path="DB.SCHEMA.SNOWPARK_TEMP_STAGE_ABCDEF0123",
                model_deployment_spec_file_rel_path=self.m_ops._model_deployment_spec.DEPLOY_SPEC_FILE_REL_PATH,
                statement_params=self.m_statement_params,
            )
            mock_get_service_status.assert_called_once_with(
                service_name="SERVICE",
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_get_model_build_service_name(self) -> None:
        query_id = str(uuid.uuid4())
        most_significant_bits = uuid.UUID(query_id).int >> 64
        md5_hash = hashlib.md5(str(most_significant_bits).encode()).hexdigest()
        identifier = md5_hash[:6]
        service_name = ("model_build_" + identifier).upper()
        self.assertEqual(self.m_ops._get_model_build_service_name(query_id), service_name)


if __name__ == "__main__":
    absltest.main()
