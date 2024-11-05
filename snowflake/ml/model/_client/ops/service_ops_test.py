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
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Session, row
from snowflake.snowpark._internal import utils as snowpark_utils


class ModelOpsTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        # TODO(hayu): Remove mock sql after Snowflake 8.40.0 release
        query = "SELECT CURRENT_VERSION() AS CURRENT_VERSION"
        sql_result = [row.Row(CURRENT_VERSION="8.40.0 1234567890ab")]
        self.m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))

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
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
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
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
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
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_create_service_model_db_and_schema(self) -> None:
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
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
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
                service_database_name=sql_identifier.SqlIdentifier("DB"),
                service_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
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
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_create_service_default_db_and_schema(self) -> None:
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
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
                statement_params=self.m_statement_params,
            )
            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )
            mock_save.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
                service_database_name=sql_identifier.SqlIdentifier("TEMP"),
                service_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("TEMP"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                ingress_enabled=True,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
            )
            mock_upload_directory_to_stage.assert_called_once_with(
                self.c_session,
                local_path=self.m_ops._model_deployment_spec.workspace_path,
                stage_path=pathlib.PurePosixPath(
                    self.m_ops._stage_client.fully_qualified_object_name(
                        sql_identifier.SqlIdentifier("TEMP"),
                        sql_identifier.SqlIdentifier("test", case_sensitive=True),
                        sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                    )
                ),
                statement_params=self.m_statement_params,
            )
            mock_deploy_model.assert_called_once_with(
                stage_path='TEMP."test".SNOWPARK_TEMP_STAGE_ABCDEF0123',
                model_deployment_spec_file_rel_path=self.m_ops._model_deployment_spec.DEPLOY_SPEC_FILE_REL_PATH,
                statement_params=self.m_statement_params,
            )
            mock_get_service_status.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_get_model_build_service_name(self) -> None:
        query_id = "01b6fc10-0002-c121-0000-6ed10736311e"
        """
        Java code to generate the expected value:
        import java.math.BigInteger;
        import org.apache.commons.codec.digest.DigestUtils;
        String uuid = "01b6fc10-0002-c121-0000-6ed10736311e";
        String uuidString = uuid.replace("-", "");
        BigInteger bigInt = new BigInteger(uuidString, 16);
        String identifier = DigestUtils.md5Hex(bigInt.toString()).substring(0, 8);
        System.out.println(identifier);
        """
        identifier = "81edd120"
        expected = ("model_build_" + identifier).upper()
        self.assertEqual(self.m_ops._get_model_build_service_name(query_id), expected)


if __name__ == "__main__":
    absltest.main()
