import pathlib
import uuid
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml._internal import file_utils, platform_capabilities
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model._client.sql import service as service_sql
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Session, row
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


class ServiceOpsTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.m_statement_params = {"test": "1"}
        self.c_session = cast(Session, self.m_session)
        with platform_capabilities.PlatformCapabilities.mock_features():
            self.m_ops = service_ops.ServiceOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            )

    # TODO(hayu): Remove mock sql after Snowflake 8.40.0 release
    def _add_snowflake_version_check_mock_operations(
        self,
        m_session: mock_session.MockSession,
    ) -> mock_session.MockSession:
        query = "SELECT CURRENT_VERSION() AS CURRENT_VERSION"
        sql_result = [row.Row(CURRENT_VERSION="8.40.0 1234567890ab")]
        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        return m_session

    def test_create_service(self) -> None:
        self._add_snowflake_version_check_mock_operations(self.m_session)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status=service_sql.InstanceStatus.PENDING,
                container_status=service_sql.ContainerStatus.PENDING,
                message=None,
            )
        ]
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",) as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ), mock.patch.object(self.m_ops._model_deployment_spec, "save",) as mock_save, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_model_spec",
        ) as mock_add_model_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_service_spec",
        ) as mock_add_service_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_image_build_spec",
        ) as mock_add_image_build_spec, mock.patch.object(
            file_utils, "upload_directory_to_stage", return_value=None
        ) as mock_upload_directory_to_stage, mock.patch.object(
            self.m_ops._service_client,
            "deploy_model",
            return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
        ) as mock_deploy_model, mock.patch.object(
            self.m_ops._service_client,
            "get_service_container_statuses",
            return_value=m_statuses,
        ) as mock_get_service_container_statuses:
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
                block=True,
                statement_params=self.m_statement_params,
            )
            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )
            mock_add_model_spec.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
            )
            mock_add_service_spec.assert_called_once_with(
                service_database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                service_schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                ingress_enabled=True,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
            )
            mock_save.assert_called_once()
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
                model_deployment_spec_yaml_str=None,
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_create_service_model_db_and_schema(self) -> None:
        self._add_snowflake_version_check_mock_operations(self.m_session)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status=service_sql.InstanceStatus.PENDING,
                container_status=service_sql.ContainerStatus.PENDING,
                message=None,
            )
        ]
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",) as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ), mock.patch.object(self.m_ops._model_deployment_spec, "save",) as mock_save, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_model_spec",
        ) as mock_add_model_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_service_spec",
        ) as mock_add_service_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_image_build_spec",
        ) as mock_add_image_build_spec, mock.patch.object(
            file_utils, "upload_directory_to_stage", return_value=None
        ) as mock_upload_directory_to_stage, mock.patch.object(
            self.m_ops._service_client,
            "deploy_model",
            return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
        ) as mock_deploy_model, mock.patch.object(
            self.m_ops._service_client,
            "get_service_container_statuses",
            return_value=m_statuses,
        ) as mock_get_service_container_statuses:
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
                block=True,
                statement_params=self.m_statement_params,
            )
            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )
            mock_save.assert_called_once()
            mock_add_model_spec.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
            )
            mock_add_service_spec.assert_called_once_with(
                service_database_name=sql_identifier.SqlIdentifier("DB"),
                service_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                ingress_enabled=True,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                force_rebuild=True,
                image_repo_database_name=sql_identifier.SqlIdentifier("DB"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
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
                model_deployment_spec_yaml_str=None,
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_create_service_default_db_and_schema(self) -> None:
        self._add_snowflake_version_check_mock_operations(self.m_session)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status=service_sql.InstanceStatus.PENDING,
                container_status=service_sql.ContainerStatus.PENDING,
                message=None,
            )
        ]
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",) as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ), mock.patch.object(self.m_ops._model_deployment_spec, "save",) as mock_save, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_model_spec",
        ) as mock_add_model_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_service_spec",
        ) as mock_add_service_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_image_build_spec",
        ) as mock_add_image_build_spec, mock.patch.object(
            file_utils, "upload_directory_to_stage", return_value=None
        ) as mock_upload_directory_to_stage, mock.patch.object(
            self.m_ops._service_client,
            "deploy_model",
            return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
        ) as mock_deploy_model, mock.patch.object(
            self.m_ops._service_client,
            "get_service_container_statuses",
            return_value=m_statuses,
        ) as mock_get_service_container_statuses:
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
                block=True,
                statement_params=self.m_statement_params,
            )
            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )
            mock_add_model_spec.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
            )
            mock_add_service_spec.assert_called_once_with(
                service_database_name=sql_identifier.SqlIdentifier("TEMP"),
                service_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                ingress_enabled=True,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                image_repo_database_name=sql_identifier.SqlIdentifier("TEMP"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
            )
            mock_save.assert_called_once()
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
                model_deployment_spec_yaml_str=None,
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_create_service_async_job(self) -> None:
        self._add_snowflake_version_check_mock_operations(self.m_session)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status=service_sql.InstanceStatus.PENDING,
                container_status=service_sql.ContainerStatus.PENDING,
                message=None,
            )
        ]
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",), mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
        ), mock.patch.object(self.m_ops._model_deployment_spec, "save",), mock.patch.object(
            file_utils, "upload_directory_to_stage", return_value=None
        ), mock.patch.object(
            self.m_ops._service_client,
            "deploy_model",
            return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
        ), mock.patch.object(
            self.m_ops._service_client,
            "get_service_container_statuses",
            return_value=m_statuses,
        ):
            res = self.m_ops.create_service(
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
                block=False,
                statement_params=self.m_statement_params,
            )
            self.assertIsInstance(res, snowpark.AsyncJob)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "database_name": (sql_identifier.SqlIdentifier("DB"), sql_identifier.SqlIdentifier("DB")),
            "schema_name": (sql_identifier.SqlIdentifier("SCHEMA"), sql_identifier.SqlIdentifier("SCHEMA")),
            "job_database_name": (sql_identifier.SqlIdentifier("JOB_DB"), sql_identifier.SqlIdentifier("JOB_DB")),
            "job_schema_name": (sql_identifier.SqlIdentifier("JOB_SCHEMA"), sql_identifier.SqlIdentifier("JOB_SCHEMA")),
            "image_repo_database_name": (
                sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
                sql_identifier.SqlIdentifier("IMAGE_REPO_DB"),
            ),
            "image_repo_schema_name": (
                sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
                sql_identifier.SqlIdentifier("IMAGE_REPO_SCHEMA"),
            ),
            "output_table_database_name": (
                sql_identifier.SqlIdentifier("OUTPUT_TABLE_DB"),
                sql_identifier.SqlIdentifier("OUTPUT_TABLE_DB"),
            ),
            "output_table_schema_name": (
                sql_identifier.SqlIdentifier("OUTPUT_TABLE_SCHEMA"),
                sql_identifier.SqlIdentifier("OUTPUT_TABLE_SCHEMA"),
            ),
        },
        {
            "database_name": (sql_identifier.SqlIdentifier("DB"), sql_identifier.SqlIdentifier("DB")),
            "schema_name": (sql_identifier.SqlIdentifier("SCHEMA"), sql_identifier.SqlIdentifier("SCHEMA")),
            "job_database_name": (None, sql_identifier.SqlIdentifier("DB")),
            "job_schema_name": (None, sql_identifier.SqlIdentifier("SCHEMA")),
            "image_repo_database_name": (None, sql_identifier.SqlIdentifier("DB")),
            "image_repo_schema_name": (None, sql_identifier.SqlIdentifier("SCHEMA")),
            "output_table_database_name": (None, sql_identifier.SqlIdentifier("DB")),
            "output_table_schema_name": (None, sql_identifier.SqlIdentifier("SCHEMA")),
        },
        {
            "database_name": (None, sql_identifier.SqlIdentifier("TEMP")),
            "schema_name": (None, sql_identifier.SqlIdentifier("test", case_sensitive=True)),
            "job_database_name": (None, sql_identifier.SqlIdentifier("TEMP")),
            "job_schema_name": (None, sql_identifier.SqlIdentifier("test", case_sensitive=True)),
            "image_repo_database_name": (None, sql_identifier.SqlIdentifier("TEMP")),
            "image_repo_schema_name": (None, sql_identifier.SqlIdentifier("test", case_sensitive=True)),
            "output_table_database_name": (None, sql_identifier.SqlIdentifier("TEMP")),
            "output_table_schema_name": (None, sql_identifier.SqlIdentifier("test", case_sensitive=True)),
        },
    )
    def test_invoke_job_method(
        self,
        database_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        schema_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        job_database_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        job_schema_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        image_repo_database_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        image_repo_schema_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        output_table_database_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
        output_table_schema_name: tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier],
    ) -> None:
        pd_df = pd.DataFrame([["1.0"]], columns=["input"], dtype=np.float32)
        m_sig = _DUMMY_SIG["predict"]
        mock_writer = mock.MagicMock()
        mock_drop_table = mock.MagicMock()
        m_input_df = mock_data_frame.MockDataFrame()
        m_input_df.add_operation(operation="select", check_args=False, check_kwargs=False)
        m_input_df.__setattr__("columns", ["COL1", "COL2"])
        m_input_df.__setattr__("write", mock_writer)
        m_input_df.__setattr__("drop_table", mock_drop_table)
        m_output_df = mock_data_frame.MockDataFrame()
        m_output_df.add_mock_sort("_ID", ascending=True).add_mock_drop(
            snowpark_handler._KEEP_ORDER_COL_NAME
        ).add_mock_drop("COL1", "COL2")
        return_mapping = {
            f"{job_database_name[1]}.{job_schema_name[1]}.SNOWPARK_TEMP_ABCDEF0123": m_input_df,
            f"{output_table_database_name[1]}.{output_table_schema_name[1]}.OUTPUT_TABLE": m_output_df,
        }
        self.m_session.table = mock.MagicMock(
            name="table", side_effect=lambda arg, *args, **kwargs: return_mapping.get(arg)
        )
        stage_path = self.m_ops._stage_client.fully_qualified_object_name(
            database_name[1],
            schema_name[1],
            sql_identifier.SqlIdentifier("SNOWPARK_TEMP_ABCDEF0123"),
        )
        with mock.patch.object(self.m_ops._stage_client, "create_tmp_stage",) as mock_create_stage, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_ABCDEF0123"
        ), mock.patch.object(self.m_ops._model_deployment_spec, "save",) as mock_save, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_model_spec",
        ) as mock_add_model_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_job_spec",
        ) as mock_add_job_spec, mock.patch.object(
            self.m_ops._model_deployment_spec,
            "add_image_build_spec",
        ) as mock_add_image_build_spec, mock.patch.object(
            file_utils, "upload_directory_to_stage", return_value=None
        ) as mock_upload_directory_to_stage, mock.patch.object(
            self.m_ops._service_client,
            "deploy_model",
            return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
        ) as mock_deploy_model, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_from_df", return_value=m_input_df
        ) as mock_convert_from_df, mock.patch.object(
            snowpark_handler.SnowparkDataFrameHandler, "convert_to_df", return_value=pd_df
        ) as mock_convert_to_df:
            self.m_ops.invoke_job_method(
                target_method="predict",
                signature=m_sig,
                X=pd_df,
                database_name=database_name[0],
                schema_name=schema_name[0],
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                job_database_name=job_database_name[0],
                job_schema_name=job_schema_name[0],
                job_name=sql_identifier.SqlIdentifier("JOB"),
                compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                warehouse_name=sql_identifier.SqlIdentifier("WAREHOUSE"),
                image_repo_database_name=image_repo_database_name[0],
                image_repo_schema_name=image_repo_schema_name[0],
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                output_table_database_name=output_table_database_name[0],
                output_table_schema_name=output_table_schema_name[0],
                output_table_name=sql_identifier.SqlIdentifier("OUTPUT_TABLE"),
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EAI")],
                statement_params=self.m_statement_params,
            )
            mock_create_stage.assert_called_once_with(
                database_name=database_name[1],
                schema_name=schema_name[1],
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )
            mock_save.assert_called_once()
            mock_add_model_spec.assert_called_once_with(
                database_name=database_name[1],
                schema_name=schema_name[1],
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
            )
            mock_add_job_spec.assert_called_once_with(
                job_database_name=job_database_name[1],
                job_schema_name=job_schema_name[1],
                job_name=sql_identifier.SqlIdentifier("JOB"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                warehouse=sql_identifier.SqlIdentifier("WAREHOUSE"),
                target_method="predict",
                input_table_database_name=job_database_name[1],
                input_table_schema_name=job_schema_name[1],
                input_table_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_ABCDEF0123"),
                output_table_database_name=output_table_database_name[1],
                output_table_schema_name=output_table_schema_name[1],
                output_table_name=sql_identifier.SqlIdentifier("OUTPUT_TABLE"),
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_repo_database_name=image_repo_database_name[1],
                image_repo_schema_name=image_repo_schema_name[1],
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                image_repo_name=sql_identifier.SqlIdentifier("IMAGE_REPO"),
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EAI")],
            )
            mock_upload_directory_to_stage.assert_called_once_with(
                self.c_session,
                local_path=self.m_ops._model_deployment_spec.workspace_path,
                stage_path=pathlib.PurePosixPath(stage_path),
                statement_params=self.m_statement_params,
            )
            mock_deploy_model.assert_called_once_with(
                stage_path=stage_path,
                model_deployment_spec_file_rel_path=self.m_ops._model_deployment_spec.DEPLOY_SPEC_FILE_REL_PATH,
                model_deployment_spec_yaml_str=None,
                statement_params=self.m_statement_params,
            )
            mock_convert_from_df.assert_called_once_with(
                self.c_session, mock.ANY, keep_order=True, features=m_sig.inputs
            )
            mock_convert_to_df.assert_called_once_with(m_output_df, features=m_sig.outputs)

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
