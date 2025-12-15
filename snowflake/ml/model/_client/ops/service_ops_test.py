import pathlib
import uuid
from typing import Any, Optional, cast
from unittest import mock

from absl.testing import absltest, parameterized
from packaging import version

from snowflake import snowpark
from snowflake.ml import version as snowml_version
from snowflake.ml._internal import file_utils, platform_capabilities
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.jobs import job
from snowflake.ml.model import inference_engine, model_signature
from snowflake.ml.model._client.model import batch_inference_specs
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.model._client.sql import service as service_sql
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.test_utils.mock_progress import create_mock_progress_status
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

    _default_hf_args = {
        "hf_model_name": None,
        "hf_task": None,
        "hf_token": None,
        "hf_tokenizer": None,
        "hf_revision": None,
        "hf_trust_remote_code": False,
        "pip_requirements": None,
        "conda_dependencies": None,
        "comment": None,
        "warehouse": None,
    }

    def _get_hugging_face_model_save_args(
        self,
        huggingface_args: Optional[dict[str, Any]] = None,
        *,
        use_inlined_deployment_spec: bool = False,
    ) -> dict[str, Any]:
        if huggingface_args is None:
            return self._default_hf_args
        else:
            # union huggingface_args with _default_hf_args
            if use_inlined_deployment_spec and "hf_token" in huggingface_args:
                return {
                    **self._default_hf_args,
                    **huggingface_args,
                    "hf_token": service_sql.QMARK_RESERVED_TOKEN,
                }
            else:
                return {**self._default_hf_args, **huggingface_args}

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

    def _create_mock_async_job(self) -> mock.MagicMock:
        """Create a mock async job that prevents infinite loops in log streaming."""
        mock_async_job = mock.MagicMock(spec=snowpark.AsyncJob)
        mock_async_job.is_done.return_value = True  # Prevents infinite loop in _stream_service_logs
        return mock_async_job

    @parameterized.parameters(  # type: ignore[misc]
        {"huggingface_args": {}},
        {
            "huggingface_args": {
                "hf_model_name": "gpt2",
                "hf_task": "text-generation",
                "hf_token": "token",
            }
        },
        {
            "huggingface_args": {
                "hf_model_name": "gpt2",
                "hf_task": "text-generation",
            }
        },
    )
    def test_create_service_basic(self, huggingface_args: dict[str, Any]) -> None:
        self._add_snowflake_version_check_mock_operations(self.m_session)
        current_version = version.Version(snowml_version.VERSION)
        # force enable inlined deployment spec
        with platform_capabilities.PlatformCapabilities.mock_features(
            features={platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: current_version}
        ):
            self.m_ops = service_ops.ServiceOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            )
            self.assertTrue(self.m_ops._use_inlined_deployment_spec)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status=service_sql.InstanceStatus.PENDING,
                container_status=service_sql.ContainerStatus.PENDING,
                message=None,
            )
        ]
        with (
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
            ) as mock_save,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ) as mock_add_model_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ) as mock_add_service_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_hf_logger_spec",
            ) as mock_add_hf_logger,
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ) as mock_deploy_model,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_logs",
                return_value="",  # Return empty logs to prevent SQL calls
            ),
            mock.patch.object(
                self.m_ops,
                "_wait_for_service_status",
                return_value=None,
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_inference_engine_spec",
            ) as mock_add_inference_engine_spec,
        ):
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
                image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
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
                hf_model_args=service_ops.HFModelArgs(**huggingface_args) if huggingface_args else None,
                progress_status=create_mock_progress_status(),
                inference_engine_args=None,
                autocapture=None,
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
                autocapture=None,
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                fully_qualified_image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
            )
            if huggingface_args:
                mock_add_hf_logger.assert_called_once_with(
                    **self._get_hugging_face_model_save_args(
                        use_inlined_deployment_spec=True,
                        huggingface_args=huggingface_args,
                    )
                )

            mock_save.assert_called_once()
            mock_deploy_model.assert_called_once_with(
                stage_path=None,
                model_deployment_spec_file_rel_path=None,
                model_deployment_spec_yaml_str=self.m_ops._model_deployment_spec.save(),
                statement_params=self.m_statement_params,
                query_params=["token"] if "hf_token" in huggingface_args else [],
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

            # by default, no inference engine spec is added
            mock_add_inference_engine_spec.assert_not_called()

    @parameterized.parameters(  # type: ignore[misc]
        {"huggingface_args": {}},
        {
            "huggingface_args": {
                "hf_model_name": "gpt2",
                "hf_task": "text-generation",
                "hf_token": "token",
            }
        },
    )
    def test_create_service_model_db_and_schema(self, huggingface_args: dict[str, Any]) -> None:
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
        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ) as mock_create_stage,
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
            ) as mock_save,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ) as mock_add_model_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ) as mock_add_service_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_hf_logger_spec",
            ) as mock_add_hf_logger,
            mock.patch.object(
                file_utils, "upload_directory_to_stage", return_value=None
            ) as mock_upload_directory_to_stage,
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ) as mock_deploy_model,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_logs",
                return_value="",  # Return empty logs to prevent SQL calls
            ),
            mock.patch.object(
                self.m_ops,
                "_wait_for_service_status",
                return_value=None,
            ),
        ):
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
                image_repo_name="IMAGE_REPO",
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
                hf_model_args=service_ops.HFModelArgs(**huggingface_args) if huggingface_args else None,
                progress_status=create_mock_progress_status(),
                inference_engine_args=None,
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
                autocapture=None,
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
            )
            if huggingface_args:
                mock_add_hf_logger.assert_called_once_with(**self._get_hugging_face_model_save_args(huggingface_args))
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
                query_params=[],
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"huggingface_args": {}},
        {
            "huggingface_args": {
                "hf_model_name": "gpt2",
                "hf_task": "text-generation",
                "hf_token": "token",
            }
        },
    )
    def test_create_service_default_db_and_schema(self, huggingface_args: dict[str, Any]) -> None:
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
        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ) as mock_create_stage,
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
            ) as mock_save,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ) as mock_add_model_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ) as mock_add_service_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_hf_logger_spec",
            ) as mock_add_hf_logger,
            mock.patch.object(
                file_utils, "upload_directory_to_stage", return_value=None
            ) as mock_upload_directory_to_stage,
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ) as mock_deploy_model,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_logs",
                return_value="",  # Return empty logs to prevent SQL calls
            ),
            mock.patch.object(
                self.m_ops,
                "_wait_for_service_status",
                return_value=None,
            ),
        ):
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
                image_repo_name="IMAGE_REPO",
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
                hf_model_args=service_ops.HFModelArgs(**huggingface_args) if huggingface_args else None,
                progress_status=create_mock_progress_status(),
                inference_engine_args=None,
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
                autocapture=None,
            )
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("IMAGE_BUILD_COMPUTE_POOL"),
                fully_qualified_image_repo_name='TEMP."test".IMAGE_REPO',
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
            )
            if huggingface_args:
                mock_add_hf_logger.assert_called_once_with(**self._get_hugging_face_model_save_args(huggingface_args))
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
                query_params=[],
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"huggingface_args": {}},
        {
            "huggingface_args": {
                "hf_model_name": "gpt2",
                "hf_task": "text-generation",
                "hf_token": "token",
            }
        },
    )
    def test_create_service_async_job(self, huggingface_args: dict[str, Any]) -> None:
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
        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ),
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
            ),
            mock.patch.object(file_utils, "upload_directory_to_stage", return_value=None),
        ):
            mock_async_job = mock.MagicMock(spec=snowpark.AsyncJob)
            mock_async_job.is_done.return_value = True

            with (
                mock.patch.object(
                    self.m_ops._service_client,
                    "deploy_model",
                    return_value=(str(uuid.uuid4()), mock_async_job),
                ),
                mock.patch.object(
                    self.m_ops._service_client,
                    "get_service_container_statuses",
                    return_value=m_statuses,
                ),
                mock.patch.object(
                    self.m_ops._service_client,
                    "get_service_logs",
                    return_value="",  # Return empty logs
                ),
                mock.patch.object(
                    self.m_ops,
                    "_wait_for_service_status",
                    return_value=None,
                ),
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
                    image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
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
                    hf_model_args=service_ops.HFModelArgs(**huggingface_args) if huggingface_args else None,
                    progress_status=create_mock_progress_status(),
                    inference_engine_args=None,
                )
                self.assertIsInstance(res, snowpark.AsyncJob)

    def test_create_service_uses_operation_id_for_logging(self) -> None:
        """Test that create_service generates operation_id and passes it to service loggers."""
        self._add_snowflake_version_check_mock_operations(self.m_session)

        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.DONE,
                instance_id=0,
                instance_status=service_sql.InstanceStatus.PENDING,
                container_status=service_sql.ContainerStatus.PENDING,
                message=None,
            )
        ]

        with (
            mock.patch.object(self.m_ops._stage_client, "create_tmp_stage"),
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(self.m_ops._model_deployment_spec, "save"),
            mock.patch.object(self.m_ops._model_deployment_spec, "add_model_spec"),
            mock.patch.object(self.m_ops._model_deployment_spec, "add_service_spec"),
            mock.patch.object(self.m_ops._model_deployment_spec, "add_image_build_spec"),
            mock.patch.object(file_utils, "upload_directory_to_stage", return_value=None),
        ):
            mock_async_job = mock.MagicMock(spec=snowpark.AsyncJob)
            mock_async_job.is_done.return_value = True

            with (
                mock.patch.object(
                    self.m_ops._service_client,
                    "deploy_model",
                    return_value=(str(uuid.uuid4()), mock_async_job),
                ),
                mock.patch.object(
                    self.m_ops._service_client,
                    "get_service_container_statuses",
                    return_value=m_statuses,
                ),
                mock.patch.object(
                    self.m_ops._service_client,
                    "get_service_logs",
                    return_value="",  # Return empty logs
                ),
                mock.patch.object(
                    self.m_ops,
                    "_wait_for_service_status",
                    return_value=None,
                ),
            ):
                # This test just verifies the service can be called without timeout
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
                    image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
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
                    hf_model_args=None,
                    progress_status=create_mock_progress_status(),
                    inference_engine_args=None,
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
        self.assertEqual(
            self.m_ops._get_service_id_from_deployment_step(
                query_id,
                service_ops.DeploymentStep.MODEL_BUILD,
            ),
            expected,
        )

    def test_create_service_custom_inference_engine(self) -> None:
        """Test create_service with custom inference engine parameters."""
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

        # Define test inference engine kwargs
        test_inference_engine_args = [
            "--tensor-parallel-size=2",
            "--max_tokens=1000",
            "--temperature=0.8",
        ]

        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ) as mock_create_stage,
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
            ) as mock_save,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ) as mock_add_model_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ) as mock_add_service_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_inference_engine_spec",
            ) as mock_add_inference_engine_spec,
            mock.patch.object(
                file_utils, "upload_directory_to_stage", return_value=None
            ) as mock_upload_directory_to_stage,
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ) as mock_deploy_model,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_logs",
                return_value="",
            ),
            mock.patch.object(
                self.m_ops,
                "_wait_for_service_status",
                return_value=None,
            ),
        ):
            # Call create_service with inference engine parameters
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
                image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                ingress_enabled=True,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="2",  # This should match tensor-parallel-size
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
                block=True,
                statement_params=self.m_statement_params,
                inference_engine_args=service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.VLLM,
                    inference_engine_args_override=test_inference_engine_args,
                ),
                progress_status=create_mock_progress_status(),
            )

            # Verify all the standard method calls
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
                gpu="2",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
            )

            # This is the key assertion - verify add_inference_engine_spec was called
            mock_add_inference_engine_spec.assert_called_once_with(
                inference_engine=inference_engine.InferenceEngine.VLLM, inference_engine_args=test_inference_engine_args
            )

            mock_add_image_build_spec.assert_not_called()
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
                query_params=[],
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    def test_create_service_with_inference_engine_and_no_image_build(self) -> None:
        """Test create_service with custom inference engine parameters and no image build."""
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

        # Define test inference engine kwargs
        test_inference_engine_args = [
            "--tensor-parallel-size=2",
            "--max_tokens=1000",
            "--temperature=0.8",
        ]

        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ) as mock_create_stage,
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
            ) as mock_save,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ) as mock_add_model_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ) as mock_add_service_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_inference_engine_spec",
            ) as mock_add_inference_engine_spec,
            mock.patch.object(
                file_utils, "upload_directory_to_stage", return_value=None
            ) as mock_upload_directory_to_stage,
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), mock.MagicMock(spec=snowpark.AsyncJob)),
            ) as mock_deploy_model,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ) as mock_get_service_container_statuses,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_logs",
                return_value="",  # Return empty logs to prevent SQL calls
            ),
            mock.patch.object(
                self.m_ops,
                "_wait_for_service_status",
                return_value=None,
            ),
        ):
            # Call create_service with inference engine parameters
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
                image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                ingress_enabled=True,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="2",  # This should match tensor-parallel-size
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=True,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
                block=True,
                statement_params=self.m_statement_params,
                inference_engine_args=service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.VLLM,
                    inference_engine_args_override=test_inference_engine_args,
                ),
                progress_status=create_mock_progress_status(),
            )

            # Verify all the standard method calls
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
                gpu="2",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
            )

            # key assertions -- image build is not called and inference engine model is called
            # when inference engine is specified
            mock_add_image_build_spec.assert_not_called()
            mock_add_inference_engine_spec.assert_called_once_with(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args=test_inference_engine_args,
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
                query_params=[],
                statement_params=self.m_statement_params,
            )
            mock_get_service_container_statuses.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                include_message=False,
                statement_params=self.m_statement_params,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {
            "job_name": "BATCH_JOB",
            "expected_job_db": "TEMP",
            "expected_job_schema": "test",
            "expected_result_id": 'TEMP."test".BATCH_JOB',
        },
        {
            "job_name": "JOB_DB.JOB_SCHEMA.BATCH_JOB",
            "expected_job_db": "JOB_DB",
            "expected_job_schema": "JOB_SCHEMA",
            "expected_result_id": "JOB_DB.JOB_SCHEMA.BATCH_JOB",
        },
    )
    def test_invoke_batch_job_method(
        self, job_name: str, expected_job_db: str, expected_job_schema: str, expected_result_id: str
    ) -> None:
        """Test invoke_batch_job_method with different job name formats."""
        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ) as mock_create_stage,
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._stage_client,
                "fully_qualified_object_name",
                return_value="TEMP.test.SNOWPARK_TEMP_STAGE_ABCDEF0123",
            ),
            mock.patch.object(
                file_utils, "upload_directory_to_stage", return_value=None
            ) as mock_upload_directory_to_stage,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "clear",
            ) as mock_clear,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ) as mock_add_model_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_job_spec",
            ) as mock_add_job_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
                return_value=pathlib.Path("/mock/spec/path"),
            ) as mock_save,
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ) as mock_deploy_model,
        ):
            result = self.m_ops.invoke_batch_job_method(
                function_name="predict",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
                job_name=job_name,
                compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                warehouse=sql_identifier.SqlIdentifier("WAREHOUSE"),
                image_repo_name="IMAGE_REPO",
                input_stage_location="@input_stage/",
                input_file_pattern="*.parquet",
                output_stage_location="@output_stage/",
                completion_filename="completion.txt",
                force_rebuild=True,
                statement_params=self.m_statement_params,
                num_workers=2,
                max_batch_rows=1000,
                cpu_requests="1",
                memory_requests="4GiB",
                gpu_requests=None,
                replicas=1,
            )

            # Verify all method calls
            mock_clear.assert_called_once()

            mock_add_model_spec.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
            )

            # Handle case sensitivity for expected schema
            expected_schema_identifier = (
                sql_identifier.SqlIdentifier("test", case_sensitive=True)
                if expected_job_schema == "test"
                else sql_identifier.SqlIdentifier(expected_job_schema)
            )

            mock_add_job_spec.assert_called_once_with(
                job_database_name=sql_identifier.SqlIdentifier(expected_job_db),
                job_schema_name=expected_schema_identifier,
                job_name="BATCH_JOB",
                inference_compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                num_workers=2,
                max_batch_rows=1000,
                input_stage_location="@input_stage/",
                input_file_pattern="*.parquet",
                output_stage_location="@output_stage/",
                completion_filename="completion.txt",
                function_name="predict",
                warehouse=sql_identifier.SqlIdentifier("WAREHOUSE"),
                cpu="1",
                memory="4GiB",
                gpu=None,
                replicas=1,
            )

            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                fully_qualified_image_repo_name='TEMP."test".IMAGE_REPO',
                force_rebuild=True,
            )

            mock_save.assert_called_once()

            mock_create_stage.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                statement_params=self.m_statement_params,
            )

            mock_upload_directory_to_stage.assert_called_once()

            mock_deploy_model.assert_called_once_with(
                stage_path="TEMP.test.SNOWPARK_TEMP_STAGE_ABCDEF0123",
                model_deployment_spec_file_rel_path=model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH,
                model_deployment_spec_yaml_str=None,
                statement_params=self.m_statement_params,
            )

            # Verify returned MLJob
            self.assertIsInstance(result, job.MLJob)
            self.assertEqual(result.id, expected_result_id)

    def test_invoke_batch_job_method_with_workspace(self) -> None:
        """Test invoke_batch_job_method when using workspace (temporary directory)."""
        # Create a new ServiceOperator without the platform capability mock
        # so it uses the workspace path
        with platform_capabilities.PlatformCapabilities.mock_features({"inlined_deployment_spec": False}):
            m_ops_with_workspace = service_ops.ServiceOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            )
            with (
                mock.patch.object(
                    m_ops_with_workspace._stage_client,
                    "create_tmp_stage",
                ) as mock_create_stage,
                mock.patch.object(
                    snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
                ),
                mock.patch.object(
                    m_ops_with_workspace._stage_client,
                    "fully_qualified_object_name",
                    return_value="TEMP.test.SNOWPARK_TEMP_STAGE_ABCDEF0123",
                ),
                mock.patch.object(
                    m_ops_with_workspace._model_deployment_spec,
                    "clear",
                ),
                mock.patch.object(
                    m_ops_with_workspace._model_deployment_spec,
                    "add_model_spec",
                ),
                mock.patch.object(
                    m_ops_with_workspace._model_deployment_spec,
                    "add_job_spec",
                ),
                mock.patch.object(
                    m_ops_with_workspace._model_deployment_spec,
                    "add_image_build_spec",
                ),
                mock.patch.object(
                    m_ops_with_workspace._model_deployment_spec,
                    "save",
                    return_value=pathlib.Path("/mock/spec/path"),
                ),
                mock.patch.object(
                    file_utils, "upload_directory_to_stage", return_value=None
                ) as mock_upload_directory_to_stage,
                mock.patch.object(
                    m_ops_with_workspace._service_client,
                    "deploy_model",
                    return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
                ) as mock_deploy_model,
            ):
                result = m_ops_with_workspace.invoke_batch_job_method(
                    function_name="predict",
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("VERSION"),
                    job_name="BATCH_JOB",
                    compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                    warehouse=sql_identifier.SqlIdentifier("WAREHOUSE"),
                    image_repo_name="IMAGE_REPO",
                    input_stage_location="@input_stage/",
                    input_file_pattern="*.parquet",
                    output_stage_location="@output_stage/",
                    completion_filename="completion.txt",
                    force_rebuild=True,
                    statement_params=self.m_statement_params,
                    num_workers=2,
                    max_batch_rows=1000,
                    cpu_requests="1",
                    memory_requests="4GiB",
                    gpu_requests=None,
                    replicas=1,
                )

                # Verify stage operations
                mock_create_stage.assert_called_once_with(
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                    stage_name=sql_identifier.SqlIdentifier("SNOWPARK_TEMP_STAGE_ABCDEF0123"),
                    statement_params=self.m_statement_params,
                )

                mock_upload_directory_to_stage.assert_called_once()

                # Verify deploy_model called with stage path
                mock_deploy_model.assert_called_once_with(
                    stage_path="TEMP.test.SNOWPARK_TEMP_STAGE_ABCDEF0123",
                    model_deployment_spec_file_rel_path=(
                        model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH
                    ),
                    model_deployment_spec_yaml_str=None,
                    statement_params=self.m_statement_params,
                )

                # Verify returned MLJob
                self.assertIsInstance(result, job.MLJob)

    def test_invoke_batch_job_method_minimal_params(self) -> None:
        """Test invoke_batch_job_method with minimal required parameters only."""
        with (
            mock.patch.object(
                self.m_ops._stage_client,
                "create_tmp_stage",
            ),
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(
                self.m_ops._stage_client,
                "fully_qualified_object_name",
                return_value="TEMP.test.SNOWPARK_TEMP_STAGE_ABCDEF0123",
            ),
            mock.patch.object(file_utils, "upload_directory_to_stage", return_value=None),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "clear",
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_job_spec",
            ) as mock_add_job_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "save",
                return_value=pathlib.Path("/mock/spec/path"),
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ),
        ):
            result = self.m_ops.invoke_batch_job_method(
                function_name="predict",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
                job_name="BATCH_JOB",
                compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                warehouse=sql_identifier.SqlIdentifier("WAREHOUSE"),
                image_repo_name=None,
                input_stage_location="@input_stage/",
                input_file_pattern="*.parquet",
                output_stage_location="@output_stage/",
                completion_filename="completion.txt",
                force_rebuild=False,
                num_workers=None,
                max_batch_rows=None,
                cpu_requests=None,
                memory_requests=None,
                gpu_requests=None,
                replicas=None,
                statement_params=self.m_statement_params,
            )

            # Verify job spec called with None for optional parameters
            mock_add_job_spec.assert_called_once_with(
                job_database_name=sql_identifier.SqlIdentifier("TEMP"),
                job_schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                job_name="BATCH_JOB",
                inference_compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                num_workers=None,
                max_batch_rows=None,
                input_stage_location="@input_stage/",
                input_file_pattern="*.parquet",
                output_stage_location="@output_stage/",
                completion_filename="completion.txt",
                function_name="predict",
                warehouse=sql_identifier.SqlIdentifier("WAREHOUSE"),
                cpu=None,
                memory=None,
                gpu=None,
                replicas=None,
            )

            # Verify image build spec called with None for image repo
            mock_add_image_build_spec.assert_called_once_with(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("COMPUTE_POOL"),
                fully_qualified_image_repo_name=None,
                force_rebuild=False,
            )

            # Verify returned MLJob
            self.assertIsInstance(result, job.MLJob)

    def test_enforce_save_mode_error_with_empty_stage(self) -> None:
        """Test _enforce_save_mode with ERROR mode and empty stage location."""
        # Mock stage_client.list_stage to return no files
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.return_value = []
        self.m_ops._stage_client = mock_stage_client

        self.m_ops._enforce_save_mode(batch_inference_specs.SaveMode.ERROR, "@test_stage/")

        mock_stage_client.list_stage.assert_called_once_with("@test_stage/")

    def test_enforce_save_mode_error_with_files_raises_exception(self) -> None:
        """Test _enforce_save_mode with ERROR mode and files in stage location."""
        mock_file_row = mock.MagicMock()

        # Mock stage_client.list_stage to return files
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.return_value = [mock_file_row, mock_file_row]
        self.m_ops._stage_client = mock_stage_client

        with self.assertRaises(FileExistsError) as cm:
            self.m_ops._enforce_save_mode(batch_inference_specs.SaveMode.ERROR, "@test_stage/")

        self.assertIn("Output stage location '@test_stage/' is not empty", str(cm.exception))
        self.assertIn("Found 2 existing files", str(cm.exception))
        self.assertIn("When using ERROR mode", str(cm.exception))

    def test_enforce_save_mode_error_with_stage_exception(self) -> None:
        """Test _enforce_save_mode with ERROR mode when stage list operation fails."""
        # Mock stage_client.list_stage to raise exception
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.side_effect = Exception("Stage not found")
        self.m_ops._stage_client = mock_stage_client

        with self.assertRaises(Exception) as cm:
            self.m_ops._enforce_save_mode(batch_inference_specs.SaveMode.ERROR, "@test_stage/")

        self.assertIn("Stage not found", str(cm.exception))

    def test_enforce_save_mode_overwrite_with_empty_stage(self) -> None:
        """Test _enforce_save_mode with OVERWRITE mode and empty stage location."""
        # Mock stage_client.list_stage to return no files
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.return_value = []
        self.m_ops._stage_client = mock_stage_client

        with mock.patch("warnings.warn") as mock_warn:
            self.m_ops._enforce_save_mode(batch_inference_specs.SaveMode.OVERWRITE, "@test_stage/")
            mock_warn.assert_not_called()

        mock_stage_client.list_stage.assert_called_once_with("@test_stage/")

    def test_enforce_save_mode_overwrite_with_files_shows_warning(self) -> None:
        """Test _enforce_save_mode with OVERWRITE mode and files in stage location."""
        mock_file_row = mock.MagicMock()

        # Mock stage_client.list_stage to return files
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.return_value = [mock_file_row, mock_file_row, mock_file_row]
        self.m_ops._stage_client = mock_stage_client

        # Mock session for REMOVE command
        mock_session = mock.MagicMock()
        self.m_ops._session = mock_session

        with mock.patch("warnings.warn") as mock_warn:
            self.m_ops._enforce_save_mode(batch_inference_specs.SaveMode.OVERWRITE, "@test_stage/")

            mock_warn.assert_called_once()
            warning_message = mock_warn.call_args[0][0]
            self.assertIn("Output stage location '@test_stage/' is not empty", warning_message)
            self.assertIn("Found 3 existing files", warning_message)
            self.assertIn("OVERWRITE mode will remove all existing files", warning_message)

        # Verify stage list was called and session REMOVE was called
        mock_stage_client.list_stage.assert_called_once_with("@test_stage/")
        mock_session.sql.assert_called_once_with("REMOVE @test_stage/")

    def test_enforce_save_mode_overwrite_remove_fails(self) -> None:
        """Test _enforce_save_mode with OVERWRITE mode when REMOVE command fails."""
        mock_session = mock.MagicMock()
        mock_file_row = mock.MagicMock()

        # Mock stage_client.list_stage to return files
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.return_value = [mock_file_row]
        self.m_ops._stage_client = mock_stage_client

        # Mock session.sql for REMOVE to fail
        mock_session.sql.return_value.collect.side_effect = Exception("Permission denied")
        self.m_ops._session = mock_session

        with self.assertRaises(RuntimeError) as cm:
            self.m_ops._enforce_save_mode(batch_inference_specs.SaveMode.OVERWRITE, "@test_stage/")

        self.assertIn("OVERWRITE was specified", str(cm.exception))
        self.assertIn("failed to remove existing files", str(cm.exception))
        self.assertIn("Permission denied", str(cm.exception))

    def test_enforce_save_mode_invalid_mode(self) -> None:
        """Test _enforce_save_mode with invalid SaveMode."""
        # Mock stage_client.list_stage to return no files
        mock_stage_client = mock.MagicMock()
        mock_stage_client.list_stage.return_value = []
        self.m_ops._stage_client = mock_stage_client

        invalid_mode = "INVALID_MODE"

        with self.assertRaises(ValueError) as cm:
            self.m_ops._enforce_save_mode(invalid_mode, "@test_stage/")  # type: ignore[arg-type]

        self.assertIn("Invalid SaveMode: INVALID_MODE", str(cm.exception))
        self.assertIn("Must be one of", str(cm.exception))

        mock_stage_client.list_stage.assert_called_once_with("@test_stage/")


if __name__ == "__main__":
    absltest.main()
