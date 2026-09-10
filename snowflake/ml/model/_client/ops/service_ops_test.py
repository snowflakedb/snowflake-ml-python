import pathlib
import threading
import time
import uuid
from typing import Any, cast
from unittest import mock

from absl.testing import absltest, parameterized
from packaging import version

from snowflake import snowpark
from snowflake.ml import version as snowml_version
from snowflake.ml._internal import file_utils, platform_capabilities
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import inference_engine, model_signature
from snowflake.ml.model._client.model import batch_inference_job_specs
from snowflake.ml.model._client.ops import deployment_step, service_ops
from snowflake.ml.model._client.sql import service as service_sql
from snowflake.ml.model._signatures import core
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.test_utils.mock_progress import create_mock_progress_status
from snowflake.snowpark import Session, dataframe, row
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

_DUMMY_SIG_WITH_PARAMS = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
    ],
    outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    params=[
        core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
    ],
)


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
        huggingface_args: dict[str, Any] | None = None,
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
                instance_status="PENDING",
                container_status="PENDING",
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
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],  # service does not exist; create_service takes the deploy path
            ) as mock_show_services,
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
                min_instances=0,
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
                autocapture=False,
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
                min_instances=0,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=False,
                feature_sources_per_function=None,
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
            mock_show_services.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                starts_with="MYSERVICE",
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
                instance_status="PENDING",
                container_status="PENDING",
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
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],  # service does not exist; create_service takes the deploy path
            ) as mock_show_services,
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
                min_instances=1,
                max_instances=3,
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
                min_instances=1,
                max_instances=3,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
                feature_sources_per_function=None,
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
            mock_show_services.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                starts_with="MYSERVICE",
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
                instance_status="PENDING",
                container_status="PENDING",
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
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],  # service does not exist; create_service takes the deploy path
            ) as mock_show_services,
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
                min_instances=0,
                max_instances=4,
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
                min_instances=0,
                max_instances=4,
                cpu="1",
                memory="6GiB",
                gpu="1",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
                feature_sources_per_function=None,
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
            mock_show_services.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                starts_with="MYSERVICE",
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
                instance_status="PENDING",
                container_status="PENDING",
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
                    "show_services",
                    return_value=[],
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
                    min_instances=0,
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
                instance_status="PENDING",
                container_status="PENDING",
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
                    "show_services",
                    return_value=[],
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
                    min_instances=0,
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
            deployment_step.get_service_id_from_deployment_step(
                query_id,
                deployment_step.DeploymentStep.MODEL_BUILD,
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
                instance_status="PENDING",
                container_status="PENDING",
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
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],  # service does not exist; create_service takes the deploy path
            ) as mock_show_services,
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
                min_instances=0,
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
                min_instances=0,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="2",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
                feature_sources_per_function=None,
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
            mock_show_services.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                starts_with="MYSERVICE",
                statement_params=self.m_statement_params,
            )

    def test_create_service_with_inference_engine_and_no_image_build(self) -> None:
        """Test create_service with custom inference engine parameters and no image build."""
        self._add_snowflake_version_check_mock_operations(self.m_session)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
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
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],  # service does not exist; create_service takes the deploy path
            ) as mock_show_services,
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
                min_instances=2,
                max_instances=7,
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
                min_instances=2,
                max_instances=7,
                cpu="1",
                memory="6GiB",
                gpu="2",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
                feature_sources_per_function=None,
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
            mock_show_services.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                starts_with="MYSERVICE",
                statement_params=self.m_statement_params,
            )

    def test_create_service_with_python_generic_inference_engine(self) -> None:
        """Test create_service with PYTHON_GENERIC inference engine.

        When PYTHON_GENERIC is specified, it should:
        - Call add_inference_engine_spec with PYTHON_GENERIC
        - Skip image build (same as VLLM)
        """
        self._add_snowflake_version_check_mock_operations(self.m_session)
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
                self.m_ops._stage_client,
                "create_tmp_stage",
            ) as _mock_create_stage,  # noqa: F841
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
            ) as _mock_upload_directory_to_stage,  # noqa: F841
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ) as mock_deploy_model,
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],  # service does not exist; create_service takes the deploy path
            ) as mock_show_services,
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
            # Call create_service with PYTHON_GENERIC inference engine (no args override)
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
                min_instances=0,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="2",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=False,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
                block=True,
                statement_params=self.m_statement_params,
                inference_engine_args=service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.PYTHON_GENERIC,
                    inference_engine_args_override=None,
                ),
                progress_status=create_mock_progress_status(),
            )

            # Verify model spec was added
            mock_add_model_spec.assert_called_once_with(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("VERSION"),
            )

            # Verify service spec was added
            mock_add_service_spec.assert_called_once_with(
                service_database_name=sql_identifier.SqlIdentifier("SERVICE_DB"),
                service_schema_name=sql_identifier.SqlIdentifier("SERVICE_SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("SERVICE_COMPUTE_POOL"),
                ingress_enabled=True,
                min_instances=0,
                max_instances=1,
                cpu="1",
                memory="6GiB",
                gpu="2",
                num_workers=1,
                max_batch_rows=1024,
                autocapture=None,
                feature_sources_per_function=None,
            )

            # Key assertion: verify add_inference_engine_spec was called with PYTHON_GENERIC
            mock_add_inference_engine_spec.assert_called_once_with(
                inference_engine=inference_engine.InferenceEngine.PYTHON_GENERIC,
                inference_engine_args=None,
            )

            # Image build should be skipped when inference engine is specified
            mock_add_image_build_spec.assert_not_called()

            mock_save.assert_called_once()
            mock_deploy_model.assert_called_once()
            mock_show_services.assert_called_once()

    def test_create_service_with_python_generic_inference_engine_with_args(self) -> None:
        """Test create_service with PYTHON_GENERIC inference engine and custom args override."""
        self._add_snowflake_version_check_mock_operations(self.m_session)
        m_statuses = [
            service_sql.ServiceStatusInfo(
                service_status=service_sql.ServiceStatus.PENDING,
                instance_id=0,
                instance_status="PENDING",
                container_status="PENDING",
                message=None,
            )
        ]

        # Custom args for PYTHON_GENERIC
        test_inference_engine_args = ["--custom-arg=value", "--another-arg"]

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
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_model_spec",
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_image_build_spec",
            ) as mock_add_image_build_spec,
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_inference_engine_spec",
            ) as mock_add_inference_engine_spec,
            mock.patch.object(file_utils, "upload_directory_to_stage", return_value=None),
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=m_statuses,
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],
            ),
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
            # Call create_service with PYTHON_GENERIC inference engine with custom args
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
                min_instances=0,
                max_instances=1,
                cpu_requests="1",
                memory_requests="6GiB",
                gpu_requests="2",
                num_workers=1,
                max_batch_rows=1024,
                force_rebuild=False,
                build_external_access_integrations=[sql_identifier.SqlIdentifier("EXTERNAL_ACCESS_INTEGRATION")],
                block=True,
                statement_params=self.m_statement_params,
                inference_engine_args=service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.PYTHON_GENERIC,
                    inference_engine_args_override=test_inference_engine_args,
                ),
                progress_status=create_mock_progress_status(),
            )

            # Key assertion: verify add_inference_engine_spec was called with PYTHON_GENERIC and custom args
            mock_add_inference_engine_spec.assert_called_once_with(
                inference_engine=inference_engine.InferenceEngine.PYTHON_GENERIC,
                inference_engine_args=test_inference_engine_args,
            )

            # Image build should be skipped when inference engine is specified
            mock_add_image_build_spec.assert_not_called()

    def test_create_service_block_error_stops_log_thread(self) -> None:
        """Test that log-streaming thread is stopped when create_service(block=True) fails.

        Regression test for thread leak: before the fix, the log thread could outlive
        the error handler because join(timeout=5) returned before the thread finished
        _finalize_logs. The fix adds a stop_event that the thread checks each iteration.
        """
        self._add_snowflake_version_check_mock_operations(self.m_session)

        # async_job.is_done() returns False initially (so the log thread enters its loop),
        # then True (so it can exit once stop_event is set).
        mock_async_job = mock.MagicMock(spec=snowpark.AsyncJob)
        mock_async_job.is_done.side_effect = [False, True, True, True]
        mock_async_job.result.side_effect = RuntimeError("Compute pool does not exist")

        threads_before = set(threading.enumerate())

        with (
            mock.patch.object(self.m_ops._stage_client, "create_tmp_stage"),
            mock.patch.object(
                snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_STAGE_ABCDEF0123"
            ),
            mock.patch.object(self.m_ops._model_deployment_spec, "save"),
            mock.patch.object(file_utils, "upload_directory_to_stage", return_value=None),
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), mock_async_job),
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                side_effect=snowpark.exceptions.SnowparkSQLException(
                    "002003 (02000): Service 'MODEL_BUILD_ABCD1234' does not exist"
                ),
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_logs",
                return_value="",
            ),
            mock.patch.object(
                self.m_ops,
                "_wait_for_service_status",
                side_effect=RuntimeError("Service deployment failed: Compute pool does not exist"),
            ),
        ):
            with self.assertRaises(RuntimeError):
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
                    ingress_enabled=False,
                    min_instances=0,
                    max_instances=1,
                    cpu_requests=None,
                    memory_requests=None,
                    gpu_requests=None,
                    num_workers=None,
                    max_batch_rows=None,
                    force_rebuild=False,
                    build_external_access_integrations=None,
                    block=True,
                    statement_params=self.m_statement_params,
                    hf_model_args=None,
                    progress_status=create_mock_progress_status(),
                    inference_engine_args=None,
                )

        # Wait for any lingering threads to finish
        time.sleep(2)
        new_threads = set(threading.enumerate()) - threads_before
        leaked = {t for t in new_threads if t.name == "service-log-streamer"}
        self.assertEmpty(leaked, f"Log-streaming thread leaked after error: {leaked}")

    def test_create_service_forwards_feature_sources_per_function(self) -> None:
        # Pure passthrough check: ensure ServiceOperator hands the (sentinel)
        # FeatureView mapping to ModelDeploymentSpec.add_service_spec
        # untouched. Validation/serialization is covered in
        # model_deployment_spec_test; here we just pin the wiring.
        self._add_snowflake_version_check_mock_operations(self.m_session)
        current_version = version.Version(snowml_version.VERSION)
        # Inlined deployment spec mode skips the temp-stage SQL path, keeping
        # this wiring test free of unrelated mocks.
        with platform_capabilities.PlatformCapabilities.mock_features(
            features={platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: current_version}
        ):
            self.m_ops = service_ops.ServiceOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            )
        sentinel_feature_sources = {"predict": [mock.sentinel.feature_view]}
        with (
            mock.patch.object(self.m_ops._model_deployment_spec, "save"),
            mock.patch.object(self.m_ops._model_deployment_spec, "add_model_spec"),
            mock.patch.object(
                self.m_ops._model_deployment_spec,
                "add_service_spec",
            ) as mock_add_service_spec,
            mock.patch.object(self.m_ops._model_deployment_spec, "add_image_build_spec"),
            mock.patch.object(
                self.m_ops._service_client,
                "deploy_model",
                return_value=(str(uuid.uuid4()), self._create_mock_async_job()),
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "get_service_container_statuses",
                return_value=[
                    service_sql.ServiceStatusInfo(
                        service_status=service_sql.ServiceStatus.PENDING,
                        instance_id=0,
                        instance_status="PENDING",
                        container_status="PENDING",
                        message=None,
                    )
                ],
            ),
            mock.patch.object(
                self.m_ops._service_client,
                "show_services",
                return_value=[],
            ),
            mock.patch.object(self.m_ops._service_client, "get_service_logs", return_value=""),
            mock.patch.object(self.m_ops, "_wait_for_service_status", return_value=None),
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
                min_instances=0,
                max_instances=1,
                cpu_requests=None,
                memory_requests=None,
                gpu_requests=None,
                num_workers=None,
                max_batch_rows=None,
                force_rebuild=False,
                build_external_access_integrations=None,
                block=True,
                statement_params=self.m_statement_params,
                progress_status=create_mock_progress_status(),
                feature_sources_per_function=sentinel_feature_sources,
            )
        _, kwargs = mock_add_service_spec.call_args
        self.assertIs(kwargs["feature_sources_per_function"], sentinel_feature_sources)

    def test_show_service_returns_match(self) -> None:
        Outcome = row.Row("name", "status", "database_name", "schema_name")
        rows = [Outcome("MYSERVICE", "RUNNING", "DB", "SCHEMA")]
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=rows) as mock_show_services:
            res = self.m_ops._show_service(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                statement_params=None,
            )
        self.assertIsNotNone(res)
        assert res is not None
        self.assertEqual(res["name"], "MYSERVICE")
        self.assertEqual(res["status"], "RUNNING")
        mock_show_services.assert_called_once_with(
            database_name=sql_identifier.SqlIdentifier("DB"),
            schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
            starts_with="MYSERVICE",
            statement_params=None,
        )

    def test_show_service_returns_none_when_no_match(self) -> None:
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=[]):
            res = self.m_ops._show_service(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("GHOST"),
                statement_params=None,
            )
        self.assertIsNone(res)

    def test_show_service_filters_prefix_collisions(self) -> None:
        # STARTS WITH 'FOO' returns FOOBAR; _show_service must reject it.
        Outcome = row.Row("name", "status", "database_name", "schema_name")
        rows = [Outcome("FOOBAR", "RUNNING", "DB", "SCHEMA")]
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=rows):
            res = self.m_ops._show_service(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("FOO"),
                statement_params=None,
            )
        self.assertIsNone(res)

    def test_show_service_quoted_identifier(self) -> None:
        # Case-sensitive (quoted) service name. SHOW SERVICES returns the stored
        # name unquoted in the `name` column; SqlIdentifier.resolved() also
        # produces the stored form. The comparison must hold and STARTS WITH
        # must receive the unquoted form.
        #
        # Note: a lowercase service name like "my-service" cannot actually be
        # created in Snowflake — services require uppercase names because the
        # name is compiled into a DNS label. We keep this unit test anyway to
        # guard the code path: if Snowflake ever relaxes that rule, or if a
        # caller constructs a SqlIdentifier from arbitrary user input, the
        # resolved/identifier round-trip needs to keep working.
        Outcome = row.Row("name", "status", "database_name", "schema_name")
        rows = [Outcome("my-service", "RUNNING", "DB", "SCHEMA")]
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=rows) as mock_show_services:
            res = self.m_ops._show_service(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier('"my-service"'),
                statement_params=None,
            )
        self.assertIsNotNone(res)
        assert res is not None
        self.assertEqual(res["name"], "my-service")
        mock_show_services.assert_called_once_with(
            database_name=sql_identifier.SqlIdentifier("DB"),
            schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
            starts_with="my-service",
            statement_params=None,
        )

    def test_get_service_status_returns_status_when_match_exists(self) -> None:
        Outcome = row.Row("name", "status", "database_name", "schema_name")
        rows = [Outcome("MYSERVICE", "RUNNING", "DB", "SCHEMA")]
        with mock.patch.object(
            self.m_ops._service_client,
            "show_services",
            return_value=rows,
        ) as mock_show_services:
            status = self.m_ops._get_service_status(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                statement_params=None,
            )
        self.assertEqual(status, service_sql.ServiceStatus.RUNNING)
        mock_show_services.assert_called_once_with(
            database_name=sql_identifier.SqlIdentifier("DB"),
            schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
            starts_with="MYSERVICE",
            statement_params=None,
        )

    def test_get_service_status_returns_none_when_no_match(self) -> None:
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=[]):
            status = self.m_ops._get_service_status(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("GHOST"),
                statement_params=None,
            )
        self.assertIsNone(status)

    def test_check_if_service_exists_returns_true_when_match_exists(self) -> None:
        Outcome = row.Row("name", "status", "database_name", "schema_name")
        rows = [Outcome("MYSERVICE", "RUNNING", "DB", "SCHEMA")]
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=rows):
            exists = self.m_ops._check_if_service_exists(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
                statement_params=None,
            )
        self.assertTrue(exists)

    def test_check_if_service_exists_returns_false_when_no_match(self) -> None:
        with mock.patch.object(self.m_ops._service_client, "show_services", return_value=[]):
            exists = self.m_ops._check_if_service_exists(
                database_name=sql_identifier.SqlIdentifier("DB"),
                schema_name=sql_identifier.SqlIdentifier("SCHEMA"),
                service_name=sql_identifier.SqlIdentifier("GHOST"),
                statement_params=None,
            )
        self.assertFalse(exists)

    def test_execute_inference_job_service_with_explicit_job_name(self) -> None:
        m_async_job = self._create_mock_async_job()
        m_async_job.result.return_value = [row.Row("Batch inference job IGNORED with model ...")]
        input_df = mock.MagicMock(spec=dataframe.DataFrame)
        fake_uuid = uuid.UUID("abcdef0123456789abcdef0123456789")
        with (
            mock.patch.object(
                self.m_ops._service_client,
                "execute_inference_job_service",
                return_value=("query_id", m_async_job),
            ) as mock_execute,
            mock.patch("snowflake.ml.model._client.ops.service_ops.uuid.uuid4", return_value=fake_uuid),
        ):
            result = self.m_ops.execute_inference_job_service(
                X=input_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                input_spec=batch_inference_job_specs.InputSpec(params={"k": "v"}),
                output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out"),
                resources_spec=None,
                inference_spec=None,
                image_build_spec=None,
                function_name="predict",
                job_name="JOB",
                replicas=2,
                async_=False,
                statement_params={"test": "1"},
            )
        expected_input_stage = f"@DB.SCHEMA.STAGE/out/_snowflake_temporary/{fake_uuid.hex}/"
        input_df.write.copy_into_location.assert_called_once_with(
            location=expected_input_stage, file_format_type="parquet", header=True, overwrite=True
        )
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args.kwargs
        self.assertIn("input:", call_kwargs["yaml_body"])
        # output_spec.stage_location is normalized to add trailing / before being emitted.
        self.assertIn("stage_location: '@DB.SCHEMA.STAGE/out/'", call_kwargs["yaml_body"])
        self.assertEqual(call_kwargs["model_fqn"], 'TEMP."test".MODEL')
        self.assertEqual(call_kwargs["version"], sql_identifier.SqlIdentifier("V1"))
        self.assertEqual(call_kwargs["function_name"], "predict")
        self.assertEqual(call_kwargs["job_fqn"], 'TEMP."test".JOB')
        self.assertEqual(call_kwargs["replicas"], 2)
        self.assertFalse(call_kwargs["async_"])
        self.assertEqual(call_kwargs["from_stage_path"], expected_input_stage)
        self.assertEqual(result.id, 'TEMP."test".JOB')

    def test_execute_inference_job_service_input_stage_location_skips_materialization(self) -> None:
        m_async_job = self._create_mock_async_job()
        m_async_job.result.return_value = [row.Row("Batch inference job DB.SCHEMA.SRV_GEN with model M ...")]
        with mock.patch.object(
            self.m_ops._service_client,
            "execute_inference_job_service",
            return_value=("query_id", m_async_job),
        ) as mock_execute:
            self.m_ops.execute_inference_job_service(
                input_stage_location="@DB.SCHEMA.STAGE/input",
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                input_spec=None,
                output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                resources_spec=None,
                inference_spec=None,
                image_build_spec=None,
                function_name=None,
                job_name=None,
                replicas=None,
                async_=True,
                statement_params=None,
            )
        from_stage_path = mock_execute.call_args.kwargs["from_stage_path"]
        # Read in place: the user path flows through as the FROM path (only trailing-slash normalized),
        # and no _snowflake_temporary/<uuid>/ staging target is created -- i.e. no COPY INTO materialization.
        self.assertEqual(from_stage_path, "@DB.SCHEMA.STAGE/input/")
        self.assertNotIn(service_ops._BATCH_INFERENCE_RESERVED_INPUT_SUBDIR, from_stage_path)

    def test_execute_inference_job_service_rejects_bad_input_stage_location(self) -> None:
        cases = [
            # (input_stage_location, output_stage_location, expected error fragment)
            ("DB.SCHEMA.STAGE/input", "@DB.SCHEMA.STAGE/out/", "must be a stage path starting with '@'"),
            ("@", "@DB.SCHEMA.STAGE/out/", "not a valid Snowflake stage path"),
            # input nested under the output stage location would be scanned/overwritten as output.
            ("@DB.SCHEMA.STAGE/out/input/", "@DB.SCHEMA.STAGE/out/", "must not be inside output_spec.stage_location"),
        ]
        for input_loc, output_loc, err in cases:
            with self.subTest(input_stage_location=input_loc):
                with mock.patch.object(self.m_ops._service_client, "execute_inference_job_service") as mock_execute:
                    with self.assertRaisesRegex(ValueError, err):
                        self.m_ops.execute_inference_job_service(
                            input_stage_location=input_loc,
                            model_name=sql_identifier.SqlIdentifier("MODEL"),
                            version_name=sql_identifier.SqlIdentifier("V1"),
                            compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                            input_spec=None,
                            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_loc),
                            resources_spec=None,
                            inference_spec=None,
                            image_build_spec=None,
                            function_name=None,
                            job_name=None,
                            replicas=None,
                            async_=True,
                            statement_params=None,
                        )
                mock_execute.assert_not_called()

    def test_execute_inference_job_service_overlap_uses_session_namespace(self) -> None:
        # An unqualified input resolves against the session's current db/schema, so it overlaps a
        # fully-qualified output on that same session stage (case-insensitive on identifiers).
        with (
            mock.patch.object(self.m_ops._session, "get_current_database", return_value="DB"),
            mock.patch.object(self.m_ops._session, "get_current_schema", return_value="SCHEMA"),
            mock.patch.object(self.m_ops._service_client, "execute_inference_job_service") as mock_execute,
        ):
            with self.assertRaisesRegex(ValueError, "must not be inside output_spec.stage_location"):
                self.m_ops.execute_inference_job_service(
                    input_stage_location="@stage/out/input/",
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                    input_spec=None,
                    output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                    resources_spec=None,
                    inference_spec=None,
                    image_build_spec=None,
                    function_name=None,
                    job_name=None,
                    replicas=None,
                    async_=True,
                    statement_params=None,
                )
        mock_execute.assert_not_called()

    def test_execute_inference_job_service_does_not_remove_user_stage_on_failure(self) -> None:
        with (
            mock.patch.object(
                self.m_ops._service_client,
                "execute_inference_job_service",
                side_effect=RuntimeError("server rejected"),
            ),
            mock.patch.object(self.m_ops._session, "sql") as mock_sql,
        ):
            with self.assertRaisesRegex(RuntimeError, "server rejected"):
                self.m_ops.execute_inference_job_service(
                    input_stage_location="@DB.SCHEMA.STAGE/input/",
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                    input_spec=None,
                    output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                    resources_spec=None,
                    inference_spec=None,
                    image_build_spec=None,
                    function_name=None,
                    job_name=None,
                    replicas=None,
                    async_=True,
                    statement_params=None,
                )
        # The caller-owned stage must not be removed: no REMOVE statement is issued. (The operator's
        # only REMOVE runs in the staged-input cleanup, which is skipped for a caller stage path.)
        remove_calls = [c for c in mock_sql.call_args_list if c.args and str(c.args[0]).startswith("REMOVE")]
        self.assertEqual(remove_calls, [])

    def test_execute_inference_job_service_parses_server_generated_name(self) -> None:
        m_async_job = self._create_mock_async_job()
        m_async_job.result.return_value = [row.Row("Batch inference job DB.SCHEMA.SRV_GEN with model M ...")]
        input_df = mock.MagicMock(spec=dataframe.DataFrame)
        with mock.patch.object(
            self.m_ops._service_client,
            "execute_inference_job_service",
            return_value=("query_id", m_async_job),
        ):
            result = self.m_ops.execute_inference_job_service(
                X=input_df,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                input_spec=None,
                output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                resources_spec=None,
                inference_spec=None,
                image_build_spec=None,
                function_name=None,
                job_name=None,
                replicas=None,
                async_=True,
                statement_params=None,
            )
        self.assertEqual(result.id, "DB.SCHEMA.SRV_GEN")

    def test_execute_inference_job_service_unparsable_response_raises(self) -> None:
        m_async_job = self._create_mock_async_job()
        m_async_job.result.return_value = [row.Row("not the expected format")]
        input_df = mock.MagicMock(spec=dataframe.DataFrame)
        with mock.patch.object(
            self.m_ops._service_client,
            "execute_inference_job_service",
            return_value=("query_id", m_async_job),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to parse job name"):
                self.m_ops.execute_inference_job_service(
                    X=input_df,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                    input_spec=None,
                    output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                    resources_spec=None,
                    inference_spec=None,
                    image_build_spec=None,
                    function_name=None,
                    job_name=None,
                    replicas=None,
                    async_=True,
                    statement_params=None,
                )

    def test_execute_inference_job_service_raises_when_copy_into_fails(self) -> None:
        input_df = mock.MagicMock(spec=dataframe.DataFrame)
        input_df.write.copy_into_location.side_effect = Exception("staging boom")
        with mock.patch.object(
            self.m_ops._service_client,
            "execute_inference_job_service",
        ) as mock_execute:
            with self.assertRaisesRegex(RuntimeError, "Failed to process input data"):
                self.m_ops.execute_inference_job_service(
                    X=input_df,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                    input_spec=None,
                    output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                    resources_spec=None,
                    inference_spec=None,
                    image_build_spec=None,
                    function_name=None,
                    job_name=None,
                    replicas=None,
                    async_=True,
                    statement_params=None,
                )
        mock_execute.assert_not_called()

    def test_execute_inference_job_service_cleans_up_staged_input_on_sql_failure(self) -> None:
        """When the SQL call raises, the orphaned staged input must be REMOVED."""
        input_df = mock.MagicMock(spec=dataframe.DataFrame)
        fake_uuid = uuid.UUID("abcdef0123456789abcdef0123456789")
        expected_stage = f"@DB.SCHEMA.STAGE/out/_snowflake_temporary/{fake_uuid.hex}/"
        with (
            mock.patch.object(
                self.m_ops._service_client,
                "execute_inference_job_service",
                side_effect=RuntimeError("server rejected"),
            ),
            mock.patch("snowflake.ml.model._client.ops.service_ops.uuid.uuid4", return_value=fake_uuid),
            mock.patch.object(self.m_ops._session, "sql") as mock_sql,
        ):
            with self.assertRaisesRegex(RuntimeError, "server rejected"):
                self.m_ops.execute_inference_job_service(
                    X=input_df,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                    input_spec=None,
                    output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                    resources_spec=None,
                    inference_spec=None,
                    image_build_spec=None,
                    function_name=None,
                    job_name=None,
                    replicas=None,
                    async_=True,
                    statement_params=None,
                )
        mock_sql.assert_called_once_with(f"REMOVE {expected_stage}")

    def test_execute_inference_job_service_validates_job_name_before_staging(self) -> None:
        """An invalid job_name must raise before COPY INTO so nothing orphans."""
        input_df = mock.MagicMock(spec=dataframe.DataFrame)
        with (
            mock.patch(
                "snowflake.ml.model._client.ops.service_ops.sql_identifier.parse_fully_qualified_name",
                side_effect=ValueError("bad job_name"),
            ),
            mock.patch.object(self.m_ops._service_client, "execute_inference_job_service") as mock_execute,
        ):
            with self.assertRaisesRegex(ValueError, "bad job_name"):
                self.m_ops.execute_inference_job_service(
                    X=input_df,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    compute_pool_name=sql_identifier.SqlIdentifier("POOL"),
                    input_spec=None,
                    output_spec=batch_inference_job_specs.OutputSpec(stage_location="@DB.SCHEMA.STAGE/out/"),
                    resources_spec=None,
                    inference_spec=None,
                    image_build_spec=None,
                    function_name=None,
                    job_name="malformed",
                    replicas=None,
                    async_=True,
                    statement_params=None,
                )
        input_df.write.copy_into_location.assert_not_called()
        mock_execute.assert_not_called()


if __name__ == "__main__":
    absltest.main()
