import http
import inspect
import logging
import os
import pathlib
import random
import string
import tempfile
import time
import uuid
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import pytest
import requests
import retrying
import yaml
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from snowflake.ml._internal import file_utils, platform_capabilities as pc
from snowflake.ml._internal.utils import identifier, jwt_generator, sql_identifier
from snowflake.ml.model import ModelVersion, model_signature, type_hints as model_types
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.ml.registry import registry
from snowflake.ml.utils import authentication, connection_params
from snowflake.snowpark import row
from snowflake.snowpark._internal import utils as snowpark_utils
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)


@pytest.mark.spcs_deployment_image
class RegistryModelDeploymentTestBase(common_test_base.CommonTestBase):
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"
    _TEST_SPCS_WH = "REGTEST_ML_SMALL"

    BUILDER_IMAGE_PATH = os.getenv("BUILDER_IMAGE_PATH", None)
    BASE_CPU_IMAGE_PATH = os.getenv("BASE_CPU_IMAGE_PATH", None)
    BASE_GPU_IMAGE_PATH = os.getenv("BASE_GPU_IMAGE_PATH", None)
    PROXY_IMAGE_PATH = os.getenv("PROXY_IMAGE_PATH", None)
    MODEL_LOGGER_PATH = os.getenv("MODEL_LOGGER_PATH", None)

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        # Get login options BEFORE session creation (which clears password for security)
        login_options = connection_params.SnowflakeLoginOptions()

        # Capture password from login options before session creation clears it
        pat_token = login_options.get("password")

        # Now create session (this will clear password in session._conn._lower_case_parameters)
        super().setUp()

        # Set log level to INFO so that service logs are visible
        logging.basicConfig(level=logging.INFO)

        # Read private_key_path from session connection parameters (after session creation)
        conn_params = self.session._conn._lower_case_parameters
        private_key_path = conn_params.get("private_key_path")

        if private_key_path:
            # Try to load private key for JWT authentication
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=backends.default_backend()
                )
            self.pat_token = None
        elif pat_token:
            # Use PAT token from password parameter
            self.private_key = None
            self.pat_token = pat_token
        else:
            # No authentication credentials available
            self.private_key = None
            self.pat_token = None
            raise ValueError("No authentication credentials found: neither private_key_path nor password parameter set")

        self.snowflake_account_url = self.session._conn._lower_case_parameters.get("host", None)
        if self.snowflake_account_url:
            self.snowflake_account_url = f"https://{self.snowflake_account_url}"

        self._run_id = uuid.uuid4().hex[:4]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = "PUBLIC"
        self._test_image_repo = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "image_repo"
        ).upper()
        self._test_stage = "TEST_STAGE"

        if not self.session.get_current_warehouse():
            self.session.sql(f"USE WAREHOUSE {self._TEST_SPCS_WH}").collect()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_stage(self._test_stage)
        self._db_manager.create_image_repo(self._test_image_repo)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _has_image_override(self) -> bool:
        """Check if image override environment variables are set.

        Returns:
            True if all image override environment variables are set, False otherwise.

        Raises:
            ValueError: If some but not all of the required variables are set.
        """
        image_paths = [
            self.BUILDER_IMAGE_PATH,
            self.BASE_CPU_IMAGE_PATH,
            self.BASE_GPU_IMAGE_PATH,
            self.MODEL_LOGGER_PATH,
        ]

        if all(image_paths):
            return True
        elif not any(image_paths):
            return False
        else:
            raise ValueError(
                "Please set or unset BUILDER_IMAGE_PATH, BASE_CPU_IMAGE_PATH, BASE_GPU_IMAGE_PATH, "
                "and MODEL_LOGGER_PATH at the same time."
            )

    def _deploy_model_with_image_override(
        self,
        mv: ModelVersion,
        service_name: str,
        service_compute_pool: str,
        gpu_requests: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = True,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        experimental_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Deploy model with image override."""
        # Extract autocapture from experimental_options
        autocapture = experimental_options.get("autocapture") if experimental_options else None
        is_gpu = gpu_requests is not None
        image_path = self.BASE_GPU_IMAGE_PATH if is_gpu else self.BASE_CPU_IMAGE_PATH
        build_compute_pool = sql_identifier.SqlIdentifier(self._TEST_CPU_COMPUTE_POOL)

        # create a temp stage
        database_name_id, schema_name_id, service_name_id = sql_identifier.parse_fully_qualified_name(service_name)
        database_name_id = database_name_id or sql_identifier.SqlIdentifier(self._test_db)
        schema_name_id = schema_name_id or sql_identifier.SqlIdentifier(self._test_schema)
        stage_name = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        )
        image_repo_name = sql_identifier.SqlIdentifier(self._test_image_repo)

        mv._service_ops._model_deployment_spec.add_model_spec(
            database_name=mv._model_ops._model_version_client._database_name,
            schema_name=mv._model_ops._model_version_client._schema_name,
            model_name=mv._model_name,
            version_name=mv._version_name,
        )

        image_repo_fqn = identifier.get_schema_level_object_identifier(
            database_name_id.identifier(), schema_name_id.identifier(), image_repo_name.identifier()
        )
        mv._service_ops._model_deployment_spec.add_image_build_spec(
            image_build_compute_pool_name=build_compute_pool,
            fully_qualified_image_repo_name=image_repo_fqn,
            force_rebuild=force_rebuild,
            external_access_integrations=None,
        )

        mv._service_ops._model_deployment_spec.add_service_spec(
            service_name=service_name_id,
            inference_compute_pool_name=sql_identifier.SqlIdentifier(service_compute_pool),
            service_database_name=database_name_id,
            service_schema_name=schema_name_id,
            ingress_enabled=True,
            max_instances=max_instances,
            num_workers=num_workers,
            cpu=cpu_requests,
            memory=memory_requests,
            gpu=gpu_requests,
            max_batch_rows=max_batch_rows,
            autocapture=autocapture,
        )

        deploy_spec = mv._service_ops._model_deployment_spec.save()

        inline_deploy_spec_enabled = pc.PlatformCapabilities.get_instance().is_inlined_deployment_spec_enabled()
        if mv._service_ops._model_deployment_spec.workspace_path:
            with pathlib.Path(deploy_spec).open("r", encoding="utf-8") as f:
                deploy_spec_dict = yaml.safe_load(f)
        else:
            deploy_spec_dict = yaml.safe_load(deploy_spec)

        deploy_spec_dict["image_build"]["builder_image"] = self.BUILDER_IMAGE_PATH
        deploy_spec_dict["image_build"]["base_image"] = image_path
        deploy_spec_dict["service"]["proxy_image"] = self.PROXY_IMAGE_PATH

        if inline_deploy_spec_enabled:
            # dict to yaml string
            deploy_spec_yaml_str = yaml.dump(deploy_spec_dict)
            # deploy the model service
            query_id, async_job = mv._service_ops._service_client.deploy_model(
                model_deployment_spec_yaml_str=deploy_spec_yaml_str,
            )
        else:
            temp_dir = tempfile.TemporaryDirectory()
            workspace_path = pathlib.Path(temp_dir.name)
            deploy_spec_file_rel_path = model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH
            stage_path = mv._service_ops._stage_client.create_tmp_stage(
                database_name=database_name_id,
                schema_name=schema_name_id,
                stage_name=stage_name,
            )
            with (workspace_path / deploy_spec_file_rel_path).open("w", encoding="utf-8") as f:
                yaml.dump(deploy_spec_dict, f)
            file_utils.upload_directory_to_stage(
                self.session,
                local_path=workspace_path,
                stage_path=pathlib.PurePosixPath(stage_path),
            )
            # deploy the model service
            query_id, async_job = mv._service_ops._service_client.deploy_model(
                stage_path=stage_path, model_deployment_spec_file_rel_path=deploy_spec_file_rel_path
            )

        # stream service logs in a thread
        model_build_service_name = sql_identifier.SqlIdentifier(
            mv._service_ops._get_service_id_from_deployment_step(query_id, service_ops.DeploymentStep.MODEL_BUILD)
        )
        model_build_service = service_ops.ServiceLogInfo(
            database_name=database_name_id,
            schema_name=schema_name_id,
            service_name=model_build_service_name,
            deployment_step=service_ops.DeploymentStep.MODEL_BUILD,
        )
        model_inference_service = service_ops.ServiceLogInfo(
            database_name=database_name_id,
            schema_name=schema_name_id,
            service_name=service_name_id,
            deployment_step=service_ops.DeploymentStep.MODEL_INFERENCE,
        )

        log_thread = mv._service_ops._start_service_log_streaming(
            async_job=async_job,
            model_logger_service=None,
            model_build_service=model_build_service,
            model_inference_service=model_inference_service,
            model_inference_service_exists=False,
            force_rebuild=True,
            operation_id=query_id,
        )
        log_thread.join()

        res = cast(str, cast(list[row.Row], async_job.result())[0][0])
        logging.info(f"Inference service {service_name} deployment complete: {res}")

    def _test_registry_model_deployment(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: dict[str, tuple[Any, Callable[[Any], Any]]],
        service_name: Optional[str] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        gpu_requests: Optional[str] = None,
        service_compute_pool: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = True,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        use_default_repo: bool = False,
        experimental_options: Optional[dict[str, Any]] = None,
        use_model_logging: bool = False,
    ) -> ModelVersion:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{self._run_id}"

        mv = None
        if not use_model_logging:
            mv = self.registry.log_model(
                model=model,
                model_name=name,
                version_name=version,
                sample_input_data=sample_input_data,
                conda_dependencies=conda_dependencies,
                pip_requirements=pip_requirements,
                options=options,
            )

        return self._deploy_model_service(
            mv=mv,
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            service_name=service_name,
            gpu_requests=gpu_requests,
            service_compute_pool=service_compute_pool,
            num_workers=num_workers,
            max_instances=max_instances,
            max_batch_rows=max_batch_rows,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            use_default_repo=use_default_repo,
            experimental_options=experimental_options,
            pip_requirements=pip_requirements,
            conda_dependencies=conda_dependencies,
        )

    def _deploy_model_service(
        self,
        model: Optional[model_types.SupportedModelType],
        mv: Optional[ModelVersion],
        prediction_assert_fns: dict[str, tuple[Any, Callable[[Any], Any]]],
        service_name: Optional[str] = None,
        gpu_requests: Optional[str] = None,
        service_compute_pool: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        use_default_repo: bool = False,
        experimental_options: Optional[dict[str, Any]] = None,
        pip_requirements: Optional[list[str]] = None,
        conda_dependencies: Optional[list[str]] = None,
    ) -> ModelVersion:
        with_image_override = self._has_image_override()

        if service_name is None:
            service_name = f"service_{inspect.stack()[1].function}_{self._run_id}"
        if service_compute_pool is None:
            service_compute_pool = self._TEST_CPU_COMPUTE_POOL if gpu_requests is None else self._TEST_GPU_COMPUTE_POOL

        if mv is not None:
            if with_image_override:
                self._deploy_model_with_image_override(
                    mv,
                    service_name=service_name,
                    service_compute_pool=sql_identifier.SqlIdentifier(service_compute_pool),
                    gpu_requests=gpu_requests,
                    num_workers=num_workers,
                    max_instances=max_instances,
                    max_batch_rows=max_batch_rows,
                    force_rebuild=False,
                    cpu_requests=cpu_requests,
                    memory_requests=memory_requests,
                    experimental_options=experimental_options,
                )
            else:
                mv.create_service(
                    service_name=service_name,
                    image_build_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                    service_compute_pool=service_compute_pool,
                    image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                    gpu_requests=gpu_requests,
                    force_rebuild=True,
                    num_workers=num_workers,
                    max_instances=max_instances,
                    max_batch_rows=max_batch_rows,
                    ingress_enabled=True,
                    cpu_requests=cpu_requests,
                    memory_requests=memory_requests,
                    experimental_options=experimental_options,
                )
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            assert model is not None
            if with_image_override:
                self.session.sql(
                    f"ALTER SESSION SET SPCS_MODEL_LOGGER_ARCH_AGNOSTIC_CONTAINER_URL = '{self.MODEL_LOGGER_PATH}'"
                ).collect()
            model_name = "".join(random.choices(string.ascii_uppercase, k=5))
            version_name = "".join(random.choices(string.ascii_uppercase, k=5))
            model.log_model_and_create_service(
                session=self.session,
                model_name=model_name,
                version_name=version_name,
                pip_requirements=pip_requirements,
                conda_dependencies=conda_dependencies,
                service_name=service_name,
                service_compute_pool=service_compute_pool,
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                gpu_requests=gpu_requests,
                force_rebuild=True,
                num_workers=num_workers,
                max_instances=max_instances,
                max_batch_rows=max_batch_rows,
                ingress_enabled=True,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
                experimental_options=experimental_options,
            )

            mv = self.registry.get_model(model_name).version(version_name)
            if with_image_override:
                self.session.sql("ALTER SESSION UNSET SPCS_MODEL_LOGGER_ARCH_AGNOSTIC_CONTAINER_URL").collect()

        assert mv is not None
        while True:
            service_status = mv.list_services().loc[0, "status"]
            if service_status != "PENDING":
                break
            time.sleep(10)

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv.run(test_input, function_name=target_method, service_name=service_name)
            check_func(res)

        endpoint = RegistryModelDeploymentTestBase._ensure_ingress_url(mv)
        jwt_token_generator = self._get_jwt_token_generator()

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res_df = self._inference_using_rest_api(
                test_input, endpoint=endpoint, jwt_token_generator=jwt_token_generator, target_method=target_method
            )
            check_func(res_df)

        return mv

    @staticmethod
    def retry_if_result_status_retriable(result: requests.Response) -> bool:
        if result.status_code in [
            http.HTTPStatus.SERVICE_UNAVAILABLE,
            http.HTTPStatus.TOO_MANY_REQUESTS,
            http.HTTPStatus.GATEWAY_TIMEOUT,
        ]:
            return True
        return False

    @staticmethod
    def _ensure_ingress_url(mv: ModelVersion) -> str:
        while True:
            endpoint = mv.list_services().loc[0, "inference_endpoint"]
            if endpoint is not None:
                break
            time.sleep(10)
        return endpoint

    def _get_jwt_token_generator(self) -> Optional[jwt_generator.JWTGenerator]:
        """Get JWT token generator if private key is available."""
        if self.private_key is None:
            return None

        account = identifier.get_unescaped_names(self.session.get_current_account())
        user = identifier.get_unescaped_names(self.session.get_current_user())
        if not account or not user:
            raise ValueError("Account and user must be set.")

        return authentication.get_jwt_token_generator(
            account,
            user,
            self.private_key,
        )

    def _get_auth_for_inference(self, endpoint: str):
        """Get authentication for inference requests - private key first, then PAT fallback."""
        if self.private_key:
            # Use JWT authentication if private key available
            jwt_token_generator = self._get_jwt_token_generator()
            return authentication.SnowflakeJWTTokenAuth(
                jwt_token_generator=jwt_token_generator,
                role=identifier.get_unescaped_names(self.session.get_current_role()),
                endpoint=endpoint,
                snowflake_account_url=self.snowflake_account_url,
            )
        elif self.pat_token:
            # Fallback to PAT authentication
            return authentication.SnowflakePATAuth(self.pat_token)
        else:
            raise ValueError("No authentication credentials available for inference requests")

    def _inference_using_rest_api(
        self,
        test_input: pd.DataFrame,
        *,
        endpoint: str,
        jwt_token_generator: Optional[jwt_generator.JWTGenerator] = None,
        target_method: str,
    ) -> pd.DataFrame:
        test_input_arr = model_signature._convert_local_data_to_df(test_input).values
        test_input_arr = np.column_stack([range(test_input_arr.shape[0]), test_input_arr])

        # Use automatic auth selection (jwt_token_generator kept for backward compatibility but ignored)
        auth_handler = self._get_auth_for_inference(endpoint)

        res = retrying.retry(
            wait_exponential_multiplier=100,
            wait_exponential_max=4000,
            retry_on_result=RegistryModelDeploymentTestBase.retry_if_result_status_retriable,
        )(requests.post)(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json={"data": test_input_arr.tolist()},
            auth=auth_handler,
        )
        res.raise_for_status()
        return pd.DataFrame([x[1] for x in res.json()["data"]])

    def _single_inference_request(
        self,
        test_input: pd.DataFrame,
        *,
        endpoint: str,
        jwt_token_generator: jwt_generator.JWTGenerator,
        target_method: str,
    ) -> requests.Response:
        """Make a single REST API inference request without retries.

        This method is designed for performance testing where we want to measure
        the actual response time of individual requests without retry overhead.

        Args:
            test_input: Input data for inference
            endpoint: Service endpoint URL
            jwt_token_generator: JWT token generator for authentication
            target_method: Target method name (e.g., 'predict')

        Returns:
            Raw requests.Response object (success or failure)
        """
        test_input_arr = model_signature._convert_local_data_to_df(test_input).values
        test_input_arr = np.column_stack([range(test_input_arr.shape[0]), test_input_arr])

        return requests.post(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json={"data": test_input_arr.tolist()},
            auth=authentication.SnowflakeJWTTokenAuth(
                jwt_token_generator=jwt_token_generator,
                role=identifier.get_unescaped_names(self.session.get_current_role()),
                endpoint=endpoint,
                snowflake_account_url=self.snowflake_account_url,
            ),
            timeout=60,  # 60 second timeout since ingress will timeout after 60 seconds.
            # This will help in case the service itself is not reachable.
        )
