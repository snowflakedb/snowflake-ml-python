import http
import inspect
import logging
import random
import string
import time
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import requests
import retrying

from snowflake.ml._internal.utils import identifier, jwt_generator, sql_identifier
from snowflake.ml.model import ModelVersion, model_signature, type_hints as model_types
from snowflake.ml.model._client.model import inference_engine_utils
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.ml.utils import authentication
from snowflake.snowpark import row
from tests.integ.snowflake.ml.registry import registry_spcs_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils


class RegistryModelDeploymentTestBase(registry_spcs_test_base.RegistrySPCSTestBase):
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
        autocapture: bool = False,
        inference_engine_options: Optional[dict[str, Any]] = None,
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

        # Set embed_local_ml_library to True explicitly because if we set target_platforms to
        # SNOWPARK_CONTAINER_SERVICES, we will skip the logic which automatically sets it to
        # True when the snowml package is not available in the Snowflake Anaconda Channel.
        options = options or {}
        options["embed_local_ml_library"] = True
        mv = None
        if not use_model_logging:
            mv = self.registry.log_model(
                model=model,
                model_name=name,
                version_name=version,
                sample_input_data=sample_input_data,
                conda_dependencies=conda_dependencies,
                pip_requirements=pip_requirements,
                target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
                options=options,
                signatures=signatures,
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
            autocapture=autocapture,
            inference_engine_options=inference_engine_options,
            experimental_options=experimental_options,
            pip_requirements=pip_requirements,
            conda_dependencies=conda_dependencies,
        )

    def _deploy_model_with_image_override(
        self,
        mv: ModelVersion,
        *,
        service_name: str,
        service_compute_pool: str,
        gpu_requests: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = True,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        autocapture: bool = False,
        inference_engine_options: Optional[dict[str, Any]] = None,
        experimental_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Deploy model with image override."""
        is_gpu = gpu_requests is not None
        image_path = self.BASE_GPU_IMAGE_PATH if is_gpu else self.BASE_CPU_IMAGE_PATH
        assert image_path is not None, "Base image path must be set for image override deployment."
        database, schema, service = self._get_fully_qualified_service_or_job_name(service_name)
        compute_pool = sql_identifier.SqlIdentifier(service_compute_pool)

        self._add_common_model_deployment_spec_options(
            mv=mv, database=database, schema=schema, force_rebuild=force_rebuild
        )
        inference_engine_args = inference_engine_utils._get_inference_engine_args(inference_engine_options)
        # Set inference engine spec if specified
        if inference_engine_args is not None:
            inference_engine_args = inference_engine_utils._enrich_inference_engine_args(
                inference_engine_args,
                gpu_requests,
            )
            mv._service_ops._model_deployment_spec.add_inference_engine_spec(
                inference_engine=inference_engine_args.inference_engine,
                inference_engine_args=inference_engine_args.inference_engine_args_override,
            )

        mv._service_ops._model_deployment_spec.add_service_spec(
            service_name=service,
            inference_compute_pool_name=compute_pool,
            service_database_name=database,
            service_schema_name=schema,
            ingress_enabled=True,
            max_instances=max_instances,
            num_workers=num_workers,
            cpu=cpu_requests,
            memory=memory_requests,
            gpu=gpu_requests,
            max_batch_rows=max_batch_rows,
            autocapture=autocapture,
        )

        query_id, async_job = self._deploy_override_model(
            mv=mv,
            database=database,
            schema=schema,
            inference_image=image_path,
            is_batch_inference=False,
        )

        # stream service logs in a thread
        model_build_service_name = sql_identifier.SqlIdentifier(
            mv._service_ops._get_service_id_from_deployment_step(query_id, service_ops.DeploymentStep.MODEL_BUILD)
        )
        model_build_service = service_ops.ServiceLogInfo(
            database_name=database,
            schema_name=schema,
            service_name=model_build_service_name,
            deployment_step=service_ops.DeploymentStep.MODEL_BUILD,
        )
        model_inference_service = service_ops.ServiceLogInfo(
            database_name=database,
            schema_name=schema,
            service_name=service,
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
        autocapture: bool = False,
        inference_engine_options: Optional[dict[str, Any]] = None,
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
                    autocapture=autocapture,
                    inference_engine_options=inference_engine_options,
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
                    autocapture=autocapture,
                    inference_engine_options=inference_engine_options,
                    experimental_options=experimental_options,
                )
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            assert model is not None
            image_overrides = {
                "SPCS_MODEL_LOGGER_ARCH_AGNOSTIC_CONTAINER_URL": self.MODEL_LOGGER_PATH,
                "SPCS_MODEL_BASE_GPU_INFERENCE_CONTAINER_URL": self.BASE_GPU_IMAGE_PATH,
                "SPCS_MODEL_BASE_CPU_INFERENCE_CONTAINER_URL": self.BASE_CPU_IMAGE_PATH,
                "SPCS_MODEL_INFERENCE_PROXY_CONTAINER_URL": self.PROXY_IMAGE_PATH,
                "SPCS_MODEL_BUILD_CONTAINER_URL": self.BUILDER_IMAGE_PATH,
                "SPCS_MODEL_INFERENCE_ENGINE_CONTAINER_URLS": f'{{"vllm": "{self.VLLM_IMAGE_PATH}"}}',
            }
            if with_image_override:
                for key, value in image_overrides.items():
                    self.session.sql(f"ALTER SESSION SET {key} = '{value}'").collect()
            try:
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
                    inference_engine_options=inference_engine_options,
                    experimental_options=experimental_options,
                )
                mv = self.registry.get_model(model_name).version(version_name)
            finally:
                if with_image_override:
                    for key in image_overrides.keys():
                        self.session.sql(f"ALTER SESSION UNSET {key}").collect()

        assert mv is not None
        while True:
            service_status = mv.list_services().loc[0, "status"]
            if service_status != "PENDING":
                break
            time.sleep(10)

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            # Retry logic for inference calls as Proxy doesn't wait for model loading in the inference server.
            # The inference server status could be RUNNING but the model might not be loaded in memory yet.
            max_retries = 3
            retry_delays = [30, 60, 90]

            for attempt in range(max_retries):
                try:
                    res = mv.run(test_input, function_name=target_method, service_name=service_name)
                    check_func(res)
                    break
                except Exception as e:
                    error_str = str(e)

                    # Check if it's a connection refused error (inference server not ready yet)
                    if "connection refused" in error_str.lower() or "502" in error_str:
                        if attempt < max_retries - 1:
                            wait_time = retry_delays[attempt]
                            logging.warning(
                                f"Inference failed (attempt {attempt + 1}/{max_retries}): {e}. "
                                f"Retrying in {wait_time} seconds..."
                            )
                            time.sleep(wait_time)
                        else:
                            logging.error(f"Inference failed after {max_retries} attempts")
                            raise
                    else:
                        # Not a connection error, raise immediately
                        raise

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
