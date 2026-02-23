import http
import inspect
import logging
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import requests
import retrying

from snowflake.ml._internal.utils import identifier, jwt_generator
from snowflake.ml.model import (
    ModelVersion,
    inference_engine,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.utils import authentication
from tests.integ.snowflake.ml.registry import registry_spcs_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

# Builder type constants for parameterized tests
KANIKO_BUILDER = "kaniko"
INFERENCE_IMAGE_BUILDER = "inference_image_builder"


class RegistryModelDeploymentTestBase(registry_spcs_test_base.RegistrySPCSTestBase):

    # Session parameter names used to override the images.
    _BUILDER_SESSION_PARAM = "SPCS_MODEL_BUILD_CONTAINER_URL"

    _INFERENCE_ENGINE_SESSION_PARAM = "SPCS_MODEL_INFERENCE_ENGINE_CONTAINER_URLS"

    _PROXY_SESSION_PARAM = "SPCS_MODEL_INFERENCE_PROXY_CONTAINER_URL"

    _MODEL_LOGGER_SESSION_PARAM = "SPCS_MODEL_LOGGER_ARCH_AGNOSTIC_CONTAINER_URL"

    _BASE_CPU_IMAGE_SESSION_PARAM = "SPCS_MODEL_BASE_CPU_INFERENCE_CONTAINER_URL"
    _BASE_GPU_IMAGE_SESSION_PARAM = "SPCS_MODEL_BASE_GPU_INFERENCE_CONTAINER_URL"

    def _get_image_override_session_params(self) -> dict[str, str]:
        overrides: dict[str, str] = {
            self._MODEL_LOGGER_SESSION_PARAM: self.MODEL_LOGGER_PATH,
            self._BASE_GPU_IMAGE_SESSION_PARAM: self.BASE_GPU_IMAGE_PATH,
            self._BASE_CPU_IMAGE_SESSION_PARAM: self.BASE_CPU_IMAGE_PATH,
            self._PROXY_SESSION_PARAM: self.PROXY_IMAGE_PATH,
            self._BUILDER_SESSION_PARAM: self.BUILDER_IMAGE_PATH,
        }
        result = {k: v for k, v in overrides.items() if v is not None}

        if self.VLLM_IMAGE_PATH is not None:
            result[self._INFERENCE_ENGINE_SESSION_PARAM] = f'{{"vllm": "{self.VLLM_IMAGE_PATH}"}}'

        return result

    def setUp(self) -> None:
        super().setUp()
        # When running with image override, set the model logger session parameter once
        # so all deployment paths use the locally built image instead of the production one.
        if self._has_image_override():
            for key, value in self._get_image_override_session_params().items():
                self.session.sql(f"ALTER SESSION SET {key} = '{value}'").collect()

    def tearDown(self) -> None:
        if self._has_image_override():
            for key in self._get_image_override_session_params():
                self.session.sql(f"ALTER SESSION UNSET {key}").collect()
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
        autocapture: Optional[bool] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
        experimental_options: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        skip_rest_api_test: bool = False,
        use_inference_image_builder: bool = False,
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
            params=params,
            skip_rest_api_test=skip_rest_api_test,
            use_inference_image_builder=use_inference_image_builder,
        )

    def _deploy_model_service(
        self,
        mv: ModelVersion,
        prediction_assert_fns: dict[str, tuple[Any, Callable[[Any], Any]]],
        service_name: Optional[str] = None,
        gpu_requests: Optional[str] = None,
        service_compute_pool: Optional[str] = None,
        num_workers: Optional[int] = None,
        min_instances: int = 0,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        use_default_repo: bool = False,
        autocapture: Optional[bool] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
        experimental_options: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        skip_rest_api_test: bool = False,
        use_inference_image_builder: bool = False,
    ) -> ModelVersion:

        if service_name is None:
            service_name = f"service_{inspect.stack()[1].function}_{self._run_id}"
        if service_compute_pool is None:
            service_compute_pool = self._TEST_CPU_COMPUTE_POOL if gpu_requests is None else self._TEST_GPU_COMPUTE_POOL

        builder_swapped = False
        if use_inference_image_builder and self._has_image_override():
            assert (
                self.INFERENCE_IMAGE_BUILDER_PATH is not None
            ), "INFERENCE_IMAGE_BUILDER_PATH must be set when use_inference_image_builder=True"
            self.session.sql(
                f"ALTER SESSION SET {self._BUILDER_SESSION_PARAM} = '{self.INFERENCE_IMAGE_BUILDER_PATH}'"
            ).collect()
            builder_swapped = True

        mv.create_service(
            service_name=service_name,
            service_compute_pool=service_compute_pool,
            gpu_requests=gpu_requests,
            num_workers=num_workers,
            min_instances=min_instances,
            max_instances=max_instances,
            max_batch_rows=max_batch_rows,
            ingress_enabled=True,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            autocapture=autocapture,
            inference_engine_options=inference_engine_options,
            experimental_options=experimental_options,
        )
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
                    res = mv.run(test_input, function_name=target_method, service_name=service_name, params=params)
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

        if skip_rest_api_test:
            return mv

        endpoint = RegistryModelDeploymentTestBase._ensure_ingress_url(mv)
        jwt_token_generator = self._get_jwt_token_generator()

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            # For REST API, params need to be included as columns in the DataFrame
            rest_api_input = test_input.copy()
            if params:
                for param_name, param_value in params.items():
                    rest_api_input[param_name] = [param_value] * len(rest_api_input)
            res_df = self._inference_using_rest_api(
                self._to_external_data_format(rest_api_input),
                endpoint=endpoint,
                jwt_token_generator=jwt_token_generator,
                target_method=target_method,
            )
            check_func(res_df)

        if builder_swapped:
            self.session.sql(f"ALTER SESSION SET {self._BUILDER_SESSION_PARAM} = '{self.BUILDER_IMAGE_PATH}'").collect()

        return mv

    def _get_inference_engine_options_for_inference_engine(
        self,
        inference_engine_type: str,
        base_inference_engine_options: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Helper method to generate inference_engine_options based on inference engine type.

        Args:
            inference_engine_type: Inference engine type - either "Default" (Python) or "vLLM"
            base_inference_engine_options: Base inference engine options to merge with inference engine-specific options

        Returns:
            Dictionary of inference engine options or None for Default backend

        Raises:
            ValueError: If an unknown inference engine type is provided. Must be 'Default' or 'vLLM'.
        """
        inference_engine_options = base_inference_engine_options.copy() if base_inference_engine_options else {}

        if inference_engine_type == "vLLM":
            inference_engine_options["engine"] = inference_engine.InferenceEngine.VLLM
        elif inference_engine_type != "Default":
            raise ValueError(f"Unknown inference engine type: {inference_engine_type}. Must be 'Default' or 'vLLM'")

        # Return None for Default backend if no other options are set
        if inference_engine_type == "Default" and not inference_engine_options:
            return None

        return inference_engine_options if inference_engine_options else None

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

    def _to_external_data_format(self, test_input: pd.DataFrame) -> dict[str, Any]:
        test_input_arr = model_signature._convert_local_data_to_df(test_input).values
        test_input_arr = np.column_stack([range(test_input_arr.shape[0]), test_input_arr])
        return {"data": test_input_arr.tolist()}

    def _inference_using_rest_api(
        self,
        request_payload: dict[str, Any],
        *,
        endpoint: str,
        jwt_token_generator: Optional[jwt_generator.JWTGenerator] = None,
        target_method: str,
    ) -> pd.DataFrame:
        # Use automatic auth selection (jwt_token_generator kept for backward compatibility but ignored)
        auth_handler = self._get_auth_for_inference(endpoint)

        res = retrying.retry(
            wait_exponential_multiplier=100,
            wait_exponential_max=4000,
            retry_on_result=RegistryModelDeploymentTestBase.retry_if_result_status_retriable,
        )(requests.post)(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json=request_payload,
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
