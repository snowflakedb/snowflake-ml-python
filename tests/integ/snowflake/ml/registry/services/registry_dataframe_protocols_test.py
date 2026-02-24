"""Test dataframe protocols for model deployment."""

import json
import logging
import os
import tempfile
from typing import Any, Callable, Optional

import pandas as pd
import pytest
import requests
from absl.testing import absltest, parameterized

from snowflake.ml.model import ModelVersion
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.models import huggingface_pipeline
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)


class TestRegistryDataframeProtocolsInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_hf_endpoint = None
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        # Get HF token if available (used for gated models)
        cls.hf_token = os.getenv("HF_TOKEN", None)
        # Unset HF_ENDPOINT to avoid artifactory errors
        # TODO: Remove this once artifactory is fixed
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint

    def setUp(self) -> None:
        """Set up test - track services created during test."""
        super().setUp()
        self._test_services = []  # Track services created in this test

    def tearDown(self) -> None:
        """Clean up services created during test."""
        # Delete all services tracked during this test
        for mv, service_name in self._test_services:
            try:
                logger.info(f"Deleting service: {service_name}")
                mv.delete_service(service_name)
                logger.info(f"Service {service_name} deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete service {service_name}: {e}")

        # Call parent tearDown
        super().tearDown()

    def _create_payload_for_protocol(
        self,
        input_data: pd.DataFrame,
        protocol: str,
    ) -> dict[str, Any]:
        orient = protocol.replace("dataframe_", "")
        res = json.loads(input_data.to_json(orient=orient))
        return {f"{protocol}": res}

    def _create_payload_for_protocol_with_extra_columns(
        self,
        input_data: pd.DataFrame,
        protocol: str,
    ) -> dict[str, Any]:
        input_data_copy = input_data.copy()
        # Add extra column to input data
        input_data_copy["extra_column"] = pd.Timestamp.now()
        # Create payload for protocol
        req_data = self._create_payload_for_protocol(input_data_copy, protocol)
        # Add top level key to specify extra columns
        req_data["extra_columns"] = ["extra_column"]
        return req_data

    def _make_rest_api_call(
        self,
        endpoint: str,
        request_payload: dict[str, Any],
        target_method: str,
    ) -> requests.Response:
        auth_handler = self._get_auth_for_inference(endpoint)

        response = requests.post(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json=request_payload,
            auth=auth_handler,
            timeout=30,
        )
        return response

    def _test_with_model_logging(
        self,
        model: Any,
        method_name: str,
        inference_engine: str = "Default",
        input_data: Optional[pd.DataFrame] = None,
        validator_fn: Callable[[pd.DataFrame], None] = None,
        input_data_batch: Optional[pd.DataFrame] = None,
        validator_fn_batch: Callable[[pd.DataFrame], None] = None,
    ) -> ModelVersion:
        # TODO remove this once param rollout is complete
        self.session.sql(
            "alter session set SPCS_MODEL_INFERENCE_SERVER_ONLINE_REST_INFERENCE_PANDAS_PROTOCOLS_ENABLED = true;"
        ).collect()

        # Deploy model and test with single row
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                method_name: (
                    input_data,
                    validator_fn,
                ),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(
                inference_engine,
            ),
        )

        service_name = mv.list_services().loc[0, "name"]
        # Track service for cleanup in tearDown
        self._test_services.append((mv, service_name))

        endpoint = self._ensure_ingress_url(mv)
        jwt_token_generator = self._get_jwt_token_generator()

        res_service = mv.run(input_data_batch, function_name=method_name, service_name=service_name)
        validator_fn_batch(res_service)

        # External data format
        res_api = self._inference_using_rest_api(
            endpoint=endpoint,
            request_payload=self._to_external_data_format(input_data_batch),
            jwt_token_generator=jwt_token_generator,
            target_method=method_name,
        )
        validator_fn_batch(res_api)

        # Use only batch data for the other protocols

        # Split format
        request_body = self._create_payload_for_protocol(input_data_batch, "dataframe_split")
        res = self._inference_using_rest_api(
            endpoint=endpoint,
            request_payload=request_body,
            jwt_token_generator=jwt_token_generator,
            target_method=method_name,
        )
        validator_fn_batch(res)

        # Split format extra columns (passes)
        request_body = self._create_payload_for_protocol_with_extra_columns(input_data_batch, "dataframe_split")
        res = self._inference_using_rest_api(
            endpoint=endpoint,
            request_payload=request_body,
            jwt_token_generator=jwt_token_generator,
            target_method=method_name,
        )
        validator_fn_batch(res)

        # Records format
        request_body = self._create_payload_for_protocol(input_data_batch, "dataframe_records")
        res = self._inference_using_rest_api(
            endpoint=endpoint,
            request_payload=request_body,
            jwt_token_generator=jwt_token_generator,
            target_method=method_name,
        )
        validator_fn_batch(res)

        # Records format extra columns (passes)
        request_body = self._create_payload_for_protocol_with_extra_columns(input_data_batch, "dataframe_records")
        res = self._inference_using_rest_api(
            endpoint=endpoint,
            request_payload=request_body,
            jwt_token_generator=jwt_token_generator,
            target_method=method_name,
        )
        validator_fn_batch(res)

        # Return the ModelVersion object so the test can delete the service
        return mv

    def _cleanup_services(self, mv: ModelVersion) -> None:
        services_df = mv.list_services()
        services = services_df["name"]
        self.assertLen(services, 1)

        for service in services:
            logger.info(f"Deleting service: {service}")
            mv.delete_service(service)

        services_df = mv.list_services()
        self.assertLen(services_df, 0)
        logger.info("Service cleanup completed successfully")

    @parameterized.parameters(  # type: ignore[misc]
        ("Default",),
        ("vLLM",),
    )
    @pytest.mark.conda_incompatible
    @absltest.skip("Skipping test until inference server release 1.0.0")
    def test_inference_with_HF_model(self, inference_engine: str) -> None:
        # Define model
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B",
            download_snapshot=False,
        )

        # Define test input data
        x_df_single = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Complete the sentence.",
                        },
                        {
                            "role": "user",
                            "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",
                        },
                    ],
                    "temperature": 0.9,
                    "max_completion_tokens": 250,
                    "stop": None,
                    "n": 3,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                }
            ],
        )

        # Define validator function to check response
        def check_single_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 3)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])
                self.assertGreater(len(row[0]["message"]["content"]), 0)

        # Define test input data (batch)
        test_prompts = [
            "What is the capital of France?",
            "Write a short poem about the ocean.",
            "Explain what machine learning is in one sentence.",
        ]

        x_df_batch = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.7,
                    "max_completion_tokens": 200,
                    "stop": None,
                    "n": 2,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                }
                for prompt in test_prompts
            ],
        )

        # Define validator function to check batch response
        def check_batch_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            self.assertEqual(len(res), len(test_prompts))

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 2)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])
                self.assertGreater(len(row[0]["message"]["content"]), 0)

        # Run test
        mv = self._test_with_model_logging(
            model=model,
            inference_engine=inference_engine,
            method_name="__call__",
            input_data=x_df_single,
            validator_fn=check_single_res,
            input_data_batch=x_df_batch,
            validator_fn_batch=check_batch_res,
        )

        self._cleanup_services(mv)


if __name__ == "__main__":
    absltest.main()
