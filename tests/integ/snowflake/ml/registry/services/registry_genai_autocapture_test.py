import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

import pandas as pd
import pytest
import requests
from absl.testing import absltest
from retrying import retry

from snowflake.ml.model import ModelVersion
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface_pipeline
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)


@pytest.mark.spcs_deployment_image
class RegistryGenAIAutocaptureTest(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for Gen AI autocapture on vLLM /{method} path.

    Verifies that vLLM chat completion requests emit gen_ai.* attributes
    to INFERENCE_TABLE when autocapture is enabled.
    """

    _endpoint: Optional[str] = None
    _model_version: Optional[ModelVersion] = None
    _service_name: Optional[str] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_hf_endpoint = None
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    def setUp(self) -> None:
        super().setUp()

        if not self._has_image_override():
            self.skipTest("Skipping: image override environment variables not set.")

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        try:
            self.session.sql("ALTER SESSION SET SPCS_MODEL_INFERENCE_SERVER_ENABLE_AUTOCAPTURE_FOR_VLLM=true").collect()
        except Exception as e:
            self.skipTest(f"Failed to set ENABLE_AUTOCAPTURE_FOR_VLLM session param: {e}")

        if RegistryGenAIAutocaptureTest._endpoint is None:
            self._deploy_test_service()

    def tearDown(self) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_db_manager") and hasattr(cls, "_test_db"):
            cls._db_manager.drop_database(cls._test_db)
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint

    def _deploy_test_service(self) -> None:
        """Deploy a vLLM-backed text-generation model with autocapture enabled."""
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            download_snapshot=False,
        )

        input_data = pd.DataFrame.from_records(
            [
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7,
                    "max_completion_tokens": 50,
                    "stop": None,
                    "n": 1,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
            ]
        )

        def check_response(result: pd.DataFrame) -> None:
            self.assertGreater(len(result), 0)

        service_name = f"genai_autocapture_test_{self._run_id}"

        model_version = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={"__call__": (input_data, check_response)},
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(InferenceEngine.VLLM),
            autocapture=True,
            service_name=service_name,
        )

        RegistryGenAIAutocaptureTest._model_version = model_version
        RegistryGenAIAutocaptureTest._endpoint = self._ensure_ingress_url(model_version)
        RegistryGenAIAutocaptureTest._service_name = service_name
        logger.info(f"Gen AI autocapture test service deployed: {service_name} at {self._endpoint}")

    def _make_chat_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send a chat completion request via /{method} external function format."""
        row = [
            0,  # row index
            messages,  # messages
            None,  # temperature
            None,  # max_completion_tokens
            None,  # stop
            None,  # n
            None,  # stream
            None,  # top_p
            None,  # frequency_penalty
            None,  # presence_penalty
            None,  # response_format
        ]
        payload = {"data": [row]}

        auth_handler = self._get_auth_for_inference(self._endpoint)

        @retry(
            wait_exponential_multiplier=1000,
            wait_exponential_max=30000,
            retry_on_result=self.retry_if_result_status_retriable,
        )
        def _post() -> requests.Response:
            return requests.post(
                f"https://{self._endpoint}/__call__",
                json=payload,
                auth=auth_handler,
                timeout=120,
            )

        response = _post()
        response.raise_for_status()
        return response.json()

    def _query_inference_table(
        self,
        expected_record_count: int,
        timeout_seconds: int = 120,
    ) -> pd.DataFrame:
        """Poll INFERENCE_TABLE until expected number of records appear."""
        model_name = self._model_version.model_name

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception:
            pass

        start_time = time.time()
        last_count = 0

        while time.time() - start_time < timeout_seconds:
            query = f"SELECT * FROM TABLE(INFERENCE_TABLE('{model_name}', SERVICE => '{self._service_name}'))"
            result_df = self.session.sql(query).to_pandas()
            last_count = len(result_df)

            if last_count >= expected_record_count:
                return result_df

            time.sleep(2)

        raise TimeoutError(
            f"Timeout after {timeout_seconds}s: expected {expected_record_count} records, "
            f"found {last_count} in INFERENCE_TABLE"
        )

    def test_genai_autocapture_basic(self):
        """Verify gen_ai.* attributes appear in INFERENCE_TABLE for a vLLM chat request."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = self._make_chat_request(messages)

        self.assertIn("data", response)
        self.assertGreater(len(response["data"]), 0)

        # Wait for records to land — deployment validation call + our test call.
        # Deployment validation sends 1 row (__call__), we send 1 row.
        inference_df = self._query_inference_table(expected_record_count=2, timeout_seconds=120)

        # Find our record (the most recent one).
        record_attributes = json.loads(inference_df["RECORD_ATTRIBUTES"].iloc[-1])

        # Assert gen_ai semantic convention attributes are present.
        expected_keys = [
            "gen_ai.operation.name",
            "gen_ai.request.model",
            "gen_ai.response.model",
            "gen_ai.response.id",
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.response.finish_reasons",
            "gen_ai.output.type",
        ]
        for key in expected_keys:
            self.assertIn(key, record_attributes, f"Missing gen_ai attribute: {key}")

        # Verify values make sense.
        self.assertEqual(record_attributes["gen_ai.operation.name"], "chat")
        self.assertIsInstance(record_attributes["gen_ai.usage.input_tokens"], int)
        self.assertIsInstance(record_attributes["gen_ai.usage.output_tokens"], int)
        self.assertGreater(record_attributes["gen_ai.usage.input_tokens"], 0)
        self.assertGreater(record_attributes["gen_ai.usage.output_tokens"], 0)

        # Verify input messages contain our prompt.
        input_messages = json.loads(record_attributes["gen_ai.input.messages"])
        self.assertGreater(len(input_messages), 0)
        user_messages = [m for m in input_messages if m.get("role") == "user"]
        self.assertGreater(len(user_messages), 0)

        # Verify output messages have assistant response.
        output_messages = json.loads(record_attributes["gen_ai.output.messages"])
        self.assertGreater(len(output_messages), 0)
        self.assertEqual(output_messages[0]["role"], "assistant")

        # Verify standard metadata is also present.
        self.assertIn("snow.model_serving.function.name", record_attributes)
        self.assertIn("snow.model_serving.request.timestamp", record_attributes)
        self.assertIn("snow.model_serving.response.timestamp", record_attributes)
        self.assertIn("snow.model_serving.response.code", record_attributes)

    def test_genai_autocapture_with_params(self):
        """Verify optional request params are captured when explicitly set."""
        row = [
            0,  # row index
            [{"role": "user", "content": "Say hi"}],  # messages
            0.5,  # temperature
            32,  # max_completion_tokens
            None,  # stop
            None,  # n
            None,  # stream
            0.9,  # top_p
            None,  # frequency_penalty
            None,  # presence_penalty
            None,  # response_format
        ]
        payload = {"data": [row]}

        auth_handler = self._get_auth_for_inference(self._endpoint)

        @retry(
            wait_exponential_multiplier=1000,
            wait_exponential_max=30000,
            retry_on_result=self.retry_if_result_status_retriable,
        )
        def _post() -> requests.Response:
            return requests.post(
                f"https://{self._endpoint}/__call__",
                json=payload,
                auth=auth_handler,
                timeout=120,
            )

        response = _post()
        response.raise_for_status()

        # Query and find the latest record.
        inference_df = self._query_inference_table(expected_record_count=3, timeout_seconds=120)
        record_attributes = json.loads(inference_df["RECORD_ATTRIBUTES"].iloc[-1])

        # Verify optional params were captured.
        self.assertIn("gen_ai.request.temperature", record_attributes)
        self.assertAlmostEqual(record_attributes["gen_ai.request.temperature"], 0.5, places=2)
        self.assertIn("gen_ai.request.max_tokens", record_attributes)
        self.assertEqual(record_attributes["gen_ai.request.max_tokens"], 32)
        self.assertIn("gen_ai.request.top_p", record_attributes)
        self.assertAlmostEqual(record_attributes["gen_ai.request.top_p"], 0.9, places=2)


if __name__ == "__main__":
    absltest.main()
