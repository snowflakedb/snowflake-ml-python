"""Integration test for vLLM inference server REST API with flat format params.

This test verifies that parameter values passed via REST API flat format work correctly.
"""

import logging

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import registry_vllm_params_test_base

logger = logging.getLogger(__name__)


class TestRegistryVLLMParamsRestApiFlatInteg(registry_vllm_params_test_base.RegistryVLLMParamsTestBase):
    """Integration test for vLLM inference with params via REST API flat format."""

    @pytest.mark.conda_incompatible
    def test_vllm_rest_api_flat_format_with_params(self) -> None:
        """Test vLLM REST API with flat format where params are columns in the data."""
        logger.info("Testing vLLM REST API with flat format params")

        model = self._create_vllm_model()

        # Deploy model - input only has messages, params passed separately
        test_input = self._create_test_input()

        # Deployment params (n=1 for single choice)
        deploy_params = {
            "temperature": 0.1,
            "max_completion_tokens": 100,
            "stop": ["."],
            "n": 1,
            "stream": False,
            "top_p": 0.9,
            "frequency_penalty": 0.01,
            "presence_penalty": 0.01,
        }

        def check_result(res: pd.DataFrame) -> None:
            self._validate_openai_response(res, expected_num_choices=1)

        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (test_input, check_result),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_vllm_inference_engine_options(),
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            params=deploy_params,
            skip_rest_api_test=True,
        )

        # Test REST API with different params via flat format
        endpoint = self._ensure_ingress_url(mv)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Write a haiku about mountains."}]},
        ]

        # For flat format, all params are included as columns in the data array
        flat_payload = self._create_flat_format_payload_with_params(
            messages=messages,
            temperature=0.8,
            max_completion_tokens=50,
            n=2,
            top_p=0.9,
        )

        response = self._make_rest_api_request(endpoint, flat_payload, "__call__")
        res_df = self._parse_response_data(response)

        self._validate_openai_response(res_df, expected_num_choices=2)
        logger.info("vLLM REST API flat format with params test completed successfully")


if __name__ == "__main__":
    absltest.main()
