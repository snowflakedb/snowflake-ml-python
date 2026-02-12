"""Integration test for vLLM inference server REST API with split format params.

This test verifies that parameter values passed via REST API split format work correctly.
"""

import logging

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import registry_vllm_params_test_base

logger = logging.getLogger(__name__)


class TestRegistryVLLMParamsRestApiSplitInteg(registry_vllm_params_test_base.RegistryVLLMParamsTestBase):
    """Integration test for vLLM inference with params via REST API split format."""

    @pytest.mark.conda_incompatible
    def test_vllm_rest_api_split_format_with_params(self) -> None:
        """Test vLLM REST API with SPLIT format and top-level params."""
        logger.info("Testing vLLM REST API with SPLIT format params")

        model = self._create_vllm_model()

        # Deploy model - input only has messages
        test_input = self._create_test_input()

        deploy_params = {
            "temperature": 1.0,
            "max_completion_tokens": 100,
            "n": 1,
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

        # Test SPLIT format with top-level params
        endpoint = self._ensure_ingress_url(mv)

        messages_data = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Name three colors."}]},
        ]

        split_payload = {
            "dataframe_split": {
                "index": [0],
                "columns": ["messages"],
                "data": [[messages_data]],
            },
            "params": {
                "temperature": 0.6,
                "max_completion_tokens": 80,
                "n": 2,
                "top_p": 0.85,
                "frequency_penalty": 0.05,
                "presence_penalty": 0.05,
            },
        }

        response = self._make_rest_api_request(endpoint, split_payload, "__call__")
        res_df = self._parse_response_data(response)

        self._validate_openai_response(res_df, expected_num_choices=2)
        logger.info("vLLM REST API SPLIT format with params test completed successfully")

    @pytest.mark.conda_incompatible
    def test_vllm_rest_api_split_format_with_params_and_extra_columns(self) -> None:
        """Test vLLM REST API with SPLIT format and top-level params and extra columns."""
        logger.info("Testing vLLM REST API with SPLIT format params and extra columns")

        model = self._create_vllm_model()

        # Deploy model - input only has messages
        test_input = self._create_test_input("What is 2 + 2? Answer with just the number.")

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
            # Prompt asks "What is 2 + 2?", expect "4" in response
            self._validate_openai_response_with_content(res, expected_phrases=["4"], expected_num_choices=1)
            logger.info("SPLIT format with params and extra columns test completed successfully")

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

        # Test SPLIT format with top-level params and extra columns
        endpoint = self._ensure_ingress_url(mv)

        messages_data = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Give brief, factual answers."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "What planet do we live on?"}]},
        ]

        split_payload = {
            "dataframe_split": {
                "index": [0],
                "columns": ["messages", "extra_column"],
                "data": [[messages_data, "extra_value"]],
            },
            "params": {
                "temperature": 0.1,
                "max_completion_tokens": 50,
                "n": 1,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "extra_columns": ["extra_column"],
        }

        response = self._make_rest_api_request(endpoint, split_payload, "__call__")
        res_df = self._parse_response_data(response)

        # Prompt asks what planet we live on, expect "Earth"
        self._validate_openai_response_with_content(res_df, expected_phrases=["Earth"], expected_num_choices=1)
        logger.info("vLLM REST API SPLIT format with params and extra columns test completed successfully")


if __name__ == "__main__":
    absltest.main()
