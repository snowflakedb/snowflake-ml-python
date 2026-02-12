"""Integration test for vLLM inference server with partial params.

This test verifies that partial params work (some set, some using defaults).
"""

import logging

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import registry_vllm_params_test_base

logger = logging.getLogger(__name__)


class TestRegistryVLLMParamsPartialInteg(registry_vllm_params_test_base.RegistryVLLMParamsTestBase):
    """Integration test for vLLM inference with partial params."""

    @pytest.mark.conda_incompatible
    def test_vllm_inference_with_partial_params(self) -> None:
        """Test vLLM inference with partial params (some set, some using defaults)."""
        logger.info("Testing vLLM inference with partial params")

        model = self._create_vllm_model()

        # Input only has messages
        test_input = self._create_test_input()

        # Only provide some params, others should use signature defaults
        # Signature defaults: temperature=1.0, max_completion_tokens=250, n=1, top_p=1.0, etc.
        test_params = {
            "temperature": 0.5,
            "max_completion_tokens": 200,
            # n, top_p, frequency_penalty, presence_penalty use signature defaults
        }

        def check_result(res: pd.DataFrame) -> None:
            # Default n=1, so expect 1 choice
            self._validate_openai_response(res, expected_num_choices=1)
            logger.info("Partial params validated - used signature defaults for missing params")

        self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (test_input, check_result),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_vllm_inference_engine_options(),
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            params=test_params,
            skip_rest_api_test=True,
        )

        logger.info("vLLM with partial params test completed successfully")


if __name__ == "__main__":
    absltest.main()
