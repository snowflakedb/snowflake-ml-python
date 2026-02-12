"""Integration test for vLLM inference server with default params.

This test verifies that default values are used when parameters are not provided.
"""

import logging

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import registry_vllm_params_test_base

logger = logging.getLogger(__name__)


class TestRegistryVLLMParamsDefaultInteg(registry_vllm_params_test_base.RegistryVLLMParamsTestBase):
    """Integration test for vLLM inference with default params."""

    @pytest.mark.conda_incompatible
    def test_vllm_inference_with_default_params(self) -> None:
        """Test vLLM inference using all default params from signature.

        When no params are passed, signature defaults are used:
        - temperature=1.0
        - max_completion_tokens=250
        - n=1
        - top_p=1.0
        - frequency_penalty=0.0
        - presence_penalty=0.0
        """
        logger.info("Testing vLLM inference with default params")

        model = self._create_vllm_model()

        # Input only has messages, no params passed - signature defaults will be used
        test_input = self._create_test_input()

        def check_result(res: pd.DataFrame) -> None:
            # Default n=1
            self._validate_openai_response(res, expected_num_choices=1)
            logger.info("Default params validated successfully")

        self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (test_input, check_result),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_vllm_inference_engine_options(),
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            # No params passed - signature defaults will be used
            skip_rest_api_test=True,
        )

        logger.info("vLLM with default params test completed successfully")


if __name__ == "__main__":
    absltest.main()
