"""Integration test for vLLM inference server with params passed via mv.run().

This test verifies that parameter values passed at inference time via mv.run() are correctly received.
"""

import logging

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import registry_vllm_params_test_base

logger = logging.getLogger(__name__)


class TestRegistryVLLMParamsMvRunInteg(registry_vllm_params_test_base.RegistryVLLMParamsTestBase):
    """Integration test for vLLM inference with params via mv.run()."""

    @pytest.mark.conda_incompatible
    def test_vllm_inference_with_params_via_mv_run(self) -> None:
        """Test vLLM inference with params passed via mv.run()."""
        logger.info("Testing vLLM inference with params via mv.run()")

        model = self._create_vllm_model()

        # Input with only messages (params are passed separately via params argument)
        test_input = self._create_test_input()

        # Params to pass via mv.run()
        test_params = {
            "temperature": 0.5,
            "max_completion_tokens": 150,
            "n": 2,
            "top_p": 0.95,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
        }

        def check_result(res: pd.DataFrame) -> None:
            self._validate_openai_response(res, expected_num_choices=2)
            logger.info(f"vLLM response validated successfully: {len(res['choices'].iloc[0])} choices")

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

        logger.info("vLLM with params via mv.run() test completed successfully")


if __name__ == "__main__":
    absltest.main()
