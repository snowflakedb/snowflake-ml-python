"""Integration tests for mv.run() against a service with model params (ParamSpec).

This test validates that mv.run() correctly invokes a deployed service when the model
has ParamSpec parameters defined in its signature.

Test methods (4):
- test_mv_run_full_param_override: All params passed via mv.run(params={...})
- test_mv_run_partial_param_override: Some params passed via mv.run(params={...}), others use defaults
- test_mv_run_no_param_override: No params passed, use signature defaults
- test_mv_run_with_params_in_dataframe: Params included as columns in the input DataFrame
"""

import os
import tempfile
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import model_signature, openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.models import huggingface_pipeline
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

# Full set of parameters
_FULL_PARAMS = {
    "temperature": 0.9,
    "max_completion_tokens": 20,
    "stop": None,
    "n": 1,
    "stream": False,
    "top_p": 1.0,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
}

# Partial parameters (only some overridden)
_PARTIAL_PARAMS = {
    "temperature": 0.7,
    "max_completion_tokens": 800,
}

_SIGNATURE = openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE


def _get_input_df(include_params_in_df: bool) -> pd.DataFrame:
    """Get input DataFrame for the test."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Complete the sentence."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of Canada?"}],
        },
    ]

    record: dict[str, Any] = {"messages": messages}

    if include_params_in_df:
        record.update(_FULL_PARAMS)

    return pd.DataFrame.from_records([record])


def _check_openai_response(test_case: absltest.TestCase, res: pd.DataFrame) -> None:
    """Verify the response is in OpenAI chat completion format."""
    pd.testing.assert_index_equal(
        res.columns,
        pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
        check_order=False,
    )

    for row in res["choices"]:
        test_case.assertIsInstance(row, list)
        test_case.assertIn("message", row[0])
        test_case.assertIn("content", row[0]["message"])


def _verify_signature(
    test_case: absltest.TestCase,
    mv: object,
    expected_signature: model_signature.ModelSignature,
) -> None:
    """Verify the logged model's signature matches the expected signature."""
    functions = mv.show_functions()  # type: ignore[attr-defined]

    func_info = None
    for f in functions:
        if f["target_method"] == "__call__":
            func_info = f
            break

    test_case.assertIsNotNone(func_info, "Method __call__ not found")
    actual_sig = func_info["signature"]

    # Verify inputs
    test_case.assertEqual(len(actual_sig.inputs), len(expected_signature.inputs))

    # Verify params (critical for stop param array bug)
    expected_params = expected_signature.params or []
    actual_params = actual_sig.params or []
    test_case.assertEqual(len(actual_params), len(expected_params))

    for expected, actual in zip(expected_params, actual_params):
        test_case.assertEqual(expected.name, actual.name)
        test_case.assertEqual(expected._shape, actual._shape)


class TestRegistryMvRun(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for mv.run() against a service with ParamSpec."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_hf_endpoint = None
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        # Unset HF_ENDPOINT to avoid artifactory errors
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir is not None:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if cls._original_hf_home is not None:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        # These tests require the model logger image that supports params.
        # Skip if not running with image override (i.e., using the latest built images).
        if not self._has_image_override():
            self.skipTest("Skipping model logger tests: requires image override with updated model logger.")

    def _run_test(self, include_params_in_df: bool, params: Optional[dict[str, Any]]) -> None:
        """Run the test with given configuration."""
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            download_snapshot=False,
        )

        input_df = _get_input_df(include_params_in_df)

        def check_res(res: pd.DataFrame) -> None:
            _check_openai_response(self, res)

        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={"__call__": (input_df, check_res)},
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            inference_engine_options=self._get_inference_engine_options_for_inference_engine("vLLM"),
            gpu_requests="1",
            use_model_logging=True,
            signatures=_SIGNATURE,
            params=params,
            skip_rest_api_test=True,
        )

        expected_sig = list(_SIGNATURE.values())[0]
        _verify_signature(self, mv, expected_sig)

    def test_mv_run_full_param_override(self) -> None:
        """Test with all params passed via mv.run(params={...})."""
        self._run_test(include_params_in_df=False, params=_FULL_PARAMS)

    def test_mv_run_partial_param_override(self) -> None:
        """Test with some params passed via mv.run(params={...}), others use defaults."""
        self._run_test(include_params_in_df=False, params=_PARTIAL_PARAMS)

    def test_mv_run_no_param_override(self) -> None:
        """Test with no params passed, use signature defaults."""
        self._run_test(include_params_in_df=False, params=None)

    def test_mv_run_with_params_in_dataframe(self) -> None:
        """Test with params included as columns in the input DataFrame."""
        self._run_test(include_params_in_df=True, params=None)


if __name__ == "__main__":
    absltest.main()
