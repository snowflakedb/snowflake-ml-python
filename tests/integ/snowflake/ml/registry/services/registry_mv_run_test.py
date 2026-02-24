"""Integration tests for mv.run() against a service with OpenAI signatures.

This test validates that mv.run() correctly invokes a deployed service across:
- OpenAI signatures (with/without ParamSpec, string/object content format)
- Inference engines (Default, vLLM)
- Logging methods (local, remote)
- Parameter passing methods (for ParamSpec signatures)
"""

import os
import tempfile
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import (
    ModelVersion,
    compute_pool,
    model_signature,
    openai_signatures,
)
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.models import huggingface
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


def _get_input_df(use_string_content: bool, include_params_in_df: bool) -> pd.DataFrame:
    """Get input DataFrame for the test.

    Args:
        use_string_content: If True, use simple string content format (e.g., "Hello").
            If False, use object content format (e.g., [{"type": "text", "text": "Hello"}]).
            This corresponds to *_WITH_CONTENT_FORMAT_STRING signatures vs standard signatures.
        include_params_in_df: If True, include params as columns in the DataFrame.
    """
    if use_string_content:
        messages = [
            {"role": "system", "content": "Complete the sentence."},
            {"role": "user", "content": "What is the capital of Canada?"},
        ]
    else:
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
    """Integration tests for mv.run() against a service with OpenAI signatures."""

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

    def _run_test(
        self,
        signature: dict[str, model_signature.ModelSignature],
        use_string_content: bool,
        engine: str,
        include_params_in_df: bool,
        params: Optional[dict[str, Any]],
    ) -> None:
        """Run the test with given configuration."""
        model = huggingface.TransformersPipeline(
            task="text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            compute_pool_for_log=None,
        )

        input_df = _get_input_df(use_string_content, include_params_in_df)

        def check_res(res: pd.DataFrame) -> None:
            _check_openai_response(self, res)

        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={"__call__": (input_df, check_res)},
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(engine),
            signatures=signature,
            params=params,
            skip_rest_api_test=True,
        )

        expected_sig = list(signature.values())[0]
        _verify_signature(self, mv, expected_sig)

    # ========================================================================
    # Tests for LOCAL logging with ParamSpec signatures (signature is configurable)
    # ========================================================================

    @parameterized.product(  # type: ignore[misc]
        signature=[
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        ],
        engine=["Default", "vLLM"],
    )
    def test_local_logging_full_param_override(
        self,
        signature: dict[str, model_signature.ModelSignature],
        engine: str,
    ) -> None:
        """Test local logging with all params passed via mv.run(params={...})."""
        use_string_content = signature == openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING
        self._run_test(
            signature=signature,
            use_string_content=use_string_content,
            engine=engine,
            include_params_in_df=False,
            params=_FULL_PARAMS,
        )

    @parameterized.product(  # type: ignore[misc]
        signature=[
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        ],
        engine=["Default", "vLLM"],
    )
    def test_local_logging_partial_param_override(
        self,
        signature: dict[str, model_signature.ModelSignature],
        engine: str,
    ) -> None:
        """Test local logging with some params passed, others use defaults."""
        use_string_content = signature == openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING
        self._run_test(
            signature=signature,
            use_string_content=use_string_content,
            engine=engine,
            include_params_in_df=False,
            params=_PARTIAL_PARAMS,
        )

    @parameterized.product(  # type: ignore[misc]
        signature=[
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        ],
        engine=["Default", "vLLM"],
    )
    def test_local_logging_no_param_override(
        self,
        signature: dict[str, model_signature.ModelSignature],
        engine: str,
    ) -> None:
        """Test local logging with no params passed, use signature defaults."""
        use_string_content = signature == openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING
        self._run_test(
            signature=signature,
            use_string_content=use_string_content,
            engine=engine,
            include_params_in_df=False,
            params=None,
        )

    @parameterized.product(  # type: ignore[misc]
        signature=[
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        ],
        engine=["Default", "vLLM"],
    )
    def test_local_logging_params_in_dataframe(
        self,
        signature: dict[str, model_signature.ModelSignature],
        engine: str,
    ) -> None:
        """Test local logging with params included as columns in the input DataFrame."""
        use_string_content = signature == openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING
        self._run_test(
            signature=signature,
            use_string_content=use_string_content,
            engine=engine,
            include_params_in_df=True,
            params=None,
        )

    # ========================================================================
    # Negative test cases - Invalid params and type mismatches
    # ========================================================================

    @parameterized.product(  # type: ignore[misc]
        compute_pool_for_log=[
            compute_pool.DEFAULT_CPU_COMPUTE_POOL,
            None,
        ],
    )
    def test_invalid_params(self, compute_pool_for_log: Optional[str]) -> None:
        """Test that mv.run raises errors for unknown params and type mismatches.

        Parameterized by compute_pool_for_log to cover both local (None) and
        remote (DEFAULT_CPU_COMPUTE_POOL) logging paths. The model is logged once
        per compute_pool_for_log value, then deployed with both Default and vLLM
        engines.
        """
        self._run_invalid_params_test(compute_pool_for_log=compute_pool_for_log)

    def _run_invalid_params_test(self, compute_pool_for_log: Optional[str]) -> None:
        """Log one model and deploy with each engine to test invalid params.

        Args:
            compute_pool_for_log: Compute pool for logging. None for local logging,
                a pool name for remote logging via model_logger.
        """
        model = huggingface.TransformersPipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-gpt2-with-chatml-template",
            compute_pool_for_log=compute_pool_for_log,
        )

        input_df = _get_input_df(use_string_content=False, include_params_in_df=False)

        def check_res(res: pd.DataFrame) -> None:
            _check_openai_response(self, res)

        logging_type = "remote" if compute_pool_for_log else "local"
        prediction_assert_fns = {"__call__": (input_df, check_res)}
        mv: Optional[ModelVersion] = None

        service_name = f"service_test_invalid_params_{logging_type}_vLLM_{self._run_id}"

        if mv is None:
            # First engine: log the model and deploy
            mv = self._test_registry_model_deployment(
                model=model,
                prediction_assert_fns=prediction_assert_fns,
                service_name=service_name,
                options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
                gpu_requests="1",
                inference_engine_options=self._get_inference_engine_options_for_inference_engine("vLLM"),
                signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
                params=_FULL_PARAMS,
                skip_rest_api_test=True,
            )
        else:
            # Subsequent engines: reuse the logged model, only deploy
            self._deploy_model_service(
                mv=mv,
                model=None,
                prediction_assert_fns=prediction_assert_fns,
                service_name=service_name,
                gpu_requests="1",
                inference_engine_options=self._get_inference_engine_options_for_inference_engine("vLLM"),
                params=_FULL_PARAMS,
                skip_rest_api_test=True,
            )

            self._assert_invalid_params(mv, input_df, service_name)

    def _assert_invalid_params(self, mv: ModelVersion, input_df: pd.DataFrame, service_name: str) -> None:
        """Assert that mv.run raises errors for unknown params and type mismatches."""
        with self.assertRaisesRegex(ValueError, r"Unknown parameter.*unknown_param"):
            mv.run(
                input_df,
                function_name="__call__",
                service_name=service_name,
                params={"unknown_param": 0.5},
            )

        with self.assertRaisesRegex(ValueError, r"not compatible with dtype"):
            mv.run(
                input_df,
                function_name="__call__",
                service_name=service_name,
                params={"temperature": "not_a_float"},
            )

    # ========================================================================
    # Tests for LOCAL logging with non-ParamSpec signatures (signature is configurable)
    # ========================================================================

    @parameterized.product(  # type: ignore[misc]
        signature=[
            openai_signatures.OPENAI_CHAT_SIGNATURE,
            openai_signatures.OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        ],
        engine=["Default", "vLLM"],
    )
    def test_local_without_params_signature(
        self,
        signature: dict[str, model_signature.ModelSignature],
        engine: str,
    ) -> None:
        """Test local logging with signatures without ParamSpec (params are inputs)."""
        use_string_content = signature == openai_signatures.OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING
        # For non-ParamSpec signatures, params must be in the DataFrame
        self._run_test(
            signature=signature,
            use_string_content=use_string_content,
            engine=engine,
            include_params_in_df=True,
            params=None,
        )

    # ========================================================================
    # Tests for REMOTE logging (signature is hardcoded in model_logger)
    # ========================================================================

    @parameterized.product(  # type: ignore[misc]
        engine=["Default", "vLLM"],
    )
    def test_remote_logging_full_param_override(self, engine: str) -> None:
        """Test remote logging with all params passed via mv.run(params={...})."""
        signature = openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE
        self._run_test(
            signature=signature,
            use_string_content=False,
            engine=engine,
            include_params_in_df=False,
            params=_FULL_PARAMS,
        )

    @parameterized.product(  # type: ignore[misc]
        engine=["Default", "vLLM"],
    )
    def test_remote_logging_partial_param_override(self, engine: str) -> None:
        """Test remote logging with some params passed, others use defaults."""
        signature = openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE
        self._run_test(
            signature=signature,
            use_string_content=False,
            engine=engine,
            include_params_in_df=False,
            params=_PARTIAL_PARAMS,
        )

    @parameterized.product(  # type: ignore[misc]
        engine=["Default", "vLLM"],
    )
    def test_remote_logging_no_param_override(self, engine: str) -> None:
        """Test remote logging with no params passed, use signature defaults."""
        signature = openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE
        self._run_test(
            signature=signature,
            use_string_content=False,
            engine=engine,
            include_params_in_df=False,
            params=None,
        )

    @parameterized.product(  # type: ignore[misc]
        engine=["Default", "vLLM"],
    )
    def test_remote_logging_params_in_dataframe(self, engine: str) -> None:
        """Test remote logging with params included as columns in the input DataFrame."""
        signature = openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE
        self._run_test(
            signature=signature,
            use_string_content=False,
            engine=engine,
            include_params_in_df=True,
            params=None,
        )


if __name__ == "__main__":
    absltest.main()
