"""Integration tests for transformers using non-ParamSpec OpenAI signatures.

Non-ParamSpec signatures have no params field; inference parameters (temperature,
n, stop, etc.) are regular input columns in the DataFrame rather than ParamSpec
entries. This is distinct from registry_transformer_params_test which only
exercises ParamSpec signatures.

Signatures are discovered dynamically from _OPENAI_CHAT_SIGNATURE_SPECS by
filtering for specs where ``not spec.params``. If a future change adds ParamSpec
to all signatures, this test will have no parameterized cases and can be removed.

Each deployment is reused across run and REST API subtests.

Deployments: 2 signatures × 2 engines × local logging style = 4
"""

import os
import tempfile
from typing import Any, Optional

import pandas as pd
import requests
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature, openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

_TINY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_FULL_PARAMS: dict[str, Any] = {
    "temperature": 0.9,
    "max_completion_tokens": 20,
    "stop": None,
    "n": 1,
    "stream": False,
    "top_p": 1.0,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
}

_OBJECT_CONTENT_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": [{"type": "text", "text": "Complete the sentence."}]},
    {"role": "user", "content": [{"type": "text", "text": "What is the capital of Canada?"}]},
]

_STRING_CONTENT_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Complete the sentence."},
    {"role": "user", "content": "What is the capital of Canada?"},
]

# ---------------------------------------------------------------------------
# Discover non-ParamSpec signatures dynamically so this test auto-adjusts
# when openai_signatures adds ParamSpec to previously non-ParamSpec specs.
# ---------------------------------------------------------------------------

_NON_PARAM_SPEC_SIGNATURES: list[dict[str, model_signature.ModelSignature]] = [
    {"__call__": spec} for spec in openai_signatures._OPENAI_CHAT_SIGNATURE_SPECS if not spec.params
]


def _is_object_content(sig: dict[str, model_signature.ModelSignature]) -> bool:
    """True when the signature uses structured object content format."""
    messages_spec = sig["__call__"].inputs[0]
    assert isinstance(messages_spec, model_signature.FeatureGroupSpec), "Expected messages to be a FeatureGroupSpec"
    content_spec = next(s for s in messages_spec._specs if s.name == "content")
    return isinstance(content_spec, model_signature.FeatureGroupSpec)


def _get_messages(sig: dict[str, model_signature.ModelSignature]) -> list[dict[str, Any]]:
    """Return messages matching the signature's content format."""
    if _is_object_content(sig):
        return _OBJECT_CONTENT_MESSAGES
    return _STRING_CONTENT_MESSAGES


class TestRegistryTransformerNonParamSpecSignatureInteg(
    registry_model_deployment_test_base.RegistryModelDeploymentTestBase
):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_hf_endpoint = None
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
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

    # ========================================================================
    # Helpers
    # ========================================================================

    def _validate_openai_response(self, res: pd.DataFrame) -> None:
        pd.testing.assert_index_equal(
            res.columns,
            pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
            check_order=False,
        )
        for row in res["choices"]:
            self.assertIsInstance(row, list)
            self.assertIn("message", row[0])
            self.assertIn("content", row[0]["message"])

    def _verify_signature(
        self,
        mv: object,
        expected_signature: model_signature.ModelSignature,
    ) -> None:
        """Verify the logged model's signature matches expected (critical for stop param array bug)."""
        functions = mv.show_functions()  # type: ignore[attr-defined]
        func_info = next((f for f in functions if f["target_method"] == "__call__"), None)
        self.assertIsNotNone(func_info, "Method __call__ not found")
        actual_sig = func_info["signature"]

        self.assertEqual(len(actual_sig.inputs), len(expected_signature.inputs))

        expected_params = expected_signature.params or []
        actual_params = actual_sig.params or []
        self.assertEqual(len(actual_params), len(expected_params))
        for expected, actual in zip(expected_params, actual_params):
            self.assertEqual(expected.name, actual.name)
            self.assertEqual(expected._shape, actual._shape)

    def _rest_post(self, endpoint: str, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            f"https://{endpoint}/__call__",
            json=payload,
            auth=self._get_auth_for_inference(endpoint),
            timeout=60,
        )

    def _flat_payload(self, messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        """FLAT format: row_index followed by all input columns in signature order."""
        return {
            "data": [
                [
                    0,
                    messages,
                    params.get("temperature", 1.0),
                    params.get("max_completion_tokens", 250),
                    params.get("stop", None),
                    params.get("n", 1),
                    params.get("stream", False),
                    params.get("top_p", 1.0),
                    params.get("frequency_penalty", 0.0),
                    params.get("presence_penalty", 0.0),
                ]
            ]
        }

    def _assert_rest_ok(self, endpoint: str, payload: dict[str, Any], label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        response = self._rest_post(endpoint, payload)
        self.assertEqual(
            response.status_code,
            200,
            f"{tag}Expected 200, got {response.status_code}: {response.text[:300]}",
        )
        res = pd.DataFrame([x[1] for x in response.json()["data"]])
        self._validate_openai_response(res)

    def _assert_rest_400(self, endpoint: str, payload: dict[str, Any], label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        response = self._rest_post(endpoint, payload)
        self.assertEqual(
            response.status_code,
            400,
            f"{tag}Expected 400, got {response.status_code}: {response.text[:300]}",
        )

    # ========================================================================
    # Non-ParamSpec signatures
    # Parameterized by: signature × engine × logging style
    # ========================================================================

    @parameterized.named_parameters(  # type: ignore[misc]
        *[
            dict(
                testcase_name=f"{engine}_{log}_{fmt}",
                engine=engine,
                compute_pool_for_log=pool,
                signature=sig,
            )
            for engine in ["Default", "vLLM"]
            for log, pool in [("local", None)]  # Remote logging hardcodes the signature
            for sig, fmt in zip(
                _NON_PARAM_SPEC_SIGNATURES,
                ["object" if _is_object_content(s) else "string" for s in _NON_PARAM_SPEC_SIGNATURES],
            )
        ]
    )
    def test_non_param_spec_signature(
        self,
        engine: str,
        compute_pool_for_log: Optional[str],
        signature: dict[str, model_signature.ModelSignature],
    ) -> None:
        """Test non-ParamSpec signatures where params are passed as input columns."""
        messages = _get_messages(signature)
        input_df = pd.DataFrame.from_records([{"messages": messages, **_FULL_PARAMS}])
        logging_style = "remote" if compute_pool_for_log else "local"
        content_fmt = "object" if _is_object_content(signature) else "string"
        ctx = f"{engine}/{logging_style}/{content_fmt}"

        model = huggingface.TransformersPipeline(
            task="text-generation",
            model=_TINY_MODEL,
            compute_pool_for_log=compute_pool_for_log,
        )

        def check_res(res: pd.DataFrame) -> None:
            self._validate_openai_response(res)

        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={"__call__": (input_df, check_res)},
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(engine),
            signatures=signature,
            skip_rest_api_test=True,
        )

        endpoint = self._ensure_ingress_url(mv)

        with self.subTest("signature_verification"):
            self._verify_signature(mv, list(signature.values())[0])

        with self.subTest("rest_flat / full"):
            self._assert_rest_ok(endpoint, self._flat_payload(messages, **_FULL_PARAMS), f"{ctx}/flat/full")

        with self.subTest("rest_flat / defaults"):
            self._assert_rest_ok(endpoint, self._flat_payload(messages), f"{ctx}/flat/defaults")

        # TODO(SNOW-3186308): Uncomment this test when we have proxy-side type validation.
        # with self.subTest("rest_flat / invalid_type"):
        #     self._assert_rest_400(
        #         endpoint, self._flat_payload(messages, temperature="not_a_float"), f"{ctx}/flat/invalid_type"
        #     )

        with self.subTest("rest_flat / too_many_cols"):
            row = [0, messages, 0.9, 20, None, 1, False, 1.0, 0.1, 0.2, "extra"]
            self._assert_rest_400(endpoint, {"data": [row]}, f"{ctx}/flat/too_many_cols")

        with self.subTest("rest_flat / too_few_cols"):
            self._assert_rest_400(endpoint, {"data": [[0, messages, 0.9]]}, f"{ctx}/flat/too_few_cols")


if __name__ == "__main__":
    absltest.main()
