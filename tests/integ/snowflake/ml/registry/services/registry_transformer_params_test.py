"""Integration tests for transformer model runtime parameter passing.

Tests that ParamSpec parameters (temperature, n, stop, etc.) are correctly
handled across all invocation paths: mv.run, REST flat, REST split, REST records.

Uses SmolLM2-135M-Instruct (supports both object and string content formats) so both
ParamSpec signatures are tested. Deploys one model per (engine × logging × signature)
combination (no explicit gpu_requests variant), then reuses each across subtests covering
full/partial/default params,
extra columns, and invalid inputs.
"""

import logging
import os
import tempfile
from typing import Any

import pandas as pd
import pytest
import requests
import retrying
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature, openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.compute_pool import DEFAULT_CPU_COMPUTE_POOL
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)

_SMALL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _param_specs(signature: dict[str, model_signature.ModelSignature]) -> list[model_signature.BaseParamSpec]:
    return list(signature["__call__"].params or [])


def _flat_column_specs(signature: dict[str, model_signature.ModelSignature]) -> list[Any]:
    """Columns of a flat row after the row index, in the order the proxy expects.

    The proxy validates a flat row against 1 + len(inputs) + len(params). The OpenAI
    params sit in inputs on some chat signatures and in params on others, so both are
    walked here rather than assuming a particular shape.
    """
    call_signature = signature["__call__"]
    return list(call_signature.inputs) + list(call_signature.params or [])


def _deployed_signature(mv: Any) -> dict[str, model_signature.ModelSignature]:
    """Read back the signature the model version was actually logged with.

    Remote logging goes through SYSTEM$IMPORT_MODEL, which derives the signature
    server-side and ignores the signature requested by the client, so the deployed
    param list can lag the client's openai_signatures. Request payloads must be
    built against this signature rather than the requested one.
    """
    functions = mv.show_functions()
    call_function = next((f for f in functions if f["target_method"] == "__call__"), None)
    if call_function is None:
        raise AssertionError(f"__call__ not found among logged functions: {[f['target_method'] for f in functions]}")
    return {"__call__": call_function["signature"]}


def _params_in_signature(
    signature: dict[str, model_signature.ModelSignature], params: dict[str, Any]
) -> dict[str, Any]:
    """Drop params the deployed signature does not declare.

    Passing a param the signature does not know about is rejected client-side, so
    signatures predating a given param (e.g. response_format) must not receive it.
    """
    declared = {spec.name for spec in _param_specs(signature)}
    return {name: value for name, value in params.items() if name in declared}


# All OpenAI params — n=2 so we can verify params actually reach the model
# (2 choices in response proves n was honoured).
_FULL_PARAMS: dict[str, Any] = {
    "temperature": 0.8,
    "max_completion_tokens": 50,
    "stop": ["<|stop_unused|>"],
    "n": 2,
    "stream": False,
    "top_p": 0.9,
    "frequency_penalty": 0.05,
    "presence_penalty": 0.05,
    "response_format": None,
}

_PARTIAL_PARAMS: dict[str, Any] = {
    "temperature": 0.5,
    "max_completion_tokens": 80,
    "n": 3,
}

_INVALID_NAME_PARAMS: dict[str, Any] = {"unknown_param": 0.5}
_INVALID_TYPE_PARAMS: dict[str, Any] = {"temperature": "not_a_float"}


# ---------------------------------------------------------------------------
# Signatures & messages
# ---------------------------------------------------------------------------

_PARAM_SPEC_SIGNATURES: list[dict[str, model_signature.ModelSignature]] = [
    {"__call__": spec} for spec in openai_signatures._OPENAI_CHAT_SIGNATURE_SPECS if spec.params
]

_OBJECT_CONTENT_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
]

_STRING_CONTENT_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
]


def _is_object_content(sig: dict[str, model_signature.ModelSignature]) -> bool:
    """True when the signature uses object (structured) content format."""
    content_spec = next(s for s in sig["__call__"].inputs[0]._specs if s.name == "content")
    return hasattr(content_spec, "_specs")


def _get_messages(sig: dict[str, model_signature.ModelSignature]) -> list[dict[str, Any]]:
    """Return messages matching the signature's content format (object vs string)."""
    if _is_object_content(sig):
        return _OBJECT_CONTENT_MESSAGES
    return _STRING_CONTENT_MESSAGES


class TestTransformerParamsInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.session.sql("ALTER SESSION SET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL = TRUE").collect()

    def tearDown(self) -> None:
        self.session.sql("ALTER SESSION UNSET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL").collect()
        super().tearDown()

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name

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

    # ========================================================================
    # Small helpers
    # ========================================================================

    def _validate_openai_response(self, res: pd.DataFrame, expected_choices: int = 1, label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        pd.testing.assert_index_equal(
            res.columns,
            pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
            check_order=False,
        )
        self.assertEqual(len(res), 1, f"{tag}Expected single response row")
        for row in res["choices"]:
            self.assertIsInstance(row, list, f"{tag}choices should be a list")
            self.assertEqual(len(row), expected_choices, f"{tag}Expected {expected_choices} choices, got {len(row)}")
            self.assertIn("message", row[0], f"{tag}Missing 'message' in choice")
            self.assertIn("content", row[0]["message"], f"{tag}Missing 'content' in message")

    def _rest_post(self, endpoint: str, payload: dict[str, Any]) -> requests.Response:
        return retrying.retry(
            wait_exponential_multiplier=1000,
            wait_exponential_max=10000,
            stop_max_attempt_number=3,
            retry_on_result=(
                registry_model_deployment_test_base.RegistryModelDeploymentTestBase.retry_if_result_status_retriable
            ),
        )(requests.post)(
            f"https://{endpoint}/__call__",
            json=payload,
            auth=self._get_auth_for_inference(endpoint),
            timeout=60,
        )

    def _assert_rest_ok(self, endpoint: str, payload: dict[str, Any], expected_choices: int, label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        response = self._rest_post(endpoint, payload)
        self.assertEqual(
            response.status_code,
            200,
            f"{tag}Expected 200, got {response.status_code}: {response.text[:300]}",
        )
        res = pd.DataFrame([x[1] for x in response.json()["data"]])
        self._validate_openai_response(res, expected_choices, label=label)

    def _assert_rest_400(self, endpoint: str, payload: dict[str, Any], label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        response = self._rest_post(endpoint, payload)
        self.assertEqual(
            response.status_code,
            400,
            f"{tag}Expected 400, got {response.status_code}: {response.text[:300]}",
        )

    def _flat_payload(
        self,
        signature: dict[str, model_signature.ModelSignature],
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> dict[str, Any]:
        """Flat format: row_index, then one column per signature input and param, in order.

        The proxy rejects any row whose column count differs from the deployed signature,
        so the columns are derived from the signature rather than hardcoded. A column the
        caller does not supply falls back to the spec default, or to None for input
        columns, which the proxy omits so the engine default applies.
        """
        row: list[Any] = [0]
        for spec in _flat_column_specs(signature):
            if spec.name == "messages":
                row.append(messages)
            else:
                row.append(params.get(spec.name, getattr(spec, "default_value", None)))
        return {"data": [row]}

    # ========================================================================
    # Deploy (called once per parameterized test)
    # ========================================================================

    def _deploy(
        self,
        engine: InferenceEngine,
        compute_pool_for_log: str | None,
        signature: dict[str, model_signature.ModelSignature],
    ) -> tuple[Any, str]:
        logging_style = "remote" if compute_pool_for_log else "local"
        deploy_label = f"{engine.name}/{logging_style}"
        logger.info("Deploying model: %s", deploy_label)

        model = huggingface.TransformersPipeline(
            task="text-generation",
            model=_SMALL_MODEL,
            compute_pool_for_log=compute_pool_for_log,
        )
        messages = _get_messages(signature)
        test_input = pd.DataFrame.from_records([{"messages": messages}])
        is_vllm = engine == InferenceEngine.VLLM
        options: dict[str, Any] = {"cuda_version": model_env.DEFAULT_CUDA_VERSION} if is_vllm else {}
        service_compute_pool = self._TEST_GPU_COMPUTE_POOL if is_vllm else None

        def check_result(res: pd.DataFrame) -> None:
            self.assertEqual(len(res), 1, f"[{deploy_label}/deploy] Expected single response row")

        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={"__call__": (test_input, check_result)},
            options=options,
            service_compute_pool=service_compute_pool,
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(engine),
            signatures=signature,
            skip_rest_api_test=True,
        )
        endpoint = self._ensure_ingress_url(mv)
        return mv, endpoint

    # ========================================================================
    # Test groups — each uses subTests so failures pinpoint the exact case
    # ========================================================================

    def _test_mv_run(
        self,
        mv: Any,
        messages: list[dict[str, Any]],
        ctx: str,
        *,
        signature: dict[str, model_signature.ModelSignature],
    ) -> None:
        service_name = mv.list_services().loc[0, "name"]
        input_df = pd.DataFrame.from_records([{"messages": messages}])
        full_params = _params_in_signature(signature, _FULL_PARAMS)

        with self.subTest("mv_run / full"):
            res = mv.run(input_df, function_name="__call__", service_name=service_name, params=full_params)
            self._validate_openai_response(res, expected_choices=2, label=f"{ctx}/mv_run/full")

        with self.subTest("mv_run / partial"):
            res = mv.run(input_df, function_name="__call__", service_name=service_name, params=_PARTIAL_PARAMS)
            self._validate_openai_response(res, expected_choices=3, label=f"{ctx}/mv_run/partial")

        with self.subTest("mv_run / default"):
            res = mv.run(input_df, function_name="__call__", service_name=service_name)
            self._validate_openai_response(res, expected_choices=1, label=f"{ctx}/mv_run/default")

        with self.subTest("mv_run / invalid_name"):
            with self.assertRaisesRegex(ValueError, r"Unknown parameter"):
                mv.run(input_df, function_name="__call__", service_name=service_name, params=_INVALID_NAME_PARAMS)

        with self.subTest("mv_run / invalid_type"):
            with self.assertRaisesRegex(ValueError, r"not compatible with dtype"):
                mv.run(input_df, function_name="__call__", service_name=service_name, params=_INVALID_TYPE_PARAMS)

    def _test_rest_flat(
        self,
        endpoint: str,
        messages: list[dict[str, Any]],
        ctx: str,
        *,
        signature: dict[str, model_signature.ModelSignature],
    ) -> None:
        with self.subTest("rest_flat / full"):
            self._assert_rest_ok(
                endpoint,
                self._flat_payload(signature, messages, **_FULL_PARAMS),
                2,
                f"{ctx}/flat/full",
            )

        with self.subTest("rest_flat / partial"):
            self._assert_rest_ok(
                endpoint,
                self._flat_payload(signature, messages, **_PARTIAL_PARAMS),
                3,
                f"{ctx}/flat/partial",
            )

        with self.subTest("rest_flat / default"):
            self._assert_rest_ok(
                endpoint,
                self._flat_payload(signature, messages),
                1,
                f"{ctx}/flat/default",
            )

        # vLLM Go proxy silently drops invalid types (toFloat64 type assertion fails,
        # field stays nil, vLLM uses its default). No server-side type validation.
        # TODO(SNOW-3186308): Uncomment this test when we have proxy-side type validation.
        # with self.subTest("rest_flat / invalid_type"):
        #     self._assert_rest_400(
        #         endpoint,
        #         self._flat_payload(signature, messages, temperature="not_a_float"),
        #         f"{ctx}/flat/invalid_type",
        #     )

        with self.subTest("rest_flat / too_many_cols"):
            payload = self._flat_payload(signature, messages, **_FULL_PARAMS)
            payload["data"][0].append("extra")
            self._assert_rest_400(endpoint, payload, f"{ctx}/flat/too_many_cols")

        with self.subTest("rest_flat / too_few_cols"):
            self._assert_rest_400(endpoint, {"data": [[0, messages, 0.8]]}, f"{ctx}/flat/too_few_cols")

    def _test_rest_split(
        self,
        endpoint: str,
        messages: list[dict[str, Any]],
        ctx: str,
        *,
        signature: dict[str, model_signature.ModelSignature],
    ) -> None:
        base = {"dataframe_split": {"index": [0], "columns": ["messages"], "data": [[messages]]}}
        full_params = _params_in_signature(signature, _FULL_PARAMS)

        with self.subTest("rest_split / full"):
            self._assert_rest_ok(endpoint, {**base, "params": full_params}, 2, f"{ctx}/split/full")

        with self.subTest("rest_split / partial"):
            self._assert_rest_ok(endpoint, {**base, "params": _PARTIAL_PARAMS}, 3, f"{ctx}/split/partial")

        with self.subTest("rest_split / default"):
            self._assert_rest_ok(endpoint, base, 1, f"{ctx}/split/default")

        with self.subTest("rest_split / extra_cols"):
            self._assert_rest_ok(
                endpoint,
                {
                    "dataframe_split": {
                        "index": [0],
                        "columns": ["messages", "extra_col"],
                        "data": [[messages, "x"]],
                    },
                    "params": full_params,
                    "extra_columns": ["extra_col"],
                },
                2,
                f"{ctx}/split/extra_cols",
            )

        # TODO(SNOW-3186308): Uncomment this test when we have proxy-side type validation.
        # with self.subTest("rest_split / invalid_name"):
        #     self._assert_rest_400(endpoint, {**base, "params": _INVALID_NAME_PARAMS}, f"{ctx}/split/invalid_name")

        # TODO(SNOW-3186308): Uncomment this test when we have proxy-side type validation.
        # with self.subTest("rest_split / invalid_type"):
        #     self._assert_rest_400(endpoint, {**base, "params": _INVALID_TYPE_PARAMS}, f"{ctx}/split/invalid_type")

    def _test_rest_records(
        self,
        endpoint: str,
        messages: list[dict[str, Any]],
        ctx: str,
        *,
        signature: dict[str, model_signature.ModelSignature],
    ) -> None:
        base: dict[str, Any] = {"dataframe_records": [{"messages": messages}]}
        full_params = _params_in_signature(signature, _FULL_PARAMS)

        with self.subTest("rest_records / full"):
            self._assert_rest_ok(endpoint, {**base, "params": full_params}, 2, f"{ctx}/records/full")

        with self.subTest("rest_records / partial"):
            self._assert_rest_ok(endpoint, {**base, "params": _PARTIAL_PARAMS}, 3, f"{ctx}/records/partial")

        with self.subTest("rest_records / default"):
            self._assert_rest_ok(endpoint, base, 1, f"{ctx}/records/default")

        with self.subTest("rest_records / extra_cols"):
            self._assert_rest_ok(
                endpoint,
                {
                    "dataframe_records": [{"messages": messages, "extra_col": "x"}],
                    "params": full_params,
                    "extra_columns": ["extra_col"],
                },
                2,
                f"{ctx}/records/extra_cols",
            )

        # vLLM Go proxy ResolveParams iterates over known paramSpecs only;
        # unknown params are silently ignored, invalid types are silently dropped.
        # TODO(SNOW-3186308): Uncomment this test when we have proxy-side type validation.
        # with self.subTest("rest_records / invalid_name"):
        #     self._assert_rest_400(endpoint, {**base, "params": _INVALID_NAME_PARAMS}, f"{ctx}/records/invalid_name")

        # TODO(SNOW-3186308): Uncomment this test when we have proxy-side type validation.
        # with self.subTest("rest_records / invalid_type"):
        #     self._assert_rest_400(endpoint, {**base, "params": _INVALID_TYPE_PARAMS}, f"{ctx}/records/invalid_type")

    # ========================================================================
    # Entry point — one deployment per (engine × logging × signature)
    # ========================================================================

    @parameterized.named_parameters(  # type: ignore[misc]
        *[
            dict(
                testcase_name=f"{engine.name}_{log}_{fmt}",
                engine=engine,
                compute_pool_for_log=pool,
                signature=sig,
            )
            for engine, pool_pairs in [
                (InferenceEngine.VLLM, [("local", None), ("remote", DEFAULT_CPU_COMPUTE_POOL)]),
                # TODO(jack-douglas_snow): Uncomment this when transformer signatures are supported for PYTHON_GENERIC
                # (InferenceEngine.PYTHON_GENERIC, [("local", None), ("remote", DEFAULT_CPU_COMPUTE_POOL)]),
            ]
            for log, pool in pool_pairs
            for sig, fmt in zip(
                _PARAM_SPEC_SIGNATURES,
                ["object" if _is_object_content(s) else "string" for s in _PARAM_SPEC_SIGNATURES],
            )
        ]
    )
    @pytest.mark.conda_incompatible
    def test_params(
        self,
        engine: InferenceEngine,
        compute_pool_for_log: str | None,
        signature: dict[str, model_signature.ModelSignature],
    ) -> None:
        """Deploy once, then run all param variants across every invocation path."""
        logging_style = "remote" if compute_pool_for_log else "local"
        content_fmt = "object" if _is_object_content(signature) else "string"
        ctx = f"{engine.name}/{logging_style}/{content_fmt}"

        mv, endpoint = self._deploy(engine, compute_pool_for_log, signature)
        messages = _get_messages(signature)
        # Payloads follow the signature the model was logged with, which is not always the
        # one requested above: remote logging derives it server-side.
        deployed_signature = _deployed_signature(mv)

        with self.subTest("mv_run"):
            self._test_mv_run(mv, messages, ctx, signature=deployed_signature)
        with self.subTest("rest_flat"):
            self._test_rest_flat(endpoint, messages, ctx, signature=deployed_signature)
        with self.subTest("rest_split"):
            self._test_rest_split(endpoint, messages, ctx, signature=deployed_signature)
        with self.subTest("rest_records"):
            self._test_rest_records(endpoint, messages, ctx, signature=deployed_signature)


if __name__ == "__main__":
    absltest.main()
