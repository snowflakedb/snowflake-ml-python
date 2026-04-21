"""Backwards compatibility tests for custom model inference with older snowflake-ml-python versions.

Scenario: a customer on an older snowflake-ml-python version logs and deploys a
model. The model artifact bundles or pins that older library version, but the
inference server and proxy images are always the latest release. These tests
verify that the current images correctly serve models logged with older client
versions — catching undocumented or unexpected breaking changes in image code
(inference_server, proxy) before they reach production.

The latest snowflake-ml-python version is already covered by
registry_custom_model_params_test. This suite only tests older versions.

Tested versions:
    - snowflake-ml-python 1.5.0 (early release: broadest backwards compat coverage)
    - snowflake-ml-python 1.18.0 (older release: broader backwards compat coverage)
    - snowflake-ml-python 1.25.0 (pre-ParamSpec: no runtime parameters)
    - snowflake-ml-python 1.26.0 (first version to introduce ParamSpec)

Tested invocation paths:
    - mv.run (SQL service function path)
    - REST flat (external function / positional format)
    - REST split (dataframe_split with params dict)
    - REST records (dataframe_records with params dict)

See model_container_services_deployment/CONTEXT/backwards_compatibility.md
for the full backwards compatibility policy.
"""

import datetime
import logging
import unittest
from typing import Any

import pandas as pd
from absl.testing import absltest
from packaging import version

from snowflake.ml.model import custom_model, model_signature
from tests.integ.snowflake.ml.registry.services import registry_param_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

logger = logging.getLogger(__name__)

_DEFAULT_TIMESTAMP = datetime.datetime(2024, 1, 1, 12, 0, 0)


def _format_timestamp(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _normalize_timestamp(value: Any) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        return datetime.datetime.fromisoformat(value)
    raise ValueError(f"Cannot convert {type(value)} to datetime")


def _serialize_for_rest(params: dict[str, Any]) -> dict[str, Any]:
    """Convert native Python params to JSON-serializable format for REST payloads."""
    result = {}
    for k, v in params.items():
        if isinstance(v, bytes):
            result[k] = v.hex()
        elif isinstance(v, datetime.datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


# ===========================================================================
# 1.26.0 param test constants
# ===========================================================================

_FULL_PARAMS: dict[str, Any] = {
    "temperature": 2.5,
    "max_tokens": 200,
    "label": "custom",
    "verbose": True,
    "tag": b"hello",
    "created_at": datetime.datetime(2025, 6, 15, 8, 30, 0),
}

_PARTIAL_PARAMS: dict[str, Any] = {
    "temperature": 0.5,
    "max_tokens": 50,
}

_REST_DEFAULT_PARAMS: dict[str, Any] = _serialize_for_rest(
    {
        "temperature": 1.0,
        "max_tokens": 100,
        "label": "default",
        "verbose": False,
        "tag": b"default",
        "created_at": _DEFAULT_TIMESTAMP,
    }
)

_FULL_EXPECTED: dict[str, Any] = {
    "input_value": 10.0,
    "received_temperature": 2.5,
    "received_max_tokens": 200,
    "received_label": "custom",
    "received_verbose": True,
    "received_tag": b"hello".hex().upper(),
    "received_created_at": _format_timestamp(datetime.datetime(2025, 6, 15, 8, 30, 0)),
}

_PARTIAL_EXPECTED: dict[str, Any] = {
    "input_value": 10.0,
    "received_temperature": 0.5,
    "received_max_tokens": 50,
    "received_label": "default",
    "received_verbose": False,
    "received_tag": b"default".hex().upper(),
    "received_created_at": _format_timestamp(_DEFAULT_TIMESTAMP),
}

_DEFAULT_EXPECTED: dict[str, Any] = {
    "input_value": 10.0,
    "received_temperature": 1.0,
    "received_max_tokens": 100,
    "received_label": "default",
    "received_verbose": False,
    "received_tag": b"default".hex().upper(),
    "received_created_at": _format_timestamp(_DEFAULT_TIMESTAMP),
}


def _to_raw_expected(expected: dict[str, Any], *, bytes_overridden: bool = True) -> dict[str, Any]:
    """Adapt expected output for raw JSON REST response paths.

    mv.run passes bytes as actual bytes objects -> model does .hex().upper() -> uppercase hex.
    REST paths pass bytes as strings:
    - Explicit override: hex string arrives lowercase -> model returns as-is -> lowercase hex.
    - Server-resolved default: Go proxy serializes bytes default as raw string "default".
    """
    result = dict(expected)
    if bytes_overridden:
        result["received_tag"] = result["received_tag"].lower()
    else:
        result["received_tag"] = "default"
    return result


# ===========================================================================
# Models
# ===========================================================================


class SimpleModel(custom_model.CustomModel):
    """Minimal custom model with no runtime parameters.

    Used to test backwards compatibility with pre-ParamSpec versions (e.g. 1.25.0).
    Reports the runtime snowflake-ml-python version so tests can verify
    the model is actually running against the expected pinned version.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        from snowflake.ml import version as snowml_version_module

        return pd.DataFrame(
            {
                "input_value": input_df["value"].tolist(),
                "doubled": (input_df["value"] * 2).tolist(),
                "snowml_version": [snowml_version_module.VERSION] * len(input_df),
            }
        )


class ModelWithScalarParams(custom_model.CustomModel):
    """Custom model covering all scalar ParamSpec types available in 1.26.0.

    Includes float, int, string, bool, bytes, and timestamp.
    Reports the runtime snowflake-ml-python version so tests can verify
    the model is actually running against the expected pinned version.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        temperature: float = 1.0,
        max_tokens: int = 100,
        label: str = "default",
        verbose: bool = False,
        tag: bytes = b"default",
        created_at: datetime.datetime = _DEFAULT_TIMESTAMP,
    ) -> pd.DataFrame:
        from snowflake.ml import version as snowml_version_module

        n = len(input_df)
        tag_str = tag.hex().upper() if isinstance(tag, bytes) else tag
        ts_str = _format_timestamp(_normalize_timestamp(created_at))
        return pd.DataFrame(
            {
                "input_value": input_df["value"].tolist(),
                "received_temperature": [temperature] * n,
                "received_max_tokens": [max_tokens] * n,
                "received_label": [label] * n,
                "received_verbose": [verbose] * n,
                "received_tag": [tag_str] * n,
                "received_created_at": [ts_str] * n,
                "snowml_version": [snowml_version_module.VERSION] * n,
            }
        )


# ===========================================================================
# Test: pre-ParamSpec (1.25.0) — basic inference, no params
# ===========================================================================


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_version() >= version.parse("10.0.0"),
    "Model method signature parameters only available when Snowflake Version >= 10.0.0",
)
class TestBackwardsCompatNoParams(registry_param_test_base.ParamTestBase):
    """Backwards compatibility test for models logged before ParamSpec existed.

    Logs a simple custom model with older snowflake-ml-python versions, deploys
    with the current images, and verifies basic inference works via mv.run and all
    REST formats. Tests both a significantly older version (1.18.0) for broad
    backwards compat coverage and the immediate pre-ParamSpec version (1.25.0)
    to ensure the param infrastructure doesn't break non-param models.
    """

    @staticmethod
    def _get_simple_signature() -> model_signature.ModelSignature:
        return model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="value", dtype=model_signature.DataType.FLOAT)],
            outputs=[
                model_signature.FeatureSpec(name="input_value", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="doubled", dtype=model_signature.DataType.DOUBLE),
                model_signature.FeatureSpec(name="snowml_version", dtype=model_signature.DataType.STRING),
            ],
        )

    def _log_and_deploy_simple_model(self, snowml_version: str) -> Any:
        """Log a simple model with a pinned older snowflake-ml-python version and deploy it."""
        model = SimpleModel(custom_model.ModelContext())
        sig = self._get_simple_signature()
        test_input = pd.DataFrame({"value": [10.0]})

        def check_deploy(res: pd.DataFrame) -> None:
            self.assertEqual(len(res), 1)

        return self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_deploy)},
            pip_requirements=[f"snowflake-ml-python=={snowml_version}"],
            options={"embed_local_ml_library": False},
            skip_rest_api_test=True,
        )

    def _assert_snowml_version(self, mv: Any, expected_version: str) -> None:
        service_name = self._get_service_name(mv)
        input_df = pd.DataFrame({"value": [10.0]})
        res = mv.run(input_df, function_name="predict", service_name=service_name)
        actual_version = res.iloc[0]["snowml_version"]
        self.assertTrue(
            actual_version.startswith(expected_version),
            f"Expected snowml runtime version to start with {expected_version}, got {actual_version}",
        )

    def _check_simple_row(self, row: dict[str, Any], input_value: float, label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        self.assertAlmostEqual(row["input_value"], input_value, places=5, msg=f"{tag}input_value")
        self.assertAlmostEqual(row["doubled"], input_value * 2, places=5, msg=f"{tag}doubled")

    def _test_mv_run_simple(self, mv: Any, ctx: str) -> None:
        service_name = self._get_service_name(mv)
        input_df = pd.DataFrame({"value": [10.0]})

        with self.subTest("mv_run / basic"):
            res = mv.run(input_df, function_name="predict", service_name=service_name)
            self.assertEqual(len(res), 1, f"[{ctx}] Expected single response row")
            self._check_simple_row(res.iloc[0], 10.0, f"{ctx}/mv_run/basic")

    def _test_rest_flat_simple(self, endpoint: str, ctx: str) -> None:
        test_input = pd.DataFrame({"value": [10.0]})
        payload = self._to_external_data_format(test_input)

        with self.subTest("rest_flat / basic"):
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/flat/basic")
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self.assertEqual(len(res_df), 1, f"[{ctx}/flat/basic] Expected single response row")
            self._check_simple_row(res_df.iloc[0], 10.0, f"{ctx}/flat/basic")

    def _test_rest_split_simple(self, endpoint: str, ctx: str) -> None:
        payload = {"dataframe_split": {"index": [0], "columns": ["value"], "data": [[10.0]]}}

        with self.subTest("rest_split / basic"):
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/split/basic")
            row = self._parse_rest_rows(response)[0]
            self._check_simple_row(row, 10.0, f"{ctx}/split/basic")

    def _test_rest_records_simple(self, endpoint: str, ctx: str) -> None:
        payload: dict[str, Any] = {"dataframe_records": [{"value": 10.0}]}

        with self.subTest("rest_records / basic"):
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/records/basic")
            row = self._parse_rest_rows(response)[0]
            self._check_simple_row(row, 10.0, f"{ctx}/records/basic")

    def _run_no_params_test(self, snowml_version: str) -> None:
        """Deploy with a pinned snowflake-ml-python version and verify basic inference."""
        ctx = f"snowml_v{snowml_version}"

        mv = self._log_and_deploy_simple_model(snowml_version)
        endpoint = self._ensure_ingress_url(mv)

        with self.subTest("version_check"):
            self._assert_snowml_version(mv, snowml_version)
        with self.subTest("mv_run"):
            self._test_mv_run_simple(mv, ctx)
        with self.subTest("rest_flat"):
            self._test_rest_flat_simple(endpoint, ctx)
        with self.subTest("rest_split"):
            self._test_rest_split_simple(endpoint, ctx)
        with self.subTest("rest_records"):
            self._test_rest_records_simple(endpoint, ctx)

    def test_no_params_v1_5_0(self) -> None:
        """Deploy with snowflake-ml-python 1.5.0, verify basic inference.

        Tests that the current images correctly serve a model from an early
        SDK version, providing the broadest backwards compatibility coverage.
        """
        self._run_no_params_test("1.5.0")

    def test_no_params_v1_18_0(self) -> None:
        """Deploy with snowflake-ml-python 1.18.0, verify basic inference.

        Tests that the current images correctly serve a model from a significantly
        older SDK version (~6 months old), covering model artifact format changes
        and env handling changes beyond just the ParamSpec boundary.
        """
        self._run_no_params_test("1.18.0")

    def test_no_params_v1_25_0(self) -> None:
        """Deploy with snowflake-ml-python 1.25.0 (pre-ParamSpec), verify basic inference.

        Tests that the current inference server images correctly handle models
        logged immediately before ParamSpec was introduced. Ensures the param
        infrastructure does not break models that have no ParamSpec at all.
        """
        self._run_no_params_test("1.25.0")


# ===========================================================================
# Test: ParamSpec (1.26.0) — param forwarding
# ===========================================================================


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_version() >= version.parse("10.0.0"),
    "Model method signature parameters only available when Snowflake Version >= 10.0.0",
)
class TestBackwardsCompatWithParams(registry_param_test_base.ParamTestBase):
    """Backwards compatibility tests for param forwarding with older snowflake-ml-python versions.

    Logs the model with a pinned older snowflake-ml-python version
    (embed_local_ml_library=False), deploys it with the current images,
    and verifies param handling via mv.run and all REST formats.
    """

    _SNOWML_VERSION = "1.26.0"

    # ===================================================================
    # Signature
    # ===================================================================

    @staticmethod
    def _get_signature() -> model_signature.ModelSignature:
        return model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="value", dtype=model_signature.DataType.FLOAT)],
            outputs=[
                model_signature.FeatureSpec(name="input_value", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="received_temperature", dtype=model_signature.DataType.DOUBLE),
                model_signature.FeatureSpec(name="received_max_tokens", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="received_label", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(name="received_verbose", dtype=model_signature.DataType.BOOL),
                model_signature.FeatureSpec(name="received_tag", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(name="received_created_at", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(name="snowml_version", dtype=model_signature.DataType.STRING),
            ],
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
                model_signature.ParamSpec(name="max_tokens", dtype=model_signature.DataType.INT64, default_value=100),
                model_signature.ParamSpec(name="label", dtype=model_signature.DataType.STRING, default_value="default"),
                model_signature.ParamSpec(name="verbose", dtype=model_signature.DataType.BOOL, default_value=False),
                model_signature.ParamSpec(name="tag", dtype=model_signature.DataType.BYTES, default_value=b"default"),
                model_signature.ParamSpec(
                    name="created_at",
                    dtype=model_signature.DataType.TIMESTAMP_NTZ,
                    default_value=_DEFAULT_TIMESTAMP,
                ),
            ],
        )

    # ===================================================================
    # Logging helper
    # ===================================================================

    def _log_model_pinned(self, snowml_version: str) -> Any:
        """Log with embed_local_ml_library=False and a pinned snowflake-ml-python version."""
        model = ModelWithScalarParams(custom_model.ModelContext())
        sig = self._get_signature()
        test_input = pd.DataFrame({"value": [10.0]})

        def check_deploy(res: pd.DataFrame) -> None:
            self.assertEqual(len(res), 1)

        return self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_deploy)},
            pip_requirements=[f"snowflake-ml-python=={snowml_version}"],
            options={"embed_local_ml_library": False},
            skip_rest_api_test=True,
        )

    # ===================================================================
    # Version validation
    # ===================================================================

    def _assert_snowml_version(self, mv: Any, expected_version: str) -> None:
        service_name = self._get_service_name(mv)
        input_df = pd.DataFrame({"value": [10.0]})
        res = mv.run(input_df, function_name="predict", service_name=service_name)
        actual_version = res.iloc[0]["snowml_version"]
        self.assertTrue(
            actual_version.startswith(expected_version),
            f"Expected snowml runtime version to start with {expected_version}, got {actual_version}",
        )

    # ===================================================================
    # Assertion helpers
    # ===================================================================

    def _check_row(self, row: dict[str, Any], expected: dict[str, Any], label: str = "") -> None:
        tag = f"[{label}] " if label else ""
        self.assertAlmostEqual(row["input_value"], expected["input_value"], places=5, msg=f"{tag}input_value")
        self.assertAlmostEqual(
            row["received_temperature"], expected["received_temperature"], places=5, msg=f"{tag}received_temperature"
        )
        self.assertEqual(row["received_max_tokens"], expected["received_max_tokens"], f"{tag}received_max_tokens")
        self.assertEqual(row["received_label"], expected["received_label"], f"{tag}received_label")
        self.assertEqual(row["received_verbose"], expected["received_verbose"], f"{tag}received_verbose")
        self.assertEqual(row["received_tag"], expected["received_tag"], f"{tag}received_tag")
        self.assertEqual(row["received_created_at"], expected["received_created_at"], f"{tag}received_created_at")

    def _check_df(self, res: pd.DataFrame, expected: dict[str, Any], label: str = "") -> None:
        self._check_row(res.iloc[0], expected, label)

    # ===================================================================
    # Payload builders
    # ===================================================================

    def _flat_payload(self, value: float, params: dict[str, Any]) -> dict[str, Any]:
        """Build flat format payload: {"data": [[row_id, feature, param1, param2, ...]]}."""
        test_input = pd.DataFrame({"value": [value], **{k: [v] for k, v in params.items()}})
        return self._to_external_data_format(test_input)

    # ===================================================================
    # Subtests: mv.run
    # ===================================================================

    def _test_mv_run(self, mv: Any, ctx: str) -> None:
        service_name = self._get_service_name(mv)
        input_df = pd.DataFrame({"value": [10.0]})

        with self.subTest("mv_run / full"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_FULL_PARAMS)
            self._check_df(res, _FULL_EXPECTED, f"{ctx}/mv_run/full")

        with self.subTest("mv_run / partial"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_PARTIAL_PARAMS)
            self._check_df(res, _PARTIAL_EXPECTED, f"{ctx}/mv_run/partial")

        with self.subTest("mv_run / default"):
            res = mv.run(input_df, function_name="predict", service_name=service_name)
            self._check_df(res, _DEFAULT_EXPECTED, f"{ctx}/mv_run/default")

    # ===================================================================
    # Subtests: REST flat
    # ===================================================================

    def _test_rest_flat(self, endpoint: str, ctx: str) -> None:
        with self.subTest("rest_flat / full"):
            flat_params = {**_REST_DEFAULT_PARAMS, **_serialize_for_rest(_FULL_PARAMS)}
            response = self._assert_rest_ok(endpoint, self._flat_payload(10.0, flat_params), label=f"{ctx}/flat/full")
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self._check_df(res_df, _to_raw_expected(_FULL_EXPECTED), f"{ctx}/flat/full")

        with self.subTest("rest_flat / partial"):
            flat_params = {**_REST_DEFAULT_PARAMS, **_serialize_for_rest(_PARTIAL_PARAMS)}
            response = self._assert_rest_ok(
                endpoint, self._flat_payload(10.0, flat_params), label=f"{ctx}/flat/partial"
            )
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self._check_df(res_df, _to_raw_expected(_PARTIAL_EXPECTED), f"{ctx}/flat/partial")

        with self.subTest("rest_flat / default"):
            response = self._assert_rest_ok(
                endpoint, self._flat_payload(10.0, _REST_DEFAULT_PARAMS), label=f"{ctx}/flat/default"
            )
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self._check_df(res_df, _to_raw_expected(_DEFAULT_EXPECTED), f"{ctx}/flat/default")

    # ===================================================================
    # Subtests: REST split
    # ===================================================================

    def _test_rest_split(self, endpoint: str, ctx: str) -> None:
        base = {"dataframe_split": {"index": [0], "columns": ["value"], "data": [[10.0]]}}

        with self.subTest("rest_split / full"):
            payload = {**base, "params": _serialize_for_rest(_FULL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/split/full")
            row = self._parse_rest_rows(response)[0]
            self._check_row(row, _to_raw_expected(_FULL_EXPECTED), f"{ctx}/split/full")

        with self.subTest("rest_split / partial"):
            payload = {**base, "params": _serialize_for_rest(_PARTIAL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/split/partial")
            row = self._parse_rest_rows(response)[0]
            self._check_row(row, _to_raw_expected(_PARTIAL_EXPECTED, bytes_overridden=False), f"{ctx}/split/partial")

        with self.subTest("rest_split / default"):
            response = self._assert_rest_ok(endpoint, base, label=f"{ctx}/split/default")
            row = self._parse_rest_rows(response)[0]
            self._check_row(row, _to_raw_expected(_DEFAULT_EXPECTED, bytes_overridden=False), f"{ctx}/split/default")

    # ===================================================================
    # Subtests: REST records
    # ===================================================================

    def _test_rest_records(self, endpoint: str, ctx: str) -> None:
        base: dict[str, Any] = {"dataframe_records": [{"value": 10.0}]}

        with self.subTest("rest_records / full"):
            payload = {**base, "params": _serialize_for_rest(_FULL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/records/full")
            row = self._parse_rest_rows(response)[0]
            self._check_row(row, _to_raw_expected(_FULL_EXPECTED), f"{ctx}/records/full")

        with self.subTest("rest_records / partial"):
            payload = {**base, "params": _serialize_for_rest(_PARTIAL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label=f"{ctx}/records/partial")
            row = self._parse_rest_rows(response)[0]
            self._check_row(row, _to_raw_expected(_PARTIAL_EXPECTED, bytes_overridden=False), f"{ctx}/records/partial")

        with self.subTest("rest_records / default"):
            response = self._assert_rest_ok(endpoint, base, label=f"{ctx}/records/default")
            row = self._parse_rest_rows(response)[0]
            self._check_row(row, _to_raw_expected(_DEFAULT_EXPECTED, bytes_overridden=False), f"{ctx}/records/default")

    # ===================================================================
    # Entry point
    # ===================================================================

    def test_params_v1_26_0(self) -> None:
        """Deploy with snowflake-ml-python 1.26.0, then verify param handling.

        Tests that the current inference server images correctly handle
        param forwarding for a model logged with the first ParamSpec-capable
        snowflake-ml-python release across all invocation paths.
        """
        ctx = f"snowml_v{self._SNOWML_VERSION}"

        mv = self._log_model_pinned(self._SNOWML_VERSION)
        endpoint = self._ensure_ingress_url(mv)

        with self.subTest("version_check"):
            self._assert_snowml_version(mv, self._SNOWML_VERSION)
        with self.subTest("mv_run"):
            self._test_mv_run(mv, ctx)
        with self.subTest("rest_flat"):
            self._test_rest_flat(endpoint, ctx)
        with self.subTest("rest_split"):
            self._test_rest_split(endpoint, ctx)
        with self.subTest("rest_records"):
            self._test_rest_records(endpoint, ctx)


if __name__ == "__main__":
    absltest.main()
