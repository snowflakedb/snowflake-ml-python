"""Integration tests for custom model runtime parameter passing.

Tests that ParamSpec parameters of all supported data types are correctly
handled across all invocation paths: mv.run, REST flat, REST split, REST records,
and REST wide (500+ features).

Deploys one model per test method, then reuses each across subtests covering
full/partial/default/none params, error cases (invalid_name, invalid_type,
too_many_cols, too_few_cols, extra_cols).

Coverage matrix (38 subtests + 7 TODO, 2 deployments):

    Invocation   | full | partial | default | none | invalid_name | invalid_type | too_many | too_few | extra_cols
    -------------|------|---------|---------|------|--------------|--------------|----------|---------|----------
    mv.run       |  Y   |   Y     |   Y     |  Y   | Y (VE)       |   Y (VE)     |   N/A    |  N/A    |   N/A
    REST flat    |  Y   |   Y     |   Y     |  Y   |    N/A       |   Y (400)    |  Y (400) | Y (400) |   N/A
    REST split   |  Y   |   Y     |   Y     |  Y   |   TODO†      |   Y (400)    |  TODO†   |  N/A    | Y (200)
    REST records |  Y   |   Y     |   Y     |  Y   |   TODO†      |   Y (400)    |  TODO†   |  N/A    | Y (200)
    REST wide    |  Y   |   Y     |   Y     |  Y   |   TODO†      |   Y (400)    |  TODO†   |  N/A    |   N/A

    † = commented out — server currently ignores silently instead of returning 400

    Additional edge cases:
    - mv.run / multi_row_with_params: params applied consistently across all rows
    - REST flat / varying_params_across_rows: TODO† — should reject varying rows
    - REST wide / trailing_positional_params: params after feature dict use defaults
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

# ---------------------------------------------------------------------------
# Defaults used by ModelWithAllDataTypes
# ---------------------------------------------------------------------------

_DEFAULT_TIMESTAMP = datetime.datetime(2024, 1, 1, 12, 0, 0)
_DEFAULT_WEIGHTS = [1.0, 2.0, 3.0]
_DEFAULT_NESTED_LIST = [[1, 2], [3, 4]]
_DEFAULT_MODELS = [{"name": "default", "weight": 1}, {"name": "default", "weight": 1}]


def _format_timestamp(dt: datetime.datetime) -> str:
    """Format timestamp with milliseconds for consistent output.

    Snowflake returns timestamps with milliseconds, so we need to match that format.
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _normalize_timestamp(value: Any) -> datetime.datetime:
    """Convert timestamp value to datetime, handling both datetime objects and ISO strings.

    This handles the difference between SQL path (receives datetime) and REST path (receives string).
    """
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        # Try parsing ISO format strings (handles both 'T' and space separators)
        return datetime.datetime.fromisoformat(value)
    raise ValueError(f"Cannot convert {type(value)} to datetime")


_DEFAULT_TIMESTAMP_STR = _format_timestamp(_DEFAULT_TIMESTAMP)

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_for_rest(params: dict[str, Any]) -> dict[str, Any]:
    """Convert native Python params to JSON-serializable format for REST payloads.

    bytes -> hex string, datetime -> ISO format string, everything else passed through.
    """
    result = {}
    for k, v in params.items():
        if isinstance(v, bytes):
            result[k] = v.hex()
        elif isinstance(v, datetime.datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


def _to_raw_expected(expected: dict[str, Any], bytes_overridden: bool = True) -> dict[str, Any]:
    """Adapt expected output for raw JSON response paths (REST flat/split/records).

    mv.run passes bytes as actual bytes objects → model does .hex().upper() → uppercase hex.
    REST paths pass bytes as strings:
    - Explicit override: hex string arrives lowercase → model returns as-is → lowercase hex.
    - Server-resolved default: Go proxy serializes bytes default as raw string "default".
    """
    result = dict(expected)
    if "received_bytes" in result and isinstance(result["received_bytes"], str):
        if bytes_overridden:
            result["received_bytes"] = result["received_bytes"].lower()
        else:
            # Server default: Go proxy passes raw string, not hex
            result["received_bytes"] = "default"
    return result


# ---------------------------------------------------------------------------
# AllDataTypes: param input variants (native Python types, used by mv.run)
# REST paths derive JSON-serializable versions via _serialize_for_rest().
# ---------------------------------------------------------------------------

_FULL_PARAMS: dict[str, Any] = {
    "int8_param": 10,
    "int16_param": 200,
    "int32_param": 3000,
    "int64_param": 40000,
    "uint8_param": 15,
    "uint16_param": 300,
    "uint32_param": 4000,
    "uint64_param": 50000,
    "float_param": 1.25,
    "double_param": 2.75,
    "bool_param": False,
    "string_param": "custom_value",
    "bytes_param": b"hello",
    "timestamp_param": datetime.datetime(2025, 1, 2, 3, 4, 5),
    "weights_param": [4.5, 3.5, 2.5],
    "nested_list": [[4, 3], [2, 1]],
    "config": {
        "temperature": 2.0,
        "top_k": 10,
        "use_cache": False,
        "stop_token": "STOP",
        "penalties": [0.9, 0.1],
        "sampling": {"seed": 123, "strategy": "beam"},
        "models": [{"name": "fast", "weight": 10}, {"name": "slow", "weight": 5}],
    },
}

_PARTIAL_PARAMS: dict[str, Any] = {
    "timestamp_param": datetime.datetime(2025, 1, 2, 3, 4, 5),
    "weights_param": [4.5, 3.5, 2.5],
    "nested_list": [[4, 3], [2, 1]],
    # Deep merge: override temperature at top level + strategy in nested sampling dict.
    # Unspecified keys (top_k, use_cache, stop_token, penalties, sampling.seed) keep defaults.
    "config": {
        "temperature": 3.0,
        "sampling": {"strategy": "beam"},
    },
}

_NONE_PARAMS: dict[str, Any] = {
    "int8_param": None,
    "int16_param": None,
    "int32_param": None,
    "int64_param": None,
    "uint8_param": None,
    "uint16_param": None,
    "uint32_param": None,
    "uint64_param": None,
    "float_param": None,
    "double_param": None,
    "bool_param": None,
    "string_param": None,
    "bytes_param": None,
    "timestamp_param": None,
    "weights_param": None,
    "nested_list": None,
    "config": None,
}

_INVALID_NAME_PARAMS: dict[str, Any] = {"unknown_param": 42}
_INVALID_TYPE_PARAMS: dict[str, Any] = {"int8_param": "not_an_int"}

# ---------------------------------------------------------------------------
# AllDataTypes: expected outputs (DataFrame/uppercase hex format).
# Raw JSON paths derive lowercase hex via _to_raw_expected().
# ---------------------------------------------------------------------------

_DEFAULT_EXPECTED: dict[str, Any] = {
    "input_value": 10.0,
    "received_int8": 1,
    "received_int16": 2,
    "received_int32": 3,
    "received_int64": 4,
    "received_uint8": 5,
    "received_uint16": 6,
    "received_uint32": 7,
    "received_uint64": 8,
    "received_float": 1.5,
    "received_double": 2.5,
    "received_bool": True,
    "received_string": "default",
    # TODO (SNOW-3045092): Fix byte output serialization issue
    "received_bytes": b"default".hex().upper(),  # hex of b"default"
    # TODO (SNOW-3045092): Fix timestamp output serialization issue
    "received_timestamp": _DEFAULT_TIMESTAMP_STR,
    "received_weights": _DEFAULT_WEIGHTS,
    "received_nested_list": _DEFAULT_NESTED_LIST,
    "received_temperature": 1.0,
    "received_top_k": 50,
    "received_use_cache": True,
    "received_stop_token": "END",
    "received_penalties": [0.5, 0.3],
    "received_seed": 42,
    "received_strategy": "greedy",
    "received_models": _DEFAULT_MODELS,
}

_FULL_EXPECTED: dict[str, Any] = {
    "input_value": 10.0,
    "received_int8": 10,
    "received_int16": 200,
    "received_int32": 3000,
    "received_int64": 40000,
    "received_uint8": 15,
    "received_uint16": 300,
    "received_uint32": 4000,
    "received_uint64": 50000,
    "received_float": 1.25,
    "received_double": 2.75,
    "received_bool": False,
    "received_string": "custom_value",
    # Inference server converts bytes to uppercase hex string for output
    # TODO (SNOW-3045092): Fix byte output serialization issue
    "received_bytes": b"hello".hex().upper(),  # hex of b"hello"
    # TODO (SNOW-3045092): Fix timestamp output serialization issue
    "received_timestamp": _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5)),
    "received_weights": [4.5, 3.5, 2.5],
    "received_nested_list": [[4, 3], [2, 1]],
    "received_temperature": 2.0,
    "received_top_k": 10,
    "received_use_cache": False,
    "received_stop_token": "STOP",
    "received_penalties": [0.9, 0.1],
    "received_seed": 123,
    "received_strategy": "beam",
    "received_models": [{"name": "fast", "weight": 10}, {"name": "slow", "weight": 5}],
}

_PARTIAL_EXPECTED: dict[str, Any] = {
    **_DEFAULT_EXPECTED,
    "received_timestamp": _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5)),
    "received_weights": [4.5, 3.5, 2.5],
    "received_nested_list": [[4, 3], [2, 1]],
    # Deep merge results: temperature overridden, sampling.strategy overridden,
    # everything else (top_k, use_cache, stop_token, penalties, sampling.seed) keeps defaults
    "received_temperature": 3.0,
    "received_top_k": 50,
    "received_use_cache": True,
    "received_stop_token": "END",
    "received_penalties": [0.5, 0.3],
    "received_seed": 42,
    "received_strategy": "beam",
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 50,
    "use_cache": True,
    "stop_token": "END",
    "penalties": [0.5, 0.3],
    "sampling": {"seed": 42, "strategy": "greedy"},
    "models": _DEFAULT_MODELS,
}

# JSON-serializable version of all param defaults (bytes as hex, datetime as ISO).
# Flat format requires ALL columns present, so this is used to fill missing columns.
_REST_DEFAULT_PARAMS: dict[str, Any] = {
    **_serialize_for_rest(
        {
            "int8_param": 1,
            "int16_param": 2,
            "int32_param": 3,
            "int64_param": 4,
            "uint8_param": 5,
            "uint16_param": 6,
            "uint32_param": 7,
            "uint64_param": 8,
            "float_param": 1.5,
            "double_param": 2.5,
            "bool_param": True,
            "string_param": "default",
            "bytes_param": b"default",
            "timestamp_param": _DEFAULT_TIMESTAMP,
            "weights_param": _DEFAULT_WEIGHTS,
            "nested_list": _DEFAULT_NESTED_LIST,
        }
    ),
    "config": _DEFAULT_CONFIG,
}

# ---------------------------------------------------------------------------
# Wide model param variants (different model, different signature)
# ---------------------------------------------------------------------------

_WIDE_FORMAT_NUM_FEATURES = 501  # Must exceed 500 to trigger wide format at deployment time
_WIDE_FEATURE_SUM = float(_WIDE_FORMAT_NUM_FEATURES)  # Each feature is 1.0

_WIDE_FULL_PARAMS: dict[str, Any] = {"multiplier": 2.0, "offset": 100}
_WIDE_PARTIAL_PARAMS: dict[str, Any] = {"multiplier": 3.0}
_WIDE_NONE_PARAMS: dict[str, Any] = {"multiplier": None, "offset": None}
_WIDE_INVALID_NAME_PARAMS: dict[str, Any] = {"unknown_param": 42}
_WIDE_INVALID_TYPE_PARAMS: dict[str, Any] = {"multiplier": "not_a_float"}

_WIDE_FULL_EXPECTED: dict[str, Any] = {
    "feature_sum": _WIDE_FEATURE_SUM,
    "output": _WIDE_FEATURE_SUM * 2.0 + 100,
    "received_multiplier": 2.0,
    "received_offset": 100,
}

_WIDE_PARTIAL_EXPECTED: dict[str, Any] = {
    "feature_sum": _WIDE_FEATURE_SUM,
    "output": _WIDE_FEATURE_SUM * 3.0,
    "received_multiplier": 3.0,
    "received_offset": 0,
}

_WIDE_DEFAULT_EXPECTED: dict[str, Any] = {
    "feature_sum": _WIDE_FEATURE_SUM,
    "output": _WIDE_FEATURE_SUM,
    "received_multiplier": 1.0,
    "received_offset": 0,
}


# ===========================================================================
# Model classes
# ===========================================================================


class ModelWithAllDataTypes(custom_model.CustomModel):
    """A custom model that accepts params of all supported data types."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        # Signed integers
        int8_param: int = 1,
        int16_param: int = 2,
        int32_param: int = 3,
        int64_param: int = 4,
        # Unsigned integers
        uint8_param: int = 5,
        uint16_param: int = 6,
        uint32_param: int = 7,
        uint64_param: int = 8,
        # Floating point
        float_param: float = 1.5,
        double_param: float = 2.5,
        # Other scalars
        bool_param: bool = True,
        string_param: str = "default",
        # Extended types
        bytes_param: bytes = b"default",
        timestamp_param: datetime.datetime = _DEFAULT_TIMESTAMP,
        weights_param: list[float] = _DEFAULT_WEIGHTS,
        nested_list: list[list[int]] = _DEFAULT_NESTED_LIST,
        config: dict = _DEFAULT_CONFIG,  # noqa: B006
    ) -> pd.DataFrame:
        n = len(input)
        bytes_val = bytes_param.hex().upper() if isinstance(bytes_param, bytes) else bytes_param
        ts_val = _format_timestamp(_normalize_timestamp(timestamp_param))
        cfg = config or {}
        sampling = cfg.get("sampling", {}) or {}
        models = cfg.get("models", _DEFAULT_MODELS)
        return pd.DataFrame(
            {
                "input_value": input["value"].tolist(),
                "received_int8": [int8_param] * n,
                "received_int16": [int16_param] * n,
                "received_int32": [int32_param] * n,
                "received_int64": [int64_param] * n,
                "received_uint8": [uint8_param] * n,
                "received_uint16": [uint16_param] * n,
                "received_uint32": [uint32_param] * n,
                "received_uint64": [uint64_param] * n,
                "received_float": [float_param] * n,
                "received_double": [double_param] * n,
                "received_bool": [bool_param] * n,
                "received_string": [string_param] * n,
                # TODO (SNOW-3045092): Fix byte output serialization issue
                "received_bytes": [bytes_val] * n,
                # TODO (SNOW-3045092): Fix timestamp output serialization issue
                "received_timestamp": [ts_val] * n,
                "received_weights": [weights_param] * n,
                "received_nested_list": [nested_list] * n,
                "received_temperature": [cfg.get("temperature", 1.0)] * n,
                "received_top_k": [cfg.get("top_k", 50)] * n,
                "received_use_cache": [cfg.get("use_cache", True)] * n,
                "received_stop_token": [cfg.get("stop_token", "END")] * n,
                "received_penalties": [cfg.get("penalties", [0.5, 0.3])] * n,
                "received_seed": [sampling.get("seed", 42)] * n,
                "received_strategy": [sampling.get("strategy", "greedy")] * n,
                "received_models": [models] * n,
            }
        )


class ModelWithManyFeatures(custom_model.CustomModel):
    """A custom model with 500+ features to test wide format.

    Wide format is triggered when the model has 500+ features.
    The model sums all feature values and applies the multiplier and offset params.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        multiplier: float = 1.0,
        offset: int = 0,
    ) -> pd.DataFrame:
        if multiplier is None:
            multiplier = 0.0
        if offset is None:
            offset = 0

        # Sum all feature columns (f_0 through f_499)
        feature_sum = input.sum(axis=1).iloc[0]
        output_value = feature_sum * multiplier + offset

        return pd.DataFrame(
            {
                "output": [output_value],
                "feature_sum": [feature_sum],
                "received_multiplier": [multiplier],
                "received_offset": [offset],
            }
        )


# ===========================================================================
# Test class
# ===========================================================================


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_version() >= version.parse("10.0.0"),
    "Model method signature parameters only available when Snowflake Version >= 10.0.0",
)
class TestRegistryCustomModelParamsInteg(registry_param_test_base.ParamTestBase):
    """Integration tests for custom model inference with runtime parameters.

    Deploy each model once, then reuse across subtests covering every
    (invocation_path x param_variant) combination.
    """

    # ===================================================================
    # Signatures
    # ===================================================================

    def _get_all_data_types_signature(self) -> model_signature.ModelSignature:
        return model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="value", dtype=model_signature.DataType.FLOAT)],
            outputs=[
                model_signature.FeatureSpec(name="input_value", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="received_int8", dtype=model_signature.DataType.INT8),
                model_signature.FeatureSpec(name="received_int16", dtype=model_signature.DataType.INT16),
                model_signature.FeatureSpec(name="received_int32", dtype=model_signature.DataType.INT32),
                model_signature.FeatureSpec(name="received_int64", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="received_uint8", dtype=model_signature.DataType.UINT8),
                model_signature.FeatureSpec(name="received_uint16", dtype=model_signature.DataType.UINT16),
                model_signature.FeatureSpec(name="received_uint32", dtype=model_signature.DataType.UINT32),
                model_signature.FeatureSpec(name="received_uint64", dtype=model_signature.DataType.UINT64),
                model_signature.FeatureSpec(name="received_float", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="received_double", dtype=model_signature.DataType.DOUBLE),
                model_signature.FeatureSpec(name="received_bool", dtype=model_signature.DataType.BOOL),
                model_signature.FeatureSpec(name="received_string", dtype=model_signature.DataType.STRING),
                # TODO (SNOW-3045092): Fix byte output serialization issue
                model_signature.FeatureSpec(name="received_bytes", dtype=model_signature.DataType.STRING),
                # TODO (SNOW-3045092): Fix timestamp output serialization issue
                model_signature.FeatureSpec(name="received_timestamp", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(name="received_weights", dtype=model_signature.DataType.DOUBLE, shape=(3,)),
                model_signature.FeatureSpec(
                    name="received_nested_list", dtype=model_signature.DataType.INT64, shape=(2, 2)
                ),
                model_signature.FeatureSpec(name="received_temperature", dtype=model_signature.DataType.DOUBLE),
                model_signature.FeatureSpec(name="received_top_k", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="received_use_cache", dtype=model_signature.DataType.BOOL),
                model_signature.FeatureSpec(name="received_stop_token", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(
                    name="received_penalties", dtype=model_signature.DataType.DOUBLE, shape=(2,)
                ),
                model_signature.FeatureSpec(name="received_seed", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="received_strategy", dtype=model_signature.DataType.STRING),
                model_signature.FeatureGroupSpec(
                    name="received_models",
                    specs=[
                        model_signature.FeatureSpec(name="name", dtype=model_signature.DataType.STRING),
                        model_signature.FeatureSpec(name="weight", dtype=model_signature.DataType.INT64),
                    ],
                    shape=(2,),
                ),
            ],
            params=[
                model_signature.ParamSpec(name="int8_param", dtype=model_signature.DataType.INT8, default_value=1),
                model_signature.ParamSpec(name="int16_param", dtype=model_signature.DataType.INT16, default_value=2),
                model_signature.ParamSpec(name="int32_param", dtype=model_signature.DataType.INT32, default_value=3),
                model_signature.ParamSpec(name="int64_param", dtype=model_signature.DataType.INT64, default_value=4),
                model_signature.ParamSpec(name="uint8_param", dtype=model_signature.DataType.UINT8, default_value=5),
                model_signature.ParamSpec(name="uint16_param", dtype=model_signature.DataType.UINT16, default_value=6),
                model_signature.ParamSpec(name="uint32_param", dtype=model_signature.DataType.UINT32, default_value=7),
                model_signature.ParamSpec(name="uint64_param", dtype=model_signature.DataType.UINT64, default_value=8),
                model_signature.ParamSpec(name="float_param", dtype=model_signature.DataType.FLOAT, default_value=1.5),
                model_signature.ParamSpec(
                    name="double_param", dtype=model_signature.DataType.DOUBLE, default_value=2.5
                ),
                model_signature.ParamSpec(name="bool_param", dtype=model_signature.DataType.BOOL, default_value=True),
                model_signature.ParamSpec(
                    name="string_param", dtype=model_signature.DataType.STRING, default_value="default"
                ),
                model_signature.ParamSpec(
                    name="bytes_param", dtype=model_signature.DataType.BYTES, default_value=b"default"
                ),
                model_signature.ParamSpec(
                    name="timestamp_param",
                    dtype=model_signature.DataType.TIMESTAMP_NTZ,
                    default_value=_DEFAULT_TIMESTAMP,
                ),
                model_signature.ParamSpec(
                    name="weights_param",
                    dtype=model_signature.DataType.DOUBLE,
                    default_value=list(_DEFAULT_WEIGHTS),
                    shape=(3,),
                ),
                model_signature.ParamSpec(
                    name="nested_list",
                    dtype=model_signature.DataType.INT64,
                    default_value=_DEFAULT_NESTED_LIST,
                    shape=(2, 2),
                ),
                model_signature.ParamGroupSpec(
                    name="config",
                    specs=[
                        model_signature.ParamSpec(
                            name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=1.0
                        ),
                        model_signature.ParamSpec(name="top_k", dtype=model_signature.DataType.INT64, default_value=50),
                        model_signature.ParamSpec(
                            name="use_cache", dtype=model_signature.DataType.BOOL, default_value=True
                        ),
                        model_signature.ParamSpec(
                            name="stop_token", dtype=model_signature.DataType.STRING, default_value="END"
                        ),
                        model_signature.ParamSpec(
                            name="penalties",
                            dtype=model_signature.DataType.DOUBLE,
                            default_value=[0.5, 0.3],
                            shape=(2,),
                        ),
                        model_signature.ParamGroupSpec(
                            name="sampling",
                            specs=[
                                model_signature.ParamSpec(
                                    name="seed", dtype=model_signature.DataType.INT64, default_value=42
                                ),
                                model_signature.ParamSpec(
                                    name="strategy", dtype=model_signature.DataType.STRING, default_value="greedy"
                                ),
                            ],
                        ),
                        model_signature.ParamGroupSpec(
                            name="models",
                            specs=[
                                model_signature.ParamSpec(
                                    name="name", dtype=model_signature.DataType.STRING, default_value="default"
                                ),
                                model_signature.ParamSpec(
                                    name="weight", dtype=model_signature.DataType.INT64, default_value=1
                                ),
                            ],
                            shape=(2,),
                        ),
                    ],
                ),
            ],
        )

    def _get_wide_model_signature(self) -> model_signature.ModelSignature:
        feature_specs = [
            model_signature.FeatureSpec(name=f"f_{i}", dtype=model_signature.DataType.FLOAT)
            for i in range(_WIDE_FORMAT_NUM_FEATURES)
        ]
        return model_signature.ModelSignature(
            inputs=feature_specs,
            outputs=[
                model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="feature_sum", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="received_multiplier", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="received_offset", dtype=model_signature.DataType.INT64),
            ],
            params=[
                model_signature.ParamSpec(name="multiplier", dtype=model_signature.DataType.FLOAT, default_value=1.0),
                model_signature.ParamSpec(name="offset", dtype=model_signature.DataType.INT64, default_value=0),
            ],
        )

    # ===================================================================
    # Assertion helpers
    # ===================================================================

    def _check_all_data_types(self, row: dict[str, Any], expected: dict[str, Any], label: str = "") -> None:
        """Validate every received param value in a response row dict or DataFrame row."""
        tag = f"[{label}] " if label else ""

        self.assertAlmostEqual(row["input_value"], expected["input_value"], places=5, msg=f"{tag}input_value")
        for key in [
            "received_int8",
            "received_int16",
            "received_int32",
            "received_int64",
            "received_uint8",
            "received_uint16",
            "received_uint32",
            "received_uint64",
        ]:
            self.assertEqual(row[key], expected[key], f"{tag}{key}")
        self.assertAlmostEqual(row["received_float"], expected["received_float"], places=5, msg=f"{tag}received_float")
        self.assertAlmostEqual(
            row["received_double"], expected["received_double"], places=5, msg=f"{tag}received_double"
        )
        self.assertEqual(row["received_bool"], expected["received_bool"], f"{tag}received_bool")
        self.assertEqual(row["received_string"], expected["received_string"], f"{tag}received_string")
        self.assertEqual(row["received_bytes"], expected["received_bytes"], f"{tag}received_bytes")
        self.assertEqual(row["received_timestamp"], expected["received_timestamp"], f"{tag}received_timestamp")
        self.assertEqual(row["received_weights"], expected["received_weights"], f"{tag}received_weights")
        self.assertEqual(row["received_nested_list"], expected["received_nested_list"], f"{tag}received_nested_list")
        self.assertAlmostEqual(
            row["received_temperature"], expected["received_temperature"], places=5, msg=f"{tag}received_temperature"
        )
        self.assertEqual(row["received_top_k"], expected["received_top_k"], f"{tag}received_top_k")
        self.assertEqual(row["received_use_cache"], expected["received_use_cache"], f"{tag}received_use_cache")
        self.assertEqual(row["received_stop_token"], expected["received_stop_token"], f"{tag}received_stop_token")
        self.assertEqual(row["received_penalties"], expected["received_penalties"], f"{tag}received_penalties")
        self.assertEqual(row["received_seed"], expected["received_seed"], f"{tag}received_seed")
        self.assertEqual(row["received_strategy"], expected["received_strategy"], f"{tag}received_strategy")
        self.assertEqual(row["received_models"], expected["received_models"], f"{tag}received_models")

    def _check_all_data_types_df(self, res: pd.DataFrame, expected: dict[str, Any], label: str = "") -> None:
        """Validate a DataFrame result (mv.run / REST flat path)."""
        self._check_all_data_types(res.iloc[0], expected, label)

    def _check_wide(self, row: dict[str, Any], expected: dict[str, Any], label: str = "") -> None:
        """Validate wide model response values."""
        tag = f"[{label}] " if label else ""
        self.assertAlmostEqual(row["feature_sum"], expected["feature_sum"], places=5, msg=f"{tag}feature_sum")
        self.assertAlmostEqual(row["output"], expected["output"], places=5, msg=f"{tag}output")
        self.assertAlmostEqual(
            row["received_multiplier"], expected["received_multiplier"], places=5, msg=f"{tag}received_multiplier"
        )
        self.assertEqual(row["received_offset"], expected["received_offset"], f"{tag}received_offset")

    # ===================================================================
    # Payload builders
    # ===================================================================

    def _flat_payload(self, value: float, params: dict[str, Any]) -> dict[str, Any]:
        """Build flat format payload: {"data": [[row_id, feature, param1, param2, ...]]}."""
        test_input = pd.DataFrame({"value": [value], **{k: [v] for k, v in params.items()}})
        return self._to_external_data_format(test_input)

    def _build_wide_payload(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build WIDE format payload: {"data": [[row_id, {features + params}]]}."""
        row_dict = {f"f_{i}": 1.0 for i in range(_WIDE_FORMAT_NUM_FEATURES)}
        if params:
            row_dict.update(params)
        return {"data": [[0, row_dict]]}

    # ===================================================================
    # Deploy helpers
    # ===================================================================

    def _deploy_all_data_types(self) -> tuple[Any, str]:
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_all_data_types_signature()
        test_input = pd.DataFrame({"value": [10.0]})

        def check_deploy(res: pd.DataFrame) -> None:
            self.assertEqual(len(res), 1, "Expected single response row from deploy check")

        mv = self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_deploy)},
            skip_rest_api_test=True,
        )
        endpoint = self._ensure_ingress_url(mv)
        return mv, endpoint

    def _deploy_wide(self) -> tuple[Any, str]:
        model = ModelWithManyFeatures(custom_model.ModelContext())
        sig = self._get_wide_model_signature()
        data = {f"f_{i}": [float(1.0)] for i in range(_WIDE_FORMAT_NUM_FEATURES)}
        test_input = pd.DataFrame(data)

        def check_deploy(res: pd.DataFrame) -> None:
            self.assertEqual(len(res), 1, "Expected single response row from deploy check")

        mv = self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_deploy)},
            skip_rest_api_test=True,
        )
        endpoint = self._ensure_ingress_url(mv)
        return mv, endpoint

    # ===================================================================
    # Flat: mv.run subtests
    # ===================================================================

    def _test_mv_run(self, mv: Any) -> None:
        service_name = self._get_service_name(mv)
        input_df = pd.DataFrame({"value": [10.0]})

        with self.subTest("mv_run_flat/ full"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_FULL_PARAMS)
            self._check_all_data_types_df(res, _FULL_EXPECTED, "mv_run/full")

        with self.subTest("mv_run_flat/ partial"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_PARTIAL_PARAMS)
            self._check_all_data_types_df(res, _PARTIAL_EXPECTED, "mv_run/partial")

        with self.subTest("mv_run_flat/ default"):
            res = mv.run(input_df, function_name="predict", service_name=service_name)
            self._check_all_data_types_df(res, _DEFAULT_EXPECTED, "mv_run/default")

        with self.subTest("mv_run_flat/ none"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_NONE_PARAMS)
            self._check_all_data_types_df(res, _DEFAULT_EXPECTED, "mv_run/none")

        with self.subTest("mv_run_flat/ invalid_name"):
            with self.assertRaisesRegex(ValueError, r"Unknown parameter"):
                mv.run(input_df, function_name="predict", service_name=service_name, params=_INVALID_NAME_PARAMS)

        with self.subTest("mv_run_flat/ invalid_type"):
            with self.assertRaisesRegex(ValueError, r"not compatible with dtype"):
                mv.run(input_df, function_name="predict", service_name=service_name, params=_INVALID_TYPE_PARAMS)

        with self.subTest("mv_run / multi_row_with_params"):
            # Verify params are applied consistently across all rows in a multi-row DataFrame.
            # mv.run() params are request-level (single dict), so all rows should receive the same values.
            multi_row_df = pd.DataFrame({"value": [10.0, 20.0, 30.0]})
            res = mv.run(multi_row_df, function_name="predict", service_name=service_name, params=_FULL_PARAMS)
            self.assertEqual(len(res), 3, "Expected 3 response rows")
            for i in range(3):
                row = res.iloc[i]
                self.assertAlmostEqual(row["input_value"], [10.0, 20.0, 30.0][i], places=5)
                self.assertEqual(row["received_int8"], 10, f"Row {i} should use int8_param=10")
                self.assertEqual(row["received_string"], "custom_value", f"Row {i} should use string_param override")

    # ===================================================================
    # REST flat subtests
    # ===================================================================

    def _test_rest_flat(self, endpoint: str) -> None:
        # Flat format sends params as columns alongside features (positional, all columns required).
        # All params are always present as serialized strings, so bytes arrive as lowercase hex.

        with self.subTest("rest_flat / full"):
            flat_params = {**_REST_DEFAULT_PARAMS, **_serialize_for_rest(_FULL_PARAMS)}
            response = self._assert_rest_ok(endpoint, self._flat_payload(10.0, flat_params), label="flat/full")
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self._check_all_data_types_df(res_df, _to_raw_expected(_FULL_EXPECTED), "flat/full")

        with self.subTest("rest_flat / partial"):
            flat_params = {**_REST_DEFAULT_PARAMS, **_serialize_for_rest(_PARTIAL_PARAMS)}
            response = self._assert_rest_ok(endpoint, self._flat_payload(10.0, flat_params), label="flat/partial")
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self._check_all_data_types_df(res_df, _to_raw_expected(_PARTIAL_EXPECTED), "flat/partial")

        with self.subTest("rest_flat / default"):
            response = self._assert_rest_ok(
                endpoint, self._flat_payload(10.0, _REST_DEFAULT_PARAMS), label="flat/default"
            )
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            self._check_all_data_types_df(res_df, _to_raw_expected(_DEFAULT_EXPECTED), "flat/default")

        with self.subTest("rest_flat / none"):
            flat_params = {**_REST_DEFAULT_PARAMS, **_serialize_for_rest(_NONE_PARAMS)}
            response = self._assert_rest_ok(endpoint, self._flat_payload(10.0, flat_params), label="flat/none")
            res_df = pd.DataFrame([x[1] for x in response.json()["data"]])
            # bytes_param=None → server resolves default as actual bytes → model does .hex().upper() → uppercase hex
            self._check_all_data_types_df(res_df, _DEFAULT_EXPECTED, "flat/none")

        # Flat-specific edge cases (invalid_name is N/A for positional format)
        with self.subTest("rest_flat / invalid_type"):
            bad_type = {**_REST_DEFAULT_PARAMS, "int8_param": "not_an_int"}
            self._assert_rest_400(endpoint, self._flat_payload(10.0, bad_type), label="flat/invalid_type")

        with self.subTest("rest_flat / too_many_cols"):
            payload = self._flat_payload(10.0, _REST_DEFAULT_PARAMS)
            payload["data"][0].append("extra_value")
            self._assert_rest_400(endpoint, payload, label="flat/too_many_cols")

        with self.subTest("rest_flat / too_few_cols"):
            self._assert_rest_400(endpoint, {"data": [[0, 10.0, 1, 2]]}, label="flat/too_few_cols")

        # TODO: Service path should reject rows with varying param values.
        #  Currently first-row-wins silently. Uncomment once the server validates param equality.
        # with self.subTest("rest_flat / varying_params_across_rows"):
        #     row0_params = {**_REST_DEFAULT_PARAMS, "int8_param": 10}
        #     row1_params = {**_REST_DEFAULT_PARAMS, "int8_param": 99}
        #     row0 = pd.DataFrame({"value": [10.0], **{k: [v] for k, v in row0_params.items()}})
        #     row1 = pd.DataFrame({"value": [20.0], **{k: [v] for k, v in row1_params.items()}})
        #     payload0 = self._to_external_data_format(row0)
        #     payload1 = self._to_external_data_format(row1)
        #     payload0["data"].extend(payload1["data"])
        #     self._assert_rest_400(endpoint, payload0, label="flat/varying_params")

    # ===================================================================
    # REST split subtests
    # ===================================================================

    def _test_rest_split(self, endpoint: str) -> None:
        base = {"dataframe_split": {"index": [0], "columns": ["value"], "data": [[10.0]]}}

        with self.subTest("rest_split / full"):
            payload = {**base, "params": _serialize_for_rest(_FULL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label="split/full")
            row = self._parse_rest_rows(response)[0]
            self._check_all_data_types(row, _to_raw_expected(_FULL_EXPECTED), "split/full")

        with self.subTest("rest_split / partial"):
            payload = {**base, "params": _serialize_for_rest(_PARTIAL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label="split/partial")
            row = self._parse_rest_rows(response)[0]
            # bytes_param not in _PARTIAL_PARAMS → Go proxy resolves default as raw string
            self._check_all_data_types(
                row, _to_raw_expected(_PARTIAL_EXPECTED, bytes_overridden=False), "split/partial"
            )

        with self.subTest("rest_split / default"):
            response = self._assert_rest_ok(endpoint, base, label="split/default")
            row = self._parse_rest_rows(response)[0]
            self._check_all_data_types(
                row, _to_raw_expected(_DEFAULT_EXPECTED, bytes_overridden=False), "split/default"
            )

        with self.subTest("rest_split / none"):
            payload = {**base, "params": _serialize_for_rest(_NONE_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label="split/none")
            row = self._parse_rest_rows(response)[0]
            # Explicit null → server resolves default as actual bytes → model hex-encodes
            self._check_all_data_types(row, _DEFAULT_EXPECTED, "split/none")

        # TODO: Go proxy silently ignores unknown params instead of returning 400.
        #  Uncomment once the proxy validates param names.
        # with self.subTest("rest_split / invalid_name"):
        #     self._assert_rest_400(endpoint, {**base, "params": _INVALID_NAME_PARAMS}, label="split/invalid_name")

        with self.subTest("rest_split / invalid_type"):
            self._assert_rest_400(endpoint, {**base, "params": _INVALID_TYPE_PARAMS}, label="split/invalid_type")

        # TODO: Go proxy silently ignores extra data columns instead of returning 400.
        #  Uncomment once the proxy validates column counts.
        # with self.subTest("rest_split / too_many_cols"):
        #     self._assert_rest_400(
        #         endpoint,
        #         {"dataframe_split": {"index": [0], "columns": ["value", "extra_col"], "data": [[10.0, "x"]]}},
        #         label="split/too_many_cols",
        #     )

        with self.subTest("rest_split / extra_cols"):
            # Extra column WITH extra_columns key → server should accept and ignore
            rest_params = _serialize_for_rest(_FULL_PARAMS)
            response = self._assert_rest_ok(
                endpoint,
                {
                    "dataframe_split": {"index": [0], "columns": ["value", "extra_col"], "data": [[10.0, "x"]]},
                    "params": rest_params,
                    "extra_columns": ["extra_col"],
                },
                label="split/extra_cols",
            )
            row = self._parse_rest_rows(response)[0]
            self._check_all_data_types(row, _to_raw_expected(_FULL_EXPECTED), "split/extra_cols")

    # ===================================================================
    # REST records subtests
    # ===================================================================

    def _test_rest_records(self, endpoint: str) -> None:
        base: dict[str, Any] = {"dataframe_records": [{"value": 10.0}]}

        with self.subTest("rest_records / full"):
            payload = {**base, "params": _serialize_for_rest(_FULL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label="records/full")
            row = self._parse_rest_rows(response)[0]
            self._check_all_data_types(row, _to_raw_expected(_FULL_EXPECTED), "records/full")

        with self.subTest("rest_records / partial"):
            payload = {**base, "params": _serialize_for_rest(_PARTIAL_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label="records/partial")
            row = self._parse_rest_rows(response)[0]
            # bytes_param not in _PARTIAL_PARAMS → Go proxy resolves default as raw string
            self._check_all_data_types(
                row, _to_raw_expected(_PARTIAL_EXPECTED, bytes_overridden=False), "records/partial"
            )

        with self.subTest("rest_records / default"):
            response = self._assert_rest_ok(endpoint, base, label="records/default")
            row = self._parse_rest_rows(response)[0]
            self._check_all_data_types(
                row, _to_raw_expected(_DEFAULT_EXPECTED, bytes_overridden=False), "records/default"
            )

        with self.subTest("rest_records / none"):
            payload = {**base, "params": _serialize_for_rest(_NONE_PARAMS)}
            response = self._assert_rest_ok(endpoint, payload, label="records/none")
            row = self._parse_rest_rows(response)[0]
            # Explicit null → server resolves default as actual bytes → model hex-encodes
            self._check_all_data_types(row, _DEFAULT_EXPECTED, "records/none")

        # TODO: Go proxy silently ignores unknown params instead of returning 400.
        #  Uncomment once the proxy validates param names.
        # with self.subTest("rest_records / invalid_name"):
        #     self._assert_rest_400(endpoint, {**base, "params": _INVALID_NAME_PARAMS}, label="records/invalid_name")

        with self.subTest("rest_records / invalid_type"):
            self._assert_rest_400(endpoint, {**base, "params": _INVALID_TYPE_PARAMS}, label="records/invalid_type")

        # TODO: Go proxy silently ignores extra data columns instead of returning 400.
        #  Uncomment once the proxy validates column counts.
        # with self.subTest("rest_records / too_many_cols"):
        #     self._assert_rest_400(
        #         endpoint,
        #         {"dataframe_records": [{"value": 10.0, "extra_col": "x"}]},
        #         label="records/too_many_cols",
        #     )

        with self.subTest("rest_records / extra_cols"):
            # Extra column WITH extra_columns key → server should accept and ignore
            rest_params = _serialize_for_rest(_FULL_PARAMS)
            response = self._assert_rest_ok(
                endpoint,
                {
                    "dataframe_records": [{"value": 10.0, "extra_col": "x"}],
                    "params": rest_params,
                    "extra_columns": ["extra_col"],
                },
                label="records/extra_cols",
            )
            row = self._parse_rest_rows(response)[0]
            self._check_all_data_types(row, _to_raw_expected(_FULL_EXPECTED), "records/extra_cols")

    # ===================================================================
    # Wide: mv.run subtests
    # ===================================================================

    def _test_mv_run_wide(self, mv: Any) -> None:
        service_name = self._get_service_name(mv)
        data = {f"f_{i}": [float(1.0)] for i in range(_WIDE_FORMAT_NUM_FEATURES)}
        input_df = pd.DataFrame(data)

        with self.subTest("mv_run_wide / full"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_WIDE_FULL_PARAMS)
            self._check_wide(res.iloc[0], _WIDE_FULL_EXPECTED, "mv_run_wide/full")

        with self.subTest("mv_run_wide / partial"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_WIDE_PARTIAL_PARAMS)
            self._check_wide(res.iloc[0], _WIDE_PARTIAL_EXPECTED, "mv_run_wide/partial")

        with self.subTest("mv_run_wide / default"):
            res = mv.run(input_df, function_name="predict", service_name=service_name)
            self._check_wide(res.iloc[0], _WIDE_DEFAULT_EXPECTED, "mv_run_wide/default")

        with self.subTest("mv_run_wide / none"):
            res = mv.run(input_df, function_name="predict", service_name=service_name, params=_WIDE_NONE_PARAMS)
            # None params → server substitutes defaults
            self._check_wide(res.iloc[0], _WIDE_DEFAULT_EXPECTED, "mv_run_wide/none")

        with self.subTest("mv_run_wide / invalid_name"):
            with self.assertRaisesRegex(ValueError, r"Unknown parameter"):
                mv.run(input_df, function_name="predict", service_name=service_name, params=_WIDE_INVALID_NAME_PARAMS)

        with self.subTest("mv_run_wide / invalid_type"):
            with self.assertRaisesRegex(ValueError, r"not compatible with dtype"):
                mv.run(input_df, function_name="predict", service_name=service_name, params=_WIDE_INVALID_TYPE_PARAMS)

    # ===================================================================
    # Wide: REST wide subtests
    # ===================================================================

    def _test_rest_wide(self, endpoint: str) -> None:
        with self.subTest("rest_wide / full"):
            response = self._assert_rest_ok(endpoint, self._build_wide_payload(_WIDE_FULL_PARAMS), label="wide/full")
            row = self._parse_rest_rows(response)[0]
            self._check_wide(row, _WIDE_FULL_EXPECTED, "wide/full")

        with self.subTest("rest_wide / partial"):
            response = self._assert_rest_ok(
                endpoint, self._build_wide_payload(_WIDE_PARTIAL_PARAMS), label="wide/partial"
            )
            row = self._parse_rest_rows(response)[0]
            self._check_wide(row, _WIDE_PARTIAL_EXPECTED, "wide/partial")

        with self.subTest("rest_wide / default"):
            response = self._assert_rest_ok(endpoint, self._build_wide_payload(), label="wide/default")
            row = self._parse_rest_rows(response)[0]
            self._check_wide(row, _WIDE_DEFAULT_EXPECTED, "wide/default")

        with self.subTest("rest_wide / none"):
            response = self._assert_rest_ok(endpoint, self._build_wide_payload(_WIDE_NONE_PARAMS), label="wide/none")
            row = self._parse_rest_rows(response)[0]
            # None params → server substitutes defaults
            self._check_wide(row, _WIDE_DEFAULT_EXPECTED, "wide/none")

        # TODO: Server silently ignores unknown keys in the wide dict instead of returning 400.
        #  Uncomment once the inference server validates param names in wide format.
        # with self.subTest("rest_wide / invalid_name"):
        #     self._assert_rest_400(
        #         endpoint, self._build_wide_payload(_WIDE_INVALID_NAME_PARAMS), label="wide/invalid_name"
        #     )

        with self.subTest("rest_wide / invalid_type"):
            self._assert_rest_400(
                endpoint, self._build_wide_payload(_WIDE_INVALID_TYPE_PARAMS), label="wide/invalid_type"
            )

        # TODO: Server silently ignores extra keys in the wide dict instead of returning 400.
        #  Uncomment once the inference server validates column counts in wide format.
        # with self.subTest("rest_wide / too_many_cols"):
        #     payload = self._build_wide_payload()
        #     payload["data"][0][1]["extra_col"] = "unexpected"
        #     self._assert_rest_400(endpoint, payload, label="wide/too_many_cols")

        with self.subTest("rest_wide / trailing_positional_params"):
            # Alternate wire format: params as trailing positional elements AFTER the feature dict.
            # Format: [index, {features_only}, param1, param2, ...]
            # This occurs when SQL passes OBJECT_CONSTRUCT for features with separate param args.
            # The server extracts params from the dict only (not trailing args), so defaults are used.
            features_only = {f"f_{i}": 1.0 for i in range(_WIDE_FORMAT_NUM_FEATURES)}
            payload = {"data": [[0, features_only, 2.0, 100]]}  # trailing multiplier=2.0, offset=100
            response = self._assert_rest_ok(endpoint, payload, label="wide/trailing_positional")
            row = self._parse_rest_rows(response)[0]
            # Trailing positional params are ignored — server uses defaults
            self._check_wide(row, _WIDE_DEFAULT_EXPECTED, "wide/trailing_positional")

    # ===================================================================
    # Entry points — one deployment per test method
    # ===================================================================

    def test_all_data_types_params(self) -> None:
        """Deploy ModelWithAllDataTypes once, then run 27 subtests across all paths and param variants."""
        mv, endpoint = self._deploy_all_data_types()

        with self.subTest("mv_run"):
            self._test_mv_run(mv)
        with self.subTest("rest_flat"):
            self._test_rest_flat(endpoint)
        with self.subTest("rest_split"):
            self._test_rest_split(endpoint)
        with self.subTest("rest_records"):
            self._test_rest_records(endpoint)

    def test_wide_format_params(self) -> None:
        """Deploy ModelWithManyFeatures once, then run 12 subtests across mv.run and REST wide."""
        mv, endpoint = self._deploy_wide()

        with self.subTest("mv_run_wide"):
            self._test_mv_run_wide(mv)
        with self.subTest("rest_wide"):
            self._test_rest_wide(endpoint)


if __name__ == "__main__":
    absltest.main()
