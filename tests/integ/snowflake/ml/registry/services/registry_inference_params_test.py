"""Integration tests for model inference with runtime parameters (ParamSpec).

This test verifies that:
1. Models with ParamSpec parameters in their signature can be deployed
2. Parameter values passed at inference time are correctly received by the model
3. Default values are used when parameters are None or not provided
4. Partial params work (some set, some using defaults)
5. Various data types are supported
6. WIDE format (500+ features) works with params
7. SPLIT format with top-level params works
8. RECORDS format with top-level params works
"""

import datetime
import unittest
from typing import Any

import pandas as pd
import requests
from absl.testing import absltest
from packaging import version

from snowflake.ml.model import custom_model, model_signature
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import test_env_utils

_DEFAULT_TIMESTAMP = datetime.datetime(2024, 1, 1, 12, 0, 0)
_DEFAULT_WEIGHTS = [1.0, 2.0, 3.0]
_DEFAULT_NESTED_LIST = [[1, 2], [3, 4]]


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
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "input_value": [input["value"].iloc[0]],
                "received_int8": [int8_param],
                "received_int16": [int16_param],
                "received_int32": [int32_param],
                "received_int64": [int64_param],
                "received_uint8": [uint8_param],
                "received_uint16": [uint16_param],
                "received_uint32": [uint32_param],
                "received_uint64": [uint64_param],
                "received_float": [float_param],
                "received_double": [double_param],
                "received_bool": [bool_param],
                "received_string": [string_param],
                # Convert bytes to hex string for output (avoids JSON serialization issues)
                # TODO (SNOW-3045092): Fix byte output serialization issue
                "received_bytes": [bytes_param.hex().upper() if isinstance(bytes_param, bytes) else bytes_param],
                # Convert timestamp to string with milliseconds for consistent output
                # Handles both datetime objects (SQL path) and ISO strings (REST path)
                "received_timestamp": [_format_timestamp(_normalize_timestamp(timestamp_param))],
                "received_weights": [weights_param],
                "received_nested_list": [nested_list],
            }
        )


class ModelWithNestedParams(custom_model.CustomModel):
    """A custom model with nested parameters using ParamGroupSpec."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        # Handle None values by using defaults
        if learning_rate is None:
            learning_rate = 0.0
        if momentum is None:
            momentum = 0.0
        if epochs is None:
            epochs = 0
        if batch_size is None:
            batch_size = 0

        return pd.DataFrame(
            {
                "input_value": [input["value"].iloc[0]],
                "received_learning_rate": [learning_rate],
                "received_momentum": [momentum],
                "received_epochs": [epochs],
                "received_batch_size": [batch_size],
            }
        )


# Number of features needed to trigger wide format (must exceed 500 to trigger wide format at deployment time)
_WIDE_FORMAT_NUM_FEATURES = 501


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


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_version() >= version.parse("10.0.0"),
    "Model method signature parameters only available when Snowflake Version >= 10.0.0",
)
class TestRegistryInferenceParamsInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for inference with runtime parameters."""

    def _get_model_with_params_signature(self) -> model_signature.ModelSignature:
        """Shared signature for ModelWithParams tests."""
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
            ],
        )

    def test_mv_run_with_all_params_provided(self) -> None:
        """Test that all params passed at inference time are correctly received."""
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_model_with_params_signature()

        test_input = pd.DataFrame({"value": [10.0]})
        test_params = {
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
        }

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["input_value"].iloc[0], 10.0, places=5)
            self.assertEqual(res["received_int8"].iloc[0], 10)
            self.assertEqual(res["received_int16"].iloc[0], 200)
            self.assertEqual(res["received_int32"].iloc[0], 3000)
            self.assertEqual(res["received_int64"].iloc[0], 40000)
            self.assertEqual(res["received_uint8"].iloc[0], 15)
            self.assertEqual(res["received_uint16"].iloc[0], 300)
            self.assertEqual(res["received_uint32"].iloc[0], 4000)
            self.assertEqual(res["received_uint64"].iloc[0], 50000)
            self.assertEqual(res["received_float"].iloc[0], 1.25)
            self.assertEqual(res["received_double"].iloc[0], 2.75)
            self.assertEqual(res["received_bool"].iloc[0], False)
            self.assertEqual(res["received_string"].iloc[0], "custom_value")
            # Inference server converts bytes to hex string for output
            # TODO (SNOW-3045092): Fix byte output serialization issue
            self.assertEqual(res["received_bytes"].iloc[0], b"hello".hex().upper())  # hex of b"hello"
            # Inference server converts timestamp to ISO format string for output
            # TODO (SNOW-3045092): Fix timestamp output serialization issue
            self.assertEqual(
                res["received_timestamp"].iloc[0], _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5))
            )
            self.assertEqual(res["received_weights"].iloc[0], [4.5, 3.5, 2.5])
            self.assertEqual(res["received_nested_list"].iloc[0], [[4, 3], [2, 1]])

        self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_result)},
            params=test_params,
            skip_rest_api_test=True,
        )

    def test_mv_run_with_all_params_default(self) -> None:
        """Test that signature default values are used when params not provided."""
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_model_with_params_signature()

        test_input = pd.DataFrame(
            {
                "value": [10.0],
            }
        )

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["input_value"].iloc[0], 10.0, places=5)
            self.assertEqual(res["received_int8"].iloc[0], 1)
            self.assertEqual(res["received_int16"].iloc[0], 2)
            self.assertEqual(res["received_int32"].iloc[0], 3)
            self.assertEqual(res["received_int64"].iloc[0], 4)
            self.assertEqual(res["received_uint8"].iloc[0], 5)
            self.assertEqual(res["received_uint16"].iloc[0], 6)
            self.assertEqual(res["received_uint32"].iloc[0], 7)
            self.assertEqual(res["received_uint64"].iloc[0], 8)
            self.assertEqual(res["received_float"].iloc[0], 1.5)
            self.assertEqual(res["received_double"].iloc[0], 2.5)
            self.assertEqual(res["received_bool"].iloc[0], True)
            self.assertEqual(res["received_string"].iloc[0], "default")
            # Inference server converts bytes to hex string for output
            # TODO (SNOW-3045092): Fix byte output serialization issue
            self.assertEqual(res["received_bytes"].iloc[0], b"default".hex().upper())  # hex of b"default"
            # Inference server converts timestamp to ISO format string for output
            # TODO (SNOW-3045092): Fix timestamp output serialization issue
            self.assertEqual(res["received_timestamp"].iloc[0], _DEFAULT_TIMESTAMP_STR)
            self.assertEqual(res["received_weights"].iloc[0], _DEFAULT_WEIGHTS)
            self.assertEqual(res["received_nested_list"].iloc[0], _DEFAULT_NESTED_LIST)

        self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_result)},
            skip_rest_api_test=True,
        )

    def test_mv_run_with_partial_params(self) -> None:
        """Test setting some params while others use defaults."""
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_model_with_params_signature()

        # Set timestamp_param, weights_param, and nested_list only, others use signature defaults
        test_input = pd.DataFrame({"value": [10.0]})
        test_params = {
            "timestamp_param": datetime.datetime(2025, 1, 2, 3, 4, 5),
            "weights_param": [4.5, 3.5, 2.5],
            "nested_list": [[4, 3], [2, 1]],
        }

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["input_value"].iloc[0], 10.0, places=5)
            self.assertEqual(res["received_int8"].iloc[0], 1)
            self.assertEqual(res["received_int16"].iloc[0], 2)
            self.assertEqual(res["received_int32"].iloc[0], 3)
            self.assertEqual(res["received_int64"].iloc[0], 4)
            self.assertEqual(res["received_uint8"].iloc[0], 5)
            self.assertEqual(res["received_uint16"].iloc[0], 6)
            self.assertEqual(res["received_uint32"].iloc[0], 7)
            self.assertEqual(res["received_uint64"].iloc[0], 8)
            self.assertEqual(res["received_float"].iloc[0], 1.5)
            self.assertEqual(res["received_double"].iloc[0], 2.5)
            self.assertEqual(res["received_bool"].iloc[0], True)
            self.assertEqual(res["received_string"].iloc[0], "default")
            # Inference server converts bytes to hex string for output
            # TODO (SNOW-3045092): Fix byte output serialization issue
            self.assertEqual(res["received_bytes"].iloc[0], b"default".hex().upper())  # hex of b"default"
            # Inference server converts timestamp to ISO format string for output
            # TODO (SNOW-3045092): Fix timestamp output serialization issue
            self.assertEqual(
                res["received_timestamp"].iloc[0], _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5))
            )
            self.assertEqual(res["received_weights"].iloc[0], [4.5, 3.5, 2.5])
            self.assertEqual(res["received_nested_list"].iloc[0], [[4, 3], [2, 1]])

        self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_result)},
            params=test_params,
            skip_rest_api_test=True,
        )

    def test_rest_api_flat_format_with_params(self) -> None:
        """Test REST API inference with all parameters provided."""
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_model_with_params_signature()

        # Deploy model - pass params explicitly for mv.run() path
        # REST API path will add param columns automatically
        valid_input = pd.DataFrame({"value": [1.0]})
        test_params = {
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
        }

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["input_value"].iloc[0], 1.0, places=5)
            self.assertEqual(res["received_int8"].iloc[0], 10)
            self.assertEqual(res["received_int16"].iloc[0], 200)
            self.assertEqual(res["received_int32"].iloc[0], 3000)
            self.assertEqual(res["received_int64"].iloc[0], 40000)
            self.assertEqual(res["received_uint8"].iloc[0], 15)
            self.assertEqual(res["received_uint16"].iloc[0], 300)
            self.assertEqual(res["received_uint32"].iloc[0], 4000)
            self.assertEqual(res["received_uint64"].iloc[0], 50000)
            self.assertAlmostEqual(res["received_float"].iloc[0], 1.25, places=5)
            self.assertAlmostEqual(res["received_double"].iloc[0], 2.75, places=5)
            self.assertEqual(res["received_bool"].iloc[0], False)
            self.assertEqual(res["received_string"].iloc[0], "custom_value")
            # Inference server converts bytes to hex string for output
            # TODO (SNOW-3045092): Fix byte output serialization issue
            self.assertEqual(res["received_bytes"].iloc[0], b"hello".hex().upper())  # hex of b"hello"
            # Inference server converts timestamp to ISO format string for output
            # TODO (SNOW-3045092): Fix timestamp output serialization issue
            self.assertEqual(
                res["received_timestamp"].iloc[0], _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5))
            )
            self.assertEqual(res["received_weights"].iloc[0], [4.5, 3.5, 2.5])
            self.assertEqual(res["received_nested_list"].iloc[0], [[4, 3], [2, 1]])

        mv = self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (valid_input, check_result)},
            params=test_params,
            skip_rest_api_test=True,
        )

        # Test REST API with different params
        endpoint = self._ensure_ingress_url(mv)
        jwt_token_generator = self._get_jwt_token_generator()

        # Build REST API input with features and params as columns
        # Note: bytes and datetime must be serialized for JSON (hex string and ISO format)
        test_input = pd.DataFrame(
            {
                "value": [10.0],
                "int8_param": [20],
                "int16_param": [400],
                "int32_param": [6000],
                "int64_param": [80000],
                "uint8_param": [30],
                "uint16_param": [600],
                "uint32_param": [8000],
                "uint64_param": [100000],
                "float_param": [2.5],
                "double_param": [5.5],
                "bool_param": [True],
                "string_param": ["rest_api_value"],
                "bytes_param": [b"rest".hex()],  # hex string for JSON serialization
                "timestamp_param": [datetime.datetime(2026, 6, 7, 8, 9, 10).isoformat()],  # ISO string for JSON
                "weights_param": [[1.0, 2.0, 3.0]],
                "nested_list": [[[5, 6], [7, 8]]],
            }
        )

        res_df = self._inference_using_rest_api(
            self._to_external_data_format(test_input),
            endpoint=endpoint,
            jwt_token_generator=jwt_token_generator,
            target_method="predict",
        )

        self.assertAlmostEqual(res_df["input_value"].iloc[0], 10.0, places=5)
        self.assertEqual(res_df["received_int8"].iloc[0], 20)
        self.assertEqual(res_df["received_int16"].iloc[0], 400)
        self.assertEqual(res_df["received_int32"].iloc[0], 6000)
        self.assertEqual(res_df["received_int64"].iloc[0], 80000)
        self.assertEqual(res_df["received_uint8"].iloc[0], 30)
        self.assertEqual(res_df["received_uint16"].iloc[0], 600)
        self.assertEqual(res_df["received_uint32"].iloc[0], 8000)
        self.assertEqual(res_df["received_uint64"].iloc[0], 100000)
        self.assertEqual(res_df["received_float"].iloc[0], 2.5)
        self.assertEqual(res_df["received_double"].iloc[0], 5.5)
        self.assertEqual(res_df["received_bool"].iloc[0], True)
        self.assertEqual(res_df["received_string"].iloc[0], "rest_api_value")
        # Inference server converts bytes to hex string for output
        # TODO (SNOW-3045092): Fix byte output serialization issue
        self.assertEqual(res_df["received_bytes"].iloc[0], b"rest".hex().upper())  # hex of b"rest"
        # Inference server converts timestamp to ISO format string for output
        # TODO (SNOW-3045092): Fix timestamp output serialization issue
        self.assertEqual(
            res_df["received_timestamp"].iloc[0], _format_timestamp(datetime.datetime(2026, 6, 7, 8, 9, 10))
        )
        self.assertEqual(res_df["received_weights"].iloc[0], [1.0, 2.0, 3.0])
        self.assertEqual(res_df["received_nested_list"].iloc[0], [[5, 6], [7, 8]])

    def _get_wide_model_signature(self) -> model_signature.ModelSignature:
        """Signature for ModelWithManyFeatures (500+ features to trigger wide format)."""
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

    def _create_wide_format_input(self, num_features: int = _WIDE_FORMAT_NUM_FEATURES) -> pd.DataFrame:
        """Create a DataFrame with many features for wide format testing."""
        data = {f"f_{i}": [float(1.0)] for i in range(num_features)}
        return pd.DataFrame(data)

    def _make_rest_api_request(
        self,
        endpoint: str,
        request_payload: dict[str, Any],
        target_method: str,
    ) -> requests.Response:
        """Make a REST API request with a custom payload format."""
        auth_handler = self._get_auth_for_inference(endpoint)
        return requests.post(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json=request_payload,
            auth=auth_handler,
            timeout=60,
        )

    def test_rest_api_wide_format_with_params(self) -> None:
        """Test that params work correctly with WIDE format (500+ features/params)."""
        model = ModelWithManyFeatures(custom_model.ModelContext())
        sig = self._get_wide_model_signature()

        test_input = self._create_wide_format_input()
        # Expected feature_sum = 1.0 * 501 = 501
        expected_feature_sum = 501
        test_params = {"multiplier": 2.0, "offset": 100}

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["feature_sum"].iloc[0], expected_feature_sum, places=5)
            self.assertAlmostEqual(res["output"].iloc[0], expected_feature_sum * 2.0 + 100, places=5)
            self.assertAlmostEqual(res["received_multiplier"].iloc[0], 2.0, places=5)
            self.assertEqual(res["received_offset"].iloc[0], 100)

        mv = self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (test_input, check_result)},
            params=test_params,
            skip_rest_api_test=True,
        )

        # Test REST API with different params
        endpoint = self._ensure_ingress_url(mv)

        # Build WIDE format payload: {"data": [[row_id, {features + params}], ...]}
        # Features and params are combined in the same dict
        row_dict = {f"f_{i}": 1.0 for i in range(_WIDE_FORMAT_NUM_FEATURES)}
        row_dict["multiplier"] = 3.0
        row_dict["offset"] = 50
        wide_payload = {"data": [[0, row_dict]]}

        response = self._make_rest_api_request(endpoint, wide_payload, "predict")
        response.raise_for_status()

        # Parse response: {"data": [[0, {"output": val, ...}]]}
        response_data = response.json()["data"]
        res_row = response_data[0][1]

        # feature_sum = 501 (each feature is 1.0), output = 501 * 3.0 + 50 = 1553
        self.assertAlmostEqual(res_row["feature_sum"], expected_feature_sum, places=5)
        self.assertAlmostEqual(res_row["output"], expected_feature_sum * 3.0 + 50, places=5)
        self.assertAlmostEqual(res_row["received_multiplier"], 3.0, places=5)
        self.assertEqual(res_row["received_offset"], 50)

    def test_rest_api_split_format_with_params(self) -> None:
        """Test SPLIT format with top-level params."""
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_model_with_params_signature()

        # Deploy model with flat format first
        valid_input = pd.DataFrame(
            {
                "value": [1.0],
            }
        )

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["input_value"].iloc[0], 1.0, places=5)
            # These should be the model defaults since no params are passed
            self.assertEqual(res["received_int8"].iloc[0], 1)
            self.assertEqual(res["received_int16"].iloc[0], 2)
            self.assertEqual(res["received_int32"].iloc[0], 3)
            self.assertEqual(res["received_int64"].iloc[0], 4)
            self.assertEqual(res["received_uint8"].iloc[0], 5)
            self.assertEqual(res["received_uint16"].iloc[0], 6)
            self.assertEqual(res["received_uint32"].iloc[0], 7)
            self.assertEqual(res["received_uint64"].iloc[0], 8)
            self.assertEqual(res["received_float"].iloc[0], 1.5)
            self.assertEqual(res["received_double"].iloc[0], 2.5)
            self.assertEqual(res["received_bool"].iloc[0], True)
            self.assertEqual(res["received_string"].iloc[0], "default")
            # Inference server converts bytes to uppercase hex string for output
            self.assertEqual(res["received_bytes"].iloc[0], b"default".hex().upper())
            # Inference server converts timestamp to ISO format string for output
            self.assertEqual(res["received_timestamp"].iloc[0], _DEFAULT_TIMESTAMP_STR)
            self.assertEqual(res["received_weights"].iloc[0], _DEFAULT_WEIGHTS)
            self.assertEqual(res["received_nested_list"].iloc[0], _DEFAULT_NESTED_LIST)

        mv = self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (valid_input, check_result)},
            skip_rest_api_test=True,
        )

        # Test SPLIT format with top-level params
        endpoint = self._ensure_ingress_url(mv)
        split_payload = {
            "dataframe_split": {
                "index": [0],
                "columns": ["value"],
                "data": [[10.0]],
            },
            "params": {
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
                "bytes_param": b"hello".hex(),  # hex string for JSON serialization
                "timestamp_param": datetime.datetime(2025, 1, 2, 3, 4, 5).isoformat(),  # ISO string for JSON
                "weights_param": [4.5, 3.5, 2.5],
                "nested_list": [[4, 3], [2, 1]],
            },
        }

        response = self._make_rest_api_request(endpoint, split_payload, "predict")
        response.raise_for_status()

        # Parse response: {"data": [[0, {"input_value": val, ...}]]}
        response_data = response.json()["data"]
        res_row = response_data[0][1]

        self.assertAlmostEqual(res_row["input_value"], 10.0, places=5)
        self.assertEqual(res_row["received_int8"], 10)
        self.assertEqual(res_row["received_int16"], 200)
        self.assertEqual(res_row["received_int32"], 3000)
        self.assertEqual(res_row["received_int64"], 40000)
        self.assertEqual(res_row["received_uint8"], 15)
        self.assertEqual(res_row["received_uint16"], 300)
        self.assertEqual(res_row["received_uint32"], 4000)
        self.assertEqual(res_row["received_uint64"], 50000)
        self.assertAlmostEqual(res_row["received_float"], 1.25, places=5)
        self.assertAlmostEqual(res_row["received_double"], 2.75, places=5)
        self.assertEqual(res_row["received_bool"], False)
        self.assertEqual(res_row["received_string"], "custom_value")
        # Inference server converts bytes to hex string for output
        # TODO (SNOW-3045092): Fix byte output serialization issue
        self.assertEqual(res_row["received_bytes"], b"hello".hex())
        # Model normalizes both datetime objects and ISO strings to same format
        self.assertEqual(res_row["received_timestamp"], _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5)))
        self.assertEqual(res_row["received_weights"], [4.5, 3.5, 2.5])
        self.assertEqual(res_row["received_nested_list"], [[4, 3], [2, 1]])

    def test_rest_api_records_format_with_params(self) -> None:
        """Test RECORDS format with top-level params."""
        model = ModelWithAllDataTypes(custom_model.ModelContext())
        sig = self._get_model_with_params_signature()

        # Deploy model with flat format first
        valid_input = pd.DataFrame(
            {
                "value": [1.0],
            }
        )

        def check_result(res: pd.DataFrame) -> None:
            self.assertAlmostEqual(res["input_value"].iloc[0], 1.0, places=5)
            # Validate params passed via flat format columns
            self.assertEqual(res["received_int8"].iloc[0], 1)
            self.assertEqual(res["received_int16"].iloc[0], 2)
            self.assertEqual(res["received_int32"].iloc[0], 3)
            self.assertEqual(res["received_int64"].iloc[0], 4)
            self.assertEqual(res["received_uint8"].iloc[0], 5)
            self.assertEqual(res["received_uint16"].iloc[0], 6)
            self.assertEqual(res["received_uint32"].iloc[0], 7)
            self.assertEqual(res["received_uint64"].iloc[0], 8)
            self.assertAlmostEqual(res["received_float"].iloc[0], 1.5, places=5)
            self.assertAlmostEqual(res["received_double"].iloc[0], 2.5, places=5)
            self.assertEqual(res["received_bool"].iloc[0], True)
            self.assertEqual(res["received_string"].iloc[0], "default")
            self.assertEqual(res["received_bytes"].iloc[0], b"default".hex().upper())
            self.assertEqual(res["received_timestamp"].iloc[0], _DEFAULT_TIMESTAMP_STR)
            self.assertEqual(res["received_weights"].iloc[0], _DEFAULT_WEIGHTS)
            self.assertEqual(res["received_nested_list"].iloc[0], _DEFAULT_NESTED_LIST)

        mv = self._test_registry_model_deployment(
            model=model,
            signatures={"predict": sig},
            prediction_assert_fns={"predict": (valid_input, check_result)},
            skip_rest_api_test=True,
        )

        # Test RECORDS format with top-level params
        endpoint = self._ensure_ingress_url(mv)
        records_payload = {
            "dataframe_records": [{"value": 10.0}],
            # Note: bytes and datetime must be serialized for JSON (hex string and ISO format)
            "params": {
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
                "bytes_param": b"hello".hex(),  # hex string for JSON serialization
                "timestamp_param": datetime.datetime(2025, 1, 2, 3, 4, 5).isoformat(),  # ISO string for JSON
                "weights_param": [4.5, 3.5, 2.5],
                "nested_list": [[4, 3], [2, 1]],
            },
        }

        response = self._make_rest_api_request(endpoint, records_payload, "predict")
        response.raise_for_status()

        response_data = response.json()["data"]
        res_row = response_data[0][1]

        self.assertAlmostEqual(res_row["input_value"], 10.0, places=5)
        self.assertEqual(res_row["received_int8"], 10)
        self.assertEqual(res_row["received_int16"], 200)
        self.assertEqual(res_row["received_int32"], 3000)
        self.assertEqual(res_row["received_int64"], 40000)
        self.assertEqual(res_row["received_uint8"], 15)
        self.assertEqual(res_row["received_uint16"], 300)
        self.assertEqual(res_row["received_uint32"], 4000)
        self.assertEqual(res_row["received_uint64"], 50000)
        self.assertAlmostEqual(res_row["received_float"], 1.25, places=5)
        self.assertAlmostEqual(res_row["received_double"], 2.75, places=5)
        self.assertEqual(res_row["received_bool"], False)
        self.assertEqual(res_row["received_string"], "custom_value")
        # Inference server converts bytes to hex string for output
        self.assertEqual(res_row["received_bytes"], b"hello".hex())
        # Model normalizes both datetime objects and ISO strings to same format
        self.assertEqual(res_row["received_timestamp"], _format_timestamp(datetime.datetime(2025, 1, 2, 3, 4, 5)))
        self.assertEqual(res_row["received_weights"], [4.5, 3.5, 2.5])
        self.assertEqual(res_row["received_nested_list"], [[4, 3], [2, 1]])


if __name__ == "__main__":
    absltest.main()
