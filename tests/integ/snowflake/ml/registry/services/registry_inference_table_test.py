import base64
import datetime
import json
import time
from typing import Any, Optional

import pandas as pd
import pytest
import requests
from absl.testing import absltest, parameterized
from sklearn.ensemble import RandomForestRegressor

from snowflake.ml.model import ModelVersion, custom_model, model_signature
from tests.integ.snowflake.ml.registry.services.registry_model_deployment_test_base import (
    RegistryModelDeploymentTestBase,
)


class ModelWithScalarParams(custom_model.CustomModel):
    """Custom model with scalar parameters for testing autocapture with records format."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        temperature: float = 0.2,  # intentionally different from ParamSpec default
        max_tokens: int = 10,  # intentionally different from ParamSpec default
    ) -> pd.DataFrame:
        input_value = input_df["value"].iloc[0]
        output_value = input_value * temperature + max_tokens
        return pd.DataFrame(
            {
                "output": [output_value],
                "received_temperature": [temperature],
                "received_max_tokens": [max_tokens],
            }
        )


class ModelWithComplexParams(custom_model.CustomModel):
    """Custom model with complex param types (list, nested list, bool)."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        learning_rate: float = -1.0,
        use_gpu: bool = False,
        weights: list[float] = [],  # noqa: B006
        nested_config: list[list[int]] = [],  # noqa: B006
    ) -> pd.DataFrame:
        input_value = input_df["value"].iloc[0]
        weights_sum = sum(weights) if weights else 0.0
        return pd.DataFrame(
            {
                "output": [input_value * learning_rate + weights_sum],
                "received_learning_rate": [learning_rate],
                "received_use_gpu": [use_gpu],
                "received_weights": [weights],
                "received_nested_config": [nested_config],
            }
        )


class ModelWithBytesAndDatetimeParams(custom_model.CustomModel):
    """Custom model with bytes and datetime param types for testing autocapture coverage.

    The inference server passes params as their serialized forms (base64 strings for bytes,
    ISO strings for datetime), so this model handles conversion internally.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        data_bytes: bytes = b"",
        timestamp: datetime.datetime = datetime.datetime(2020, 1, 1, 0, 0, 0),  # noqa: B008
    ) -> pd.DataFrame:
        # Handle bytes: base64 string from inference server
        if isinstance(data_bytes, str):
            data_bytes = base64.b64decode(data_bytes)

        # Handle timestamp: ISO string from inference server
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.fromisoformat(timestamp)

        input_value = input_df["value"].iloc[0]
        bytes_len = len(data_bytes) if data_bytes else 0
        ts_year = timestamp.year if timestamp else 0
        return pd.DataFrame(
            {
                "output": [input_value + bytes_len + ts_year],
                "received_bytes_length": [bytes_len],
                "received_timestamp_year": [ts_year],
                "received_bytes_decoded": [data_bytes.decode("utf-8") if data_bytes else ""],
                "received_timestamp_iso": [timestamp.isoformat() if timestamp else ""],
            }
        )


class ModelWithNullableFeatures(custom_model.CustomModel):
    """Custom model that accepts nullable string features for testing null value autocapture.

    Also exposes predict_proba (API path predict-proba) for testing hyphenated method name autocapture.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    def _predict_impl(self, input_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in input_df.iterrows():
            a = row["feature_a"]
            b = row["feature_b"]
            label = f"{a}_{b}" if b is not None and pd.notna(b) else f"{a}_missing"
            results.append({"output": label})
        return pd.DataFrame(results)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return self._predict_impl(input_df)

    @custom_model.inference_api
    def predict_proba(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Exposed as predict-proba in the API; tests autocapture with hyphenated method name."""
        return self._predict_impl(input_df)


@pytest.mark.spcs_deployment_image
class RegistryInferenceTableTest(RegistryModelDeploymentTestBase):
    """Integration tests for inference request/response data capture to inference table."""

    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()

    def _deploy_simple_sklearn_model(
        self,
        autocapture_param: bool,
        autocapture_deployment: bool,
        service_name: Optional[str] = None,
    ) -> ModelVersion:
        """Deploy a simple sklearn model for testing.

        Args:
            autocapture_param: Whether to set session parameter FEATURE_MODEL_INFERENCE_AUTOCAPTURE
                (True=ENABLED, False=DISABLED)
            autocapture_deployment: Whether to enable autocapture in deployment spec (True/False)
            service_name: Optional service name for deployment
        """
        # Set Snowflake session parameter if specified
        parameter_value = "ENABLED" if autocapture_param else "DISABLED"
        param = "FEATURE_MODEL_INFERENCE_AUTOCAPTURE"
        try:
            self.session.sql(f"ALTER SESSION SET {param} = {parameter_value}").collect()
        except Exception as e:
            if autocapture_param:
                self.skipTest(f"Failed to set {param} parameter: {e}")
            else:
                # If we can't set to DISABLED, that's fine - it defaults to DISABLED
                print(f"DEBUG: Note: Could not set {param} to DISABLED: {e}")

        # Create simple model
        model = RandomForestRegressor(n_estimators=2, random_state=42, max_depth=2)
        X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
        y = [10.0, 20.0, 30.0, 40.0]
        model.fit(X, y)

        prediction_assert_fns = {
            "predict": (
                pd.DataFrame({"feature": [1.5, 2.5]}),
                lambda res: self.assertEqual(len(res), 2),
            )
        }

        # Use the base class method for deployment with autocapture
        return self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=X,
            autocapture=autocapture_deployment,
            service_name=service_name,
        )

    def _query_inference_table(
        self,
        mv: ModelVersion,
        service_name: str,
        expected_record_count: int,
        timeout_seconds: int,
    ) -> pd.DataFrame:
        """Query INFERENCE_TABLE and wait for expected number of records.

        Args:
            mv: Model version to query
            service_name: Name of the service
            expected_record_count: Expected number of records
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            DataFrame with inference table results

        Raises:
            TimeoutError: If expected records are not found within timeout
        """
        # Get model information
        services_df = mv.list_services()
        if len(services_df) == 0:
            raise RuntimeError("No services found for the model version")

        # Get model name from ModelVersion object
        model_name = mv.model_name

        # Enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE parameter for INFERENCE_TABLE function
        # This is required to query the inference table regardless of capture settings
        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
            print("DEBUG: Enabled FEATURE_MODEL_INFERENCE_AUTOCAPTURE for INFERENCE_TABLE query")
        except Exception as e:
            print(f"DEBUG: Warning - Could not enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        start_time = time.time()
        last_count = 0

        for _attempt in range(timeout_seconds):
            # Query INFERENCE_TABLE
            query = f"SELECT * FROM TABLE(INFERENCE_TABLE('{model_name}', SERVICE => '{service_name}'))"
            print(f"DEBUG: Querying inference table: {query}")

            result_df = self.session.sql(query).to_pandas()
            record_count = len(result_df)
            last_count = record_count

            print(f"DEBUG: Found {record_count} records (expected {expected_record_count})")

            if record_count == expected_record_count:
                self.assertEqual(
                    record_count, expected_record_count, f"Expected {expected_record_count} records, got {record_count}"
                )
                return result_df

            # Check if we've exceeded timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                break

            time.sleep(1)

        # Timeout reached without finding expected records
        raise TimeoutError(
            f"Timeout after {timeout_seconds} seconds: Expected {expected_record_count} records, "
            f"but found {last_count} records in INFERENCE_TABLE for model '{model_name}' "
            f"and service '{service_name}'"
        )

    def _verify_list_service(self, mv: ModelVersion, expected_autocapture: bool) -> str:
        """Verify list_service show autocapture is enabled or disabled on the service.

        Args:
            mv: ModelVersion model version.
            expected_autocapture: bool expected value for autocapture_enabeld column.
        """
        services = mv.list_services()
        self.assertIn(
            "autocapture_enabled",
            services.columns,
            "Expect autocapture_enabled column to exist",
        )
        actual_autocapture = services.loc[0, "autocapture_enabled"]
        self.assertEqual(
            actual_autocapture,
            expected_autocapture,
            "Actual autocapture_enabled value does not match with expected",
        )

    def _assert_autocapture_record(
        self,
        record_attributes: dict,
        expected_request_data: dict[str, Any] | None = None,
        expected_response_data: dict[str, Any] | None = None,
        expected_params: dict[str, Any] | None = None,
        unexpected_params: list[str] | None = None,
        record_idx: int = 0,
    ) -> None:
        """Validate an autocapture record's request data, response data, params, and standard metadata.

        Args:
            record_attributes: Parsed RECORD_ATTRIBUTES dict from the inference table.
            expected_request_data: Feature name -> expected value mapping for request data.
            expected_response_data: Output name -> expected value mapping for response data.
            expected_params: Param name -> expected value mapping (should be captured).
            unexpected_params: Param names that should NOT appear in the record.
            record_idx: Index for error messages.
        """
        prefix = f"Record {record_idx}"

        # Standard metadata keys must always be present
        for key in [
            "snow.model_serving.request.timestamp",
            "snow.model_serving.response.timestamp",
            "snow.model_serving.response.code",
            "snow.model_serving.function.name",
        ]:
            self.assertIn(key, record_attributes, f"{prefix}: Missing {key}")

        if expected_request_data:
            for name, expected_value in expected_request_data.items():
                key = f"snow.model_serving.request.data.{name}"
                self.assertIn(key, record_attributes, f"{prefix}: Expected feature '{name}' to be captured")
                if isinstance(expected_value, float):
                    self.assertAlmostEqual(record_attributes[key], expected_value, places=5, msg=f"{prefix}: {name}")
                else:
                    self.assertEqual(record_attributes[key], expected_value, f"{prefix}: {name}")

        if expected_response_data:
            for name, expected_value in expected_response_data.items():
                key = f"snow.model_serving.response.data.{name}"
                self.assertIn(key, record_attributes, f"{prefix}: Expected response '{name}' to be captured")
                if isinstance(expected_value, float):
                    self.assertAlmostEqual(record_attributes[key], expected_value, places=5, msg=f"{prefix}: {name}")
                else:
                    self.assertEqual(record_attributes[key], expected_value, f"{prefix}: {name}")

        if expected_params:
            for name, expected_value in expected_params.items():
                key = f"snow.model_serving.request.params.{name}"
                self.assertIn(key, record_attributes, f"{prefix}: Expected param '{name}' to be captured")
                if isinstance(expected_value, float):
                    self.assertAlmostEqual(record_attributes[key], expected_value, places=5, msg=f"{prefix}: {name}")
                else:
                    self.assertEqual(record_attributes[key], expected_value, f"{prefix}: {name}")

        if unexpected_params:
            for name in unexpected_params:
                key = f"snow.model_serving.request.params.{name}"
                self.assertNotIn(key, record_attributes, f"{prefix}: Param '{name}' should NOT be captured")

    def _create_payload_for_protocol(self, input_data: pd.DataFrame, protocol: str) -> dict[str, Any]:
        """Create a payload for the specified protocol format.

        Args:
            input_data: Input dataframe
            protocol: Protocol format ('dataframe_split' or 'dataframe_records')

        Returns:
            Dict with protocol as key and JSON data as value
        """
        # Extract orient by removing 'dataframe_' prefix
        orient = protocol.replace("dataframe_", "")
        # Convert dataframe to JSON using the appropriate orient
        json_data = input_data.to_json(orient=orient)
        # Return dict with protocol as key
        return {protocol: json.loads(json_data)}

    def _create_payload_for_protocol_with_extra_columns(
        self, input_data: pd.DataFrame, protocol: str
    ) -> dict[str, Any]:
        """Create a payload with extra columns for the specified protocol format.

        Args:
            input_data: Input dataframe
            protocol: Protocol format ('dataframe_split' or 'dataframe_records')

        Returns:
            Dict with protocol data and extra_columns list
        """
        # Copy input_data and add extra columns
        data_with_extras = input_data.copy()
        data_with_extras["extra_string_col"] = "test_value"
        data_with_extras["extra_timestamp_col"] = pd.Timestamp("2024-01-01 12:00:00")

        # Get base payload
        payload = self._create_payload_for_protocol(data_with_extras, protocol)

        # Add extra_columns list
        payload["extra_columns"] = ["extra_string_col", "extra_timestamp_col"]

        return payload

    def test_inference_table_autocapture(self):
        """Test basic inference data capture to INFERENCE_TABLE."""
        # Generate unique service name for this test
        service_name = f"inference_service_test_{self._run_id}"

        mv = self._deploy_simple_sklearn_model(
            autocapture_param=True, autocapture_deployment=True, service_name=service_name
        )
        endpoint = self._ensure_ingress_url(mv)

        # List service show autocapture enabled
        self._verify_list_service(mv, expected_autocapture=True)

        # Send inference request
        test_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Send request and verify response works
        response = self._inference_using_rest_api(
            self._to_external_data_format(test_input), endpoint=endpoint, target_method="predict"
        )
        self.assertIsNotNone(response)
        self.assertEqual(len(response), len(test_input), "Response should match input size")

        # Query INFERENCE_TABLE expecting 7 records (3 from test + 4 from deployment validation)
        inference_results = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=7, timeout_seconds=120
        )
        self.assertIsNotNone(inference_results, "Should have inference table results")

    @parameterized.parameters(  # type: ignore[misc]
        {
            "autocapture_param": False,
            "autocapture_deployment": False,
            "test_description": "both param and deployment disabled",
        },
        {
            "autocapture_param": True,
            "autocapture_deployment": False,
            "test_description": "param enabled and deployment disabled",
        },
    )
    def test_inference_table_disabled_scenarios(
        self,
        autocapture_param: bool,
        autocapture_deployment: bool,
        test_description: str,
    ):
        """Test scenarios where create service succeeds but
        inference data capture should be disabled.

        Args:
            autocapture_param: Session parameter setting (True/False)
            autocapture_deployment: Deployment spec setting (True/False)
            test_description: Description of the test scenario
        """
        # Generate unique service name for this test
        service_name = f"inference_service_test_{self._run_id}"
        mv = self._deploy_simple_sklearn_model(
            autocapture_param=autocapture_param,
            autocapture_deployment=autocapture_deployment,
            service_name=service_name,
        )
        endpoint = self._ensure_ingress_url(mv)

        # List service show autocapture disabled when param is disabled or deployment flag is disabled
        self._verify_list_service(mv, expected_autocapture=False)

        # Send inference request
        test_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Send request and verify response works
        response = self._inference_using_rest_api(
            self._to_external_data_format(test_input), endpoint=endpoint, target_method="predict"
        )
        self.assertIsNotNone(response, f"Response should not be None when {test_description}")
        self.assertEqual(len(response), len(test_input), f"Response should match input size when {test_description}")

    def test_inference_table_service_creation_fail(self):
        """Test create service fails when autocapture param disabled but deployment enabled."""
        # Generate unique service name for this test
        service_name = f"inference_service_test_{self._run_id}"

        with self.assertRaises(Exception) as context:
            self._deploy_simple_sklearn_model(
                autocapture_param=False,
                autocapture_deployment=True,
                service_name=service_name,
            )
        self.assertIn("Feature autocapture is not supported.", str(context.exception))

    def test_autocapture_records_format_with_partial_params(self):
        """Test that autocapture only captures explicitly provided params, not defaults."""
        # TODO: Remove this once the proxy image that can handle partial parameter autocapture
        # is available in system repository.
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        service_name = f"autocapture_params_test_{self._run_id}"

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        model = ModelWithScalarParams(custom_model.ModelContext())
        sample_input = pd.DataFrame({"value": [1.0, 2.0]})
        sample_output = model.predict(sample_input, temperature=0.7, max_tokens=100)

        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=0.7,
            ),
            model_signature.ParamSpec(
                name="max_tokens",
                dtype=model_signature.DataType.INT64,
                default_value=100,
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        prediction_assert_fns: dict[str, tuple[pd.DataFrame, Any]] = {}
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=sample_input,
            signatures={"predict": sig},
            autocapture=True,
            service_name=service_name,
            skip_rest_api_test=True,
        )

        endpoint = self._ensure_ingress_url(mv)
        self._verify_list_service(mv, expected_autocapture=True)

        # Send request with only temperature param (not max_tokens)
        records_payload = {
            "dataframe_records": [{"value": 10.0}],
            "params": {"temperature": 0.9},
        }

        result_df = self._inference_using_rest_api(records_payload, endpoint=endpoint, target_method="predict")

        # Verify inference used provided temperature=0.9 and ParamSpec default max_tokens=100
        # output = 10.0 * 0.9 + 100 = 109.0
        self.assertAlmostEqual(result_df["output"].iloc[0], 109.0, places=5)
        self.assertAlmostEqual(result_df["received_temperature"].iloc[0], 0.9, places=5)
        self.assertEqual(result_df["received_max_tokens"].iloc[0], 100)

        # Query INFERENCE_TABLE expecting 1 record from the records_payload request
        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=1, timeout_seconds=120
        )
        self.assertIsNotNone(inference_table_df, "Should have inference table results")

        record_attributes = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[0])

        self._assert_autocapture_record(
            record_attributes,
            expected_request_data={"value": 10.0},
            expected_params={"temperature": 0.9},
            unexpected_params=["max_tokens"],
        )

    def test_autocapture_split_format_with_complex_params(self):
        """Test autocapture with split format and complex param types (list, nested list, bool).
        Only some params are provided; defaults should be excluded from capture.
        """
        # TODO: Remove this once the proxy image that can handle partial parameter autocapture
        # is available in system repository.
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        service_name = f"autocapture_split_complex_test_{self._run_id}"

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        model = ModelWithComplexParams(custom_model.ModelContext())
        sample_input = pd.DataFrame({"value": [1.0, 2.0]})
        sample_output = model.predict(
            sample_input,
            learning_rate=0.01,
            use_gpu=True,
            weights=[1.0, 2.0, 3.0],
            nested_config=[[1, 2], [3, 4]],
        )

        params = [
            model_signature.ParamSpec(name="learning_rate", dtype=model_signature.DataType.DOUBLE, default_value=0.01),
            model_signature.ParamSpec(name="use_gpu", dtype=model_signature.DataType.BOOL, default_value=False),
            model_signature.ParamSpec(
                name="weights",
                dtype=model_signature.DataType.DOUBLE,
                default_value=[1.0, 2.0, 3.0],
                shape=(-1,),
            ),
            model_signature.ParamSpec(
                name="nested_config",
                dtype=model_signature.DataType.INT64,
                default_value=[[1, 2], [3, 4]],
                shape=(2, 2),
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        prediction_assert_fns: dict[str, tuple[pd.DataFrame, Any]] = {}
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=sample_input,
            signatures={"predict": sig},
            autocapture=True,
            service_name=service_name,
            skip_rest_api_test=True,
        )

        endpoint = self._ensure_ingress_url(mv)
        self._verify_list_service(mv, expected_autocapture=True)

        # Send split format with weights (list) and use_gpu (bool) provided.
        # learning_rate (scalar) and nested_config (nested list) use defaults -- excluded from capture.
        split_payload = {
            "dataframe_split": {
                "index": [0],
                "columns": ["value"],
                "data": [[10.0]],
            },
            "params": {
                "weights": [4.5, 3.5, 2.5],
                "use_gpu": True,
            },
        }

        result_df = self._inference_using_rest_api(split_payload, endpoint=endpoint, target_method="predict")

        # output = 10.0 * 0.01 (default learning_rate) + sum([4.5, 3.5, 2.5]) = 0.1 + 10.5 = 10.6
        self.assertAlmostEqual(result_df["output"].iloc[0], 10.6, places=3)

        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=1, timeout_seconds=120
        )
        self.assertIsNotNone(inference_table_df, "Should have inference table results")

        record_attributes = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[0])

        self._assert_autocapture_record(
            record_attributes,
            expected_request_data={"value": 10.0},
            expected_params={"weights": "[4.5,3.5,2.5]", "use_gpu": True},
            unexpected_params=["learning_rate", "nested_config"],
        )

    def test_autocapture_split_format_with_bytes_and_datetime_params(self):
        """Test autocapture with dataframe_split format using bytes and datetime param types."""
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        service_name = f"autocapture_bytes_datetime_test_{self._run_id}"

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        model = ModelWithBytesAndDatetimeParams(custom_model.ModelContext())
        sample_input = pd.DataFrame({"value": [1.0, 2.0]})

        test_bytes = b"hello"
        test_timestamp = datetime.datetime(2025, 6, 15, 12, 30, 45)
        sample_output = model.predict(sample_input, data_bytes=test_bytes, timestamp=test_timestamp)

        params = [
            model_signature.ParamSpec(
                name="data_bytes",
                dtype=model_signature.DataType.BYTES,
                default_value=b"",
            ),
            model_signature.ParamSpec(
                name="timestamp",
                dtype=model_signature.DataType.TIMESTAMP_NTZ,
                default_value=datetime.datetime(2020, 1, 1, 0, 0, 0),
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        prediction_assert_fns: dict[str, tuple[pd.DataFrame, Any]] = {}
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=sample_input,
            signatures={"predict": sig},
            autocapture=True,
            service_name=service_name,
            skip_rest_api_test=True,
        )

        endpoint = self._ensure_ingress_url(mv)
        self._verify_list_service(mv, expected_autocapture=True)

        # Send dataframe_split format with bytes (base64 encoded) and datetime (ISO format) params
        request_bytes = b"test_data"
        request_timestamp = datetime.datetime(2026, 3, 20, 14, 45, 30)
        split_payload = {
            "dataframe_split": {
                "index": [0],
                "columns": ["value"],
                "data": [[10.0]],
            },
            "params": {
                "data_bytes": base64.b64encode(request_bytes).decode("utf-8"),
                "timestamp": request_timestamp.isoformat(),
            },
        }

        result_df = self._inference_using_rest_api(split_payload, endpoint=endpoint, target_method="predict")

        # Verify inference received correct params
        # output = 10.0 + len(b"test_data") + 2026 = 10 + 9 + 2026 = 2045
        self.assertAlmostEqual(result_df["output"].iloc[0], 2045.0, places=5)
        self.assertEqual(result_df["received_bytes_length"].iloc[0], 9)
        self.assertEqual(result_df["received_timestamp_year"].iloc[0], 2026)
        self.assertEqual(result_df["received_bytes_decoded"].iloc[0], "test_data")

        # Query INFERENCE_TABLE expecting 1 record
        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=1, timeout_seconds=120
        )
        self.assertIsNotNone(inference_table_df, "Should have inference table results")

        record_attributes = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[0])

        self._assert_autocapture_record(
            record_attributes,
            expected_request_data={"value": 10.0},
            expected_params={"data_bytes": base64.b64encode(request_bytes).decode("utf-8")},
        )
        self.assertIn(
            "snow.model_serving.request.params.timestamp",
            record_attributes,
            "Expected 'timestamp' to be captured since it was explicitly provided.",
        )

    def test_autocapture_external_function_formats(self):
        """Test autocapture with ExternalFunction FLAT and WIDE formats.

        This test verifies that:
        1. Inference works correctly for both FLAT and WIDE formats with params
        2. Autocapture captures feature data but NOT params for ExternalFunction formats

        Note: Params are intentionally NOT captured for ExternalFunction formats because:
        - Inference extracts params from the first record only (request-level)
        - Each record could have different param values in the data (user modified)
        - Capturing per-record param values would be misleading
        """
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        service_name = f"autocapture_external_format_test_{self._run_id}"

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        model = ModelWithScalarParams(custom_model.ModelContext())
        sample_input = pd.DataFrame({"value": [1.0, 2.0]})
        sample_output = model.predict(sample_input, temperature=0.7, max_tokens=100)

        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=0.7,
            ),
            model_signature.ParamSpec(
                name="max_tokens",
                dtype=model_signature.DataType.INT64,
                default_value=100,
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        prediction_assert_fns: dict[str, tuple[pd.DataFrame, Any]] = {}
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=sample_input,
            signatures={"predict": sig},
            autocapture=True,
            service_name=service_name,
            skip_rest_api_test=True,
        )

        endpoint = self._ensure_ingress_url(mv)
        self._verify_list_service(mv, expected_autocapture=True)

        # WIDE FORMAT TESTS

        # Test 1: WIDE format with all params
        # Format: [index, {value: v, temperature: t, max_tokens: m}]
        wide_all_payload = {"data": [[0, {"value": 10.0, "temperature": 0.9, "max_tokens": 150}]]}
        result_df = self._inference_using_rest_api(wide_all_payload, endpoint=endpoint, target_method="predict")

        # output = 10.0 * 0.9 + 150 = 159.0
        self.assertAlmostEqual(result_df["output"].iloc[0], 159.0, places=5)
        self.assertAlmostEqual(result_df["received_temperature"].iloc[0], 0.9, places=5)
        self.assertEqual(result_df["received_max_tokens"].iloc[0], 150)

        # Test 2: WIDE format with partial params (only temperature, max_tokens uses default)
        wide_partial_payload = {"data": [[1, {"value": 20.0, "temperature": 0.5}]]}
        result_df = self._inference_using_rest_api(wide_partial_payload, endpoint=endpoint, target_method="predict")

        # output = 20.0 * 0.5 + 100 (default) = 110.0
        self.assertAlmostEqual(result_df["output"].iloc[0], 110.0, places=5)
        self.assertAlmostEqual(result_df["received_temperature"].iloc[0], 0.5, places=5)
        self.assertEqual(result_df["received_max_tokens"].iloc[0], 100)  # default

        # Test 3: WIDE format with no params (all use defaults)
        wide_no_params_payload = {"data": [[2, {"value": 30.0}]]}
        result_df = self._inference_using_rest_api(wide_no_params_payload, endpoint=endpoint, target_method="predict")

        # output = 30.0 * 0.7 (default) + 100 (default) = 121.0
        self.assertAlmostEqual(result_df["output"].iloc[0], 121.0, places=5)
        self.assertAlmostEqual(result_df["received_temperature"].iloc[0], 0.7, places=5)  # default
        self.assertEqual(result_df["received_max_tokens"].iloc[0], 100)  # default

        # FLAT FORMAT TESTS

        # Test 4: FLAT format with all params
        # Format: [index, value, temperature, max_tokens]
        flat_all_payload = {"data": [[3, 40.0, 0.8, 200]]}
        result_df = self._inference_using_rest_api(flat_all_payload, endpoint=endpoint, target_method="predict")

        # output = 40.0 * 0.8 + 200 = 232.0
        self.assertAlmostEqual(result_df["output"].iloc[0], 232.0, places=5)
        self.assertAlmostEqual(result_df["received_temperature"].iloc[0], 0.8, places=5)
        self.assertEqual(result_df["received_max_tokens"].iloc[0], 200)

        # Test 5: FLAT format with partial params - should FAIL (row too short)
        # Format: [index, value, temperature] - missing max_tokens
        flat_partial_payload = {"data": [[4, 50.0, 0.6]]}
        with self.assertRaises(requests.exceptions.HTTPError) as context:
            self._inference_using_rest_api(flat_partial_payload, endpoint=endpoint, target_method="predict")
        self.assertEqual(context.exception.response.status_code, 400, "FLAT with partial params should fail with 400")

        # Test 6: FLAT format with no params - should FAIL (row too short)
        # Format: [index, value] - missing both params
        flat_no_params_payload = {"data": [[5, 60.0]]}
        with self.assertRaises(requests.exceptions.HTTPError) as context:
            self._inference_using_rest_api(flat_no_params_payload, endpoint=endpoint, target_method="predict")
        self.assertEqual(context.exception.response.status_code, 400, "FLAT with no params should fail with 400")

        # VERIFY AUTOCAPTURE
        # Only 4 successful records (Tests 1-4), failed requests (Tests 5-6) NOT captured
        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=4, timeout_seconds=120
        )
        self.assertIsNotNone(inference_table_df, "Should have inference table results")
        self.assertEqual(
            len(inference_table_df), 4, "Should have exactly 4 records (failed FLAT requests should NOT be captured)"
        )

        # Expected input->output pairs from the 4 successful requests (Tests 1-4)
        expected_pairs = {
            10.0: 159.0,  # Test 1: WIDE all params - value=10.0, output=10.0*0.9+150=159.0
            20.0: 110.0,  # Test 2: WIDE partial - value=20.0, output=20.0*0.5+100=110.0
            30.0: 121.0,  # Test 3: WIDE no params - value=30.0, output=30.0*0.7+100=121.0
            40.0: 232.0,  # Test 4: FLAT all params - value=40.0, output=40.0*0.8+200=232.0
        }
        captured_pairs = {}

        for i, row in inference_table_df.iterrows():
            record_attributes = json.loads(row["RECORD_ATTRIBUTES"])

            self._assert_autocapture_record(
                record_attributes,
                unexpected_params=["temperature", "max_tokens"],
                record_idx=i,
            )

            self.assertIn("snow.model_serving.request.data.value", record_attributes)
            self.assertIn("snow.model_serving.response.data.output", record_attributes)

            req_value = record_attributes["snow.model_serving.request.data.value"]
            resp_output = record_attributes["snow.model_serving.response.data.output"]
            captured_pairs[req_value] = resp_output

        # Verify all expected request->response pairs were captured correctly
        self.assertEqual(
            captured_pairs, expected_pairs, f"Captured pairs {captured_pairs} should match expected {expected_pairs}"
        )

    def test_autocapture_with_null_values(self):
        """Test that rows with null feature values are captured in the inference table, not dropped.

        Also tests hyphenated method name (predict_proba -> predict-proba) autocapture.
        When running against pp8, uses custom proxy image via SPCS_MODEL_INFERENCE_PROXY_CONTAINER_URL.
        """
        # Skip test unless image override env vars are set (uncomment to require full image override).
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        service_name = f"autocapture_null_test_{self._run_id}"

        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        model = ModelWithNullableFeatures(custom_model.ModelContext())
        sample_input = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": ["hello", "world"]})
        sample_output = model.predict(sample_input)

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
        )

        prediction_assert_fns: dict[str, tuple[pd.DataFrame, Any]] = {}
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=sample_input,
            signatures={"predict": sig, "predict_proba": sig},
            autocapture=True,
            service_name=service_name,
            skip_rest_api_test=True,
        )

        endpoint = self._ensure_ingress_url(mv)
        self._verify_list_service(mv, expected_autocapture=True)

        # Send two rows: one with all values present, one with a null feature_b
        records_payload = {
            "dataframe_records": [
                {"feature_a": 1.0, "feature_b": "hello"},
                {"feature_a": 2.0, "feature_b": None},
            ],
        }

        result_df = self._inference_using_rest_api(records_payload, endpoint=endpoint, target_method="predict")
        self.assertEqual(len(result_df), 2, "Response should contain 2 rows")

        # Both rows should be captured -- null values must not cause rows to be dropped
        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=2, timeout_seconds=120
        )
        self.assertIsNotNone(inference_table_df, "Should have inference table results")
        self.assertEqual(len(inference_table_df), 2, "Both rows (including null) should be captured")

        for i in range(len(inference_table_df)):
            record_attributes = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[i])
            self._assert_autocapture_record(record_attributes, record_idx=i)
            self.assertIn("snow.model_serving.request.data.feature_a", record_attributes)

        # Test hyphenated method name: predict_proba is exposed as predict-proba in the API; autocapture
        # must resolve it so the inference table gets a record with function name predict_proba.
        proba_payload = {
            "dataframe_records": [{"feature_a": 3.0, "feature_b": "proba"}],
        }
        result_proba = self._inference_using_rest_api(proba_payload, endpoint=endpoint, target_method="predict_proba")
        self.assertEqual(len(result_proba), 1, "predict_proba response should have 1 row")
        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=3, timeout_seconds=120
        )
        proba_records = [
            row
            for _, row in inference_table_df.iterrows()
            if json.loads(row["RECORD_ATTRIBUTES"]).get("snow.model_serving.function.name") == "predict_proba"
        ]
        self.assertEqual(
            len(proba_records),
            1,
            "Exactly one autocapture record should be for predict_proba (hyphenated method name)",
        )

    def test_autocapture_with_extra_columns(self):
        """Test that extra_columns are captured correctly in autocapture for both split and records formats."""
        # Enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE session parameter
        try:
            self.session.sql("ALTER SESSION SET FEATURE_MODEL_INFERENCE_AUTOCAPTURE = ENABLED").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable FEATURE_MODEL_INFERENCE_AUTOCAPTURE: {e}")

        # Create unique service name with run_id
        service_name = f"autocapture_extra_cols_test_{self._run_id}"

        # Deploy ModelWithScalarParams with autocapture=True
        model = ModelWithScalarParams(custom_model.ModelContext())
        sample_input = pd.DataFrame({"value": [1.0, 2.0]})
        sample_output = model.predict(sample_input, temperature=0.7, max_tokens=100)

        # Create signature with ParamSpec for temperature and max_tokens
        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=0.7,
            ),
            model_signature.ParamSpec(
                name="max_tokens",
                dtype=model_signature.DataType.INT64,
                default_value=100,
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        prediction_assert_fns: dict[str, tuple[pd.DataFrame, Any]] = {}
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=sample_input,
            signatures={"predict": sig},
            autocapture=True,
            service_name=service_name,
            skip_rest_api_test=True,
        )

        # Get endpoint and verify autocapture enabled
        endpoint = self._ensure_ingress_url(mv)
        self._verify_list_service(mv, expected_autocapture=True)

        # Send dataframe_split request with extra_columns
        input_data_split = pd.DataFrame({"value": [10.0]})
        split_payload = self._create_payload_for_protocol_with_extra_columns(input_data_split, "dataframe_split")
        result_split = self._inference_using_rest_api(split_payload, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(result_split, "Split format request should return a result")

        # Send dataframe_records request with extra_columns
        input_data_records = pd.DataFrame({"value": [20.0]})
        records_payload = self._create_payload_for_protocol_with_extra_columns(input_data_records, "dataframe_records")
        result_records = self._inference_using_rest_api(records_payload, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(result_records, "Records format request should return a result")

        # Query inference table expecting 2 records
        inference_table_df = self._query_inference_table(
            mv=mv, service_name=service_name, expected_record_count=2, timeout_seconds=120
        )
        self.assertIsNotNone(inference_table_df, "Should have inference table results")
        self.assertEqual(len(inference_table_df), 2, "Should have exactly 2 records in inference table")

        for idx in range(2):
            record_attributes = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[idx])

            self._assert_autocapture_record(record_attributes, record_idx=idx)
            self.assertIn("snow.model_serving.request.data.value", record_attributes)

            # Verify extra_columns captured correctly
            self.assertIn(
                "snow.model_serving.request.extra_columns.extra_string_col",
                record_attributes,
                f"Record {idx}: Expected extra_string_col to be captured in extra_columns",
            )
            self.assertEqual(
                record_attributes["snow.model_serving.request.extra_columns.extra_string_col"],
                "test_value",
                f"Record {idx}: Extra string column value should match",
            )

            self.assertIn(
                "snow.model_serving.request.extra_columns.extra_timestamp_col",
                record_attributes,
                f"Record {idx}: Expected extra_timestamp_col to be captured in extra_columns",
            )
            timestamp_value = record_attributes["snow.model_serving.request.extra_columns.extra_timestamp_col"]
            self.assertIsNotNone(timestamp_value, f"Record {idx}: Timestamp value should not be None")
            self.assertTrue(len(str(timestamp_value)) > 0, f"Record {idx}: Timestamp value should not be empty")

        # Verify the input values are correct for each record
        record_0_attrs = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[0])
        record_1_attrs = json.loads(inference_table_df["RECORD_ATTRIBUTES"].iloc[1])

        # The records may not be in order, so we need to check both possibilities
        values = {
            record_0_attrs["snow.model_serving.request.data.value"],
            record_1_attrs["snow.model_serving.request.data.value"],
        }
        expected_values = {10.0, 20.0}
        self.assertEqual(values, expected_values, "Should have captured both input values (10.0 and 20.0)")


if __name__ == "__main__":
    absltest.main()
