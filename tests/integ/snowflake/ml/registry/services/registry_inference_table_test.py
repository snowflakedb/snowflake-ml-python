import json
import time
from typing import Any, Optional

import pandas as pd
import pytest
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

        # temperature was explicitly provided -- should be captured with the correct value
        self.assertIn(
            "snow.model_serving.request.params.temperature",
            record_attributes,
            "Expected 'temperature' to be captured since it was explicitly provided.",
        )
        self.assertAlmostEqual(
            record_attributes["snow.model_serving.request.params.temperature"],
            0.9,
            places=5,
            msg="Captured temperature value should match the request-provided value.",
        )

        # max_tokens was NOT provided -- should not be captured
        self.assertNotIn(
            "snow.model_serving.request.params.max_tokens",
            record_attributes,
            "Expected 'max_tokens' to NOT be captured since only the default value was used.",
        )

        # Validate standard keys are present with correct values
        self.assertIn("snow.model_serving.request.data.value", record_attributes)
        self.assertAlmostEqual(
            record_attributes["snow.model_serving.request.data.value"],
            10.0,
            places=5,
            msg="Captured input feature value should match the request.",
        )
        self.assertIn("snow.model_serving.request.timestamp", record_attributes)
        self.assertIn("snow.model_serving.response.timestamp", record_attributes)
        self.assertIn("snow.model_serving.response.code", record_attributes)
        self.assertIn("snow.model_serving.function.name", record_attributes)

    def test_autocapture_split_format_with_complex_params(self):
        """Test autocapture with split format and complex param types (list, nested list, bool).
        Only some params are provided; defaults should be excluded from capture.
        """
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

        # Provided list param should be captured with correct value
        self.assertIn(
            "snow.model_serving.request.params.weights",
            record_attributes,
            "Expected list param 'weights' to be captured since it was explicitly provided.",
        )
        self.assertEqual(
            record_attributes["snow.model_serving.request.params.weights"],
            "[4.5,3.5,2.5]",
            "Captured weights value should match the request-provided list.",
        )
        # Provided bool param should be captured with correct value
        self.assertIn(
            "snow.model_serving.request.params.use_gpu",
            record_attributes,
            "Expected bool param 'use_gpu' to be captured since it was explicitly provided.",
        )
        self.assertEqual(
            record_attributes["snow.model_serving.request.params.use_gpu"],
            True,
            "Captured use_gpu value should be True.",
        )

        # Default scalar param should NOT be captured
        self.assertNotIn(
            "snow.model_serving.request.params.learning_rate",
            record_attributes,
            "Expected 'learning_rate' to NOT be captured since only the default was used.",
        )
        # Default nested list param should NOT be captured
        self.assertNotIn(
            "snow.model_serving.request.params.nested_config",
            record_attributes,
            "Expected 'nested_config' to NOT be captured since only the default was used.",
        )

        # Validate standard keys are present with correct values
        self.assertIn("snow.model_serving.request.data.value", record_attributes)
        self.assertAlmostEqual(
            record_attributes["snow.model_serving.request.data.value"],
            10.0,
            places=5,
            msg="Captured input feature value should match the request.",
        )
        self.assertIn("snow.model_serving.request.timestamp", record_attributes)
        self.assertIn("snow.model_serving.response.timestamp", record_attributes)
        self.assertIn("snow.model_serving.response.code", record_attributes)
        self.assertIn("snow.model_serving.function.name", record_attributes)


if __name__ == "__main__":
    absltest.main()
