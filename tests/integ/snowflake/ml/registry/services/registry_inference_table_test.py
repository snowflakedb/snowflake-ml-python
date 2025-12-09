import time
from typing import Optional

import pandas as pd
import pytest
from absl.testing import absltest, parameterized
from sklearn.ensemble import RandomForestRegressor

from snowflake.ml._internal import platform_capabilities
from snowflake.ml.model import ModelVersion
from tests.integ.snowflake.ml.registry.services.registry_model_deployment_test_base import (
    RegistryModelDeploymentTestBase,
)


@pytest.mark.spcs_deployment_image
class RegistryInferenceTableTest(RegistryModelDeploymentTestBase):
    """Integration tests for inference request/response data capture to inference table."""

    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()

        # Skip tests if image override environment variables are not set
        # These tests require custom proxy image with inference table support
        if not self._has_image_override() or not self.PROXY_IMAGE_PATH:
            self.skipTest(
                "Skipping inference table tests: image override environment variables not set. "
                "Required: BUILDER_IMAGE_PATH, BASE_CPU_IMAGE_PATH, BASE_GPU_IMAGE_PATH, PROXY_IMAGE_PATH"
            )

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

        # TODO (SNOW-2862108): remove mock after GS release.
        with platform_capabilities.PlatformCapabilities.mock_features(
            {"FEATURE_MODEL_INFERENCE_AUTOCAPTURE": parameter_value}
        ):
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

    def _verify_list_service(
        self, mv: ModelVersion, autocapture_param: bool, expected_autocapture: Optional[bool] = None
    ) -> str:
        """Verify list_service show autocapture is enabled or disabled on the service.

        Args:
            mv: ModelVersion model version.
            autocapture_param: bool whether autocapture param is enabeld when call list service
            expected_autocapture: Optional[bool] expected value for autocapture_enabeld columns.
            None if param was disabled and column should be excluded.
        """
        if autocapture_param:
            # TODO (SNOW-2862108): remove mock after GS changes.
            # Platform capabilities should pick up ALTER SET SESSION PARAM value
            with platform_capabilities.PlatformCapabilities.mock_features(
                {"FEATURE_MODEL_INFERENCE_AUTOCAPTURE": "ENABLED"}
            ):
                services = mv.list_services()
                self.assertIn(
                    "autocapture_enabled",
                    services.columns,
                    "Expect autocapture_enabled column to exist when param is enabled",
                )
                actual_autocapture = services.loc[0, "autocapture_enabled"]
                self.assertEqual(
                    actual_autocapture,
                    expected_autocapture,
                    "Actual autocapture_enabled value does not match with expected",
                )
        else:
            with platform_capabilities.PlatformCapabilities.mock_features(
                {"FEATURE_MODEL_INFERENCE_AUTOCAPTURE": "DISABLED"}
            ):
                services = mv.list_services()
                self.assertNotIn(
                    "autocapture_enabled",
                    services.columns,
                    "Expect autocapture_enabled column to be excluded when param is disabled",
                )

    def test_inference_table_autocapture(self):
        """Test basic inference data capture to INFERENCE_TABLE."""
        # Generate unique service name for this test
        service_name = f"inference_service_test_{self._run_id}"

        mv = self._deploy_simple_sklearn_model(
            autocapture_param=True, autocapture_deployment=True, service_name=service_name
        )
        endpoint = self._ensure_ingress_url(mv)

        # List service show autocapture enabled with param enabled
        self._verify_list_service(mv, autocapture_param=True, expected_autocapture=True)

        # Send inference request
        test_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Send request and verify response works
        response = self._inference_using_rest_api(test_input, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(response)
        self.assertEqual(len(response), len(test_input), "Response should match input size")

        # Query INFERENCE_TABLE expecting 7 records (3 from test + 4 from deployment validation)
        # TODO (SNOW-2862108): uncomment after GS release caught up.
        # inference_results = self._query_inference_table(
        #     mv=mv, service_name=service_name, expected_record_count=7, timeout_seconds=120
        # )
        # self.assertIsNotNone(inference_results, "Should have inference table results")

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

        # List service show autocapture disabled when param is enabled and deployment is disabled.
        # And should exclude ``autocapture_enabled`` column in output when param is disabled
        expected_autocapture = False if autocapture_param else None
        self._verify_list_service(mv, autocapture_param=autocapture_param, expected_autocapture=expected_autocapture)

        # Send inference request
        test_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Send request and verify response works
        response = self._inference_using_rest_api(test_input, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(response, f"Response should not be None when {test_description}")
        self.assertEqual(len(response), len(test_input), f"Response should match input size when {test_description}")

    def test_inference_table_service_creation_fail(self):
        """Test create service fails when autocapture param disabled but deployment enabled."""
        # TODO (SNOW-2883225): Enable after fix failure
        self.skipTest("Skipping test_inference_table_service_creation_fail test: Re-enable after fix failure. ")

        # Generate unique service name for this test
        service_name = f"inference_service_test_{self._run_id}"

        with self.assertRaises(ValueError) as context:
            self._deploy_simple_sklearn_model(
                autocapture_param=False,
                autocapture_deployment=True,
                service_name=service_name,
            )
        self.assertIn("Invalid Argument: Autocapture feature is not supported", str(context.exception))


if __name__ == "__main__":
    absltest.main()
