import json
import time
from typing import Any, Optional

import pandas as pd
import pytest
from absl.testing import absltest, parameterized
from sklearn.ensemble import RandomForestRegressor

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
        # if not self._has_image_override() or not self.PROXY_IMAGE_PATH:
        #     self.skipTest(
        #         "Skipping inference table tests: image override environment variables not set. "
        #         "Required: BUILDER_IMAGE_PATH, BASE_CPU_IMAGE_PATH, BASE_GPU_IMAGE_PATH, PROXY_IMAGE_PATH"
        #     )
        self.skipTest("Skipping inference table tests. Pending SPCS cluster upgrade.")

        self.model_name = f"inference_table_model_{self._run_id}"
        self.version_name = "v1"

    def _deploy_simple_sklearn_model(
        self, autocapture_param: Optional[bool] = None, autocapture_deployment: Optional[bool] = None
    ) -> ModelVersion:
        """Deploy a simple sklearn model for testing.

        Args:
            autocapture_param: Whether to set session parameter FEATURE_MODEL_INFERENCE_AUTOCAPTURE
                (True=ENABLED, False=DISABLED, None=don't set)
            autocapture_deployment: Whether to enable autocapture in deployment spec
                (True/False/None passed to experimental_options)
        """
        # Set Snowflake session parameter if specified
        if autocapture_param is not None:
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

        # Build experimental_options based on autocapture_deployment
        experimental_options = None
        if autocapture_deployment is not None:
            experimental_options = {"autocapture": autocapture_deployment}

        # Use the base class method for deployment with autocapture in experimental_options
        return self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns=prediction_assert_fns,
            sample_input_data=X,
            experimental_options=experimental_options,
        )

    def _extract_logs(self, mv: ModelVersion, max_wait_seconds: int = 30) -> tuple[list[dict[str, Any]], list[str]]:
        """Extract both inference logs and system logs from SPCS service."""
        # Get the actual service name from the deployed service
        services_df = mv.list_services()
        if len(services_df) == 0:
            raise RuntimeError("No services found for the model version")
        service_name = services_df.iloc[0]["name"]
        inference_logs = []
        system_logs = []

        # Wait for logs to appear and be processed
        for attempt in range(max_wait_seconds):
            try:
                logs_query = f"""
                SELECT system$get_service_logs(
                    '{service_name}',
                    '0', 'proxy'
                )
                """

                raw_logs = self.session.sql(logs_query).collect()
                print(f"DEBUG: Raw logs: {raw_logs}")

                for log_row in raw_logs:
                    log_text = log_row[0]
                    # Split the big blob by escaped newlines
                    log_lines = log_text.replace("\\n", "\n").split("\n")

                    for line in log_lines:
                        print(f"DEBUG: Line: {line}")
                        line = line.strip()

                        # Extract inference logs (slog JSON)
                        if '"msg":"inference_logs"' in line and '"severity_text":"INFO"' in line:
                            try:
                                json_start = line.find('{"time":')
                                if json_start >= 0:
                                    json_text = line[json_start:]
                                    log_json = json.loads(json_text)
                                    if (
                                        log_json.get("msg") == "inference_logs"
                                        and log_json.get("severity_text") == "INFO"
                                    ):
                                        inference_logs.append(log_json)
                            except json.JSONDecodeError as e:
                                print(f"DEBUG: Failed to parse JSON: {e}, line: {line[:100]}...")
                                continue

                        # Extract system logs (processing path indicators)
                        if any(
                            keyword in line
                            for keyword in [
                                "Completed batch processing",
                                "Completed streaming request processing",
                                "Processing completed batch",
                                "Processing streaming pass-through request",
                            ]
                        ):
                            system_logs.append(line)

                if len(inference_logs) > 0:
                    return inference_logs, system_logs

                time.sleep(1)  # Wait and retry

            except Exception as e:
                if attempt == max_wait_seconds - 1:
                    raise e
                time.sleep(1)

        return inference_logs, system_logs

    def _verify_log_structure(self, log_entry: dict[str, Any]) -> None:
        """Verify the slog JSON has the expected inference table schema with flat attributes."""
        # Standard slog fields
        self.assertIn("time", log_entry)
        self.assertIn("level", log_entry)
        self.assertEqual(log_entry["msg"], "inference_logs")

        # Our custom fields
        self.assertEqual(log_entry["severity_text"], "INFO")
        self.assertEqual(log_entry["body"], "inference_logs")
        self.assertIn("attributes", log_entry)

        # Verify attributes structure with flat dot-notation keys
        attrs = log_entry["attributes"]

        # Verify request data fields (at least one request data field should exist)
        request_data_keys = [k for k in attrs.keys() if k.startswith("snow.model_serving.request.data.")]
        self.assertGreater(len(request_data_keys), 0, "Should have at least one request data field")

        # Verify request timestamp
        self.assertIn("snow.model_serving.request.timestamp", attrs, "Should have request timestamp")
        self.assertIsInstance(attrs["snow.model_serving.request.timestamp"], str, "Request timestamp should be string")

        # Verify response data fields (at least one response data field should exist)
        response_data_keys = [k for k in attrs.keys() if k.startswith("snow.model_serving.response.data.")]
        self.assertGreater(len(response_data_keys), 0, "Should have at least one response data field")

        # Verify response timestamp and code
        self.assertIn("snow.model_serving.response.timestamp", attrs, "Should have response timestamp")
        self.assertIsInstance(
            attrs["snow.model_serving.response.timestamp"], str, "Response timestamp should be string"
        )
        self.assertIn("snow.model_serving.response.code", attrs, "Should have response code")
        self.assertIsInstance(attrs["snow.model_serving.response.code"], int, "Response code should be integer")

        # Verify model metadata
        self.assertIn("snow.model.function.name", attrs, "Should have model function name")
        self.assertIn("snow.model.id", attrs, "Should have model id")
        self.assertIn("snow.model.name", attrs, "Should have model name")
        self.assertIn("snow.model.version.id", attrs, "Should have model version id")
        self.assertIn("snow.model.version.name", attrs, "Should have model version name")

    def _verify_processing_path(self, system_logs: list[str], expected_path: str) -> None:
        """Verify the correct processing path was taken."""
        batch_logs = [log for log in system_logs if "Completed batch processing" in log]
        streaming_logs = [log for log in system_logs if "Completed streaming request processing" in log]
        if expected_path == "batch":

            self.assertGreater(len(batch_logs), 0, "Expected batch path but found no 'Completed batch processing' logs")

        elif expected_path == "streaming":
            self.assertGreater(
                len(streaming_logs),
                0,
                "Expected streaming path but found no 'Completed streaming request processing' logs",
            )
        else:
            self.fail(f"Unknown expected_path: {expected_path}")

    def test_inference_table_batch_path(self):
        """Test inference data capture to table for small requests that trigger batch processing."""
        mv = self._deploy_simple_sklearn_model(autocapture_param=True, autocapture_deployment=True)
        endpoint = self._ensure_ingress_url(mv)

        # Small request < 1KB -> should trigger batch path
        test_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Send request and verify response works
        """
        response = self._inference_using_rest_api(
            test_input, endpoint=endpoint, jwt_token_generator=self._get_jwt_token_generator(), target_method="predict"
        )
        """
        response = self._inference_using_rest_api(test_input, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(response)
        self.assertEqual(len(response), len(test_input), "Response should match input size")

        # Wait for logs to be processed
        time.sleep(3)

        # Extract both inference and system logs in one pass
        inference_logs, system_logs = self._extract_logs(mv)

        # Verify inference data capture worked
        self.assertGreater(len(inference_logs), 0, "Should have inference logs for batch request")

        # Verify log structure
        for log in inference_logs:
            self._verify_log_structure(log)

        # Verify batch path was used
        self._verify_processing_path(system_logs, "batch")

    def test_inference_table_streaming_path(self):
        """Test inference data capture to table for large requests that trigger streaming processing."""
        mv = self._deploy_simple_sklearn_model(autocapture_param=True, autocapture_deployment=True)
        endpoint = self._ensure_ingress_url(mv)

        # Large request > 1KB -> should trigger streaming path
        # Create many rows with valid float data to exceed 1KB threshold
        large_data = [float(i) for i in range(200)]  # 200 rows of float data
        test_input = pd.DataFrame({"feature": large_data})  # Should be > 1KB when serialized

        # Send request and verify response works
        """
        response = self._inference_using_rest_api(
            test_input, endpoint=endpoint, jwt_token_generator=self._get_jwt_token_generator(), target_method="predict"
        )
        """
        response = self._inference_using_rest_api(test_input, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(response)
        self.assertEqual(len(response), len(test_input), "Response should match input size")

        # Wait for logs to be processed
        time.sleep(3)

        # Extract both inference and system logs in one pass
        inference_logs, system_logs = self._extract_logs(mv)

        # Verify inference data capture worked
        self.assertGreater(len(inference_logs), 0, "Should have inference logs for streaming request")

        # Verify log structure
        for log in inference_logs:
            self._verify_log_structure(log)

        # Verify streaming path was used
        self._verify_processing_path(system_logs, "streaming")

    @parameterized.parameters(  # type: ignore[misc]
        {
            "autocapture_param": False,
            "autocapture_deployment": False,
            "test_description": "both param and deployment disabled",
        },
        {
            "autocapture_param": True,
            "autocapture_deployment": False,
            "test_description": "deployment spec overrides session param (param=ENABLED, deployment=False)",
        },
    )
    def test_inference_table_disabled_scenarios(
        self,
        autocapture_param: Optional[bool],
        autocapture_deployment: Optional[bool],
        test_description: str,
    ):
        """Test scenarios where inference data capture should be disabled.

        Args:
            autocapture_param: Session parameter setting (True/False/None)
            autocapture_deployment: Deployment spec setting (True/False/None)
            test_description: Description of the test scenario
        """
        mv = self._deploy_simple_sklearn_model(
            autocapture_param=autocapture_param, autocapture_deployment=autocapture_deployment
        )
        endpoint = self._ensure_ingress_url(mv)

        # Small request that would normally trigger batch data capture
        test_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Send request and verify response works
        response = self._inference_using_rest_api(test_input, endpoint=endpoint, target_method="predict")
        self.assertIsNotNone(response)
        self.assertEqual(len(response), len(test_input), "Response should match input size")

        # Wait for any potential logs
        time.sleep(3)

        # Extract logs
        inference_logs, system_logs = self._extract_logs(mv)

        # Verify NO inference logs are captured in all disabled scenarios
        self.assertEqual(len(inference_logs), 0, f"Should have NO inference logs when {test_description}")

        # But system logs should still exist (proxy still works)
        self.assertGreater(len(system_logs), 0, "System logs should still exist when autocapture is disabled")


if __name__ == "__main__":
    absltest.main()
