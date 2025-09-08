"""Integration tests for wide input format with QUOTED_IDENTIFIERS_IGNORE_CASE parameter.

Tests the end-to-end flow of:
1. Go proxy querying QUOTED_IDENTIFIERS_IGNORE_CASE session parameter
2. Writing parameter to sentinel file in shared volume
3. Python inference server reading sentinel and applying column resolution logic
4. REST API handling wide input format with UPPERCASE column names

Note: case_sensitive=True model logging is currently not tested here due to
known issues with inference server handling of case-sensitive column names.
This will be addressed in a separate test once the inference server behavior
is fixed to properly handle case-sensitive model signatures.
See: TODO(snowflake-dev) Add JIRA ticket for inference server case_sensitive handling
"""

import logging
from typing import Any

import pandas as pd
import requests
import xgboost
from absl.testing import absltest

from snowflake.ml._internal.utils import identifier, jwt_generator
from snowflake.ml.utils import authentication
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)


class TestRegistryQuotedIdentifiersInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Test QUOTED_IDENTIFIERS_IGNORE_CASE parameter handling in model deployment."""

    def setUp(self) -> None:
        super().setUp()
        # Create a simple XGBoost model for testing
        self._prepare_test_model()

    def _prepare_test_model(self) -> None:
        """Create a simple XGBoost model with mixed case column names."""
        import numpy as np

        np.random.seed(42)
        n_samples = 100

        # Create data with mixed case column names to test case sensitivity
        data = {
            "Feature_One": np.random.uniform(0, 10, n_samples),
            "FEATURE_TWO": np.random.uniform(0, 5, n_samples),
            "feature_three": np.random.randint(0, 3, n_samples),
            "Mixed_Case_Col": np.random.choice([0, 1], n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }

        self.train_data = pd.DataFrame(data)
        self.feature_cols = ["Feature_One", "FEATURE_TWO", "feature_three", "Mixed_Case_Col"]
        self.target_col = "target"

        # Train simple XGBoost model
        self.model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = self.train_data[self.feature_cols]
        y = self.train_data[self.target_col]
        self.model.fit(X, y)

        # Prepare test data with different case variations
        self.test_data_exact = self.train_data[self.feature_cols].head(2)

        # Test data with different cases (for case sensitivity testing)
        self.test_data_lower = self.test_data_exact.copy()
        self.test_data_lower.columns = [col.lower() for col in self.test_data_lower.columns]

        self.test_data_upper = self.test_data_exact.copy()
        self.test_data_upper.columns = [col.upper() for col in self.test_data_upper.columns]

    def _create_dictionary_payload(self, test_data: pd.DataFrame) -> dict[str, Any]:
        """Create dictionary format payload."""
        data_with_dict = []
        for idx, row in test_data.iterrows():
            row_dict = row.to_dict()
            data_with_dict.append([idx, row_dict])

        # DEBUG: Log what dictionary we're actually creating
        logger.info(f"🔍 DEBUG: Created dictionary payload with columns: {list(test_data.columns)}")
        logger.info(f"🔍 DEBUG: Dictionary content: {data_with_dict[0][1] if data_with_dict else 'No data'}")

        return {"data": data_with_dict}

    def _make_rest_api_call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        jwt_token_generator: jwt_generator.JWTGenerator,
        target_method: str = "predict",
    ) -> requests.Response:
        """Make a REST API call for testing."""
        response = requests.post(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json=payload,
            auth=authentication.SnowflakeJWTTokenAuth(
                jwt_token_generator=jwt_token_generator,
                role=identifier.get_unescaped_names(self.session.get_current_role()),
                endpoint=endpoint,
                snowflake_account_url=self.snowflake_account_url,
            ),
            timeout=30,
        )
        return response

    def _rest_api_call_with_dictionary(
        self, test_data: pd.DataFrame, endpoint: str, jwt_token_generator, target_method: str = "predict"
    ):
        """Helper to make dictionary format REST API calls."""
        payload = self._create_dictionary_payload(test_data)
        response = self._make_rest_api_call(endpoint, payload, jwt_token_generator, target_method)
        return response.status_code

    def test_wide_input_format_regression(self) -> None:
        """
        End-to-end regression test for wide input format path with parameter optimization.

        Tests the recent optimization changes to ensure no regression in:
        - Proxy parameter reading from account level
        - Column caching optimization (read columns only once)
        - Wide input format processing with nested objects
        - QUOTED_IDENTIFIERS_IGNORE_CASE parameter handling (false case only)
        """
        logger.info("🧪 WIDE INPUT FORMAT REGRESSION TEST")
        logger.info("   Testing proxy parameter → column cache → wide input processing")

        # Check that QUOTED_IDENTIFIERS_IGNORE_CASE is false at account level
        logger.info("🔍 Checking account-level QUOTED_IDENTIFIERS_IGNORE_CASE parameter...")
        try:
            result = self.session.sql("SHOW PARAMETERS LIKE 'QUOTED_IDENTIFIERS_IGNORE_CASE' IN ACCOUNT").collect()
            if result:
                param_value = str(result[0].value).lower()
                logger.info(f"Account-level QUOTED_IDENTIFIERS_IGNORE_CASE = {param_value}")

                if param_value != "false":
                    self.fail(
                        f"Regression test requires QUOTED_IDENTIFIERS_IGNORE_CASE = false at account level, "
                        f"but found: {param_value}. Cannot run test safely without affecting parallel tests."
                    )
            else:
                logger.info("No explicit account-level setting found - using default (false)")
        except Exception as e:
            logger.warning(f"Could not query account parameter (using default false): {e}")

        # Test parameter = false (safe for parallel tests)
        logger.info("\n" + "=" * 60)
        logger.info("🔬 TESTING QUOTED_IDENTIFIERS_IGNORE_CASE = false (account level)")
        logger.info("=" * 60)
        self._test_wide_input_regression_false()

    def test_quoted_identifiers_basic_functionality(self) -> None:
        """
        Basic inference functionality test for parameter optimization.

        Verifies that the Go proxy parameter caching optimization works correctly
        when QUOTED_IDENTIFIERS_IGNORE_CASE = false (account level).

        Tests:
        - Parameter is correctly read by proxy at startup
        - Parameter value propagates to Python inference server
        - REST API inference works with correct parameter behavior
        """
        logger.info("🧪 PARAMETER OPTIMIZATION TEST")

        # Check that QUOTED_IDENTIFIERS_IGNORE_CASE is false at account level
        logger.info("🔍 Checking account-level QUOTED_IDENTIFIERS_IGNORE_CASE parameter...")
        try:
            result = self.session.sql("SHOW PARAMETERS LIKE 'QUOTED_IDENTIFIERS_IGNORE_CASE' IN ACCOUNT").collect()
            if result:
                param_value = str(result[0].value).lower()
                logger.info(f"Account-level QUOTED_IDENTIFIERS_IGNORE_CASE = {param_value}")

                if param_value != "false":
                    self.fail(
                        f"Test requires QUOTED_IDENTIFIERS_IGNORE_CASE = false at account level, "
                        f"but found: {param_value}. Cannot run test safely without affecting parallel tests."
                    )
            else:
                logger.info("No explicit account-level setting found - using default (false)")
        except Exception as e:
            logger.warning(f"Could not query account parameter (using default false): {e}")

        # Test with parameter = false (safe for parallel tests)
        logger.info("\n" + "=" * 60)
        logger.info("🔬 TESTING QUOTED_IDENTIFIERS_IGNORE_CASE = false (account level)")
        logger.info("=" * 60)
        self._test_inference_with_parameter_false()

    def _test_inference_with_parameter_false(self) -> None:
        """Test inference functionality with QUOTED_IDENTIFIERS_IGNORE_CASE = false (account level)"""

        logger.info("📋 Testing with account-level parameter: false")

        # Use base class method for robust model deployment and testing
        logger.info("🚀 Using _test_registry_model_deployment() for robust testing...")

        def assert_prediction_shape(res):
            """Assert that prediction has expected shape and log success"""
            self.assertIsNotNone(res, "Prediction result must not be None")
            self.assertGreater(len(res), 0, "Prediction must return results")
            logger.info(f"✅ Base class prediction test passed: {res.shape}")

        # This does full end-to-end testing: model registration, service deployment, mv.run(), and REST API
        mv = self._test_registry_model_deployment(
            model=self.model,
            sample_input_data=self.train_data[self.feature_cols].head(1),
            prediction_assert_fns={"predict": (self.train_data[self.feature_cols].head(1), assert_prediction_shape)},
            service_name=f"basic_test_false_{self._run_id}",
        )
        logger.info("✅ Base class deployment and testing completed successfully")

        # Additional custom dictionary format testing using the deployed service
        logger.info("🧪 Testing custom dictionary format...")
        services = mv.list_services()  # This is the correct method, not get_service()

        if len(services) > 0:
            # Service is deployed, get endpoint for dictionary format testing
            endpoint = self._ensure_ingress_url(mv)
            self.assertIsNotNone(endpoint, "Ingress endpoint must be available")

            jwt_token_generator = self._get_jwt_token_generator()

            # Test dictionary format behavior with parameter = false
            # Model signature is stored in UPPERCASE (case_sensitive=False default)
            # Dictionary format must match stored signature regardless of QUOTED_IDENTIFIERS_IGNORE_CASE

            # Test 1: Exact case should FAIL (400) - doesn't match UPPERCASE stored signature
            try:
                exact_case_result = self._rest_api_call_with_dictionary(
                    self.train_data[self.feature_cols].head(1),
                    endpoint=endpoint,
                    jwt_token_generator=jwt_token_generator,
                )
                logger.info(f"   📋 Dictionary format (exact case): Status {exact_case_result}")

                # Exact case should fail because stored signature is UPPERCASE
                self.assertEqual(
                    exact_case_result,
                    400,
                    "Dictionary format with exact case should FAIL - stored signature is UPPERCASE "
                    "(case_sensitive=False default)",
                )

            except Exception as e:
                self.fail(f"Exact case dictionary testing failed unexpectedly: {e}")

            # Test 2: UPPERCASE case should SUCCEED (200) - matches UPPERCASE stored signature
            try:
                uppercase_data = self.train_data[self.feature_cols].head(1).copy()
                uppercase_data.columns = [col.upper() for col in uppercase_data.columns]

                uppercase_result = self._rest_api_call_with_dictionary(
                    uppercase_data,
                    endpoint=endpoint,
                    jwt_token_generator=jwt_token_generator,
                )
                logger.info(f"   📋 Dictionary format (uppercase): Status {uppercase_result}")

                # UPPERCASE should succeed because it matches stored signature
                self.assertEqual(
                    uppercase_result,
                    200,
                    "Dictionary format with UPPERCASE should SUCCEED - matches stored signature",
                )

            except Exception as e:
                self.fail(f"Uppercase dictionary testing failed unexpectedly: {e}")

            # Test 3: Verify flat format still works (should be unaffected by parameter optimization)
            logger.info("📊 Testing flat format with parameter optimization...")
            try:
                flat_result = self._inference_using_rest_api(
                    self.train_data[self.feature_cols].head(2),
                    endpoint=endpoint,
                    jwt_token_generator=jwt_token_generator,
                    target_method="predict",
                )
                logger.info(f"   ✅ Flat format: {flat_result.shape}")
                self.assertIsNotNone(flat_result, "Flat format should work with parameter optimization")
                self.assertGreater(len(flat_result), 0, "Flat format should return results")
            except Exception as e:
                self.fail(f"Flat format testing failed unexpectedly: {e}")

        else:
            self.fail("No services were deployed by base class method")

        logger.info("🏁 Parameter false testing complete")

    def _test_wide_input_regression_false(self) -> None:
        """Test wide input regression with QUOTED_IDENTIFIERS_IGNORE_CASE = false (account level)"""

        logger.info("📋 Testing with account-level parameter: false")

        # Use base class method for consistent service deployment
        logger.info("🚀 Using _test_registry_model_deployment() for regression testing...")

        def assert_regression_baseline(res):
            """Assert that basic prediction works as regression baseline"""
            self.assertIsNotNone(res, "Regression baseline: prediction must work")
            self.assertGreater(len(res), 0, "Regression baseline: must return predictions")
            logger.info(f"✅ Regression baseline passed: {res.shape}")

        # Use base class for robust deployment and basic testing
        mv = self._test_registry_model_deployment(
            model=self.model,
            sample_input_data=self.train_data[self.feature_cols].head(1),
            prediction_assert_fns={"predict": (self.train_data[self.feature_cols].head(1), assert_regression_baseline)},
            service_name=f"regression_test_false_{self._run_id}",
        )
        logger.info("✅ Base class deployment completed - now testing regression scenarios")

        # Get the deployed service for additional regression testing
        services = mv.list_services()
        self.assertGreater(len(services), 0, "Service must be deployed for regression testing")

        endpoint = self._ensure_ingress_url(mv)
        self.assertIsNotNone(endpoint, "Service endpoint must be available for regression testing")

        jwt_token_generator = self._get_jwt_token_generator()
        logger.info(f"🔗 Service ready for regression testing: {endpoint}")

        # Test 1: FLAT format regression test (ensure optimization didn't break flat format)
        logger.info("📊 Testing FLAT FORMAT (regression test)...")
        flat_format_passed = self._test_flat_format_regression(endpoint, jwt_token_generator)
        self.assertTrue(
            flat_format_passed, "FLAT FORMAT REGRESSION DETECTED! Parameter optimization broke flat format processing!"
        )

        # Test 2: Wide input format (the main regression test)
        logger.info("🌐 Testing WIDE INPUT FORMAT (regression test)...")
        wide_format_passed = self._test_wide_input_format_regression(endpoint, jwt_token_generator, False)

        # Wide input format should work - this is a regression test to ensure no breaking changes
        self.assertTrue(
            wide_format_passed,
            "WIDE INPUT FORMAT REGRESSION DETECTED! Parameter optimization broke wide input format processing!",
        )
        logger.info("✅ Wide input format regression test passed")

        # Test 3: Verify parameter was read correctly by proxy
        logger.info("🔍 Testing parameter propagation to inference server...")
        self._verify_parameter_propagation(endpoint, jwt_token_generator, False)
        logger.info("✅ Parameter propagation working correctly")

        logger.info("🏁 Wide input regression test complete for parameter: false")

    def _test_flat_format_regression(self, endpoint: str, jwt_token_generator) -> bool:
        """Test that flat format still works after parameter optimization changes."""

        try:
            # Use robust base class method for flat format testing
            logger.info("   📊 Testing flat format using base class method...")
            flat_result = self._inference_using_rest_api(
                self.train_data[self.feature_cols].head(2),  # Test with multiple rows
                endpoint=endpoint,
                jwt_token_generator=jwt_token_generator,
                target_method="predict",
            )
            logger.info(f"   ✅ Base class flat format test: {flat_result.shape}")
            return True

        except Exception as e:
            logger.error(f"   ❌ Flat format regression test failed: {e}")
            return False

    def _test_wide_input_format_regression(self, endpoint: str, jwt_token_generator, param_on: bool) -> bool:
        """Test wide input format with nested objects (triggers column caching optimization)."""

        try:
            # Create wide input format data (nested objects that require json_normalize)
            # For case_sensitive=False (default), model signature columns are stored in UPPERCASE
            # So we need to use UPPERCASE column names in the wide input format
            wide_input_payload = {
                "data": [
                    [0, {"FEATURE_ONE": 1.5, "FEATURE_TWO": 2.0, "FEATURE_THREE": 1, "MIXED_CASE_COL": 0}],
                    [1, {"FEATURE_ONE": 2.5, "FEATURE_TWO": 3.0, "FEATURE_THREE": 2, "MIXED_CASE_COL": 1}],
                ]
            }

            logger.info("   🌐 Testing wide input format (nested objects with UPPERCASE columns)...")
            response = self._make_rest_api_call(endpoint, wide_input_payload, jwt_token_generator)

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    "   ✅ Wide input format: Status %d, %d predictions",
                    response.status_code,
                    len(result.get("data", [])),
                )
                return True
            else:
                logger.error(f"   ❌ Wide input format failed: Status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"   ❌ Wide input format regression test failed: {e}")
            return False

    def _verify_parameter_propagation(self, endpoint: str, jwt_token_generator, param_on: bool):
        """Verify that QUOTED_IDENTIFIERS_IGNORE_CASE parameter was read correctly (param_on should always be False)."""

        if param_on:
            raise ValueError("Test only supports param_on=False to avoid affecting parallel tests")

        # With QUOTED_IDENTIFIERS_IGNORE_CASE = false and case_sensitive=False (default):
        # Model signature is stored in UPPERCASE, so exact case should FAIL, UPPERCASE should succeed

        # Test 1: Exact case should FAIL (400) - doesn't match UPPERCASE stored signature
        exact_case_data = {"Feature_One": 1.0, "FEATURE_TWO": 2.0, "feature_three": 1, "Mixed_Case_Col": 0}
        exact_payload = {"data": [[0, exact_case_data]]}

        exact_response = self._make_rest_api_call(endpoint, exact_payload, jwt_token_generator)
        logger.info(f"   📋 Exact case test: Status {exact_response.status_code}")

        # Exact case should fail because signature is stored in UPPERCASE
        self.assertEqual(
            exact_response.status_code,
            400,
            "Exact case dictionary format should FAIL with case_sensitive=False - signature stored in UPPERCASE",
        )

        # Test 2: UPPERCASE should SUCCEED (200) - matches UPPERCASE stored signature
        uppercase_data = {"FEATURE_ONE": 1.0, "FEATURE_TWO": 2.0, "FEATURE_THREE": 1, "MIXED_CASE_COL": 0}
        uppercase_payload = {"data": [[0, uppercase_data]]}

        uppercase_response = self._make_rest_api_call(endpoint, uppercase_payload, jwt_token_generator)
        logger.info(f"   📋 Uppercase case test: Status {uppercase_response.status_code}")

        self.assertEqual(
            uppercase_response.status_code,
            200,
            "UPPERCASE dictionary format should SUCCEED - matches UPPERCASE stored signature",
        )

        logger.info("   ✅ Parameter propagation verified: parameter=false, signature=UPPERCASE, behavior=correct")


if __name__ == "__main__":
    absltest.main()
