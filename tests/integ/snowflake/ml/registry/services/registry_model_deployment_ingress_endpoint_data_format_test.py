import logging
from typing import Any

import numpy as np
import pandas as pd
import requests
import retrying
import xgboost
from absl.testing import absltest

from snowflake.ml._internal.utils import identifier, jwt_generator
from snowflake.ml.model import model_signature
from snowflake.ml.utils import authentication
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)


class TestRegistryModelDeploymentIngressEndpointDataFormatInteg(
    registry_model_deployment_test_base.RegistryModelDeploymentTestBase
):
    def _make_ingress_endpoint_call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        jwt_token_generator: jwt_generator.JWTGenerator,
        target_method: str = "predict",
    ) -> requests.Response:
        """Make a REST API call to the ingress endpoint for testing."""
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

    def _create_json_payload_from_pandas_dataframe(self, test_data: pd.DataFrame) -> dict[str, Any]:
        """Create JSON payload from pandas DataFrame for REST API calls."""
        data_with_dict = []
        for idx, row in test_data.iterrows():
            row_dict = row.to_dict()
            data_with_dict.append([idx, row_dict])

        return {"data": data_with_dict}

    def _make_ingress_call_with_pandas_dataframe(
        self, test_data: pd.DataFrame, endpoint: str, jwt_token_generator, target_method: str = "predict"
    ):
        """Helper to make JSON format REST API calls via ingress endpoint."""
        payload = self._create_json_payload_from_pandas_dataframe(test_data)
        response = self._make_ingress_endpoint_call(endpoint, payload, jwt_token_generator, target_method)
        return response.status_code

    def _test_flat_format_inference(self, endpoint: str, payload: pd.DataFrame, jwt_token_generator) -> bool:
        """Test that flat format still works after parameter optimization changes."""

        try:
            # Use robust base class method for flat format testing
            logger.info("Testing flat format using base class method...")
            flat_result = self._inference_using_rest_api(
                payload,  # Test with multiple rows
                endpoint=endpoint,
                jwt_token_generator=jwt_token_generator,
                target_method="predict",
            )
            logger.info(f"Base class flat format test: {flat_result.shape}")
            return True

        except Exception as e:
            logger.error(f"Flat format inference test failed: {e}")
            return False

    def _test_json_format_inference(
        self, endpoint: str, payload: dict[str, Any], jwt_token_generator, expected_error_message: str = None
    ) -> bool:
        """Test JSON format inference with optional failure error message validation."""

        try:
            response = self._make_ingress_endpoint_call(endpoint, payload, jwt_token_generator)

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    "Wide input format: Status %d, %d predictions",
                    response.status_code,
                    len(result.get("data", [])),
                )
                return True
            else:
                error_text = response.text
                logger.error(f"Wide input format failed: Status {response.status_code}: {error_text}")

                if expected_error_message:
                    if expected_error_message in error_text:
                        logger.info(f"Got expected error: '{expected_error_message}'")
                        return False
                    else:
                        logger.error(f"Expected error '{expected_error_message}' but got: {error_text}")
                        raise AssertionError(
                            f"Wrong error type. Expected: '{expected_error_message}', Got: {error_text}"
                        )

                return False

        except Exception as e:
            logger.error(f"Wide input format regression test failed: {e}")

            if expected_error_message:
                if expected_error_message in str(e):
                    logger.info(f"Got expected error in exception: '{expected_error_message}'")
                    return False
                else:
                    logger.error(f"Expected error '{expected_error_message}' but got exception: {str(e)}")
                    raise AssertionError(
                        f"Wrong error type. Expected: '{expected_error_message}', Got exception: {str(e)}"
                    )

            return False

    def _assert_prediction_result(self, res):
        """Assert that prediction has expected result"""
        self.assertIsNotNone(res, "Prediction result must not be None")
        self.assertGreater(len(res), 0, "Prediction must return results")
        logger.info(f"Prediction result shape: {res.shape}")

    def _inference_using_rest_api(
        self,
        test_input: pd.DataFrame,
        *,
        endpoint: str,
        jwt_token_generator: jwt_generator.JWTGenerator,
        target_method: str,
    ) -> pd.DataFrame:
        """Override base class method to handle case sensitive method names.

        For case sensitive models, target_method comes as '"predict"' but the REST API
        endpoint should be /predict (without quotes).
        """
        # Strip quotes from target_method for URL construction
        clean_target_method = target_method.strip('"')

        test_input_arr = model_signature._convert_local_data_to_df(test_input).values
        test_input_arr = np.column_stack([range(test_input_arr.shape[0]), test_input_arr])
        res = retrying.retry(
            wait_exponential_multiplier=100,
            wait_exponential_max=4000,
            retry_on_result=(
                registry_model_deployment_test_base.RegistryModelDeploymentTestBase.retry_if_result_status_retriable
            ),
        )(requests.post)(
            f"https://{endpoint}/{clean_target_method.replace('_', '-')}",
            json={"data": test_input_arr.tolist()},
            auth=authentication.SnowflakeJWTTokenAuth(
                jwt_token_generator=jwt_token_generator,
                role=identifier.get_unescaped_names(self.session.get_current_role()),
                endpoint=endpoint,
                snowflake_account_url=self.snowflake_account_url,
            ),
        )
        res.raise_for_status()
        return pd.DataFrame([x[1] for x in res.json()["data"]])

    @absltest.skip("Skipping test until inference server release 0.0.21")
    def test_ingress_endpoint_with_case_insensitive_model(self):
        logger.info("Testing ingress endpoint with case insensitive model")

        n_samples = 100
        np.random.seed(42)

        data = {
            "feature_one": np.random.uniform(0, 10, n_samples),
            "FEATURE_TWO": np.random.uniform(0, 5, n_samples),
            "Feature_Three": np.random.randint(0, 3, n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }

        train_data = pd.DataFrame(data)
        feature_cols = ["feature_one", "FEATURE_TWO", "Feature_Three"]
        target_col = "target"

        model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = train_data[feature_cols]
        y = train_data[target_col]
        model.fit(X, y)

        logger.info(f"Model trained with features: {feature_cols}")

        # Model is logged without method options, case sensitive is False by default
        mv = self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_data[feature_cols].head(1),
            prediction_assert_fns={"predict": (train_data[feature_cols].head(1), self._assert_prediction_result)},
            service_name=f"case_insensitive_test_{self._run_id}",
        )
        logger.info("Model deployed successfully")

        services = mv.list_services()
        self.assertGreater(len(services), 0, "Service must be deployed")

        endpoint = self._ensure_ingress_url(mv)
        self.assertIsNotNone(endpoint, "Ingress endpoint must be available")

        jwt_token_generator = self._get_jwt_token_generator()
        logger.info(f"Testing with endpoint: {endpoint}")

        flat_format_success = self._test_flat_format_inference(
            endpoint, train_data[feature_cols].head(2), jwt_token_generator
        )
        self.assertTrue(flat_format_success, "Flat format inference should work")
        logger.info("Flat format test passed")

        exact_case_payload = {
            "data": [
                [0, {"feature_one": 1.5, "FEATURE_TWO": 2.0, "Feature_Three": 1}],
                [1, {"feature_one": 2.5, "FEATURE_TWO": 3.0, "Feature_Three": 2}],
            ]
        }

        exact_case_success = self._test_json_format_inference(endpoint, exact_case_payload, jwt_token_generator)
        self.assertTrue(exact_case_success, "Exact case should PASS")
        logger.info("Exact case test passed")

        uppercase_payload = {
            "data": [
                [0, {"FEATURE_ONE": 1.5, "FEATURE_TWO": 2.0, "FEATURE_THREE": 1}],
                [1, {"FEATURE_ONE": 2.5, "FEATURE_TWO": 3.0, "FEATURE_THREE": 2}],
            ]
        }

        uppercase_success = self._test_json_format_inference(endpoint, uppercase_payload, jwt_token_generator)
        self.assertTrue(uppercase_success, "Uppercase case should PASS - matches normalized identifier signature")
        logger.info("Uppercase case test passed")

        logger.info("Case insensitive model test completed successfully")

    @absltest.skip("Skipping test until inference server release 0.0.21")
    def test_ingress_endpoint_with_case_sensitive_model(self):
        logger.info("Testing ingress endpoint with case sensitive model")

        n_samples = 100
        np.random.seed(42)

        data = {
            "feature_one": np.random.uniform(0, 10, n_samples),
            "FEATURE_TWO": np.random.uniform(0, 5, n_samples),
            "Feature_Three": np.random.randint(0, 3, n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }

        train_data = pd.DataFrame(data)
        feature_cols = ["feature_one", "FEATURE_TWO", "Feature_Three"]
        target_col = "target"

        model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = train_data[feature_cols]
        y = train_data[target_col]
        model.fit(X, y)

        logger.info(f"Model trained with features: {feature_cols}")

        options = {"method_options": {"predict": {"case_sensitive": True}, "explain": {"case_sensitive": True}}}
        mv = self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_data[feature_cols].head(1),
            prediction_assert_fns={'"predict"': (train_data[feature_cols].head(1), self._assert_prediction_result)},
            service_name=f"case_sensitive_test_{self._run_id}",
            options=options,
        )
        logger.info("Case sensitive model deployed successfully")

        services = mv.list_services()
        self.assertGreater(len(services), 0, "Service must be deployed")

        endpoint = self._ensure_ingress_url(mv)
        self.assertIsNotNone(endpoint, "Ingress endpoint must be available")

        jwt_token_generator = self._get_jwt_token_generator()
        logger.info(f"Testing with endpoint: {endpoint}")

        flat_format_success = self._test_flat_format_inference(
            endpoint, train_data[feature_cols].head(2), jwt_token_generator
        )
        self.assertTrue(flat_format_success, "Flat format inference should work")
        logger.info("Flat format test passed")

        exact_case_payload = {
            "data": [
                [0, {"feature_one": 1.5, "FEATURE_TWO": 2.0, "Feature_Three": 1}],
                [1, {"feature_one": 2.5, "FEATURE_TWO": 3.0, "Feature_Three": 2}],
            ]
        }

        exact_case_success = self._test_json_format_inference(endpoint, exact_case_payload, jwt_token_generator)
        self.assertTrue(exact_case_success, "Exact case should PASS")
        logger.info("Exact case test passed")

        uppercase_payload = {
            "data": [
                [0, {"FEATURE_ONE": 1.5, "FEATURE_TWO": 2.0, "FEATURE_THREE": 1}],
                [1, {"FEATURE_ONE": 2.5, "FEATURE_TWO": 3.0, "FEATURE_THREE": 2}],
            ]
        }

        uppercase_success = self._test_json_format_inference(endpoint, uppercase_payload, jwt_token_generator)
        self.assertTrue(uppercase_success, "Uppercase case should PASS - matches normalized identifier signature")
        logger.info("Uppercase case test passed")

        logger.info("Case sensitive model test completed successfully")

    @absltest.skip("Skipping test until inference server release 0.0.21")
    def test_ingress_endpoint_with_quoted_case_insensitive_model_1(self):
        logger.info("Testing ingress endpoint with quoted case insensitive model")

        n_samples = 100
        np.random.seed(42)

        data = {
            "feature_1": np.random.uniform(0, 10, n_samples),
            "FEATURE_2": np.random.uniform(0, 5, n_samples),
            "FEAture_3": np.random.randint(0, 3, n_samples),
            '"feature_4"': np.random.uniform(0, 8, n_samples),
            '"FEATURE_5"': np.random.uniform(0, 6, n_samples),
            '"FEAture_6"': np.random.randint(0, 4, n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }

        # Some feature names are quoted
        train_data = pd.DataFrame(data)
        feature_cols = ["feature_1", "FEATURE_2", "FEAture_3", '"feature_4"', '"FEATURE_5"', '"FEAture_6"']
        target_col = "target"

        model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = train_data[feature_cols]
        y = train_data[target_col]
        model.fit(X, y)

        logger.info(f"Model trained with features (including quoted): {feature_cols}")

        # Model is logged without method options, case sensitive is False by default
        options = {"enable_explainability": False}
        mv = self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_data[feature_cols].head(1),
            prediction_assert_fns={"predict": (train_data[feature_cols].head(1), self._assert_prediction_result)},
            service_name=f"quoted_case_insensitive_test_{self._run_id}",
            options=options,
        )
        logger.info("Quoted case insensitive model deployed successfully")

        services = mv.list_services()
        self.assertGreater(len(services), 0, "Service must be deployed")

        endpoint = self._ensure_ingress_url(mv)
        self.assertIsNotNone(endpoint, "Ingress endpoint must be available")

        jwt_token_generator = self._get_jwt_token_generator()
        logger.info(f"Testing with endpoint: {endpoint}")

        flat_format_success = self._test_flat_format_inference(
            endpoint, train_data[feature_cols].head(2), jwt_token_generator
        )
        self.assertTrue(flat_format_success, "Flat format inference should work")
        logger.info("Flat format test passed")

        exact_case_payload = {
            "data": [
                [
                    0,
                    {
                        "feature_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEAture_3": 1,
                        '"feature_4"': 1.5,
                        '"FEATURE_5"': 2.5,
                        '"FEAture_6"': 0,
                    },
                ],
                [
                    1,
                    {
                        "feature_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEAture_3": 2,
                        '"feature_4"': 2.0,
                        '"FEATURE_5"': 3.0,
                        '"FEAture_6"': 1,
                    },
                ],
            ]
        }

        exact_case_success = self._test_json_format_inference(endpoint, exact_case_payload, jwt_token_generator)
        self.assertTrue(exact_case_success, "Exact case should PASS")
        logger.info("Exact case test passed")

        uppercase_payload = {
            "data": [
                [
                    0,
                    {
                        "FEATURE_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEATURE_3": 1,
                        '"FEATURE_4"': 1.5,  # Columns with quotes maintain their case with normalized identifier rule
                        '"FEATURE_5"': 2.5,  # Hence uppercasing all the columns should fail
                        '"FEATURE_6"': 0,
                    },
                ],
                [
                    1,
                    {
                        "FEATURE_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEATURE_3": 2,
                        '"FEATURE_4"': 2.0,
                        '"FEATURE_5"': 3.0,
                        '"FEATURE_6"': 1,
                    },
                ],
            ]
        }

        uppercase_success = self._test_json_format_inference(
            endpoint,
            uppercase_payload,
            jwt_token_generator,
            expected_error_message="Input data columns do not match the model signature",
        )
        self.assertFalse(uppercase_success, "Uppercase case should FAIL - column signature mismatch")
        logger.info("Uppercase case test failed with expected column mismatch error")

        # This payload is generated to test the signature column cache built for
        # wide input service function flow using the normalized identifier rule
        normalized_identifier_payload = {
            "data": [
                [
                    0,
                    {
                        "FEATURE_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEATURE_3": 1,
                        '"feature_4"': 1.5,  # Columns with quotes maintain their case with normalized identifier rule
                        "FEATURE_5": 2.5,  # Normalized identifier rule removes the quotes if possible
                        '"FEAture_6"': 0,  # Hence this payload should pass
                    },
                ],
                [
                    1,
                    {
                        "FEATURE_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEATURE_3": 2,
                        '"feature_4"': 2.0,
                        "FEATURE_5": 3.0,
                        '"FEAture_6"': 1,
                    },
                ],
            ]
        }

        normalized_identifier_success = self._test_json_format_inference(
            endpoint, normalized_identifier_payload, jwt_token_generator
        )
        self.assertTrue(
            normalized_identifier_success,
            "Normalized identifier case should PASS - matches normalized identifier signature",
        )
        logger.info("Normalized identifier case test passed")

        logger.info("Quoted case insensitive model test completed successfully")

    @absltest.skip("Skipping test until inference server release 0.0.21")
    def test_ingress_endpoint_with_quoted_case_insensitive_model_2(self):
        logger.info("Testing ingress endpoint with quoted case insensitive model")

        n_samples = 100
        np.random.seed(42)

        data = {
            "feature_1": np.random.uniform(0, 10, n_samples),
            "FEATURE_2": np.random.uniform(0, 5, n_samples),
            "FEAture_3": np.random.randint(0, 3, n_samples),
            '"feature_4"': np.random.uniform(0, 8, n_samples),
            '"FEATURE 5"': np.random.uniform(0, 6, n_samples),
            '"FEAture_6"': np.random.randint(0, 4, n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }

        # Some feature names are quoted and also have spaces
        train_data = pd.DataFrame(data)
        feature_cols = ["feature_1", "FEATURE_2", "FEAture_3", '"feature_4"', '"FEATURE 5"', '"FEAture_6"']
        target_col = "target"

        model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = train_data[feature_cols]
        y = train_data[target_col]
        model.fit(X, y)

        logger.info(f"Model trained with features (including quoted): {feature_cols}")

        # Model is logged without method options, case sensitive is False by default
        options = {"enable_explainability": False}
        mv = self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_data[feature_cols].head(1),
            prediction_assert_fns={"predict": (train_data[feature_cols].head(1), self._assert_prediction_result)},
            service_name=f"quoted_case_insensitive_test_{self._run_id}",
            options=options,
        )
        logger.info("Quoted case insensitive model deployed successfully")

        services = mv.list_services()
        self.assertGreater(len(services), 0, "Service must be deployed")

        endpoint = self._ensure_ingress_url(mv)
        self.assertIsNotNone(endpoint, "Ingress endpoint must be available")

        jwt_token_generator = self._get_jwt_token_generator()
        logger.info(f"Testing with endpoint: {endpoint}")

        flat_format_success = self._test_flat_format_inference(
            endpoint, train_data[feature_cols].head(2), jwt_token_generator
        )
        self.assertTrue(flat_format_success, "Flat format inference should work")
        logger.info("Flat format test passed")

        exact_case_payload = {
            "data": [
                [
                    0,
                    {
                        "feature_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEAture_3": 1,
                        '"feature_4"': 1.5,
                        '"FEATURE 5"': 2.5,
                        '"FEAture_6"': 0,
                    },
                ],
                [
                    1,
                    {
                        "feature_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEAture_3": 2,
                        '"feature_4"': 2.0,
                        '"FEATURE 5"': 3.0,
                        '"FEAture_6"': 1,
                    },
                ],
            ]
        }

        exact_case_success = self._test_json_format_inference(endpoint, exact_case_payload, jwt_token_generator)
        self.assertTrue(exact_case_success, "Exact case should PASS")
        logger.info("Exact case test passed")

        uppercase_payload = {
            "data": [
                [
                    0,
                    {
                        "FEATURE_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEATURE_3": 1,
                        '"FEATURE_4"': 1.5,  # Columns with quotes maintain their case with normalized identifier rule
                        '"FEATURE 5"': 2.5,  # Hence uppercasing all the columns should fail
                        '"FEATURE_6"': 0,
                    },
                ],
                [
                    1,
                    {
                        "FEATURE_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEATURE_3": 2,
                        '"FEATURE_4"': 2.0,
                        '"FEATURE 5"': 3.0,
                        '"FEATURE_6"': 1,
                    },
                ],
            ]
        }

        uppercase_success = self._test_json_format_inference(
            endpoint,
            uppercase_payload,
            jwt_token_generator,
            expected_error_message="Input data columns do not match the model signature",
        )
        self.assertFalse(uppercase_success, "Uppercase case should FAIL - column signature mismatch")
        logger.info("Uppercase case test failed with expected column mismatch error")

        # This payload is generated to test the signature column cache built for
        # wide input service function flow using the normalized identifier rule
        normalized_identifier_payload = {
            "data": [
                [
                    0,
                    {
                        "FEATURE_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEATURE_3": 1,
                        '"feature_4"': 1.5,  # Columns with quotes maintain their case with normalized identifier rule
                        '"FEATURE 5"': 2.5,  # Normalized identifier does not remove quotes here due to the space
                        '"FEAture_6"': 0,  # Hence this payload should pass
                    },
                ],
                [
                    1,
                    {
                        "FEATURE_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEATURE_3": 2,
                        '"feature_4"': 2.0,
                        '"FEATURE 5"': 3.0,
                        '"FEAture_6"': 1,
                    },
                ],
            ]
        }

        normalized_identifier_success = self._test_json_format_inference(
            endpoint, normalized_identifier_payload, jwt_token_generator
        )
        self.assertTrue(
            normalized_identifier_success,
            "Normalized identifier case should PASS - matches normalized identifier signature",
        )
        logger.info("Normalized identifier case test passed")

        logger.info("Quoted case insensitive model test completed successfully")

    @absltest.skip("Skipping test until inference server release 0.0.21")
    def test_ingress_endpoint_with_quoted_case_sensitive_model(self):
        logger.info("Testing ingress endpoint with quoted case sensitive model")

        n_samples = 100
        np.random.seed(42)

        data = {
            "feature_1": np.random.uniform(0, 10, n_samples),
            "FEATURE_2": np.random.uniform(0, 5, n_samples),
            "FEAture_3": np.random.randint(0, 3, n_samples),
            '"feature_4"': np.random.uniform(0, 8, n_samples),
            '"FEATURE_5"': np.random.uniform(0, 6, n_samples),
            '"FEAture_6"': np.random.randint(0, 4, n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }

        # Some feature names are quoted
        train_data = pd.DataFrame(data)
        feature_cols = ["feature_1", "FEATURE_2", "FEAture_3", '"feature_4"', '"FEATURE_5"', '"FEAture_6"']
        target_col = "target"

        model = xgboost.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = train_data[feature_cols]
        y = train_data[target_col]
        model.fit(X, y)

        logger.info(f"Model trained with features (including quoted): {feature_cols}")

        options = {"method_options": {"predict": {"case_sensitive": True}, "explain": {"case_sensitive": True}}}
        mv = self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_data[feature_cols].head(1),
            prediction_assert_fns={'"predict"': (train_data[feature_cols].head(1), self._assert_prediction_result)},
            service_name=f"quoted_case_sensitive_test_{self._run_id}",
            options=options,
        )
        logger.info("Quoted case sensitive model deployed successfully")

        services = mv.list_services()
        self.assertGreater(len(services), 0, "Service must be deployed")

        endpoint = self._ensure_ingress_url(mv)
        self.assertIsNotNone(endpoint, "Ingress endpoint must be available")

        jwt_token_generator = self._get_jwt_token_generator()
        logger.info(f"Testing with endpoint: {endpoint}")

        flat_format_success = self._test_flat_format_inference(
            endpoint, train_data[feature_cols].head(2), jwt_token_generator
        )
        self.assertTrue(flat_format_success, "Flat format inference should work")
        logger.info("Flat format test passed")

        exact_case_payload = {
            "data": [
                [
                    0,
                    {
                        "feature_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEAture_3": 1,
                        '"feature_4"': 1.5,
                        '"FEATURE_5"': 2.5,
                        '"FEAture_6"': 0,
                    },
                ],
                [
                    1,
                    {
                        "feature_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEAture_3": 2,
                        '"feature_4"': 2.0,
                        '"FEATURE_5"': 3.0,
                        '"FEAture_6"': 1,
                    },
                ],
            ]
        }

        exact_case_success = self._test_json_format_inference(endpoint, exact_case_payload, jwt_token_generator)
        self.assertTrue(exact_case_success, "Exact case should PASS")
        logger.info("Exact case test passed")

        uppercase_payload = {
            "data": [
                [
                    0,
                    {
                        "FEATURE_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEATURE_3": 1,
                        '"FEATURE_4"': 1.5,  # Columns with quotes maintain their case with normalized identifier rule
                        '"FEATURE_5"': 2.5,  # Hence uppercasing all the columns should fail
                        '"FEATURE_6"': 0,
                    },
                ],
                [
                    1,
                    {
                        "FEATURE_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEATURE_3": 2,
                        '"FEATURE_4"': 2.0,
                        '"FEATURE_5"': 3.0,
                        '"FEATURE_6"': 1,
                    },
                ],
            ]
        }

        uppercase_success = self._test_json_format_inference(
            endpoint,
            uppercase_payload,
            jwt_token_generator,
            expected_error_message="Input data columns do not match the model signature",
        )
        self.assertFalse(uppercase_success, "Uppercase case should FAIL - column signature mismatch")
        logger.info("Uppercase case test failed with expected column mismatch error")

        # This payload is generated to test the signature column cache built for
        # wide input service function flow using the normalized identifier rule
        normalized_identifier_payload = {
            "data": [
                [
                    0,
                    {
                        "FEATURE_1": 1.0,
                        "FEATURE_2": 2.0,
                        "FEATURE_3": 1,
                        '"feature_4"': 1.5,  # Columns with quotes maintain their case with normalized identifier rule
                        "FEATURE_5": 2.5,  # Normalized identifier removes the quotes if possible
                        '"FEAture_6"': 0,  # Hence this payload should pass
                    },
                ],
                [
                    1,
                    {
                        "FEATURE_1": 1.5,
                        "FEATURE_2": 2.5,
                        "FEATURE_3": 2,
                        '"feature_4"': 2.0,
                        "FEATURE_5": 3.0,
                        '"FEAture_6"': 1,
                    },
                ],
            ]
        }

        normalized_identifier_success = self._test_json_format_inference(
            endpoint, normalized_identifier_payload, jwt_token_generator
        )
        self.assertTrue(
            normalized_identifier_success,
            "Normalized identifier case should PASS - matches normalized identifier signature",
        )
        logger.info("Normalized identifier case test passed")

        logger.info("Quoted case sensitive model test completed successfully")


if __name__ == "__main__":
    absltest.main()
