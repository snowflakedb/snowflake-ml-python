import json
from typing import cast

from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml._internal import platform_capabilities
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import session as snowpark_session


class PlatformCapabilitiesTest(parameterized.TestCase):
    """Testing PlatformCapabilities class."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = mock_session.MockSession(conn=None, test_case=self)

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

    def _add_session_mock_sql(self, query: str, result: mock_data_frame.MockDataFrame) -> None:
        """Helper to add expected sql calls."""
        self._session.add_mock_sql(query=query, result=result)

    def test_enabled_inline_deployment_spec_full_version(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: "1.8.6",
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_empty_string(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: "",
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inlined_deployment_spec_enabled_no_value(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES="{ }")]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_full_version(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps({platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: "999.999.999"})
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_inline_deployment_spec_minor_version(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: "1.9",
                            }
                        )
                    )
                ],
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_minor_version(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: "99.9",
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_live_commit_bool(self) -> None:
        """Test is_live_commit_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.LIVE_COMMIT_PARAMETER: True,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_live_commit_enabled())

    def test_disabled_live_commit_bool(self) -> None:
        """Test is_live_commit_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.LIVE_COMMIT_PARAMETER: False,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_live_commit_enabled())

    def test_is_live_commit_enabled_false(self) -> None:
        """Test is_live_commit_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES="{ }")]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_live_commit_enabled())

    def test_enabled_live_commit_true(self) -> None:
        """Test is_live_commit_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.LIVE_COMMIT_PARAMETER: True,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_live_commit_enabled())

    def test_enabled_set_module_functions_volatility_from_manifest_false(self) -> None:
        """Test is_set_module_functions_volatility_from_manifest method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES="{ }")]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_set_module_functions_volatility_from_manifest())

    def test_enabled_set_module_functions_volatility_from_manifest_true(self) -> None:
        """Test is_set_module_functions_volatility_from_manifest method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_set_module_functions_volatility_from_manifest())

    @parameterized.product(enabled_param_value=["ENABLED", "ENABLED_PUBLIC_PREVIEW"])  # type: ignore[misc]
    def test_enabled_feature_model_inference_autocapture(self, enabled_param_value: str) -> None:
        """Test is_inference_autocapture_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.FEATURE_MODEL_INFERENCE_AUTOCAPTURE: enabled_param_value,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inference_autocapture_enabled())

    @parameterized.product(disabled_param_value=["DISABLED", "DISABLED_PRIVATE_PREVIEW"])  # type: ignore[misc]
    def test_disabled_feature_model_inference_autocapture(self, disabled_param_value: str) -> None:
        """Test is_inference_autocapture_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.FEATURE_MODEL_INFERENCE_AUTOCAPTURE: disabled_param_value,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inference_autocapture_enabled())

    def test_unknown_value_feature_model_inference_autocapture(self) -> None:
        """Test is_inference_autocapture_enabled method raises ValueError for unknown values."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.FEATURE_MODEL_INFERENCE_AUTOCAPTURE: "UNKNOWN VALUE",
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        with self.assertRaises(ValueError) as context:
            pc.is_inference_autocapture_enabled()
        self.assertIn(
            "Invalid feature parameter value: UNKNOWN VALUE for feature FEATURE_MODEL_INFERENCE_AUTOCAPTURE",
            str(context.exception),
        )

    def test_disabled_feature_model_inference_autocapture_param_not_set(self) -> None:
        """Test is_inference_autocapture_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES="{ }")]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inference_autocapture_enabled())

    @parameterized.product(live_commit=[True, False])  # type: ignore[misc]
    def test_mocking(self, live_commit: bool) -> None:
        """Test mocking of platform capabilities."""
        with platform_capabilities.PlatformCapabilities.mock_features(
            {
                platform_capabilities.LIVE_COMMIT_PARAMETER: live_commit,
            }
        ):
            pc = platform_capabilities.PlatformCapabilities.get_instance()
            self.assertEqual(pc.is_live_commit_enabled(), live_commit, "Live commit")


if __name__ == "__main__":
    absltest.main()
