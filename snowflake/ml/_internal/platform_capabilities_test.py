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

    def test_enabled_inline_deployment_spec_bool(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: True,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_bool(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: False,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_is_inlined_deployment_spec_enabled_false(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES="{ }")]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_inline_deployment_spec_true(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: True,
                            }
                        )
                    )
                ]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_inline_deployment_spec_str(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(FEATURES=json.dumps({platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: False}))]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_inline_deployment_spec_int(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: 1,
                            }
                        )
                    )
                ],
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_int(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        FEATURES=json.dumps(
                            {
                                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: 0,
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

    @parameterized.product(inline_deployment_spec=[True, False], live_commit=[True, False])  # type: ignore[misc]
    def test_mocking(self, inline_deployment_spec: bool, live_commit: bool) -> None:
        """Test mocking of platform capabilities."""
        with platform_capabilities.PlatformCapabilities.mock_features(
            {
                platform_capabilities.INLINE_DEPLOYMENT_SPEC_PARAMETER: inline_deployment_spec,
                platform_capabilities.LIVE_COMMIT_PARAMETER: live_commit,
            }
        ):
            pc = platform_capabilities.PlatformCapabilities.get_instance()
            self.assertEqual(pc.is_inlined_deployment_spec_enabled(), inline_deployment_spec, "Inline deployment spec")
            self.assertEqual(pc.is_live_commit_enabled(), live_commit, "Live commit")


if __name__ == "__main__":
    absltest.main()
