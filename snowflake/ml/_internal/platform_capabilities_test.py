from typing import cast

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal import platform_capabilities
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import session as snowpark_session


class PlatformCapabilitiesTest(absltest.TestCase):
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
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES='{ "ENABLE_INLINE_DEPLOYMENT_SPEC": true }')]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_bool(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES='{ "ENABLE_INLINE_DEPLOYMENT_SPEC": false }')]),
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
                [snowpark.Row(FEATURES='{ "ENABLE_INLINE_DEPLOYMENT_SPEC": "true" }')]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_inline_deployment_spec_str(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(FEATURES='{ "ENABLE_INLINE_DEPLOYMENT_SPEC": "false" }')]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_enabled_inline_deployment_spec_int(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES='{ "ENABLE_INLINE_DEPLOYMENT_SPEC": 1 }')]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_inlined_deployment_spec_enabled())

    def test_disabled_inline_deployment_spec_int(self) -> None:
        """Test is_inlined_deployment_spec_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame([snowpark.Row(FEATURES='{ "ENABLE_INLINE_DEPLOYMENT_SPEC": 0 }')]),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_inlined_deployment_spec_enabled())

    def test_mocking(self) -> None:
        with platform_capabilities.PlatformCapabilities.mock_features({"ENABLE_INLINE_DEPLOYMENT_SPEC": True}):
            pc = platform_capabilities.PlatformCapabilities.get_instance()
            self.assertTrue(pc.is_inlined_deployment_spec_enabled())
        with platform_capabilities.PlatformCapabilities.mock_features({"ENABLE_INLINE_DEPLOYMENT_SPEC": False}):
            pc = platform_capabilities.PlatformCapabilities.get_instance()
            self.assertFalse(pc.is_inlined_deployment_spec_enabled())


if __name__ == "__main__":
    absltest.main()
