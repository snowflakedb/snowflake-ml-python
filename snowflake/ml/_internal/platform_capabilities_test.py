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

    def test_nested_function_enabled(self) -> None:
        """Test is_nested_function_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(FEATURES='{ "SPCS_MODEL_ENABLE_EMBEDDED_SERVICE_FUNCTIONS": true }')]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertTrue(pc.is_nested_function_enabled())

    def test_nested_function_disabled(self) -> None:
        """Test is_nested_function_enabled method."""
        self._add_session_mock_sql(
            query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(FEATURES='{ "SPCS_MODEL_ENABLE_EMBEDDED_SERVICE_FUNCTIONS": false }')]
            ),
        )

        pc = platform_capabilities.PlatformCapabilities(session=cast(snowpark_session.Session, self._session))
        self.assertFalse(pc.is_nested_function_enabled())


if __name__ == "__main__":
    absltest.main()
