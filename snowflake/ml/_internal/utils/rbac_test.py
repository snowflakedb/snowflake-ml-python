import datetime
from typing import cast

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal.utils import rbac
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Session


class RbacTest(absltest.TestCase):
    """Testing RBAC utils."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environemnts for testing."""
        self._session = mock_session.MockSession(conn=None, test_case=self)

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

    def add_session_mock_sql(self, query: str, result: mock_data_frame.MockDataFrame) -> None:
        """Helper to add expected sql calls."""
        self._session.add_mock_sql(query=query, result=result)

    def test_get_role_privileges(self) -> None:
        """Test the normal retrieval of privileges for a role."""
        self.add_session_mock_sql(
            query="SHOW GRANTS ON ACCOUNT",
            result=mock_data_frame.MockDataFrame(
                [
                    snowpark.Row(
                        created_on=datetime.datetime(2022, 5, 31, 15, 50, 23, 531000, tzinfo=datetime.timezone.utc),
                        privilege=rbac.PRIVILEGE_CREATE_DATABASE,
                        granted_on="ACCOUNT",
                        name="DEPLOYMENT",
                        granted_to="ROLE",
                        grantee_name="ROLE_A",
                        grant_option="false",
                        granted_by="SYSADMIN",
                    ),
                    snowpark.Row(
                        created_on=datetime.datetime(2022, 5, 31, 15, 50, 22, 531000, tzinfo=datetime.timezone.utc),
                        privilege=rbac.PRIVILEGE_CREATE_ROLE,
                        granted_on="ACCOUNT",
                        name="DEPLOYMENT",
                        granted_to="ROLE",
                        grantee_name="ROLE_B",
                        grant_option="false",
                        granted_by="SYSADMIN",
                    ),
                ],
            ),
        )
        self.assertEqual(
            rbac.get_role_privileges(cast(Session, self._session), "ROLE_A"), {rbac.PRIVILEGE_CREATE_DATABASE}
        )


if __name__ == "__main__":
    absltest.main()
