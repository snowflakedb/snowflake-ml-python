import datetime
from typing import cast

from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml._internal.utils import rbac, sql_identifier
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Session


class RbacTest(parameterized.TestCase):
    """Testing RBAC utils."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
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

    @parameterized.parameters(  # type: ignore[misc]
        (True,),
        (False,),
    )
    def test_is_role_in_session(self, in_session: bool) -> None:
        """Test checking whether a role is active in the session."""
        self.add_session_mock_sql(
            query="SELECT IS_ROLE_IN_SESSION('PROD_OWNER')",
            result=mock_data_frame.MockDataFrame(collect_result=[snowpark.Row(in_session)]),
        )
        self.assertEqual(
            rbac.is_role_in_session(cast(Session, self._session), sql_identifier.SqlIdentifier("prod_owner")),
            in_session,
        )

    def test_is_role_in_session_escapes_role_name(self) -> None:
        """Test that quoted role names with single quotes are escaped."""
        self.add_session_mock_sql(
            query="""SELECT IS_ROLE_IN_SESSION('"it''s role"')""",
            result=mock_data_frame.MockDataFrame(collect_result=[snowpark.Row(False)]),
        )
        self.assertFalse(
            rbac.is_role_in_session(
                cast(Session, self._session), sql_identifier.SqlIdentifier("it's role", case_sensitive=True)
            )
        )


if __name__ == "__main__":
    absltest.main()
