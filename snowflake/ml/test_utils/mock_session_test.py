from absl.testing.absltest import TestCase, main

from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row


class MockSessionTest(TestCase):
    """Testing MockSession function."""

    def test_sql_match_verify_result(self) -> None:
        """Test that MockSession responds correctly for a matching query."""
        query = "SELECT COUNT(*) FROM TEST;"
        result_df = mock_data_frame.MockDataFrame([Row(count=23)])
        session = mock_session.MockSession(conn=None, test_case=self)
        session.add_mock_sql(query=query, result=result_df)

        actual_df = session.sql(query)
        self.assertEqual(actual_df, result_df)
        self.assertEqual(actual_df.collect(), [Row(count=23)])

    def test_sql_no_match(self) -> None:
        """Test that MockSession fails with a non-matching query."""
        query = "SELECT COUNT(*) FROM TEST;"
        result_df = mock_data_frame.MockDataFrame()
        session = mock_session.MockSession(conn=None, test_case=self)
        session.add_mock_sql(query=query, result=result_df)

        with self.assertRaises(AssertionError):
            session.sql("SELECT * FROM ALL;")

    def test_get_current_role(self) -> None:
        """Test get_current_role normal operation."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(operation="get_current_role", result="SuperAdmin")
            self.assertEqual(session.get_current_role(), "SuperAdmin")

    def test_generic_operation_success(self) -> None:
        """Test the successfull validation of a dynamically added function call."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(
                operation="unknown_operation", args=("nkw",), kwargs={"kw": "v"}, result="unknown_result"
            )
            self.assertEqual(session.unknown_operation("nkw", kw="v"), "unknown_result")

    def test_generic_operation_failure(self) -> None:
        """Test the unsuccessful validation of a dynamically added function call."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(
                operation="unknown_operation",
                args=("wrong_nkw",),
                kwargs={"wrong_kw": "v"},
                result="unknown_result",
            )
            with self.assertRaises(AssertionError):
                session.unknown_operation("nkw", kw="v")

    def test_generic_operation_ignore_args_and_kwargs(self) -> None:
        """Test the validation of a dynamically added function call without checking arguments ."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(
                operation="unknown_operation",
                args=("wrong_nkw",),
                kwargs={"wrong_kw": "v"},
                result="unknown_result",
                check_args=False,
                check_kwargs=False,
            )

            # TODO(sdas): mypy fails on following line saying '"MockSession" has no attribute "unknown_operation"'.
            # [Suggested fix](https://stackoverflow.com/a/50889782) did not work. Investigate.
            self.assertEqual(session.unknown_operation("nkw", kw="v"), "unknown_result")

    def test_check_statement_params_success(self) -> None:
        """Test the unsuccessful validation of statement_params."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(
                operation="operation",
                args=("nkw",),
                kwargs={"kw": "v", "statement_params": {"project": "SnowML"}},
                result="result",
                check_args=False,
                check_kwargs=False,
                check_statement_params=True,
            )

            self.assertEqual(session.operation("nkw", kw="v", statement_params={"project": "SnowML"}), "result")

    def test_check_statement_params_failure(self) -> None:
        """Test the unsuccessful validation of statement_params."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(
                operation="operation",
                args=("nkw",),
                kwargs={"kw": "v", "statement_params": {"project": "SnowML"}},
                result="result",
                check_args=False,
                check_kwargs=False,
                check_statement_params=True,
            )

            with self.assertRaises(AssertionError):
                self.assertEqual(session.operation("nkw", kw="v", statement_params={"project": "SnowPark"}), "result")

    def test_check_statement_params_missing(self) -> None:
        """Test the unsuccessful validation of statement_params."""
        with mock_session.MockSession(conn=None, test_case=self) as session:
            session.add_operation(
                operation="operation",
                args=("nkw",),
                kwargs={"kw": "v", "statement_params": {"project": "SnowML"}},
                result="result",
                check_args=False,
                check_kwargs=False,
                check_statement_params=True,
            )

            with self.assertRaises(AssertionError):
                self.assertEqual(session.operation("nkw", kw="v"), "result")


if __name__ == "__main__":
    main()
