from __future__ import annotations  # for return self methods

from typing import Any, Optional, Union
from unittest import TestCase

from snowflake import snowpark
from snowflake.ml._internal.utils import string_matcher as smatcher
from snowflake.ml.test_utils import mock_data_frame, mock_snowml_base
from snowflake.snowpark import Session


class MockSession(mock_snowml_base.MockSnowMLBase):
    """Mock version of snowflake.snowpark.Session.

    This class simulates a Snowpark session and allows to return canned responses to specific queries. Currently only
    calls to `Session.sql(query: str)` are supported.

    To add query-response pairs for unit-testing, call add_session_mock_sql(query, response). The calls are assumed to
    be executed in order and not repeated.

    When `session.sql(query)` is called, the `query` is checked against the next expected call. If it matches, the
    pre-defined response is returned, if `query` does not match the next expected call, an assertion will fail.

    Example:

        from snowflake.snowpark import Row

        # Setup:
        session = MockSession()

        session.add_mock_sql(
            query="CREATE DATABASE TEST;",
            result=MockDataFrame())

        # Code to be tested calls:
        #   session.sql("CREATE DATABASE TEST;").collect()

        # Result is [Row(status='Database TEST successfully created.')]

    TODOs:
        * Create a robust SQL string matcher that closely models actual query equivalence.
          * Criteria should be: whitespace independence, accurate treatment of case, including quoted identifiers, ...
            If the query change does not create different results, it should not fail the assertion (within limits).
        * Cover more functionality e.g. get_current_(account|database|...),  (add|clear|get)_(imports|packages),
          create_dataframe, etc.
        * Allow a test mode with unordered result matching (i.e. use internal dict not list, both should be possible).
        * Allow some side-effects e.g. a call to `use_schema` can change the result of `current_schema()`. This might be
          hard in general.
    """

    def __init__(self, conn: Any, test_case: TestCase | None, check_call_sequence_completion: bool = True) -> None:
        super().__init__(check_call_sequence_completion)
        self._test_case = test_case if test_case else TestCase()

    @property  # type: ignore[misc]
    def __class__(self) -> type[Session]:  # type: ignore[override]
        return Session

    # Helpers for setting up the MockDataFrame.

    def add_mock_sql(
        self,
        query: str,
        result: mock_data_frame.MockDataFrame,
        matcher: Union[
            type[smatcher.StringMatcherSql], type[smatcher.StringMatcherIgnoreWhitespace]
        ] = smatcher.StringMatcherSql,
        *,
        params: Optional[list[Any]] = None,
    ) -> mock_snowml_base.MockSnowMLBase:
        """Add an expected query-result pair to the session."""
        return self.add_operation(
            operation="sql", args=(), kwargs={"query": matcher(query), "params": params}, result=result
        )

    # DataFrame Operations

    def get_current_role(self) -> Any:
        mo = self._check_operation(
            operation="get_current_role",
            check_args=False,
            check_kwargs=False,
        )
        return mo.result

    def sql(self, query: str, params: Optional[list[Any]] = None) -> Any:
        """Execute a mock sql call.

        This will compare the `query` string against the stored expected calls and if it matches, the corresponding
        result will be returned.

        String normalization is performed on the given `query` before matching. This tries to ensure that immaterial
        changes to the query string (e.g. whitespace) do not break the test.

        Args:
            query (str): Query string.
            params (list[Any]): Query parameters.

        Returns:
            MockDataFrame result given.
        """
        mo = self._check_operation(
            operation="sql",
            args=(),
            kwargs={"query": query, "params": params},
            check_args=False,
            check_kwargs=True,
        )
        return mo.result

    def add_mock_query_history(self, result: snowpark.QueryHistory) -> mock_snowml_base.MockSnowMLBase:
        """Add an expected query history to the session."""
        return self.add_operation(operation="query_history", args=(), kwargs={}, result=result)

    def query_history(self) -> Any:
        """Execute a mock query_history call."""
        mo = self._check_operation(
            operation="query_history",
            args=(),
            kwargs={},
            check_args=False,
            check_kwargs=False,
        )
        return mo.result

    def get_current_database(self) -> Optional[str]:
        return None

    def get_current_schema(self) -> Optional[str]:
        return None

    def get_current_warehouse(self) -> Optional[str]:
        return None
