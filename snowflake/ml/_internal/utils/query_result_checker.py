from __future__ import annotations  # for return self methods

from functools import partial
from typing import Any, Callable, Optional

from snowflake import connector, snowpark
from snowflake.ml._internal.utils import formatting


def _query_log(sql: str | None) -> str:
    """Returns the query string to log if valid. Returns empty otherwise."""
    return f" Query: {sql}" if sql else ""


def result_dimension_matcher(
    expected_rows: Optional[int], expected_cols: Optional[int], result: list[snowpark.Row], sql: str | None = None
) -> bool:
    """Check result dimensions of the collected result dataframe of a Snowflake SQL operation.

    If rows or cols are None, they are not checked.

    Args:
        expected_rows (int): Number of expected rows in the result.
        expected_cols (int): Number of expected columns in the result:
        result (List[snowpark.Row]): Collected dataframe containing the result status of a Snowflake SQL operation.
        sql (str): Query string to be included in the error messages. Will not affect the functionality otherwise.

    Returns:
        true if `expected_{rows|cols}` number of rows and columns found in `result`.

    Raises:
        DataError: In case the validation failed.
    """
    actual_rows = len(result)
    if expected_rows is not None and actual_rows != expected_rows:
        raise connector.DataError(
            formatting.unwrap(
                f"""Query Result did not match expected number of rows. Expected {expected_rows} rows, found:
                {actual_rows} rows. Result from operation was: {result}.{_query_log(sql)}"""
            )
        )

    if expected_cols is not None:
        if not result:
            raise connector.DataError(
                formatting.unwrap(
                    f"""Query Result is empty but a number of columns expected. Expected {expected_cols} columns.
                    Result from operation was: {result}.{_query_log(sql)}"""
                )
            )

        actual_cols = len(result[0])
        if actual_cols != expected_cols:
            raise connector.DataError(
                formatting.unwrap(
                    f"""Query Result did not match expected number of columns. Expected {expected_cols} columns,
                    found {actual_cols} columns. Result from operation was: {result}.{_query_log(sql)}"""
                )
            )
    return True


def column_name_matcher(
    expected_col_name: str, allow_empty: bool, result: list[snowpark.Row], sql: str | None = None
) -> bool:
    """Returns true if `expected_col_name` is found. Raise exception otherwise."""
    if not result:
        if allow_empty:
            return True
        raise connector.DataError(f"Query Result is empty.{_query_log(sql)}")
    if expected_col_name not in result[0]:
        raise connector.DataError(
            formatting.unwrap(
                f"""Query Result did not have expected column {expected_col_name}. Result from operation was:
                    {result}.{_query_log(sql)}"""
            )
        )
    return True


def cell_value_by_column_matcher(
    row_idx: int, expected_col_name: str, expected_value: Any, result: list[snowpark.Row], sql: str | None = None
) -> bool:
    """Returns true if `expected_col_name` is found in `result` and the value in row `row_idx` matches `value`. Raises a
    connector.DataError otherwise.

    Args:
        row_idx (int): index of the row to check the value
        expected_col_name (str): name of the column to check
        expected_value (Any): What to compare the value in the result against.
        result (list[snowpark.Row]): result to check. Typically the result of a .collect() call on a DataFrame.
        sql (Optional[str]): Sql string that generated this result. This is optional and for more detailed error
            messages only.

    Returns:
        True iff the given result matches the expectation given by the arguments.

    Raises:
        DataError: There is a mismatch between the given result and the stated expectation.
    """
    if len(result) <= row_idx or len(result[row_idx]) < 1:
        raise connector.DataError(
            formatting.unwrap(
                f"""Query Result did not have required number of rows x col [{row_idx}][{expected_col_name}]. Result
                from operation was: {result}.{_query_log(sql)}"""
            )
        )
    if expected_col_name not in result[row_idx]:
        raise connector.DataError(
            formatting.unwrap(
                f"""Query Result did not have the expected value column '{expected_col_name}' in row {row_idx}.
                Actual row looked like {result[row_idx]}.{_query_log(sql)}"""
            )
        )
    if result[row_idx][expected_col_name] != expected_value:
        raise connector.DataError(
            formatting.unwrap(
                f"""Query Result did not have the expected value '{expected_value}' at expected position
                [{row_idx}][{expected_col_name}]. Actual value at position [{row_idx}][{expected_col_name}] was
                '{result[row_idx][expected_col_name]}'.{_query_log(sql)}"""
            )
        )
    return True


_DEFAULT_MATCHERS: list[Callable[[list[snowpark.Row], Optional[str]], bool]] = [
    partial(result_dimension_matcher, 1, 1),
    partial(column_name_matcher, "status"),
]


class ResultValidator:
    """Convenience wrapper for validation SnowPark DataFrames that are returned as the result of session operations.

    Usage Example:
        result = (
            ResultValidator(
                session=self._session,
                query="UPDATE table SET NAME = 'name'",
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_partial_match(row_idx=0, col_idx=0, expected_value="number of rows updated=1")
            .validate()
        )
    """

    def __init__(self, result: list[snowpark.Row], query: str | None = None) -> None:
        self._result: list[snowpark.Row] = result
        self._query: str | None = query
        self._success_matchers: list[Callable[[list[snowpark.Row], Optional[str]], bool]] = []

    def has_dimensions(self, expected_rows: int | None = None, expected_cols: int | None = None) -> ResultValidator:
        """Validate that the result of the operation has the right shape of `expected_rows` rows and `expected_cols`
        columns.

        Args:
            expected_rows: Number of rows expected in the result.
            expected_cols: Number of columns expected in the result.

        Returns:
            ResultValidator object (self)
        """
        self._success_matchers.append(partial(result_dimension_matcher, expected_rows, expected_cols))
        return self

    def has_column(self, expected_col_name: str, allow_empty: bool = False) -> ResultValidator:
        """Validate that the a column with the name `expected_column_name` exists in the result.

        Args:
            expected_col_name: Name of the column that is expected to be present in the result (case sensitive).
            allow_empty: If the check will fail if the result is empty.

        Returns:
            ResultValidator object (self)
        """
        self._success_matchers.append(partial(column_name_matcher, expected_col_name, allow_empty))
        return self

    def has_named_value_match(self, row_idx: int, col_name: str, expected_value: Any) -> ResultValidator:
        """Validate that the column `col_name` in row `row_idx` of the results exists and matches `expected_value`.

        Args:
            row_idx: Row index of the cell that needs to match.
            col_name: Column name of the cell that needs to match.
            expected_value: Value that the cell needs to match. For strings it is treated as a substring match, all
                other types will expect an exact match.

        Returns:
            ResultValidator object (self)
        """
        self._success_matchers.append(partial(cell_value_by_column_matcher, row_idx, col_name, expected_value))
        return self

    def insertion_success(self, expected_num_rows: int) -> ResultValidator:
        """Validate that `expected_num_rows` have been inserted successfully.

        Args:
            expected_num_rows: Number of rows that are expected to be inserted successfully.

        Returns:
            ResultValidator object (self)
        """
        self._success_matchers.append(
            partial(cell_value_by_column_matcher, 0, "number of rows inserted", expected_num_rows)
        )
        return self

    def deletion_success(self, expected_num_rows: int) -> ResultValidator:
        """Validate that `expected_num_rows` have been deleted successfully.

        Args:
            expected_num_rows: Number of rows that are expected to be deleted successfully.

        Returns:
            ResultValidator object (self)
        """
        self._success_matchers.append(
            partial(cell_value_by_column_matcher, 0, "number of rows deleted", expected_num_rows)
        )
        return self

    def _get_result(self) -> list[snowpark.Row]:
        """Return the given result DataFrame."""
        return self._result

    def validate(self) -> list[snowpark.Row]:
        """Execute the query and validate the result.

        Returns:
            Query result.
        """
        result = self._get_result()
        for matcher in self._success_matchers:
            assert matcher(result, self._query)
        return result


class SqlResultValidator(ResultValidator):
    """Convenience wrapper for validation of SQL queries.

    Usage Example:
        result = (
            SqlResultValidator(
                session=self._session,
                query="UPDATE table SET NAME = 'name'",
                statement_params=statement_params,
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_partial_match(row_idx=0, col_idx=0, expected_value="number of rows updated=1")
            .validate()
        )
    """

    def __init__(
        self, session: snowpark.Session, query: str, statement_params: Optional[dict[str, Any]] = None
    ) -> None:
        self._session: snowpark.Session = session
        self._query: str = query
        self._success_matchers: list[Callable[[list[snowpark.Row], Optional[str]], bool]] = []
        self._statement_params: Optional[dict[str, Any]] = statement_params

    def _get_result(self) -> list[snowpark.Row]:
        """Collect the result of the given SQL query."""
        return self._session.sql(self._query).collect(statement_params=self._statement_params)
