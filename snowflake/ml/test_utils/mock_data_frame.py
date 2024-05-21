from __future__ import annotations  # for return self methods

from typing import Any, Dict, List, Optional, Type, Union

from snowflake import snowpark
from snowflake.ml._internal.utils import formatting
from snowflake.ml._internal.utils.string_matcher import StringMatcherIgnoreWhitespace
from snowflake.ml.test_utils import mock_snowml_base
from snowflake.snowpark import Column, DataFrame
from snowflake.snowpark._internal import type_utils


class MockAsyncJob:
    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self, *args: Any, **kwargs: Any) -> Any:
        return self._result


class MockDataFrame(mock_snowml_base.MockSnowMLBase):
    """Mock version of snowflake.snowpark.DataFrame.

    This class has very limited functionality and currently only implements the collect() method. It is most useful in
    combination with MockSession to return results from SQL queries.

    Example:
        MockDataFrame([Row(status='Database TEST successfully created.')]).collect()

        will return: [Row(status='Database TEST successfully created.')]
    """

    def __init__(
        self,
        collect_result: Union[list[snowpark.Row], MockAsyncJob] | None = None,
        count_result: int | None = None,
        collect_statement_params: dict[str, str] | None = None,
        count_statement_params: dict[str, str] | None = None,
        check_call_sequence_completion: bool = True,
        columns: list[str] | None = None,
        collect_block: Optional[bool] = None,
    ) -> None:
        """Initializes MockDataFrame.

        Args:
            collect_result: Expected result for a `collect()` call to the dataframe. Typically a list of Row() objects.
            count_result: Expected result of a `count()` call to the dataframe. Typically and integer number.
            collect_statement_params: It will check if the same `statement_params` specified during collect() call.
            count_statement_params: It will check if the same `statement_params` specified during collect() call.
            check_call_sequence_completion: If True (default), we will check the completion of all expected operation
                calls in the finalize() call and when leaving a `with MockDataFrame()` context.
            columns: List of columns available in the mock dataframe. This value is exposed as a property.
            collect_block: It will check if the same `block` parameter is specified during collect() call.
        """
        super().__init__(
            check_call_sequence_completion,
        )
        self.columns = columns
        # We cannot specify both collect_result and count_result for the same DataFrame. No shorter version of this
        # expression worked.
        assert not (collect_result is not None and count_result is not None)
        # If collect_statement_params given, collect_result must be present.
        assert not collect_statement_params or collect_result

        # We need to differentiate between None and empty list. Empty list is a valid response to collect call.
        if collect_result is not None:
            self.add_collect_result(collect_result, collect_statement_params, collect_block)

        # Similarly we need to differentiate between None and 0. Dataframe with result count 0 is a valid scenario.
        if count_result is not None:
            self.add_count_result(count_result, count_statement_params)

    @property  # type: ignore[misc]
    def __class__(self) -> Type[DataFrame]:  # type: ignore[override]
        return DataFrame

    def __repr__(self) -> str:
        result = "MockDataFrame"
        indent = 8
        # return result
        for i in range(len(self._call_sequence)):
            prefix = "\n" + " " * indent
            prefix += "------> " if i == self._call_sequence_index else "        "
            op = self._call_sequence[i]
            if isinstance(op.result, MockDataFrame):
                result += f"{prefix}.{op.operation}(args={op.args}, kwargs={op.kwargs})"
            else:
                result += formatting.unwrap(
                    f"""{prefix}.{op.operation}(args={op.args}, kwargs={op.kwargs}) ->
                            {op.result.__class__.__name__}: {op.result}"""
                )
        return result

    def add_collect_result(
        self,
        result: Union[List[snowpark.Row], MockAsyncJob],
        statement_params: Optional[Dict[str, Any]] = None,
        block: Optional[bool] = None,
    ) -> mock_snowml_base.MockSnowMLBase:
        """Convenience helper to set the expected result of a `collect()` operation."""
        check_statement_params = False
        kwargs: Optional[Dict[str, Any]] = None
        if statement_params:
            kwargs = {}
            kwargs["statement_params"] = statement_params
            check_statement_params = True
        if block is not None:
            kwargs = kwargs or {}
            kwargs["block"] = block

        return self.add_operation(
            operation="collect", kwargs=kwargs, result=result, check_statement_params=check_statement_params
        )

    def add_count_result(
        self,
        result: int | None,
        statement_params: dict[str, str] | None = None,
    ) -> mock_snowml_base.MockSnowMLBase:
        """Convenience helper to set the expected result of a `count()` operation."""
        check_statement_params = False
        kwargs = None
        if statement_params:
            kwargs = {}
            kwargs["statement_params"] = statement_params
            check_statement_params = True
        return self.add_operation(
            operation="count", kwargs=kwargs, result=result, check_statement_params=check_statement_params
        )

    def add_mock_filter(self, expr: str, result: MockDataFrame | None = None) -> MockDataFrame:
        return self.add_operation(
            operation="filter", args=(), kwargs={"expr": StringMatcherIgnoreWhitespace(expr)}, result=result
        )

    def add_mock_drop(self, *cols: type_utils.ColumnOrName, result: MockDataFrame | None = None) -> MockDataFrame:
        return self.add_operation(operation="drop", args=cols, kwargs={}, result=result)

    def add_mock_sort(
        self,
        *cols: type_utils.ColumnOrName,
        ascending: Optional[Union[bool, int, List[Union[bool, int]]]] = None,
        result: MockDataFrame | None = None,
    ) -> MockDataFrame:
        if ascending:
            kwargs = {"ascending": ascending}
        else:
            kwargs = {}
        return self.add_operation(operation="sort", args=cols, kwargs=kwargs, result=result)

    def add_mock_with_columns(
        self,
        col_names: List[str],
        values: List[Column],
        result: MockDataFrame | None = None,
    ) -> MockDataFrame:
        return self.add_operation(
            operation="with_columns", args=(), kwargs={"col_names": col_names, "values": values}, result=result
        )

    def collect(self, *args: Any, **kwargs: Any) -> Any:
        """Collect a dataframe. Corresponds to DataFrame.collect."""
        mdfo = self._check_operation("collect", args, kwargs)
        return mdfo.result

    def collect_nowait(self, *args: Any, **kwargs: Any) -> Any:
        """Collect a dataframe. Corresponds to DataFrame.collect_nowait."""
        return self.collect(*args, **kwargs)

    def filter(self, *args: Any, **kwargs: Any) -> Any:
        """Filter a dataframe. Corresponds to DataFrame.filter.

        We currently cannot check the arguments to filter. Argument checking for the call is disabled.

        # noqa: DAR101
        # noqa: DAR201
        """
        mdfo = self._check_operation("filter", args=args, kwargs=kwargs, check_args=False, check_kwargs=False)
        return mdfo.result

    def drop(self, *args: Any, **kwargs: Any) -> Any:
        mdfo = self._check_operation("drop", args=args, kwargs=kwargs)
        return mdfo.result

    def sort(self, *args: Any, **kwargs: Any) -> Any:
        mdfo = self._check_operation("sort", args=args, kwargs=kwargs)
        return mdfo.result

    def with_columns(self, *args: Any, **kwargs: Any) -> Any:
        mdfo = self._check_operation("with_columns", args=args, kwargs=kwargs, check_args=False, check_kwargs=False)
        return mdfo.result

    def count(self, *args: Any, **kwargs: Any) -> Any:
        """Count the number of rows in the dataframe. Corresponds to DataFrame.count."""
        # Result should be int if block=True and AsyncJob if block=False.
        mdfo = self._check_operation("count", args, kwargs)
        return mdfo.result

    def select(self, *args: Any, **kwargs: Any) -> Any:
        """Select columns from the dataframe. Corresponds to DataFrame.select."""
        # Result should be int if block=True and AsyncJob if block=False.
        mdfo = self._check_operation("select", args, kwargs)
        return mdfo.result
