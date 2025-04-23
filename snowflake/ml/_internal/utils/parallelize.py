import math
from contextlib import contextmanager
from timeit import default_timer
from typing import Any, Callable, Generator, Iterable, Optional

import snowflake.snowpark.functions as F
from snowflake import snowpark


@contextmanager
def timer() -> Generator[Callable[[], float], None, None]:
    start: float = default_timer()

    def elapser() -> float:
        return default_timer() - start

    yield lambda: elapser()


def _flatten(L: Iterable[list[Any]]) -> list[Any]:
    return [val for sublist in L for val in sublist]


def map_dataframe_by_column(
    df: snowpark.DataFrame,
    cols: list[str],
    map_func: Callable[[snowpark.DataFrame, list[str]], snowpark.DataFrame],
    partition_size: int,
    statement_params: Optional[dict[str, Any]] = None,
) -> list[list[Any]]:
    """Applies the `map_func` to the input DataFrame by parallelizing it over subsets of the column.

    Because the return results are materialized as Python lists *in memory*, this method should
    not be used on operations that are expected to return many rows.

    The `map_func` must satisfy the property that for an input `df` with columns C, then for any
    partition [*c1, *c2, ..., *cp] on C, then `map_func(df,C) == [*map_func(df,c1), ..., *map_func(df,cp)]`,
    where * is the list unpacking operator. This means that `map_func(df, col_subset)` should
    return r rows and c*|col_subset| columns, for constants r and c.

    Args:
        df: Input dataset to operate on.
        cols: List of column names to compute on. Must index into `df`.
        map_func: The map function applied on each partition of the DataFrame.
        partition_size: The number of columns to include in each partition. Must be a positive integer.
        statement_params: Statement parameters for query telemetry.

    Returns:
        A Python list representation of the output of the query.

    Raises:
        Exception: If the pre-conditions above are not met.
    """
    partition_id_col = "_PARTITION_ID"
    n_output_cols = 0
    unioned_df: Optional[snowpark.DataFrame] = None
    last_partition_df = None

    if partition_size < 1:
        raise Exception(f"Partition size must be a positive integer, but got {partition_size}.")

    try:
        n_partitions = math.ceil(len(df[cols].columns) / partition_size)
    except Exception:
        raise Exception(f"Provided column names {cols} does not index into the dataset.")

    # This should never happen
    if n_partitions == 0:
        return [[]]

    # Create one DataFrame for the first n-1 partitions, and one for the last partition.
    for partition_id in range(n_partitions):
        cols_subset = cols[(partition_id * partition_size) : ((partition_id + 1) * partition_size)]
        mapped_df = map_func(df, cols_subset)
        if partition_id == 0:
            n_output_cols = len(mapped_df.columns)

        if partition_id == n_partitions - 1:
            last_partition_df = mapped_df
        else:
            if n_output_cols != len(mapped_df.columns):
                raise Exception("All partitions must contain the same number of columns.")
            mapped_df = mapped_df.with_column(partition_id_col, F.lit(partition_id))
            unioned_df = mapped_df if unioned_df is None else unioned_df.union_all(mapped_df)

    # Store results in a list of size |n_partitions| x |n_rows| x |n_output_cols|
    all_results: list[list[list[Any]]] = [[] for _ in range(n_partitions - 1)]

    # Collect the results of the first n-1 partitions, removing the partition_id column
    unioned_result = unioned_df.collect(statement_params=statement_params) if unioned_df is not None else []
    for row in unioned_result:
        row_dict = row.as_dict()
        partition_id = row_dict.pop(partition_id_col, None)
        if partition_id is None:
            raise Exception(f"Found unknown partition id {partition_id}.")
        all_results[partition_id].append(list(row_dict.values()))

    # Collect the results of the last partition
    last_partition_result = (
        [[]]
        if last_partition_df is None
        else [list(row) for row in last_partition_df.collect(statement_params=statement_params)]
    )
    all_results.append(last_partition_result)

    row_counts = {len(res) for res in all_results}
    if len(row_counts) > 1:
        raise Exception(
            f"All partitions must return the same number of rows, but found multiple row counts: {row_counts}."
        )

    return [_flatten(row) for row in list(zip(*all_results))]
