import collections
import json
from typing import Optional

import pandas as pd
from pandas import arrays as pandas_arrays
from pandas.core.arrays import sparse as pandas_sparse

from snowflake.snowpark import DataFrame


def _pandas_to_sparse_pandas(pandas_df: pd.DataFrame, sparse_cols: list[str]) -> Optional[pd.DataFrame]:
    """Convert the pandas df into pandas df with multiple SparseArray columns."""
    num_rows = pandas_df.shape[0]
    if num_rows == 0:
        # Okay since pd.contact() ignores None.
        return None

    for col_name in sparse_cols:
        array_length, dtype = None, None
        indices, vals = collections.defaultdict(list), collections.defaultdict(list)
        for row_i, v in enumerate(pandas_df[col_name]):
            value_map = json.loads(v) if v else None
            if not value_map:
                continue

            array_length = array_length or value_map["array_length"]
            if array_length != value_map.pop("array_length"):
                raise ValueError("array_length mismatch")

            for key, value in value_map.items():
                dtype = dtype or type(value)
                if dtype != type(value):
                    raise ValueError("data type mismatch")

                key = int(key)
                if key >= array_length:
                    raise ValueError("index greater than array_length")

                indices[key].append(row_i)
                vals[key].append(value)

        for col_i in range(array_length or 0):
            # for each index create a SparseArray column. For now use sequential number as suffix for column name
            col_name_index = col_name + "_" + str(col_i)
            pandas_df[col_name_index] = pandas_arrays.SparseArray(
                vals[col_i], pandas_sparse.make_sparse_index(num_rows, indices[col_i], "integer"), dtype=dtype
            )

        del pandas_df[col_name]

    return pandas_df


def to_pandas_with_sparse(df: DataFrame, sparse_cols: list[str]) -> pd.DataFrame:
    """Load a Snowpark df with sparse columns represented in JSON strings into pandas df with multiple SparseArray
        columns.

       For example, for below input:
       ----------------------------------------------
       |'COL1'|'COL2'                               |
       ----------------------------------------------
       |'a'   |'{"array_length": 4, "1": 1}'        |
       |'b'   |'{"array_length": 4, "0": 3}'        |
       |'c'   |'{"array_length": 4}'                |
       |'d'   |'{"array_length": 4, "1": 2, "2": 1}'|
       ----------------------------------------------
       The call to `to_pandas_with_sparse(df, ['COL2'])` will return a pandas df:
       --------------------------------------------
       |'COL1'|'COL2_0'|'COL2_1'|'COL2_2'|'COL2_3'|
       --------------------------------------------
       |'a'   | 0      | 1      | 0      | 0      |
       |'b'   | 3      | 0      | 0      | 0      |
       |'c'   | 0      | 0      | 0      | 0      |
       |'d'   | 0      | 2      | 1      | 0      |
       --------------------------------------------
       The dtype of 'COL2_0', 'COL2_1', 'COL2_2', 'COL2_3' columns is 'Sparse[int64, 0]'.

    Args:
        df: A Snowpark data frame contains column(s) of sparse data represented as JSON strings. An example of such
            JSON strings: '{"3": 1, "4": 2, "array_length": 5}'. This can come from transformation results of
            `OneHotEncoder`.
        sparse_cols: names of sparse data columns.

    Returns:
        A pandas dataframe with each of the sparse data column from input expanded to multiple SparseArray columns.

    """
    pandas_dfs = [_pandas_to_sparse_pandas(pandas_df_batch, sparse_cols) for pandas_df_batch in df.to_pandas_batches()]
    return pd.concat(pandas_dfs)
