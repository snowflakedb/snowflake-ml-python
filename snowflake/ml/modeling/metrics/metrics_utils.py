import math
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import cloudpickle
import numpy as np

import snowflake.snowpark._internal.utils as snowpark_utils
from snowflake import snowpark
from snowflake.snowpark import Session, functions as F, types as T

LABEL = "LABEL"
INDEX = "INDEX"


def register_accumulator_udtf(*, session: Session, statement_params: Dict[str, Any]) -> str:
    """Registers accumulator UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session: Snowpark session.
        statement_params: Dictionary used for tagging queries for tracking purposes.

    Returns:
        Name of the UDTF.
    """

    class Accumulator:
        """This class is registered as a UDTF. It accumulates all the rows passed to the UDTF."""

        def __init__(self) -> None:
            self._accumulated_row = None

        def process(self, input_row: bytes) -> None:
            """Accumulates rows.

            Args:
                input_row: numpy array serialized using cloudpickle.
            """
            row = cloudpickle.loads(input_row)
            if self._accumulated_row is None:
                self._accumulated_row = row
            else:
                self._accumulated_row = self._accumulated_row + row

        def end_partition(self) -> Iterable[Tuple[bytes]]:
            yield (cloudpickle.dumps(self._accumulated_row),)

    accumulator = "Accumulator_{}".format(str(uuid4()).replace("-", "_").upper())
    session.udtf.register(
        Accumulator,
        output_schema=T.StructType(
            [
                T.StructField("result", T.BinaryType()),
            ]
        ),
        input_types=[T.BinaryType()],
        packages=["numpy", "cloudpickle"],
        name=accumulator,
        is_permanent=False,
        replace=True,
        statement_params=statement_params,
    )
    return accumulator


def register_sharded_dot_sum_computer(*, session: Session, statement_params: Dict[str, Any]) -> str:
    """Registers dot and sum computation UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session: Snowpark session.
        statement_params: Dictionary used for tagging queries for tracking purposes.

    Returns:
        Name of the UDTF.
    """

    class ShardedDotAndSumComputer:
        """This class is registered as a UDTF and computes the sum and dot product
        of columns for each partition of rows. The computations across all the partitions happens
        in parallel using the nodes in the warehouse. In order to avoid keeping the entire partition
        in memory, we batch the rows (size is 1000) and maintain a running sum and dot prod in self._sum_by_count,
        self._sum_by_countd and self._dot_prod respectively. We return these at the end of the partition.
        """

        def __init__(self) -> None:
            self._variables_initialized = False
            # 2d array containing pairwise dot product of all columns - sum(col1*col2/num_rows).
            self._dot_prod = np.zeros((1, 1))
            # number of rows
            self._count = 0
            # delta degree of freedom
            self._ddof = 0
            # Setting the batch size to 1000 based on experimentation. Can be fine tuned later.
            self._batch_size = 1000
            # 2d array containing a batch of input rows. A batch contains self._batch_size rows.
            self._batched_rows = np.zeros((self._batch_size, 1))
            # 1d array of length = # of cols. Contains sum(col/count) for each column.
            self._sum_by_count = np.zeros(1)
            # 1d array of length = # of cols. Contains sum(col/(count-ddof)) for each column.
            self._sum_by_countd = np.zeros(1)
            # Number of columns in the dataset.
            self._n_cols = -1
            # Running count of number of rows added to self._batched_rows.
            self._cur_count = 0
            # Square root of count - ddof
            self._sqrt_count_d = -1.0

        def process(self, input_row: List[float], count: int, ddof: int) -> None:
            """Computes sum and dot product.

            Args:
                input_row: List of floats.
                count: Number of rows in the table.
                ddof: delta degree of freedom
            """
            # 1. initialization of variables
            if not self._variables_initialized:
                self._n_cols = len(input_row)
                self._count = count
                self._ddof = ddof
                self._sqrt_count_d = math.sqrt(self._count - self._ddof)
                self._sum_by_count = np.zeros(self._n_cols)
                self._sum_by_countd = np.zeros(self._n_cols)
                self._batched_rows = np.zeros((self._batch_size, self._n_cols))
                self._dot_prod = np.zeros((self._n_cols, self._n_cols))
            self._variables_initialized = True

            self._batched_rows[self._cur_count, :] = input_row
            self._cur_count += 1

            # 2. Compute incremental sum and dot_prod for the batch.
            if self._cur_count >= self._batch_size:
                self.accumulate_batch_sum_and_dot_prod()
                self._cur_count = 0

        def end_partition(self) -> Iterable[Tuple[bytes, str]]:
            # 3. Compute sum and dot_prod for the remaining rows in the batch.
            if self._cur_count > 0:
                self.accumulate_batch_sum_and_dot_prod()
            for i in range(self._n_cols):
                yield (cloudpickle.dumps(self._dot_prod[i, :]), "row_" + str(i))
            yield (cloudpickle.dumps(self._sum_by_count), "sum_by_count")
            if self._ddof != 0:
                yield (cloudpickle.dumps(self._sum_by_countd), "sum_by_countd")

        def accumulate_batch_sum_and_dot_prod(self) -> None:
            rows_by_sqrt_countd = self._batched_rows / self._sqrt_count_d
            self._dot_prod += np.einsum(
                "nt, nm -> tm",
                rows_by_sqrt_countd[0 : self._cur_count, :],
                rows_by_sqrt_countd[0 : self._cur_count, :],
                optimize="optimal",
            )
            rows_by_count = self._batched_rows / self._count
            self._sum_by_count += np.sum(rows_by_count[0 : self._cur_count, :], axis=0)
            if self._ddof != 0:
                rows_by_count_d = self._batched_rows / (self._count - self._ddof)
                self._sum_by_countd += np.sum(rows_by_count_d[0 : self._cur_count, :], axis=0)

    sharded_dot_and_sum_computer = snowpark_utils.random_name_for_temp_object(
        snowpark_utils.TempObjectType.TABLE_FUNCTION
    )
    # TODO (SNOW-897239): make this an anonymous temp UDTF for it to work in a SPROC
    session.udtf.register(
        ShardedDotAndSumComputer,
        output_schema=T.StructType(
            [
                T.StructField("result", T.BinaryType()),
                T.StructField("part", T.StringType()),
            ]
        ),
        input_types=[T.ArrayType(), T.IntegerType(), T.IntegerType()],
        packages=["numpy", "cloudpickle"],
        name=sharded_dot_and_sum_computer,
        is_permanent=False,
        replace=True,
        statement_params=statement_params,
    )
    return sharded_dot_and_sum_computer


def validate_and_return_dataframe_and_columns(
    *, df: snowpark.DataFrame, columns: Optional[Collection[str]] = None
) -> Tuple[snowpark.DataFrame, Collection[str]]:
    """Validates that the columns are all numeric and returns a dataframe with those columns.

    Args:
        df: Input snowpark dataframe.
        columns: Columns that need to be validated.

    Returns:
        Tuple with snowpark dataframe and list of columns.

    Raises:
        ValueError: If non-numeric columns are provided in the input arg, columns.
    """
    input_df = df
    if columns is None:
        columns = [c.name for c in input_df.schema.fields if issubclass(type(c.datatype), T._NumericType)]
        input_df = input_df.select(columns)
    else:
        input_df = input_df.select(columns)
        for c in input_df.schema.fields:
            if not issubclass(type(c.datatype), T._NumericType):
                msg = "Column: {} is not a numeric column"
                raise ValueError(msg.format(c.name))
    return (input_df, columns)


def check_label_columns(
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
) -> None:
    """Check y true and y pred columns.

    Args:
        y_true_col_names: Column name(s) representing actual values.
        y_pred_col_names: Column name(s) representing predicted values.

    Raises:
        TypeError: `y_true_col_names` and `y_pred_col_names` are of different types.
        ValueError: Multilabel `y_true_col_names` and `y_pred_col_names` are of different lengths.
    """
    if type(y_true_col_names) != type(y_pred_col_names):
        raise TypeError(
            "Label columns should be of the same type."
            f"Got y_true_col_names={type(y_true_col_names)} vs y_pred_col_names={type(y_pred_col_names)}."
        )
    if isinstance(y_true_col_names, list) and (len(y_true_col_names) != len(y_pred_col_names)):
        raise ValueError(
            "Length of multilabel label columns should be of the same between y_true_col_names and y_pred_col_names."
            f"Got y_true_col_names={y_true_col_names} (length: {len(y_true_col_names)}) vs "
            f"y_pred_col_names={y_pred_col_names} (length: {len(y_pred_col_names)})."
        )


def flatten_cols(cols: List[Optional[Union[str, List[str]]]]) -> List[str]:
    res = []
    for col in cols:
        if isinstance(col, str):
            res.append(col)
        if isinstance(col, list):
            res.extend(col)
    return res


def unique_labels(
    *,
    df: snowpark.DataFrame,
    columns: List[snowpark.Column],
) -> snowpark.DataFrame:
    """Extract indexed ordered unique labels as a dataframe.

    Args:
        df: Input dataframe.
        columns: Columns to extract labels from.

    Returns:
        Dataframe with ordered unique labels and indices.
        Columns: [LABEL, INDEX].
    """
    union_df = None
    for col in columns:
        temp_df = df.select(col.alias(LABEL))
        if union_df:
            # uniqueness guaranteed through `DataFrame.union`
            union_df = union_df.union(temp_df)
        else:
            union_df = temp_df

    # append an index column dense ranking labels
    assert union_df is not None
    res: snowpark.DataFrame = union_df.with_column(INDEX, F.dense_rank().over(snowpark.Window.order_by(LABEL)) - 1)
    return res
