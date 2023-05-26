#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import math
from typing import Collection, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import cloudpickle
import numpy as np

from snowflake import snowpark
from snowflake.snowpark import Session, functions as F, types as T

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


def register_accumulator_udtf(*, session: Session, statement_params: Dict[str, str]) -> str:
    """Registers accumulator UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session (Session): Snowpark session.
        statement_params (Dict[str, str]): Dictionary used for tagging queries for tracking purposes.

    Returns:
        Name of the UDTF.
    """

    class DotAndSumAccumulator:
        """This class is registered as a UDTF. It accumulates all the rows passed to the UDTF."""

        def __init__(self) -> None:
            self._accumulated_row = None

        def process(self, input_row: bytes) -> None:
            """Accumulates rows.

            Args:
                input_row (bytes): numpy array serialized using cloudpickle.
            """
            row = cloudpickle.loads(input_row)
            if self._accumulated_row is None:
                self._accumulated_row = row
            else:
                self._accumulated_row = self._accumulated_row + row

        def end_partition(self) -> Iterable[Tuple[bytes]]:
            yield (cloudpickle.dumps(self._accumulated_row),)

    dot_and_sum_accumulator = "DotAndSumAccumulator_{}".format(str(uuid4()).replace("-", "_").upper())
    session.udtf.register(
        DotAndSumAccumulator,
        output_schema=T.StructType(
            [
                T.StructField("result", T.BinaryType()),
            ]
        ),
        input_types=[T.BinaryType()],
        packages=["numpy", "cloudpickle"],
        name=dot_and_sum_accumulator,
        is_permanent=False,
        replace=True,
        statement_params=statement_params,
    )
    return dot_and_sum_accumulator


def register_sharded_dot_sum_computer(*, session: Session, statement_params: Dict[str, str]) -> str:
    """Registers dot and sum computation UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session (Session): Snowpark session.
        statement_params (Dict[str, str]): Dictionary used for tagging queries for tracking purposes.

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

        def process(self, input_row: List[float], count: str, ddof: str) -> None:
            """Computes sum and dot product.

            Args:
                input_row (List[float]): List of floats.
                count (str): Number of rows in the table.
                ddof (str): delta degree of freedom
            """
            # 1. initialization of variables
            if not self._variables_initialized:
                self._n_cols = len(input_row)
                self._count = int(count)
                self._ddof = int(ddof)
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

    sharded_dot_and_sum_computer = "ShardedDotAndSumComputer_{}".format(str(uuid4()).replace("-", "_").upper())
    session.udtf.register(
        ShardedDotAndSumComputer,
        output_schema=T.StructType(
            [
                T.StructField("result", T.BinaryType()),
                T.StructField("part", T.StringType()),
            ]
        ),
        input_types=[T.ArrayType(), T.StringType(), T.StringType()],
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
        df (snowpark.DataFrame): Input snowpark dataframe.
        columns (Optional[Collection[str]]): Columns that need to be validated.

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


def weighted_sum(
    *,
    df: snowpark.DataFrame,
    sample_score_column: snowpark.Column,
    sample_weight_column: Optional[snowpark.Column] = None,
    normalize: bool = False,
    statement_params: Dict[str, str],
) -> float:
    """Weighted sum of the sample score column.

    Args:
        df: Input dataframe.
        sample_score_column: Sample score column.
        sample_weight_column: Sample weight column.
        normalize: If ``False``, return the weighted sum.
            Otherwise, return the fraction of weighted sum.
        statement_params: Dictionary used for tagging queries for tracking purposes.

    Returns:
        If ``normalize == True``, return the fraction of weighted sum (float),
        else returns the weighted sum (int).
    """
    if normalize:
        if sample_weight_column is not None:
            res = F.sum(sample_score_column * sample_weight_column) / F.sum(  # type: ignore[arg-type, operator]
                sample_weight_column  # type: ignore[arg-type]
            )
        else:
            res = F.avg(sample_score_column)  # type: ignore[arg-type]
    elif sample_weight_column is not None:
        res = F.sum(sample_score_column * sample_weight_column)  # type: ignore[arg-type, operator]
    else:
        res = F.sum(sample_score_column)  # type: ignore[arg-type]

    return float(df.select(res).collect(statement_params=statement_params)[0][0])
