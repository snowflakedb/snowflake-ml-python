from typing import Collection, Optional

import cloudpickle
import numpy as np
import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
def correlation(*, df: snowpark.DataFrame, columns: Optional[Collection[str]] = None) -> pd.DataFrame:
    """Pearson correlation matrix for the columns in a snowpark dataframe.
    NaNs and Nulls are not ignored, i.e. correlation on columns containing NaN or Null
    results in NaN correlation values.
    Returns a pandas dataframe containing the correlation matrix.

    The below steps explain how correlation matrix is computed in a distributed way:
    Let n = # of rows in the dataframe; sqrt_n = sqrt(n); X, Y are 2 columns in the dataframe
    Correlation(X, Y) = numerator/denominator where
    numerator = dot(X/sqrt_n, Y/sqrt_n) - sum(X/n)*sum(X/n)
    denominator = std_dev(X)*std_dev(Y)
    std_dev(X) = sqrt(dot(X/sqrt_n, X/sqrt_n) - sum(X/n)*sum(X/n))

    Note that the formula is entirely written using dot and sum operators on columns. Using first UDTF, we compute the
    dot and sum of columns for different shards in parallel. In the second UDTF, dot and sum is accumulated
    from all the shards. The final computation for numerator, denominator and division is done on the client side
    as a post-processing step.

    Args:
        df (snowpark.DataFrame): Snowpark Dataframe for which correlation matrix has to be computed.
        columns (Optional[Collection[str]]): List of column names for which the correlation matrix has to be computed.
            If None, correlation matrix is computed for all numeric columns in the snowpark dataframe.

    Returns:
        Correlation matrix in pandas.DataFrame format.
    """
    assert df._session is not None
    session = df._session
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    input_df, columns = metrics_utils.validate_and_return_dataframe_and_columns(df=df, columns=columns)
    count = input_df.count(statement_params=statement_params)

    # Register UDTFs.
    sharded_dot_and_sum_computer = metrics_utils.register_sharded_dot_sum_computer(
        session=session, statement_params=statement_params
    )
    sharded_dot_and_sum_computer_udtf = F.table_function(sharded_dot_and_sum_computer)
    accumulator = metrics_utils.register_accumulator_udtf(session=session, statement_params=statement_params)
    accumulator_udtf = F.table_function(accumulator)

    # Compute the confusion matrix.
    temp_df1 = input_df.select(F.array_construct(*input_df.columns).alias("ARR_COL"))
    temp_df2 = temp_df1.select(
        sharded_dot_and_sum_computer_udtf(F.col("ARR_COL"), F.lit(count), F.lit(0))
    ).with_column_renamed("RESULT", "RES")
    res_df = temp_df2.select(accumulator_udtf(F.col("RES")).over(partition_by="PART"), F.col("PART"))
    results = res_df.collect(statement_params=statement_params)

    # The below computation can be moved to a third udtf. But there is not much benefit in terms of client side
    # resource consumption as the below computation is very fast (< 1 sec for 1000 cols). Memory is in the same order
    # as the resultant correlation matrix.
    # Pushing this to a udtf requires creating a temp udtf which takes about 20 secs, so it doesn't make sense
    # to have this in a udtf.
    n_cols = len(columns)
    sum_arr = np.zeros(n_cols)
    squared_sum_arr = np.zeros(n_cols)
    dot_prod = np.zeros((n_cols, n_cols))
    # Get sum, dot_prod and squared sum array from the results.
    for i in range(len(results)):
        x = results[i]
        if x[1] == "sum_by_count":
            sum_arr = cloudpickle.loads(x[0])
        else:
            row = int(x[1].strip("row_"))
            dot_prod[row, :] = cloudpickle.loads(x[0])
            squared_sum_arr[row] = dot_prod[row, row]

    # sum(X/n)*sum(Y/n) is computed for all combinations of X,Y (columns in the dataframe)
    exey_arr = np.einsum("t,m->tm", sum_arr, sum_arr, optimize="optimal")
    numerator_matrix = dot_prod - exey_arr

    # standard deviation for all columns in the dataframe
    stddev_arr = np.sqrt(squared_sum_arr - np.einsum("i, i -> i", sum_arr, sum_arr, optimize="optimal"))
    # std_dev(X)*std_dev(Y) is computed for all combinations of X,Y (columns in the dataframe)
    denominator_matrix = np.einsum("t,m->tm", stddev_arr, stddev_arr, optimize="optimal")
    corr_res = numerator_matrix / denominator_matrix
    correlation_matrix = pd.DataFrame(corr_res, columns=columns, index=columns)
    return correlation_matrix
