from typing import Collection, Optional

import cloudpickle
import numpy as np
import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import DataFrame, functions as F

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
def covariance(*, df: DataFrame, columns: Optional[Collection[str]] = None, ddof: int = 1) -> pd.DataFrame:
    """Covariance matrix for the columns in a snowpark dataframe.
    NaNs and Nulls are not ignored, i.e. covariance on columns containing NaN or Null
    results in NaN covariance values.
    Returns a pandas dataframe containing the covariance matrix.

    The below steps explain how covariance matrix is computed in a distributed way:
    Let n = # of rows in the dataframe; ddof = delta degree of freedom; X, Y are 2 columns in the dataframe
    Covariance(X, Y) = term1 - term2 - term3 + term4 where
    term1 = dot(X/sqrt(n-ddof), Y/sqrt(n-ddof))
    term2 = sum(Y/n)*sum(X/(n-ddof))
    term3 = sum(X/n)*sum(Y/(n-ddof))
    term4 = (n/(n-ddof))*sum(X/n)*sum(Y/n)

    Note that the formula is entirely written using dot and sum operators on columns. Using first UDTF, we compute the
    dot and sum of columns for different shards in parallel. In the second UDTF, dot and sum is accumulated
    from all the shards. The final computation for covariance matrix is done on the client side
    as a post-processing step.

    Args:
        df (DataFrame): Snowpark Dataframe for which covariance matrix has to be computed.
        columns (Optional[Collection[str]]): List of column names for which the covariance matrix has to be computed.
            If None, covariance matrix is computed for all numeric columns in the snowpark dataframe.
        ddof (int): default 1. Delta degrees of freedom.
            The divisor used in calculations is N - ddof, where N represents the number of rows.

    Returns:
        Covariance matrix in pandas.DataFrame format.
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
        sharded_dot_and_sum_computer_udtf(F.col("ARR_COL"), F.lit(count), F.lit(ddof))
    ).with_column_renamed("RESULT", "RES")
    res_df = temp_df2.select(accumulator_udtf(F.col("RES")).over(partition_by="PART"), F.col("PART"))
    results = res_df.collect(statement_params=statement_params)

    # The below computation can be moved to a third udtf. But there is not much benefit in terms of client side
    # resource consumption as the below computation is very fast (< 1 sec for 1000 cols). Memory is in the same order
    # as the resultant covariance matrix.
    # Pushing this to a udtf requires creating a temp udtf which takes about 20 secs, so it doesn't make sense
    # to have this in a udtf.
    n_cols = len(columns)
    sum_by_count = np.zeros(n_cols)
    sum_by_countd = np.zeros(n_cols)
    term1 = np.zeros((n_cols, n_cols))
    # Get sum and dot prod from the results.
    for i in range(len(results)):
        x = results[i]
        if x[1] == "sum_by_count":
            sum_by_count = cloudpickle.loads(x[0])
        elif x[1] == "sum_by_countd":
            sum_by_countd = cloudpickle.loads(x[0])
        else:
            row = int(x[1].strip("row_"))
            term1[row, :] = cloudpickle.loads(x[0])
    if ddof == 0:
        sum_by_countd = sum_by_count

    term2 = np.matmul(sum_by_count[:, np.newaxis], sum_by_countd[:, np.newaxis].T)
    term3 = term2.T
    factor = count * 1.0 / (count - ddof)
    term4 = factor * np.matmul(sum_by_count[:, np.newaxis], sum_by_count[:, np.newaxis].T)

    covariance_matrix = term1 - term2 - term3 + term4
    return pd.DataFrame(covariance_matrix, columns=columns, index=columns)
