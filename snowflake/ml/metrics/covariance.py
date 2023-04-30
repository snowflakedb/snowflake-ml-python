#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import inspect
from typing import Collection, Optional

import cloudpickle
import numpy as np
import pandas as pd

from snowflake.ml.metrics import _utils
from snowflake.ml.utils import telemetry
from snowflake.snowpark import DataFrame

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(  # type: ignore[misc]
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

    statement_params = telemetry.get_function_usage_statement_params(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), None),
    )

    input_df, columns = _utils.validate_and_return_dataframe_and_columns(df=df, columns=columns)
    sharded_dot_and_sum_computer = _utils.register_sharded_dot_sum_computer(
        session=input_df._session, statement_params=statement_params
    )
    dot_and_sum_accumulator = _utils.register_accumulator_udtf(
        session=input_df._session, statement_params=statement_params
    )
    count = input_df.count(statement_params=statement_params)

    # TODO: Move the below to snowpark dataframe operations
    input_query = input_df.queries["queries"][-1]
    query = f"""
        with temp_table1 as
        (select array_construct(*) as col from ({input_query})),
        temp_table2 as
        (select result as res, part from temp_table1,
        table({sharded_dot_and_sum_computer}(temp_table1.col, '{str(count)}', '{str(ddof)}')))
        select result, temp_table2.part from temp_table2,
        table({dot_and_sum_accumulator}(temp_table2.res) over (partition by part))
    """
    results = input_df._session.sql(query).collect(statement_params=statement_params)

    # The below computation can be moved to a third udtf. But there is not much benefit in terms of client side
    # resource consumption as the below computation is very fast (< 1 sec for 1000 cols). Memory is in the same order
    # as the resultant covariance matrix.
    # Pushing this to a udtf requires creating a temp udtf which takes about 20 secs, so it doesn't make sense
    # to have this in a udtf.
    n_cols = len(columns)  # type: ignore
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
