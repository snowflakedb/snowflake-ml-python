#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import inspect

from snowflake.ml._internal import telemetry
from snowflake.snowpark import DataFrame, functions as F

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
def r2_score(*, df: DataFrame, y_true_col_name: str, y_pred_col_name: str) -> float:
    """:math:`R^2` (coefficient of determination) regression score function.
    Returns R squared metric on 2 columns in the dataframe.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    TODO(pdorairaj): Implement other params from sklearn - sample_weight, multi_output, force_finite.

    Args:
        df (DataFrame): Input dataframe.
        y_true_col_name (str): Column name representing actual values.
        y_pred_col_name (str): Column name representing predicted values.

    Returns:
        R squared metric.
    """

    df_avg = df.select(F.avg(y_true_col_name).as_("avg_y_true"))  # type: ignore[arg-type]
    df_r_square = df.join(df_avg).select(
        F.lit(1)  # type: ignore[arg-type]
        - F.sum((df[y_true_col_name] - df[y_pred_col_name]) ** 2)  # type: ignore[operator]
        / F.sum((df[y_true_col_name] - df_avg["avg_y_true"]) ** 2)  # type: ignore[operator]
    )

    statement_params = telemetry.get_function_usage_statement_params(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), None),
    )
    return float(df_r_square.collect(statement_params=statement_params)[0][0])
