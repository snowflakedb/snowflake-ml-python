from typing import List, Optional, Union

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def accuracy_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    normalize: bool = True,
    sample_weight_col_name: Optional[str] = None,
) -> float:
    """
    Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in the y true columns.

    Args:
        df: Input dataframe.
        y_true_col_names: Column name(s) representing actual values.
        y_pred_col_names: Column name(s) representing predicted values.
        normalize: If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
        sample_weight_col_name: Column name representing sample weights.

    Returns:
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

    if isinstance(y_true_col_names, str) or (len(y_true_col_names) == 1):
        score_column = F.iff(df[y_true_col_names] == df[y_pred_col_names], 1, 0)  # type: ignore[arg-type]
    # multilabel
    else:
        expr = " and ".join([f"({y_true_col_names[i]} = {y_pred_col_names[i]})" for i in range(len(y_true_col_names))])
        score_column = F.iff(expr, 1, 0)  # type: ignore[arg-type]
    return metrics_utils.weighted_sum(
        df=df,
        sample_score_column=score_column,
        sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
        normalize=normalize,
        statement_params=telemetry.get_statement_params(_PROJECT, _SUBPROJECT),
    )
