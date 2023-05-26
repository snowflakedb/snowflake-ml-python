from typing import Optional

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.metrics import _utils
from snowflake.snowpark import functions as F

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def accuracy_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_name: str,
    y_pred_col_name: str,
    normalize: bool = True,
    sample_weight_col_name: Optional[str] = None,
) -> float:
    """
    Accuracy classification score.

    Note: Currently multilabel classification is not supported.

    Args:
        df: Input dataframe.
        y_true_col_name: Column name representing actual values.
        y_pred_col_name: Column name representing predicted values.
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
    # TODO: Support multilabel classification.
    score_column = F.iff(df[y_true_col_name] == df[y_pred_col_name], 1, 0)  # type: ignore[arg-type]
    return _utils.weighted_sum(
        df=df,
        sample_score_column=score_column,
        sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
        normalize=normalize,
        statement_params=telemetry.get_statement_params(_PROJECT, _SUBPROJECT),
    )
