from typing import List, Optional, Union

import numpy.typing as npt

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling import metrics

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def precision_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.ArrayLike]:
    """
    Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Args:
        df: Input dataframe.
        y_true_col_names: Column name(s) representing actual values.
        y_pred_col_names: Column name(s) representing predicted values.
        labels: The set of labels to include when ``average != 'binary'``, and
            their order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in the y true and
            y pred columns are used in sorted order.
        pos_label: The class to report if ``average='binary'`` and the data is
            binary. If the data are multiclass or multilabel, this will be ignored;
            setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
            scores for that label only.
        average: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (y true, y pred) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division. If set to
            "warn", this acts as 0, but warnings are also raised.

    Returns:
        precision: float (if average is not None) or array of float, shape = (n_unique_labels,)
            Precision of the positive class in binary classification or weighted
            average of the precision of each class for the multiclass task.
    """
    p, _, _, _ = metrics.precision_recall_fscore_support(
        df=df,
        y_true_col_names=y_true_col_names,
        y_pred_col_names=y_pred_col_names,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("precision",),
        sample_weight_col_name=sample_weight_col_name,
        zero_division=zero_division,
    )
    return p
