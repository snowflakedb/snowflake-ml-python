from typing import Optional, Tuple, Union

import cloudpickle
import numpy.typing as npt
from sklearn import metrics

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.snowpark import functions as F
from snowflake.snowpark._internal import utils as snowpark_utils

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def roc_curve(
    *,
    df: snowpark.DataFrame,
    y_true_col_name: str,
    y_score_col_name: str,
    pos_label: Optional[Union[str, int]] = None,
    sample_weight_col_name: Optional[str] = None,
    drop_intermediate: bool = True,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Compute Receiver operating characteristic (ROC).

    Note: this implementation is restricted to the binary classification task.

    Args:
        df: Input dataframe.
        y_true_col_name: Column name representing true binary labels.
            If labels are not either {-1, 1} or {0, 1}, then pos_label should be
            explicitly given.
        y_score_col_name: Column name representing target scores, can either
            be probability estimates of the positive class, confidence values,
            or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).
        pos_label: The label of the positive class.
            When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
            ``pos_label`` is set to 1, otherwise an error will be raised.
        sample_weight_col_name: Column name representing sample weights.
        drop_intermediate: Whether to drop some suboptimal thresholds which would
            not appear on a plotted ROC curve. This is useful in order to create
            lighter ROC curves.

    Returns:
        fpr: ndarray of shape (>2,)
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr : ndarray of shape (>2,)
            Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds : ndarray of shape = (n_thresholds,)
            Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    """
    session = df._session
    assert session is not None
    sproc_name = f"roc_curve_{snowpark_utils.generate_random_alphanumeric()}"
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    cols = [y_true_col_name, y_score_col_name]
    if sample_weight_col_name:
        cols.append(sample_weight_col_name)
    query = df[cols].queries["queries"][-1]

    @F.sproc(  # type: ignore[misc]
        session=session,
        name=sproc_name,
        replace=True,
        packages=["cloudpickle", "scikit-learn", "snowflake-snowpark-python"],
        statement_params=statement_params,
    )
    def roc_curve_sproc(session: snowpark.Session) -> bytes:
        df = session.sql(query).to_pandas(statement_params=statement_params)
        y_true = df[y_true_col_name]
        y_score = df[y_score_col_name]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true,
            y_score,
            pos_label=pos_label,
            sample_weight=sample_weight,
            drop_intermediate=drop_intermediate,
        )

        return cloudpickle.dumps((fpr, tpr, thresholds))  # type: ignore[no-any-return]

    loaded_data = cloudpickle.loads(session.call(sproc_name))
    res: Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike] = loaded_data
    return res
