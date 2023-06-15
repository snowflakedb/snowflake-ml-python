from typing import List, Optional, Tuple, Union

import cloudpickle
import numpy as np
import numpy.typing as npt
import sklearn
from packaging import version
from sklearn import metrics

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F
from snowflake.snowpark._internal import utils as snowpark_utils

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def roc_auc_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_score_col_names: Union[str, List[str]],
    average: Optional[str] = "macro",
    sample_weight_col_name: Optional[str] = None,
    max_fpr: Optional[float] = None,
    multi_class: str = "raise",
    labels: Optional[npt.ArrayLike] = None,
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply.

    Args:
        df: Input dataframe.
        y_true_col_names: Column name(s) representing true labels or binary label indicators.
            The binary and multiclass cases expect labels with shape (n_samples,)
            while the multilabel case expects binary label indicators with shape
            (n_samples, n_classes).
        y_score_col_names: Column name(s) representing target scores.
            * In the binary case, it corresponds to an array of shape
              `(n_samples,)`. Both probability estimates and non-thresholded
              decision values can be provided. The probability estimates correspond
              to the **probability of the class with the greater label**.
              The decision values correspond to the output of
              `estimator.decision_function`.
            * In the multiclass case, it corresponds to an array of shape
              `(n_samples, n_classes)` of probability estimates provided by the
              `predict_proba` method. The probability estimates **must**
              sum to 1 across the possible classes. In addition, the order of the
              class scores must correspond to the order of ``labels``,
              if provided, or else to the numerical or lexicographical order of
              the labels in ``y_true``.
            * In the multilabel case, it corresponds to an array of shape
              `(n_samples, n_classes)`. Probability estimates are provided by the
              `predict_proba` method and the non-thresholded decision values by
              the `decision_function` method. The probability estimates correspond
              to the **probability of the class with the greater label for each
              output** of the classifier.
        average: {'micro', 'macro', 'samples', 'weighted'} or None, default='macro'
            If ``None``, the scores for each class are returned.
            Otherwise, this determines the type of averaging performed on the data.
            Note: multiclass ROC AUC currently only handles the 'macro' and
            'weighted' averages. For multiclass targets, `average=None` is only
            implemented for `multi_class='ovr'` and `average='micro'` is only
            implemented for `multi_class='ovr'`.
            ``'micro'``:
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).
            ``'samples'``:
                Calculate metrics for each instance, and find their average.
            Will be ignored when ``y_true`` is binary.
        sample_weight_col_name: Column name representing sample weights.
        max_fpr: float > 0 and <= 1, default=None
            If not ``None``, the standardized partial AUC [2]_ over the range
            [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
            should be either equal to ``None`` or ``1.0`` as AUC ROC partial
            computation currently is not supported for multiclass.
        multi_class: {'raise', 'ovr', 'ovo'}, default='raise'
            Only used for multiclass targets. Determines the type of configuration
            to use. The default value raises an error, so either
            ``'ovr'`` or ``'ovo'`` must be passed explicitly.
            ``'ovr'``:
                Stands for One-vs-rest. Computes the AUC of each class
                against the rest [3]_ [4]_. This
                treats the multiclass case in the same way as the multilabel case.
                Sensitive to class imbalance even when ``average == 'macro'``,
                because class imbalance affects the composition of each of the
                'rest' groupings.
            ``'ovo'``:
                Stands for One-vs-one. Computes the average AUC of all
                possible pairwise combinations of classes [5]_.
                Insensitive to class imbalance when
                ``average == 'macro'``.
        labels: Only used for multiclass targets. List of labels that index the
            classes in ``y_score``. If ``None``, the numerical or lexicographical
            order of the labels in ``y_true`` is used.

    Returns:
        auc: Area Under the Curve score.
    """
    session = df._session
    assert session is not None
    sproc_name = f"roc_auc_score_{snowpark_utils.generate_random_alphanumeric()}"
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    cols = metrics_utils.flatten_cols([y_true_col_names, y_score_col_names, sample_weight_col_name])
    query = df[cols].queries["queries"][-1]

    @F.sproc(  # type: ignore[misc]
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            "cloudpickle",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            "snowflake-snowpark-python",
        ],
        statement_params=statement_params,
    )
    def roc_auc_score_sproc(session: snowpark.Session) -> bytes:
        df = session.sql(query).to_pandas(statement_params=statement_params)
        y_true = df[y_true_col_names]
        y_score = df[y_score_col_names]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None
        auc = metrics.roc_auc_score(
            y_true,
            y_score,
            average=average,
            sample_weight=sample_weight,
            max_fpr=max_fpr,
            multi_class=multi_class,
            labels=labels,
        )

        return cloudpickle.dumps(auc)  # type: ignore[no-any-return]

    auc: Union[float, npt.NDArray[np.float_]] = cloudpickle.loads(
        session.call(sproc_name, statement_params=statement_params)
    )
    return auc


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
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    cols = metrics_utils.flatten_cols([y_true_col_name, y_score_col_name, sample_weight_col_name])
    query = df[cols].queries["queries"][-1]

    @F.sproc(  # type: ignore[misc]
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            "cloudpickle",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            "snowflake-snowpark-python",
        ],
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

    loaded_data = cloudpickle.loads(session.call(sproc_name, statement_params=statement_params))
    res: Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike] = loaded_data
    return res
