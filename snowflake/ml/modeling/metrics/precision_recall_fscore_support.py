import warnings
from typing import List, Optional, Set, Tuple, Union

import cloudpickle
import numpy.typing as npt
from sklearn import exceptions, metrics

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F
from snowflake.snowpark._internal import utils as snowpark_utils

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def precision_recall_fscore_support(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    beta: float = 1.0,
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = None,
    warn_for: Union[Tuple[str, ...], Set[str]] = ("precision", "recall", "f-score"),
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[Tuple[float, float, float, None], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]:
    """
    Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label a negative sample as
    positive.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in the y true column(s).

    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Args:
        df: Input dataframe.
        y_true_col_names: Column name(s) representing actual values.
        y_pred_col_names: Column name(s) representing predicted values.
        beta: The strength of recall versus precision in the F-score.
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
        average: {'binary', 'micro', 'macro', 'samples', 'weighted'}, default=None
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
        warn_for: This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns:
        precision: float (if average is not None) or array of float, shape = [n_unique_labels]
            Precision score.
        recall: float (if average is not None) or array of float, shape = [n_unique_labels]
            Recall score.
        fbeta_score: float (if average is not None) or array of float, shape = [n_unique_labels]
            F-beta score.
        support: None (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in the y true column(s).
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

    session = df._session
    assert session is not None
    query = df.queries["queries"][-1]
    sproc_name = f"precision_recall_fscore_support_{snowpark_utils.generate_random_alphanumeric()}"
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    @F.sproc(  # type: ignore[misc]
        session=session,
        name=sproc_name,
        replace=True,
        packages=["cloudpickle", "scikit-learn", "snowflake-snowpark-python"],
        statement_params=statement_params,
    )
    def precision_recall_fscore_support_sproc(session: snowpark.Session) -> bytes:
        df = session.sql(query).to_pandas(statement_params=statement_params)
        y_true = df[y_true_col_names]
        y_pred = df[y_pred_col_names]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None

        with warnings.catch_warnings(record=True) as w:
            p, r, f, s = metrics.precision_recall_fscore_support(
                y_true,
                y_pred,
                beta=beta,
                labels=labels,
                pos_label=pos_label,
                average=average,
                warn_for=warn_for,
                sample_weight=sample_weight,
                zero_division=zero_division,
            )

            # handle zero_division warnings
            warning = None
            if len(w) > 0 and issubclass(w[-1].category, exceptions.UndefinedMetricWarning):
                warning = w[-1]

        return cloudpickle.dumps((p, r, f, s, warning))  # type: ignore[no-any-return]

    loaded_data = cloudpickle.loads(session.call(sproc_name))
    res: Union[
        Tuple[float, float, float, None], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]
    ] = loaded_data[:4]
    warning = loaded_data[-1]
    if warning:
        warnings.warn(warning.message, category=warning.category)
    return res
