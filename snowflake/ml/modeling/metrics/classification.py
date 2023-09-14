import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import cloudpickle
import numpy as np
import numpy.typing as npt
import sklearn
from packaging import version
from sklearn import exceptions, metrics

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    generate_random_alphanumeric,
    random_name_for_temp_object,
)

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
        y_true, y_pred = (
            (y_true_col_names, y_pred_col_names)
            if isinstance(y_true_col_names, str)
            else (y_true_col_names[0], y_pred_col_names[0])
        )
        score_column = F.iff(df[y_true] == df[y_pred], 1, 0)
    # multilabel
    else:
        expr = " and ".join([f"({y_true_col_names[i]} = {y_pred_col_names[i]})" for i in range(len(y_true_col_names))])
        score_column = F.iff(expr, 1, 0)
    return _weighted_sum(
        df=df,
        sample_score_column=score_column,
        sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
        normalize=normalize,
        statement_params=telemetry.get_statement_params(_PROJECT, _SUBPROJECT),
    )


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def confusion_matrix(
    *,
    df: snowpark.DataFrame,
    y_true_col_name: str,
    y_pred_col_name: str,
    labels: Optional[npt.ArrayLike] = None,
    sample_weight_col_name: Optional[str] = None,
    normalize: Optional[str] = None,
) -> Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Args:
        df: Input dataframe.
        y_true_col_name: Column name representing actual values.
        y_pred_col_name: Column name representing predicted values.
        labels: List of labels to index the matrix. This may be used to
            reorder or select a subset of labels.
            If ``None`` is given, those that appear at least once in the
            y true or y pred column are used in sorted order.
        sample_weight_col_name: Column name representing sample weights.
        normalize: {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.

    Returns:
        C: ndarray of shape (n_classes, n_classes)
            Confusion matrix whose i-th row and j-th
            column entry indicates the number of
            samples with true label being i-th class
            and predicted label being j-th class.

    Raises:
        ValueError: The given ``labels`` is empty.
        ValueError: No label specified in the given ``labels`` is in the y true column.
        ValueError: ``normalize`` is not one of {'true', 'pred', 'all', None}.
    """
    assert df._session is not None
    session = df._session

    # Get a label df with columns: [LABEL, INDEX].
    if labels is None:
        label_df = metrics_utils.unique_labels(df=df, columns=[df[y_true_col_name], df[y_pred_col_name]])
    else:
        _labels = np.array(labels)
        label_data = np.vstack((_labels, np.arange(_labels.size))).T.tolist()
        label_df = session.create_dataframe(label_data, schema=[metrics_utils.LABEL, metrics_utils.INDEX])

    n_labels = label_df.count()
    if labels is not None:
        if n_labels == 0:
            raise ValueError("'labels' should contains at least one label.")
        elif df[[y_true_col_name]].filter(~F.is_null(df[y_true_col_name])).count() == 0:
            return np.zeros((n_labels, n_labels), dtype=int)
        elif df[[y_true_col_name]].join(label_df, df[y_true_col_name] == label_df[metrics_utils.LABEL]).count() == 0:
            raise ValueError("At least one label specified must be in the y true column")

    rand = generate_random_alphanumeric()
    if sample_weight_col_name is None:
        sample_weight_col_name = f'"_SAMPLE_WEIGHT_{rand}"'
        df = df.with_column(sample_weight_col_name, F.lit(1))

    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    # Get indices of true and pred data.
    label_col = f'"_LABEL_{rand}"'
    y_true_index_col = f'"_Y_TRUE_INDEX_{rand}"'
    y_pred_index_col = f'"_Y_PRED_INDEX_{rand}"'
    label_df = label_df.with_column_renamed(metrics_utils.LABEL, label_col)
    ind_df = (
        df.join(
            label_df.with_column_renamed(metrics_utils.INDEX, y_true_index_col),
            df[y_true_col_name] == label_df[label_col],
        )
        .drop(label_col)
        .join(
            label_df.with_column_renamed(metrics_utils.INDEX, y_pred_index_col),
            df[y_pred_col_name] == label_df[label_col],
        )
        .drop(label_col)
    )

    # Register UDTFs.
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    confusion_matrix_computer = _register_confusion_matrix_computer(session=session, statement_params=statement_params)
    confusion_matrix_computer_udtf = F.table_function(confusion_matrix_computer)
    accumulator = metrics_utils.register_accumulator_udtf(session=session, statement_params=statement_params)
    accumulator_udtf = F.table_function(accumulator)

    # Compute the confusion matrix.
    temp_df1 = ind_df.select(
        F.array_construct(sample_weight_col_name, y_true_index_col, y_pred_index_col).alias("ARR_COL")
    )
    temp_df2 = temp_df1.select(confusion_matrix_computer_udtf(F.col("ARR_COL"), F.lit(n_labels))).with_column_renamed(
        "RESULT", "RES"
    )
    res_df = temp_df2.select(accumulator_udtf(F.col("RES")).over(partition_by="PART"), F.col("PART"))
    results = res_df.collect(statement_params=statement_params)

    cm = np.zeros((n_labels, n_labels))
    for i in range(len(results)):
        row = int(results[i][1].strip("row_"))
        cm[row, :] = cloudpickle.loads(results[i][0])

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm


def _register_confusion_matrix_computer(*, session: snowpark.Session, statement_params: Dict[str, Any]) -> str:
    """Registers confusion matrix computation UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session: Snowpark session.
        statement_params: Dictionary used for tagging queries for tracking purposes.

    Returns:
        Name of the UDTF.
    """

    class ConfusionMatrixComputer:
        BATCH_SIZE = 1000

        def __init__(self) -> None:
            self._initialized = False
            self._confusion_matrix = np.zeros((1, 1))
            # 2d array containing a batch of input rows. A batch contains self.BATCH_SIZE rows.
            # [sample_weight, y_true, y_pred]
            self._batched_rows = np.zeros((self.BATCH_SIZE, 1))
            # Number of columns in the dataset.
            self._n_cols = -1
            # Running count of number of rows added to self._batched_rows.
            self._cur_count = 0
            # Number of labels.
            self._n_label = 0

        def process(self, input_row: List[float], n_label: int) -> None:
            """Computes confusion matrix.

            Args:
                input_row: List of floats: [sample_weight, y_true, y_pred].
                n_label: Number of labels.
            """
            # 1. Initialize variables.
            if not self._initialized:
                self._n_cols = len(input_row)
                self._batched_rows = np.zeros((self.BATCH_SIZE, self._n_cols))
                self._n_label = n_label
                self._confusion_matrix = np.zeros((self._n_label, self._n_label))
            self._initialized = True

            self._batched_rows[self._cur_count, :] = input_row
            self._cur_count += 1

            # 2. Compute incremental sum and dot_prod for the batch.
            if self._cur_count >= self.BATCH_SIZE:
                self.update_confusion_matrix()
                self._cur_count = 0

        def end_partition(self) -> Iterable[Tuple[bytes, str]]:
            # 3. Compute sum and dot_prod for the remaining rows in the batch.
            if self._cur_count > 0:
                self.update_confusion_matrix()
            for i in range(self._n_label):
                yield cloudpickle.dumps(self._confusion_matrix[i, :]), "row_" + str(i)

        def update_confusion_matrix(self) -> None:
            np.add.at(
                self._confusion_matrix,
                (self._batched_rows[:, 1].astype(int), self._batched_rows[:, 2].astype(int)),
                self._batched_rows[:, 0],
            )

    # TODO(SNANDAMURI): Should we convert it to temp anonymous UDTF for it to work in Sproc?
    confusion_matrix_computer = "ConfusionMatrixComputer_{}".format(str(uuid.uuid4()).replace("-", "_").upper())
    session.udtf.register(
        ConfusionMatrixComputer,
        output_schema=T.StructType(
            [
                T.StructField("result", T.BinaryType()),
                T.StructField("part", T.StringType()),
            ]
        ),
        input_types=[T.ArrayType(), T.IntegerType()],
        packages=["numpy", "cloudpickle"],
        name=confusion_matrix_computer,
        is_permanent=False,
        replace=True,
        statement_params=statement_params,
    )
    return confusion_matrix_computer


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def f1_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a harmonic mean of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.

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
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (y true, y pred) are binary.
            ``'micro'``
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                func`accuracy_score`).
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division, i.e. when all
            predictions and labels are negative. If set to "warn", this acts as 0,
            but warnings are also raised.

    Returns:
        f1_score - float or array of float, shape = [n_unique_labels]
            F1 score of the positive class in binary classification or weighted
            average of the F1 scores of each class for the multiclass task.
    """
    return fbeta_score(
        df=df,
        y_true_col_names=y_true_col_names,
        y_pred_col_names=y_pred_col_names,
        beta=1.0,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight_col_name=sample_weight_col_name,
        zero_division=zero_division,
    )


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def fbeta_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    beta: float,
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of recall in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).

    Args:
        df: Input dataframe.
        y_true_col_names: Column name(s) representing actual values.
        y_pred_col_names: Column name(s) representing predicted values.
        beta: Determines the weight of recall in the combined score.
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
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (y true, y pred) are binary.
            ``'micro'``
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                func`accuracy_score`).
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division, i.e. when all
            predictions and labels are negative. If set to "warn", this acts as 0,
            but warnings are also raised.

    Returns:
        fbeta_score - float (if average is not None) or array of float, shape = [n_unique_labels]
            F-beta score of the positive class in binary classification or weighted
            average of the F-beta score of each class for the multiclass task.
    """
    _, _, f, _ = precision_recall_fscore_support(
        df=df,
        y_true_col_names=y_true_col_names,
        y_pred_col_names=y_pred_col_names,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("f-score",),
        sample_weight_col_name=sample_weight_col_name,
        zero_division=zero_division,
    )
    return f


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def log_loss(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    eps: Union[float, str] = "auto",
    normalize: bool = True,
    sample_weight_col_name: Optional[str] = None,
    labels: Optional[npt.ArrayLike] = None,
) -> float:
    r"""
    Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.
    For a single sample with true label :math:`y \in \{0,1\}` and
    a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
    loss is:

    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))

    Args:
        df: Input dataframe.
        y_true_col_names: Column name(s) representing actual values.
        y_pred_col_names: Column name(s) representing predicted probabilities,
            as returned by a classifier's predict_proba method.
            If ``y_pred.shape = (n_samples,)`` the probabilities provided are
            assumed to be that of the positive class. The labels in ``y_pred``
            are assumed to be ordered alphabetically, as done by `LabelBinarizer`.
        eps: float or "auto", default="auto"
            Log loss is undefined for p=0 or p=1, so probabilities are
            clipped to `max(eps, min(1 - eps, p))`. The default will depend on the
            data type of `y_pred` and is set to `np.finfo(y_pred.dtype).eps`.
        normalize: If true, return the mean loss per sample.
            Otherwise, return the sum of the per-sample losses.
        sample_weight_col_name: Column name representing sample weights.
        labels: If not provided, labels will be inferred from y_true. If ``labels``
            is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
            assumed to be binary and are inferred from ``y_true``.

    Returns:
        Log loss, aka logistic loss or cross-entropy loss.
    """
    session = df._session
    assert session is not None
    sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    cols = metrics_utils.flatten_cols([y_true_col_names, y_pred_col_names, sample_weight_col_name])
    queries = df[cols].queries["queries"]

    @F.sproc(  # type: ignore[misc]
        is_permanent=False,
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            "cloudpickle",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            "snowflake-snowpark-python",
        ],
        statement_params=statement_params,
        anonymous=True,
    )
    def log_loss_anon_sproc(session: snowpark.Session) -> float:
        for query in queries[:-1]:
            _ = session.sql(query).collect(statement_params=statement_params)
        df = session.sql(queries[-1]).to_pandas(statement_params=statement_params)
        y_true = df[y_true_col_names]
        y_pred = df[y_pred_col_names]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None
        return metrics.log_loss(  # type: ignore[no-any-return]
            y_true,
            y_pred,
            eps=eps,
            normalize=normalize,
            sample_weight=sample_weight,
            labels=labels,
        )

    loss: float = log_loss_anon_sproc(session)
    return loss


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
) -> Union[
    Tuple[float, float, float, None],
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]],
]:
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
            ``'binary'``
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (y true, y pred) are binary.
            ``'micro'``
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        warn_for: This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               * recall - when there are no positive labels
               * precision - when there are no positive predictions
               * f-score - both
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns:
        Tuple containing following items
            precision - float (if average is not None) or array of float, shape = [n_unique_labels]
                Precision score.
            recall - float (if average is not None) or array of float, shape = [n_unique_labels]
                Recall score.
            fbeta_score - float (if average is not None) or array of float, shape = [n_unique_labels]
                F-beta score.
            support - None (if average is not None) or array of int, shape = [n_unique_labels]
                The number of occurrences of each label in the y true column(s).
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

    session = df._session
    assert session is not None
    sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    cols = metrics_utils.flatten_cols([y_true_col_names, y_pred_col_names, sample_weight_col_name])
    queries = df[cols].queries["queries"]

    @F.sproc(  # type: ignore[misc]
        is_permanent=False,
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            "cloudpickle",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            "snowflake-snowpark-python",
        ],
        statement_params=statement_params,
        anonymous=True,
    )
    def precision_recall_fscore_support_anon_sproc(session: snowpark.Session) -> bytes:
        for query in queries[:-1]:
            _ = session.sql(query).collect(statement_params=statement_params)
        df = session.sql(queries[-1]).to_pandas(statement_params=statement_params)
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

    loaded_data = cloudpickle.loads(precision_recall_fscore_support_anon_sproc(session))
    res: Union[
        Tuple[float, float, float, None],
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]],
    ] = loaded_data[:4]
    warning = loaded_data[-1]
    if warning:
        warnings.warn(warning.message, category=warning.category)
    return res


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
) -> Union[float, npt.NDArray[np.float_]]:
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
            ``'binary'``
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (y true, y pred) are binary.
            ``'micro'``
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                func`accuracy_score`).
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division. If set to
            "warn", this acts as 0, but warnings are also raised.

    Returns:
        precision - float (if average is not None) or array of float, shape = (n_unique_labels,)
            Precision of the positive class in binary classification or weighted
            average of the precision of each class for the multiclass task.
    """
    p, _, _, _ = precision_recall_fscore_support(
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


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def recall_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, List[str]],
    y_pred_col_names: Union[str, List[str]],
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

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
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (y true, y pred) are binary.
            ``'micro'``
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall. Weighted recall
                is equal to accuracy.
            ``'samples'``
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                func`accuracy_score`).
        sample_weight_col_name: Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division. If set to
            "warn", this acts as 0, but warnings are also raised.

    Returns:
        recall - float (if average is not None) or array of float of shape (n_unique_labels,)
            Recall of the positive class in binary classification or weighted
            average of the recall of each class for the multiclass task.
    """
    _, r, _, _ = precision_recall_fscore_support(
        df=df,
        y_true_col_names=y_true_col_names,
        y_pred_col_names=y_pred_col_names,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("recall",),
        sample_weight_col_name=sample_weight_col_name,
        zero_division=zero_division,
    )
    return r


def _weighted_sum(
    *,
    df: snowpark.DataFrame,
    sample_score_column: snowpark.Column,
    sample_weight_column: Optional[snowpark.Column] = None,
    normalize: bool = False,
    statement_params: Dict[str, str],
) -> float:
    """Weighted sum of the sample score column.

    Args:
        df: Input dataframe.
        sample_score_column: Sample score column.
        sample_weight_column: Sample weight column.
        normalize: If ``False``, return the weighted sum.
            Otherwise, return the fraction of weighted sum.
        statement_params: Dictionary used for tagging queries for tracking purposes.

    Returns:
        If ``normalize == True``, return the fraction of weighted sum (float),
        else returns the weighted sum (int).
    """
    if normalize:
        if sample_weight_column is not None:
            res = F.sum(sample_score_column * sample_weight_column) / F.sum(sample_weight_column)
        else:
            res = F.avg(sample_score_column)
    elif sample_weight_column is not None:
        res = F.sum(sample_score_column * sample_weight_column)
    else:
        res = F.sum(sample_score_column)

    return float(df.select(res).collect(statement_params=statement_params)[0][0])
