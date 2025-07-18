import inspect
import json
import math
import warnings
from typing import Any, Iterable, Optional, Union

import cloudpickle
import numpy as np
import numpy.typing as npt
import sklearn
from packaging import version
from sklearn import exceptions, metrics

import snowflake.snowpark._internal.utils as snowpark_utils
from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import result
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    generate_random_alphanumeric,
    random_name_for_temp_object,
)
from snowflake.snowpark.functions import udtf

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def accuracy_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    normalize: bool = True,
    sample_weight_col_name: Optional[str] = None,
) -> float:
    """
    Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in the y true columns.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        normalize: boolean, default=True
            If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.

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
    return metrics_utils.weighted_sum(
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
) -> Union[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_name: string or list of strings
            Column name representing actual values.
        y_pred_col_name: string or list of strings
            Column name representing predicted values.
        labels: list of labels, default=None
            List of labels to index the matrix. This may be used to
            reorder or select a subset of labels.
            If ``None`` is given, those that appear at least once in the
            y true or y pred column are used in sorted order.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
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


def _register_confusion_matrix_computer(*, session: snowpark.Session, statement_params: dict[str, Any]) -> str:
    """Registers confusion matrix computation UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session: Snowpark session.
        statement_params: Dictionary used for tagging queries for tracking purposes.

    Returns:
        Name of the UDTF.
    """
    batch_size = metrics_utils.BATCH_SIZE

    class ConfusionMatrixComputer:
        def __init__(self) -> None:
            self._initialized = False
            self._confusion_matrix = np.zeros((1, 1))
            # 2d array containing a batch of input rows. A batch contains metrics_utils.BATCH_SIZE rows.
            # [sample_weight, y_true, y_pred]
            self._batched_rows = np.zeros((batch_size, 1))
            # Number of columns in the dataset.
            self._n_cols = -1
            # Running count of number of rows added to self._batched_rows.
            self._cur_count = 0
            # Number of labels.
            self._n_label = 0

        def process(self, input_row: list[float], n_label: int) -> None:
            """Computes confusion matrix.

            Args:
                input_row: List of floats: [sample_weight, y_true, y_pred].
                n_label: Number of labels.
            """
            # 1. Initialize variables.
            if not self._initialized:
                self._n_cols = len(input_row)
                self._batched_rows = np.zeros((batch_size, self._n_cols))
                self._n_label = n_label
                self._confusion_matrix = np.zeros((self._n_label, self._n_label))
            self._initialized = True

            self._batched_rows[self._cur_count, :] = input_row
            self._cur_count += 1

            # 2. Compute incremental confusion matrix for the batch.
            if self._cur_count >= batch_size:
                self.update_confusion_matrix()
                self._cur_count = 0

        def end_partition(self) -> Iterable[tuple[bytes, str]]:
            # 3. Compute sum and dot_prod for the remaining rows in the batch.
            if self._cur_count > 0:
                self.update_confusion_matrix()
            for i in range(self._n_label):
                yield cloudpickle.dumps(self._confusion_matrix[i, :]), "row_" + str(i)

        def update_confusion_matrix(self) -> None:
            # Update the confusion matrix by adding values from the 1st column of the batched rows to specific
            # locations in the confusion matrix determined by row and column indices from the 2nd and 3rd columns of
            # the batched rows.
            np.add.at(
                self._confusion_matrix,
                (
                    self._batched_rows[: self._cur_count][:, 1].astype(int),
                    self._batched_rows[: self._cur_count][:, 2].astype(int),
                ),
                self._batched_rows[: self._cur_count][:, 0],
            )

    confusion_matrix_computer = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE_FUNCTION)
    session.udtf.register(
        ConfusionMatrixComputer,
        output_schema=T.StructType(
            [
                T.StructField("result", T.BinaryType()),
                T.StructField("part", T.StringType()),
            ]
        ),
        input_types=[T.ArrayType(), T.IntegerType()],
        packages=[f"numpy=={np.__version__}", f"cloudpickle=={cloudpickle.__version__}"],
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
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float64]]:
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
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        labels: list of labels, default=None
            The set of labels to include when ``average != 'binary'``, and
            their order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in the y true and
            y pred columns are used in sorted order.
        pos_label:  string or integer, default=1
            The class to report if ``average='binary'`` and the data is
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
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
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
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    beta: float,
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of recall in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        beta: float
            Determines the weight of recall in the combined score.
        labels: list of labels, default=None
            The set of labels to include when ``average != 'binary'``, and
            their order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in the y true and
            y pred columns are used in sorted order.
        pos_label: string or integer, default=1
            The class to report if ``average='binary'`` and the data is
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
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
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
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
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
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted probabilities,
            as returned by a classifier's predict_proba method.
            If ``y_pred.shape = (n_samples,)`` the probabilities provided are
            assumed to be that of the positive class. The labels in ``y_pred``
            are assumed to be ordered alphabetically, as done by `LabelBinarizer`.
        eps: float or "auto", default="auto"
            Deprecated: if specified, it will be ignored and a warning emitted. Retained
            for backward compatibility.
        normalize: boolean, default=True
            If true, return the mean loss per sample.
            Otherwise, return the sum of the per-sample losses.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        labels: list of labels, default=None
            If not provided, labels will be inferred from y_true. If ``labels``
            is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
            assumed to be binary and are inferred from ``y_true``.

    Returns:
        Log loss, aka logistic loss or cross-entropy loss.
    """
    session = df._session
    assert session is not None
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    y_true = y_true_col_names if isinstance(y_true_col_names, list) else [y_true_col_names]
    y_pred = y_pred_col_names if isinstance(y_pred_col_names, list) else [y_pred_col_names]

    if eps != "auto":
        warnings.warn("log_loss eps argument is deprecated and will be ignored.", DeprecationWarning, stacklevel=2)

    # If it is binary classification, use SQL because it is faster.
    if len(y_pred) == 1:
        metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)
        eps = float(np.finfo(float).eps)
        y_true_col = y_true[0]
        y_pred_col = y_pred[0]
        y_pred_eps_min = F.iff(df[y_pred_col] < (1 - eps), df[y_pred_col], 1 - eps)
        y_pred_eps = F.iff(y_pred_eps_min > eps, y_pred_eps_min, eps)
        neg_loss_column = F.iff(df[y_true_col] == 1, F.log(math.e, y_pred_eps), F.log(math.e, 1 - y_pred_eps))
        loss_column = F.negate(neg_loss_column)
        return metrics_utils.weighted_sum(
            df=df,
            sample_score_column=loss_column,
            sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
            normalize=normalize,
            statement_params=statement_params,
        )

    # Since we are processing samples individually, we need to explicitly specify the output labels
    # in the case that there is one output label.
    if len(y_true) == 1 and not labels:
        labels = json.loads(df.select(F.array_unique_agg(y_true[0])).collect(statement_params=statement_params)[0][0])

    normalize_sum = None
    if normalize:
        if sample_weight_col_name:
            normalize_sum = float(
                df.select(F.sum(df[sample_weight_col_name])).collect(statement_params=statement_params)[0][0]
            )
        else:
            normalize_sum = df.count()

    log_loss_computer = _register_log_loss_computer(
        session=session,
        statement_params=statement_params,
        labels=labels,
    )
    log_loss_computer_udtf = F.table_function(log_loss_computer)

    if sample_weight_col_name:
        temp_df = df.select(
            F.array_construct(*y_true).alias("y_true_cols"),
            F.array_construct(*y_pred).alias("y_pred_cols"),
            sample_weight_col_name,
        )
        res_df = temp_df.select(
            log_loss_computer_udtf(F.col("y_true_cols"), F.col("y_pred_cols"), F.col(sample_weight_col_name))
        )
    else:
        temp_df = df.select(
            F.array_construct(*y_true).alias("y_true_cols"),
            F.array_construct(*y_pred).alias("y_pred_cols"),
        )
        temp_df = temp_df.with_column("sample_weight_col", F.lit(1.0))
        res_df = temp_df.select(
            log_loss_computer_udtf(F.col("y_true_cols"), F.col("y_pred_cols"), F.col("sample_weight_col"))
        )

    total_loss = float(res_df.select(F.sum(res_df["log_loss"])).collect(statement_params=statement_params)[0][0])

    return total_loss / normalize_sum if normalize_sum and normalize_sum > 0 else total_loss


def _register_log_loss_computer(
    *,
    session: snowpark.Session,
    statement_params: dict[str, Any],
    labels: Optional[npt.ArrayLike] = None,
) -> str:
    """Registers log loss computation UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session: Snowpark session.
        statement_params: Dictionary used for tagging queries for tracking purposes.
        labels: If not provided, labels will be inferred from y_true. If ``labels``
            is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
            assumed to be binary and are inferred from ``y_true``.

    Returns:
        Name of the UDTF.
    """

    class LogLossComputer:
        def __init__(self) -> None:
            self._labels = labels
            self._y_true: list[list[int]] = []
            self._y_pred: list[list[float]] = []
            self._sample_weight: list[float] = []

        def process(self, y_true: list[int], y_pred: list[float], sample_weight: float) -> None:
            self._y_true.append(y_true)
            self._y_pred.append(y_pred)
            self._sample_weight.append(sample_weight)

        def end_partition(self) -> Iterable[tuple[float]]:
            res = metrics.log_loss(
                self._y_true,
                self._y_pred,
                normalize=False,
                sample_weight=self._sample_weight,
                labels=self._labels,
            )
            yield (float(res),)

    log_loss_computer = random_name_for_temp_object(TempObjectType.TABLE_FUNCTION)
    sklearn_release = version.parse(sklearn.__version__).release
    session.udtf.register(
        LogLossComputer,
        output_schema=T.StructType(
            [
                T.StructField("log_loss", T.FloatType()),
            ]
        ),
        packages=[f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*"],
        name=log_loss_computer,
        is_permanent=False,
        replace=True,
        statement_params=statement_params,
    )
    return log_loss_computer


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def precision_recall_fscore_support(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    beta: float = 1.0,
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = None,
    warn_for: Union[tuple[str, ...], set[str]] = ("precision", "recall", "f-score"),
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[
    tuple[float, float, float, None],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
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
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        beta: float, default=1.0
            The strength of recall versus precision in the F-score.
        labels: list of labels, default=None
            The set of labels to include when ``average != 'binary'``, and
            their order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in the y true and
            y pred columns are used in sorted order.
        pos_label: string or integer, default=1
            The class to report if ``average='binary'`` and the data is
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
        warn_for: tuple or set containing "precision", "recall", or "f-score"
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
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
    if average == "samples":
        metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

        session = df._session
        assert session is not None
        sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)
        sklearn_release = version.parse(sklearn.__version__).release
        statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

        cols = metrics_utils.flatten_cols([y_true_col_names, y_pred_col_names, sample_weight_col_name])
        queries = df[cols].queries["queries"]

        pickled_result_module = cloudpickle.dumps(result)

        @F.sproc(  # type: ignore[misc]
            is_permanent=False,
            session=session,
            name=sproc_name,
            replace=True,
            packages=[
                f"cloudpickle=={cloudpickle.__version__}",
                f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
                "snowflake-snowpark-python",
            ],
            statement_params=statement_params,
            anonymous=True,
        )
        def precision_recall_fscore_support_anon_sproc(session: snowpark.Session) -> bytes:
            for query in queries[:-1]:
                _ = session.sql(query).collect(statement_params=statement_params)
            sp_df = session.sql(queries[-1])
            df = sp_df.to_pandas(statement_params=statement_params)
            df.columns = sp_df.columns

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

            result_module = cloudpickle.loads(pickled_result_module)
            return result_module.serialize(session, (p, r, f, s, warning))  # type: ignore[no-any-return]

        kwargs = telemetry.get_sproc_statement_params_kwargs(
            precision_recall_fscore_support_anon_sproc, statement_params
        )
        result_object = result.deserialize(session, precision_recall_fscore_support_anon_sproc(session, **kwargs))

        res: Union[
            tuple[float, float, float, None],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ] = result_object[:4]
        warning = result_object[-1]
        if warning:
            warnings.warn(warning.message, category=warning.category, stacklevel=2)
        return res

    # Distributed when average != "samples"
    session = df._session
    assert session is not None

    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)
    metrics_utils.validate_average_pos_label(average, pos_label)
    zero_division_value = _check_zero_division(zero_division)

    y_true = y_true_col_names if isinstance(y_true_col_names, list) else [y_true_col_names]
    y_pred = y_pred_col_names if isinstance(y_pred_col_names, list) else [y_pred_col_names]
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    samplewise = average == "samples"

    # Explicitly specify labels.
    if not labels:
        lbls = set()
        cols = y_true + y_pred
        col_lbls_lists = df.select(*[F.array_unique_agg(col) for col in cols]).collect(
            statement_params=statement_params
        )[0]
        for col_lbls_list in col_lbls_lists:
            for col_lbl in json.loads(col_lbls_list):
                lbls.add(col_lbl)
        labels = sorted(list(lbls))
        if average == "binary":
            labels = _check_binary_labels(labels, pos_label=pos_label)

    normalize_sum: float = 0
    if samplewise:
        if sample_weight_col_name:
            normalize_sum = float(
                df.select(F.sum(df[sample_weight_col_name])).collect(statement_params=statement_params)[0][0]
            )
        else:
            normalize_sum = df.count()

    multilabel_confusion_matrix_computer = _register_multilabel_confusion_matrix_computer(
        session=session, labels=labels, samplewise=samplewise
    )
    multilabel_confusion_matrix_udtf = F.table_function(multilabel_confusion_matrix_computer)
    if sample_weight_col_name:
        temp_df = df.select(
            F.array_construct(*y_true).alias("y_true_cols"),
            F.array_construct(*y_pred).alias("y_pred_cols"),
            F.col(sample_weight_col_name),
        )
    else:
        temp_df = df.select(
            F.array_construct(*y_true).alias("y_true_cols"),
            F.array_construct(*y_pred).alias("y_pred_cols"),
        )
        sample_weight_col_name = "sample_weight_col"
        temp_df = temp_df.with_column(sample_weight_col_name, F.lit(1.0))
    mcm_df = temp_df.select(
        multilabel_confusion_matrix_udtf(F.col("y_true_cols"), F.col("y_pred_cols"), F.col(sample_weight_col_name))
    )

    if samplewise:
        # Each column already contains samplewise results. Will be processed differently
        # than others, with everything done in SQL because it is row by row.
        zero_division_val = float(zero_division_value) if zero_division_value != np.nan else "NaN"
        mcm_df = mcm_df.with_column(
            "precision",
            F.iff(mcm_df["PRED_SUM"][0] == 0, zero_division_val, mcm_df["TP_SUM"][0] / mcm_df["PRED_SUM"][0]),
        )
        mcm_df = mcm_df.with_column(
            "recall", F.iff(mcm_df["TRUE_SUM"][0] == 0, zero_division_val, mcm_df["TP_SUM"][0] / mcm_df["TRUE_SUM"][0])
        )
        if np.isposinf(beta):
            mcm_df = mcm_df.with_column("f_score", mcm_df["recall"])
        elif beta == 0:
            mcm_df = mcm_df.with_column("f_score", mcm_df["precision"])
        else:
            beta_squared = beta**2
            mcm_df = mcm_df.with_column("denom", beta_squared * mcm_df["precision"] + mcm_df["recall"])
            mcm_df = mcm_df.with_column(
                "f-score",
                F.iff(
                    mcm_df["denom"] == 0,
                    zero_division_val,
                    (1 + beta_squared) * mcm_df["precision"] * mcm_df["recall"] / mcm_df["denom"],
                ),
            )

        total_precision, total_recall, total_fscore = mcm_df.select(
            F.sum(mcm_df["precision"]), F.sum(mcm_df["recall"]), F.sum(mcm_df["f-score"])
        ).collect(statement_params=statement_params)[0]
        return (
            (total_precision / normalize_sum, total_recall / normalize_sum, total_fscore / normalize_sum, None)
            if normalize_sum > 0
            else (0, 0, 0, None)
        )

    tp_sum_df = _sum_array_col(mcm_df, "TP_SUM")
    pred_sum_df = _sum_array_col(mcm_df, "PRED_SUM")
    true_sum_df = _sum_array_col(mcm_df, "TRUE_SUM")

    # Aggregated TP_SUM, PRED_SUM, TRUE_SUM as 1D arrays.
    tp_sum = np.array(json.loads(tp_sum_df.collect(statement_params=statement_params)[0][0]))
    pred_sum = np.array(json.loads(pred_sum_df.collect(statement_params=statement_params)[0][0]))
    true_sum = np.array(json.loads(true_sum_df.collect(statement_params=statement_params)[0][0]))

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta_squared = beta**2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division)
    recall = _prf_divide(tp_sum, true_sum, "recall", "true", average, warn_for, zero_division)

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    if np.isposinf(beta):
        f_score = recall
    elif beta == 0:
        f_score = precision
    else:
        # The score is defined as:
        # score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        # We set to `zero_division_value` if the denominator is 0 **or** if **both**
        # precision and recall are ill-defined.
        denom = beta_squared * precision + recall
        mask = np.isclose(denom, 0) | np.isclose(pred_sum + true_sum, 0)
        denom[mask] = 1  # avoid division by 0
        f_score = (1 + beta_squared) * precision * recall / denom
        f_score[mask] = zero_division_value

    # Average the results
    if average == "weighted":
        weights = true_sum
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision_avg = _nanaverage(precision, weights=weights)
        recall_avg = _nanaverage(recall, weights=weights)
        f_score_avg = _nanaverage(f_score, weights=weights)

        return precision_avg, recall_avg, f_score_avg, None  # return no support

    return precision, recall, f_score, true_sum


def _register_multilabel_confusion_matrix_computer(
    *,
    session: snowpark.Session,
    labels: Optional[npt.ArrayLike] = None,
    samplewise: bool,
) -> str:
    """Registers multilabel confusion matrix computation UDTF in Snowflake and returns the name of the UDTF.

    Args:
        session: Snowpark session.
        labels : array-like of shape (n_classes,), default=None
            A list of classes or column indices to select some (or to force
            inclusion of classes absent from the data).
        samplewise : bool, default=False
            In the multilabel case, this calculates a confusion matrix per sample.

    Returns:
        Name of the UDTF.
    """

    class MultilabelConfusionMatrixComputer:
        def __init__(self) -> None:
            self._labels = labels
            self._samplewise = samplewise
            self._y_true: list[list[int]] = []
            self._y_pred: list[list[int]] = []
            self._sample_weight: list[float] = []

        def process(self, y_true: list[int], y_pred: list[int], sample_weight: float) -> None:
            self._y_true.append(y_true)
            self._y_pred.append(y_pred)
            self._sample_weight.append(sample_weight)

        def end_partition(
            self,
        ) -> Iterable[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
            MCM = metrics.multilabel_confusion_matrix(
                self._y_true,
                self._y_pred,
                sample_weight=self._sample_weight,
                labels=self._labels,
                samplewise=self._samplewise,
            )
            tp_sum = MCM[:, 1, 1]
            pred_sum = tp_sum + MCM[:, 0, 1]
            true_sum = tp_sum + MCM[:, 1, 0]
            if samplewise:
                tp_sum = tp_sum * self._sample_weight
            yield (tp_sum, pred_sum, true_sum)

    multilabel_confusion_matrix_computer = random_name_for_temp_object(TempObjectType.TABLE_FUNCTION)
    sklearn_release = version.parse(sklearn.__version__).release
    session.udtf.register(
        MultilabelConfusionMatrixComputer,
        output_schema=T.StructType(
            [
                T.StructField("TP_SUM", T.ArrayType()),
                T.StructField("PRED_SUM", T.ArrayType()),
                T.StructField("TRUE_SUM", T.ArrayType()),
            ]
        ),
        packages=[f"numpy=={np.__version__}", f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*"],
        name=multilabel_confusion_matrix_computer,
        is_permanent=False,
        replace=True,
        statement_params=telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), None),
            api_calls=[udtf],
        ),
    )
    return multilabel_confusion_matrix_computer


def _binary_precision_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    pos_label: Union[str, int] = 1,
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float64]]:

    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    if isinstance(y_true_col_names, str):
        y_true_col_names = [y_true_col_names]
    if isinstance(y_pred_col_names, str):
        y_pred_col_names = [y_pred_col_names]

    if len(y_pred_col_names) != len(y_true_col_names):
        raise ValueError(
            "precision_score: `y_true_col_names` and `y_pred_column_names` must be lists of the same length "
            "or both strings."
        )

    # Confirm that the data is binary.
    labels_set = set()
    columns = y_true_col_names + y_pred_col_names
    column_labels_lists = df.select(*[F.array_unique_agg(col) for col in columns]).collect(
        statement_params=statement_params
    )[0]
    for column_labels_list in column_labels_lists:
        for column_label in json.loads(column_labels_list):
            labels_set.add(column_label)
    labels = sorted(list(labels_set))
    _ = _check_binary_labels(labels, pos_label=pos_label)

    sample_weight_column = df[sample_weight_col_name] if sample_weight_col_name else None

    scores = []
    for y_true, y_pred in zip(y_true_col_names, y_pred_col_names):
        tp_col = F.iff((F.col(y_true) == pos_label) & (F.col(y_pred) == pos_label), 1, 0)
        fp_col = F.iff((F.col(y_true) != pos_label) & (F.col(y_pred) == pos_label), 1, 0)
        tp = metrics_utils.weighted_sum(
            df=df,
            sample_score_column=tp_col,
            sample_weight_column=sample_weight_column,
            statement_params=statement_params,
        )
        fp = metrics_utils.weighted_sum(
            df=df,
            sample_score_column=fp_col,
            sample_weight_column=sample_weight_column,
            statement_params=statement_params,
        )

        try:
            score = tp / (tp + fp)
        except ZeroDivisionError:
            if zero_division == "warn":
                msg = "precision_score: division by zero: score value will be 0."
                warnings.warn(msg, exceptions.UndefinedMetricWarning, stacklevel=2)
                score = 0.0
            else:
                score = float(zero_division)

        scores.append(score)

    if len(scores) == 1:
        return scores[0]

    return np.array(scores)


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def precision_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        labels: list of labels, default=None
            The set of labels to include when ``average != 'binary'``, and
            their order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in the y true and
            y pred columns are used in sorted order.
        pos_label: string or integer, default=1
            The class to report if ``average='binary'`` and the data is
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
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division. If set to
            "warn", this acts as 0, but warnings are also raised.

    Returns:
        precision - float (if average is not None) or array of float, shape = (n_unique_labels,)
            Precision of the positive class in binary classification or weighted
            average of the precision of each class for the multiclass task.
    """
    if average == "binary":
        return _binary_precision_score(
            df=df,
            y_true_col_names=y_true_col_names,
            y_pred_col_names=y_pred_col_names,
            pos_label=pos_label,
            sample_weight_col_name=sample_weight_col_name,
            zero_division=zero_division,
        )

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
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    labels: Optional[npt.ArrayLike] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight_col_name: Optional[str] = None,
    zero_division: Union[str, int] = "warn",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        labels: list of labels, default=None
            The set of labels to include when ``average != 'binary'``, and
            their order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in the y true and
            y pred columns are used in sorted order.
        pos_label: string or integer, default=1
            The class to report if ``average='binary'`` and the data is
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
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
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


def _sum_array_col(df: snowpark.DataFrame, col_name: str) -> snowpark.DataFrame:
    """Calculates element-wise sum of an array column in a DataFrame

    For example, if we have this:

    col_name
    [1, 2, 3]
    [4, 5, 6]

    The result will be:

    col_name
    [5, 7, 9]

    Args:
        df: Input dataframe.
        col_name: Array column to sum.

    Returns:
        snowpark.DataFrame: Returns a dataframe with the sum array in a column with name col_name.
    """
    temp_df = df.flatten(col_name).group_by(F.col("INDEX")).agg(F.sum("VALUE").alias("VALUE_SUM"))
    res_df: snowpark.DataFrame = temp_df.select(F.array_agg(temp_df["VALUE_SUM"]).within_group("INDEX").alias(col_name))
    return res_df


def _check_binary_labels(
    labels: list[Any],
    pos_label: Union[str, int] = 1,
) -> list[Any]:
    """Validation associated with binary average labels.

    Args:
        labels: List of labels.
        pos_label: The class to report if ``average='binary'`` and the data is
            binary.

    Returns:
        List[Any]: Identified labels.

    Raises:
        ValueError: Average setting of binary is incorrect or pos_label is invalid.
    """
    if len(labels) <= 2:
        if len(labels) == 2 and pos_label not in labels:
            raise ValueError(f"pos_label={pos_label} is not a valid label. It must be one of {labels}")
        labels = [pos_label]
    else:
        raise ValueError(
            "Cannot compute precision score with binary average: there are more than two labels present."
            "Please choose another average setting."
        )

    return labels


def _prf_divide(
    numerator: npt.NDArray[np.float64],
    denominator: npt.NDArray[np.float64],
    metric: str,
    modifier: str,
    average: Optional[str] = None,
    warn_for: Union[tuple[str, ...], set[str]] = ("precision", "recall", "f-score"),
    zero_division: Union[str, int] = "warn",
) -> npt.NDArray[np.float64]:
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0, 1 or np.nan (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.

    Args:
        numerator: Numerator in the division, an array of floats.
        denominator: Denominator in the division, an array of floats.
        metric: Name of the metric.
        modifier: Name of the modifier.
        average: Type of average being calculated, or None.
        warn_for: This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        zero_division: "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division. If set to
            "warn", this acts as 0, but warnings are also raised.

    Returns:
        npt.NDArray[np.float64]: Result of the division, an array of floats.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # set those with 0 denominator to `zero_division`, and 0 when "warn"
    zero_division_value = _check_zero_division(zero_division)
    result[mask] = zero_division_value

    # we assume the user will be removing warnings if zero_division is set
    # to something different than "warn". If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn" or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and "f-score" in warn_for:
        msg_start = f"{metric.title()} and F-score are"
    elif metric in warn_for:
        msg_start = f"{metric.title()} is"
    elif "f-score" in warn_for:
        msg_start = "F-score is"
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result


def _warn_prf(
    average: Optional[str],
    modifier: str,
    msg_start: str,
    result_size: int,
) -> None:
    """Function for precision, recall, f-score warnings.

    Args:
        average: Type of average being calculated, or None.
        modifier: String representing modifier.
        msg_start: String representing start of message.
        result_size: Result size as an int.
    """

    axis0, axis1 = "sample", "label"
    if average == "samples":
        axis0, axis1 = axis1, axis0
    msg = (
        "{0} ill-defined and being set to 0.0 {{0}} "
        "no {1} {2}s. Use `zero_division` parameter to control"
        " this behavior.".format(msg_start, modifier, axis0)
    )
    if result_size == 1:
        msg = msg.format("due to")
    else:
        msg = msg.format(f"in {axis1}s with")
    warnings.warn(msg, exceptions.UndefinedMetricWarning, stacklevel=2)


def _check_zero_division(zero_division: Union[int, float, str]) -> float:
    """Returns the value to use when division by zero occurs.

    Args:
        zero_division: The value to use, or the string "warn".

    Returns:
        float: The zero division value.
    """
    if isinstance(zero_division, str) and zero_division == "warn":
        return 0.0
    elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:
        return float(zero_division)
    else:  # np.isnan(zero_division)
        return np.nan


def _nanaverage(a: npt.NDArray[np.float64], weights: Optional[npt.ArrayLike] = None) -> Any:
    """Compute the weighted average, ignoring NaNs.

    Args:
        a: ndarray
            Array containing data to be averaged.
        weights: array-like, default=None
            An array of weights associated with the values in a. Each value in a
            contributes to the average according to its associated weight. The
            weights array can either be 1-D of the same shape as a. If `weights=None`,
            then all data in a are assumed to have a weight equal to one.

    Returns:
        Any: The weighted average or NaN.

    Notes
    -----
    This wrapper to combine :func:`numpy.average` and :func:`numpy.nanmean`, so
    that :func:`np.nan` values are ignored from the average and weights can
    be passed. Note that when possible, we delegate to the prime methods.
    """

    if len(a) == 0:
        return np.nan

    mask = np.isnan(a)
    if mask.all():
        return np.nan

    if weights is None:
        return np.nanmean(a)

    weights = np.array(weights, copy=False)
    a, weights = a[~mask], weights[~mask]
    try:
        return np.average(a, weights=weights)
    except ZeroDivisionError:
        # this is when all weights are zero, then ignore them
        return np.average(a)
