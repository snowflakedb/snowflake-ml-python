import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cloudpickle
import numpy as np
import numpy.typing as npt

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal import utils as snowpark_utils

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


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
        Confusion matrix whose i-th row and j-th column entry indicates the number of
        samples with true label being i-th class and predicted label being j-th class.

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

    rand = snowpark_utils.generate_random_alphanumeric()
    if sample_weight_col_name is None:
        sample_weight_col_name = f'"_SAMPLE_WEIGHT_{rand}"'
        df = df.with_column(sample_weight_col_name, F.lit(1))  # type: ignore[arg-type]

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
        F.array_construct(sample_weight_col_name, y_true_index_col, y_pred_index_col).alias(  # type: ignore[arg-type]
            "ARR_COL"
        )
    )
    temp_df2 = temp_df1.select(
        confusion_matrix_computer_udtf(F.col("ARR_COL"), F.lit(n_labels))  # type: ignore[arg-type]
    ).with_column_renamed("RESULT", "RES")
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
