from typing import Optional

import numpy as np
import numpy.typing as npt
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils
from tests.integ.snowflake.ml.modeling.metrics import generator

_TYPES = [utils.DataType.INTEGER] + [utils.DataType.FLOAT] * 4
_BINARY_LOW, _BINARY_HIGH = 0, [2, 1, 1, 1, 1]
_MULTICLASS_LOW, _MULTICLASS_HIGH = 0, [3, 1, 1, 1, 1]
_BINARY_DATA_LIST, _SF_SCHEMA = generator.gen_test_cases(_TYPES, _BINARY_LOW, _BINARY_HIGH)
_MULTICLASS_DATA_LIST, _ = generator.gen_test_cases(_TYPES, _MULTICLASS_LOW, _MULTICLASS_HIGH)
_REGULAR_BINARY_DATA_LIST, _LARGE_BINARY_DATA = _BINARY_DATA_LIST[:-1], _BINARY_DATA_LIST[-1]
_REGULAR_MULTICLASS_DATA_LIST, _LARGE_MULTICLASS_DATA = _MULTICLASS_DATA_LIST[:-1], _MULTICLASS_DATA_LIST[-1]
_BINARY_Y_TRUE_COL = _SF_SCHEMA[1]
_BINARY_Y_PRED_COL = _SF_SCHEMA[2]
_MULTICLASS_Y_TRUE_COL = _SF_SCHEMA[1]
_MULTICLASS_Y_PRED_COLS = [_SF_SCHEMA[2], _SF_SCHEMA[3], _SF_SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SF_SCHEMA[5]

_MULTILABEL_TYPES = [utils.DataType.INTEGER] * 3 + [utils.DataType.FLOAT] * 3
_MULTILABEL_LOW, _MULTILABEL_HIGH = 0, [2, 2, 2, 1, 1, 1]
_MULTILABEL_DATA_LIST, _MULTILABEL_SCHEMA = generator.gen_test_cases(
    _MULTILABEL_TYPES, _MULTILABEL_LOW, _MULTILABEL_HIGH
)
_REGULAR_MULTILABEL_DATA_LIST, _LARGE_MULTILABEL_DATA = _MULTILABEL_DATA_LIST[:-1], _MULTILABEL_DATA_LIST[-1]
_MULTILABEL_Y_TRUE_COLS = [_MULTILABEL_SCHEMA[1], _MULTILABEL_SCHEMA[2], _MULTILABEL_SCHEMA[3]]
_MULTILABEL_Y_PRED_COLS = [_MULTILABEL_SCHEMA[4], _MULTILABEL_SCHEMA[5], _MULTILABEL_SCHEMA[6]]


class LogLossTest(parameterized.TestCase):
    """Test log loss."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
    )
    def test_binary(self, data_index: int) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_BINARY_Y_TRUE_COL,
            y_pred_col_names=_BINARY_Y_PRED_COL,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_BINARY_Y_TRUE_COL],
            pandas_df[_BINARY_Y_PRED_COL],
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
    )
    def test_multiclass(self, data_index: int) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTICLASS_Y_TRUE_COL,
            y_pred_col_names=_MULTICLASS_Y_PRED_COLS,
            eps="auto",
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTICLASS_Y_TRUE_COL],
            pandas_df[_MULTICLASS_Y_PRED_COLS],
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
        normalize=[True, False],
    )
    def test_normalize_binary(self, data_index: int, normalize: bool) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_BINARY_Y_TRUE_COL,
            y_pred_col_names=_BINARY_Y_PRED_COL,
            normalize=normalize,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_BINARY_Y_TRUE_COL],
            pandas_df[_BINARY_Y_PRED_COL],
            normalize=normalize,
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
        normalize=[True, False],
    )
    def test_normalize_multiclass(self, data_index: int, normalize: bool) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTICLASS_Y_TRUE_COL,
            y_pred_col_names=_MULTICLASS_Y_PRED_COLS,
            normalize=normalize,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTICLASS_Y_TRUE_COL],
            pandas_df[_MULTICLASS_Y_PRED_COLS],
            normalize=normalize,
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
        sample_weight_col_name=[None, _SAMPLE_WEIGHT_COL],
    )
    def test_sample_weight_binary(self, data_index: int, sample_weight_col_name: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_BINARY_Y_TRUE_COL,
            y_pred_col_names=_BINARY_Y_PRED_COL,
            sample_weight_col_name=sample_weight_col_name,
        )
        sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_BINARY_Y_TRUE_COL],
            pandas_df[_BINARY_Y_PRED_COL],
            sample_weight=sample_weight,
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
        sample_weight_col_name=[None, _SAMPLE_WEIGHT_COL],
    )
    def test_sample_weight_multiclass(self, data_index: int, sample_weight_col_name: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTICLASS_Y_TRUE_COL,
            y_pred_col_names=_MULTICLASS_Y_PRED_COLS,
            sample_weight_col_name=sample_weight_col_name,
        )
        sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTICLASS_Y_TRUE_COL],
            pandas_df[_MULTICLASS_Y_PRED_COLS],
            sample_weight=sample_weight,
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
        labels=[None, [2, 0, 4]],
    )
    def test_labels(self, data_index: int, labels: Optional[npt.ArrayLike]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTICLASS_Y_TRUE_COL,
            y_pred_col_names=_MULTICLASS_Y_PRED_COLS,
            labels=labels,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTICLASS_Y_TRUE_COL],
            pandas_df[_MULTICLASS_Y_PRED_COLS],
            labels=labels,
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTILABEL_DATA_LIST))),
    )
    def test_multilabel(self, data_index: int) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTILABEL_DATA_LIST[data_index], _MULTILABEL_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_PRED_COLS],
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    def test_with_large_num_of_rows_binary(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _LARGE_BINARY_DATA, _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_BINARY_Y_TRUE_COL,
            y_pred_col_names=_BINARY_Y_PRED_COL,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_BINARY_Y_TRUE_COL],
            pandas_df[_BINARY_Y_PRED_COL],
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    def test_with_large_num_of_rows_multiclass(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _LARGE_MULTICLASS_DATA, _SF_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTICLASS_Y_TRUE_COL,
            y_pred_col_names=_MULTICLASS_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTICLASS_Y_TRUE_COL],
            pandas_df[_MULTICLASS_Y_PRED_COLS],
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)

    def test_with_large_num_of_rows_multilabel(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _LARGE_MULTILABEL_DATA, _MULTILABEL_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_PRED_COLS],
        )
        np.testing.assert_allclose(actual_loss, sklearn_loss)


if __name__ == "__main__":
    main()
