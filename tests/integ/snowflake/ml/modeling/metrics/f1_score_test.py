from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import exceptions, metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils
from tests.integ.snowflake.ml.modeling.metrics import generator

_TYPES = [utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT]
_BINARY_LOW, _BINARY_HIGH = 0, 2
_MULTICLASS_LOW, _MULTICLASS_HIGH = 0, 5
_BINARY_DATA_LIST, _SF_SCHEMA = generator.gen_test_cases(_TYPES, _BINARY_LOW, _BINARY_HIGH)
_MULTICLASS_DATA_LIST, _ = generator.gen_test_cases(_TYPES, _MULTICLASS_LOW, _MULTICLASS_HIGH)
_REGULAR_BINARY_DATA_LIST, _LARGE_BINARY_DATA = _BINARY_DATA_LIST[:-1], _BINARY_DATA_LIST[-1]
_REGULAR_MULTICLASS_DATA_LIST, _LARGE_MULTICLASS_DATA = _MULTICLASS_DATA_LIST[:-1], _MULTICLASS_DATA_LIST[-1]
_Y_TRUE_COL = _SF_SCHEMA[1]
_Y_PRED_COL = _SF_SCHEMA[2]
_Y_TRUE_COLS = [_SF_SCHEMA[1], _SF_SCHEMA[2]]
_Y_PRED_COLS = [_SF_SCHEMA[3], _SF_SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SF_SCHEMA[5]


class F1ScoreTest(parameterized.TestCase):
    """Test F1 score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
        labels=[None, [2, 0, 4]],
    )
    def test_labels(self, data_index: int, labels: Optional[npt.ArrayLike]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COL,
            y_pred_col_names=_Y_PRED_COL,
            average=None,
            labels=labels,
        )
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            average=None,
            labels=labels,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
        pos_label=[0, 2, 4],
    )
    def test_pos_label(self, data_index: int, pos_label: Union[str, int]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COL,
            y_pred_col_names=_Y_PRED_COL,
            average=None,
            pos_label=pos_label,
        )
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            average=None,
            pos_label=pos_label,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
        average=[None, "micro", "macro", "weighted"],
    )
    def test_average_multiclass(self, data_index: int, average: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COL,
            y_pred_col_names=_Y_PRED_COL,
            average=average,
        )
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            average=average,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.product(
        (
            dict(y_true=_Y_TRUE_COL, y_pred=_Y_PRED_COL, average="binary"),
            dict(y_true=_Y_TRUE_COLS, y_pred=_Y_PRED_COLS, average="samples"),
        ),
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
    )
    def test_average_binary_samples(
        self,
        y_true: Union[str, list[str]],
        y_pred: Union[str, list[str]],
        average: Optional[str],
        data_index: int,
    ) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=y_true,
            y_pred_col_names=y_pred,
            average=average,
        )
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[y_true],
            pandas_df[y_pred],
            average=average,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
        sample_weight_col_name=[None, _SAMPLE_WEIGHT_COL],
    )
    def test_sample_weight_binary(self, data_index: int, sample_weight_col_name: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COLS,
            y_pred_col_names=_Y_PRED_COLS,
            average=None,
            sample_weight_col_name=sample_weight_col_name,
        )
        sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COLS],
            pandas_df[_Y_PRED_COLS],
            average=None,
            sample_weight=sample_weight,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_MULTICLASS_DATA_LIST))),
        sample_weight_col_name=[None, _SAMPLE_WEIGHT_COL],
    )
    def test_sample_weight_multiclass(self, data_index: int, sample_weight_col_name: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_MULTICLASS_DATA_LIST[data_index], _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COL,
            y_pred_col_names=_Y_PRED_COL,
            average=None,
            sample_weight_col_name=sample_weight_col_name,
        )
        sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            average=None,
            sample_weight=sample_weight,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.product(  # type: ignore[misc]
        zero_division=[0, 1],
    )
    def test_zero_division(self, zero_division: Union[str, int]) -> None:
        data = [
            [0, 0, 0, 0, 0, 0],
        ]
        pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

        if zero_division != "warn":
            actual_f = snowml_metrics.f1_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                zero_division=zero_division,
            )
            sklearn_f = sklearn_metrics.f1_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                zero_division=zero_division,
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

        # warn
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            zero_division="warn",
        )

        with self.assertWarns(exceptions.UndefinedMetricWarning):
            actual_f = snowml_metrics.f1_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                zero_division="warn",
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

    def test_with_large_num_of_rows_binary(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _LARGE_BINARY_DATA, _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COLS,
            y_pred_col_names=_Y_PRED_COLS,
            average=None,
        )
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COLS],
            pandas_df[_Y_PRED_COLS],
            average=None,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)

    def test_with_large_num_of_rows_multiclass(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _LARGE_MULTICLASS_DATA, _SF_SCHEMA)

        actual_f = snowml_metrics.f1_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COL,
            y_pred_col_names=_Y_PRED_COL,
            average=None,
        )
        sklearn_f = sklearn_metrics.f1_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            average=None,
        )
        np.testing.assert_allclose(actual_f, sklearn_f)


if __name__ == "__main__":
    main()
