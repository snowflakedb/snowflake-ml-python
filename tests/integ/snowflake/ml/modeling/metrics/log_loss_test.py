from typing import Any, Dict

from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_ROWS = 100
_TYPES = [utils.DataType.INTEGER] + [utils.DataType.FLOAT] * 4
_BINARY_DATA, _SF_SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=[2, 1, 1, 1, 1],
)
_BINARY_Y_TRUE_COL = _SF_SCHEMA[1]
_BINARY_Y_PRED_COL = _SF_SCHEMA[2]
_MULTICLASS_DATA = [
    [0, 2, 0.29, 0.49, 0.22, 0.18],
    [1, 0, 0.33, 0.16, 0.51, 0.69],
    [2, 1, 0.54, 0.29, 0.17, 0.04],
    [3, 2, 0.27, 0.68, 0.05, 0.17],
    [4, 1, 0.82, 0.12, 0.06, 0.91],
    [5, 2, 0.08, 0.46, 0.46, 0.76],
]
_MULTICLASS_Y_TRUE_COL = _SF_SCHEMA[1]
_MULTICLASS_Y_PRED_COLS = [_SF_SCHEMA[2], _SF_SCHEMA[3], _SF_SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SF_SCHEMA[5]
_MULTILABEL_DATA = [
    [1, 0, 1, 0.8, 0.3, 0.6],
    [0, 1, 0, 0.2, 0.7, 0.4],
    [1, 1, 0, 0.9, 0.6, 0.2],
    [0, 0, 1, 0.1, 0.4, 0.8],
]
_MULTILABEL_SCHEMA = ["Y_0", "Y_1", "Y_2", "S_0", "S_1", "S_2"]
_MULTILABEL_Y_TRUE_COLS = [_MULTILABEL_SCHEMA[0], _MULTILABEL_SCHEMA[1], _MULTILABEL_SCHEMA[2]]
_MULTILABEL_Y_PRED_COLS = [_MULTILABEL_SCHEMA[3], _MULTILABEL_SCHEMA[4], _MULTILABEL_SCHEMA[5]]


class LogLossTest(parameterized.TestCase):
    """Test log loss."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "eps": ["auto", 0.1, 0.5, 0.99],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _BINARY_Y_TRUE_COL, "y_pred": _BINARY_Y_PRED_COL},
                    {"data": _MULTICLASS_DATA, "y_true": _MULTICLASS_Y_TRUE_COL, "y_pred": _MULTICLASS_Y_PRED_COLS},
                ],
            }
        },
    )
    def test_eps(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

            for eps in params["eps"]:
                actual_loss = snowml_metrics.log_loss(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    eps=eps,
                )
                sklearn_loss = sklearn_metrics.log_loss(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    eps=eps,
                )
                self.assertAlmostEqual(sklearn_loss, actual_loss)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "normalize": [True, False],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _BINARY_Y_TRUE_COL, "y_pred": _BINARY_Y_PRED_COL},
                    {"data": _MULTICLASS_DATA, "y_true": _MULTICLASS_Y_TRUE_COL, "y_pred": _MULTICLASS_Y_PRED_COLS},
                ],
            }
        },
    )
    def test_normalize(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

            for normalize in params["normalize"]:
                actual_loss = snowml_metrics.log_loss(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    normalize=normalize,
                )
                sklearn_loss = sklearn_metrics.log_loss(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    normalize=normalize,
                )
                self.assertAlmostEqual(sklearn_loss, actual_loss)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _BINARY_Y_TRUE_COL, "y_pred": _BINARY_Y_PRED_COL},
                    {"data": _MULTICLASS_DATA, "y_true": _MULTICLASS_Y_TRUE_COL, "y_pred": _MULTICLASS_Y_PRED_COLS},
                ],
            }
        },
    )
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

            for sample_weight_col_name in params["sample_weight_col_name"]:
                actual_loss = snowml_metrics.log_loss(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    sample_weight_col_name=sample_weight_col_name,
                )
                sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
                sklearn_loss = sklearn_metrics.log_loss(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    sample_weight=sample_weight,
                )
                self.assertAlmostEqual(sklearn_loss, actual_loss)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": [None, [2, 0, 4]]}},
    )
    def test_labels(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTICLASS_DATA, _SF_SCHEMA)

        for labels in params["labels"]:
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
            self.assertAlmostEqual(sklearn_loss, actual_loss)

    def test_multilabel(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTILABEL_DATA, _MULTILABEL_SCHEMA)

        actual_loss = snowml_metrics.log_loss(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.log_loss(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_PRED_COLS],
        )
        self.assertAlmostEqual(sklearn_loss, actual_loss)


if __name__ == "__main__":
    main()
