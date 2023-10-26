from typing import Any, Dict

import numpy as np
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import exceptions, metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_ROWS = 100
_TYPES = [utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT]
_BINARY_DATA, _PD_SCHEMA, _SF_SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=2,
)
_MULTICLASS_DATA, _, _ = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=5,
)
_Y_TRUE_COL = _SF_SCHEMA[1]
_Y_PRED_COL = _SF_SCHEMA[2]
_Y_TRUE_COLS = [_SF_SCHEMA[1], _SF_SCHEMA[2]]
_Y_PRED_COLS = [_SF_SCHEMA[3], _SF_SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SF_SCHEMA[5]


class PrecisionScoreTest(parameterized.TestCase):
    """Test precision score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": [None, [2, 0, 4]]}},
    )
    def test_labels(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTICLASS_DATA, _PD_SCHEMA)

        for labels in params["labels"]:
            actual_p = snowml_metrics.precision_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                labels=labels,
                average=None,
            )
            sklearn_p = sklearn_metrics.precision_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                labels=labels,
                average=None,
            )
            np.testing.assert_allclose(actual_p, sklearn_p)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"pos_label": [0, 2, 4]}},
    )
    def test_pos_label(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTICLASS_DATA, _PD_SCHEMA)

        for pos_label in params["pos_label"]:
            actual_p = snowml_metrics.precision_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                pos_label=pos_label,
                average="micro",
            )
            sklearn_p = sklearn_metrics.precision_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                pos_label=pos_label,
                average="micro",
            )
            np.testing.assert_allclose(actual_p, sklearn_p)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _Y_TRUE_COLS, "y_pred": _Y_PRED_COLS},
                    {"data": _MULTICLASS_DATA, "y_true": _Y_TRUE_COL, "y_pred": _Y_PRED_COL},
                ],
            }
        },
    )
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df, input_df = utils.get_df(self._session, data, _PD_SCHEMA)

            for sample_weight_col_name in params["sample_weight_col_name"]:
                actual_p = snowml_metrics.precision_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    sample_weight_col_name=sample_weight_col_name,
                    average="micro",
                )
                sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
                sklearn_p = sklearn_metrics.precision_score(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    sample_weight=sample_weight,
                    average="micro",
                )
                np.testing.assert_allclose(actual_p, sklearn_p)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"average": [None, "micro", "macro", "weighted"]}},
    )
    def test_average_multiclass(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTICLASS_DATA, _PD_SCHEMA)

        for average in params["average"]:
            actual_p = snowml_metrics.precision_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                average=average,
            )
            sklearn_p = sklearn_metrics.precision_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                average=average,
            )
            np.testing.assert_allclose(actual_p, sklearn_p)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "average": ["binary", "samples"],
                "y_true": [_Y_TRUE_COL, _Y_TRUE_COLS],
                "y_pred": [_Y_PRED_COL, _Y_PRED_COLS],
            }
        },
    )
    def test_average_binary(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _BINARY_DATA, _PD_SCHEMA)

        for idx, average in enumerate(params["average"]):
            y_true = params["y_true"][idx]
            y_pred = params["y_pred"][idx]
            actual_p = snowml_metrics.precision_score(
                df=input_df,
                y_true_col_names=y_true,
                y_pred_col_names=y_pred,
                average=average,
            )
            sklearn_p = sklearn_metrics.precision_score(
                pandas_df[y_true],
                pandas_df[y_pred],
                average=average,
            )
            np.testing.assert_allclose(actual_p, sklearn_p)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"zero_division": ["warn", 0, 1]}},
    )
    def test_zero_division(self, params: Dict[str, Any]) -> None:
        data = [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
        pandas_df, input_df = utils.get_df(self._session, data, _PD_SCHEMA)

        for zero_division in params["zero_division"]:
            if zero_division == "warn":
                continue

            actual_p = snowml_metrics.precision_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                zero_division=zero_division,
            )
            sklearn_p = sklearn_metrics.precision_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                zero_division=zero_division,
            )
            np.testing.assert_allclose(actual_p, sklearn_p)

        # warn
        sklearn_p = sklearn_metrics.precision_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            zero_division="warn",
        )

        with self.assertWarns(exceptions.UndefinedMetricWarning):
            actual_p = snowml_metrics.precision_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                zero_division="warn",
            )
            np.testing.assert_allclose(actual_p, sklearn_p)


if __name__ == "__main__":
    main()
