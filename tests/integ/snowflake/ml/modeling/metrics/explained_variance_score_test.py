from typing import Any
from unittest import mock

import numpy as np
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_ROWS = 100
_TYPES = [utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT]
_BINARY_DATA, _SF_SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=2,
)
_MULTICLASS_DATA, _ = utils.gen_fuzz_data(
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
_MULTILABEL_DATA = [
    [1, 0, 1, 0.8, 0.3, 0.6],
    [0, 1, 0, 0.2, 0.7, 0.4],
    [1, 1, 0, 0.9, 0.6, 0.2],
    [0, 0, 1, 0.1, 0.4, 0.8],
]
_MULTILABEL_SCHEMA = ["Y_0", "Y_1", "Y_2", "S_0", "S_1", "S_2"]
_MULTILABEL_Y_TRUE_COLS = [_MULTILABEL_SCHEMA[0], _MULTILABEL_SCHEMA[1], _MULTILABEL_SCHEMA[2]]
_MULTILABEL_Y_PRED_COLS = [_MULTILABEL_SCHEMA[3], _MULTILABEL_SCHEMA[4], _MULTILABEL_SCHEMA[5]]


class ExplainedVarianceScoreTest(parameterized.TestCase):
    """Test explained variance regression score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

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
    def test_sample_weight(self, params: dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

            for sample_weight_col_name in params["sample_weight_col_name"]:
                actual_loss = snowml_metrics.explained_variance_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    sample_weight_col_name=sample_weight_col_name,
                )
                sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
                sklearn_loss = sklearn_metrics.explained_variance_score(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    sample_weight=sample_weight,
                )
                self.assertAlmostEqual(sklearn_loss, actual_loss)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"multioutput": ["raw_values", "uniform_average", [0.2, 1.0, 1.66]]}},
    )
    def test_multioutput(self, params: dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTILABEL_DATA, _MULTILABEL_SCHEMA)

        for multioutput in params["multioutput"]:
            actual_loss = snowml_metrics.explained_variance_score(
                df=input_df,
                y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
                y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
                multioutput=multioutput,
            )
            sklearn_loss = sklearn_metrics.explained_variance_score(
                pandas_df[_MULTILABEL_Y_TRUE_COLS],
                pandas_df[_MULTILABEL_Y_PRED_COLS],
                multioutput=multioutput,
            )
            np.testing.assert_allclose(actual_loss, sklearn_loss)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "force_finite": [True, False],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _Y_TRUE_COLS, "y_pred": _Y_PRED_COLS},
                    {"data": _MULTICLASS_DATA, "y_true": _Y_TRUE_COL, "y_pred": _Y_PRED_COL},
                ],
            }
        },
    )
    def test_force_finite(self, params: dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

            for force_finite in params["force_finite"]:
                actual_loss = snowml_metrics.explained_variance_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    force_finite=force_finite,
                )
                sklearn_loss = sklearn_metrics.explained_variance_score(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    force_finite=force_finite,
                )
                self.assertAlmostEqual(sklearn_loss, actual_loss)

    def test_multilabel(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _MULTILABEL_DATA, _MULTILABEL_SCHEMA)

        actual_loss = snowml_metrics.explained_variance_score(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.explained_variance_score(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_PRED_COLS],
        )
        self.assertAlmostEqual(sklearn_loss, actual_loss)

    @mock.patch("snowflake.ml.modeling.metrics.regression.result._RESULT_SIZE_THRESHOLD", 0)
    def test_metric_size_threshold(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _BINARY_DATA, _SF_SCHEMA)

        actual_loss = snowml_metrics.explained_variance_score(
            df=input_df,
            y_true_col_names=_Y_TRUE_COLS,
            y_pred_col_names=_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.explained_variance_score(
            pandas_df[_Y_TRUE_COLS],
            pandas_df[_Y_PRED_COLS],
        )
        self.assertAlmostEqual(sklearn_loss, actual_loss)


if __name__ == "__main__":
    main()
