#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Any, Dict

import numpy as np
import pandas as pd
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import exceptions, metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_ROWS = 100
_TYPES = [utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT]
_BINARY_DATA, _SCHEMA = utils.gen_fuzz_data(
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
_Y_TRUE_COL = _SCHEMA[1]
_Y_PRED_COL = _SCHEMA[2]
_Y_TRUE_COLS = [_SCHEMA[1], _SCHEMA[2]]
_Y_PRED_COLS = [_SCHEMA[3], _SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SCHEMA[5]


class FbetaScoreTest(parameterized.TestCase):
    """Test F-beta score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "beta": [1.0, 0.5],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _Y_TRUE_COLS, "y_pred": _Y_PRED_COLS},
                    {"data": _MULTICLASS_DATA, "y_true": _Y_TRUE_COL, "y_pred": _Y_PRED_COL},
                ],
            }
        },
    )
    def test_beta(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df = pd.DataFrame(data, columns=_SCHEMA)
            input_df = self._session.create_dataframe(pandas_df)

            for beta in params["beta"]:
                actual_f = snowml_metrics.fbeta_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    beta=beta,
                    average=None,
                )
                sklearn_f = sklearn_metrics.fbeta_score(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    beta=beta,
                    average=None,
                )
                np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": [None, [2, 0, 4]]}},
    )
    def test_labels(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for labels in params["labels"]:
            actual_f = snowml_metrics.fbeta_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                beta=0.5,
                average=None,
                labels=labels,
            )
            sklearn_f = sklearn_metrics.fbeta_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                beta=0.5,
                average=None,
                labels=labels,
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"pos_label": [0, 2, 4]}},
    )
    def test_pos_label(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for pos_label in params["pos_label"]:
            actual_f = snowml_metrics.fbeta_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                beta=0.5,
                average=None,
                pos_label=pos_label,
            )
            sklearn_f = sklearn_metrics.fbeta_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                beta=0.5,
                average=None,
                pos_label=pos_label,
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"average": [None, "micro", "macro", "weighted"]}},
    )
    def test_average_multiclass(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for average in params["average"]:
            actual_f = snowml_metrics.fbeta_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                beta=0.5,
                average=average,
            )
            sklearn_f = sklearn_metrics.fbeta_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                beta=0.5,
                average=average,
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

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
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for idx, average in enumerate(params["average"]):
            y_true = params["y_true"][idx]
            y_pred = params["y_pred"][idx]
            actual_f = snowml_metrics.fbeta_score(
                df=input_df,
                y_true_col_names=y_true,
                y_pred_col_names=y_pred,
                beta=0.5,
                average=average,
            )
            sklearn_f = sklearn_metrics.fbeta_score(
                pandas_df[y_true],
                pandas_df[y_pred],
                beta=0.5,
                average=average,
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

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
            pandas_df = pd.DataFrame(data, columns=_SCHEMA)
            input_df = self._session.create_dataframe(pandas_df)

            for sample_weight_col_name in params["sample_weight_col_name"]:
                actual_f = snowml_metrics.fbeta_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    beta=0.5,
                    average=None,
                    sample_weight_col_name=sample_weight_col_name,
                )
                sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
                sklearn_f = sklearn_metrics.fbeta_score(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    beta=0.5,
                    average=None,
                    sample_weight=sample_weight,
                )
                np.testing.assert_allclose(actual_f, sklearn_f)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"zero_division": [0, 1]}},
    )
    def test_zero_division(self, params: Dict[str, Any]) -> None:
        data = [
            [0, 0, 0, 0, 0, 0],
        ]
        pandas_df = pd.DataFrame(data, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for zero_division in params["zero_division"]:
            if zero_division == "warn":
                continue

            actual_f = snowml_metrics.fbeta_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                beta=0.5,
                zero_division=zero_division,
            )
            sklearn_f = sklearn_metrics.fbeta_score(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                beta=0.5,
                zero_division=zero_division,
            )
            np.testing.assert_allclose(actual_f, sklearn_f)

        # warn
        sklearn_f = sklearn_metrics.fbeta_score(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            beta=0.5,
            zero_division="warn",
        )

        with self.assertWarns(exceptions.UndefinedMetricWarning):
            actual_f = snowml_metrics.fbeta_score(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                beta=0.5,
                zero_division="warn",
            )
            np.testing.assert_allclose(actual_f, sklearn_f)


if __name__ == "__main__":
    main()
