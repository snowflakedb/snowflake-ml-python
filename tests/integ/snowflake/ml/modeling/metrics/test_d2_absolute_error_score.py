#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Any, Dict

import numpy as np
import pandas as pd
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

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
_MULTILABEL_DATA = [
    [1, 0, 1, 0.8, 0.3, 0.6],
    [0, 1, 0, 0.2, 0.7, 0.4],
    [1, 1, 0, 0.9, 0.6, 0.2],
    [0, 0, 1, 0.1, 0.4, 0.8],
]
_MULTILABEL_SCHEMA = ["Y_0", "Y_1", "Y_2", "S_0", "S_1", "S_2"]
_MULTILABEL_Y_TRUE_COLS = [_MULTILABEL_SCHEMA[0], _MULTILABEL_SCHEMA[1], _MULTILABEL_SCHEMA[2]]
_MULTILABEL_Y_PRED_COLS = [_MULTILABEL_SCHEMA[3], _MULTILABEL_SCHEMA[4], _MULTILABEL_SCHEMA[5]]


class D2AbsoluteErrorScoreTest(parameterized.TestCase):
    """Test d2_absolute_error_score."""

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
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            pandas_df = pd.DataFrame(data, columns=_SCHEMA)
            input_df = self._session.create_dataframe(pandas_df)

            for sample_weight_col_name in params["sample_weight_col_name"]:
                actual_loss = snowml_metrics.d2_absolute_error_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_pred_col_names=y_pred,
                    sample_weight_col_name=sample_weight_col_name,
                )
                sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
                sklearn_loss = sklearn_metrics.d2_absolute_error_score(
                    pandas_df[y_true],
                    pandas_df[y_pred],
                    sample_weight=sample_weight,
                )
                self.assertAlmostEqual(sklearn_loss, actual_loss)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"multioutput": ["raw_values", "uniform_average", [0.2, 1.0, 1.66]]}},
    )
    def test_multioutput(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTILABEL_DATA, columns=_MULTILABEL_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for multioutput in params["multioutput"]:
            actual_loss = snowml_metrics.d2_absolute_error_score(
                df=input_df,
                y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
                y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
                multioutput=multioutput,
            )
            sklearn_loss = sklearn_metrics.d2_absolute_error_score(
                pandas_df[_MULTILABEL_Y_TRUE_COLS],
                pandas_df[_MULTILABEL_Y_PRED_COLS],
                multioutput=multioutput,
            )
            np.testing.assert_allclose(actual_loss, sklearn_loss)

    def test_multilabel(self) -> None:
        pandas_df = pd.DataFrame(_MULTILABEL_DATA, columns=_MULTILABEL_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        actual_loss = snowml_metrics.d2_absolute_error_score(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_pred_col_names=_MULTILABEL_Y_PRED_COLS,
        )
        sklearn_loss = sklearn_metrics.d2_absolute_error_score(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_PRED_COLS],
        )
        self.assertAlmostEqual(sklearn_loss, actual_loss)


if __name__ == "__main__":
    main()
