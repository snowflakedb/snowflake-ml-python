#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Any, Dict

import numpy as np
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_BINARY_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=100,
    types=[utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT],
    low=0,
    high=2,
)
_MULTICLASS_DATA, _ = utils.gen_fuzz_data(
    rows=100,
    types=[utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT],
    low=0,
    high=5,
)
_Y_TRUE_COL = _SCHEMA[1]
_Y_PRED_COL = _SCHEMA[2]
_Y_TRUE_COLS = [_SCHEMA[1], _SCHEMA[2]]
_Y_PRED_COLS = [_SCHEMA[3], _SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SCHEMA[5]


class AccuracyScoreTest(parameterized.TestCase):
    """Test accuracy score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

        self._binary_input_df = self._session.create_dataframe(_BINARY_DATA, schema=_SCHEMA)
        self._binary_pandas_df = self._binary_input_df.to_pandas()
        self._multiclass_input_df = self._session.create_dataframe(_MULTICLASS_DATA, schema=_SCHEMA)
        self._multiclass_pandas_df = self._multiclass_input_df.to_pandas()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"y_true_col_names": [_Y_TRUE_COL, _Y_TRUE_COLS], "y_pred_col_names": [_Y_PRED_COL, _Y_PRED_COLS]}},
    )
    def test_accuracy_score(self, params: Dict[str, Any]) -> None:
        for i in range(len(params["y_true_col_names"])):
            y_true_col_names = params["y_true_col_names"][i]
            y_pred_col_names = params["y_pred_col_names"][i]
            input_df = self._multiclass_input_df if isinstance(y_true_col_names, str) else self._binary_input_df
            pandas_df = self._multiclass_pandas_df if isinstance(y_true_col_names, str) else self._binary_pandas_df

            score = snowml_metrics.accuracy_score(
                df=input_df, y_true_col_names=y_true_col_names, y_pred_col_names=y_pred_col_names, normalize=False
            )
            score_sklearn = sklearn_metrics.accuracy_score(
                pandas_df[y_true_col_names], pandas_df[y_pred_col_names], normalize=False
            )
            np.testing.assert_allclose(score, score_sklearn)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"y_true_col_names": [_Y_TRUE_COL, _Y_TRUE_COLS], "y_pred_col_names": [_Y_PRED_COL, _Y_PRED_COLS]}},
    )
    def test_accuracy_score_sample_weight(self, params: Dict[str, Any]) -> None:
        for i in range(len(params["y_true_col_names"])):
            y_true_col_names = params["y_true_col_names"][i]
            y_pred_col_names = params["y_pred_col_names"][i]
            input_df = self._multiclass_input_df if isinstance(y_true_col_names, str) else self._binary_input_df
            pandas_df = self._multiclass_pandas_df if isinstance(y_true_col_names, str) else self._binary_pandas_df

            score = snowml_metrics.accuracy_score(
                df=input_df,
                y_true_col_names=y_true_col_names,
                y_pred_col_names=y_pred_col_names,
                sample_weight_col_name=_SAMPLE_WEIGHT_COL,
                normalize=False,
            )
            score_sklearn = sklearn_metrics.accuracy_score(
                pandas_df[y_true_col_names],
                pandas_df[y_pred_col_names],
                sample_weight=pandas_df[_SAMPLE_WEIGHT_COL].to_numpy(),
                normalize=False,
            )
            np.testing.assert_allclose(score, score_sklearn)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"y_true_col_names": [_Y_TRUE_COL, _Y_TRUE_COLS], "y_pred_col_names": [_Y_PRED_COL, _Y_PRED_COLS]}},
    )
    def test_accuracy_score_normalized(self, params: Dict[str, Any]) -> None:
        for i in range(len(params["y_true_col_names"])):
            y_true_col_names = params["y_true_col_names"][i]
            y_pred_col_names = params["y_pred_col_names"][i]
            input_df = self._multiclass_input_df if isinstance(y_true_col_names, str) else self._binary_input_df
            pandas_df = self._multiclass_pandas_df if isinstance(y_true_col_names, str) else self._binary_pandas_df

            score = snowml_metrics.accuracy_score(
                df=input_df, y_true_col_names=y_true_col_names, y_pred_col_names=y_pred_col_names, normalize=True
            )
            score_sklearn = sklearn_metrics.accuracy_score(
                pandas_df[y_true_col_names], pandas_df[y_pred_col_names], normalize=True
            )
            np.testing.assert_allclose(score, score_sklearn)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"y_true_col_names": [_Y_TRUE_COL, _Y_TRUE_COLS], "y_pred_col_names": [_Y_PRED_COL, _Y_PRED_COLS]}},
    )
    def test_accuracy_score_sample_weight_normalized(self, params: Dict[str, Any]) -> None:
        for i in range(len(params["y_true_col_names"])):
            y_true_col_names = params["y_true_col_names"][i]
            y_pred_col_names = params["y_pred_col_names"][i]
            input_df = self._multiclass_input_df if isinstance(y_true_col_names, str) else self._binary_input_df
            pandas_df = self._multiclass_pandas_df if isinstance(y_true_col_names, str) else self._binary_pandas_df

            score = snowml_metrics.accuracy_score(
                df=input_df,
                y_true_col_names=y_true_col_names,
                y_pred_col_names=y_pred_col_names,
                sample_weight_col_name=_SAMPLE_WEIGHT_COL,
                normalize=True,
            )
            score_sklearn = sklearn_metrics.accuracy_score(
                pandas_df[y_true_col_names],
                pandas_df[y_pred_col_names],
                sample_weight=pandas_df[_SAMPLE_WEIGHT_COL].to_numpy(),
                normalize=True,
            )
            np.testing.assert_allclose(score, score_sklearn)


if __name__ == "__main__":
    main()
