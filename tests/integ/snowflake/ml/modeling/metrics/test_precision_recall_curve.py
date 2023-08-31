#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Any, Dict
from unittest import mock

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
_TYPES = [utils.DataType.INTEGER] + [utils.DataType.FLOAT] * 2
_BINARY_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=[2, 1, 1],
)
_Y_TRUE_COL = _SCHEMA[1]
_PROBAS_PRED_COL = _SCHEMA[2]
_SAMPLE_WEIGHT_COL = _SCHEMA[3]


class PrecisionRecallCurveTest(parameterized.TestCase):
    """Test precision-recall curve."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"pos_label": [0, 2, 4]}},
    )
    def test_pos_label(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for pos_label in params["pos_label"]:
            actual_precision, actual_recall, actual_thresholds = snowml_metrics.precision_recall_curve(
                df=input_df,
                y_true_col_name=_Y_TRUE_COL,
                probas_pred_col_name=_PROBAS_PRED_COL,
                pos_label=pos_label,
            )
            sklearn_precision, sklearn_recall, sklearn_thresholds = sklearn_metrics.precision_recall_curve(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_PROBAS_PRED_COL],
                pos_label=pos_label,
            )
            np.testing.assert_allclose(actual_precision, sklearn_precision)
            np.testing.assert_allclose(actual_recall, sklearn_recall)
            np.testing.assert_allclose(actual_thresholds, sklearn_thresholds)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL]}},
    )
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for sample_weight_col_name in params["sample_weight_col_name"]:
            actual_precision, actual_recall, actual_thresholds = snowml_metrics.precision_recall_curve(
                df=input_df,
                y_true_col_name=_Y_TRUE_COL,
                probas_pred_col_name=_PROBAS_PRED_COL,
                sample_weight_col_name=sample_weight_col_name,
            )
            sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
            sklearn_precision, sklearn_recall, sklearn_thresholds = sklearn_metrics.precision_recall_curve(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_PROBAS_PRED_COL],
                sample_weight=sample_weight,
            )
            np.testing.assert_allclose(actual_precision, sklearn_precision)
            np.testing.assert_allclose(actual_recall, sklearn_recall)
            np.testing.assert_allclose(actual_thresholds, sklearn_thresholds)

    @mock.patch("snowflake.ml.modeling.metrics.ranking.result._RESULT_SIZE_THRESHOLD", 0)
    def test_metric_size_threshold(self) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        actual_precision, actual_recall, actual_thresholds = snowml_metrics.precision_recall_curve(
            df=input_df,
            y_true_col_name=_Y_TRUE_COL,
            probas_pred_col_name=_PROBAS_PRED_COL,
        )
        sklearn_precision, sklearn_recall, sklearn_thresholds = sklearn_metrics.precision_recall_curve(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_PROBAS_PRED_COL],
        )
        np.testing.assert_allclose(actual_precision, sklearn_precision)
        np.testing.assert_allclose(actual_recall, sklearn_recall)
        np.testing.assert_allclose(actual_thresholds, sklearn_thresholds)


if __name__ == "__main__":
    main()
