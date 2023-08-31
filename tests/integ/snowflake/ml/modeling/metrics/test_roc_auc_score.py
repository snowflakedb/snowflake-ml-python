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
_TYPES = [utils.DataType.INTEGER] + [utils.DataType.FLOAT] * 4
_BINARY_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=[2, 1, 1, 1, 1],
)
_BINARY_Y_TRUE_COL = _SCHEMA[1]
_BINARY_Y_SCORE_COL = _SCHEMA[2]
_MULTICLASS_DATA = [
    [0, 2, 0.29, 0.49, 0.22, 0.18],
    [1, 0, 0.33, 0.16, 0.51, 0.69],
    [2, 1, 0.54, 0.29, 0.17, 0.04],
    [3, 2, 0.27, 0.68, 0.05, 0.17],
    [4, 1, 0.82, 0.12, 0.06, 0.91],
    [5, 2, 0.08, 0.46, 0.46, 0.76],
]
_MULTICLASS_Y_TRUE_COL = _SCHEMA[1]
_MULTICLASS_Y_SCORE_COLS = [_SCHEMA[2], _SCHEMA[3], _SCHEMA[4]]
_SAMPLE_WEIGHT_COL = _SCHEMA[5]
_MULTILABEL_DATA = [
    [1, 0, 1, 0.8, 0.3, 0.6],
    [0, 1, 0, 0.2, 0.7, 0.4],
    [1, 1, 0, 0.9, 0.6, 0.2],
    [0, 0, 1, 0.1, 0.4, 0.8],
]
_MULTILABEL_SCHEMA = ["Y_0", "Y_1", "Y_2", "S_0", "S_1", "S_2"]
_MULTILABEL_Y_TRUE_COLS = [_MULTILABEL_SCHEMA[0], _MULTILABEL_SCHEMA[1], _MULTILABEL_SCHEMA[2]]
_MULTILABEL_Y_SCORE_COLS = [_MULTILABEL_SCHEMA[3], _MULTILABEL_SCHEMA[4], _MULTILABEL_SCHEMA[5]]


class RocAucScoreTest(parameterized.TestCase):
    """Test ROC AUC score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"average": [None, "micro", "macro", "samples", "weighted"]}},
    )
    def test_average_binary(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for average in params["average"]:
            actual_auc = snowml_metrics.roc_auc_score(
                df=input_df,
                y_true_col_names=_BINARY_Y_TRUE_COL,
                y_score_col_names=_BINARY_Y_SCORE_COL,
                average=average,
            )
            sklearn_auc = sklearn_metrics.roc_auc_score(
                pandas_df[_BINARY_Y_TRUE_COL],
                pandas_df[_BINARY_Y_SCORE_COL],
                average=average,
            )
            self.assertAlmostEqual(sklearn_auc, actual_auc)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "average": [None, "micro", "macro", "weighted"],
                "multi_class": ["ovr", "ovr", "ovo", "ovo"],
            }
        },
    )
    def test_average_multiclass(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for idx, average in enumerate(params["average"]):
            multi_class = params["multi_class"][idx]
            actual_auc = snowml_metrics.roc_auc_score(
                df=input_df,
                y_true_col_names=_MULTICLASS_Y_TRUE_COL,
                y_score_col_names=_MULTICLASS_Y_SCORE_COLS,
                average=average,
                multi_class=multi_class,
            )
            sklearn_auc = sklearn_metrics.roc_auc_score(
                pandas_df[_MULTICLASS_Y_TRUE_COL],
                pandas_df[_MULTICLASS_Y_SCORE_COLS],
                average=average,
                multi_class=multi_class,
            )
            np.testing.assert_allclose(actual_auc, sklearn_auc)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL],
                "values": [
                    {"data": _BINARY_DATA, "y_true": _BINARY_Y_TRUE_COL, "y_score": _BINARY_Y_SCORE_COL},
                    {"data": _MULTICLASS_DATA, "y_true": _MULTICLASS_Y_TRUE_COL, "y_score": _MULTICLASS_Y_SCORE_COLS},
                ],
            }
        },
    )
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        for values in params["values"]:
            data = values["data"]
            y_true = values["y_true"]
            y_score = values["y_score"]
            pandas_df = pd.DataFrame(data, columns=_SCHEMA)
            input_df = self._session.create_dataframe(pandas_df)

            for sample_weight_col_name in params["sample_weight_col_name"]:
                actual_auc = snowml_metrics.roc_auc_score(
                    df=input_df,
                    y_true_col_names=y_true,
                    y_score_col_names=y_score,
                    sample_weight_col_name=sample_weight_col_name,
                    multi_class="ovr",
                )
                sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
                sklearn_auc = sklearn_metrics.roc_auc_score(
                    pandas_df[y_true],
                    pandas_df[y_score],
                    sample_weight=sample_weight,
                    multi_class="ovr",
                )
                self.assertAlmostEqual(sklearn_auc, actual_auc)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"max_fpr": [None, 0.1, 0.5, 1]}},
    )
    def test_max_fpr(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for max_fpr in params["max_fpr"]:
            actual_auc = snowml_metrics.roc_auc_score(
                df=input_df,
                y_true_col_names=_BINARY_Y_TRUE_COL,
                y_score_col_names=_BINARY_Y_SCORE_COL,
                max_fpr=max_fpr,
            )
            sklearn_auc = sklearn_metrics.roc_auc_score(
                pandas_df[_BINARY_Y_TRUE_COL],
                pandas_df[_BINARY_Y_SCORE_COL],
                max_fpr=max_fpr,
            )
            self.assertAlmostEqual(sklearn_auc, actual_auc)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"multi_class": ["ovr", "ovo"]}},
    )
    def test_multi_class(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for multi_class in params["multi_class"]:
            actual_auc = snowml_metrics.roc_auc_score(
                df=input_df,
                y_true_col_names=_MULTICLASS_Y_TRUE_COL,
                y_score_col_names=_MULTICLASS_Y_SCORE_COLS,
                multi_class=multi_class,
            )
            sklearn_auc = sklearn_metrics.roc_auc_score(
                pandas_df[_MULTICLASS_Y_TRUE_COL],
                pandas_df[_MULTICLASS_Y_SCORE_COLS],
                multi_class=multi_class,
            )
            self.assertAlmostEqual(sklearn_auc, actual_auc)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": [None, [0, 1, 2]]}},
    )
    def test_labels(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for labels in params["labels"]:
            actual_auc = snowml_metrics.roc_auc_score(
                df=input_df,
                y_true_col_names=_MULTICLASS_Y_TRUE_COL,
                y_score_col_names=_MULTICLASS_Y_SCORE_COLS,
                labels=labels,
                multi_class="ovr",
            )
            sklearn_auc = sklearn_metrics.roc_auc_score(
                pandas_df[_MULTICLASS_Y_TRUE_COL],
                pandas_df[_MULTICLASS_Y_SCORE_COLS],
                labels=labels,
                multi_class="ovr",
            )
            self.assertAlmostEqual(sklearn_auc, actual_auc)

    def test_multilabel(self) -> None:
        pandas_df = pd.DataFrame(_MULTILABEL_DATA, columns=_MULTILABEL_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        actual_auc = snowml_metrics.roc_auc_score(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_score_col_names=_MULTILABEL_Y_SCORE_COLS,
        )
        sklearn_auc = sklearn_metrics.roc_auc_score(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_SCORE_COLS],
        )
        self.assertAlmostEqual(sklearn_auc, actual_auc)

    @mock.patch("snowflake.ml.modeling.metrics.ranking.result._RESULT_SIZE_THRESHOLD", 0)
    def test_metric_size_threshold(self) -> None:
        pandas_df = pd.DataFrame(_MULTILABEL_DATA, columns=_MULTILABEL_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        actual_auc = snowml_metrics.roc_auc_score(
            df=input_df,
            y_true_col_names=_MULTILABEL_Y_TRUE_COLS,
            y_score_col_names=_MULTILABEL_Y_SCORE_COLS,
        )
        sklearn_auc = sklearn_metrics.roc_auc_score(
            pandas_df[_MULTILABEL_Y_TRUE_COLS],
            pandas_df[_MULTILABEL_Y_SCORE_COLS],
        )
        self.assertAlmostEqual(sklearn_auc, actual_auc)


if __name__ == "__main__":
    main()
