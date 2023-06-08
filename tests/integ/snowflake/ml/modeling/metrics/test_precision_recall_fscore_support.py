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
_DATA, _SCHEMA = utils.gen_fuzz_data(
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


class PrecisionRecallFscoreSupportTest(parameterized.TestCase):
    """Test precision_recall_fscore_support."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": [None, [2, 0, 4]]}},
    )
    def test_precision_recall_fscore_support_labels(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for labels in params["labels"]:
            actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                labels=labels,
            )
            sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                labels=labels,
            )
            np.testing.assert_allclose(
                np.array((actual_p, actual_r, actual_f, actual_s)),
                np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s)),
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL]}},
    )
    def test_precision_recall_fscore_support_sample_weight(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for sample_weight_col_name in params["sample_weight_col_name"]:
            actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                sample_weight_col_name=sample_weight_col_name,
            )
            sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
            sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                sample_weight=sample_weight,
            )
            np.testing.assert_allclose(
                np.array((actual_p, actual_r, actual_f, actual_s)),
                np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s)),
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"average": [None, "binary", "micro", "macro", "samples", "weighted"]}},
    )
    def test_precision_recall_fscore_support_average(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for average in params["average"]:
            if average == "binary" or average == "samples":
                continue

            actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                average=average,
            )
            sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                average=average,
            )
            np.testing.assert_allclose(
                np.array((actual_p, actual_r, actual_f, actual_s), dtype=np.float_),
                np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s), dtype=np.float_),
            )

        data, _ = utils.gen_fuzz_data(
            rows=_ROWS,
            types=_TYPES,
            low=0,
            high=2,
        )
        binary_pandas_df = pd.DataFrame(data, columns=_SCHEMA)
        binary_input_df = self._session.create_dataframe(binary_pandas_df)

        # binary
        actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
            df=binary_input_df,
            y_true_col_names=_Y_TRUE_COL,
            y_pred_col_names=_Y_PRED_COL,
            average="binary",
        )
        sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
            binary_pandas_df[_Y_TRUE_COL],
            binary_pandas_df[_Y_PRED_COL],
            average="binary",
        )
        np.testing.assert_allclose(
            np.array((actual_p, actual_r, actual_f, actual_s), dtype=np.float_),
            np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s), dtype=np.float_),
        )

        # samples
        actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
            df=binary_input_df,
            y_true_col_names=_Y_TRUE_COLS,
            y_pred_col_names=_Y_PRED_COLS,
            average="samples",
        )
        sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
            binary_pandas_df[_Y_TRUE_COLS],
            binary_pandas_df[_Y_PRED_COLS],
            average="samples",
        )
        np.testing.assert_allclose(
            np.array((actual_p, actual_r, actual_f, actual_s), dtype=np.float_),
            np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s), dtype=np.float_),
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"zero_division": ["warn", 0, 1]}},
    )
    def test_precision_recall_fscore_support_zero_division(self, params: Dict[str, Any]) -> None:
        data = [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
        pandas_df = pd.DataFrame(data, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for zero_division in params["zero_division"]:
            if zero_division == "warn":
                continue

            actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                zero_division=zero_division,
            )
            sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_PRED_COL],
                zero_division=zero_division,
            )
            np.testing.assert_allclose(
                np.array((actual_p, actual_r, actual_f, actual_s)),
                np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s)),
            )

        # warn
        sklearn_p, sklearn_r, sklearn_f, sklearn_s = sklearn_metrics.precision_recall_fscore_support(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            zero_division="warn",
        )

        with self.assertWarns(exceptions.UndefinedMetricWarning):
            actual_p, actual_r, actual_f, actual_s = snowml_metrics.precision_recall_fscore_support(
                df=input_df,
                y_true_col_names=_Y_TRUE_COL,
                y_pred_col_names=_Y_PRED_COL,
                zero_division="warn",
            )
            np.testing.assert_allclose(
                np.array((actual_p, actual_r, actual_f, actual_s)),
                np.array((sklearn_p, sklearn_r, sklearn_f, sklearn_s)),
            )


if __name__ == "__main__":
    main()
