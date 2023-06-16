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
_TYPES = [utils.DataType.INTEGER] + [utils.DataType.FLOAT] * 2
_BINARY_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=[2, 1, 1],
)
_MULTICLASS_DATA, _ = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=[5, 1, 1],
)
_Y_TRUE_COL = _SCHEMA[1]
_Y_SCORE_COL = _SCHEMA[2]
_SAMPLE_WEIGHT_COL = _SCHEMA[3]


class RocCurveTest(parameterized.TestCase):
    """Test ROC."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"pos_label": [0, 2, 4]}},
    )
    def test_pos_label(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_MULTICLASS_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for pos_label in params["pos_label"]:
            actual_fpr, actual_tpr, actual_thresholds = snowml_metrics.roc_curve(
                df=input_df,
                y_true_col_name=_Y_TRUE_COL,
                y_score_col_name=_Y_SCORE_COL,
                pos_label=pos_label,
            )
            sklearn_fpr, sklearn_tpr, sklearn_thresholds = sklearn_metrics.roc_curve(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_SCORE_COL],
                pos_label=pos_label,
            )
            np.testing.assert_allclose(
                np.array((actual_fpr, actual_tpr, actual_thresholds)),
                np.array((sklearn_fpr, sklearn_tpr, sklearn_thresholds)),
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL]}},
    )
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for sample_weight_col_name in params["sample_weight_col_name"]:
            actual_fpr, actual_tpr, actual_thresholds = snowml_metrics.roc_curve(
                df=input_df,
                y_true_col_name=_Y_TRUE_COL,
                y_score_col_name=_Y_SCORE_COL,
                sample_weight_col_name=sample_weight_col_name,
            )
            sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
            sklearn_fpr, sklearn_tpr, sklearn_thresholds = sklearn_metrics.roc_curve(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_SCORE_COL],
                sample_weight=sample_weight,
            )
            np.testing.assert_allclose(
                np.array((actual_fpr, actual_tpr, actual_thresholds)),
                np.array((sklearn_fpr, sklearn_tpr, sklearn_thresholds)),
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"drop_intermediate": [True, False]}},
    )
    def test_drop_intermediate(self, params: Dict[str, Any]) -> None:
        pandas_df = pd.DataFrame(_BINARY_DATA, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        for drop_intermediate in params["drop_intermediate"]:
            actual_fpr, actual_tpr, actual_thresholds = snowml_metrics.roc_curve(
                df=input_df,
                y_true_col_name=_Y_TRUE_COL,
                y_score_col_name=_Y_SCORE_COL,
                drop_intermediate=drop_intermediate,
            )
            sklearn_fpr, sklearn_tpr, sklearn_thresholds = sklearn_metrics.roc_curve(
                pandas_df[_Y_TRUE_COL],
                pandas_df[_Y_SCORE_COL],
                drop_intermediate=drop_intermediate,
            )
            np.testing.assert_allclose(
                np.array((actual_fpr, actual_tpr, actual_thresholds)),
                np.array((sklearn_fpr, sklearn_tpr, sklearn_thresholds)),
            )


if __name__ == "__main__":
    main()
