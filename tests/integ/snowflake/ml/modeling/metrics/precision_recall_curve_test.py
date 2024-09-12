from typing import Optional, Union
from unittest import mock

import numpy as np
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils
from tests.integ.snowflake.ml.modeling.metrics import generator

_TYPES = [utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT]
_BINARY_LOW, _BINARY_HIGH = 0, 2
_BINARY_DATA_LIST, _SF_SCHEMA = generator.gen_test_cases(_TYPES, _BINARY_LOW, _BINARY_HIGH)
_REGULAR_BINARY_DATA_LIST, _LARGE_BINARY_DATA = _BINARY_DATA_LIST[:-1], _BINARY_DATA_LIST[-1]
_Y_TRUE_COL = _SF_SCHEMA[1]
_PROBAS_PRED_COL = _SF_SCHEMA[2]
_SAMPLE_WEIGHT_COL = _SF_SCHEMA[5]


class PrecisionRecallCurveTest(parameterized.TestCase):
    """Test precision-recall curve."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_BINARY_DATA_LIST))),
        pos_label=[0, 2, 4],
    )
    def test_pos_label(self, data_index: int, pos_label: Union[str, int]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _BINARY_DATA_LIST[data_index], _SF_SCHEMA)

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

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
        sample_weight_col_name=[None, _SAMPLE_WEIGHT_COL],
    )
    def test_sample_weight(self, data_index: int, sample_weight_col_name: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

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

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_BINARY_DATA_LIST))),
    )
    @mock.patch("snowflake.ml.modeling.metrics.ranking.result._RESULT_SIZE_THRESHOLD", 0)
    def test_metric_size_threshold(self, data_index: int) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_BINARY_DATA_LIST[data_index], _SF_SCHEMA)

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
