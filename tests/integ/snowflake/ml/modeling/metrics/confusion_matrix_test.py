from typing import Optional

import numpy as np
import numpy.typing as npt
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils
from tests.integ.snowflake.ml.modeling.metrics import generator

_TYPES = [utils.DataType.INTEGER] * 2 + [utils.DataType.FLOAT]
_LOW, _HIGH = 1, 5
_DATA_LIST, _SF_SCHEMA = generator.gen_test_cases(_TYPES, _LOW, _HIGH)
_REGULAR_DATA_LIST, _LARGE_DATA = _DATA_LIST[:-1], _DATA_LIST[-1]
_Y_TRUE_COL = _SF_SCHEMA[1]
_Y_PRED_COL = _SF_SCHEMA[2]
_SAMPLE_WEIGHT_COL = _SF_SCHEMA[3]


class ConfusionMatrixTest(parameterized.TestCase):
    """Test confusion matrix."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_DATA_LIST))), labels=[None, [2, 0, 4]]
    )
    def test_labels(self, data_index: int, labels: Optional[npt.ArrayLike]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_DATA_LIST[data_index], _SF_SCHEMA)

        actual_cm = snowml_metrics.confusion_matrix(
            df=input_df,
            y_true_col_name=_Y_TRUE_COL,
            y_pred_col_name=_Y_PRED_COL,
            labels=labels,
        )
        sklearn_cm = sklearn_metrics.confusion_matrix(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            labels=labels,
        )
        np.testing.assert_allclose(actual_cm, sklearn_cm)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_DATA_LIST))), sample_weight_col_name=[None, _SAMPLE_WEIGHT_COL]
    )
    def test_sample_weight(self, data_index: int, sample_weight_col_name: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_DATA_LIST[data_index], _SF_SCHEMA)

        actual_cm = snowml_metrics.confusion_matrix(
            df=input_df,
            y_true_col_name=_Y_TRUE_COL,
            y_pred_col_name=_Y_PRED_COL,
            sample_weight_col_name=sample_weight_col_name,
        )
        sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
        sklearn_cm = sklearn_metrics.confusion_matrix(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            sample_weight=sample_weight,
        )
        np.testing.assert_allclose(actual_cm, sklearn_cm)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_DATA_LIST))), normalize=["true", "pred", "all", None]
    )
    def test_normalize(self, data_index: int, normalize: Optional[str]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _REGULAR_DATA_LIST[data_index], _SF_SCHEMA)

        actual_cm = snowml_metrics.confusion_matrix(
            df=input_df,
            y_true_col_name=_Y_TRUE_COL,
            y_pred_col_name=_Y_PRED_COL,
            normalize=normalize,
        )
        sklearn_cm = sklearn_metrics.confusion_matrix(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
            normalize=normalize,
        )
        np.testing.assert_allclose(actual_cm, sklearn_cm)

    @parameterized.product(  # type: ignore[misc]
        data_index=list(range(len(_REGULAR_DATA_LIST))), labels=[None, [], [100, -10]], normalize=[None, "invalid"]
    )
    def test_invalid_params(self, data_index: int, labels: Optional[npt.ArrayLike], normalize: Optional[str]) -> None:
        input_df = self._session.create_dataframe(_REGULAR_DATA_LIST[data_index], schema=_SF_SCHEMA)

        if labels is not None or normalize is not None:
            with self.assertRaises(ValueError):
                snowml_metrics.confusion_matrix(
                    df=input_df,
                    y_true_col_name=_Y_TRUE_COL,
                    y_pred_col_name=_Y_PRED_COL,
                    labels=labels,
                    normalize=normalize,
                )

    def test_with_large_num_of_rows(self) -> None:
        pandas_df, input_df = utils.get_df(self._session, _LARGE_DATA, _SF_SCHEMA)

        actual_cm = snowml_metrics.confusion_matrix(
            df=input_df,
            y_true_col_name=_Y_TRUE_COL,
            y_pred_col_name=_Y_PRED_COL,
        )
        sklearn_cm = sklearn_metrics.confusion_matrix(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
        )
        np.testing.assert_allclose(actual_cm, sklearn_cm)

    def test_with_divisible_num_of_rows(self) -> None:
        data, _ = utils.gen_fuzz_data(
            rows=metrics_utils.BATCH_SIZE * 4,
            types=_TYPES,
            low=-_LOW,
            high=_HIGH,
        )
        pandas_df, input_df = utils.get_df(self._session, data, _SF_SCHEMA)

        actual_cm = snowml_metrics.confusion_matrix(
            df=input_df,
            y_true_col_name=_Y_TRUE_COL,
            y_pred_col_name=_Y_PRED_COL,
        )
        sklearn_cm = sklearn_metrics.confusion_matrix(
            pandas_df[_Y_TRUE_COL],
            pandas_df[_Y_PRED_COL],
        )
        np.testing.assert_allclose(actual_cm, sklearn_cm)


if __name__ == "__main__":
    main()
