from typing import Any, Dict

import numpy as np
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml.modeling import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_DATA, _SF_SCHEMA = utils.gen_fuzz_data(
    rows=100,
    types=[utils.DataType.INTEGER] * 2 + [utils.DataType.FLOAT],
    low=-1,
    high=5,
)
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

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": [None, [2, 0, 4]]}},
    )
    def test_labels(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _DATA, _SF_SCHEMA)

        for labels in params["labels"]:
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

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"sample_weight_col_name": [None, _SAMPLE_WEIGHT_COL]}},
    )
    def test_sample_weight(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _DATA, _SF_SCHEMA)

        for sample_weight_col_name in params["sample_weight_col_name"]:
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

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"normalize": ["true", "pred", "all", None]}},
    )
    def test_normalize(self, params: Dict[str, Any]) -> None:
        pandas_df, input_df = utils.get_df(self._session, _DATA, _SF_SCHEMA)

        for normalize in params["normalize"]:
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

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"labels": []}},
        {"params": {"labels": [100, -10]}},
        {"params": {"normalize": "invalid"}},
    )
    def test_invalid_params(self, params: Dict[str, Any]) -> None:
        input_df = self._session.create_dataframe(_DATA, schema=_SF_SCHEMA)

        if "labels" in params:
            with self.assertRaises(ValueError):
                snowml_metrics.confusion_matrix(
                    df=input_df,
                    y_true_col_name=_Y_TRUE_COL,
                    y_pred_col_name=_Y_PRED_COL,
                    labels=params["labels"],
                )

        if "normalize" in params:
            with self.assertRaises(ValueError):
                snowml_metrics.confusion_matrix(
                    df=input_df,
                    y_true_col_name=_Y_TRUE_COL,
                    y_pred_col_name=_Y_PRED_COL,
                    normalize=params["normalize"],
                )


if __name__ == "__main__":
    main()
