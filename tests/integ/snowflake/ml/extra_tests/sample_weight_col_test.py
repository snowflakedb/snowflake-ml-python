import random
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest
from importlib_resources import files
from sklearn.model_selection import GridSearchCV as SkGridSearchCV
from xgboost import XGBClassifier as XGB_XGBClassifier

from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

numerical_columns = [
    "CONS_CONF_IDX",
    "CONS_PRICE_IDX",
    "DURATION",
    "EMP_VAR_RATE",
    "EURIBOR3M",
    "NR_EMPLOYED",
    "PDAYS",
    "PREVIOUS",
]
label_column = ["LABEL"]
feature_cols = numerical_columns


class XGBSampleWeightTest(absltest.TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        data_file = files("tests.integ.snowflake.ml.test_data").joinpath("UCI_BANK_MARKETING_20COLUMNS.csv")
        self._test_data = pd.read_csv(data_file, index_col=0)

    def tearDown(self):
        self._session.close()

    def test_fit_and_compare_results(self) -> None:
        pd_data = self._test_data
        pd_data["ROW_INDEX"] = pd_data.reset_index().index
        sample_weight_col = "SAMPLE_WEIGHT"
        pd_data[sample_weight_col] = np.array([random.randint(0, 100) for _ in range(pd_data.shape[0])])

        snowml_classifier = XGBClassifier(
            input_cols=feature_cols,
            label_cols=label_column,
            passthrough_cols="ROW_INDEX",
            sample_weight_col=sample_weight_col,
        )
        xgb_classifier = XGB_XGBClassifier()

        xgb_classifier.fit(pd_data[feature_cols], pd_data[label_column], sample_weight=pd_data[sample_weight_col])
        predictions = xgb_classifier.predict(pd_data[feature_cols])

        raw_data = self._session.create_dataframe(pd_data)
        snowml_classifier.fit(raw_data)
        snowml_predictions = (
            snowml_classifier.predict(raw_data).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()
        )

        np.testing.assert_allclose(predictions.flatten(), snowml_predictions.flatten(), rtol=1.0e-3, atol=1.0e-3)

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_grid_search_on_xgboost_sample_weight(self, is_single_node_mock: mock.Mock) -> None:
        for v in [True, False]:
            is_single_node_mock.return_value = v

            pd_data = self._test_data
            pd_data["ROW_INDEX"] = pd_data.reset_index().index
            sample_weight_col = "SAMPLE_WEIGHT"
            pd_data[sample_weight_col] = np.array([random.randint(0, 100) for _ in range(pd_data.shape[0])])

            snowml_classifier = XGBClassifier(
                input_cols=feature_cols,
                label_cols=label_column,
                passthrough_cols="ROW_INDEX",
            )
            xgb_classifier = XGB_XGBClassifier()

            param_grid = {
                "max_depth": [80, 100],
            }

            grid_search = GridSearchCV(
                param_grid=param_grid,
                estimator=snowml_classifier,
                input_cols=feature_cols,
                label_cols=label_column,
                passthrough_cols="ROW_INDEX",
                sample_weight_col=sample_weight_col,
            )
            sk_grid_search = SkGridSearchCV(param_grid=param_grid, estimator=xgb_classifier)

            sk_grid_search.fit(pd_data[feature_cols], pd_data[label_column], sample_weight=pd_data[sample_weight_col])
            predictions = sk_grid_search.predict(pd_data[feature_cols])

            raw_data = self._session.create_dataframe(pd_data)
            grid_search.fit(raw_data)
            snowml_predictions = (
                grid_search.predict(raw_data).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()
            )

            np.testing.assert_allclose(predictions.flatten(), snowml_predictions.flatten(), rtol=1.0e-3, atol=1.0e-3)


if __name__ == "__main__":
    absltest.main()
