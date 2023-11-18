import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from importlib_resources import files
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import GridSearchCV as SkGridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    MinMaxScaler as SkMinMaxScaler,
    OneHotEncoder as SkOneHotEncoder,
)
from snowflake.ml.modeling.linear_model.logistic_regression import LogisticRegression

from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.model_selection._internal import GridSearchCV
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import MinMaxScaler, OneHotEncoder
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

categorical_columns = [
    "AGE",
    "CAMPAIGN",
    "CONTACT",
    "DAY_OF_WEEK",
    "EDUCATION",
    "HOUSING",
    "JOB",
    "LOAN",
    "MARITAL",
    "MONTH",
    "POUTCOME",
]
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
feature_cols = categorical_columns + numerical_columns


class GridSearchCVTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_fit_and_compare_results(self) -> None:
        data_file = files("tests.integ.snowflake.ml.test_data").joinpath("UCI_BANK_MARKETING_20COLUMNS.csv")
        pd_data = pd.read_csv(data_file, index_col=0)
        pd_data["INDEX"] = pd_data.reset_index().index
        raw_data = self._session.create_dataframe(pd_data)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    ColumnTransformer(
                        transformers=[
                            ("OHE", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_columns),
                            ("MMS", MinMaxScaler(clip=True), numerical_columns),
                        ]
                    ),
                ),
                ("CLF", LogisticRegression(solver="saga", label_cols=label_column)),
            ]
        )

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid={"CLF__penalty": ["l1", "l2"]},
            input_cols=feature_cols,
            label_cols=label_column,
            drop_input_cols=True,
        )
        gs.fit(raw_data)
        predicted = gs.predict(raw_data.to_pandas().sort_values(by="INDEX")).to_numpy()

        sk_pipeline = SkPipeline(
            steps=[
                (
                    "preprocessing",
                    SkColumnTransformer(
                        transformers=[
                            ("OHE", SkOneHotEncoder(handle_unknown="ignore", sparse=False), categorical_columns),
                            ("MMS", SkMinMaxScaler(clip=True), numerical_columns),
                        ]
                    ),
                ),
                ("CLF", SkLogisticRegression(solver="saga")),
            ]
        )
        sk_gs = SkGridSearchCV(
            estimator=sk_pipeline,
            param_grid={"CLF__penalty": ["l1", "l2"]},
        )
        sk_gs.fit(pd_data[feature_cols], pd_data[label_column])
        sk_predicted = sk_gs.predict(pd_data[feature_cols])

        assert gs._sklearn_object.best_params_ == sk_gs.best_params_
        np.testing.assert_allclose(gs._sklearn_object.best_score_, sk_gs.best_score_)
        np.testing.assert_allclose(predicted.flatten(), sk_predicted.flatten(), rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    main()
