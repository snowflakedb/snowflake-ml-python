#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import pytest
from absl.testing.absltest import TestCase, main
from snowflake.ml.modeling.linear_model.logistic_regression import LogisticRegression

from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import MinMaxScaler, OneHotEncoder
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Column, Session

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


@pytest.mark.pip_incompatible
class GridSearchCVTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_fit_and_compare_results(self) -> None:
        raw_data = self._session.sql(
            """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
                FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS
                LIMIT 2000"""
        ).drop(Column("Y"))

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
        gs.predict(raw_data).to_pandas()


if __name__ == "__main__":
    main()
