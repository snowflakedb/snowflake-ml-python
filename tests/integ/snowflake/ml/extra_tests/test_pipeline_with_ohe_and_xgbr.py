#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from absl.testing import absltest

from snowflake.ml.framework.pipeline import Pipeline
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.preprocessing import MinMaxScaler, OneHotEncoder
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
    "DEFAULT",
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


class GridSearchCVTest(absltest.TestCase):
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
                    "OHE",
                    OneHotEncoder(
                        input_cols=categorical_columns, output_cols=categorical_columns, drop_input_cols=True
                    ),
                ),
                (
                    "MMS",
                    MinMaxScaler(
                        clip=True,
                        input_cols=numerical_columns,
                        output_cols=numerical_columns,
                    ),
                ),
                ("regression", XGBRegressor(label_cols=label_column)),
            ]
        )

        pipeline.fit(raw_data)
        pipeline.predict(raw_data).to_pandas()

    def test_fit_and_compare_results_pandas_dataframe(self) -> None:
        raw_data = self._session.sql(
            """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
                FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS
                LIMIT 2000"""
        ).drop(Column("Y"))
        raw_data_pandas = raw_data.to_pandas()

        pipeline = Pipeline(
            steps=[
                (
                    "OHE",
                    OneHotEncoder(
                        input_cols=categorical_columns, output_cols=categorical_columns, drop_input_cols=True
                    ),
                ),
                (
                    "MMS",
                    MinMaxScaler(
                        clip=True,
                        input_cols=numerical_columns,
                        output_cols=numerical_columns,
                    ),
                ),
                ("regression", XGBRegressor(label_cols=label_column)),
            ]
        )

        pipeline.fit(raw_data_pandas)
        pipeline.predict(raw_data_pandas)


if __name__ == "__main__":
    absltest.main()
