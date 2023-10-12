import numpy as np
from absl.testing import absltest

from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Column, Session, functions as F

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

    def test_fit_and_compare_results_pandas(self) -> None:
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
        pipeline.predict(raw_data.to_pandas())

    def test_pipeline_export(self) -> None:
        snow_df = (
            self._session.sql(
                """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
                FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS
                LIMIT 2000"""
            )
            .drop("Y")
            .withColumn("ROW_INDEX", F.monotonically_increasing_id())
        )
        pd_df = snow_df.to_pandas().sort_values(by=["ROW_INDEX"]).drop("LABEL", axis=1)

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
                (
                    "SS",
                    StandardScaler(input_cols=(numerical_columns[0:2]), output_cols=(numerical_columns[0:2])),
                ),
                ("regression", XGBRegressor(label_cols=label_column)),
            ]
        )

        pipeline.fit(snow_df)
        snow_results = pipeline.predict(snow_df).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()

        sk_pipeline = pipeline.to_sklearn()
        sk_results = sk_pipeline.predict(pd_df)
        np.testing.assert_allclose(snow_results.flatten(), sk_results.flatten(), rtol=1.0e-1, atol=1.0e-2)

    def test_pipeline_with_limited_number_of_columns_in_estimator_export(self) -> None:
        snow_df = (
            self._session.sql(
                """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
                FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS
                LIMIT 2000"""
            )
            .drop("Y", "DEFAULT")
            .withColumn("ROW_INDEX", F.monotonically_increasing_id())
        )
        pd_df = snow_df.to_pandas().sort_values(by=["ROW_INDEX"]).drop("LABEL", axis=1)

        pipeline = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(
                        clip=True,
                        input_cols=numerical_columns,
                        output_cols=numerical_columns,
                    ),
                ),
                (
                    "SS",
                    StandardScaler(input_cols=(numerical_columns[0:2]), output_cols=(numerical_columns[0:2])),
                ),
                ("regression", XGBRegressor(input_cols=numerical_columns, label_cols=label_column)),
            ]
        )

        pipeline.fit(snow_df)
        snow_results = pipeline.predict(snow_df).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()

        sk_pipeline = pipeline.to_sklearn()
        sk_results = sk_pipeline.predict(pd_df)
        np.testing.assert_allclose(snow_results.flatten(), sk_results.flatten(), rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    absltest.main()
