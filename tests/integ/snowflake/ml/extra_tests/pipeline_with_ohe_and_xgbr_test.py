import os

import numpy as np
import pandas as pd
import pytest
from absl.testing import absltest
from importlib_resources import files
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.impute import KNNImputer as SkKNNImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    MinMaxScaler as SkMinMaxScaler,
    OneHotEncoder as SkOneHotEncoder,
)
from xgboost import XGBClassifier as XGB_XGBClassifier

from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.impute import KNNImputer
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import MinMaxScaler, OneHotEncoder
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import test_env_utils

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
IN_ML_RUNTIME_ENV_VAR = "IN_SPCS_ML_RUNTIME"
feature_cols = categorical_columns + numerical_columns


class PipelineXGBRTest(absltest.TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        data_file = files("tests.integ.snowflake.ml.test_data").joinpath("UCI_BANK_MARKETING_20COLUMNS.csv")
        self._test_data = pd.read_csv(data_file, index_col=0)

    def tearDown(self):
        self._session.close()

    def _get_preprocessor(self, categorical_columns, numerical_columns, use_knn_imputer=True):
        """Helper method to create the ColumnTransformer for preprocessing."""
        transformers = [
            ("ohe", OneHotEncoder(drop_input_cols=True), categorical_columns),
            ("mms", MinMaxScaler(clip=True), numerical_columns),
        ]

        if use_knn_imputer:
            transformers.append(("knn_imputer", KNNImputer(), numerical_columns))

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",  # Ensures columns not specified are passed through without transformation
        )

    def _get_pipeline(self, categorical_columns, numerical_columns, label_column, use_knn_imputer=True):
        """Helper method to create the Pipeline with the appropriate preprocessor and XGBClassifier."""

        # Check if the environment variable is set to True
        if os.environ.get(IN_ML_RUNTIME_ENV_VAR):
            # Create the preprocessor using the helper method
            preprocessor = self._get_preprocessor(categorical_columns, numerical_columns, use_knn_imputer)

            # Create and return the pipeline with the preprocessor
            return Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regression", XGBClassifier(label_cols=label_column, passthrough_cols="ROW_INDEX")),
                ]
            )

        # When the environment variable is not set
        steps = [
            (
                "OHE",
                OneHotEncoder(input_cols=categorical_columns, output_cols=categorical_columns, drop_input_cols=True),
            ),
            (
                "MMS",
                MinMaxScaler(
                    clip=True,
                    input_cols=numerical_columns,
                    output_cols=numerical_columns,
                ),
            ),
            ("regression", XGBClassifier(label_cols=label_column, passthrough_cols="ROW_INDEX")),
        ]

        if use_knn_imputer:
            steps.insert(2, ("KNNImputer", KNNImputer(input_cols=numerical_columns, output_cols=numerical_columns)))

        return Pipeline(steps=steps)

    def test_fit_and_compare_results(self) -> None:
        pd_data = self._test_data
        pd_data["ROW_INDEX"] = pd_data.reset_index().index
        raw_data = self._session.create_dataframe(pd_data)

        pipeline = self._get_pipeline(categorical_columns, numerical_columns, label_column)
        pipeline.fit(raw_data)
        results = pipeline.predict(raw_data).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()

        sk_pipeline = SkPipeline(
            steps=[
                (
                    "Preprocessing",
                    SkColumnTransformer(
                        [
                            ("cat_transformer", SkOneHotEncoder(), categorical_columns),
                            ("num_transforms", SkMinMaxScaler(), numerical_columns),
                            ("num_imputer", SkKNNImputer(), numerical_columns),
                        ]
                    ),
                ),
                ("Training", XGB_XGBClassifier()),
            ]
        )
        sk_pipeline.fit(pd_data[numerical_columns + categorical_columns], pd_data[label_column])
        sk_results = sk_pipeline.predict(pd_data[numerical_columns + categorical_columns])

        np.testing.assert_allclose(results.flatten(), sk_results.flatten(), rtol=1.0e-1, atol=1.0e-2)

    @pytest.mark.skipif(
        os.getenv("IN_SPCS_ML_RUNTIME") == "True",
        reason=(
            "Skipping this test on Container Runtimes. "
            "See: https://snowflakecomputing.atlassian.net/browse/SNOW-1648870"
        ),
    )
    def test_fit_predict_proba_and_compare_results(self) -> None:
        pd_data = self._test_data
        pd_data["ROW_INDEX"] = pd_data.reset_index().index
        raw_data = self._session.create_dataframe(pd_data)

        pipeline = self._get_pipeline(categorical_columns, numerical_columns, label_column)
        pipeline.fit(raw_data)
        results = pipeline.predict_proba(raw_data).to_pandas().sort_values(by=["ROW_INDEX"])
        proba_cols = [c for c in results.columns if c.startswith("PREDICT_PROBA")]
        proba_results = results[proba_cols].to_numpy()

        sk_pipeline = SkPipeline(
            steps=[
                (
                    "Preprocessing",
                    SkColumnTransformer(
                        [
                            ("cat_transformer", SkOneHotEncoder(), categorical_columns),
                            ("num_transforms", SkMinMaxScaler(), numerical_columns),
                            ("num_imputer", SkKNNImputer(), numerical_columns),
                        ]
                    ),
                ),
                ("Training", XGB_XGBClassifier()),
            ]
        )
        sk_pipeline.fit(pd_data[numerical_columns + categorical_columns], pd_data[label_column])
        sk_proba_results = sk_pipeline.predict_proba(pd_data[numerical_columns + categorical_columns])

        np.testing.assert_allclose(proba_results.flatten(), sk_proba_results.flatten(), rtol=1.0e-1, atol=1.0e-2)

    @pytest.mark.skipif(
        os.getenv("IN_SPCS_ML_RUNTIME") == "True",
        reason=(
            "Skipping this test, as we go ahead with this PR"
            "See: https://github.com/snowflakedb/snowml/pull/2651/files"
        ),
    )
    def test_fit_and_compare_results_pandas(self) -> None:
        pd_data = self._test_data
        pd_data["ROW_INDEX"] = pd_data.reset_index().index
        raw_data = self._session.create_dataframe(pd_data)

        pipeline = self._get_pipeline(categorical_columns, numerical_columns, label_column, use_knn_imputer=False)

        pipeline.fit(raw_data)
        pipeline.predict(raw_data.to_pandas())

    def test_pipeline_squash(self) -> None:
        pd_data = self._test_data
        pd_data["ROW_INDEX"] = pd_data.reset_index().index
        raw_data = self._session.create_dataframe(pd_data)

        pipeline = self._get_pipeline(categorical_columns, numerical_columns, label_column)

        pipeline._deps.append(
            test_env_utils.get_latest_package_version_spec_in_server(self._session, "snowflake-snowpark-python")
        )

        p1 = pipeline.fit(raw_data)
        results1 = p1.predict(raw_data).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()

        p2 = pipeline.fit(raw_data, squash=True)
        results2 = p2.predict(raw_data).to_pandas().sort_values(by=["ROW_INDEX"])["OUTPUT_LABEL"].to_numpy()

        self.assertEqual(hash(p1), hash(p2))

        np.testing.assert_allclose(results1.flatten(), results2.flatten(), rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    absltest.main()
