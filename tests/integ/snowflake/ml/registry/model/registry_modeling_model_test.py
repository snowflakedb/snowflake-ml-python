import numpy as np
from absl.testing import absltest
from sklearn import datasets

from snowflake.ml.modeling.lightgbm import LGBMRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.xgboost import XGBRegressor
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistryModelingModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_snowml_model_deploy_snowml_sklearn(
        self,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
        )

    def test_snowml_model_deploy_xgboost(
        self,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
        )

    def test_snowml_model_deploy_lightgbm(
        self,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
