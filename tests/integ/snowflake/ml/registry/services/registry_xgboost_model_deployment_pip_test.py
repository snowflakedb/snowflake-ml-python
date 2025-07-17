import inflection
import numpy as np
import pandas as pd
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection
from sklearn.preprocessing import LabelEncoder

from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryXGBoostModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.product(  # type: ignore[misc]
        gpu_requests=[None, "1"],
    )
    def test_xgb(
        self,
        gpu_requests: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        self._test_registry_model_deployment(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                        rtol=1e-3,
                        atol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            options=(
                {"cuda_version": model_env.DEFAULT_CUDA_VERSION, "enable_explainability": False}
                if gpu_requests
                else {"enable_explainability": False}
            ),
            gpu_requests=gpu_requests,
            pip_requirements=[f"xgboost=={xgboost.__version__}"],
        )

    @absltest.skipIf(True, "Temporarily quarantined until Inference server release")
    def test_xgb_wide_input(self) -> None:
        n_samples = 10
        n_features = 750
        data = {}

        for i in range(n_features):
            if i % 3 == 0:
                col_name = f'"z_feature_{i:03d}"'
                data[col_name] = np.random.randint(0, 10, n_samples)
            elif i % 3 == 1:
                col_name = f'"FEATURE_{i:03d}"'
                data[col_name] = np.random.choice(["A", "B", "C"], n_samples)
            else:
                col_name = f"a_feature_{i:03d}"
                data[col_name] = np.random.normal(0, 1, n_samples)

        train_df = pd.DataFrame(data)

        categorical_cols = [col for col in train_df.columns if train_df[col].dtype == "object"]

        for col in categorical_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])

        y_train = np.random.choice([0, 1, 2], n_samples)

        model = xgboost.XGBClassifier(n_estimators=50, reg_lambda=1, gamma=0, max_depth=3)
        model.fit(train_df, y_train)

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_df,
            prediction_assert_fns={
                "predict": (
                    train_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(model.predict(train_df), columns=res.columns),
                        rtol=1e-3,
                        atol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
            pip_requirements=[f"xgboost=={xgboost.__version__}"],
        )


if __name__ == "__main__":
    absltest.main()
