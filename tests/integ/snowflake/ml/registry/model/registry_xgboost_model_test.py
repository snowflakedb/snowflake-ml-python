import inflection
import numpy as np
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class TestRegistryXGBoostModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_xgb(
        self,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        self._test_registry_model(
            model=regressor,
            sample_input=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(regressor.predict(cal_X_test), axis=1)
                    ),
                ),
            },
        )

    def test_xgb_sp(
        self,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True).frame
        cal_data.columns = [inflection.parameterize(c, "_") for c in cal_data]
        cal_data_sp_df = self._session.create_dataframe(cal_data)
        cal_data_sp_df_train, cal_data_sp_df_test = tuple(cal_data_sp_df.random_split([0.25, 0.75], seed=2568))
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        cal_data_pd_df_train = cal_data_sp_df_train.to_pandas()
        regressor.fit(cal_data_pd_df_train.drop(columns=["target"]), cal_data_pd_df_train["target"])
        cal_data_sp_df_test_X = cal_data_sp_df_test.drop('"target"')

        y_df_expected = pd.concat(
            [
                cal_data_sp_df_test_X.to_pandas(),
                pd.DataFrame(regressor.predict(cal_data_sp_df_test_X.to_pandas()), columns=["output_feature_0"]),
            ],
            axis=1,
        )
        self._test_registry_model(
            model=regressor,
            sample_input=cal_data_sp_df_train.drop('"target"'),
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test_X,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
        )

    def test_xgb_booster(
        self,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(xgboost.DMatrix(data=cal_X_test))
        self._test_registry_model(
            model=regressor,
            sample_input=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(res.values, np.expand_dims(y_pred, axis=1), rtol=1e-6),
                ),
            },
        )

    def test_xgb_booster_sp(
        self,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True).frame
        cal_data.columns = [inflection.parameterize(c, "_") for c in cal_data]
        cal_data_sp_df = self._session.create_dataframe(cal_data)
        cal_data_sp_df_train, cal_data_sp_df_test = tuple(cal_data_sp_df.random_split([0.25, 0.75], seed=2568))
        cal_data_pd_df_train = cal_data_sp_df_train.to_pandas()
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(
            params,
            xgboost.DMatrix(data=cal_data_pd_df_train.drop(columns=["target"]), label=cal_data_pd_df_train["target"]),
        )
        cal_data_sp_df_test_X = cal_data_sp_df_test.drop('"target"')
        y_df_expected = pd.concat(
            [
                cal_data_sp_df_test_X.to_pandas(),
                pd.DataFrame(
                    regressor.predict(xgboost.DMatrix(data=cal_data_sp_df_test_X.to_pandas())),
                    columns=["output_feature_0"],
                ),
            ],
            axis=1,
        )
        self._test_registry_model(
            model=regressor,
            sample_input=cal_data_sp_df_train.drop('"target"'),
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test_X,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
