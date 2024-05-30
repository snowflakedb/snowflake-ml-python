import inflection
import lightgbm
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class TestRegistryLightGBMModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_lightgbm_classifier(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)

        getattr(self, registry_test_fn)(
            model=classifier,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(classifier.predict(cal_X_test), axis=1)
                    ),
                ),
                "predict_proba": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(res.values, classifier.predict_proba(cal_X_test)),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_lightgbm_classifier_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)

        y_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(classifier.predict(cal_X_test), columns=["output_feature_0"]),
            ],
            axis=1,
        )
        y_df_expected_proba = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(classifier.predict_proba(cal_X_test), columns=["output_feature_0", "output_feature_1"]),
            ],
            axis=1,
        )

        cal_data_sp_df_train = self.session.create_dataframe(cal_X_train)
        cal_data_sp_df_test = self.session.create_dataframe(cal_X_test)
        getattr(self, registry_test_fn)(
            model=classifier,
            sample_input_data=cal_data_sp_df_train,
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
                "predict_proba": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected_proba, check_dtype=False),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_lightgbm_booster(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train({"objective": "regression"}, lightgbm.Dataset(cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(cal_X_test)

        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(res.values, np.expand_dims(y_pred, axis=1), rtol=1e-6),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_lightgbm_booster_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train({"objective": "regression"}, lightgbm.Dataset(cal_X_train, label=cal_y_train))
        y_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(regressor.predict(cal_X_test), columns=["output_feature_0"]),
            ],
            axis=1,
        )

        cal_data_sp_df_train = self.session.create_dataframe(cal_X_train)
        cal_data_sp_df_test = self.session.create_dataframe(cal_X_test)
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_data_sp_df_train,
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
