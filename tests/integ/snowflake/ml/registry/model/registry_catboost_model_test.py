import catboost
import inflection
import numpy as np
import pandas as pd
import shap
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model import model_signature
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class TestRegistryCatBoostModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_catboost_classifier_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
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
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_catboost_classifier_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.Explainer(classifier)(cal_X_test).values

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
                "explain": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(res.values, expected_explanations),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_catboost_classifier_sp_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
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
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_catboost_classifier_explain_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
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
        explanation_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(
                    shap.Explainer(classifier)(cal_X_test).values,
                    columns=[f"{c}_explanation" for c in cal_X_test.columns],
                ),
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
                "explain": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, explanation_df_expected, check_dtype=False),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_catboost_with_signature_and_sample_data(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)
        y_pred_log_proba = classifier.predict_log_proba(cal_X_test)
        sig = {
            "predict": model_signature.infer_signature(cal_X_test, y_pred),
            "predict_proba": model_signature.infer_signature(cal_X_test, y_pred_proba),
            "predict_log_proba": model_signature.infer_signature(cal_X_test, y_pred_log_proba),
        }
        expected_explanations = shap.Explainer(classifier)(cal_X_test).values

        # passing both signature and sample_input_data when enable_explainability is True
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
                "predict_log_proba": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(res.values, classifier.predict_log_proba(cal_X_test)),
                ),
                "explain": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(res.values, expected_explanations),
                ),
            },
            options={"enable_explainability": True},
            signatures=sig,
        )

        with self.assertRaisesRegex(
            ValueError, "Signatures and sample_input_data both cannot be specified at the same time."
        ):
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
                    "predict_log_proba": (
                        cal_X_test,
                        lambda res: np.testing.assert_allclose(res.values, classifier.predict_log_proba(cal_X_test)),
                    ),
                    "explain": (
                        cal_X_test,
                        lambda res: np.testing.assert_allclose(res.values, expected_explanations),
                    ),
                },
                signatures=sig,
                additional_version_suffix="v2",
            )


if __name__ == "__main__":
    absltest.main()
