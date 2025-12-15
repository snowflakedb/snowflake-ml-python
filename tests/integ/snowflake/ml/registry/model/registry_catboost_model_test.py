import catboost
import inflection
import pandas as pd
import shap
from absl.testing import absltest, parameterized
from sklearn import (
    compose,
    datasets,
    model_selection,
    pipeline as SK_pipeline,
    preprocessing,
)

from snowflake.ml.model import model_signature
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class TestRegistryCatBoostModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_catboost_classifier_no_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier(thread_count=1)
        classifier.fit(cal_X_train, cal_y_train)

        self._test_registry_model(
            model=classifier,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "predict_proba": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict_proba(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    def test_catboost_classifier_pipeline_no_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = SK_pipeline.Pipeline(
            steps=[
                ("regressor", catboost.CatBoostClassifier(thread_count=1)),
            ]
        )
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
        self._test_registry_model(
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

    def test_catboost_classifier_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier(thread_count=1)
        classifier.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.Explainer(classifier)(cal_X_test).values

        self._test_registry_model(
            model=classifier,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "predict_proba": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict_proba(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "explain": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(expected_explanations, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_catboost_classifier_sp_no_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier(thread_count=1)
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
        self._test_registry_model(
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

    def test_catboost_classifier_explain_sp(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier(thread_count=1)
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
        self._test_registry_model(
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
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_catboost_with_signature_and_sample_data(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier(thread_count=1)
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
        self._test_registry_model(
            model=classifier,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "predict_proba": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict_proba(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "predict_log_proba": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict_log_proba(cal_X_test), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "explain": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(expected_explanations, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": True},
            signatures=sig,
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_log_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_catboost_model_with_categorical_dtype_columns(
        self,
        registry_test_fn: str,
    ) -> None:
        data = {
            "color": ["red", "blue", "green", "red"],
            "size": [1, 2, 2, 4],
            "price": [10, 15, 20, 25],
            "target": [0, 1, 1, 0],
        }
        input_features = ["color", "size", "price"]

        df = pd.DataFrame(data)
        df["color"] = df["color"].astype("category")
        df["size"] = df["size"].astype("category")

        # Define categorical columns
        categorical_columns = ["color", "size"]

        # Create a column transformer
        preprocessor = compose.ColumnTransformer(
            transformers=[
                ("cat", preprocessing.OneHotEncoder(), categorical_columns),
            ],
            remainder="passthrough",
        )

        pipeline = SK_pipeline.Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", catboost.CatBoostClassifier()),
            ]
        )
        pipeline.fit(df[input_features], df["target"])

        def _check_predict_fn(res) -> None:
            pd.testing.assert_frame_equal(
                res["output_feature_0"].to_frame(),
                pd.DataFrame(pipeline.predict(df[input_features]), columns=["output_feature_0"]),
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=pipeline,
            sample_input_data=df[input_features],
            prediction_assert_fns={
                "predict": (
                    df[input_features],
                    _check_predict_fn,
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
