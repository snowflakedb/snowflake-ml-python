from typing import Any

import inflection
import lightgbm
import numpy as np
import numpy.typing as npt
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
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class TestRegistryLightGBMModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_lightgbm_classifier_no_explain(
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

    def test_lightgbm_classifier_pipeline_no_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        pipeline = SK_pipeline.Pipeline(
            [
                ("classifier", lightgbm.LGBMClassifier()),
            ]
        )
        pipeline.fit(cal_X_train, cal_y_train)

        def _check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(pipeline.predict(cal_X_test), columns=res.columns),
                check_dtype=False,
            )

        def _check_predict_proba_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(pipeline.predict_proba(cal_X_test), columns=res.columns),
                check_dtype=False,
            )

        self._test_registry_model(
            model=pipeline,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    _check_predict_fn,
                ),
                "predict_proba": (
                    cal_X_test,
                    _check_predict_proba_fn,
                ),
            },
            options={"enable_explainability": False},
        )

    def test_lightgbm_classifier_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        expected_explanations: npt.NDArray[Any] = shap.Explainer(classifier)(cal_X_test).values
        if expected_explanations.ndim == 3 and expected_explanations.shape[2] == 2:
            expected_explanations = np.apply_along_axis(lambda arr: arr[1], -1, expected_explanations)

        def check_explain_fn(res) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(expected_explanations, columns=res.columns),
                check_dtype=False,
            )

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
                    check_explain_fn,
                ),
            },
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_lightgbm_classifier_sp_no_explain(self) -> None:
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

    def test_lightgbm_classifier_explain_sp(self) -> None:
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
        explain_df = handlers_utils.convert_explanations_to_2D_df(
            classifier, shap.Explainer(classifier)(cal_X_test).values
        ).set_axis([f"{c}_explanation" for c in cal_X_test.columns], axis=1)
        explanation_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                explain_df,
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
                    lambda res: dataframe_utils.check_sp_df_res(
                        res, explanation_df_expected, check_dtype=False, deserialize_json=True
                    ),
                ),
            },
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_lightgbm_booster_no_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train({"objective": "regression"}, lightgbm.Dataset(cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(cal_X_test)

        self._test_registry_model(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(y_pred, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    def test_lightgbm_booster_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train({"objective": "regression"}, lightgbm.Dataset(cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(cal_X_test)
        expected_explanations = shap.Explainer(regressor)(cal_X_test).values

        self._test_registry_model(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(y_pred, columns=res.columns),
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
            },
        )

    def test_lightgbm_booster_sp_no_explain(self) -> None:
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
        self._test_registry_model(
            model=regressor,
            sample_input_data=cal_data_sp_df_train,
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            options={"enable_explainability": False},
        )

    def test_lightgbm_booster_explain_sp(self) -> None:
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

        explanation_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(
                    shap.Explainer(regressor)(cal_X_test).values,
                    columns=[f"{c}_explanation" for c in cal_X_test.columns],
                ),
            ],
            axis=1,
        )

        cal_data_sp_df_train = self.session.create_dataframe(cal_X_train)
        cal_data_sp_df_test = self.session.create_dataframe(cal_X_test)
        self._test_registry_model(
            model=regressor,
            sample_input_data=cal_data_sp_df_train,
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
                "explain": (
                    cal_data_sp_df_test,
                    lambda res: dataframe_utils.check_sp_df_res(res, explanation_df_expected, check_dtype=False),
                ),
            },
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_lightgbm_with_signature_and_sample_data(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = pd.DataFrame(classifier.predict(cal_X_test), columns=["output_feature_0"])
        sig = {
            "predict": model_signature.infer_signature(cal_X_test, y_pred),
        }

        expected_explanations: npt.NDArray[Any] = shap.Explainer(classifier)(cal_X_test).values
        if expected_explanations.ndim == 3 and expected_explanations.shape[2] == 2:
            expected_explanations = np.apply_along_axis(lambda arr: arr[1], -1, expected_explanations)

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
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_lightgbm_model_with_categorical_dtype_columns(
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
            transformers=[("cat", preprocessing.OneHotEncoder(), categorical_columns)],
            remainder="passthrough",
        )

        pipeline = SK_pipeline.Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", lightgbm.LGBMClassifier()),
            ]
        )
        pipeline.fit(df.drop("target", axis=1), df["target"])

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
