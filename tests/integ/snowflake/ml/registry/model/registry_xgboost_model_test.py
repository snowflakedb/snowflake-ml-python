import inflection
import numpy as np
import pandas as pd
import shap
import xgboost
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


class TestRegistryXGBoostModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_manual_shap_override(self, registry_test_fn: str) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "explain": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(expected_explanations, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            # pin version of shap for tests
            additional_dependencies=[f"shap=={shap.__version__}"],
            function_type_assert={"explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)

        def _check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    _check_predict_fn,
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_pipeline_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = SK_pipeline.Pipeline(
            steps=[
                ("regressor", xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)),
            ]
        )

        regressor.fit(cal_X_train, cal_y_train)
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(
                            regressor.predict(cal_X_test),
                            columns=res.columns,
                        ),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_explain_by_default(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "explain": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(expected_explanations, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            function_type_assert={"explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_explain_explicitly_enabled(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
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
            function_type_assert={"explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_sp_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True).frame
        cal_data.columns = [inflection.parameterize(c, "_") for c in cal_data]
        cal_data_sp_df = self.session.create_dataframe(cal_data)
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
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_data_sp_df_train.drop('"target"'),
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test_X,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_explain_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True).frame
        cal_data.columns = [inflection.parameterize(c, "_") for c in cal_data]
        cal_data_sp_df = self.session.create_dataframe(cal_data)
        cal_data_sp_df_train, cal_data_sp_df_test = tuple(cal_data_sp_df.random_split([0.25, 0.75], seed=2568))
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        cal_data_pd_df_train = cal_data_sp_df_train.to_pandas()
        regressor.fit(cal_data_pd_df_train.drop(columns=["target"]), cal_data_pd_df_train["target"])
        cal_data_sp_df_test_X = cal_data_sp_df_test.drop('"target"')

        explanation_df_expected = pd.concat(
            [
                cal_data_sp_df_test_X.to_pandas(),
                pd.DataFrame(
                    shap.TreeExplainer(regressor)(cal_data_sp_df_test_X.to_pandas()).values,
                    columns=[f"{c}_explanation" for c in cal_data_sp_df_test_X.to_pandas().columns],
                ),
            ],
            axis=1,
        )
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_data_sp_df_train.drop('"target"'),
            prediction_assert_fns={
                "explain": (
                    cal_data_sp_df_test_X,
                    lambda res: dataframe_utils.check_sp_df_res(
                        res, explanation_df_expected, check_dtype=False, rtol=1e-4
                    ),
                ),
            },
            function_type_assert={"explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_booster_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(xgboost.DMatrix(data=cal_X_test))

        def _check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(y_pred, columns=res.columns),
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    _check_predict_fn,
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_booster_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "explain": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(expected_explanations, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            function_type_assert={"explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_booster_sp_no_explain(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True).frame
        cal_data.columns = [inflection.parameterize(c, "_") for c in cal_data]
        cal_data_sp_df = self.session.create_dataframe(cal_data)
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
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_data_sp_df_train.drop('"target"'),
            prediction_assert_fns={
                "predict": (
                    cal_data_sp_df_test_X,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_booster_explain_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True).frame
        cal_data.columns = [inflection.parameterize(c, "_") for c in cal_data]
        cal_data_sp_df = self.session.create_dataframe(cal_data)
        cal_data_sp_df_train, cal_data_sp_df_test = tuple(cal_data_sp_df.random_split([0.25, 0.75], seed=2568))
        cal_data_pd_df_train = cal_data_sp_df_train.to_pandas()
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(
            params,
            xgboost.DMatrix(data=cal_data_pd_df_train.drop(columns=["target"]), label=cal_data_pd_df_train["target"]),
        )
        cal_data_sp_df_test_X = cal_data_sp_df_test.drop('"target"')
        explanations_df_expected = pd.concat(
            [
                cal_data_sp_df_test_X.to_pandas(),
                pd.DataFrame(
                    shap.TreeExplainer(regressor)(cal_data_sp_df_test_X.to_pandas()).values,
                    columns=[f"{c}_explanation" for c in cal_data_sp_df_test_X.to_pandas().columns],
                ),
            ],
            axis=1,
        )

        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_data_sp_df_train.drop('"target"'),
            prediction_assert_fns={
                "explain": (
                    cal_data_sp_df_test_X,
                    lambda res: dataframe_utils.check_sp_df_res(
                        res, explanations_df_expected, check_dtype=False, rtol=1e-4
                    ),
                ),
            },
            function_type_assert={"explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_booster_with_signature_and_sample_data(
        self,
        registry_test_fn: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        y_pred = pd.DataFrame(
            regressor.predict(xgboost.DMatrix(data=cal_X_test)),
            columns=["output_feature_0"],
        )
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        sig = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
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
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_xgb_model_with_categorical_dtype_columns(
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
                ("classifier", xgboost.XGBClassifier(tree_method="hist")),
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
            # TODO(SNOW-1677301): Add support for explainability for categorical columns
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    @absltest.skip("SNOW-1752904")
    def test_xgb_model_with_native_categorical_dtype_columns(
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
        # categorical_columns = ["color", "size"]

        classifier = xgboost.XGBClassifier(tree_method="hist", enable_categorical=True)
        classifier.fit(df[input_features], df["target"])

        getattr(self, registry_test_fn)(
            model=classifier,
            sample_input_data=df[input_features],
            prediction_assert_fns={
                "predict": (
                    df[input_features],
                    lambda res: np.testing.assert_allclose(
                        res["output_feature_0"].values, classifier.predict(df[input_features])
                    ),
                ),
            },
            # TODO(SNOW-1677301): Add support for explainability for categorical columns
            options={"enable_explainability": False},
        )

        # TODO(SNOW-1752904):
        # The inference fails with message
        # ValueError: DataFrame.dtypes for data must be int, float, bool or category.
        # When categorical type is supplied, The experimental DMatrix parameter`enable_categorical`
        # must be set to `True`.  Invalid columns:color: object
        #  in function PREDICT with handler predict.infer


if __name__ == "__main__":
    absltest.main()
