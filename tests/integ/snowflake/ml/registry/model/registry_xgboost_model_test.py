from typing import Callable

import inflection
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

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class TestRegistryXGBoostModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
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

    def test_xgb_no_explain(self) -> None:
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

        self._test_registry_model(
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

    def test_xgb_pipeline_no_explain(self) -> None:
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
        self._test_registry_model(
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

    def test_xgb_explain_by_default(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        self._test_registry_model(
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

    def test_xgb_explain_explicitly_enabled(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        self._test_registry_model(
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

    def test_xgb_sp_no_explain(self) -> None:
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
        self._test_registry_model(
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

    def test_xgb_explain_sp(self) -> None:
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
        self._test_registry_model(
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

    def test_xgb_booster_no_explain(self) -> None:
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

        self._test_registry_model(
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

    def test_xgb_booster_explain(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        self._test_registry_model(
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

    def test_xgb_booster_sp_no_explain(self) -> None:
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
        self._test_registry_model(
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

    def test_xgb_booster_explain_sp(self) -> None:
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

        self._test_registry_model(
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

    def test_xgb_booster_with_signature_and_sample_data(self) -> None:
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
        self._test_registry_model(
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

    def test_xgb_model_with_categorical_dtype_columns(self) -> None:
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

        self._test_registry_model(
            model=pipeline,
            sample_input_data=df[input_features],
            prediction_assert_fns={
                "predict": (
                    df[input_features],
                    _check_predict_fn,
                ),
            },
        )

    def test_xgb_model_with_dmatrix_input(self) -> None:
        data = {
            "size": [1, 2, 2, 4],
            "price": [10, 15, 20, 25],
            "dimension": [0, 2.2, -3, 4.665656],
            "target": [0, 1, 1, 0],
        }
        input_features = ["size", "price", "dimension"]

        df = pd.DataFrame(data)

        d_matrix = xgboost.DMatrix(data=df[input_features], label=df["target"])
        xgb_model = xgboost.train(
            params={"objective": "binary:logistic", "eval_metric": "logloss"},
            dtrain=d_matrix,
            num_boost_round=100,
        )

        d_matrix_input = xgboost.DMatrix(
            data=df[input_features],
        )

        def check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(xgb_model.predict(d_matrix_input), columns=res.columns),
                check_dtype=False,
            )

        self._test_registry_model(
            model=xgb_model,
            sample_input_data=d_matrix,
            prediction_assert_fns={
                "predict": (
                    d_matrix_input,
                    check_predict_fn,
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
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
        input_features = [
            "color",
            "size",
            "price",
        ]

        df = pd.DataFrame(data)
        df["color"] = df["color"].astype("category")
        df["size"] = df["size"].astype("category")

        # Define categorical columns
        # categorical_columns = ["color", "size"]

        d_matrix = xgboost.DMatrix(data=df[input_features], label=df["target"], enable_categorical=True)
        xgb_model = xgboost.train(
            params={"objective": "binary:logistic", "eval_metric": "logloss"},
            dtrain=d_matrix,
            num_boost_round=100,
        )

        d_matrix_input = xgboost.DMatrix(
            data=df[input_features],
            enable_categorical=True,
        )

        def check_predict_fn(res) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(xgb_model.predict(d_matrix_input), columns=res.columns),
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=xgb_model,
            sample_input_data=d_matrix,
            prediction_assert_fns={
                "predict": (
                    d_matrix_input,
                    check_predict_fn,
                ),
            },
        )

    def test_xgb_model_with_quoted_identifiers_ignore_case(self):
        cal_X, cal_y = datasets.load_breast_cancer(return_X_y=True)
        cal_X_df = pd.DataFrame(cal_X, columns=[f"col_{i}" for i in range(cal_X.shape[1])])

        regressor = xgboost.XGBRegressor()
        regressor.fit(cal_X_df, cal_y)

        name = "xgb_model_test_quoted_identifiers_param"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=regressor,
            model_name=name,
            version_name=version,
            sample_input_data=cal_X_df,
        )

        statement_params = {"QUOTED_IDENTIFIERS_IGNORE_CASE": "TRUE"}

        functions = mv._functions
        predict_name = sql_identifier.SqlIdentifier("predict").identifier()
        find_method: Callable[[model_manifest_schema.ModelFunctionInfo], bool] = (
            lambda method: method["name"] == predict_name
        )
        target_function_info = next(
            filter(find_method, functions),
            None,
        )
        self.assertIsNotNone(target_function_info, "predict function not found")

        result = mv._model_ops.invoke_method(
            method_name=sql_identifier.SqlIdentifier(target_function_info["name"]),
            method_function_type=target_function_info["target_method_function_type"],
            signature=target_function_info["signature"],
            X=cal_X_df[:5],
            database_name=None,
            schema_name=None,
            model_name=mv._model_name,
            version_name=mv._version_name,
            strict_input_validation=False,
            statement_params=statement_params,
            is_partitioned=target_function_info["is_partitioned"],
        )

        result_cols = list(result.columns)

        for col in result_cols:
            self.assertTrue(
                col.isupper(), f"Expected column {col} to be uppercase with QUOTED_IDENTIFIERS_IGNORE_CASE=TRUE"
            )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0, "Result should not be empty")

        self.registry.delete_model(model_name=name)


if __name__ == "__main__":
    absltest.main()
