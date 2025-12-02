from typing import Callable, Optional, cast

import numpy as np
import pandas as pd
import shap
from absl.testing import absltest, parameterized
from sklearn import (
    compose,
    datasets,
    ensemble,
    linear_model,
    multioutput,
    neighbors,
    pipeline as SK_pipeline,
    preprocessing,
)

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, test_env_utils


class TestRegistrySKLearnModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_model(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classification task, such as iris
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X, iris_y)
        getattr(self, registry_test_fn)(
            model=classifier,
            sample_input_data=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: pd.testing.assert_frame_equal(
                        res["output_feature_0"].to_frame("output_feature_0"),
                        pd.DataFrame(classifier.predict(iris_X), columns=["output_feature_0"]),
                        check_dtype=False,
                    ),
                ),
                "predict_proba": (
                    iris_X[:10],
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict_proba(iris_X[:10]), columns=res.columns),
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

    def test_skl_model_explain(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # sample input needs to be pandas dataframe for now
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X_df, iris_y)
        expected_explanations = shap.Explainer(classifier, iris_X_df)(iris_X_df).values

        def _check_explain(res: pd.DataFrame) -> None:
            actual_explain_df = handlers_utils.convert_explanations_to_2D_df(classifier, expected_explanations)
            rename_columns = {
                old_col_name: new_col_name for old_col_name, new_col_name in zip(actual_explain_df.columns, res.columns)
            }
            actual_explain_df.rename(columns=rename_columns, inplace=True)
            pd.testing.assert_frame_equal(
                res,
                actual_explain_df,
                check_dtype=False,
            )

        self._test_registry_model(
            model=classifier,
            sample_input_data=iris_X_df,
            prediction_assert_fns={
                "predict": (
                    iris_X_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res["output_feature_0"].to_frame("output_feature_0"),
                        pd.DataFrame(classifier.predict(iris_X), columns=["output_feature_0"]),
                        check_dtype=False,
                    ),
                ),
                "predict_proba": (
                    iris_X_df.iloc[:10],
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict_proba(iris_X_df[:10]), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "explain": (
                    iris_X_df,
                    _check_explain,
                ),
            },
            options={"enable_explainability": True},
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_sklearn_explain_sp(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        iris_X_sp_df = self.session.create_dataframe(iris_X_df)
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X_df, iris_y)

        explain_df = handlers_utils.convert_explanations_to_2D_df(
            classifier, shap.Explainer(classifier, iris_X_df)(iris_X_df).values
        ).set_axis([f"{c}_explanation" for c in iris_X_df.columns], axis=1)

        explanation_df_expected = pd.concat([iris_X_df, explain_df], axis=1)
        self._test_registry_model(
            model=classifier,
            sample_input_data=iris_X_sp_df,
            prediction_assert_fns={
                "explain": (
                    iris_X_sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(
                        res, explanation_df_expected, check_dtype=False, rtol=1e-4
                    ),
                ),
            },
            options={"enable_explainability": True},
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
                "predict_proba": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    def test_skl_model_case_sensitive(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classification task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        self._test_registry_model(
            model=regr,
            sample_input_data=iris_X,
            options={
                "method_options": {"predict": {"case_sensitive": True}, "predict_proba": {"case_sensitive": True}},
                "target_methods": ["predict", "predict_proba"],
            },
            prediction_assert_fns={
                '"predict"': (
                    iris_X,
                    lambda res: pd.testing.assert_frame_equal(
                        res["output_feature_0"].to_frame("output_feature_0"),
                        pd.DataFrame(regr.predict(iris_X), columns=["output_feature_0"]),
                        check_dtype=False,
                    ),
                ),
                '"predict_proba"': (
                    iris_X[:10],
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(regr.predict_proba(iris_X[:10]), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
        )

    def test_skl_multiple_output_model(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42, n_jobs=1))
        model.fit(iris_X[:10], dual_target[:10])
        self._test_registry_model(
            model=model,
            sample_input_data=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X[-10:],
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(model.predict(iris_X[-10:]), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
                "predict_proba": (
                    iris_X[-10:],
                    lambda res: np.testing.assert_allclose(
                        np.hstack([np.array(res[col].to_list()) for col in cast(pd.DataFrame, res)]),
                        np.hstack(model.predict_proba(iris_X[-10:])),
                    ),
                ),
            },
        )

    def test_skl_unsupported_explain(
        self,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42, n_jobs=1))
        model.fit(iris_X[:10], dual_target[:10])
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        name = "model_test_skl_unsupported_explain"
        version = f"ver_{self._run_id}"

        with self.assertRaisesRegex(
            ValueError, "Explainability for this model is not supported. Please set `enable_explainability=False`"
        ):
            self.registry.log_model(
                model=model,
                model_name=name,
                # TODO(SNOW-2210046): Remove this once the live-commit deletes the version upon error
                version_name=version + "_error",
                sample_input_data=iris_X_df,
                conda_dependencies=conda_dependencies,
                options={"enable_explainability": True},
            )

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=iris_X_df,
            conda_dependencies=conda_dependencies,
        )

        res = mv.run(iris_X[-10:], function_name="predict")
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(model.predict(iris_X[-10:]), columns=res.columns),
            check_dtype=False,
        )

        res = mv.run(iris_X[-10:], function_name="predict_proba")
        np.testing.assert_allclose(
            np.hstack([np.array(res[col].to_list()) for col in cast(pd.DataFrame, res)]),
            np.hstack(model.predict_proba(iris_X[-10:])),
        )

        self.registry.delete_model(model_name=name)

    def test_skl_pipeline_explain_case_sensitive_with_quoted_identifiers_ignore_case(self) -> None:
        # Build a pipeline with OneHotEncoder to simulate transformed feature names
        data = {
            "Color": ["red eyes", "blue", "green", "red eyes", "blue", "green"],
            "size": [1, 2, 2, 4, 3, 1],
            "price": [10, 15, 20, 25, 18, 12],
            "target": [0, 1, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        df["Color"] = df["Color"].astype("category")
        input_features = ["Color", "size", "price"]

        preprocessor = compose.ColumnTransformer(
            transformers=[
                ("cat", preprocessing.OneHotEncoder(), ["Color"]),
            ],
            remainder="passthrough",
        )

        pipeline = SK_pipeline.Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", linear_model.LogisticRegression(max_iter=1000)),
            ]
        )

        pipeline.fit(df[input_features], df["target"])

        name = "skl_pipeline_test_quoted_identifiers_case_sensitive_explain"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=pipeline,
            model_name=name,
            version_name=version,
            sample_input_data=df[input_features],
            options={
                "enable_explainability": True,
                # Ensure some methods are registered as case-sensitive, including explain
                "method_options": {
                    "predict": {"case_sensitive": True},
                    "predict_proba": {"case_sensitive": True},
                },
            },
        )

        functions = mv._functions
        find_method: Callable[[model_manifest_schema.ModelFunctionInfo], bool] = (
            lambda method: "explain" in method["name"]
        )
        target_function_info = next(
            filter(find_method, functions),
            None,
        )
        self.assertIsNotNone(target_function_info, "explain function not found")

        result = mv.run(
            df[input_features],
            function_name=target_function_info["name"],
            strict_input_validation=False,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0, "Result should not be empty")

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def test_skl_model_with_signature_and_sample_data(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # sample input needs to be pandas dataframe for now
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X_df, iris_y)
        expected_explanations = shap.Explainer(classifier, iris_X_df)(iris_X_df).values

        y_pred = pd.DataFrame(classifier.predict(iris_X_df), columns=["output_feature_0"])
        sig = {
            "predict": model_signature.infer_signature(iris_X_df, y_pred),
        }

        def _check_explain(res: pd.DataFrame) -> None:
            actual_explain_df = handlers_utils.convert_explanations_to_2D_df(classifier, expected_explanations)
            rename_columns = {
                old_col_name: new_col_name for old_col_name, new_col_name in zip(actual_explain_df.columns, res.columns)
            }
            actual_explain_df.rename(columns=rename_columns, inplace=True)
            pd.testing.assert_frame_equal(
                res,
                actual_explain_df,
                check_dtype=False,
            )

        self._test_registry_model(
            model=classifier,
            sample_input_data=iris_X_df,
            prediction_assert_fns={
                "predict": (
                    iris_X_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res["output_feature_0"].to_frame("output_feature_0"),
                        pd.DataFrame(classifier.predict(iris_X), columns=["output_feature_0"]),
                        check_dtype=False,
                    ),
                ),
                "explain": (
                    iris_X_df,
                    _check_explain,
                ),
            },
            options={"enable_explainability": True},
            signatures=sig,
        )

    @parameterized.product(  # type: ignore[misc]
        enable_explainability=[True, False, None],
    )
    def test_skl_model_with_categorical_dtype_columns(
        self,
        enable_explainability: Optional[bool],
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
                ("classifier", linear_model.LogisticRegression()),
            ]
        )
        pipeline.fit(df.drop("target", axis=1), df["target"])
        expected_signatures = {
            "predict": model_signature.infer_signature(
                df[input_features],
                df["target"].rename("output_feature_0"),
            ),
        }

        def _check_explain(res: pd.DataFrame) -> None:
            expected_explanations = shap.Explainer(pipeline[-1], preprocessor.transform(df[input_features]))(
                preprocessor.transform(df[input_features])
            ).values
            actual_explain_df = handlers_utils.convert_explanations_to_2D_df(pipeline, expected_explanations)
            rename_columns = {
                old_col_name: new_col_name for old_col_name, new_col_name in zip(actual_explain_df.columns, res.columns)
            }
            actual_explain_df.rename(columns=rename_columns, inplace=True)
            pd.testing.assert_frame_equal(
                res,
                actual_explain_df,
                check_dtype=False,
            )

        prediction_assert_fns = {
            "predict": (
                df[input_features],
                lambda res: pd.testing.assert_series_equal(
                    res["output_feature_0"],
                    pd.Series(pipeline.predict(df[input_features]), name="output_feature_0"),
                    check_dtype=False,
                ),
            ),
        }
        if enable_explainability:
            prediction_assert_fns["explain"] = (df[input_features], _check_explain)

        self._test_registry_model(
            model=pipeline,
            sample_input_data=df[input_features],
            prediction_assert_fns=prediction_assert_fns,
            options={"enable_explainability": enable_explainability},
            signatures=expected_signatures,
        )

    def test_skl_KDensity_model(self) -> None:

        # Generate sample data
        X = np.arange(0, 10)[:, np.newaxis]

        # Instantiate and fit the Kernel Density Estimator
        kde = neighbors.KernelDensity(kernel="gaussian", bandwidth=0.5)
        kde.fit(X)

        expected_signatures = {
            "score_samples": model_signature.infer_signature(
                X,
                kde.score_samples(X),
            ),
        }

        def _check_score_samples(res: pd.DataFrame) -> None:
            expected = kde.score_samples(X)
            pd.testing.assert_series_equal(
                res["output_feature_0"],
                pd.Series(expected, name="output_feature_0"),
                check_dtype=False,
            )

        self._test_registry_model(
            model=kde,
            sample_input_data=X,
            prediction_assert_fns={
                "score_samples": (
                    X,
                    _check_score_samples,
                ),
            },
            signatures=expected_signatures,
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
        enable_explainability=[False, None],  # Explainability not yet supported
    )
    def test_scaler_random_forest_pipeline(self, registry_test_fn: str, enable_explainability: Optional[bool]) -> None:
        X, y = datasets.load_iris(return_X_y=True)
        pipeline = SK_pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", ensemble.RandomForestClassifier(random_state=42, n_jobs=1)),
            ]
        )
        pipeline.fit(X, y)

        getattr(self, registry_test_fn)(
            model=pipeline,
            sample_input_data=X,
            prediction_assert_fns={
                "predict": (
                    X,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(pipeline.predict(X), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": enable_explainability},
        )

    def test_skl_model_with_quoted_identifiers_ignore_case(self):
        X, y = datasets.load_iris(return_X_y=True)
        X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        model = linear_model.LogisticRegression()
        model.fit(X_df, y)

        name = "model_test_skl_quoted_identifiers_param"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=X_df,
        )

        statement_params = {"QUOTED_IDENTIFIERS_IGNORE_CASE": "TRUE"}

        functions = mv._functions
        predict_proba_name = sql_identifier.SqlIdentifier("predict_proba").identifier()
        find_method: Callable[[model_manifest_schema.ModelFunctionInfo], bool] = (
            lambda method: method["name"] == predict_proba_name
        )
        target_function_info = next(
            filter(find_method, functions),
            None,
        )
        self.assertIsNotNone(target_function_info, "predict_proba function not found")

        result = mv._model_ops.invoke_method(
            method_name=sql_identifier.SqlIdentifier(target_function_info["name"]),
            method_function_type=target_function_info["target_method_function_type"],
            signature=target_function_info["signature"],
            X=X_df,
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

        predict_name = sql_identifier.SqlIdentifier("predict").identifier()
        find_method: Callable[[model_manifest_schema.ModelFunctionInfo], bool] = (
            lambda method: method["name"] == predict_name
        )
        target_function_info = next(
            filter(find_method, functions),
            None,
        )
        self.assertIsNotNone(target_function_info, "predict function not found")

        predict_result = mv._model_ops.invoke_method(
            method_name=sql_identifier.SqlIdentifier(target_function_info["name"]),
            method_function_type=target_function_info["target_method_function_type"],
            signature=target_function_info["signature"],
            X=X_df,
            database_name=None,
            schema_name=None,
            model_name=mv._model_name,
            version_name=mv._version_name,
            strict_input_validation=False,
            statement_params=statement_params,
            is_partitioned=target_function_info["is_partitioned"],
        )

        predict_cols = list(predict_result.columns)
        for col in predict_cols:
            self.assertTrue(
                col.isupper(), f"Expected column {col} to be uppercase with QUOTED_IDENTIFIERS_IGNORE_CASE=TRUE"
            )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0, "Result should not be empty")
        self.assertIsInstance(predict_result, pd.DataFrame)
        self.assertTrue(len(predict_result) > 0, "Predict result should not be empty")

        self.registry.delete_model(model_name=name)


if __name__ == "__main__":
    absltest.main()
