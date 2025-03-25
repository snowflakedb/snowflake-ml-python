from typing import cast

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

from snowflake.ml.model import model_signature
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils
from snowflake.snowpark import exceptions as snowpark_exceptions
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

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_model_explain(
        self,
        registry_test_fn: str,
    ) -> None:
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

        getattr(self, registry_test_fn)(
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

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_sklearn_explain_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        iris_X_sp_df = self.session.create_dataframe(iris_X_df)
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X_df, iris_y)

        explain_df = handlers_utils.convert_explanations_to_2D_df(
            classifier, shap.Explainer(classifier, iris_X_df)(iris_X_df).values
        ).set_axis([f"{c}_explanation" for c in iris_X_df.columns], axis=1)

        explanation_df_expected = pd.concat([iris_X_df, explain_df], axis=1)
        getattr(self, registry_test_fn)(
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

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_model_case_sensitive(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classification task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        getattr(self, registry_test_fn)(
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

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_multiple_output_model(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        model.fit(iris_X[:10], dual_target[:10])
        getattr(self, registry_test_fn)(
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
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        model.fit(iris_X[:10], dual_target[:10])
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        name = "model_test_skl_unsupported_explain"
        version = f"ver_{self._run_id}"
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=iris_X_df,
            conda_dependencies=conda_dependencies,
            options={"enable_explainability": True},
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

        with self.assertRaises(snowpark_exceptions.SnowparkSQLException):
            mv.run(iris_X_df, function_name="explain")

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_model_with_signature_and_sample_data(
        self,
        registry_test_fn: str,
    ) -> None:
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

        getattr(self, registry_test_fn)(
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
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_model_with_categorical_dtype_columns(
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

        getattr(self, registry_test_fn)(
            model=pipeline,
            sample_input_data=df[input_features],
            prediction_assert_fns={
                "predict": (
                    df[input_features],
                    lambda res: pd.testing.assert_series_equal(
                        res["output_feature_0"],
                        pd.Series(pipeline.predict(df[input_features]), name="output_feature_0"),
                        check_dtype=False,
                    ),
                ),
            },
            # TODO(SNOW-1677301): Add support for explainability for categorical columns
            options={"enable_explainability": False},
            signatures=expected_signatures,
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_KDensity_model(
        self,
        registry_test_fn: str,
    ) -> None:

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

        getattr(self, registry_test_fn)(
            model=kde,
            sample_input_data=X,
            prediction_assert_fns={
                "score_samples": (
                    X,
                    _check_score_samples,
                ),
            },
            options={"enable_explainability": False},
            signatures=expected_signatures,
        )


if __name__ == "__main__":
    absltest.main()
