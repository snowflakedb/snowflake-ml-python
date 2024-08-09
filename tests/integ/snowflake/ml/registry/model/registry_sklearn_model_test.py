from typing import cast

import numpy as np
import pandas as pd
import shap
from absl.testing import absltest, parameterized
from sklearn import datasets, ensemble, linear_model, multioutput

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
        # LogisticRegression is for classfication task, such as iris
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X, iris_y)
        getattr(self, registry_test_fn)(
            model=classifier,
            sample_input_data=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: np.testing.assert_allclose(res["output_feature_0"].values, classifier.predict(iris_X)),
                ),
                "predict_proba": (
                    iris_X[:10],
                    lambda res: np.testing.assert_allclose(res.values, classifier.predict_proba(iris_X[:10])),
                ),
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

        with self.assertRaisesRegex(
            ValueError,
            "Sample input data is required to enable explainability. Currently we only support this for "
            + "`pandas.DataFrame` and `snowflake.snowpark.dataframe.DataFrame`.",
        ):
            getattr(self, registry_test_fn)(
                model=classifier,
                sample_input_data=iris_X,
                prediction_assert_fns={},
                options={"enable_explainability": True},
            )

        getattr(self, registry_test_fn)(
            model=classifier,
            sample_input_data=iris_X_df,
            prediction_assert_fns={
                "predict": (
                    iris_X_df,
                    lambda res: np.testing.assert_allclose(res["output_feature_0"].values, classifier.predict(iris_X)),
                ),
                "predict_proba": (
                    iris_X_df.iloc[:10],
                    lambda res: np.testing.assert_allclose(res.values, classifier.predict_proba(iris_X[:10])),
                ),
                "explain": (
                    iris_X_df,
                    lambda res: np.testing.assert_allclose(
                        dataframe_utils.convert2D_json_to_3D(res.values), expected_explanations
                    ),
                ),
            },
            options={"enable_explainability": True},
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
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_skl_model_case_sensitive(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
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
                    lambda res: np.testing.assert_allclose(res["output_feature_0"].values, regr.predict(iris_X)),
                ),
                '"predict_proba"': (
                    iris_X[:10],
                    lambda res: np.testing.assert_allclose(res.values, regr.predict_proba(iris_X[:10])),
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
                    lambda res: np.testing.assert_allclose(res.to_numpy(), model.predict(iris_X[-10:])),
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
        np.testing.assert_allclose(res.to_numpy(), model.predict(iris_X[-10:]))

        res = mv.run(iris_X[-10:], function_name="predict_proba")
        np.testing.assert_allclose(
            np.hstack([np.array(res[col].to_list()) for col in cast(pd.DataFrame, res)]),
            np.hstack(model.predict_proba(iris_X[-10:])),
        )

        with self.assertRaises(snowpark_exceptions.SnowparkSQLException):
            mv.run(iris_X_df, function_name="explain")

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])


if __name__ == "__main__":
    absltest.main()
