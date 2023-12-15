from typing import cast

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, ensemble, linear_model, multioutput

from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistrySKLearnModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_skl_model(
        self,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        self._test_registry_model(
            model=regr,
            sample_input=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: np.testing.assert_allclose(res["output_feature_0"].values, regr.predict(iris_X)),
                ),
                "predict_proba": (
                    iris_X[:10],
                    lambda res: np.testing.assert_allclose(res.values, regr.predict_proba(iris_X[:10])),
                ),
            },
        )

    def test_skl_model_case_sensitive(
        self,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        self._test_registry_model(
            model=regr,
            sample_input=iris_X,
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

    def test_skl_multiple_output_model(
        self,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        model.fit(iris_X[:10], dual_target[:10])
        self._test_registry_model(
            model=model,
            sample_input=iris_X,
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


if __name__ == "__main__":
    absltest.main()
