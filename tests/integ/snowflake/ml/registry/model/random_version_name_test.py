import numpy as np
from absl.testing import absltest
from sklearn import datasets, linear_model

from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class RandomVersionNameTest(registry_model_test_base.RegistryModelTestBase):
    def test_random_version_name(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        name = f"model_{self._run_id}"
        mv = self.registry.log_model(regr, model_name=name, sample_input_data=iris_X)
        np.testing.assert_allclose(
            mv.run(iris_X, function_name="predict")["output_feature_0"].values, regr.predict(iris_X)
        )

        self.registry._model_manager._hrid_generator.hrid_to_id(mv.version_name.lower())


if __name__ == "__main__":
    absltest.main()
