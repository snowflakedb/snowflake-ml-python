from typing import List, Optional

import numpy as np
from absl.testing import absltest, parameterized
from sklearn import datasets, svm

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistrySklearnModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters({"pip_requirements": None}, {"pip_requirements": ["scikit-learn"]})  # type: ignore[misc]
    def test_sklearn(self, pip_requirements: Optional[List[str]]) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        svc = svm.LinearSVC()
        svc.fit(iris_X, iris_y)

        self._test_registry_model_deployment(
            model=svc,
            sample_input_data=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(svc.predict(iris_X), axis=1), rtol=1e-3
                    ),
                ),
            },
            pip_requirements=pip_requirements,
        )


if __name__ == "__main__":
    absltest.main()
