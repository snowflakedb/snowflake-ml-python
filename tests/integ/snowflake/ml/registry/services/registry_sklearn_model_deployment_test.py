from typing import List, Optional

import pandas as pd
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
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(svc.predict(iris_X), columns=res.columns),
                        rtol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            pip_requirements=pip_requirements,
            options={"enable_explainability": False},
        )


if __name__ == "__main__":
    absltest.main()
