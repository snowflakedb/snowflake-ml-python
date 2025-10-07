import uuid
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, svm

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistrySklearnBatchInferenceInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters({"pip_requirements": None}, {"pip_requirements": ["scikit-learn"]})  # type: ignore[misc]
    def test_sklearn(self, pip_requirements: Optional[list[str]]) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        svc = svm.LinearSVC()
        svc.fit(iris_X, iris_y)

        # Convert numpy array to pandas DataFrame for create_dataframe
        iris_df = pd.DataFrame(iris_X, columns=[f"input_feature_{i}" for i in range(iris_X.shape[1])])

        # Generate expected predictions using the original model
        model_output = svc.predict(iris_X)
        model_output_df = pd.DataFrame({"output_feature_0": model_output})

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(iris_df, model_output_df)

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=svc,
            sample_input_data=iris_X,
            pip_requirements=pip_requirements,
            options={"enable_explainability": False},
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            num_workers=1,
            service_name=f"batch_inference_{name}",
            replicas=1,
            function_name="predict",
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
