from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, svm

from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)


class TestExecuteInferenceJobServiceSklearnInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
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
        input_df, expected_predictions = self._prepare_batch_inference_data(iris_df, model_output_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_execute_inference_job_service(
            model=svc,
            sample_input_data=iris_X,
            pip_requirements=pip_requirements,
            options={"enable_explainability": False},
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            inference_spec=batch_inference_specs.Inference(num_workers=1),
            function_name="predict",
            job_name=job_name,
            replicas=1,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
