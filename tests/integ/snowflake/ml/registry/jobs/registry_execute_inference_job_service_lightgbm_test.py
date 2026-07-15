import inflection
import lightgbm
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)


class TestExecuteInferenceJobServiceLightGbmInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
    @parameterized.parameters(  # type: ignore[misc]
        {"cpu_requests": None, "replicas": 1},
        {"cpu_requests": "3", "replicas": 2},
    )
    def test_lightgbm_batch_inference(
        self,
        replicas: int,
        cpu_requests: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)

        # Generate expected predictions using the original model
        model_output = classifier.predict(cal_X_test)
        model_output_df = pd.DataFrame({"output_feature_0": model_output})

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(cal_X_test, model_output_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_execute_inference_job_service(
            model=classifier,
            sample_input_data=cal_X_test,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            resources_spec=batch_inference_specs.Resources(cpu_requests=cpu_requests),
            inference_spec=batch_inference_specs.Inference(num_workers=2),
            function_name="predict",
            job_name=job_name,
            replicas=replicas,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
