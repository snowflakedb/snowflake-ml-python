import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._client.model import batch_inference_specs
from snowflake.ml.model._signatures import tensorflow_handler
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)
from tests.integ.snowflake.ml.test_utils import model_factory


class TestExecuteInferenceJobServiceTensorFlowInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
    @absltest.skip("SNOW-3691662")
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None},
        {"gpu_requests": "1", "cpu_requests": None, "memory_requests": None},
    )
    def test_tf(
        self,
        gpu_requests: str,
        cpu_requests: str,
        memory_requests: str,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(x_df.shape[1])]

        # Generate expected predictions using the original model
        model_output = model(data_x)
        model_output_df = pd.DataFrame({"output_feature_0": model_output.numpy().flatten()})

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(x_df, model_output_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=x_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            resources_spec=batch_inference_specs.Resources(
                gpu_requests=gpu_requests,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
            ),
            inference_spec=batch_inference_specs.Inference(num_workers=1),
            job_name=job_name,
            replicas=2,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
