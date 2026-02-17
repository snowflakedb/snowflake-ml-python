import pandas as pd
import torch
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import pytorch_handler
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import model_factory


class TestPyTorchBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None},
        {"gpu_requests": "1", "cpu_requests": None, "memory_requests": None},
    )
    def test_pt(
        self,
        gpu_requests: str,
        cpu_requests: str,
        memory_requests: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]

        # Generate expected predictions using the original model
        model_output = model.forward(data_x)
        model_output_df = pd.DataFrame({"output_feature_0": model_output.detach().numpy().flatten()})

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(x_df, model_output_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=x_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                gpu_requests=gpu_requests,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
                num_workers=1,
                replicas=2,
                function_name="forward",
            ),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
