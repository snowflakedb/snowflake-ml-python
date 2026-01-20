import uuid

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import tensorflow_handler
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import model_factory


class TestTensorFlowBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
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
        input_spec, expected_predictions = self._prepare_batch_inference_data(x_df, model_output_df)

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=x_df,
            X=input_spec,
            output_stage_location=output_stage_location,
            gpu_requests=gpu_requests,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            num_workers=1,
            service_name=f"batch_inference_{name}",
            replicas=2,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
