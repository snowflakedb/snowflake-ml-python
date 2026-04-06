import pandas as pd
import torch
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.model._signatures import pytorch_handler
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
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

    def test_pytorch_with_params_forwarding(self) -> None:
        """Params are forwarded to a PyTorch model's forward method via batch inference."""

        class ScaledModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            # `scale` is the keyword argument under test: it should be forwarded
            # from ParamSpec through the handler to the model's forward method.
            def forward(self, tensor: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
                return self.linear(tensor) * scale

        model = ScaledModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]

        y_scaled = model.forward(data_x, scale=2.0).detach()
        model_output_df = pd.DataFrame({"output_feature_0": y_scaled.numpy().flatten()})

        input_df, expected_predictions = self._prepare_batch_inference_data(x_df, model_output_df)

        sig = model_signature.infer_signature(
            x_df,
            y_scaled,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=x_df,
            signatures={"forward": sig},
            X=input_df,
            input_spec=InputSpec(params={"scale": 2.0}),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                function_name="forward",
            ),
            expected_predictions=expected_predictions,
        )

    def test_torchscript_with_params_forwarding(self) -> None:
        """Params are forwarded to a TorchScript model's forward method via batch inference."""

        @torch.jit.script
        def scaled_forward(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float = 1.0
        ) -> torch.Tensor:
            return (x @ weight.t() + bias) * scale

        class ScaledModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            # `scale` is the keyword argument under test: it should be forwarded
            # from ParamSpec through the handler to the model's forward method.
            def forward(self, tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                return scaled_forward(tensor, self.linear.weight, self.linear.bias, scale)

        model = ScaledModel()
        model.eval()
        model_script = torch.jit.script(model)

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]

        y_scaled = model_script.forward(data_x, scale=2.0).detach()
        model_output_df = pd.DataFrame({"output_feature_0": y_scaled.numpy().flatten()})

        input_df, expected_predictions = self._prepare_batch_inference_data(x_df, model_output_df)

        sig = model_signature.infer_signature(
            x_df,
            y_scaled,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model_script,
            sample_input_data=x_df,
            signatures={"forward": sig},
            X=input_df,
            input_spec=InputSpec(params={"scale": 2.0}),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                function_name="forward",
            ),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
