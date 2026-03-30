import torch
from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._signatures import pytorch_handler
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryPytorchModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    def test_pytorch_basic(self) -> None:
        """Basic PyTorch model deployment with inferred signature."""

        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                return self.linear(tensor)

        model = SimpleModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=data_x,
            prediction_assert_fns={
                "forward": (
                    x_df[:10],
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res),
                        y_pred[:10],
                        check_dtype=False,
                    ),
                ),
            },
            skip_rest_api_test=True,
        )

    def test_pytorch_with_params_forwarding(self) -> None:
        """Params are forwarded to a PyTorch model's forward method via inference service."""

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
        y_scaled = model.forward(data_x, scale=2.0).detach()

        sig = model_signature.infer_signature(
            data_x,
            y_scaled,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=data_x,
            signatures={"forward": sig},
            prediction_assert_fns={
                "forward": (
                    x_df[:10],
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res),
                        y_scaled[:10],
                        check_dtype=False,
                    ),
                ),
            },
            params={"scale": 2.0},
            skip_rest_api_test=True,
        )


if __name__ == "__main__":
    absltest.main()
