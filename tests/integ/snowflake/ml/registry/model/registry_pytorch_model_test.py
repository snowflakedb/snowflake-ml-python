from typing import Optional

import pandas as pd
import torch
from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._signatures import pytorch_handler, snowpark_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import (
    dataframe_utils,
    model_factory,
    test_env_utils,
)


class TestRegistryPytorchModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_pytorch_tensor_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float32)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model(
            model=model,
            sample_input_data=data_x,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred, check_dtype=False
                    ),
                ),
            },
        )

    def test_pytorch_df_as_sample(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred
                    ),
                ),
            },
        )

    def test_pytorch_sp(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]
        y_pred = model.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(y_pred)
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
        )

    def test_torchscript_tensor_as_sample(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float32)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        self._test_registry_model(
            model=model_script,
            sample_input_data=data_x,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred, check_dtype=False
                    ),
                ),
            },
        )

    def test_torchscript_df_as_sample(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        self._test_registry_model(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred
                    ),
                ),
            },
        )

    def test_torchscript_sp(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(y_pred)
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self._test_registry_model(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
        )

    def test_pytorch_tensor_as_sample_multiple_inputs(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float32)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_pytorch_df_as_sample_multiple_inputs(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_pytorch_sp_multiple_inputs(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_torchscript_tensor_as_sample_multiple_inputs(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float32)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        self._test_registry_model(
            model=model_script,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_torchscript_df_as_sample_multiple_inputs(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        self._test_registry_model(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_torchscript_sp_multiple_inputs(self) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self._test_registry_model(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_pytorch_with_params_forwarding(self) -> None:
        """Params are forwarded to a PyTorch model's forward method."""

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
        y_default = model.forward(data_x).detach()
        y_scaled = model.forward(data_x, scale=2.0).detach()

        sig = model_signature.infer_signature(
            data_x,
            y_default,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        name = "model_test_pytorch_params_forwarding"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res_default = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_default),
            y_default[:10],
            check_dtype=False,
        )

        res_scaled = mv.run(x_df[:10], function_name="forward", params={"scale": 2.0})
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_scaled),
            y_scaled[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    # ------------------------------------------------------------------ #
    # Forward pattern tests: previously supported                        #
    # ------------------------------------------------------------------ #

    def test_forward_extra_param_uses_default(self) -> None:
        """forward(self, x: Tensor, scale: float = 1.0) — without ParamSpec, the
        extra param silently uses its default. The user cannot change it."""

        class DefaultModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                return self.linear(x) * scale

        model = DefaultModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_default = model.forward(data_x).detach()

        sig = model_signature.infer_signature(data_x, y_default)

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_default"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res),
            y_default[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    # ------------------------------------------------------------------ #
    # Forward pattern tests: newly supported (kwargs forwarding)         #
    # ------------------------------------------------------------------ #

    def test_forward_positional_scalar_param(self) -> None:
        """forward(self, x: Tensor, scale: float = 1.0) — positional param with
        default, forwarded as a keyword argument from ParamSpec."""

        class PositionalParamModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                return self.linear(x) * scale

        model = PositionalParamModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_default = model.forward(data_x).detach()
        y_scaled = model.forward(data_x, scale=3.0).detach()

        sig = model_signature.infer_signature(
            data_x,
            y_default,
            params=[model_signature.ParamSpec("scale", model_signature.DataType.DOUBLE, 1.0)],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_positional"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res_default = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_default),
            y_default[:10],
            check_dtype=False,
        )

        res_scaled = mv.run(x_df[:10], function_name="forward", params={"scale": 3.0})
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_scaled),
            y_scaled[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def test_forward_kwargs_passthrough(self) -> None:
        """forward(self, x: Tensor, **kwargs) — model accepts **kwargs and receives
        scalar params declared in ParamSpec."""

        class KwargsModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
                scale = float(kwargs.get("scale", 1.0))
                return self.linear(x) * scale

        model = KwargsModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_default = model.forward(data_x).detach()
        y_scaled = model.forward(data_x, scale=4.0).detach()

        sig = model_signature.infer_signature(
            data_x,
            y_default,
            params=[model_signature.ParamSpec("scale", model_signature.DataType.DOUBLE, 1.0)],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_kwargs"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res_default = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_default),
            y_default[:10],
            check_dtype=False,
        )

        res_scaled = mv.run(x_df[:10], function_name="forward", params={"scale": 4.0})
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_scaled),
            y_scaled[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def test_forward_multiple_scalar_params(self) -> None:
        """forward(self, x: Tensor, *, scale: float = 1.0, offset: float = 0.0) —
        multiple scalar params forwarded simultaneously."""

        class MultiParamModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, x: torch.Tensor, *, scale: float = 1.0, offset: float = 0.0) -> torch.Tensor:
                return self.linear(x) * scale + offset

        model = MultiParamModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_default = model.forward(data_x).detach()
        y_custom = model.forward(data_x, scale=2.0, offset=5.0).detach()

        sig = model_signature.infer_signature(
            data_x,
            y_default,
            params=[
                model_signature.ParamSpec("scale", model_signature.DataType.DOUBLE, 1.0),
                model_signature.ParamSpec("offset", model_signature.DataType.DOUBLE, 0.0),
            ],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_multi"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res_default = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_default),
            y_default[:10],
            check_dtype=False,
        )

        res_custom = mv.run(x_df[:10], function_name="forward", params={"scale": 2.0, "offset": 5.0})
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_custom),
            y_custom[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def test_forward_array_param(self) -> None:
        """forward(self, x: Tensor, *, weights=None) — a 1D array parameter
        forwarded via ParamSpec with shape=(-1,). Demonstrates that ParamSpec
        supports n-dimensional data, not just scalars."""

        class WeightedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 3, dtype=torch.float64)

            def forward(self, x: torch.Tensor, *, weights: Optional[list[float]] = None) -> torch.Tensor:
                out = self.linear(x)
                if weights is not None:
                    out = out * torch.tensor(weights, dtype=out.dtype)
                return out

        model = WeightedModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_default = model.forward(data_x).detach()
        custom_weights = [2.0, 0.5, 1.0]
        y_weighted = model.forward(data_x, weights=custom_weights).detach()

        sig = model_signature.infer_signature(
            data_x,
            y_default,
            params=[
                model_signature.ParamSpec("weights", model_signature.DataType.DOUBLE, [1.0, 1.0, 1.0], shape=(-1,)),
            ],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_array_param"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res_default = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_default),
            y_default[:10],
            check_dtype=False,
        )

        res_weighted = mv.run(x_df[:10], function_name="forward", params={"weights": custom_weights})
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_weighted),
            y_weighted[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def test_forward_2d_param_transform_matrix(self) -> None:
        """forward(self, x: Tensor, *, transform=None) — a 2D parameter (3x3
        transformation matrix) forwarded via ParamSpec with shape=(-1,-1). The
        matrix is applied to each row independently, so the output shape does
        not depend on the batch size."""

        class TransformModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 3, dtype=torch.float64)

            def forward(self, x: torch.Tensor, *, transform: Optional[list[list[float]]] = None) -> torch.Tensor:
                out = self.linear(x)  # (batch, 3)
                if transform is not None:
                    t = torch.tensor(transform, dtype=out.dtype)  # (3, 3)
                    out = out @ t
                return out

        model = TransformModel()
        model.eval()

        data_x = torch.rand(100, 10, dtype=torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_default = model.forward(data_x).detach()

        # Identity-ish transform that doubles first column and zeros last
        custom_transform = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        y_transformed = model.forward(data_x, transform=custom_transform).detach()

        # Default is identity (no transform applied)
        sig = model_signature.infer_signature(
            data_x,
            y_default,
            params=[
                model_signature.ParamSpec("transform", model_signature.DataType.DOUBLE, None, shape=(-1, -1)),
            ],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_2d_param"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res_default = mv.run(x_df[:10], function_name="forward")
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_default),
            y_default[:10],
            check_dtype=False,
        )

        res_transformed = mv.run(x_df[:10], function_name="forward", params={"transform": custom_transform})
        torch.testing.assert_close(
            pytorch_handler.PyTorchTensorHandler.convert_from_df(res_transformed),
            y_transformed[:10],
            check_dtype=False,
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    # ------------------------------------------------------------------ #
    # Forward pattern tests: not yet supported                           #
    # ------------------------------------------------------------------ #

    def test_forward_per_row_tensor_kwarg_not_supported(self) -> None:
        """forward(self, src: Tensor, mask: Optional[Tensor] = None) — per-row
        tensor kwargs (where each sample has its own mask) cannot be passed.

        Batch-level masks (same for all rows) CAN be passed via ParamSpec with
        shape (see test_forward_batch_level_mask_via_param). But per-row masks
        that vary per sample have no delivery mechanism: ParamSpec is one value
        for the entire batch, and the tensor data pipeline only passes tensors
        positionally with no named routing."""

        class MaskedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                out = self.linear(src)
                if mask is not None:
                    out = out * mask.unsqueeze(-1).to(out.dtype)
                return out

        model = MaskedModel()
        model.eval()

        data_x = torch.rand(50, 10, dtype=torch.float64)
        # Per-row mask: different for each sample in the batch
        mask = torch.ones(50, dtype=torch.bool)
        mask[::2] = False

        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_no_mask = model.forward(data_x).detach()
        y_with_mask = model.forward(data_x, mask=mask).detach()

        sig = model_signature.infer_signature(data_x, y_no_mask)

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        name = "model_fwd_per_row_mask"
        version = f"ver_{self._run_id}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=data_x,
            conda_dependencies=conda_dependencies,
            signatures={"forward": sig},
            options={"embed_local_ml_library": True},
        )

        res = mv.run(x_df[:10], function_name="forward")
        res_tensor = pytorch_handler.PyTorchTensorHandler.convert_from_df(res)

        # Handler can only produce the mask=None result.
        torch.testing.assert_close(res_tensor, y_no_mask[:10], check_dtype=False)

        # The masked result is different — proving the per-row mask path is unreachable.
        self.assertFalse(
            torch.allclose(y_no_mask[:10], y_with_mask[:10]),
            "Masked and unmasked outputs should differ, confirming the mask has an effect.",
        )

        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def test_forward_required_non_tensor_arg_not_supported(self) -> None:
        """forward(self, x: Tensor, n: int) — a required non-tensor argument with
        no default. Signature inference calls forward(tensor) which raises TypeError."""

        class RequiredArgModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
                return self.linear(x) * n

        model = RequiredArgModel()
        model.eval()

        data_x = torch.rand(50, 10, dtype=torch.float64)

        with self.assertRaises(TypeError):
            self.registry.log_model(
                model=model,
                model_name="model_fwd_req",
                version_name=f"ver_{self._run_id}",
                sample_input_data=data_x,
                options={"embed_local_ml_library": True},
            )

    def test_forward_dict_output_not_supported(self) -> None:
        """forward(self, x: Tensor) -> dict[str, Tensor] — dict return types are
        not handled by the output conversion pipeline. Signature inference fails
        with SnowflakeMLException(NotImplementedError)."""

        class DictOutputModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Linear(10, 5, dtype=torch.float64)
                self.head = torch.nn.Linear(10, 1, dtype=torch.float64)

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"logits": self.head(x), "embedding": self.encoder(x)}

        model = DictOutputModel()
        model.eval()

        data_x = torch.rand(50, 10, dtype=torch.float64)

        with self.assertRaisesRegex(Exception, "Un-supported type provided"):
            self.registry.log_model(
                model=model,
                model_name="model_fwd_dict",
                version_name=f"ver_{self._run_id}",
                sample_input_data=data_x,
                options={"embed_local_ml_library": True},
            )


if __name__ == "__main__":
    absltest.main()
