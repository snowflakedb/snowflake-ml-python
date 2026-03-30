import os
import tempfile
import warnings

import numpy as np
import torch
from absl.testing import absltest

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._signatures import (
    pytorch_handler,
    utils as model_signature_utils,
)


class TorchModel(torch.nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_out, dtype=dtype),
            torch.nn.Sigmoid(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)  # type: ignore[no-any-return]


def _prepare_torch_model(
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = torch.from_numpy(x).to(dtype=dtype)
    data_y = (torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)

    model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _epoch in range(100):
        pred_y = model(data_x)
        loss = loss_function(pred_y, data_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, data_x, data_y


class PyTorchHandlerTest(absltest.TestCase):
    def test_pytorch_single_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_torch_model()
            s = {"forward": model_signature.infer_signature(data_x, data_y)}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**s, "another_forward": s["forward"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.PyTorchSaveOptions(),
                )

            with self.assertRaises(NotImplementedError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.PyTorchSaveOptions(),
            )

            model.eval()
            y_pred = model.forward(data_x).detach()

            x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x)

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, torch.nn.Module)
                torch.testing.assert_close(pk.model.forward(data_x), y_pred)

                with self.assertRaisesRegex(RuntimeError, "Attempting to deserialize object on a CUDA device"):
                    pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                    pk.load(options={"use_gpu": True})

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "forward", None)
                assert callable(predict_method)
                torch.testing.assert_close(
                    pytorch_handler.PyTorchTensorHandler.convert_from_df(predict_method(x_df)),
                    y_pred,
                    check_dtype=False,
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=model,
                sample_input_data=data_x,
                metadata={"author": "halu", "version": "1"},
                options=model_types.PyTorchSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.nn.Module)
            torch.testing.assert_close(pk.model.forward(data_x), y_pred)
            self.assertEqual(s["forward"], pk.meta.signatures["forward"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(
                pytorch_handler.PyTorchTensorHandler.convert_from_df(predict_method(x_df)),
                y_pred,
                check_dtype=False,
            )

    def test_torch_df_sample_input_single_input(self) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        s = {"forward": model_signature.infer_signature(data_x, data_y)}

        with tempfile.TemporaryDirectory() as tmpdir:
            model.eval()
            y_pred = model.forward(data_x).detach()

            x_df = model_signature_utils.rename_pandas_df(
                pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False),
                s["forward"].inputs,
            )
            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=model,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.PyTorchSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.nn.Module)
            torch.testing.assert_close(pk.model.forward(data_x), y_pred)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(
                pytorch_handler.PyTorchTensorHandler.convert_from_df(predict_method(x_df)),
                y_pred,
                check_dtype=False,
            )

            model.eval()
            y_pred = model.forward(data_x).detach()

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2")).save(
                name="model1_no_sig_2",
                model=model,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.PyTorchSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.nn.Module)
            torch.testing.assert_close(pk.model.forward(data_x), y_pred)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(
                pytorch_handler.PyTorchTensorHandler.convert_from_df(predict_method(x_df)),
                y_pred,
                check_dtype=False,
            )

    def test_pytorch_multiple_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_torch_model()
            s = {"forward": model_signature.infer_signature([data_x], [data_y])}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**s, "another_forward": s["forward"]},
                    metadata={"author": "halu", "version": "1"},
                    options={"multiple_inputs": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options={"multiple_inputs": True},
            )

            model.eval()
            y_pred = model.forward(data_x).detach()

            x_df = model_signature_utils.rename_pandas_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False),
                s["forward"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, torch.nn.Module)
                torch.testing.assert_close(pk.model.forward(data_x), y_pred)

                with self.assertRaisesRegex(RuntimeError, "Attempting to deserialize object on a CUDA device"):
                    pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                    pk.load(options={"use_gpu": True})

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "forward", None)
                assert callable(predict_method)
                torch.testing.assert_close(
                    pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                        predict_method(x_df), s["forward"].outputs
                    )[0],
                    y_pred,
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=model,
                sample_input_data=[data_x],
                metadata={"author": "halu", "version": "1"},
                options={"multiple_inputs": True},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.nn.Module)
            torch.testing.assert_close(pk.model.forward(data_x), y_pred)
            self.assertEqual(s["forward"], pk.meta.signatures["forward"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df), s["forward"].outputs)[
                    0
                ],
                y_pred,
            )

    def test_pytorch_single_input_with_kwargs(self) -> None:
        """Generated fn accepts **method_kwargs and still produces correct predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_torch_model()
            s = {"forward": model_signature.infer_signature(data_x, data_y)}

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.PyTorchSaveOptions(),
            )

            model.eval()
            y_pred = model.forward(data_x).detach()
            x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)

            torch.testing.assert_close(
                pytorch_handler.PyTorchTensorHandler.convert_from_df(predict_method(x_df)),
                y_pred,
                check_dtype=False,
            )

            # Unknown kwargs are forwarded to the model, which rejects them
            with self.assertRaises(TypeError):
                predict_method(x_df, some_param=42)

    def test_pytorch_forward_with_custom_param(self) -> None:
        """Kwargs are forwarded to the model's forward method."""

        class ScaledModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, tensor: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
                result: torch.Tensor = self.linear(tensor) * scale
                return result

        model = ScaledModel()
        n_input, batch_size = 10, 20
        data_x = torch.rand(batch_size, n_input)
        model.eval()
        y_default = model.forward(data_x).detach()
        y_scaled = model.forward(data_x, scale=2.0).detach()

        sig = model_signature.ModelSignature(
            inputs=model_signature.infer_signature(data_x, y_default).inputs,
            outputs=model_signature.infer_signature(data_x, y_default).outputs,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures={"forward": sig},
                options=model_types.PyTorchSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            forward_method = getattr(pk.model, "forward", None)
            assert callable(forward_method)

            x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x)

            res_default = pytorch_handler.PyTorchTensorHandler.convert_from_df(forward_method(x_df))
            torch.testing.assert_close(res_default, y_default, check_dtype=False)

            res_scaled = pytorch_handler.PyTorchTensorHandler.convert_from_df(forward_method(x_df, scale=2.0))
            torch.testing.assert_close(res_scaled, y_scaled, check_dtype=False)

    def test_pytorch_forward_with_untyped_positional_param(self) -> None:
        """Params forwarded to forward with untyped positional parameter (no type hint, no default)."""

        class ScaledModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                result: torch.Tensor = self.linear(tensor) * scale
                return result

        model = ScaledModel()
        data_x = torch.rand(20, 10)
        model.eval()
        y_default = model.forward(data_x).detach()
        y_scaled = model.forward(data_x, scale=3.0).detach()

        sig = model_signature.ModelSignature(
            inputs=model_signature.infer_signature(data_x, y_default).inputs,
            outputs=model_signature.infer_signature(data_x, y_default).outputs,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures={"forward": sig},
                options=model_types.PyTorchSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            forward_method = getattr(pk.model, "forward", None)
            assert callable(forward_method)

            x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x)

            res_default = pytorch_handler.PyTorchTensorHandler.convert_from_df(forward_method(x_df))
            torch.testing.assert_close(res_default, y_default, check_dtype=False)

            res_scaled = pytorch_handler.PyTorchTensorHandler.convert_from_df(forward_method(x_df, scale=3.0))
            torch.testing.assert_close(res_scaled, y_scaled, check_dtype=False)

    def test_pytorch_forward_with_kwargs_passthrough(self) -> None:
        """Params forwarded to a model whose forward uses **kwargs."""

        class FlexibleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, tensor: torch.Tensor, **kwargs: float) -> torch.Tensor:
                scale = kwargs.get("scale", 1.0)
                result: torch.Tensor = self.linear(tensor) * scale
                return result

        model = FlexibleModel()
        data_x = torch.rand(20, 10)
        model.eval()
        y_default = model.forward(data_x).detach()
        y_scaled = model.forward(data_x, scale=2.0).detach()

        sig = model_signature.ModelSignature(
            inputs=model_signature.infer_signature(data_x, y_default).inputs,
            outputs=model_signature.infer_signature(data_x, y_default).outputs,
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures={"forward": sig},
                options=model_types.PyTorchSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            forward_method = getattr(pk.model, "forward", None)
            assert callable(forward_method)

            x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x)

            res_default = pytorch_handler.PyTorchTensorHandler.convert_from_df(forward_method(x_df))
            torch.testing.assert_close(res_default, y_default, check_dtype=False)

            res_scaled = pytorch_handler.PyTorchTensorHandler.convert_from_df(forward_method(x_df, scale=2.0))
            torch.testing.assert_close(res_scaled, y_scaled, check_dtype=False)

    def test_pytorch_use_gpu_load_time_closure(self) -> None:
        """use_gpu is a load-time option captured in closure, not an inference-time kwarg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_torch_model()
            s = {"forward": model_signature.infer_signature(data_x, data_y)}

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.PyTorchSaveOptions(),
            )

            model.eval()
            y_pred = model.forward(data_x).detach()
            x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x)

            # Load without use_gpu (default False) — should work on CPU
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)

            result = predict_method(x_df)
            torch.testing.assert_close(
                pytorch_handler.PyTorchTensorHandler.convert_from_df(result),
                y_pred,
                check_dtype=False,
            )

            # Verify use_gpu=True at load time still causes CUDA error (no CUDA in test)
            with self.assertRaisesRegex(RuntimeError, "Attempting to deserialize object on a CUDA device"):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(options={"use_gpu": True})

    def test_torch_df_sample_input_multiple_inputs(self) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        s = {"forward": model_signature.infer_signature([data_x], [data_y])}

        with tempfile.TemporaryDirectory() as tmpdir:
            model.eval()
            y_pred = model.forward(data_x).detach()

            x_df = model_signature_utils.rename_pandas_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False),
                s["forward"].inputs,
            )
            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=model,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
                options={"multiple_inputs": True},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.nn.Module)
            torch.testing.assert_close(pk.model.forward(data_x), y_pred)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df))[0], y_pred
            )

            model.eval()
            y_pred = model.forward(data_x).detach()

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2")).save(
                name="model1_no_sig_2",
                model=model,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
                options={"multiple_inputs": True},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.nn.Module)
            torch.testing.assert_close(pk.model.forward(data_x), y_pred)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df))[0], y_pred
            )


if __name__ == "__main__":
    absltest.main()
