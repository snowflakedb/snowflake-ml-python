import os
import tempfile
import warnings
from typing import Tuple

import numpy as np
import torch
from absl.testing import absltest

from snowflake.ml.model import model_signature
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
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
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
    def test_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_torch_model()
            s = {"forward": model_signature.infer_signature([data_x], [data_y])}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**s, "another_forward": s["forward"]},
                    metadata={"author": "halu", "version": "1"},
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

    def test_torch_df_sample_input(self) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        model_script = torch.jit.script(model)
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

            model_script.eval()
            y_pred = model_script.forward(data_x).detach()

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2")).save(
                name="model1_no_sig_2",
                model=model_script,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, torch.jit.ScriptModule)
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
