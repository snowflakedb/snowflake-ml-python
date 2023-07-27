import asyncio
import importlib
import os
import sys
import tempfile
import uuid
import warnings
from typing import List, Tuple, cast
from unittest import mock

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import xgboost
from absl.testing import absltest
from sklearn import datasets, ensemble, linear_model, model_selection, multioutput

from snowflake.ml.model import (
    _model as model_api,
    custom_model,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._signatures import (
    pytorch_handler,
    tensorflow_handler,
    utils as model_signature_utils,
)
from snowflake.ml.modeling.linear_model import LinearRegression
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import FileOperation, Session


class DemoModelWithManyArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(os.path.join(context.path("bias"), "bias1"), encoding="utf-8") as f:
            v1 = int(f.read())
        with open(os.path.join(context.path("bias"), "bias2"), encoding="utf-8") as f:
            v2 = int(f.read())
        self.bias = v1 + v2

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class AnotherDemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(input[["c1", "c2"]])


class ComposeModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            (self.context.model_ref("m1").predict(input)["c1"] + self.context.model_ref("m2").predict(input)["output"])
            / 2,
            columns=["output"],
        )


class AsyncComposeModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    async def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        res1 = await self.context.model_ref("m1").predict.async_run(input)
        res_sum = res1["output"] + self.context.model_ref("m2").predict(input)["output"]
        return pd.DataFrame(res_sum / 2)


class DemoModelWithArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(context.path("bias"), encoding="utf-8") as f:
            v = int(f.read())
        self.bias = v

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class TorchModel(torch.nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_out, dtype=dtype),
            torch.nn.Sigmoid(),
        )

    def forward(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.model(tensors[0])]


def _prepare_torch_model(
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, List[torch.Tensor], List[torch.Tensor]]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = [torch.from_numpy(x).to(dtype=dtype)]
    data_y = [(torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)]

    model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _epoch in range(100):
        pred_y = model(data_x)
        loss = loss_function(pred_y[0], data_y[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, data_x, data_y


class SimpleModule(tf.Module):
    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    @tf.function  # type: ignore[misc]
    def __call__(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        return [self.a_variable * tensors[0] + self.non_trainable_variable]


class KerasModel(tf.keras.Model):
    def __init__(self, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
        self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

    def call(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        input = tensors[0]
        x = self.fc_1(input)
        x = self.fc_2(x)
        return [x]


def _prepare_keras_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf.keras.Model, List[tf.Tensor], List[tf.Tensor]]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = [tf.convert_to_tensor(x, dtype=dtype)]
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = [tf.cast(raw_data_y, dtype=dtype)]

    def loss_fn(y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        return tf.keras.losses.mse(y_true[0], y_pred[0])

    model = KerasModel(n_hidden, n_out)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=loss_fn)
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


PY_SRC = """\
def get_name():
    return __name__
def get_file():
    return __file__
"""


class ModelLoadHygieneTest(absltest.TestCase):
    def test_model_load_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as src_path:
                fake_mod_dirpath = os.path.join(src_path, "fake", "fake_module")
                os.makedirs(fake_mod_dirpath)

                py_file_path = os.path.join(fake_mod_dirpath, "p.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                sys.path.insert(0, src_path)

                from fake.fake_module import p

                self.assertEqual(p.__file__, py_file_path)

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(workspace, "model1"),
                    model=lm,
                    sample_input=d,
                    metadata={"author": "halu", "version": "1"},
                    code_paths=[os.path.join(src_path, "fake")],
                )

                print(list(os.walk(os.path.join(workspace, "model1"))))
                _ = model_api.load_model(model_dir_path=os.path.join(workspace, "model1"))
                from fake.fake_module import p

                self.assertEqual(p.__file__, os.path.join(workspace, "model1", "code", "fake", "fake_module", "p.py"))

                importlib.reload(p)
                self.assertEqual(p.__file__, py_file_path)
                sys.path.remove(src_path)

    def test_model_save_validation(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as src_path:
                fake_mod_dirpath = os.path.join(src_path, "snowflake", "fake_module")
                os.makedirs(fake_mod_dirpath)

                py_file_path = os.path.join(fake_mod_dirpath, "p.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                with self.assertRaises(ValueError):
                    model_api.save_model(
                        name="model1",
                        model_dir_path=os.path.join(workspace, "model1"),
                        model=lm,
                        sample_input=d,
                        metadata={"author": "halu", "version": "1"},
                        code_paths=[os.path.join(src_path, "snowflake")],
                    )

            with tempfile.TemporaryDirectory() as src_path:
                py_file_path = os.path.join(src_path, "snowflake.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                with self.assertRaises(ValueError):
                    model_api.save_model(
                        name="model1",
                        model_dir_path=os.path.join(workspace, "model1"),
                        model=lm,
                        sample_input=d,
                        metadata={"author": "halu", "version": "1"},
                        code_paths=[py_file_path],
                    )


class ModelInterfaceTest(absltest.TestCase):
    def test_save_interface(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        local_dir = "path/to/local/model/dir"
        stage_path = '@"db"."schema"."stage"/model.zip'

        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

        with self.assertRaisesRegex(
            ValueError, "model_dir_path and model_stage_file_path both cannot be None at the same time."
        ):
            model_api.save_model(name="model", model=linear_model.LinearRegression())  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "Session and model_stage_file_path must be specified at the same time."
        ):
            model_api.save_model(
                name="model", model=linear_model.LinearRegression(), session=c_session, sample_input=d
            )  # type:ignore[call-overload]

        with self.assertRaisesRegex(ValueError, "Session and model_stage_file_path must be None at the same time."):
            model_api.save_model(
                name="model", model=linear_model.LinearRegression(), model_stage_file_path=stage_path, sample_input=d
            )  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "Session and model_stage_file_path must be specified at the same time."
        ):
            model_api.save_model(
                name="model",
                model=linear_model.LinearRegression(),
                session=c_session,
                model_dir_path=local_dir,
                sample_input=d,
            )  # type:ignore[call-overload]

        with self.assertRaisesRegex(ValueError, "Session and model_stage_file_path must be None at the same time."):
            model_api.save_model(
                name="model",
                model=linear_model.LinearRegression(),
                model_stage_file_path=stage_path,
                model_dir_path=local_dir,
                sample_input=d,
            )  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "model_dir_path and model_stage_file_path both cannot be specified at the same time."
        ):
            model_api.save_model(
                name="model",
                model=linear_model.LinearRegression(),
                session=c_session,
                model_stage_file_path=stage_path,
                model_dir_path=local_dir,
                sample_input=d,
            )  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "Signatures and sample_input both cannot be None for local model at the same time."
        ):
            model_api.save_model(
                name="model1",
                model_dir_path=local_dir,
                model=linear_model.LinearRegression(),
            )

        with self.assertRaisesRegex(
            ValueError, "Signatures and sample_input both cannot be specified at the same time."
        ):
            model_api.save_model(  # type:ignore[call-overload]
                name="model1",
                model_dir_path=local_dir,
                model=linear_model.LinearRegression(),
                sample_input=d,
                signatures={"predict": model_signature.ModelSignature(inputs=[], outputs=[])},
            )

        with self.assertRaisesRegex(
            ValueError, "Signatures and sample_input both cannot be specified at the same time."
        ):
            model_api.save_model(  # type:ignore[call-overload]
                name="model1",
                model_dir_path=local_dir,
                model=LinearRegression(),
                sample_input=d,
                signatures={"predict": model_signature.ModelSignature(inputs=[], outputs=[])},
            )

        with mock.patch.object(model_api, "_save", return_value=None) as mock_save:
            model_api.save_model(
                name="model1",
                model_dir_path=local_dir,
                model=LinearRegression(),
            )

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "some_file"), "w", encoding="utf-8") as f:
                f.write("Hi Ciyana!")

            with self.assertRaisesRegex(ValueError, "Provided model directory [^\\s]* is not a directory."):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tempdir, "some_file"),
                    model=linear_model.LinearRegression(),
                    sample_input=d,
                )

            with self.assertWarnsRegex(UserWarning, "Provided model directory [^\\s]* is not an empty directory."):
                with mock.patch.object(model_api, "_save", return_value=None) as mock_save:
                    model_api.save_model(
                        name="model1",
                        model_dir_path=tempdir,
                        model=linear_model.LinearRegression(),
                        sample_input=d,
                    )
                    mock_save.assert_called_once()

        with self.assertRaisesRegex(
            ValueError, "Provided model path in the stage [^\\s]* must be a path to a zip file."
        ):
            model_api.save_model(
                name="model1",
                model=linear_model.LinearRegression(),
                session=c_session,
                model_stage_file_path='@"db"."schema"."stage"/model',
                sample_input=d,
            )

        with mock.patch.object(model_api, "_save", return_value=None) as mock_save:
            with mock.patch.object(FileOperation, "put_stream", return_value=None) as mock_put_stream:
                model_api.save_model(
                    name="model1",
                    model=linear_model.LinearRegression(),
                    session=c_session,
                    model_stage_file_path=stage_path,
                    sample_input=d,
                )
            mock_put_stream.assert_called_once_with(mock.ANY, stage_path, auto_compress=False, overwrite=False)

        with mock.patch.object(model_api, "_save", return_value=None) as mock_save:
            with mock.patch.object(FileOperation, "put_stream", return_value=None) as mock_put_stream:
                model_api.save_model(  # type:ignore[call-overload]
                    name="model1",
                    model=linear_model.LinearRegression(),
                    session=c_session,
                    model_stage_file_path=stage_path,
                    sample_input=d,
                    options={"allow_overwritten_stage_file": True},
                )
            mock_put_stream.assert_called_once_with(mock.ANY, stage_path, auto_compress=False, overwrite=True)

    def test_load_interface(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        local_dir = "path/to/local/model/dir"
        stage_path = '@"db"."schema"."stage"/model.zip'

        with self.assertRaisesRegex(
            ValueError, "Session and model_stage_file_path must be specified at the same time."
        ):
            model_api.load_model(session=c_session)  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "model_dir_path and model_stage_file_path both cannot be None at the same time."
        ):
            model_api.load_model()  # type:ignore[call-overload]

        with self.assertRaisesRegex(ValueError, "Session and model_stage_file_path must be None at the same time."):
            model_api.load_model(model_stage_file_path=stage_path)  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "model_dir_path and model_stage_file_path both cannot be specified at the same time."
        ):
            model_api.load_model(
                session=c_session, model_stage_file_path=stage_path, model_dir_path=local_dir
            )  # type:ignore[call-overload]

        with self.assertRaisesRegex(
            ValueError, "Provided model path in the stage [^\\s]* must be a path to a zip file."
        ):
            model_api.load_model(session=c_session, model_stage_file_path='@"db"."schema"."stage"/model')


class ModelTest(absltest.TestCase):
    def test_bad_save_model(self) -> None:
        tmpdir = self.create_tempdir()
        os.mkdir(os.path.join(tmpdir.full_path, "bias"))
        with open(os.path.join(tmpdir.full_path, "bias", "bias1"), "w", encoding="utf-8") as f:
            f.write("25")
        with open(os.path.join(tmpdir.full_path, "bias", "bias2"), "w", encoding="utf-8") as f:
            f.write("68")
        lm = DemoModelWithManyArtifacts(
            custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir.full_path, "bias")})
        )
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        s = {"predict": model_signature.infer_signature(d, lm.predict(d))}

        with self.assertRaises(ValueError):
            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir.full_path, "model1"),
                model=lm,
                signatures={**s, "another_predict": s["predict"]},
                metadata={"author": "halu", "version": "1"},
            )

        model_api.save_model(
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=lm,
            signatures=s,
            metadata={"author": "halu", "version": "1"},
            python_version="3.5.2",
        )

        _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"), meta_only=True)

        with self.assertRaises(RuntimeError):
            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))

    def test_custom_model_with_multiple_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "bias"))
            with open(os.path.join(tmpdir, "bias", "bias1"), "w", encoding="utf-8") as f:
                f.write("25")
            with open(os.path.join(tmpdir, "bias", "bias2"), "w", encoding="utf-8") as f:
                f.write("68")
            lm = DemoModelWithManyArtifacts(
                custom_model.ModelContext(
                    models={}, artifacts={"bias": os.path.join(tmpdir, "bias", "")}
                )  # Test sanitizing user path input.
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = {"predict": model_signature.infer_signature(d, lm.predict(d))}
            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=lm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, DemoModelWithManyArtifacts)
                res = m.predict(d)
                np.testing.assert_allclose(res["output"], pd.Series(np.array([94, 97])))

                m_UDF, meta = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                assert isinstance(m_UDF, DemoModelWithManyArtifacts)
                res = m_UDF.predict(d)
                np.testing.assert_allclose(res["output"], pd.Series(np.array([94, 97])))
                self.assertEqual(meta.metadata["author"] if meta.metadata else None, "halu")

                model_api.save_model(
                    name="model1_no_sig",
                    model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                    model=lm,
                    sample_input=d,
                    metadata={"author": "halu", "version": "1"},
                )

                m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
                assert isinstance(m, DemoModelWithManyArtifacts)
                res = m.predict(d)
                np.testing.assert_allclose(res["output"], pd.Series(np.array([94, 97])))
                self.assertEqual(s, meta.signatures)

    def test_model_composition(self) -> None:
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        aclf = AnotherDemoModel(custom_model.ModelContext())
        clf = DemoModel(custom_model.ModelContext())
        model_context = custom_model.ModelContext(
            models={
                "m1": aclf,
                "m2": clf,
            }
        )
        acm = ComposeModel(model_context)
        p1 = clf.predict(d)
        p2 = acm.predict(d)
        s = {"predict": model_signature.infer_signature(d, p2)}
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=acm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )
            lm, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(lm, ComposeModel)
            p3 = lm.predict(d)

            m_UDF, _ = model_api._load_model_for_deploy(model_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m_UDF, ComposeModel)
            p4 = m_UDF.predict(d)
            np.testing.assert_allclose(p1, p2)
            np.testing.assert_allclose(p2, p3)
            np.testing.assert_allclose(p2, p4)

    def test_async_model_composition(self) -> None:
        async def _test(self: "ModelTest") -> None:
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            clf = DemoModel(custom_model.ModelContext())
            model_context = custom_model.ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            p1 = clf.predict(d)
            p2 = await acm.predict(d)
            s = {"predict": model_signature.infer_signature(d, p2)}
            with tempfile.TemporaryDirectory() as tmpdir:
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=acm,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                )
                lm, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(lm, AsyncComposeModel)
                p3 = await lm.predict(d)  # type: ignore[misc]

                m_UDF, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                assert isinstance(m_UDF, AsyncComposeModel)
                p4 = await m_UDF.predict(d)
                np.testing.assert_allclose(p1, p2)
                np.testing.assert_allclose(p2, p3)
                np.testing.assert_allclose(p2, p4)

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_custom_model_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = {"predict": model_signature.infer_signature(d, lm.predict(d))}
            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=lm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, DemoModelWithArtifacts)
            res = m.predict(d)
            np.testing.assert_allclose(res["output"], pd.Series(np.array([11, 14])))

            # test re-init when loading the model
            with open(
                os.path.join(tmpdir, "model1", "models", "model1", "artifacts", "bias"), "w", encoding="utf-8"
            ) as f:
                f.write("20")

            m_UDF, meta = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
            assert isinstance(m_UDF, DemoModelWithArtifacts)
            res = m_UDF.predict(d)

            np.testing.assert_allclose(res["output"], pd.Series(np.array([21, 24])))
            self.assertEqual(meta.metadata["author"] if meta.metadata else None, "halu")

    def test_skl_multiple_output_proba(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df[:-10], dual_target[:-10])
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict_proba": model_signature.infer_signature(iris_X_df, model.predict_proba(iris_X_df))}
            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["scikit-learn"],
            )

            orig_res = model.predict_proba(iris_X_df[-10:])

            m: multioutput.MultiOutputClassifier
            m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))

            loaded_res = m.predict_proba(iris_X_df[-10:])
            np.testing.assert_allclose(np.hstack(orig_res), np.hstack(loaded_res))

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
            predict_method = getattr(m_udf, "predict_proba", None)
            assert callable(predict_method)
            udf_res = predict_method(iris_X_df[-10:])
            np.testing.assert_allclose(
                np.hstack(orig_res), np.hstack([np.array(udf_res[col].to_list()) for col in udf_res])
            )

            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1_no_sig_bad",
                    model_dir_path=os.path.join(tmpdir, "model1_no_sig_bad"),
                    model=model,
                    sample_input=iris_X_df,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SKLModelSaveOptions({"target_methods": ["random"]}),
                )

            model_api.save_model(
                name="model1_no_sig",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=model,
                sample_input=iris_X_df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(
                np.hstack(model.predict_proba(iris_X_df[-10:])), np.hstack(m.predict_proba(iris_X_df[-10:]))
            )
            np.testing.assert_allclose(model.predict(iris_X_df[-10:]), m.predict(iris_X_df[-10:]))
            self.assertEqual(s["predict_proba"], meta.signatures["predict_proba"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig"))

            predict_method = getattr(m_udf, "predict_proba", None)
            assert callable(predict_method)
            udf_res = predict_method(iris_X_df[-10:])
            np.testing.assert_allclose(
                np.hstack(model.predict_proba(iris_X_df[-10:])),
                np.hstack([np.array(udf_res[col].to_list()) for col in udf_res]),
            )

            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(model.predict(iris_X_df[-10:]), predict_method(iris_X_df[-10:]).to_numpy())

    def test_skl(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(iris_X_df, regr.predict(iris_X_df))}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: linear_model.LinearRegression
                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(np.array([-0.08254936]), m.predict(iris_X_df[:1]))
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))

            model_api.save_model(
                name="model1_no_sig",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=iris_X_df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([-0.08254936]), m.predict(iris_X_df[:1]))
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))

    def test_xgb(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        y_pred = regressor.predict(cal_X_test)
        y_pred_proba = regressor.predict_proba(cal_X_test)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=regressor,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=regressor,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, xgboost.XGBClassifier)
                np.testing.assert_allclose(m.predict(cal_X_test), y_pred)
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            model_api.save_model(
                name="model1_no_sig",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regressor,
                sample_input=cal_X_test,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            assert isinstance(m, xgboost.XGBClassifier)
            np.testing.assert_allclose(m.predict(cal_X_test), y_pred)
            np.testing.assert_allclose(m.predict_proba(cal_X_test), y_pred_proba)
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            predict_method = getattr(m_udf, "predict_proba", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred_proba)

    def test_snowml_all_input(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: LinearRegression
                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(predictions, m.predict(df[:1])[[OUTPUT_COLUMNS]])
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

            model_api.save_model(
                name="model1_no_sig",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=df[INPUT_COLUMNS],
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([[-0.08254936]]), m.predict(df[:1])[[OUTPUT_COLUMNS]])
            s = regr.model_signatures
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_snowml_signature_partial_input(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: LinearRegression
                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(predictions, m.predict(df[:1])[[OUTPUT_COLUMNS]])
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

            model_api.save_model(
                name="model1_no_sig",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([[0.17150434]]), m.predict(df[:1])[[OUTPUT_COLUMNS]])
            s = regr.model_signatures
            # Compare the Model Signature without indexing
            self.assertItemsEqual(s["predict"].to_dict(), meta.signatures["predict"].to_dict())

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[0.17150434]]), predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_snowml_signature_drop_input_cols(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(
            input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS, drop_input_cols=True
        )
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: LinearRegression
                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(predictions, m.predict(df[:1])[[OUTPUT_COLUMNS]])
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

            model_api.save_model(
                name="model1_no_sig",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([[-0.08254936]]), m.predict(df[:1])[[OUTPUT_COLUMNS]])
            s = regr.model_signatures
            # Compare the Model Signature without indexing
            self.assertItemsEqual(s["predict"].to_dict(), meta.signatures["predict"].to_dict())

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_torch_model()
            s = {"forward": model_signature.infer_signature(data_x, data_y)}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=model,
                    signatures={**s, "another_forward": s["forward"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            model.eval()
            y_pred = model.forward(data_x)[0].detach()

            x_df = model_signature_utils.rename_pandas_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False),
                s["forward"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, torch.nn.Module)
                torch.testing.assert_close(m.forward(data_x)[0], y_pred)  # type:ignore[attr-defined]
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "forward", None)
                assert callable(predict_method)
                torch.testing.assert_close(  # type:ignore[attr-defined]
                    pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                        predict_method(x_df), s["forward"].outputs
                    )[0],
                    y_pred,
                )

            model_api.save_model(
                name="model1_no_sig_1",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"),
                model=model,
                sample_input=data_x,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"))
            assert isinstance(m, torch.nn.Module)
            torch.testing.assert_close(m.forward(data_x)[0], y_pred)  # type:ignore[attr-defined]
            self.assertEqual(s["forward"], meta.signatures["forward"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_1"))
            predict_method = getattr(m_udf, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(  # type:ignore[attr-defined]
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df), s["forward"].outputs)[
                    0
                ],
                y_pred,
            )

    def test_torchscript(self) -> None:
        model, data_x, data_y = _prepare_torch_model()
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"forward": model_signature.infer_signature(data_x, data_y)}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=model_script,
                    signatures={**s, "another_forward": s["forward"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=model_script,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            model_script.eval()
            y_pred = model_script.forward(data_x)[0].detach()

            x_df = model_signature_utils.rename_pandas_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False),
                s["forward"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, torch.jit.ScriptModule)  # type:ignore[attr-defined]
                torch.testing.assert_close(m.forward(data_x)[0], y_pred)  # type:ignore[attr-defined]
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "forward", None)
                assert callable(predict_method)
                torch.testing.assert_close(  # type:ignore[attr-defined]
                    pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                        predict_method(x_df), s["forward"].outputs
                    )[0],
                    y_pred,
                )

            model_api.save_model(
                name="model1_no_sig_1",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"),
                model=model_script,
                sample_input=data_x,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"))
            assert isinstance(m, torch.jit.ScriptModule)  # type:ignore[attr-defined]
            torch.testing.assert_close(m.forward(data_x)[0], y_pred)  # type:ignore[attr-defined]
            self.assertEqual(s["forward"], meta.signatures["forward"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_1"))
            predict_method = getattr(m_udf, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(  # type:ignore[attr-defined]
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df), s["forward"].outputs)[
                    0
                ],
                y_pred,
            )

    def test_torch_df_sample_input(self) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        s = {"forward": model_signature.infer_signature(data_x, data_y)}

        with tempfile.TemporaryDirectory() as tmpdir:
            model.eval()
            y_pred = model.forward(data_x)[0].detach()

            x_df = model_signature_utils.rename_pandas_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False),
                s["forward"].inputs,
            )
            model_api.save_model(
                name="model1_no_sig_1",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"),
                model=model,
                sample_input=x_df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"))
            assert isinstance(m, torch.nn.Module)
            torch.testing.assert_close(m.forward(data_x)[0], y_pred)  # type:ignore[attr-defined]

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_1"))
            predict_method = getattr(m_udf, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(  # type:ignore[attr-defined]
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df))[0], y_pred
            )

            model_script.eval()
            y_pred = model_script.forward(data_x)[0].detach()

            model_api.save_model(
                name="model1_no_sig_2",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_2"),
                model=model_script,
                sample_input=x_df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig_2"))
            assert isinstance(m, torch.jit.ScriptModule)  # type:ignore[attr-defined]
            torch.testing.assert_close(m.forward(data_x)[0], y_pred)  # type:ignore[attr-defined]

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_2"))
            predict_method = getattr(m_udf, "forward", None)
            assert callable(predict_method)
            torch.testing.assert_close(  # type:ignore[attr-defined]
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(predict_method(x_df))[0], y_pred
            )

    def test_tensorflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            simple_module = SimpleModule(name="simple")
            x = [tf.constant([[5.0], [10.0]])]
            y_pred = simple_module(x)
            s = {"__call__": model_signature.infer_signature(x, y_pred)}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=simple_module,
                    signatures={**s, "another_forward": s["__call__"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=simple_module,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            x_df = model_signature_utils.rename_pandas_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(x, ensure_serializable=False),
                s["__call__"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert callable(m)
                tf.assert_equal(m.__call__(x)[0], y_pred[0])
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                assert callable(m_udf)
                tf.assert_equal(
                    tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(m_udf(x_df), s["__call__"].outputs)[
                        0
                    ],
                    y_pred[0],
                )

            model_api.save_model(
                name="model1_no_sig_1",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"),
                model=simple_module,
                sample_input=x,
                metadata={"author": "halu", "version": "1"},
            )

            m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"))
            assert callable(m)
            tf.assert_equal(m(x)[0], y_pred[0])
            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_1"))
            assert callable(m_udf)
            tf.assert_equal(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(m_udf(x_df), s["__call__"].outputs)[0],
                y_pred[0],
            )

            model_api.save_model(
                name="model1_no_sig_2",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_2"),
                model=simple_module,
                sample_input=x_df,
                metadata={"author": "halu", "version": "1"},
            )

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_2"))
            assert callable(m_udf)
            tf.assert_equal(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(m_udf(x_df), s["__call__"].outputs)[0],
                y_pred[0],
            )

    def test_tensorflow_keras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = _prepare_keras_model()
            s = {"predict": model_signature.infer_signature(data_x, data_y)}
            with self.assertRaises(ValueError):
                model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=model,
                    signatures={**s, "another_forward": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            y_pred = model.predict(data_x)[0]

            x_df = model_signature_utils.rename_pandas_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False),
                s["predict"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, tf.keras.Model)
                tf.debugging.assert_near(m.predict(data_x)[0], y_pred)
                m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                tf.debugging.assert_near(
                    tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                        predict_method(x_df), s["predict"].outputs
                    )[0],
                    y_pred,
                )

            model_api.save_model(
                name="model1_no_sig_1",
                model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"),
                model=model,
                sample_input=data_x,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1_no_sig_1"))
            assert isinstance(m, tf.keras.Model)
            tf.debugging.assert_near(m.predict(data_x)[0], y_pred)
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1_no_sig_1"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            tf.debugging.assert_near(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                    predict_method(x_df), s["predict"].outputs
                )[0],
                y_pred,
            )

    def test_mlflow_model(self) -> None:
        db = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                conda_env={
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.8.13",
                        "pip<=23.0.1",
                        {
                            "pip": [
                                "mlflow<3,>=2.3",
                                "cloudpickle==2.0.0",
                                "numpy==1.23.4",
                                "psutil==5.9.0",
                                "scikit-learn==1.2.2",
                                "scipy==1.9.3",
                                "typing-extensions==4.5.0",
                            ]
                        },
                    ],
                    "name": "mlflow-env",
                },
                signature=signature,
                metadata={"author": "halu", "version": "1"},
            )

            run_id = run.info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            saved_meta = model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=mlflow_pyfunc_model,
            )

            self.assertEqual(saved_meta.python_version, "3.8.13")
            self.assertDictEqual(saved_meta.metadata, {"author": "halu", "version": "1"})
            self.assertDictEqual(
                saved_meta.signatures,
                {
                    "predict": model_signature.ModelSignature(
                        inputs=[
                            model_signature.FeatureSpec(
                                name="input_feature_0", dtype=model_signature.DataType.DOUBLE, shape=(10,)
                            )
                        ],
                        outputs=[
                            model_signature.FeatureSpec(name="output_feature_0", dtype=model_signature.DataType.DOUBLE)
                        ],
                    )
                },
            )
            self.assertListEqual(
                sorted(saved_meta.pip_requirements),
                sorted(
                    [
                        "mlflow<3,>=2.3",
                        "cloudpickle==2.0.0",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ]
                ),
            )
            self.assertIn("pip<=23.0.1", saved_meta.conda_dependencies)

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(m.metadata.run_id, run_id)

            _ = model_api.save_model(
                name="model1_again",
                model_dir_path=os.path.join(tmpdir, "model1_again"),
                model=m,
            )

            self.assertEqual(meta.python_version, "3.8.13")
            self.assertDictEqual(meta.metadata, {"author": "halu", "version": "1"})
            self.assertDictEqual(
                meta.signatures,
                {
                    "predict": model_signature.ModelSignature(
                        inputs=[
                            model_signature.FeatureSpec(
                                name="input_feature_0", dtype=model_signature.DataType.DOUBLE, shape=(10,)
                            )
                        ],
                        outputs=[
                            model_signature.FeatureSpec(name="output_feature_0", dtype=model_signature.DataType.DOUBLE)
                        ],
                    )
                },
            )
            self.assertListEqual(
                sorted(meta.pip_requirements),
                sorted(
                    [
                        "mlflow<3,>=2.3",
                        "cloudpickle==2.0.0",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ]
                ),
            )
            self.assertIn("pip<=23.0.1", meta.conda_dependencies)

            np.testing.assert_allclose(predictions, m.predict(X_test))

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            X_df = pd.DataFrame(X_test)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_df).to_numpy())

    def test_mlflow_model_df_inputs(self) -> None:
        db = datasets.load_diabetes(as_frame=True)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                signature=signature,
            )

            run_id = run.info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            _ = model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=mlflow_pyfunc_model,
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(m.metadata.run_id, run_id)

            np.testing.assert_allclose(predictions, m.predict(X_test))

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_test).to_numpy())

    def test_mlflow_model_bad_case(self) -> None:
        db = datasets.load_diabetes(as_frame=True)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                signature=signature,
                metadata={"author": "halu", "version": "1"},
            )

            run_id = run.info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model", dst_path=tmpdir)
            mlflow_pyfunc_model = mlflow.pyfunc.load_model(local_path)
            mlflow_pyfunc_model.metadata.run_id = uuid.uuid4().hex.lower()
            with self.assertRaisesRegex(ValueError, "Cannot load MLFlow model artifacts."):
                _ = model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=mlflow_pyfunc_model,
                    options={"ignore_mlflow_dependencies": True},
                )

            saved_meta = model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=mlflow_pyfunc_model,
                options={"model_uri": local_path, "ignore_mlflow_dependencies": True},
            )

            self.assertEmpty(saved_meta.pip_requirements)

            with self.assertRaisesRegex(ValueError, "Cannot load MLFlow model dependencies."):
                _ = model_api.save_model(
                    name="model1",
                    model_dir_path=os.path.join(tmpdir, "model1"),
                    model=mlflow_pyfunc_model,
                )

            saved_meta = model_api.save_model(
                name="model2",
                model_dir_path=os.path.join(tmpdir, "model2"),
                model=mlflow_pyfunc_model,
                options={"model_uri": local_path, "ignore_mlflow_metadata": True},
            )

            self.assertIsNone(saved_meta.metadata)

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model2"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(m.metadata.run_id, run_id)

            np.testing.assert_allclose(predictions, m.predict(X_test))

            _ = model_api.save_model(
                name="model2_again",
                model_dir_path=os.path.join(tmpdir, "model2_again"),
                model=m,
            )

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model2"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_test).to_numpy())

    def test_mlflow_model_pytorch(self) -> None:
        net = torch.nn.Linear(6, 1)
        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        X = torch.randn(6)
        y = torch.randn(1)

        epochs = 5
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = net(X)

            loss = loss_function(outputs, y)
            loss.backward()

            optimizer.step()

        with mlflow.start_run():
            signature = mlflow.models.infer_signature(X.numpy(), net(X).detach().numpy())
            model_info = mlflow.pytorch.log_model(net, "model", signature=signature)

        pytorch_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
        input_x = torch.randn(6).numpy()
        predictions = pytorch_pyfunc.predict(input_x)

        with tempfile.TemporaryDirectory() as tmpdir:
            _ = model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir, "model1"),
                model=pytorch_pyfunc,
            )

            m, meta = model_api.load_model(model_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)

            np.testing.assert_allclose(predictions, m.predict(input_x))

            m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir, "model1"))
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(
                np.expand_dims(predictions, axis=1), predict_method(pd.DataFrame(input_x)).to_numpy()
            )


if __name__ == "__main__":
    absltest.main()
