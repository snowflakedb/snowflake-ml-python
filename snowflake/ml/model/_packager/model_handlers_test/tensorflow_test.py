import os
import tempfile
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._signatures import (
    tensorflow_handler,
    utils as model_signature_utils,
)


class SimpleModule(tf.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    @tf.function  # type: ignore[misc]
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        return self.a_variable * tensor + self.non_trainable_variable


def _prepare_keras_subclass_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]:
    class KerasModel(tf.keras.Model):
        def __init__(self, n_hidden: int, n_out: int) -> None:
            super().__init__()
            self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
            self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

        def call(self, tensors: tf.Tensor) -> tf.Tensor:
            input = tensors
            x = self.fc_1(input)
            x = self.fc_2(x)
            return x

    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = tf.convert_to_tensor(x, dtype=dtype)
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = tf.cast(raw_data_y, dtype=dtype)

    model = KerasModel(n_hidden, n_out)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


def _prepare_keras_sequential_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = tf.convert_to_tensor(x, dtype=dtype)
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = tf.cast(raw_data_y, dtype=dtype)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(n_hidden, activation="relu"),
            tf.keras.layers.Dense(n_out, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


def _prepare_keras_functional_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = tf.convert_to_tensor(x, dtype=dtype)
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = tf.cast(raw_data_y, dtype=dtype)

    input = tf.keras.Input(shape=(n_input,))
    x = tf.keras.layers.Dense(n_hidden, activation="relu")(input)
    output = tf.keras.layers.Dense(n_out, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


class TensorflowHandlerTest(parameterized.TestCase):
    def test_tensorflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            simple_module = SimpleModule(name="simple")
            x = tf.constant([[5.0], [10.0]])
            y_pred = simple_module(x)
            s = {"__call__": model_signature.infer_signature([x], [y_pred])}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=simple_module,
                    signatures={**s, "another_forward": s["__call__"]},
                    metadata={"author": "halu", "version": "1"},
                )

            with self.assertRaises(NotImplementedError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=simple_module,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=simple_module,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            x_df = model_signature_utils.rename_pandas_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data=[x], ensure_serializable=False),
                s["__call__"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert callable(pk.model)
                tf.assert_equal(pk.model(x), y_pred)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                assert callable(pk.model)
                tf.assert_equal(
                    tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                        pk.model(x_df), s["__call__"].outputs
                    )[0],
                    y_pred,
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=simple_module,
                sample_input_data=[x],
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            tf.assert_equal(pk.model(x), y_pred)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            tf.assert_equal(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(pk.model(x_df), s["__call__"].outputs)[
                    0
                ],
                y_pred,
            )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2")).save(
                name="model1_no_sig_2",
                model=simple_module,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            tf.assert_equal(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(pk.model(x_df), s["__call__"].outputs)[
                    0
                ],
                y_pred,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_fn": _prepare_keras_subclass_model},
        {"model_fn": _prepare_keras_sequential_model},
        {"model_fn": _prepare_keras_functional_model},
    )
    def test_tensorflow_keras(
        self, model_fn: Callable[[tf.dtypes.DType], Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = model_fn(tf.float32)
            s = {"predict": model_signature.infer_signature([data_x], [data_y])}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**s, "another_forward": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            y_pred = model.predict(data_x)

            x_df = model_signature_utils.rename_pandas_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False),
                s["predict"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, tf.keras.Model)
                tf.debugging.assert_near(pk.model.predict(data_x), y_pred)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                tf.debugging.assert_near(
                    tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                        predict_method(x_df), s["predict"].outputs
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
            assert isinstance(pk.model, tf.keras.Model)
            tf.debugging.assert_near(pk.model.predict(data_x), y_pred)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            tf.debugging.assert_near(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                    predict_method(x_df), s["predict"].outputs
                )[0],
                y_pred,
            )


if __name__ == "__main__":
    absltest.main()
