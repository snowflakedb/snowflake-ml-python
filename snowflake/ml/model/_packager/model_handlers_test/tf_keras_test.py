import os
import tempfile
import warnings
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import tf_keras
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._signatures import (
    tensorflow_handler,
    utils as model_signature_utils,
)


def _prepare_keras_subclass_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf_keras.Model, tf.Tensor, tf.Tensor]:
    class KerasModel(tf_keras.Model):
        def __init__(self, n_hidden: int, n_out: int) -> None:
            super().__init__()
            self.fc_1 = tf_keras.layers.Dense(n_hidden, activation="relu")
            self.fc_2 = tf_keras.layers.Dense(n_out, activation="sigmoid")

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
        optimizer=tf_keras.optimizers.SGD(learning_rate=learning_rate), loss=tf_keras.losses.MeanSquaredError()
    )
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


def _prepare_keras_sequential_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf_keras.Model, tf.Tensor, tf.Tensor]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = tf.convert_to_tensor(x, dtype=dtype)
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = tf.cast(raw_data_y, dtype=dtype)

    model = tf_keras.Sequential(
        [
            tf_keras.layers.Dense(n_hidden, activation="relu"),
            tf_keras.layers.Dense(n_out, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf_keras.optimizers.SGD(learning_rate=learning_rate), loss=tf_keras.losses.MeanSquaredError()
    )
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


def _prepare_keras_functional_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf_keras.Model, tf.Tensor, tf.Tensor]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = tf.convert_to_tensor(x, dtype=dtype)
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = tf.cast(raw_data_y, dtype=dtype)

    input = tf_keras.Input(shape=(n_input,))
    x = tf_keras.layers.Dense(n_hidden, activation="relu")(input)
    output = tf_keras.layers.Dense(n_out, activation="sigmoid")(x)
    model = tf_keras.Model(inputs=input, outputs=output)
    model.compile(
        optimizer=tf_keras.optimizers.SGD(learning_rate=learning_rate), loss=tf_keras.losses.MeanSquaredError()
    )
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


class TensorflowHandlerTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        {"model_fn": _prepare_keras_subclass_model},
        {"model_fn": _prepare_keras_sequential_model},
        {"model_fn": _prepare_keras_functional_model},
    )
    def test_tensorflow_keras(
        self, model_fn: Callable[[tf.dtypes.DType], Tuple[tf_keras.Model, tf.Tensor, tf.Tensor]]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = model_fn(tf.float32)
            s = {"predict": model_signature.infer_signature(data_x, data_y)}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**s, "another_forward": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.KerasSaveOptions(),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.KerasSaveOptions(),
            )

            y_pred = model.predict(data_x)

            x_df = model_signature_utils.rename_pandas_df(
                tensorflow_handler.TensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False),
                s["predict"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, tf_keras.Model)
                tf.debugging.assert_near(pk.model.predict(data_x), y_pred)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                tf.debugging.assert_near(
                    predict_method(x_df),
                    y_pred,
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=model,
                sample_input_data=data_x,
                metadata={"author": "halu", "version": "1"},
                options=model_types.KerasSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, tf_keras.Model)
            tf.debugging.assert_near(pk.model.predict(data_x), y_pred)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            tf.debugging.assert_near(
                predict_method(x_df),
                y_pred,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_fn": _prepare_keras_subclass_model},
        {"model_fn": _prepare_keras_sequential_model},
        {"model_fn": _prepare_keras_functional_model},
    )
    def test_tensorflow_keras_multiple_inputs(
        self, model_fn: Callable[[tf.dtypes.DType], Tuple[tf_keras.Model, tf.Tensor, tf.Tensor]]
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
                    options={"multiple_inputs": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options={"multiple_inputs": True},
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
                assert isinstance(pk.model, tf_keras.Model)
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
                options={"multiple_inputs": True},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, tf_keras.Model)
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
