import os
import tempfile
import warnings
from typing import Any, Callable, Dict, Tuple

import keras
import numpy as np
import numpy.typing as npt
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils


def _prepare_keras_subclass_model() -> Tuple[keras.Model, npt.NDArray[np.float64], npt.NDArray[np.float32]]:
    @keras.saving.register_keras_serializable()
    class KerasModel(keras.Model):
        def __init__(self, n_hidden: int, n_out: int, **kwargs: Any) -> None:
            super().__init__()
            self.fc_1 = keras.layers.Dense(n_hidden, activation="relu")
            self.fc_2 = keras.layers.Dense(n_out, activation="sigmoid")
            self.n_hidden = n_hidden
            self.n_out = n_out

        def call(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
            x = self.fc_1(inputs)
            x = self.fc_2(x)
            return x  # type: ignore[no-any-return]

        def get_config(self) -> Dict[str, Any]:
            base_config = super().get_config()
            config = {"n_hidden": self.n_hidden, "n_out": self.n_out}
            return {**base_config, **config}

    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    y = np.random.random_integers(0, 1, (batch_size,)).astype(np.float32)

    model = KerasModel(n_hidden, n_out)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError())
    model.fit(x, y, batch_size=batch_size, epochs=100)
    return model, x, y


def _prepare_keras_sequential_model() -> Tuple[keras.Model, npt.NDArray[np.float64], npt.NDArray[np.float32]]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    y = np.random.random_integers(0, 1, (batch_size,)).astype(np.float32)

    model = keras.Sequential(
        [
            keras.layers.Dense(n_hidden, activation="relu"),
            keras.layers.Dense(n_out, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError())
    model.fit(x, y, batch_size=batch_size, epochs=100)
    return model, x, y


def _prepare_keras_functional_model() -> Tuple[keras.Model, npt.NDArray[np.float64], npt.NDArray[np.float32]]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    y = np.random.random_integers(0, 1, (batch_size,)).astype(np.float32)

    input = keras.Input(shape=(n_input,))
    input_2 = keras.layers.Dense(n_hidden, activation="relu")(input)
    output = keras.layers.Dense(n_out, activation="sigmoid")(input_2)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError())
    model.fit(x, y, batch_size=batch_size, epochs=100)
    return model, x, y


class KerasHandlerTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        {"model_fn": _prepare_keras_subclass_model},
        {"model_fn": _prepare_keras_sequential_model},
        {"model_fn": _prepare_keras_functional_model},
    )
    def test_keras(
        self, model_fn: Callable[[], Tuple[keras.Model, npt.NDArray[np.float64], npt.NDArray[np.float32]]]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data_x, data_y = model_fn()
            s = {"predict": model_signature.infer_signature(data_x, data_y)}
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
                numpy_handler.NumpyArrayHandler.convert_to_df(data_x, ensure_serializable=False),
                s["predict"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, keras.Model)
                np.testing.assert_allclose(pk.model.predict(data_x), y_pred)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(
                    predict_method(x_df).values,
                    y_pred,
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=model,
                sample_input_data=data_x,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, keras.Model)
            np.testing.assert_allclose(pk.model.predict(data_x), y_pred)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(
                predict_method(x_df).values,
                y_pred,
            )


if __name__ == "__main__":
    absltest.main()
