from typing import Any, Callable

import keras
import numpy as np
import numpy.typing as npt
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import numpy_handler, snowpark_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


def _prepare_keras_subclass_model() -> tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]:
    @keras.saving.register_keras_serializable()
    class KerasModel(keras.Model):
        def __init__(self, n_hidden: int, n_out: int) -> None:
            super().__init__()
            self.fc_1 = keras.layers.Dense(n_hidden, activation="relu")
            self.fc_2 = keras.layers.Dense(n_out, activation="sigmoid")

        def call(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
            x = self.fc_1(inputs)
            x = self.fc_2(x)
            return x  # type: ignore[no-any-return]

        def get_config(self) -> dict[str, Any]:
            base_config = super().get_config()
            config = {
                "fc_1": keras.saving.serialize_keras_object(self.fc_1),
                "fc_2": keras.saving.serialize_keras_object(self.fc_2),
            }
            return {**base_config, **config}

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "KerasModel":
            fc_1_config = config.pop("fc_1")
            fc_1 = keras.saving.deserialize_keras_object(fc_1_config)
            fc_2_config = config.pop("fc_2")
            fc_2 = keras.saving.deserialize_keras_object(fc_2_config)
            obj = cls(1, 1)
            obj.fc_1 = fc_1
            obj.fc_2 = fc_2
            return obj

    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    y = np.random.random_integers(0, 1, (batch_size,)).astype(np.float32)

    model = KerasModel(n_hidden, n_out)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError())
    model.fit(x, y, batch_size=batch_size, epochs=10)
    return model, x, y


def _prepare_keras_sequential_model() -> tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]:
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
    model.fit(x, y, batch_size=batch_size, epochs=10)
    return model, x, y


def _prepare_keras_functional_model() -> tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    y = np.random.random_integers(0, 1, (batch_size,)).astype(np.float32)

    input = keras.Input(shape=(n_input,))
    input_2 = keras.layers.Dense(n_hidden, activation="relu")(input)
    output = keras.layers.Dense(n_out, activation="sigmoid")(input_2)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError())
    model.fit(x, y, batch_size=batch_size, epochs=10)
    return model, x, y


class TestRegistryTensorflowModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        model_fn=[_prepare_keras_subclass_model, _prepare_keras_sequential_model, _prepare_keras_functional_model],
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_keras_tensor_as_sample(
        self,
        model_fn: Callable[[], tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]],
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_fn()
        y_pred = model.predict(data_x)

        def assert_fn(res) -> None:
            y_pred_df = pd.DataFrame(y_pred, columns=res.columns)
            pd.testing.assert_frame_equal(res, y_pred_df, check_dtype=False)

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=data_x,
            prediction_assert_fns={
                "": (
                    data_x,
                    assert_fn,
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        model_fn=[_prepare_keras_subclass_model, _prepare_keras_sequential_model, _prepare_keras_functional_model],
    )
    def test_keras_df_as_sample(
        self,
        model_fn: Callable[[], tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]],
    ) -> None:
        model, data_x, data_y = model_fn()
        x_df = pd.DataFrame(data_x)
        y_pred = model.predict(data_x)

        def assert_fn(res) -> None:
            y_pred_df = pd.DataFrame(y_pred, columns=res.columns)
            pd.testing.assert_frame_equal(res, y_pred_df, check_dtype=False)

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        model_fn=[_prepare_keras_subclass_model, _prepare_keras_sequential_model, _prepare_keras_functional_model],
    )
    def test_keras_sp(
        self,
        model_fn: Callable[[], tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]],
    ) -> None:
        model, data_x, data_y = model_fn()
        x_df = numpy_handler.NumpyArrayHandler.convert_to_df(data_x)
        x_df.columns = [f"input_feature_{i}" for i in range(len(x_df.columns))]
        y_pred = model.predict(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self.session,
            x_df,
        )
        y_pred_df = numpy_handler.NumpyArrayHandler.convert_to_df(y_pred)
        y_pred_df.columns = [f"output_feature_{i}" for i in range(len(y_pred_df.columns))]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
