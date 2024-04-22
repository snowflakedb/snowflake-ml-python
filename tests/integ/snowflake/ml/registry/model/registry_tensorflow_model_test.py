from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from absl.testing import absltest

from snowflake.ml.model._signatures import (
    numpy_handler,
    snowpark_handler,
    tensorflow_handler,
)
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


def prepare_keras_model(
    dtype: "tf.dtypes.DType" = tf.float32,
) -> Tuple["tf.keras.Model", "tf.Tensor", "tf.Tensor"]:
    class KerasModel(tf.keras.Model):
        def __init__(self, n_hidden: int, n_out: int) -> None:
            super().__init__()
            self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
            self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

        def call(self, tensor: "tf.Tensor") -> "tf.Tensor":
            input = tensor
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


@pytest.mark.pip_incompatible
class TestRegistryTensorflowModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_tf_tensor_as_sample(
        self,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        self._test_registry_model(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred.numpy(),
                    ),
                ),
            },
        )

    def test_tf_df_as_sample(
        self,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred.numpy(),
                    ),
                ),
            },
        )

    def test_tf_sp(
        self,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self._session,
            x_df,
        )
        y_pred_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([y_pred])
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

    def test_keras_tensor_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)
        self._test_registry_model(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred,
                        atol=1e-6,
                    ),
                ),
            },
        )

    def test_keras_df_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)
        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred,
                        atol=1e-6,
                    ),
                ),
            },
        )

    def test_keras_sp(
        self,
    ) -> None:
        model, data_x, data_y = prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.predict(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self._session,
            x_df,
        )
        y_pred_df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df([y_pred])
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


if __name__ == "__main__":
    absltest.main()
