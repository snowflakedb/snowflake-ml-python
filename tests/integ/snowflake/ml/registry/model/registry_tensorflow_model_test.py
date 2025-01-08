from typing import Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import (
    numpy_handler,
    snowpark_handler,
    tensorflow_handler,
)
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


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


class TestRegistryTensorflowModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_tf_tensor_as_sample(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        def assert_fn(res):
            y_pred_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(
                tf.transpose(tf.expand_dims(y_pred, axis=0)),
                ensure_serializable=False,
            )
            y_pred_df.columns = res.columns
            pd.testing.assert_frame_equal(
                res,
                y_pred_df,
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_tf_df_as_sample(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        def assert_fn(res):
            y_pred_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(
                tf.transpose(tf.expand_dims(y_pred, axis=0)),
                ensure_serializable=False,
            )
            y_pred_df.columns = res.columns
            pd.testing.assert_frame_equal(
                res,
                y_pred_df,
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
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
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_tf_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self.session,
            x_df,
        )
        y_pred_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        model_fn=[_prepare_keras_subclass_model, _prepare_keras_sequential_model, _prepare_keras_functional_model],
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_keras_tensor_as_sample(
        self,
        model_fn: Callable[[tf.dtypes.DType], Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]],
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_fn(tf.float32)
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)

        def assert_fn(res):
            y_pred_df = pd.DataFrame(y_pred)
            y_pred_df.columns = res.columns
            # res's shape:         (num_rows, 1, 1)
            # y_pred_df's shape:   (num_rows, 1)
            # convert list to scalar value before comparing
            for col in res.columns:
                res[col] = res[col].apply(lambda x: x[0])
            pd.testing.assert_frame_equal(
                res,
                y_pred_df,
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        model_fn=[_prepare_keras_subclass_model, _prepare_keras_sequential_model, _prepare_keras_functional_model],
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
        relax_version=[True, False],
    )
    def test_keras_df_as_sample(
        self,
        model_fn: Callable[[tf.dtypes.DType], Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]],
        registry_test_fn: str,
        relax_version: bool,
    ) -> None:
        model, data_x, data_y = model_fn(tf.float32)
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)

        def assert_fn(res):
            y_pred_df = pd.DataFrame(y_pred)
            y_pred_df.columns = res.columns
            # res's shape:         (num_rows, 1, 1)
            # y_pred_df's shape:   (num_rows, 1)
            # convert list to scalar value before comparing
            for col in res.columns:
                res[col] = res[col].apply(lambda x: x[0])
            pd.testing.assert_frame_equal(
                res,
                y_pred_df,
                check_dtype=False,
            )

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
            options={"relax_version": relax_version},
        )

    @parameterized.product(  # type: ignore[misc]
        model_fn=[_prepare_keras_subclass_model, _prepare_keras_sequential_model, _prepare_keras_functional_model],
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_keras_sp(
        self,
        model_fn: Callable[[tf.dtypes.DType], Tuple[tf.keras.Model, tf.Tensor, tf.Tensor]],
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_fn(tf.float32)
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.predict(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self.session,
            x_df,
        )
        y_pred_df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        getattr(self, registry_test_fn)(
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
