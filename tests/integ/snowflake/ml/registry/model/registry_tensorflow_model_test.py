from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from absl.testing import absltest

from snowflake.ml.model._signatures import (
    numpy_handler,
    snowpark_handler,
    tensorflow_handler,
)
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


class SimpleModule(tf.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])  # type: ignore[misc]
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        return self.a_variable * tensor + self.non_trainable_variable


class TestRegistryTensorflowModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_tf_tensor_as_sample(
        self,
    ) -> None:
        model = SimpleModule(name="simple")
        data_x = tf.constant([[5.0], [10.0]])
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        self._test_registry_model(
            model=model,
            sample_input=[data_x],
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
        model = SimpleModule(name="simple")
        data_x = tf.constant([[5.0], [10.0]])
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        self._test_registry_model(
            model=model,
            sample_input=x_df,
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
        model = SimpleModule(name="simple")
        data_x = tf.constant([[5.0], [10.0]])
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
            sample_input=x_df,
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
        model, data_x, data_y = model_factory.ModelFactory.prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)
        self._test_registry_model(
            model=model,
            sample_input=[data_x],
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
        model, data_x, data_y = model_factory.ModelFactory.prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)
        self._test_registry_model(
            model=model,
            sample_input=x_df,
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
        model, data_x, data_y = model_factory.ModelFactory.prepare_keras_model()
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
            sample_input=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
