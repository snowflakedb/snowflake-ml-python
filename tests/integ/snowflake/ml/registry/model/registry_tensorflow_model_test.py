import pandas as pd
import tensorflow as tf
from absl.testing import absltest

from snowflake.ml.model._signatures import snowpark_handler, tensorflow_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


class TestRegistryTensorflowModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_tf_tensor_as_sample(
        self,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model(data_x)

        def assert_fn(res):
            y_pred_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(
                y_pred,
                ensure_serializable=False,
            )
            y_pred_df.columns = res.columns
            pd.testing.assert_frame_equal(
                res,
                y_pred_df,
                check_dtype=False,
            )

        self._test_registry_model(
            model=model,
            sample_input_data=data_x,
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
        )

    def test_tf_df_as_sample(self) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model(data_x)

        def assert_fn(res):
            y_pred_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(
                y_pred,
                ensure_serializable=False,
            )
            y_pred_df.columns = res.columns
            pd.testing.assert_frame_equal(
                res,
                y_pred_df,
                check_dtype=False,
            )

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

    def test_tf_sp(self) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(x_df.shape[1])]
        y_pred = model(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self.session,
            x_df,
        )
        y_pred_df = tensorflow_handler.TensorflowTensorHandler.convert_to_df(y_pred)
        y_pred_df.columns = ["output_feature_0"]
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

    def test_tf_tensor_as_sample_multiple_inputs(self) -> None:
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

        self._test_registry_model(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_tf_df_as_sample_multiple_inputs(self) -> None:
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

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    assert_fn,
                ),
            },
            options={"multiple_inputs": True},
        )

    def test_tf_sp_multiple_inputs(
        self,
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

        self._test_registry_model(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
            options={"multiple_inputs": True},
        )


if __name__ == "__main__":
    absltest.main()
