import pandas as pd
import tensorflow as tf
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import snowpark_handler, tensorflow_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


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


if __name__ == "__main__":
    absltest.main()
