import pandas as pd
import torch
from absl.testing import absltest

from snowflake.ml.model._signatures import pytorch_handler, snowpark_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


class TestRegistryPytorchModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_pytorch_tensor_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model()
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model(
            model=model,
            sample_input=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
        )

    def test_pytorch_df_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self._test_registry_model(
            model=model,
            sample_input=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
        )

    def test_pytorch_sp(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
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

    def test_torchscript_tensor_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model()
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x).detach()

        self._test_registry_model(
            model=model_script,
            sample_input=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
        )

    def test_torchscript_df_as_sample(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x).detach()

        self._test_registry_model(
            model=model_script,
            sample_input=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
        )

    def test_torchscript_sp(
        self,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self._test_registry_model(
            model=model_script,
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
