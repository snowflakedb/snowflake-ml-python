import pandas as pd
import torch
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import pytorch_handler, snowpark_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, model_factory


class TestRegistryPytorchModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_pytorch_tensor_as_sample(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float32)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=data_x,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred, check_dtype=False
                    ),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_pytorch_df_as_sample(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred
                    ),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_pytorch_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]
        y_pred = model.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(y_pred)
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
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_torchscript_tensor_as_sample(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float32)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model_script,
            sample_input_data=data_x,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred, check_dtype=False
                    ),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_torchscript_df_as_sample(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.PyTorchTensorHandler.convert_from_df(res), y_pred
                    ),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_torchscript_sp(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = [f"col_{i}" for i in range(data_x.shape[1])]
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.PyTorchTensorHandler.convert_to_df(y_pred)
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        getattr(self, registry_test_fn)(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_pytorch_tensor_as_sample_multiple_inputs(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float32)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_pytorch_df_as_sample_multiple_inputs(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_pytorch_sp_multiple_inputs(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
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
            options={"multiple_inputs": True},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_torchscript_tensor_as_sample_multiple_inputs(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float32)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model_script,
            sample_input_data=[data_x],
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_torchscript_df_as_sample_multiple_inputs(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x).detach()

        getattr(self, registry_test_fn)(
            model=model_script,
            sample_input_data=x_df,
            prediction_assert_fns={
                "": (
                    x_df,
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            options={"multiple_inputs": True},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_torchscript_sp_multiple_inputs(
        self,
        registry_test_fn: str,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        model_script = torch.jit.script(model)
        y_pred = model_script.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        getattr(self, registry_test_fn)(
            model=model_script,
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
