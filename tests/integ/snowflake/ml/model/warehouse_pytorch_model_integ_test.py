#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import db_manager


class TorchModel(torch.nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_out, dtype=dtype),
            torch.nn.Sigmoid(),
        )

    def forward(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.model(tensors[0])]


def _prepare_torch_model(
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, List[torch.Tensor], List[torch.Tensor]]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = [torch.from_numpy(x).to(dtype=dtype)]
    data_y = [(torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)]

    model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _epoch in range(100):
        pred_y = model(data_x)
        loss = loss_function(pred_y[0], data_y[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, data_x, data_y


class TestWarehousePytorchModelINteg(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.cleanup_schemas()
        self._db_manager.cleanup_stages()
        self._db_manager.cleanup_user_functions()

        # To create different UDF names among different runs
        self.run_id = uuid.uuid4().hex
        self._test_schema_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "model_deployment_pytorch_model_test_schema"
        )
        self._db_manager.create_schema(self._test_schema_name)
        self._db_manager.use_schema(self._test_schema_name)

        self.deploy_stage_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "deployment_stage"
        )
        self.full_qual_stage = self._db_manager.create_stage(
            self.deploy_stage_name, schema_name=self._test_schema_name, sse_encrypted=False
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_stage(self.deploy_stage_name, schema_name=self._test_schema_name)
        self._db_manager.drop_schema(self._test_schema_name)
        self._session.close()

    def base_test_case(
        self,
        name: str,
        model: model_types.SupportedModelType,
        sample_input: model_types.SupportedDataType,
        test_input: model_types.SupportedDataType,
        deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        warehouse_model_integ_test_utils.base_test_case(
            self._db_manager,
            run_id=self.run_id,
            full_qual_stage=self.full_qual_stage,
            name=name,
            model=model,
            sample_input=sample_input,
            test_input=test_input,
            deploy_params=deploy_params,
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_pytorch_tensor_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_torch_model()
        x_df = model_signature._SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x)[0].detach()

        self.base_test_case(
            name="pytorch_model_tensor_as_sample",
            model=model,
            sample_input=data_x,
            test_input=x_df,
            deploy_params={
                "forward": (
                    {},
                    lambda res: torch.testing.assert_close(  # type:ignore[attr-defined]
                        model_signature._SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_pytorch_df_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        x_df = model_signature._SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.forward(data_x)[0].detach()

        self.base_test_case(
            name="pytorch_model_df_as_sample",
            model=model,
            sample_input=x_df,
            test_input=x_df,
            deploy_params={
                "forward": (
                    {},
                    lambda res: torch.testing.assert_close(  # type:ignore[attr-defined]
                        model_signature._SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_pytorch_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        x_df = model_signature._SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.forward(data_x)[0].detach()
        x_df_sp = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, x_df, keep_order=True)

        self.base_test_case(
            name="pytorch_model_sp",
            model=model,
            sample_input=x_df,
            test_input=x_df_sp,
            deploy_params={
                "forward": (
                    {},
                    lambda res: torch.testing.assert_close(  # type:ignore[attr-defined]
                        model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                            model_signature._SnowparkDataFrameHandler.convert_to_df(res)
                        )[0],
                        y_pred,
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_torchscript_tensor_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_torch_model()
        x_df = model_signature._SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x)[0].detach()

        self.base_test_case(
            name="torch_script_model_tensor_as_sample",
            model=model_script,
            sample_input=data_x,
            test_input=x_df,
            deploy_params={
                "forward": (
                    {},
                    lambda res: torch.testing.assert_close(  # type:ignore[attr-defined]
                        model_signature._SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_torchscript_df_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        x_df = model_signature._SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x)[0].detach()

        self.base_test_case(
            name="torch_script_model_df_as_sample",
            model=model_script,
            sample_input=x_df,
            test_input=x_df,
            deploy_params={
                "forward": (
                    {},
                    lambda res: torch.testing.assert_close(  # type:ignore[attr-defined]
                        model_signature._SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_torchscript_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_torch_model(torch.float64)
        x_df = model_signature._SeqOfPyTorchTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = ["col_0"]
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x)[0].detach()
        x_df_sp = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, x_df, keep_order=True)

        self.base_test_case(
            name="torch_script_model_sp",
            model=model_script,
            sample_input=x_df,
            test_input=x_df_sp,
            deploy_params={
                "forward": (
                    {},
                    lambda res: torch.testing.assert_close(  # type:ignore[attr-defined]
                        model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                            model_signature._SnowparkDataFrameHandler.convert_to_df(res)
                        )[0],
                        y_pred,
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )


if __name__ == "__main__":
    absltest.main()
