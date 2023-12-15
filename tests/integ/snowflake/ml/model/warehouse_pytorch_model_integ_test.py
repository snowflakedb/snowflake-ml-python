import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from absl.testing import absltest, parameterized

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import pytorch_handler, snowpark_handler
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import (
    dataframe_utils,
    db_manager,
    model_factory,
)


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
        permanent_deploy: Optional[bool] = False,
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
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_pytorch_tensor_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model()
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self.base_test_case(
            name="pytorch_model_tensor_as_sample",
            model=model,
            sample_input=[data_x],
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_pytorch_df_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.forward(data_x).detach()

        self.base_test_case(
            name="pytorch_model_df_as_sample",
            model=model,
            sample_input=x_df,
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_pytorch_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.forward(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, x_df)
        y_pred_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([y_pred])
        y_pred_df.columns = ["output_feature_0"]
        y_df_expected = pd.concat([x_df, y_pred_df], axis=1)

        self.base_test_case(
            name="pytorch_model_sp",
            model=model,
            sample_input=x_df,
            test_input=x_df_sp,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_torchscript_tensor_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model()
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x).detach()

        self.base_test_case(
            name="torch_script_model_tensor_as_sample",
            model=model_script,
            sample_input=[data_x],
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred, check_dtype=False
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_torchscript_df_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = model_factory.ModelFactory.prepare_jittable_torch_model(torch.float64)
        x_df = pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        model_script = torch.jit.script(model)  # type:ignore[attr-defined]
        y_pred = model_script.forward(data_x).detach()

        self.base_test_case(
            name="torch_script_model_df_as_sample",
            model=model_script,
            sample_input=x_df,
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: torch.testing.assert_close(
                        pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(res)[0], y_pred
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_torchscript_sp(
        self,
        permanent_deploy: Optional[bool] = False,
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

        self.base_test_case(
            name="torch_script_model_sp",
            model=model_script,
            sample_input=x_df,
            test_input=x_df_sp,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                ),
            },
            permanent_deploy=permanent_deploy,
        )


if __name__ == "__main__":
    absltest.main()
