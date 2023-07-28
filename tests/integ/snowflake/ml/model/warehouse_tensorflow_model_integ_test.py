#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from absl.testing import absltest, parameterized

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import snowpark_handler, tensorflow_handler
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import db_manager


class SimpleModule(tf.Module):
    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    @tf.function(input_signature=[[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)]])  # type: ignore[misc]
    def __call__(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        return [self.a_variable * tensors[0] + self.non_trainable_variable]


class KerasModel(tf.keras.Model):
    def __init__(self, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
        self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

    def call(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        input = tensors[0]
        x = self.fc_1(input)
        x = self.fc_2(x)
        return [x]


def _prepare_keras_model(
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[tf.keras.Model, List[tf.Tensor], List[tf.Tensor]]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    data_x = [tf.convert_to_tensor(x, dtype=dtype)]
    raw_data_y = tf.random.uniform((batch_size, 1))
    raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
    data_y = [tf.cast(raw_data_y, dtype=dtype)]

    def loss_fn(y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        return tf.keras.losses.mse(y_true[0], y_pred[0])

    model = KerasModel(n_hidden, n_out)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=loss_fn)
    model.fit(data_x, data_y, batch_size=batch_size, epochs=100)
    return model, data_x, data_y


class TestWarehouseTensorflowModelInteg(parameterized.TestCase):
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
            self.run_id, "model_deployment_tensorflow_model_test_schema"
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
    def test_tf_tensor_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model = SimpleModule(name="simple")
        data_x = [tf.constant([[5.0], [10.0]])]
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model(data_x)

        self.base_test_case(
            name="tf_model_tensor_as_sample",
            model=model,
            sample_input=data_x,
            test_input=x_df,
            deploy_params={
                "__call__": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred[0].numpy(),
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
    def test_tf_df_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model = SimpleModule(name="simple")
        data_x = [tf.constant([[5.0], [10.0]])]
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model(data_x)

        self.base_test_case(
            name="tf_model_df_as_sample",
            model=model,
            sample_input=x_df,
            test_input=x_df,
            deploy_params={
                "__call__": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred[0].numpy(),
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
    def test_tf_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model = SimpleModule(name="simple")
        data_x = [tf.constant([[5.0], [10.0]])]
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model(data_x)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self._session,
            x_df,
            keep_order=True,
        )

        self.base_test_case(
            name="tf_model_sp",
            model=model,
            sample_input=x_df,
            test_input=x_df_sp,
            deploy_params={
                "__call__": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                            snowpark_handler.SnowparkDataFrameHandler.convert_to_df(res)
                        )[0].numpy(),
                        y_pred[0].numpy(),
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
    def test_keras_tensor_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.predict(data_x)[0]

        self.base_test_case(
            name="keras_model_tensor_as_sample",
            model=model,
            sample_input=data_x,
            test_input=x_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred,
                        atol=1e-6,
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
    def test_keras_df_as_sample(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        y_pred = model.predict(data_x)[0]

        self.base_test_case(
            name="keras_model_df_as_sample",
            model=model,
            sample_input=x_df,
            test_input=x_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred,
                        atol=1e-6,
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
    def test_keras_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = _prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data_x, ensure_serializable=False)
        x_df.columns = ["col_0"]
        y_pred = model.predict(data_x)[0]
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self._session,
            x_df,
            keep_order=True,
        )

        self.base_test_case(
            name="keras_model_sp",
            model=model,
            sample_input=x_df,
            test_input=x_df_sp,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                            snowpark_handler.SnowparkDataFrameHandler.convert_to_df(res)
                        )[0].numpy(),
                        y_pred,
                        atol=1e-6,
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )


if __name__ == "__main__":
    absltest.main()
