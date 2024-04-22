import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from absl.testing import absltest, parameterized

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import (
    numpy_handler,
    snowpark_handler,
    tensorflow_handler,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import (
    dataframe_utils,
    db_manager,
    model_factory,
)


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
        sample_input_data: model_types.SupportedDataType,
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
            sample_input_data=sample_input_data,
            test_input=test_input,
            deploy_params=deploy_params,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_tf_tensor_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:

        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        self.base_test_case(
            name="tf_model_tensor_as_sample",
            model=model,
            sample_input_data=[data_x],
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred.numpy(),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_tf_df_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x = model_factory.ModelFactory.prepare_tf_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model(data_x)

        self.base_test_case(
            name="tf_model_df_as_sample",
            model=model,
            sample_input_data=x_df,
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred.numpy(),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_tf_sp(
        self,
        permanent_deploy: Optional[bool] = False,
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

        self.base_test_case(
            name="tf_model_sp",
            model=model,
            sample_input_data=x_df,
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
    def test_keras_tensor_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)

        self.base_test_case(
            name="keras_model_tensor_as_sample",
            model=model,
            sample_input_data=[data_x],
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred,
                        atol=1e-6,
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_keras_df_as_sample(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        model, data_x, data_y = prepare_keras_model()
        x_df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df([data_x], ensure_serializable=False)
        y_pred = model.predict(data_x)

        self.base_test_case(
            name="keras_model_df_as_sample",
            model=model,
            sample_input_data=x_df,
            test_input=x_df,
            deploy_params={
                "": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(res)[0].numpy(),
                        y_pred,
                        atol=1e-6,
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_keras_sp(
        self,
        permanent_deploy: Optional[bool] = False,
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

        self.base_test_case(
            name="keras_model_sp",
            model=model,
            sample_input_data=x_df,
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
