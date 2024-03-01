import os
import sys
from typing import Any
from unittest import mock

import inflection
from absl.testing import absltest
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from snowflake.ml.modeling._internal.constants import IN_ML_RUNTIME_ENV_VAR
from snowflake.ml.modeling._internal.ml_runtime_implementations.ml_runtime_trainer import (
    MlRuntimeModelTrainer,
)
from snowflake.ml.modeling._internal.model_trainer_builder import ModelTrainerBuilder
from snowflake.ml.modeling._internal.snowpark_implementations.distributed_hpo_trainer import (
    DistributedHPOTrainer,
)
from snowflake.ml.modeling._internal.snowpark_implementations.snowpark_trainer import (
    SnowparkModelTrainer,
)
from snowflake.ml.modeling._internal.snowpark_implementations.xgboost_external_memory_trainer import (
    XGBoostExternalMemoryTrainer,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session


class SnowparkHandlersUnitTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()
        os.environ.pop(IN_ML_RUNTIME_ENV_VAR, None)

    def get_snowpark_dataset(self) -> DataFrame:
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df: DataFrame = self._session.create_dataframe(input_df_pandas)
        return input_df

    def test_sklearn_model_trainer(self) -> None:
        model = LinearRegression()
        dataset = self.get_snowpark_dataset()
        trainer = ModelTrainerBuilder.build(estimator=model, dataset=dataset, input_cols=[])

        self.assertTrue(isinstance(trainer, SnowparkModelTrainer))

    def test_sklearn_model_trainer_in_ml_runtime(self) -> None:
        model = LinearRegression()
        dataset = self.get_snowpark_dataset()
        os.environ[IN_ML_RUNTIME_ENV_VAR] = "True"

        with absltest.mock.patch.dict(sys.modules, {**sys.modules, **{"snowflake.ml.runtime": absltest.mock.Mock()}}):
            trainer = ModelTrainerBuilder.build(estimator=model, dataset=dataset, input_cols=[])
        del os.environ[IN_ML_RUNTIME_ENV_VAR]

        self.assertTrue(isinstance(trainer, MlRuntimeModelTrainer))

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_distributed_hpo_trainer(self, mock_is_single_node: Any) -> None:
        mock_is_single_node.return_value = False
        dataset = self.get_snowpark_dataset()
        model = GridSearchCV(estimator=LinearRegression(), param_grid={"loss": ["rmsqe", "mae"]})
        trainer = ModelTrainerBuilder.build(estimator=model, dataset=dataset, input_cols=[])

        self.assertTrue(isinstance(trainer, DistributedHPOTrainer))

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_single_node_hpo_trainer(self, mock_is_single_node: Any) -> None:
        mock_is_single_node.return_value = True
        dataset = self.get_snowpark_dataset()
        model = GridSearchCV(estimator=LinearRegression(), param_grid={"loss": ["rmsqe", "mae"]})
        trainer = ModelTrainerBuilder.build(estimator=model, dataset=dataset, input_cols=[])

        self.assertTrue(isinstance(trainer, SnowparkModelTrainer))

    def test_xgboost_external_memory_model_trainer(self) -> None:
        model = XGBRegressor()
        dataset = self.get_snowpark_dataset()
        trainer = ModelTrainerBuilder.build(
            estimator=model, dataset=dataset, input_cols=[], use_external_memory_version=True, batch_size=1000
        )

        self.assertTrue(isinstance(trainer, XGBoostExternalMemoryTrainer))

    def test_xgboost_standard_model_trainer(self) -> None:
        model = XGBRegressor()
        dataset = self.get_snowpark_dataset()
        trainer = ModelTrainerBuilder.build(
            estimator=model,
            dataset=dataset,
            input_cols=[],
        )

        self.assertTrue(isinstance(trainer, SnowparkModelTrainer))


if __name__ == "__main__":
    absltest.main()
