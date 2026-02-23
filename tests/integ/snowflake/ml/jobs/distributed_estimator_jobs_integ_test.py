from typing import Any

import pandas as pd
import xgboost
from absl.testing import absltest

from snowflake.ml.jobs import remote
from snowflake.snowpark import DataFrame
from snowflake.snowpark.context import get_active_session
from tests.integ.snowflake.ml.jobs.job_test_base import JobTestBase


class DistributedEstimatorTestBase(JobTestBase):
    """Base class for distributed estimator tests (XGBoost, LightGBM, etc.)."""

    table_name = "MULTINODE_CPU_TRAIN_DS"

    @staticmethod
    def _generate_dataset_sql(table_name: str, db: str, schema: str, num_rows: int = 10, num_cols: int = 5) -> str:
        """Generate SQL to create a synthetic dataset for distributed training tests."""
        sql_script = f"CREATE TABLE IF NOT EXISTS {db}.{schema}.{table_name} AS \n"
        sql_script += "SELECT \n"
        for i in range(1, num_cols):
            sql_script += f"uniform(0::FLOAT, 10::FLOAT, random()) AS FEATURE_{i}, \n"
        sql_script += "FEATURE_1 + FEATURE_1 AS TARGET_1 \n"
        sql_script += f"FROM TABLE(generator(rowcount=>({num_rows})));"
        return sql_script

    @staticmethod
    def _split_dataset(snowpark_df: DataFrame) -> tuple[DataFrame, DataFrame, str, list[str]]:
        """Split dataset into train/test and extract feature/label columns."""
        label_col = "TARGET_1"
        feature_cols = [col for col in snowpark_df.columns if col != label_col]
        train_df, test_df = snowpark_df.random_split(weights=[0.8, 0.2])
        return train_df, test_df, label_col, feature_cols

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test dataset once for all tests in the class."""
        super().setUpClass()
        if cls.table_name:
            sql = cls._generate_dataset_sql(cls.table_name, cls.db, cls.schema)
            cls.session.sql(sql).collect()

    def create_training_function(self) -> Any:
        """
        Create and return a @remote decorated training function.

        Returns:
            A remote function that trains the model and returns the trained estimator.
        """
        raise NotImplementedError

    def prepare_prediction_data(self, test_df: pd.DataFrame, feature_cols: list[str]) -> Any:
        """
        Prepare test data for prediction based on the estimator's requirements.

        Args:
            test_df: Test DataFrame
            feature_cols: List of feature column names

        Returns:
            Data in the format expected by the model's predict method.
        """
        raise NotImplementedError

    def _run_distributed_training_test(self) -> None:
        """Common test logic for distributed training."""
        # Prepare dataset and split
        cpu_df = self.session.table(self.table_name)
        _, test_df, label_col, feature_cols = self._split_dataset(cpu_df)
        test_pd = test_df.to_pandas()

        # Get training function and submit job
        training_func = self.create_training_function()
        job = training_func(self.table_name, feature_cols, label_col)

        # Verify job completion
        self.assertIsNotNone(job)
        self.assertEqual(job.wait(), "DONE", job.get_logs())

        # Get model and run inference
        model_remote = job.result()
        prediction_data = self.prepare_prediction_data(test_pd, feature_cols)
        inference_remote = model_remote.predict(prediction_data)
        self.assertIsNotNone(inference_remote)


class XGBDistributedTest(DistributedEstimatorTestBase):
    """Test XGBoost distributed training on multi-node compute pool."""

    def create_training_function(self) -> Any:
        """Create XGBoost training function with @remote decorator."""

        @remote(
            self.compute_pool,
            stage_name="payload_stage",
            min_instances=2,
            target_instances=2,
            session=self.session,
        )
        def train_xgb(table_name: str, input_cols: list[str], label_col: str) -> Any:
            from snowflake.ml.data import DataConnector
            from snowflake.ml.modeling.distributors.xgboost import (
                XGBEstimator,
                XGBScalingConfig,
            )

            params = {"random_state": 42, "n_thread": 1}
            scaling_config = XGBScalingConfig(use_gpu=False, num_workers=2)
            estimator = XGBEstimator(n_estimators=10, params=params, scaling_config=scaling_config)

            session = get_active_session()
            train_df = session.table(table_name)
            data_connector = DataConnector.from_dataframe(train_df.sample(0.8, seed=42))
            return estimator.fit(data_connector, input_cols=input_cols, label_col=label_col)

        return train_xgb

    def prepare_prediction_data(self, test_df: pd.DataFrame, feature_cols: list[str]) -> Any:
        """Prepare data as XGBoost DMatrix."""
        return xgboost.DMatrix(test_df[feature_cols])

    def test_xgb_distributed(self) -> None:
        self._run_distributed_training_test()


class LightGBMDistributedTest(DistributedEstimatorTestBase):
    """Test LightGBM distributed training on multi-node compute pool."""

    def create_training_function(self) -> Any:
        """Create LightGBM training function with @remote decorator."""

        @remote(
            self.compute_pool,
            stage_name="payload_stage",
            target_instances=2,
            session=self.session,
        )
        def train_lightgbm(table_name: str, input_cols: list[str], label_col: str) -> Any:
            from snowflake.ml.data import DataConnector
            from snowflake.ml.modeling.distributors.lightgbm import (
                LightGBMEstimator,
                LightGBMScalingConfig,
            )

            params = {"random_state": 42, "verbose": -1}
            scaling_config = LightGBMScalingConfig(use_gpu=False, num_workers=2)
            estimator = LightGBMEstimator(n_estimators=10, params=params, scaling_config=scaling_config)

            session = get_active_session()
            train_df = session.table(table_name)
            data_connector = DataConnector.from_dataframe(train_df.sample(0.8, seed=42))
            return estimator.fit(data_connector, input_cols=input_cols, label_col=label_col)

        return train_lightgbm

    def prepare_prediction_data(self, test_df: pd.DataFrame, feature_cols: list[str]) -> Any:
        """Prepare data as numpy array for LightGBM."""
        return test_df[feature_cols].values

    def test_lightgbm_distributed(self) -> None:
        self._run_distributed_training_test()


if __name__ == "__main__":
    absltest.main()
