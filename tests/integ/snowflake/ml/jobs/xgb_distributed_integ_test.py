from typing import Any

import pandas as pd
import xgboost
from absl.testing import absltest
from packaging import version

from snowflake.ml._internal import env
from snowflake.ml.jobs import remote
from snowflake.snowpark.context import get_active_session
from tests.integ.snowflake.ml.jobs.modeling_job_test_base import BaseModelTest

TEST_TABLE_NAME = "MULTINODE_CPU_TRAIN_DS"


class XGBDistributedTest(BaseModelTest):
    def prepare_dataset(self, num_rows: int = 10, num_cols: int = 5) -> None:
        self.session.sql(self.generate_dataset_sql(TEST_TABLE_NAME, num_rows, num_cols)).collect()

    def generate_dataset_sql(self, table_name: str, num_rows: int, num_cols: int) -> str:
        sql_script = f"CREATE TABLE IF NOT EXISTS {self.db}.{self.schema}.{table_name} AS \n"
        sql_script += "SELECT \n"
        for i in range(1, num_cols):
            sql_script += f"uniform(0::FLOAT, 10::FLOAT, random()) AS FEATURE_{i}, \n"
        sql_script += "FEATURE_1 + FEATURE_1 AS TARGET_1 \n"
        sql_script += f"FROM TABLE(generator(rowcount=>({num_rows})));"
        return sql_script

    def split_dataset(self) -> tuple["pd.DataFrame", "pd.DataFrame", str, list[float]]:
        cpu_df = self.session.table(TEST_TABLE_NAME)
        label_col = "TARGET_1"
        feature_cols = [col for col in cpu_df.columns if col != label_col]

        # Split the data
        train_df, test_df = cpu_df.random_split(weights=[0.8, 0.2])

        # Convert to pandas DataFrames
        train_pd = train_df.to_pandas()
        test_pd = test_df.to_pandas()

        return train_pd, test_pd, label_col, feature_cols

    @absltest.skipIf(  # type: ignore[misc]
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_xgb_distributed(self) -> None:
        @remote(
            self.compute_pool, stage_name="payload_stage", min_instances=2, target_instances=2, session=self.session
        )
        def distributed_xgb(table_name: str, input_cols: list[float], label_col: str) -> Any:
            from snowflake.ml.data import DataConnector
            from snowflake.ml.modeling.distributors.xgboost import (
                XGBEstimator,
                XGBScalingConfig,
            )

            # XGBoost parameters
            params = {"random_state": 42, "n_thread": 1}

            # Configure Ray scaling for XGBoost
            scaling_config = XGBScalingConfig(use_gpu=False, num_workers=2)

            # Create and configure the estimator
            estimator = XGBEstimator(
                n_estimators=10,
                params=params,
                scaling_config=scaling_config,
            )
            session = get_active_session()
            cpu_train_df = session.table(table_name)
            data_connector = DataConnector.from_dataframe(cpu_train_df.sample(0.8, seed=42))
            xgb_model = estimator.fit(data_connector, input_cols=input_cols, label_col=label_col)
            return xgb_model

        self.prepare_dataset()
        _, test_pd, label_col, feature_cols = self.split_dataset()
        job = distributed_xgb(TEST_TABLE_NAME, feature_cols, label_col)
        self.assertIsNotNone(job)
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        model_remote = job.result()
        inference_remote = model_remote.predict(xgboost.DMatrix(test_pd[feature_cols]))
        self.assertIsNotNone(inference_remote)


if __name__ == "__main__":
    absltest.main()
