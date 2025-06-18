from typing import Any

from absl.testing import absltest
from packaging import version

from snowflake.ml._internal import env
from snowflake.ml.jobs import remote
from snowflake.snowpark import DataFrame
from snowflake.snowpark.context import get_active_session
from tests.integ.snowflake.ml.jobs.modeling_job_test_base import BaseModelTest

TEST_TABLE_NAME = "MULTINODE_CPU_LIGHTGBM_TRAIN_DS"


def generate_dataset_sql(table_name: str, db: str, schema: str, num_rows: int = 10, num_cols: int = 5) -> str:
    sql_script = f"CREATE TABLE IF NOT EXISTS {db}.{schema}.{table_name} AS \n"
    sql_script += "SELECT \n"
    for i in range(1, num_cols):
        sql_script += f"uniform(0::FLOAT, 10::FLOAT, random()) AS FEATURE_{i}, \n"
    sql_script += "FEATURE_1 + FEATURE_1 AS TARGET_1 \n"
    sql_script += f"FROM TABLE(generator(rowcount=>({num_rows})));"
    return sql_script


def split_dataset(snowpark_df: DataFrame) -> tuple[DataFrame, DataFrame, str, list[str]]:
    label_col = "TARGET_1"
    feature_cols = [col for col in snowpark_df.columns if col != label_col]
    train_df, test_df = snowpark_df.random_split(weights=[0.8, 0.2])
    return train_df, test_df, label_col, feature_cols


class LightGBMDistributedTest(BaseModelTest):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        sql = generate_dataset_sql(TEST_TABLE_NAME, cls.db, cls.schema)
        cls.session.sql(sql).collect()

    @absltest.skipIf(  # type: ignore[misc]
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_lightgbm_distributed(self) -> None:
        @remote(self.compute_pool, stage_name="payload_stage", target_instances=2, session=self.session)
        def distributed_lightgbm(table_name: str, input_cols: list[str], label_col: str) -> Any:
            from snowflake.ml.data import DataConnector
            from snowflake.ml.modeling.distributors.lightgbm import (
                LightGBMEstimator,
                LightGBMScalingConfig,
            )

            # LightGBM parameters
            params = {"random_state": 42, "verbose": -1}

            # Configure scaling for LightGBM
            scaling_config = LightGBMScalingConfig(use_gpu=False, num_workers=2)

            # Create and configure the estimator
            estimator = LightGBMEstimator(
                n_estimators=10,
                params=params,
                scaling_config=scaling_config,
            )
            session = get_active_session()
            cpu_train_df = session.table(table_name)
            data_connector = DataConnector.from_dataframe(cpu_train_df.sample(0.8, seed=42))
            lgb_model = estimator.fit(data_connector, input_cols=input_cols, label_col=label_col)
            return lgb_model

        # Use the dataset prepared in setUpClass
        cpu_df = self.session.table(TEST_TABLE_NAME)
        _, test_df, label_col, feature_cols = split_dataset(cpu_df)

        job = distributed_lightgbm(TEST_TABLE_NAME, feature_cols, label_col)
        self.assertIsNotNone(job)
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        model_remote = job.result()

        # Convert to Pandas only when absolutely necessary for model inference
        test_pd = test_df.to_pandas()
        inference_remote = model_remote.predict(test_pd[feature_cols].values)
        self.assertIsNotNone(inference_remote)


if __name__ == "__main__":
    absltest.main()
