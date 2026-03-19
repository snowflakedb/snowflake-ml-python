import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake.ml.model import custom_model
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

_NUM_PARTITIONS = 2
_NUM_ROWS_PER_PARTITION = 10
_PARTITIONED_OUTPUT_SIZE = 5


class NonPartitionedTableFunctionModel(custom_model.CustomModel):
    """A non-partitioned table function that performs a 1:1 row mapping.

    Each row is transformed independently: OUTPUT_COL1 = INPUT_COL1 * 2, OUTPUT_COL2 = INPUT_COL2 + 1.
    """

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "OUTPUT_COL1": input_df["INPUT_COL1"] * 2,
                "OUTPUT_COL2": input_df["INPUT_COL2"] + 1,
            }
        )


class PartitionedTableFunctionModel(custom_model.CustomModel):
    """A partitioned table function that fits a LinearRegression per partition.

    Predicts a fixed number of output rows per partition (M:N mapping).
    """

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(input_df[["INPUT_COL1"]], input_df["INPUT_COL2"])
        col1_min, col1_max = input_df["INPUT_COL1"].min(), input_df["INPUT_COL1"].max()
        output_col1_values = np.linspace(col1_min, col1_max, _PARTITIONED_OUTPUT_SIZE)
        preds = model.predict(output_col1_values.reshape(-1, 1))
        return pd.DataFrame({"OUTPUT_COL1": output_col1_values, "OUTPUT_COL2": preds})


def _generate_partitioned_data(num_rows_per_partition: int = _NUM_ROWS_PER_PARTITION) -> pd.DataFrame:
    """Generate synthetic data with two partitions."""
    np.random.seed(42)
    rows = []
    for partition_id in range(1, _NUM_PARTITIONS + 1):
        col1 = np.random.rand(num_rows_per_partition) * 10
        col2 = partition_id * 2.0 * col1 + partition_id * 3.0 + np.random.randn(num_rows_per_partition) * 0.1
        for c1, c2 in zip(col1, col2):
            rows.append({"PARTITION_COL": partition_id, "INPUT_COL1": c1, "INPUT_COL2": c2})
    return pd.DataFrame(rows)


@absltest.skip("Partitioned batch inference: server side not ready")
class TestBatchInferencePartitionedInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def setUp(self) -> None:
        super().setUp()
        if not self._has_image_override():
            self.skipTest("Skipping: image override env vars not set.")

    def test_regular_table_function_with_partition_column(self) -> None:
        """Case C: Non-partitioned table function with partition column (1:1 row mapping)."""
        input_pandas_df = _generate_partitioned_data()
        model = NonPartitionedTableFunctionModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)
        sample_input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        expected_output = model.predict(input_pandas_df)
        expected_with_input = pd.concat(
            [input_pandas_df.reset_index(drop=True), expected_output.reset_index(drop=True)], axis=1
        )

        def check_non_partitioned_output(actual: pd.DataFrame) -> None:
            self.assertEqual(len(actual), len(input_pandas_df))
            self.assertCountEqual(
                actual.columns.tolist(),
                ["PARTITION_COL", "INPUT_COL1", "INPUT_COL2", "OUTPUT_COL1", "OUTPUT_COL2"],
            )

            actual_sorted = actual.sort_values(["PARTITION_COL", "INPUT_COL1"]).reset_index(drop=True)
            expected_sorted = expected_with_input.sort_values(["PARTITION_COL", "INPUT_COL1"]).reset_index(drop=True)

            np.testing.assert_allclose(
                actual_sorted["OUTPUT_COL1"].astype(float).values,
                expected_sorted["OUTPUT_COL1"].astype(float).values,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                actual_sorted["OUTPUT_COL2"].astype(float).values,
                expected_sorted["OUTPUT_COL2"].astype(float).values,
                rtol=1e-5,
            )

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_df,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL"),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            prediction_assert_fn=check_non_partitioned_output,
        )

    def test_partitioned_table_function_with_partition_column(self) -> None:
        """Case E: Partitioned table function with partition column (M:N row mapping)."""
        input_pandas_df = _generate_partitioned_data(num_rows_per_partition=20)
        model = PartitionedTableFunctionModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)
        sample_input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def check_partitioned_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                ["PARTITION_COL", "OUTPUT_COL1", "OUTPUT_COL2"],
            )

            expected_total_rows = _NUM_PARTITIONS * _PARTITIONED_OUTPUT_SIZE
            self.assertEqual(len(actual), expected_total_rows)

            for partition_id in range(1, _NUM_PARTITIONS + 1):
                partition_input = input_pandas_df[input_pandas_df["PARTITION_COL"] == partition_id]
                lr = LinearRegression()
                lr.fit(partition_input[["INPUT_COL1"]], partition_input["INPUT_COL2"])
                col1_min, col1_max = partition_input["INPUT_COL1"].min(), partition_input["INPUT_COL1"].max()
                output_col1_values = np.linspace(col1_min, col1_max, _PARTITIONED_OUTPUT_SIZE)
                expected_preds = np.sort(lr.predict(output_col1_values.reshape(-1, 1)))

                actual_partition = actual[actual["PARTITION_COL"] == partition_id]
                actual_preds = np.sort(actual_partition["OUTPUT_COL2"].astype(float).values)
                np.testing.assert_allclose(actual_preds, expected_preds, rtol=1e-5)

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_df,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL"),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict",
            ),
            additional_dependencies=["scikit-learn"],
            options={"function_type": "TABLE_FUNCTION"},
            prediction_assert_fn=check_partitioned_output,
            skip_row_count_check=True,
        )


if __name__ == "__main__":
    absltest.main()
