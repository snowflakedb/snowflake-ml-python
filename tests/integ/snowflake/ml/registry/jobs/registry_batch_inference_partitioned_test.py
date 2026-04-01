import inspect
import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake import snowpark
from snowflake.ml.model import custom_model, model_signature, openai_signatures
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

_NUM_PARTITIONS = 2
_NUM_ROWS_PER_PARTITION = 10
_PARTITIONED_REDUCED_OUTPUT_SIZE = 5
_PARTITIONED_EXPANDED_OUTPUT_SIZE = 20

_PARTITIONED_INPUT_FEATURES = [
    model_signature.FeatureSpec("PARTITION_COL", model_signature.DataType.INT64),
    model_signature.FeatureSpec("INPUT_COL1", model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec("INPUT_COL2", model_signature.DataType.DOUBLE),
]

_OUTPUT_FEATURES = [
    model_signature.FeatureSpec("OUTPUT_COL1", model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec("OUTPUT_COL2", model_signature.DataType.DOUBLE),
]

_DEFAULT_PARAM_INT = 1
_DEFAULT_PARAM_FLOAT = 0.1
_DEFAULT_PARAM_BOOL = False
_OVERRIDE_PARAM_INT = 3
_OVERRIDE_PARAM_FLOAT = 0.5
_OVERRIDE_PARAM_BOOL = True

_PARAM_SPECS = [
    model_signature.ParamSpec(name="param_int", dtype=model_signature.DataType.INT64, default_value=_DEFAULT_PARAM_INT),
    model_signature.ParamSpec(
        name="param_float", dtype=model_signature.DataType.DOUBLE, default_value=_DEFAULT_PARAM_FLOAT
    ),
    model_signature.ParamSpec(
        name="param_bool", dtype=model_signature.DataType.BOOL, default_value=_DEFAULT_PARAM_BOOL
    ),
]

_PARTITIONED_MODEL_SIGNATURES = {
    "predict_stateful": model_signature.ModelSignature(
        inputs=_PARTITIONED_INPUT_FEATURES,
        outputs=[
            model_signature.FeatureSpec("OUTPUT_PARTITION_ID", model_signature.DataType.INT64),
            *_OUTPUT_FEATURES,
        ],
    ),
    "predict_reduced": model_signature.ModelSignature(
        inputs=_PARTITIONED_INPUT_FEATURES,
        outputs=_OUTPUT_FEATURES,
    ),
    "predict_equal": model_signature.ModelSignature(
        inputs=_PARTITIONED_INPUT_FEATURES,
        outputs=_OUTPUT_FEATURES,
    ),
    "predict_expanded": model_signature.ModelSignature(
        inputs=_PARTITIONED_INPUT_FEATURES,
        outputs=_OUTPUT_FEATURES,
    ),
}

_PARAM_OUTPUT_FEATURES = [
    *_OUTPUT_FEATURES,
    model_signature.FeatureSpec("OUTPUT_PARAM_INT", model_signature.DataType.INT64),
    model_signature.FeatureSpec("OUTPUT_PARAM_FLOAT", model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec("OUTPUT_PARAM_BOOL", model_signature.DataType.BOOL),
]

_PARTITIONED_MODEL_WITH_PARAMS_SIGNATURES = {
    **_PARTITIONED_MODEL_SIGNATURES,
    "predict_stateful_with_params": model_signature.ModelSignature(
        inputs=_PARTITIONED_INPUT_FEATURES,
        outputs=[
            model_signature.FeatureSpec("OUTPUT_PARTITION_ID", model_signature.DataType.INT64),
            *_PARAM_OUTPUT_FEATURES,
        ],
        params=_PARAM_SPECS,
    ),
    "predict_expanded_with_params": model_signature.ModelSignature(
        inputs=_PARTITIONED_INPUT_FEATURES,
        outputs=_PARAM_OUTPUT_FEATURES,
        params=_PARAM_SPECS,
    ),
}


class BatchTableFunctionModel(custom_model.CustomModel):
    """A non-partitioned table function model for basic batch processing."""

    @custom_model.inference_api
    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "OUTPUT_COL1": input_df["INPUT_COL1"] * 2,
                "OUTPUT_COL2": input_df["INPUT_COL2"] + 1,
            }
        )


class PartitionedModel(custom_model.CustomModel):
    """A partitioned/non-partitioned table function model with methods for different cases.

    Includes both stateless and stateful partitioned methods.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        self.partition_id_cache = None

    @custom_model.inference_api
    def predict_stateful(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """A non-partitioned table function that validates partition uniformity (stateful).

        Caches the partition ID and asserts all rows in a batch are from the same partition,
        then performs transformation.
        """
        if input_df["PARTITION_COL"].nunique() != 1:
            raise ValueError(
                f"Mixed partitions in batch: expected all rows to be from the same partition, "
                f"but found {input_df['PARTITION_COL'].unique().tolist()}"
            )

        if self.partition_id_cache is None:
            self.partition_id_cache = input_df["PARTITION_COL"].iloc[0]

        return pd.DataFrame(
            {
                "OUTPUT_PARTITION_ID": self.partition_id_cache,
                "OUTPUT_COL1": input_df["INPUT_COL1"] * 2,
                "OUTPUT_COL2": input_df["INPUT_COL2"] + 1,
            }
        )

    @custom_model.partitioned_api
    def predict_reduced(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """M>N: Reduces input to fewer output rows via linear regression (stateless)."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(input_df[["INPUT_COL1"]], input_df["INPUT_COL2"])
        col1_min, col1_max = input_df["INPUT_COL1"].min(), input_df["INPUT_COL1"].max()
        output_col1_values = np.linspace(col1_min, col1_max, _PARTITIONED_REDUCED_OUTPUT_SIZE)
        preds = model.predict(output_col1_values.reshape(-1, 1))
        return pd.DataFrame({"OUTPUT_COL1": output_col1_values, "OUTPUT_COL2": preds})

    @custom_model.partitioned_api
    def predict_equal(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """M=N: Returns the same number of rows as input (stateless)."""
        return pd.DataFrame(
            {
                "OUTPUT_COL1": input_df["INPUT_COL1"] * 2,
                "OUTPUT_COL2": input_df["INPUT_COL2"] + 1,
            }
        )

    @custom_model.partitioned_api
    def predict_expanded(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """M<N: Expands input to more output rows (stateless)."""
        col1_min, col1_max = input_df["INPUT_COL1"].min(), input_df["INPUT_COL1"].max()
        col2_min, col2_max = input_df["INPUT_COL2"].min(), input_df["INPUT_COL2"].max()
        return pd.DataFrame(
            {
                "OUTPUT_COL1": np.linspace(col1_min, col1_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE),
                "OUTPUT_COL2": np.linspace(col2_min, col2_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE),
            }
        )

    @custom_model.inference_api
    def predict_stateful_with_params(
        self,
        input_df: pd.DataFrame,
        *,
        param_int: int = _DEFAULT_PARAM_INT,
        param_float: float = _DEFAULT_PARAM_FLOAT,
        param_bool: bool = _DEFAULT_PARAM_BOOL,
    ) -> pd.DataFrame:
        """Like predict_stateful but echoes params as additional output columns."""
        if input_df["PARTITION_COL"].nunique() != 1:
            raise ValueError(
                f"Mixed partitions in batch: expected all rows to be from the same partition, "
                f"but found {input_df['PARTITION_COL'].unique().tolist()}"
            )

        if self.partition_id_cache is None:
            self.partition_id_cache = input_df["PARTITION_COL"].iloc[0]

        return pd.DataFrame(
            {
                "OUTPUT_PARTITION_ID": self.partition_id_cache,
                "OUTPUT_COL1": input_df["INPUT_COL1"] * 2,
                "OUTPUT_COL2": input_df["INPUT_COL2"] + 1,
                "OUTPUT_PARAM_INT": param_int,
                "OUTPUT_PARAM_FLOAT": param_float,
                "OUTPUT_PARAM_BOOL": param_bool,
            }
        )

    @custom_model.partitioned_api
    def predict_expanded_with_params(
        self,
        input_df: pd.DataFrame,
        *,
        param_int: int = _DEFAULT_PARAM_INT,
        param_float: float = _DEFAULT_PARAM_FLOAT,
        param_bool: bool = _DEFAULT_PARAM_BOOL,
    ) -> pd.DataFrame:
        """Like predict_expanded but echoes params as additional output columns."""
        col1_min, col1_max = input_df["INPUT_COL1"].min(), input_df["INPUT_COL1"].max()
        col2_min, col2_max = input_df["INPUT_COL2"].min(), input_df["INPUT_COL2"].max()
        return pd.DataFrame(
            {
                "OUTPUT_COL1": np.linspace(col1_min, col1_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE),
                "OUTPUT_COL2": np.linspace(col2_min, col2_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE),
                "OUTPUT_PARAM_INT": param_int,
                "OUTPUT_PARAM_FLOAT": param_float,
                "OUTPUT_PARAM_BOOL": param_bool,
            }
        )


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


class TestBatchInferencePartitionedInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        cls.cache_dir.cleanup()

    def setUp(self) -> None:
        super().setUp()
        if not self._has_image_override():
            self.skipTest("Skipping: image override env vars not set.")

    def _compare_with_warehouse(
        self,
        output_stage_location: str,
        model_name: str,
        version_name: str,
        input_df: snowpark.DataFrame,
        function_name: str,
        sort_columns: list[str],
        partition_column: Optional[str] = None,
    ) -> None:
        """Compare batch inference job output (from stage) with warehouse output."""
        mv = self.registry.get_model(model_name).version(version_name)

        job_output = self.session.read.option("pattern", ".*\\.parquet").parquet(output_stage_location).to_pandas()

        run_kwargs: dict[str, str] = {"function_name": function_name}
        if partition_column is not None:
            run_kwargs["partition_column"] = partition_column
        warehouse_output = mv.run(input_df, **run_kwargs)
        if isinstance(warehouse_output, snowpark.DataFrame):
            warehouse_output = warehouse_output.to_pandas()

        job_cmp = job_output.sort_values(sort_columns).reset_index(drop=True)
        wh_cmp = warehouse_output.sort_values(sort_columns).reset_index(drop=True)

        pd.testing.assert_frame_equal(job_cmp, wh_cmp, check_dtype=False, rtol=1e-5)

    def test_non_partitioned_table_function_without_partition_column(self) -> None:
        """Case B: Non-partitioned table function without partition_column (1:1 row mapping, batch)."""
        input_pandas_df = pd.DataFrame(
            {
                "INPUT_COL1": [1.0, 2.0, 3.0, 4.0],
                "INPUT_COL2": [2.0, 4.0, 6.0, 8.0],
            }
        )
        model = BatchTableFunctionModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)
        sample_input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        model_name = f"model_{inspect.stack()[0].function}"
        version_name = f"ver_{self._run_id}"

        def check_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                ["INPUT_COL1", "INPUT_COL2", "OUTPUT_COL1", "OUTPUT_COL2"],
            )
            # 1:1 row mapping
            self.assertEqual(len(actual), len(input_pandas_df))

            actual_sorted = actual.sort_values("INPUT_COL1").reset_index(drop=True)
            expected_sorted = input_pandas_df.sort_values("INPUT_COL1").reset_index(drop=True)

            np.testing.assert_allclose(
                actual_sorted["OUTPUT_COL1"].astype(float).values,
                expected_sorted["INPUT_COL1"].astype(float).values * 2,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                actual_sorted["OUTPUT_COL2"].astype(float).values,
                expected_sorted["INPUT_COL2"].astype(float).values + 1,
                rtol=1e-5,
            )

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_df,
            X=input_df,
            # No InputSpec / no partition_column
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_batch",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            prediction_assert_fn=check_output,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            model_name=model_name,
            version_name=version_name,
        )

        self._compare_with_warehouse(
            output_stage_location=output_stage_location,
            model_name=model_name,
            version_name=version_name,
            input_df=input_df,
            function_name="predict_batch",
            sort_columns=["OUTPUT_COL1"],
        )

    def test_non_partitioned_table_function_with_partition_column(self) -> None:
        """Case C: Non-partitioned table function with partition column (1:1 row mapping)."""
        input_pandas_df = _generate_partitioned_data(num_rows_per_partition=2)
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        model_name = f"model_{inspect.stack()[0].function}"
        version_name = f"ver_{self._run_id}"

        expected_output = pd.DataFrame(
            {
                "OUTPUT_COL1": input_pandas_df["INPUT_COL1"] * 2,
                "OUTPUT_COL2": input_pandas_df["INPUT_COL2"] + 1,
                "OUTPUT_PARTITION_ID": input_pandas_df["PARTITION_COL"],
            }
        )
        expected_with_input = pd.concat(
            [input_pandas_df.reset_index(drop=True), expected_output.reset_index(drop=True)], axis=1
        )

        def check_non_partitioned_output(actual: pd.DataFrame) -> None:
            self.assertEqual(len(actual), len(input_pandas_df))
            self.assertCountEqual(
                actual.columns.tolist(),
                ["PARTITION_COL", "INPUT_COL1", "INPUT_COL2", "OUTPUT_COL1", "OUTPUT_COL2", "OUTPUT_PARTITION_ID"],
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
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL"),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_stateful",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_SIGNATURES,
            prediction_assert_fn=check_non_partitioned_output,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            model_name=model_name,
            version_name=version_name,
        )

        self._compare_with_warehouse(
            output_stage_location=output_stage_location,
            model_name=model_name,
            version_name=version_name,
            input_df=input_df,
            function_name="predict_stateful",
            partition_column="PARTITION_COL",
            sort_columns=["PARTITION_COL", "OUTPUT_COL1"],
        )

    def test_partitioned_table_function_without_partition_column(self) -> None:
        """Case D: Partitioned table function without partition_column (M<N, partition by 1 fallback)."""
        input_pandas_df = _generate_partitioned_data()
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        model_name = f"model_{inspect.stack()[0].function}"
        version_name = f"ver_{self._run_id}"

        def check_partition_by_1_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                ["OUTPUT_COL1", "OUTPUT_COL2"],
            )

            # Verify all data was processed as a single partition:
            # predict_expanded outputs _PARTITIONED_EXPANDED_OUTPUT_SIZE rows per call.
            # If data were split into K partitions, we'd get K * _PARTITIONED_EXPANDED_OUTPUT_SIZE rows.
            # Expecting exactly _PARTITIONED_EXPANDED_OUTPUT_SIZE proves single-partition (partition-by-1) behavior.
            expected_total_rows = _PARTITIONED_EXPANDED_OUTPUT_SIZE
            self.assertEqual(len(actual), expected_total_rows)

            # Verify output values match predict_expanded over the entire input as one partition.
            col1_min, col1_max = input_pandas_df["INPUT_COL1"].min(), input_pandas_df["INPUT_COL1"].max()
            col2_min, col2_max = input_pandas_df["INPUT_COL2"].min(), input_pandas_df["INPUT_COL2"].max()
            expected_col1 = np.sort(np.linspace(col1_min, col1_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE))
            expected_col2 = np.sort(np.linspace(col2_min, col2_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE))

            actual_col1 = np.sort(actual["OUTPUT_COL1"].astype(float).values)
            actual_col2 = np.sort(actual["OUTPUT_COL2"].astype(float).values)
            np.testing.assert_allclose(actual_col1, expected_col1, rtol=1e-5)
            np.testing.assert_allclose(actual_col2, expected_col2, rtol=1e-5)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            # No InputSpec / no partition_column
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_expanded",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_SIGNATURES,
            prediction_assert_fn=check_partition_by_1_output,
            skip_row_count_check=True,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            model_name=model_name,
            version_name=version_name,
        )

        self._compare_with_warehouse(
            output_stage_location=output_stage_location,
            model_name=model_name,
            version_name=version_name,
            input_df=input_df,
            function_name="predict_expanded",
            sort_columns=["OUTPUT_COL1"],
        )

    def test_partitioned_table_function_m_greater_than_n(self) -> None:
        """Case E1: Partitioned table function with M>N row mapping."""
        input_pandas_df = _generate_partitioned_data(num_rows_per_partition=20)
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def check_partitioned_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                ["PARTITION_COL", "OUTPUT_COL1", "OUTPUT_COL2"],
            )

            expected_total_rows = _NUM_PARTITIONS * _PARTITIONED_REDUCED_OUTPUT_SIZE
            self.assertEqual(len(actual), expected_total_rows)

            for partition_id in range(1, _NUM_PARTITIONS + 1):
                partition_input = input_pandas_df[input_pandas_df["PARTITION_COL"] == partition_id]
                lr = LinearRegression()
                lr.fit(partition_input[["INPUT_COL1"]], partition_input["INPUT_COL2"])
                col1_min, col1_max = partition_input["INPUT_COL1"].min(), partition_input["INPUT_COL1"].max()
                output_col1_values = np.linspace(col1_min, col1_max, _PARTITIONED_REDUCED_OUTPUT_SIZE)
                expected_preds = np.sort(lr.predict(output_col1_values.reshape(-1, 1)))

                actual_partition = actual[actual["PARTITION_COL"] == partition_id]
                actual_preds = np.sort(actual_partition["OUTPUT_COL2"].astype(float).values)
                np.testing.assert_allclose(actual_preds, expected_preds, rtol=1e-5)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL"),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_reduced",
            ),
            additional_dependencies=["scikit-learn"],
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_SIGNATURES,
            prediction_assert_fn=check_partitioned_output,
            skip_row_count_check=True,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        )

    def test_partitioned_table_function_m_equals_n(self) -> None:
        """Case E2: Partitioned table function with M=N row mapping."""
        input_pandas_df = _generate_partitioned_data()
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def check_partitioned_equal_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                ["PARTITION_COL", "OUTPUT_COL1", "OUTPUT_COL2"],
            )

            expected_total_rows = _NUM_PARTITIONS * _NUM_ROWS_PER_PARTITION
            self.assertEqual(len(actual), expected_total_rows)

            for partition_id in range(1, _NUM_PARTITIONS + 1):
                partition_input = input_pandas_df[input_pandas_df["PARTITION_COL"] == partition_id]
                expected_col1 = np.sort(partition_input["INPUT_COL1"].values * 2)
                expected_col2 = np.sort(partition_input["INPUT_COL2"].values + 1)

                actual_partition = actual[actual["PARTITION_COL"] == partition_id]
                actual_col1 = np.sort(actual_partition["OUTPUT_COL1"].astype(float).values)
                actual_col2 = np.sort(actual_partition["OUTPUT_COL2"].astype(float).values)
                np.testing.assert_allclose(actual_col1, expected_col1, rtol=1e-5)
                np.testing.assert_allclose(actual_col2, expected_col2, rtol=1e-5)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL"),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_equal",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_SIGNATURES,
            prediction_assert_fn=check_partitioned_equal_output,
            skip_row_count_check=True,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        )

    def test_partitioned_table_function_m_less_than_n(self) -> None:
        """Case E3: Partitioned table function with M<N row mapping."""
        input_pandas_df = _generate_partitioned_data()
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        model_name = f"model_{inspect.stack()[0].function}"
        version_name = f"ver_{self._run_id}"

        def check_partitioned_expanded_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                ["PARTITION_COL", "OUTPUT_COL1", "OUTPUT_COL2"],
            )

            expected_total_rows = _NUM_PARTITIONS * _PARTITIONED_EXPANDED_OUTPUT_SIZE
            self.assertEqual(len(actual), expected_total_rows)

            for partition_id in range(1, _NUM_PARTITIONS + 1):
                partition_input = input_pandas_df[input_pandas_df["PARTITION_COL"] == partition_id]
                col1_min, col1_max = partition_input["INPUT_COL1"].min(), partition_input["INPUT_COL1"].max()
                col2_min, col2_max = partition_input["INPUT_COL2"].min(), partition_input["INPUT_COL2"].max()
                expected_col1 = np.sort(np.linspace(col1_min, col1_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE))
                expected_col2 = np.sort(np.linspace(col2_min, col2_max, _PARTITIONED_EXPANDED_OUTPUT_SIZE))

                actual_partition = actual[actual["PARTITION_COL"] == partition_id]
                actual_col1 = np.sort(actual_partition["OUTPUT_COL1"].astype(float).values)
                actual_col2 = np.sort(actual_partition["OUTPUT_COL2"].astype(float).values)
                np.testing.assert_allclose(actual_col1, expected_col1, rtol=1e-5)
                np.testing.assert_allclose(actual_col2, expected_col2, rtol=1e-5)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL"),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_expanded",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_SIGNATURES,
            prediction_assert_fn=check_partitioned_expanded_output,
            skip_row_count_check=True,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            model_name=model_name,
            version_name=version_name,
        )

        self._compare_with_warehouse(
            output_stage_location=output_stage_location,
            model_name=model_name,
            version_name=version_name,
            input_df=input_df,
            function_name="predict_expanded",
            partition_column="PARTITION_COL",
            sort_columns=["PARTITION_COL", "OUTPUT_COL1"],
        )

    def test_non_partitioned_table_function_with_partition_column_with_params(self) -> None:
        """Case C + params: Non-partitioned table function with partition column and override params."""
        input_pandas_df = _generate_partitioned_data(num_rows_per_partition=2)
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)
        override_params = {
            "param_int": _OVERRIDE_PARAM_INT,
            "param_float": _OVERRIDE_PARAM_FLOAT,
            "param_bool": _OVERRIDE_PARAM_BOOL,
        }

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def check_stateful_with_params(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                [
                    "PARTITION_COL",
                    "INPUT_COL1",
                    "INPUT_COL2",
                    "OUTPUT_COL1",
                    "OUTPUT_COL2",
                    "OUTPUT_PARTITION_ID",
                    "OUTPUT_PARAM_INT",
                    "OUTPUT_PARAM_FLOAT",
                    "OUTPUT_PARAM_BOOL",
                ],
            )
            # 1:1 row mapping
            self.assertEqual(len(actual), len(input_pandas_df))
            # Verify override params echoed in output
            np.testing.assert_array_equal(actual["OUTPUT_PARAM_INT"].values, _OVERRIDE_PARAM_INT)
            np.testing.assert_allclose(actual["OUTPUT_PARAM_FLOAT"].values, _OVERRIDE_PARAM_FLOAT)
            np.testing.assert_array_equal(actual["OUTPUT_PARAM_BOOL"].values, _OVERRIDE_PARAM_BOOL)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL", params=override_params),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_stateful_with_params",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_WITH_PARAMS_SIGNATURES,
            prediction_assert_fn=check_stateful_with_params,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        )

    def test_partitioned_table_function_m_less_than_n_with_params(self) -> None:
        """Case E3 + params: Partitioned table function with M<N row mapping and override params."""
        input_pandas_df = _generate_partitioned_data(num_rows_per_partition=2)
        model = PartitionedModel(custom_model.ModelContext())

        input_df = self.session.create_dataframe(input_pandas_df)
        override_params = {
            "param_int": _OVERRIDE_PARAM_INT,
            "param_float": _OVERRIDE_PARAM_FLOAT,
            "param_bool": _OVERRIDE_PARAM_BOOL,
        }

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def check_expanded_with_params(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            self.assertCountEqual(
                actual.columns.tolist(),
                [
                    "PARTITION_COL",
                    "OUTPUT_COL1",
                    "OUTPUT_COL2",
                    "OUTPUT_PARAM_INT",
                    "OUTPUT_PARAM_FLOAT",
                    "OUTPUT_PARAM_BOOL",
                ],
            )
            # M<N: expanded rows per partition
            expected_total_rows = _NUM_PARTITIONS * _PARTITIONED_EXPANDED_OUTPUT_SIZE
            self.assertEqual(len(actual), expected_total_rows)
            # Verify override params echoed in output
            np.testing.assert_array_equal(actual["OUTPUT_PARAM_INT"].values, _OVERRIDE_PARAM_INT)
            np.testing.assert_allclose(actual["OUTPUT_PARAM_FLOAT"].values, _OVERRIDE_PARAM_FLOAT)
            np.testing.assert_array_equal(actual["OUTPUT_PARAM_BOOL"].values, _OVERRIDE_PARAM_BOOL)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(partition_column="PARTITION_COL", params=override_params),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                function_name="predict_expanded_with_params",
            ),
            options={"function_type": "TABLE_FUNCTION"},
            signatures=_PARTITIONED_MODEL_WITH_PARAMS_SIGNATURES,
            prediction_assert_fn=check_expanded_with_params,
            skip_row_count_check=True,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        )

    def test_hf_model_with_partition_column_raises_error(self) -> None:
        """Test that run_batch raises ValueError when partition_column is specified for an HF pipeline model."""
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-gpt2-with-chatml-template",
            max_length=200,
        )

        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    ],
                }
            ]
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name="model_test_hf_partition_column_error",
            version_name=f"ver_{self._run_id}",
            conda_dependencies=conda_dependencies,
            pip_requirements=["transformers", "torch==2.6.0"],
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        input_df = self.session.create_dataframe(x_df)

        with self.assertRaisesRegex(
            ValueError,
            "partition_column is not supported for HuggingFace pipeline models",
        ):
            mv.run_batch(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                input_spec=InputSpec(partition_column="messages"),
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(job_name=job_name),
            )

    def test_scalar_function_with_partition_column_raises_error(self) -> None:
        """Test that run_batch raises ValueError when partition_column is specified for a scalar function model."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        input_pandas_df = pd.DataFrame({"INPUT_COL1": [1.0, 2.0, 3.0], "INPUT_COL2": [2.0, 4.0, 6.0]})
        model.fit(input_pandas_df[["INPUT_COL1"]], input_pandas_df["INPUT_COL2"])

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name="model_test_scalar_partition_column_error",
            version_name=f"ver_{self._run_id}",
            sample_input_data=input_pandas_df[["INPUT_COL1"]],
            conda_dependencies=conda_dependencies,
            # TODO: (SNOW-3201744) remove this after the GS is ready
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True, "function_type": "FUNCTION"},
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        input_df = self.session.create_dataframe(
            pd.DataFrame({"PARTITION_COL": [1, 1, 2, 2], "INPUT_COL1": [1.0, 2.0, 3.0, 4.0]})
        )

        with self.assertRaisesRegex(
            ValueError,
            "partition_column is not supported for FUNCTION type methods",
        ):
            mv.run_batch(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                input_spec=InputSpec(partition_column="PARTITION_COL"),
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(job_name=job_name, function_name="predict"),
            )


if __name__ == "__main__":
    absltest.main()
