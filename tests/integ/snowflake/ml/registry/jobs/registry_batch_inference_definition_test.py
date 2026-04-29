import json
import logging
import os
import tempfile
import time
import uuid
from datetime import timedelta
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest

try:
    from snowflake.core import Root
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model.batch import (
    BatchInferenceDefinition,
    FileEncoding,
    InputFormat,
    InputSpec,
    JobSpec,
    OutputSpec,
)
from snowflake.snowpark import functions as F
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

logger = logging.getLogger(__name__)


class TestModel(custom_model.CustomModel):
    """A model with multiple inference methods for testing different BatchInferenceDefinition features."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})

    @custom_model.inference_api
    def predict_with_params(self, input: pd.DataFrame, *, float_param: float = 0.5) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"], "received_float_param": [float_param] * len(input)})

    @custom_model.inference_api
    def predict_file(self, input: pd.DataFrame) -> pd.DataFrame:
        import base64

        decoded = [base64.b64decode(v).decode("utf-8") for v in input["FILE_CONTENT"]]
        return pd.DataFrame({"output": decoded})

    @custom_model.inference_api
    def predict_quoted(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["col_a"]})


class FailureModel(custom_model.CustomModel):
    """A model that deliberately raises during inference to test failure handling."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("Intentional failure for testing")


_TEST_MODEL_SIGNATURES = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C1"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C2"),
        ],
        outputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output"),
        ],
    ),
    "predict_with_params": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C1"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C2"),
        ],
        outputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="received_float_param"),
        ],
        params=[
            model_signature.ParamSpec(name="float_param", dtype=model_signature.DataType.DOUBLE, default_value=0.5),
        ],
    ),
    "predict_file": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="FILE_CONTENT"),
        ],
        outputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="output"),
        ],
    ),
    "predict_quoted": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="col_a"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="col_b"),
        ],
        outputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output"),
        ],
    ),
}


class TestBatchInferenceDefinitionInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    _DAG_POLL_INTERVAL_SEC = 15
    _DAG_POLL_MAX_ATTEMPTS = 120  # 30 min total

    def setUp(self) -> None:
        if not _HAS_SNOWFLAKE_CORE:
            self.skipTest("snowflake.core is not installed")
        super().setUp()
        self._dag_name = f"test_dag_{uuid.uuid4().hex[:8]}"
        self._model = TestModel(custom_model.ModelContext())
        self._mv = self._log_model(self._model, signatures=_TEST_MODEL_SIGNATURES)

    def _log_model(
        self,
        model: Any,
        sample_df: Any = None,
        *,
        signatures: Any = None,
    ) -> Any:
        """Log a model to registry and return the ModelVersion."""
        name = f"model_{uuid.uuid4().hex[:8]}"
        version = f"ver_{self._run_id}"
        from tests.integ.snowflake.ml.test_utils import test_env_utils

        conda_deps = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        return self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_df,
            signatures=signatures,
            conda_dependencies=conda_deps,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

    def _set_task_image_overrides(self, task_fqn: str) -> None:
        """Set image override session parameters on a DAG task via ALTER TASK."""
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {task_fqn} SET {param} = '{value}'").collect()

    def _apply_dag_task_image_overrides(self) -> None:
        """Suspend root task, set image overrides on batch inference child, then resume.

        DAG tasks run in a system-managed context, not the user's session, so session-level
        image override parameters don't apply. This sets them at the task level instead.
        The root task must be suspended before altering any child task in the DAG.
        """
        if not self._has_image_override():
            return
        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        self._set_task_image_overrides(f"{root_task_fqn}$BATCH_INFERENCE")
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

    def _poll_dag_completion(
        self, dag_name: str, task_name: str, *, scheduled_after: Optional[str] = None
    ) -> tuple[str, str]:
        """Poll TASK_HISTORY until the DAG task completes or times out.

        Args:
            dag_name: Name of the DAG to poll.
            task_name: Uppercase name of the child task to wait on.
            scheduled_after: Optional ISO timestamp; only consider task runs scheduled after this time.

        Returns:
            A tuple of (state, error_message) for the target task.
        """
        scheduled_filter = f"AND SCHEDULED_TIME > '{scheduled_after}'" if scheduled_after else ""
        for _ in range(self._DAG_POLL_MAX_ATTEMPTS):
            result = self.session.sql(
                f"""
                SELECT NAME, STATE, ERROR_MESSAGE
                FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
                    SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP())
                ))
                WHERE NAME ILIKE '%{dag_name}%'
                {scheduled_filter}
                ORDER BY SCHEDULED_TIME DESC
                LIMIT 10
            """
            ).collect()

            for row in result:
                logger.info(f"  Task {row['NAME']}: {row['STATE']} error={row['ERROR_MESSAGE']}")

            # Check if the target child task has completed
            for row in result:
                if task_name in row["NAME"].upper() and row["STATE"] in (
                    "SUCCEEDED",
                    "FAILED",
                    "CANCELLED",
                ):
                    return row["STATE"], row["ERROR_MESSAGE"]

            time.sleep(self._DAG_POLL_INTERVAL_SEC)

        self.fail(
            f"DAG {dag_name} did not complete within {self._DAG_POLL_MAX_ATTEMPTS * self._DAG_POLL_INTERVAL_SEC}s"
        )

    def _deploy_and_run_dag(self, dag: DAG) -> None:
        """Deploy a DAG, apply image overrides on the batch inference child task, and trigger a run."""
        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()
        dag_op.run(dag)

    def _assert_batch_inference_succeeded(self, base_stage_location: str) -> None:
        """Poll for BATCH_INFERENCE task success and verify _SUCCESS output file exists."""
        state, error_msg = self._poll_dag_completion(self._dag_name, task_name="BATCH_INFERENCE")
        self.assertEqual(state, "SUCCEEDED", f"Batch inference DAG task {state}: {error_msg}")

        list_results = self.session.sql(f"LIST {base_stage_location}").collect()
        success_files = [row["name"] for row in list_results if row["name"].endswith("_SUCCESS")]
        self.assertGreater(len(success_files), 0, f"No _SUCCESS file found under {base_stage_location}")

    def _run_batch_inference_dag(self, defn: BatchInferenceDefinition, base_stage_location: str) -> None:
        """Create a basic data_preparation >> batch_inference DAG, deploy, and verify success.

        Only verifies the task succeeds and produces output; correctness is tested in other tests.
        """
        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)
        self._assert_batch_inference_succeeded(base_stage_location)

    def test_batch_inference_definition_dag(self) -> None:
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/dag_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_dag",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)

    def test_batch_inference_definition_dag_return_value(self) -> None:
        """Verify that a successor task can read the batch inference task return value."""
        if not self._has_image_override():
            self.skipTest("Requires image override configuration.")

        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
            ),
        )

        # 4. Create result table to capture the return value from the successor task
        result_table = f"{self._test_db}.{self._test_schema}.dag_result_{uuid.uuid4().hex[:8]}"
        self.session.sql(f"CREATE TABLE {result_table} (return_value VARCHAR)").collect()

        # 5. Create and deploy DAG: data_preparation >> batch_inference >> verify_return_value
        data_prep_sql = "SELECT 'data_preparation done'"
        verify_sql = f"INSERT INTO {result_table} (return_value)" f" SELECT SYSTEM$GET_PREDECESSOR_RETURN_VALUE()"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition=data_prep_sql)
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            verify_task = DAGTask("verify_return_value", definition=verify_sql)
            data_prep_task >> batch_inference_task >> verify_task

        self._deploy_and_run_dag(dag)

        # 6. Poll for verify_return_value task completion
        state, error_msg = self._poll_dag_completion(self._dag_name, task_name="VERIFY_RETURN_VALUE")
        self.assertEqual(state, "SUCCEEDED", f"Verify return value task {state}: {error_msg}")

        # 7. Assert the return value contains the expected output_stage_location
        rows = self.session.sql(f"SELECT return_value FROM {result_table}").collect()
        self.assertLen(rows, 1, f"Expected 1 row in result table, got {len(rows)}")
        result = json.loads(rows[0]["RETURN_VALUE"])
        self.assertEqual(result["output_stage_location"], output_stage_location)

    def test_batch_inference_definition_dag_failure(self) -> None:
        """Verify that a DAG task with a failing model reaches FAILED state."""
        model = FailureModel(custom_model.ModelContext())
        signature = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C2"),
            ],
            outputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="output"),
            ],
        )
        mv = self._log_model(model, signatures={"predict": signature})

        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
            ),
        )

        # 4. Create marker table and deploy DAG: data_preparation >> batch_inference >> successor
        # The successor task should NOT run because batch_inference fails.
        marker_table = f"{self._test_db}.{self._test_schema}.dag_marker_{uuid.uuid4().hex[:8]}"
        self.session.sql(f"CREATE TABLE {marker_table} (marker VARCHAR)").collect()

        data_prep_sql = "SELECT 'data_preparation done'"
        successor_sql = f"INSERT INTO {marker_table} (marker) VALUES ('successor_ran')"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition=data_prep_sql)
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            successor_task = DAGTask("successor", definition=successor_sql)
            data_prep_task >> batch_inference_task >> successor_task

        self._deploy_and_run_dag(dag)

        # 5. Poll for completion and assert failure
        state, error_msg = self._poll_dag_completion(self._dag_name, task_name="BATCH_INFERENCE")
        self.assertEqual(state, "FAILED", f"Expected FAILED but got {state}: {error_msg}")
        self.assertRegex(error_msg, r"Job .+ failed to complete.*Exited with status: FAILED")

        # 6. Verify no _SUCCESS file was created
        success_file = output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file}").collect()
        self.assertEqual(len(list_results), 0, f"_SUCCESS file should not exist at: {success_file}")

        # 7. Verify successor task was never triggered
        rows = self.session.sql(f"SELECT * FROM {marker_table}").collect()
        self.assertEqual(len(rows), 0, "Successor task should not have run after batch inference failure")

    def test_serverless_task(self) -> None:
        """Verify batch inference works when the DAG task uses serverless compute (no warehouse)."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/serverless_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_serverless",
            ),
        )

        # Serverless DAG uses managed compute instead of a user warehouse
        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            user_task_managed_initial_warehouse_size="XSMALL",
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)
        self._assert_batch_inference_succeeded(base_stage_location)

    def test_params(self) -> None:
        """Verify batch inference works with InputSpec params passed through a DAG task."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/params_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            input_spec=InputSpec(params={"float_param": 0.9}),
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict_with_params",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_params",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)

    @absltest.skip("TODO: fix column_handling")
    def test_column_handling(self) -> None:
        """Verify batch inference works with InputSpec column_handling in a DAG task."""
        # Upload a small text file to stage
        input_files_stage = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/column_handling_input_files/"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("hello from column handling test")
            tmp_path = tmp.name
        try:
            self.session.sql(
                f"PUT 'file://{tmp_path}' {input_files_stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()
        finally:
            os.unlink(tmp_path)

        stage_file_path = f"{input_files_stage}{os.path.basename(tmp_path)}"
        input_df = self.session.create_dataframe([[stage_file_path]], schema=["FILE_CONTENT"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/column_handling_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            input_spec=InputSpec(
                column_handling={
                    "FILE_CONTENT": {
                        "input_format": InputFormat.FULL_STAGE_PATH,
                        "convert_to": FileEncoding.BASE64,
                    }
                }
            ),
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict_file",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_column_handling",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)

    def test_post_actions(self) -> None:
        """Verify batch inference works when the input DataFrame has post_actions."""
        # Write data to stage as parquet, then read back via session.read.parquet().
        # Reading parquet creates temp file format objects whose cleanup generates post_actions.
        parquet_stage = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/post_actions_parquet/"
        self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"]).write.copy_into_location(
            parquet_stage, file_format_type="parquet", overwrite=True
        )
        input_df = self.session.read.parquet(parquet_stage)
        self.assertGreater(len(input_df.queries["post_actions"]), 0, "Expected post_actions")
        # Parquet columns are unnamed ($1, $2) and typed VARIANT; cast to match model signature.
        input_df = input_df.select(F.col("$1").cast("BIGINT").alias("C1"), F.col("$2").cast("BIGINT").alias("C2"))

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/post_actions_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_post_actions",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)

    def test_repeated_dag_execution(self) -> None:
        """Verify batch inference DAG task succeeds on repeated executions."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/repeated_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_repeated",
            ),
        )

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            data_prep_task >> batch_inference_task

        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()

        num_runs = 3
        for i in range(num_runs):
            logger.info(f"Starting DAG run {i + 1}/{num_runs}")
            run_triggered_at = self.session.sql("SELECT CURRENT_TIMESTAMP()::VARCHAR").collect()[0][0]
            dag_op.run(dag)
            state, error_msg = self._poll_dag_completion(
                self._dag_name, task_name="BATCH_INFERENCE", scheduled_after=run_triggered_at
            )
            self.assertEqual(state, "SUCCEEDED", f"Run {i + 1} failed: {error_msg}")

        # Verify that each run produced a _SUCCESS file under the base stage
        list_results = self.session.sql(f"LIST {base_stage_location}").collect()
        success_files = [row["name"] for row in list_results if row["name"].endswith("_SUCCESS")]
        self.assertEqual(
            len(success_files),
            num_runs,
            f"Expected {num_runs} _SUCCESS files, got {len(success_files)}: {success_files}",
        )

    def test_sql_query(self) -> None:
        """Verify batch inference works with a DataFrame created from session.sql()."""
        input_table = f"{self._test_db}.{self._test_schema}.sql_query_input_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"]).write.save_as_table(
            input_table, mode="overwrite"
        )
        input_df = self.session.sql(f"SELECT * FROM {input_table} WHERE C1 >= 0")

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/sql_query_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_sql_query",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)

    def test_complex_query(self) -> None:
        """Verify batch inference works with a complex multi-query DataFrame (JOIN + select)."""
        # Build a DataFrame from two persisted tables joined together.
        # This produces a multi-query DataFrame with multiple entries in queries["queries"].
        table_a = f"{self._test_db}.{self._test_schema}.complex_a_{uuid.uuid4().hex[:8]}"
        table_b = f"{self._test_db}.{self._test_schema}.complex_b_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe([[0, 10], [1, 20]], schema=["KEY", "C1"]).write.save_as_table(
            table_a, mode="overwrite"
        )
        self.session.create_dataframe([[0, 100], [1, 200]], schema=["KEY", "C2"]).write.save_as_table(
            table_b, mode="overwrite"
        )

        df_a = self.session.table(table_a)
        df_b = self.session.table(table_b)
        input_df = df_a.join(df_b, on="KEY").select("C1", "C2")

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/complex_query_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_complex_query",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)

    def test_quoted_identifiers(self) -> None:
        """Verify batch inference works with quoted (lowercase) model name and column names."""
        # Register a model with a quoted lowercase name; uses predict_quoted which reads "col_a".
        quoted_model_name = f'"batch_quoted_{uuid.uuid4().hex[:8]}"'
        model = TestModel(custom_model.ModelContext())
        version = f"ver_{self._run_id}"
        from tests.integ.snowflake.ml.test_utils import test_env_utils

        conda_deps = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        mv = self.registry.log_model(
            model=model,
            model_name=quoted_model_name,
            version_name=version,
            signatures=_TEST_MODEL_SIGNATURES,
            conda_dependencies=conda_deps,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

        # Create input with quoted lowercase column names, persisted to a real table
        input_table = f"{self._test_db}.{self._test_schema}.quoted_input_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe([[0, 0], [1, 1]], schema=['"col_a"', '"col_b"']).write.save_as_table(
            input_table, mode="overwrite"
        )
        input_df = self.session.table(input_table)

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/quoted_id_base_stage/"

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=self._TEST_SPCS_WH,
                function_name="predict_quoted",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_quoted",
            ),
        )
        self._run_batch_inference_dag(defn, base_stage_location)


if __name__ == "__main__":
    absltest.main()
