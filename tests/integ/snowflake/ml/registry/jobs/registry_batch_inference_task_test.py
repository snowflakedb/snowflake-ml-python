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
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGRun, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model.batch import (
    BatchInferenceTask,
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
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name='"col_a"'),
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name='"col_b"'),
        ],
        outputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output"),
        ],
    ),
}


class TestBatchInferenceTaskInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
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
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {task_fqn} SET {param} = '{value}'").collect()

    def _apply_dag_task_image_overrides(self) -> None:
        if not self._has_image_override():
            return
        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        self._set_task_image_overrides(f"{root_task_fqn}$BATCH_INFERENCE")
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

    def _poll_dag_run_completion(self, dag: "DAG", *, exclude_run_ids: Optional[set[int]] = None) -> "DAGRun":
        """Poll until a DAG run reaches a terminal state, then return it.

        Uses :meth:`DAGOperation.get_complete_dag_runs` (covers the past 60 minutes) and
        :meth:`get_current_dag_runs` for liveness logging. ``exclude_run_ids`` skips runs that
        already existed before the iteration was triggered, so repeated-execution tests can
        wait for the new run only.
        """
        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)
        terminal_states = {"SUCCEEDED", "FAILED", "CANCELLED"}

        for _ in range(self._DAG_POLL_MAX_ATTEMPTS):
            completed = list(dag_op.get_complete_dag_runs(dag, error_only=False))
            if exclude_run_ids is not None:
                completed = [r for r in completed if r.run_id not in exclude_run_ids]
            completed.sort(key=lambda r: r.run_id, reverse=True)
            for run in completed:
                if run.state in terminal_states:
                    return run

            for run in dag_op.get_current_dag_runs(dag):
                logger.info(f"  DAG run {run.run_id}: state={run.state} first_error={run.first_error_message}")

            time.sleep(self._DAG_POLL_INTERVAL_SEC)

        self.fail(
            f"DAG {dag.name} did not complete within {self._DAG_POLL_MAX_ATTEMPTS * self._DAG_POLL_INTERVAL_SEC}s"
        )

    def _deploy_and_run_dag(self, dag: "DAG") -> None:
        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()
        dag_op.run(dag)

    def _assert_dag_succeeded(self, dag: "DAG", base_stage_location: str) -> None:
        """Poll for DAG run success and verify a _SUCCESS output file exists under the stage."""
        run = self._poll_dag_run_completion(dag)
        self.assertEqual(
            run.state,
            "SUCCEEDED",
            f"DAG run {run.state}: task={run.first_error_task_name} error={run.first_error_message}",
        )

        list_results = self.session.sql(f"LIST {base_stage_location}").collect()
        success_files = [row["name"] for row in list_results if row["name"].endswith("_SUCCESS")]
        self.assertGreater(len(success_files), 0, f"No _SUCCESS file found under {base_stage_location}")

    def _run_batch_inference_dag(
        self,
        base_stage_location: str,
        *,
        model_version: Any,
        X: Any,
        compute_pool: str,
        output_spec: OutputSpec,
        job_spec: JobSpec,
        input_spec: Optional[InputSpec] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a basic data_preparation >> batch_inference DAG, deploy, and verify success."""
        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=model_version,
                X=X,
                compute_pool=compute_pool,
                output_spec=output_spec,
                input_spec=input_spec,
                job_spec=job_spec,
                inference_engine_options=inference_engine_options,
            )
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)
        self._assert_dag_succeeded(dag, base_stage_location)

    def test_batch_inference_task_dag(self) -> None:
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/dag_base_stage/"

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(function_name="predict", job_name_prefix="test_dag"),
        )

    def test_batch_inference_task_dag_return_value(self) -> None:
        """Verify that a successor task can read the batch inference task return value."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        result_table = f"{self._test_db}.{self._test_schema}.dag_result_{uuid.uuid4().hex[:8]}"
        self.session.sql(f"CREATE TABLE {result_table} (return_value VARCHAR)").collect()

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
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                X=input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(function_name="predict"),
            )
            verify_task = DAGTask("verify_return_value", definition=verify_sql)
            data_prep_task >> batch_inference_task >> verify_task

        self._deploy_and_run_dag(dag)

        run = self._poll_dag_run_completion(dag)
        self.assertEqual(
            run.state,
            "SUCCEEDED",
            f"DAG run {run.state}: task={run.first_error_task_name} error={run.first_error_message}",
        )

        rows = self.session.sql(f"SELECT return_value FROM {result_table}").collect()
        self.assertLen(rows, 1, f"Expected 1 row in result table, got {len(rows)}")
        result = json.loads(rows[0]["RETURN_VALUE"])
        self.assertEqual(result["output_stage_location"], output_stage_location)

    def test_batch_inference_task_dag_failure(self) -> None:
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
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(function_name="predict"),
            )
            successor_task = DAGTask("successor", definition=successor_sql)
            data_prep_task >> batch_inference_task >> successor_task

        self._deploy_and_run_dag(dag)

        run = self._poll_dag_run_completion(dag)
        self.assertEqual(run.state, "FAILED", f"Expected FAILED but got {run.state}: {run.first_error_message}")
        self.assertIn("BATCH_INFERENCE", (run.first_error_task_name or "").upper())
        self.assertRegex(run.first_error_message or "", r"Job .+ failed to complete.*Exited with status: FAILED")

        success_file = output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file}").collect()
        self.assertEqual(len(list_results), 0, f"_SUCCESS file should not exist at: {success_file}")

        rows = self.session.sql(f"SELECT * FROM {marker_table}").collect()
        self.assertEqual(len(rows), 0, "Successor task should not have run after batch inference failure")

    def test_serverless_task(self) -> None:
        """Verify batch inference works when the DAG task uses serverless compute (no warehouse)."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/serverless_base_stage/"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            user_task_managed_initial_warehouse_size="XSMALL",
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                X=input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(base_stage_location=base_stage_location),
                job_spec=JobSpec(function_name="predict", job_name_prefix="test_serverless"),
            )
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)
        self._assert_dag_succeeded(dag, base_stage_location)

    def test_params(self) -> None:
        """Verify batch inference works with InputSpec params passed through a DAG task."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/params_base_stage/"

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            input_spec=InputSpec(params={"float_param": 0.9}),
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(function_name="predict_with_params", job_name_prefix="test_params"),
        )

    def test_column_handling(self) -> None:
        """Verify batch inference works with InputSpec column_handling in a DAG task."""
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

        self._run_batch_inference_dag(
            base_stage_location,
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
            job_spec=JobSpec(function_name="predict_file", job_name_prefix="test_column_handling"),
        )

    def test_post_actions(self) -> None:
        """Verify batch inference works when the input DataFrame has post_actions."""
        parquet_stage = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/post_actions_parquet/"
        self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"]).write.copy_into_location(
            parquet_stage, file_format_type="parquet", overwrite=True
        )
        input_df = self.session.read.parquet(parquet_stage)
        self.assertGreater(len(input_df.queries["post_actions"]), 0, "Expected post_actions")
        input_df = input_df.select(F.col("$1").cast("BIGINT").alias("C1"), F.col("$2").cast("BIGINT").alias("C2"))

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/post_actions_base_stage/"

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(function_name="predict", job_name_prefix="test_post_actions"),
        )

    def test_repeated_dag_execution(self) -> None:
        """Verify batch inference DAG task succeeds on repeated executions."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/repeated_base_stage/"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                X=input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(base_stage_location=base_stage_location),
                job_spec=JobSpec(function_name="predict", job_name_prefix="test_repeated"),
            )
            data_prep_task >> batch_inference_task

        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()

        num_runs = 3
        for i in range(num_runs):
            logger.info(f"Starting DAG run {i + 1}/{num_runs}")
            existing_run_ids = {r.run_id for r in dag_op.get_complete_dag_runs(dag, error_only=False)}
            dag_op.run(dag)
            run = self._poll_dag_run_completion(dag, exclude_run_ids=existing_run_ids)
            self.assertEqual(run.state, "SUCCEEDED", f"Run {i + 1} failed: {run.first_error_message}")

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

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(function_name="predict", job_name_prefix="test_sql_query"),
        )

    def test_complex_query(self) -> None:
        """Verify batch inference works with a complex multi-query DataFrame (JOIN + select)."""
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

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(function_name="predict", job_name_prefix="test_complex_query"),
        )

    def test_quoted_identifiers(self) -> None:
        """Verify batch inference works with quoted (lowercase) model name and column names."""
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

        input_table = f"{self._test_db}.{self._test_schema}.quoted_input_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe([[0, 0], [1, 1]], schema=['"col_a"', '"col_b"']).write.save_as_table(
            input_table, mode="overwrite"
        )
        input_df = self.session.table(input_table)

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/quoted_id_base_stage/"

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(function_name="predict_quoted", job_name_prefix="test_quoted"),
        )

    def test_user_privileges(self) -> None:
        """Verify batch inference works when the root DAG task uses EXECUTE AS USER."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/user_privileges_base_stage/"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                X=input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(base_stage_location=base_stage_location),
                job_spec=JobSpec(function_name="predict", job_name_prefix="test_user_privs"),
            )
            data_prep_task >> batch_inference_task

        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")

        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        current_user = self.session.sql("SELECT CURRENT_USER()").collect()[0][0]

        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        self.session.sql(f"ALTER TASK {root_task_fqn} SET EXECUTE AS USER {current_user}").collect()
        if self._has_image_override():
            self._set_task_image_overrides(f"{root_task_fqn}$BATCH_INFERENCE")
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

        dag_op.run(dag)
        self._assert_dag_succeeded(dag, base_stage_location)

    def test_custom_image_repo(self) -> None:
        """BatchInferenceTask runs successfully when image_repo is explicitly set on JobSpec."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/custom_repo_base_stage/"

        self._run_batch_inference_dag(
            base_stage_location,
            model_version=self._mv,
            X=input_df,
            compute_pool=self._TEST_CPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                function_name="predict",
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
                job_name_prefix="test_custom_repo",
            ),
        )

    def test_passes_dagtask_kwargs(self) -> None:
        """Verify DAGTask kwargs (e.g. comment) flow through to the deployed Snowflake task."""
        input_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"])
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/dagtask_kwargs_base_stage/"
        expected_comment = "snowml-batch-inference-task-integ"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                X=input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(base_stage_location=base_stage_location),
                job_spec=JobSpec(function_name="predict", job_name_prefix="test_kwargs"),
                comment=expected_comment,
            )
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)

        # Verify the DAGTask kwarg landed on the deployed Snowflake task
        child_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}$BATCH_INFERENCE"
        rows = self.session.sql(
            f"SHOW TASKS LIKE '{self._dag_name}$BATCH_INFERENCE' IN SCHEMA {self._test_db}.{self._test_schema}"
        ).collect()
        self.assertGreater(len(rows), 0, f"Task {child_task_fqn} not found")
        self.assertEqual(rows[0]["comment"], expected_comment)

        self._assert_dag_succeeded(dag, base_stage_location)


if __name__ == "__main__":
    absltest.main()
