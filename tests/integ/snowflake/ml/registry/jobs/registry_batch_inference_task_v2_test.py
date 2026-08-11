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
from snowflake.ml.model.batch_inference import (
    BatchInferenceTask,
    FileEncoding,
    ImageBuildSpec,
    InferenceSpec,
    InputFormat,
    InputSpec,
    OutputSpec,
    ResourcesSpec,
)
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


class TestBatchInferenceTaskV2Integ(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Integration tests for the V2 task, which runs EXECUTE INFERENCE JOB SERVICE."""

    _DAG_POLL_INTERVAL_SEC = 15
    _DAG_POLL_MAX_ATTEMPTS = 120  # 30 min total

    # A batch deploy creates several jobs in the schema: the inference job plus server-side
    # ``MODEL_BUILD_<hash>`` / ``MODEL_LOGGING_<hash>`` sub-services.
    _SUBSERVICE_PREFIXES = ("MODEL_BUILD_", "MODEL_LOGGING_")

    def setUp(self) -> None:
        if not _HAS_SNOWFLAKE_CORE:
            self.skipTest("snowflake.core is not installed")
        super().setUp()
        self._dag_name = f"test_dag_{uuid.uuid4().hex[:8]}"
        self._jobs_before_run: Optional[set[str]] = None
        self._model = TestModel(custom_model.ModelContext())
        self._mv = self._log_model(self._model, signatures=_TEST_MODEL_SIGNATURES)

    def _log_model(self, model: Any, sample_df: Any = None, *, signatures: Any = None) -> Any:
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

    def _apply_dag_task_image_overrides(self) -> None:
        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {root_task_fqn}$BATCH_INFERENCE SET {param} = '{value}'").collect()
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

    def _dag_operation(self) -> "DAGOperation":
        api_root = Root(self.session)
        return DAGOperation(api_root.databases[self._test_db].schemas[self._test_schema])

    def _poll_dag_run_completion(self, dag: "DAG", *, exclude_run_ids: Optional[set[int]] = None) -> "DAGRun":
        """Poll until a DAG run reaches a terminal state, then return it.

        Args:
            dag: The task graph to poll.
            exclude_run_ids: Run ids that existed before the run was triggered, so a repeated
                execution waits for the new run only.

        Returns:
            The terminal DAG run.
        """
        dag_op = self._dag_operation()
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
        dag_op = self._dag_operation()
        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()
        self._jobs_before_run = self._snapshot_jobs()
        dag_op.run(dag)

    def _list_jobs(self) -> Optional[list[str]]:
        """Names of the jobs in the test schema, newest first.

        None and an empty list mean different things: a failed lookup must not be read as
        "no jobs exist", or unrelated jobs get attributed to this run.

        Returns:
            The job names, or None when they cannot be listed.
        """
        try:
            rows = self.session.sql(f"SHOW JOB SERVICES IN SCHEMA {self._test_db}.{self._test_schema}").collect()
            return [str(row["name"]) for row in sorted(rows, key=lambda row: row["created_on"], reverse=True)]
        except Exception:
            logger.warning("Could not list the jobs in %s.%s", self._test_db, self._test_schema, exc_info=True)
            return None

    def _snapshot_jobs(self) -> Optional[set[str]]:
        """Upper-cased names of the jobs that exist right now.

        Returns:
            The job names upper-cased, or None when they cannot be listed.
        """
        job_names = self._list_jobs()
        return None if job_names is None else {job_name.upper() for job_name in job_names}

    def _new_jobs(self) -> Optional[list[str]]:
        """Names of the jobs created by the most recent DAG run, newest first.

        Anchored to the pre-run snapshot rather than to timestamps, so a repeated execution
        that fails before its inference job launches never reports a job from an earlier run.

        Returns:
            The job names, or None when they cannot be attributed to the run.
        """
        job_names = self._list_jobs()
        if job_names is None or self._jobs_before_run is None:
            return None
        new_job_names = [job_name for job_name in job_names if job_name.upper() not in self._jobs_before_run]
        inference_jobs = [
            job_name for job_name in new_job_names if not job_name.upper().startswith(self._SUBSERVICE_PREFIXES)
        ]
        return inference_jobs[:1] if inference_jobs else new_job_names

    def _dump_run_logs(self, *, limit: int = 100) -> str:
        """Best-effort logs for the jobs created by the most recent DAG run.

        The task launches the inference job inside Snowflake, so no MLJob handle is available;
        the run's jobs have to be discovered from the schema.

        Args:
            limit: Number of trailing log lines per job.

        Returns:
            The formatted logs.
        """
        job_names = self._new_jobs()
        if job_names is None:
            return "(unable to list the jobs in the test schema)"
        if not job_names:
            return "(this DAG run created no jobs)"

        parts = []
        for job_name in job_names:
            job_fqn = f"{self._test_db}.{self._test_schema}.{job_name}"
            parts.append(f"Job: {job_fqn}")
            try:
                # Import the submodule, not the package: ``snowflake.ml.jobs`` resolves as a
                # namespace package under Bazel, so package-level names are not importable here.
                from snowflake.ml.jobs import job as ml_job

                batch_job = ml_job.MLJob[Any](job_fqn, session=self.session)
                parts.append(f"Last {limit} lines of job logs:\n{batch_job.get_logs(limit=limit)}")
            except Exception as e:
                parts.append(f"(failed to fetch job logs: {e})")
        return "\n\n".join(parts)

    def _assert_run_succeeded(self, run: "DAGRun") -> None:
        """Assert the DAG run SUCCEEDED, dumping batch-inference service logs on failure.

        Args:
            run: The terminal DAG run.
        """
        if run.state != "SUCCEEDED":
            logs = self._dump_run_logs()
            self.fail(
                f"DAG run {run.state}: task={run.first_error_task_name} error={run.first_error_message}\n\n{logs}"
            )

    def _input_query(self) -> str:
        return "SELECT 0::INT AS C1, 0::INT AS C2 UNION ALL SELECT 1::INT AS C1, 1::INT AS C2"

    def _stage_input(self) -> str:
        """Write the input rows to a stage and return the location.

        Returns:
            The input stage location.
        """
        _, _, input_stage_location = self._prepare_job_name_and_stage_for_batch_inference()
        self.session.create_dataframe([[0, 0], [1, 1]], schema=["C1", "C2"]).write.copy_into_location(
            location=input_stage_location, file_format_type="parquet", header=True, overwrite=True
        )
        return input_stage_location

    def _new_dag(self) -> "DAG":
        """Build an empty graph with this test's name, warehouse and stage.

        Returns:
            The graph.
        """
        return DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=self._TEST_SPCS_WH,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

    def _build_dag(self, **task_kwargs: Any) -> tuple["DAG", str]:
        """Build a data_preparation >> batch_inference graph.

        Args:
            task_kwargs: Extra arguments for the batch inference task. ``function_name``
                defaults to ``predict``.

        Returns:
            The graph and the output base location.
        """
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        task_kwargs.setdefault("function_name", "predict")

        dag = self._new_dag()
        with dag:
            prep = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            score = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                **task_kwargs,
            )
            prep >> score

        return dag, output_stage_location

    def _assert_success_file_written(self, output_stage_location: str) -> list[str]:
        """Assert a _SUCCESS marker exists under the output base, and return the per-job subdirs.

        Args:
            output_stage_location: The output base location.

        Returns:
            The distinct per-job subdirectory names found under the base.
        """
        rows = self.session.sql(f"LIST {output_stage_location}").collect()
        names = [str(row["name"]) for row in rows]
        success_files = [name for name in names if name.endswith("_SUCCESS")]
        self.assertGreater(len(success_files), 0, f"No _SUCCESS file found under {output_stage_location}. Saw: {names}")

        # Results land in <stage_location>/<job_name>/, so the segment before _SUCCESS is the job name.
        return sorted({name.rsplit("/", 2)[-2] for name in success_files})

    def test_query_input(self) -> None:
        dag, output_stage_location = self._build_dag(query=self._input_query())

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_stage_input(self) -> None:
        dag, output_stage_location = self._build_dag(input_stage_location=self._stage_input())

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_input_spec_and_resources_spec(self) -> None:
        """The resources and inference blocks are accepted alongside the input block."""
        dag, output_stage_location = self._build_dag(
            query=self._input_query(),
            input_spec=InputSpec(),
            resources_spec=ResourcesSpec(cpu_requests="1", memory_requests="4Gi"),
            inference_spec=InferenceSpec(num_workers=1, max_batch_rows=1024),
        )

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_successor_reads_output_location(self) -> None:
        """A successor loads the results using the location from the task return value.

        Asserting the payload merely contains the key would not prove the path is usable, so
        the successor copies from it and the row count is checked.
        """
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        result_table = f"{self._test_db}.{self._test_schema}.dag_result_{uuid.uuid4().hex[:8]}"
        scores_table = f"{self._test_db}.{self._test_schema}.dag_scores_{uuid.uuid4().hex[:8]}"
        self.session.sql(f"CREATE TABLE {result_table} (return_value VARCHAR)").collect()
        self.session.sql(f"CREATE TABLE {scores_table} (C1 INT, C2 INT, output INT)").collect()

        # PATTERN skips the _SUCCESS marker the job writes next to the Parquet files.
        load_sql = f"""
DECLARE
  payload STRING;
  output_location STRING;
BEGIN
  payload := SYSTEM$GET_PREDECESSOR_RETURN_VALUE();
  output_location := PARSE_JSON(:payload):output_stage_location::STRING;
  INSERT INTO {result_table} (return_value) VALUES (:payload);
  EXECUTE IMMEDIATE 'COPY INTO {scores_table} FROM ' || :output_location
    || ' FILE_FORMAT = (TYPE = PARQUET) MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE'
    || ' PATTERN = ''.*[.]parquet''';
  RETURN output_location;
END;
"""

        dag = self._new_dag()
        with dag:
            prep = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            score = BatchInferenceTask(
                "batch_inference",
                model_version=self._mv,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                query=self._input_query(),
                output_spec=OutputSpec(stage_location=output_stage_location),
                function_name="predict",
            )
            load = DAGTask("load_scores", definition=load_sql)
            prep >> score >> load

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))

        rows = self.session.sql(f"SELECT return_value FROM {result_table}").collect()
        self.assertLen(rows, 1, f"Expected 1 row in {result_table}, got {len(rows)}")
        payload = json.loads(rows[0]["RETURN_VALUE"])
        self.assertIn("output_stage_location", payload)

        # _input_query returns two rows, so the successor must have loaded two.
        score_rows = self.session.sql(f"SELECT C1, C2, output FROM {scores_table} ORDER BY C1").collect()
        self.assertLen(score_rows, 2, f"Expected 2 rows loaded into {scores_table}, got {len(score_rows)}")
        self.assertEqual([row["OUTPUT"] for row in score_rows], [0, 1])

    def test_repeated_runs_get_distinct_jobs(self) -> None:
        """No NAME clause is emitted, so each firing gets a fresh server-generated job name.

        Results land in ``<stage_location>/<job_name>/``, so a per-run job name means each
        firing writes to an empty subdirectory and the default ``ERROR`` save mode never trips.
        If that stopped holding, every recurring task would fail from its second firing on.
        """
        dag, output_stage_location = self._build_dag(query=self._input_query())

        dag_op = self._dag_operation()
        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()

        num_runs = 3
        for i in range(num_runs):
            logger.info(f"Starting DAG run {i + 1}/{num_runs}")
            existing_run_ids = {r.run_id for r in dag_op.get_complete_dag_runs(dag, error_only=False)}
            self._jobs_before_run = self._snapshot_jobs()
            dag_op.run(dag)
            run = self._poll_dag_run_completion(dag, exclude_run_ids=existing_run_ids)
            self._assert_run_succeeded(run)

        job_subdirs = self._assert_success_file_written(output_stage_location)
        self.assertLen(
            job_subdirs,
            num_runs,
            f"Expected {num_runs} per-job output subdirectories under {output_stage_location}: {job_subdirs}",
        )

    def test_dag_failure(self) -> None:
        """A failing model puts the DAG run in FAILED and stops the successor."""
        signature = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C2"),
            ],
            outputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="output"),
            ],
        )
        mv = self._log_model(FailureModel(custom_model.ModelContext()), signatures={"predict": signature})

        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        marker_table = f"{self._test_db}.{self._test_schema}.dag_marker_{uuid.uuid4().hex[:8]}"
        self.session.sql(f"CREATE TABLE {marker_table} (marker VARCHAR)").collect()

        dag = self._new_dag()
        with dag:
            prep = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            score = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                query=self._input_query(),
                output_spec=OutputSpec(stage_location=output_stage_location),
                function_name="predict",
            )
            successor = DAGTask("successor", definition=f"INSERT INTO {marker_table} (marker) VALUES ('successor_ran')")
            prep >> score >> successor

        self._deploy_and_run_dag(dag)

        run = self._poll_dag_run_completion(dag)
        self.assertEqual(run.state, "FAILED", f"Expected FAILED but got {run.state}: {run.first_error_message}")
        self.assertIn("BATCH_INFERENCE", (run.first_error_task_name or "").upper())
        self.assertRegex(run.first_error_message or "", r"Job .+ failed to complete.*Exited with status: FAILED")

        rows = self.session.sql(f"LIST {output_stage_location}").collect()
        success_files = [str(row["name"]) for row in rows if str(row["name"]).endswith("_SUCCESS")]
        self.assertEmpty(success_files, f"_SUCCESS should not exist under {output_stage_location}: {success_files}")

        marker_rows = self.session.sql(f"SELECT * FROM {marker_table}").collect()
        self.assertEmpty(marker_rows, "Successor task should not have run after batch inference failure")

    def test_params(self) -> None:
        """``InputSpec.params`` reaches the model function."""
        dag, output_stage_location = self._build_dag(
            query=self._input_query(),
            input_spec=InputSpec(params={"float_param": 0.9}),
            function_name="predict_with_params",
        )

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_column_handling(self) -> None:
        """``InputSpec.column_handling`` converts a staged file path to base64 for the model."""
        input_files_stage = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/v2_column_handling_input_files/"
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
        dag, output_stage_location = self._build_dag(
            query=f"SELECT '{stage_file_path}' AS FILE_CONTENT",
            input_spec=InputSpec(
                column_handling={
                    "FILE_CONTENT": {
                        "input_format": InputFormat.FULL_STAGE_PATH,
                        "convert_to": FileEncoding.BASE64,
                    }
                }
            ),
            function_name="predict_file",
        )

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_complex_query(self) -> None:
        """A join renders correctly through the ``FROM ( <subquery> )`` clause."""
        table_a = f"{self._test_db}.{self._test_schema}.v2_complex_a_{uuid.uuid4().hex[:8]}"
        table_b = f"{self._test_db}.{self._test_schema}.v2_complex_b_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe([[0, 10], [1, 20]], schema=["KEY", "C1"]).write.save_as_table(
            table_a, mode="overwrite"
        )
        self.session.create_dataframe([[0, 100], [1, 200]], schema=["KEY", "C2"]).write.save_as_table(
            table_b, mode="overwrite"
        )

        query = (
            f"WITH joined AS (SELECT a.C1, b.C2 FROM {table_a} a JOIN {table_b} b ON a.KEY = b.KEY) "
            "SELECT C1, C2 FROM joined WHERE C1 >= 0"
        )
        dag, output_stage_location = self._build_dag(query=query)

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    @absltest.skip("TODO(SNOW-3516871): handle quoted identifiers in batch inference")
    def test_quoted_identifiers(self) -> None:
        """Batch inference works with a quoted (lowercase) model name and column names."""
        quoted_model_name = f'"batch_v2_quoted_{uuid.uuid4().hex[:8]}"'
        from tests.integ.snowflake.ml.test_utils import test_env_utils

        conda_deps = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        mv = self.registry.log_model(
            model=TestModel(custom_model.ModelContext()),
            model_name=quoted_model_name,
            version_name=f"ver_{self._run_id}",
            signatures=_TEST_MODEL_SIGNATURES,
            conda_dependencies=conda_deps,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

        input_table = f"{self._test_db}.{self._test_schema}.v2_quoted_input_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe([[0, 0], [1, 1]], schema=['"col_a"', '"col_b"']).write.save_as_table(
            input_table, mode="overwrite"
        )

        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        dag = self._new_dag()
        with dag:
            prep = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            score = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                query=f'SELECT "col_a", "col_b" FROM {input_table}',
                output_spec=OutputSpec(stage_location=output_stage_location),
                function_name="predict_quoted",
            )
            prep >> score

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_user_privileges(self) -> None:
        """Batch inference works when the root DAG task uses EXECUTE AS USER."""
        dag, output_stage_location = self._build_dag(query=self._input_query())

        dag_op = self._dag_operation()
        dag_op.deploy(dag, mode="orReplace")

        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        current_user = self.session.sql("SELECT CURRENT_USER()").collect()[0][0]

        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        self.session.sql(f"ALTER TASK {root_task_fqn} SET EXECUTE AS USER {current_user}").collect()
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {root_task_fqn}$BATCH_INFERENCE SET {param} = '{value}'").collect()
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

        self._jobs_before_run = self._snapshot_jobs()
        dag_op.run(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_custom_image_repo(self) -> None:
        """``ImageBuildSpec.image_repo`` is honored."""
        dag, output_stage_location = self._build_dag(
            query=self._input_query(),
            image_build_spec=ImageBuildSpec(
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo])
            ),
        )

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    def test_passes_dagtask_kwargs(self) -> None:
        """DAGTask kwargs flow through to the deployed Snowflake task."""
        expected_comment = "snowml-batch-inference-task-v2-integ"
        dag, output_stage_location = self._build_dag(query=self._input_query(), comment=expected_comment)

        self._deploy_and_run_dag(dag)

        rows = self.session.sql(
            f"SHOW TASKS LIKE '{self._dag_name}$BATCH_INFERENCE' IN SCHEMA {self._test_db}.{self._test_schema}"
        ).collect()
        self.assertGreater(len(rows), 0, f"Task {self._dag_name}$BATCH_INFERENCE not found")
        self.assertEqual(rows[0]["comment"], expected_comment)

        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)


if __name__ == "__main__":
    absltest.main()
