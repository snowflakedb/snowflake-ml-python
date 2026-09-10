import logging
import os
import tempfile
import time
import uuid
from datetime import timedelta
from typing import Any

import pandas as pd
from absl.testing import absltest
from pydantic import BaseModel

try:
    from snowflake.core import Root
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGRun, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.ml.model import inference_engine, openai_signatures
from snowflake.ml.model.batch_inference import (
    BatchInferenceTask,
    EngineOptions,
    InferenceSpec,
    InputSpec,
    OutputSpec,
    ResourcesSpec,
)
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

logger = logging.getLogger(__name__)


class TestBatchInferenceTaskVllmInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test BatchInferenceTask with the vLLM inference engine."""

    _DAG_POLL_INTERVAL_SEC = 15
    _DAG_POLL_MAX_ATTEMPTS = 120  # 30 min total

    # A batch deploy creates several jobs in the schema: the inference job plus server-side
    # ``MODEL_BUILD_<hash>`` / ``MODEL_LOGGING_<hash>`` sub-services.
    _SUBSERVICE_PREFIXES = ("MODEL_BUILD_", "MODEL_LOGGING_")

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        cls.hf_token = os.getenv("HF_TOKEN", None)
        cls._original_hf_endpoint: str | None = None
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint

    def setUp(self) -> None:
        if not _HAS_SNOWFLAKE_CORE:
            self.skipTest("snowflake.core is not installed")
        super().setUp()
        self._dag_name = f"test_dag_{uuid.uuid4().hex[:8]}"
        self._jobs_before_run: set[str] | None = None

    def _apply_dag_task_image_overrides(self) -> None:
        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {root_task_fqn}$BATCH_INFERENCE SET {param} = '{value}'").collect()
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

    def _dag_operation(self) -> "DAGOperation":
        api_root = Root(self.session)
        return DAGOperation(api_root.databases[self._test_db].schemas[self._test_schema])

    def _poll_dag_run_completion(self, dag: "DAG", *, exclude_run_ids: set[int] | None = None) -> "DAGRun":
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
        """Deploy the graph, apply the container image overrides, and trigger a run.

        Args:
            dag: The task graph to deploy and run.
        """
        dag_op = self._dag_operation()
        dag_op.deploy(dag, mode="orReplace")
        self._apply_dag_task_image_overrides()
        self._jobs_before_run = self._snapshot_jobs()
        dag_op.run(dag)

    def _list_jobs(self) -> list[str] | None:
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

    def _snapshot_jobs(self) -> set[str] | None:
        """Upper-cased names of the jobs that exist right now.

        Returns:
            The job names upper-cased, or None when they cannot be listed.
        """
        job_names = self._list_jobs()
        return None if job_names is None else {job_name.upper() for job_name in job_names}

    def _new_jobs(self) -> list[str] | None:
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

    def _vllm_inference_spec(self) -> InferenceSpec:
        """Inference block selecting the vLLM engine with the test's engine arguments.

        Returns:
            The inference block.
        """
        return InferenceSpec(
            engine_options=EngineOptions(
                engine=inference_engine.InferenceEngine.VLLM,
                engine_args_override=[
                    "--gpu-memory-utilization=0.8",
                    "--max-model-len=1024",
                ],
            )
        )

    def test_vllm_inference_engine(self) -> None:
        """Verify batch inference with vLLM engine works through a BatchInferenceTask."""
        # TODO: Restore remote logging via token_or_secret once HF rate limiting is resolved.
        model = huggingface.TransformersPipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            compute_pool_for_log=None,
        )

        name = f"model_{uuid.uuid4().hex[:8]}"
        version = f"ver_{self._run_id}"
        conda_deps = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
            conda_dependencies=conda_deps,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

        input_table = f"{self._test_db}.{self._test_schema}.vllm_input_{uuid.uuid4().hex[:8]}"
        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "What is the capital of France?"}],
                        },
                    ],
                    "temperature": 0.7,
                    "max_completion_tokens": 100,
                    "stop": None,
                    "n": 1,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                    "response_format": None,
                }
            ]
        )
        self.session.create_dataframe(x_df).write.save_as_table(input_table, mode="overwrite")

        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/vllm_base_stage/"

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
                model_version=mv,
                query=f"SELECT * FROM {input_table}",
                compute_pool=self._TEST_GPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                resources_spec=ResourcesSpec(gpu_requests="1"),
                inference_spec=self._vllm_inference_spec(),
            )
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

    @absltest.skip("DAG test_dag_46edb616 did not complete within 1800s")
    def test_vllm_batch_dag_response_format_extracts_output_to_table(self) -> None:
        """Batch vLLM with ``response_format`` finishes, then SQL extracts ``city``/``country`` from assistant JSON.

        The successor task materializes parquet, parses ``choices[0].message.content``, and stores structured fields
        as columns.
        """

        model = huggingface.TransformersPipeline(
            task="image-text-to-text",
            model="google/gemma-4-E2B-it",
            compute_pool_for_log=None,
        )

        class CityCountry(BaseModel):
            city: str
            country: str

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "city_country",
                "schema": CityCountry.model_json_schema(),
            },
        }

        name = f"model_{uuid.uuid4().hex[:8]}"
        version = f"ver_{self._run_id}"
        conda_deps = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            conda_dependencies=conda_deps,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is the capital of France?"},
                            ],
                        },
                    ],
                }
            ]
        )
        input_table = f"{self._test_db}.{self._test_schema}.vllm_rf_input_{uuid.uuid4().hex[:8]}"
        self.session.create_dataframe(x_df).write.save_as_table(input_table, mode="overwrite")

        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        ff_name = f"BATCH_RF_PARQUET_FF_{uuid.uuid4().hex[:8]}"
        self.session.sql(
            f"CREATE OR REPLACE FILE FORMAT {self._test_db}.{self._test_schema}.{ff_name} TYPE = PARQUET"
        ).collect()

        result_table = f"{self._test_db}.{self._test_schema}.batch_rf_extract_{uuid.uuid4().hex[:8]}"
        ff_fqn = f"{self._test_db}.{self._test_schema}.{ff_name}"
        # Stage parquet reads expose each row as $1 (VARIANT), not named columns. Bracket paths
        # on raw:"choices" mirror SERVICE ! __call__ output; CAST(content AS OBJECT) parses JSON.
        # The results live under <output_stage_location>/<job_name>/, and the recursive PATTERN
        # picks them up without the test knowing the server-generated job name.
        extract_sql = f"""
CREATE OR REPLACE TABLE {result_table} AS
WITH src AS (
  SELECT $1 AS raw FROM {output_stage_location}
    (FILE_FORMAT => '{ff_fqn}', PATTERN => '.*\\.parquet')
),
extracted_structured_output AS (
  SELECT
    CAST(raw:"choices" AS ARRAY) AS choices,
    CAST(raw:"choices"[0]['message']['content'] AS OBJECT) AS content
  FROM src
)
SELECT
  content['city']::VARCHAR AS city,
  content['country']::VARCHAR AS country
FROM extracted_structured_output
""".strip()

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
                model_version=mv,
                query=f"SELECT * FROM {input_table}",
                compute_pool=self._TEST_GPU_COMPUTE_POOL,
                input_spec=InputSpec(params={"response_format": response_format}),
                output_spec=OutputSpec(stage_location=output_stage_location),
                resources_spec=ResourcesSpec(gpu_requests="1"),
                inference_spec=self._vllm_inference_spec(),
            )
            extract_task = DAGTask("extract_output", definition=extract_sql)
            data_prep_task >> batch_inference_task >> extract_task

        self._deploy_and_run_dag(dag)
        self._assert_run_succeeded(self._poll_dag_run_completion(dag))
        self._assert_success_file_written(output_stage_location)

        out_pdf = self.session.table(result_table).to_pandas()
        self.assertEqual(len(out_pdf), 1, f"Expected one output row, got {len(out_pdf)}: {out_pdf}")
        col_map = {c.lower(): c for c in out_pdf.columns}
        self.assertIn("city", col_map, f"Expected city column; got {list(out_pdf.columns)}")
        self.assertIn("country", col_map, f"Expected country column; got {list(out_pdf.columns)}")
        city_val = out_pdf.iloc[0][col_map["city"]]
        country_val = out_pdf.iloc[0][col_map["country"]]
        self.assertIsInstance(city_val, str)
        self.assertIsInstance(country_val, str)
        self.assertGreater(len(city_val), 0, "city should be non-empty")
        self.assertGreater(len(country_val), 0, "country should be non-empty")


if __name__ == "__main__":
    absltest.main()
