from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from datetime import timedelta
from typing import Optional

import pandas as pd
from absl.testing import absltest
from pydantic import BaseModel

try:
    from snowflake.core import Root
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGRun, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.core import _common

from snowflake.ml.model import inference_engine, openai_signatures
from snowflake.ml.model.batch import BatchInferenceTask, InputSpec, JobSpec, OutputSpec
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

logger = logging.getLogger(__name__)


class TestBatchInferenceTaskVllmInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test BatchInferenceTask DAG tasks with vLLM inference engine."""

    _DAG_POLL_INTERVAL_SEC = 15
    _DAG_POLL_MAX_ATTEMPTS = 120  # 30 min total

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        cls.hf_token = os.getenv("HF_TOKEN", None)
        cls._original_hf_endpoint: Optional[str] = None
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

    def _set_task_image_overrides(self, task_fqn: str) -> None:
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {task_fqn} SET {param} = '{value}'").collect()

    def _apply_dag_task_image_overrides(self) -> None:
        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        self._set_task_image_overrides(f"{root_task_fqn}$BATCH_INFERENCE")
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

    def _poll_dag_run_completion(self, dag: DAG) -> DAGRun:
        """Poll until a DAG run reaches a terminal state, then return it.

        Uses :meth:`DAGOperation.get_complete_dag_runs` (covers the past 60 minutes) and
        :meth:`get_current_dag_runs` for liveness logging.
        """
        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)
        terminal_states = {"SUCCEEDED", "FAILED", "CANCELLED"}

        for _ in range(self._DAG_POLL_MAX_ATTEMPTS):
            completed = sorted(
                dag_op.get_complete_dag_runs(dag, error_only=False), key=lambda r: r.run_id, reverse=True
            )
            for run in completed:
                if run.state in terminal_states:
                    return run

            for run in dag_op.get_current_dag_runs(dag):
                logger.info(f"  DAG run {run.run_id}: state={run.state} first_error={run.first_error_message}")

            time.sleep(self._DAG_POLL_INTERVAL_SEC)

        self.fail(
            f"DAG {dag.name} did not complete within {self._DAG_POLL_MAX_ATTEMPTS * self._DAG_POLL_INTERVAL_SEC}s"
        )

    def _deploy_and_run_dag(self, dag: DAG) -> None:
        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode=_common.CreateMode.or_replace)
        self._apply_dag_task_image_overrides()
        dag_op.run(dag)

    def _assert_dag_succeeded(self, dag: DAG, base_stage_location: str) -> None:
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
        input_df = self.session.table(input_table)

        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/vllm_base_stage/"

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
                X=input_df,
                compute_pool=self._TEST_GPU_COMPUTE_POOL,
                output_spec=OutputSpec(base_stage_location=base_stage_location),
                job_spec=JobSpec(
                    gpu_requests="1",
                    job_name_prefix="test_vllm_dag",
                ),
                inference_engine_options={
                    "engine": inference_engine.InferenceEngine.VLLM,
                    "engine_args_override": [
                        "--gpu-memory-utilization=0.8",
                        "--max-model-len=1024",
                    ],
                },
            )
            data_prep_task >> batch_inference_task

        self._deploy_and_run_dag(dag)
        self._assert_dag_succeeded(dag, base_stage_location)

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
        input_df = self.session.table(input_table)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        ff_name = f"BATCH_RF_PARQUET_FF_{uuid.uuid4().hex[:8]}"
        self.session.sql(
            f"CREATE OR REPLACE FILE FORMAT {self._test_db}.{self._test_schema}.{ff_name} TYPE = PARQUET"
        ).collect()

        result_table = f"{self._test_db}.{self._test_schema}.batch_rf_extract_{uuid.uuid4().hex[:8]}"
        ff_fqn = f"{self._test_db}.{self._test_schema}.{ff_name}"
        # Stage parquet reads expose each row as $1 (VARIANT), not named columns. Bracket paths
        # on raw:"choices" mirror SERVICE ! __call__ output; CAST(content AS OBJECT) parses JSON.
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
                X=input_df,
                compute_pool=self._TEST_GPU_COMPUTE_POOL,
                input_spec=InputSpec(params={"response_format": response_format}),
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(
                    job_name=job_name,
                    gpu_requests="1",
                ),
                inference_engine_options={
                    "engine": inference_engine.InferenceEngine.VLLM,
                    "engine_args_override": [
                        "--gpu-memory-utilization=0.8",
                        "--max-model-len=1024",
                    ],
                },
            )
            extract_task = DAGTask("extract_output", definition=extract_sql)
            data_prep_task >> batch_inference_task >> extract_task

        self._deploy_and_run_dag(dag)

        run = self._poll_dag_run_completion(dag)
        self.assertEqual(
            run.state,
            "SUCCEEDED",
            f"DAG run {run.state}: task={run.first_error_task_name} error={run.first_error_message}",
        )

        success_check = self.session.sql(f"LIST {output_stage_location}").collect()
        has_success = any(str(r["name"]).endswith("_SUCCESS") for r in success_check)
        self.assertTrue(has_success, f"Expected batch completion marker under {output_stage_location}")

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
