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

try:
    from snowflake.core import Root
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGRun, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.ml.model import inference_engine, openai_signatures
from snowflake.ml.model.batch import BatchInferenceTask, JobSpec, OutputSpec
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
        if not self._has_image_override():
            return
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

        dag_op.deploy(dag, mode="orReplace")
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
        model = huggingface.TransformersPipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
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


if __name__ == "__main__":
    absltest.main()
