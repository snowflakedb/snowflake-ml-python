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
    from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.ml.model import inference_engine, openai_signatures
from snowflake.ml.model.batch import BatchInferenceDefinition, JobSpec, OutputSpec
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

logger = logging.getLogger(__name__)


class TestBatchInferenceDefinitionVllmInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test BatchInferenceDefinition DAG tasks with vLLM inference engine."""

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
        """Set image override session parameters on a DAG task via ALTER TASK."""
        for param, value in self._get_batch_image_override_session_params().items():
            self.session.sql(f"ALTER TASK IF EXISTS {task_fqn} SET {param} = '{value}'").collect()

    def _apply_dag_task_image_overrides(self) -> None:
        """Suspend root task, set image overrides on batch inference child, then resume."""
        if not self._has_image_override():
            return
        root_task_fqn = f"{self._test_db}.{self._test_schema}.{self._dag_name}"
        self.session.sql(f"ALTER TASK {root_task_fqn} SUSPEND").collect()
        self._set_task_image_overrides(f"{root_task_fqn}$BATCH_INFERENCE")
        self.session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()

    def _poll_dag_completion(self, dag_name: str, task_name: str) -> tuple[str, str]:
        """Poll TASK_HISTORY until the DAG task completes or times out."""
        for _ in range(self._DAG_POLL_MAX_ATTEMPTS):
            result = self.session.sql(
                f"""
                SELECT NAME, STATE, ERROR_MESSAGE
                FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
                    SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP())
                ))
                WHERE NAME ILIKE '%{dag_name}%'
                ORDER BY SCHEDULED_TIME DESC
                LIMIT 10
            """
            ).collect()

            for row in result:
                logger.info(f"  Task {row['NAME']}: {row['STATE']} error={row['ERROR_MESSAGE']}")

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

    def test_vllm_inference_engine(self) -> None:
        """Verify batch inference with vLLM engine works through a DAG task.

        This test only verifies that the DAG task completes successfully and produces output.
        Output correctness is tested in other tests.
        """
        # 1. Create and register a HuggingFace text-generation model
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

        # 2. Create input DataFrame with OpenAI chat messages
        # Persist to a real table so the DAG task can access it later (temp tables may be GC'd).
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

        # 3. Construct BatchInferenceDefinition with vLLM engine
        base_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/vllm_base_stage/"
        warehouse = self._TEST_SPCS_WH
        image_repo = ".".join([self._test_db, self._test_schema, self._test_image_repo])

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=input_df,
            compute_pool=self._TEST_GPU_COMPUTE_POOL,
            output_spec=OutputSpec(base_stage_location=base_stage_location),
            job_spec=JobSpec(
                warehouse=warehouse,
                image_repo=image_repo,
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

        # 4. Create and deploy DAG
        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=warehouse,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition="SELECT 'data_preparation done'")
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            data_prep_task >> batch_inference_task

        # 5. Deploy, run, poll for completion, and verify output
        self._deploy_and_run_dag(dag)
        self._assert_batch_inference_succeeded(base_stage_location)


if __name__ == "__main__":
    absltest.main()
