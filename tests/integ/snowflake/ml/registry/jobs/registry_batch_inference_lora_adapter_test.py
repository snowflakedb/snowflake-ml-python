import json
import logging
import os
import tempfile
from typing import Any

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import (
    ModelVersion,
    PeftAdapter,
    inference_engine,
    openai_signatures,
)
from snowflake.ml.model._client.model import batch_inference_job_specs
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.snowpark import dataframe as snowpark_dataframe
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    lora_adapter_account_gate,
    lora_adapters_enabled_patch,
    test_env_utils,
)

lora_adapters_enabled_patch.enable()

logger = logging.getLogger(__name__)

_TINY_GPT2 = "hf-internal-testing/tiny-gpt2-with-chatml-template"
_SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"
_SMOLLM2_ADAPTER = "Miladsaeedi70/smollm2-135m-scientific-sft-lora"
_SMOLLM2_ADAPTER_REVISION = "c344de0f75239d4de9341f7d38a580cdd8b10d25"


def _write_stub_adapter_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8}, f)
    with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
        f.write(b"")
    return path


class TestBatchInferenceLoraAdapterInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    cache_dir: tempfile.TemporaryDirectory
    _original_cache_dir: str | None = None
    _original_hf_accept_encoding: str | None = None
    _original_hf_home: str | None = None
    _original_hf_endpoint: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        import huggingface_hub

        super().setUpClass()
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]
        hf_session = huggingface_hub.get_session()
        cls._original_hf_accept_encoding = hf_session.headers.get("Accept-Encoding")
        hf_session.headers["Accept-Encoding"] = "identity"

    @classmethod
    def tearDownClass(cls) -> None:
        import huggingface_hub

        if cls._original_cache_dir is not None:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if cls._original_hf_home is not None:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint is not None:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint
        hf_session = huggingface_hub.get_session()
        if cls._original_hf_accept_encoding is None:
            hf_session.headers.pop("Accept-Encoding", None)
        else:
            hf_session.headers["Accept-Encoding"] = cls._original_hf_accept_encoding
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        lora_adapter_account_gate.skip_unless_lora_adapters_account_enabled(self.session)
        self.session.sql("ALTER SESSION SET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL = TRUE").collect()

    def tearDown(self) -> None:
        self.session.sql("ALTER SESSION UNSET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL").collect()
        super().tearDown()

    def _name(self, suffix: str) -> str:
        return db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, suffix).upper()

    def _chat_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "Complete the sentence."}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "A descendant of the Lost City of Atlantis, who swam to Earth while saying, "
                                    ),
                                }
                            ],
                        },
                    ],
                }
            ]
        )

    def _chat_validator(self, *, expected_model_substr: str | None = None) -> Any:
        return registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
            expected_phrases=[],
            test_case=self,
            expected_model_substr=expected_model_substr,
        )

    def _vllm_inference_spec(self) -> batch_inference_job_specs.InferenceSpec:
        return batch_inference_job_specs.InferenceSpec(
            engine_options=batch_inference_job_specs.EngineOptions(
                engine=inference_engine.InferenceEngine.VLLM,
            )
        )

    def _gpu_resources(self) -> batch_inference_job_specs.ResourcesSpec:
        return batch_inference_job_specs.ResourcesSpec(gpu_requests="1")

    def _log_serving_base(self, *, model_name: str, version_name: str) -> ModelVersion:
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model=_SMOLLM2,
            download_snapshot=False,
        )
        return self.registry.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            options={
                "embed_local_ml_library": True,
                "cuda_version": model_env.DEFAULT_CUDA_VERSION,
            },
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
            ],
        )

    def _log_serving_adapter(self, *, base_mv: ModelVersion, model_name: str, version_name: str) -> ModelVersion:
        return self.registry.log_model(
            model=PeftAdapter(
                base_model=base_mv,
                adapter_repo=_SMOLLM2_ADAPTER,
                revision=_SMOLLM2_ADAPTER_REVISION,
            ),
            model_name=model_name,
            version_name=version_name,
        )

    def _log_tiny_with_model_base(self, *, model_name: str, version_name: str) -> ModelVersion:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model=_TINY_GPT2,
            max_length=200,
        )
        return self.registry.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

    def _log_stub_adapter(self, *, base_mv: ModelVersion, model_name: str, version_name: str) -> ModelVersion:
        with tempfile.TemporaryDirectory() as tmp:
            return self.registry.log_model(
                model=PeftAdapter(base_model=base_mv, adapter_path=_write_stub_adapter_dir(tmp)),
                model_name=model_name,
                version_name=version_name,
            )

    def _run_lora_job(
        self,
        mv: ModelVersion,
        X: snowpark_dataframe.DataFrame,
        *,
        output_spec: batch_inference_job_specs.OutputSpec,
        input_spec: batch_inference_job_specs.InputSpec | None = None,
        resources_spec: batch_inference_job_specs.ResourcesSpec | None = None,
        inference_spec: batch_inference_job_specs.InferenceSpec | None = None,
        job_name: str | None = None,
        compute_pool: str | None = None,
        prediction_assert_fn: Any | None = None,
    ) -> Any:
        if (
            inference_spec is None
            or inference_spec.engine_options is None
            or inference_spec.engine_options.engine != inference_engine.InferenceEngine.VLLM
        ):
            self.fail("LoRA batch integration jobs must use vLLM")
        if compute_pool is None:
            compute_pool = self._TEST_GPU_COMPUTE_POOL
        batch_job = mv.run_batch(
            X,
            compute_pool=compute_pool,
            output_spec=output_spec,
            input_spec=input_spec,
            resources_spec=resources_spec,
            inference_spec=inference_spec,
            job_name=job_name,
        )
        try:
            batch_job.wait(timeout=1800)
        except TimeoutError:
            logger.warning("Batch job timed out after 30 minutes. Status: %s", batch_job.status)
        logs = batch_job.get_logs(limit=100) if batch_job.status != "DONE" else None
        self.assertEqual(batch_job.status, "DONE", logs)
        job_output = self._resolve_job_output_stage_location(output_spec.stage_location, batch_job)
        df = self.session.read.option("on_error", "CONTINUE").parquet(job_output)
        if prediction_assert_fn is not None:
            prediction_assert_fn(df.to_pandas())
        return batch_job

    def test_job_adapters_plus_params_model(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        support = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        sql_gen = self._log_serving_adapter(base_mv=base, model_name=self._name("SQL_GEN"), version_name="V1")
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        self._run_lora_job(
            base,
            self.session.create_dataframe(self._chat_df()),
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            input_spec=batch_inference_job_specs.InputSpec(params={"model": "support"}),
            resources_spec=self._gpu_resources(),
            inference_spec=self._vllm_inference_spec().model_copy(
                update={"adapters": {"support": support, "sql_gen": sql_gen}}
            ),
            job_name=job_name,
            prediction_assert_fn=self._chat_validator(
                expected_model_substr=support.fully_qualified_model_name,
            ),
        )

    def test_adapter_run_batch_defaults_params_model(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_serving_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        alias = f"{adapter.fully_qualified_model_name}/VERSIONS/{adapter.version_name}"
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        self._run_lora_job(
            adapter,
            self.session.create_dataframe(self._chat_df()),
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            resources_spec=self._gpu_resources(),
            inference_spec=self._vllm_inference_spec(),
            job_name=job_name,
            prediction_assert_fn=self._chat_validator(expected_model_substr=alias),
        )


if __name__ == "__main__":
    absltest.main()
