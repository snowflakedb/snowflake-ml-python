import os
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import (
    InputSpec,
    JobSpec,
    OutputSpec,
    inference_engine,
    openai_signatures,
)
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryHuggingFacePipelineVllmBatchInferenceInteg(
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase
):
    def setUp(self) -> None:
        super().setUp()
        # TODO: this is temporary since batch inference server image not released yet
        if not self._with_image_override():
            self.skipTest("Skipping multi modality tests: image override environment variables not set.")

    def tearDown(self) -> None:
        super().tearDown()

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        # Get HF token if available (used for gated models)
        cls.hf_token = os.getenv("HF_TOKEN", None)
        # Unset HF_ENDPOINT to avoid artifactory errors
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

    @parameterized.parameters(  # type: ignore[misc]
        # Default configuration with system GPU pool
        {
            "compute_pool": "SYSTEM_COMPUTE_POOL_GPU",
            "cpu_requests": None,
            "gpu_requests": "1",
            "memory_requests": None,
            "replicas": None,
            "engine_args_override": None,
        },
        # Custom resource configuration with engine args
        {
            "compute_pool": None,
            "cpu_requests": "2",
            "gpu_requests": "1",
            "memory_requests": "8Gi",
            "replicas": 2,
            "engine_args_override": ["--gpu-memory-utilization=0.9", "--max-model-len=1024"],
        },
    )
    def test_text_generation_with_vllm(
        self,
        compute_pool: Optional[str],
        cpu_requests: Optional[str],
        gpu_requests: Optional[str],
        memory_requests: Optional[str],
        replicas: Optional[int],
        engine_args_override: Optional[list[str]],
    ) -> None:
        """Test text generation with vLLM inference engine and various resource configurations."""
        model = huggingface.TransformersPipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
        )

        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant.",
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What is the capital of France?",
                                },
                            ],
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

        validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
            expected_phrases=["paris", "france"],  # Expected keywords in the response
            test_case=self,
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        input_df = self.session.create_dataframe(x_df)

        self._test_registry_batch_inference(
            model=model,
            compute_pool=compute_pool,
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                cpu_requests=cpu_requests,
                gpu_requests=gpu_requests,
                memory_requests=memory_requests,
                replicas=replicas,
            ),
            prediction_assert_fn=validator,
            inference_engine_options={
                "engine": inference_engine.InferenceEngine.VLLM,
                "engine_args_override": engine_args_override,
            },
            assert_container_count=3,
        )

    def test_text_generation_with_vllm_and_params(self) -> None:
        """Test text generation with vLLM using OPENAI_CHAT_WITH_PARAMS_SIGNATURE.

        This test uses ParamSpec to pass inference parameters via InputSpec
        instead of including them in the input DataFrame. It validates that
        the model produces output content when parameters are passed via InputSpec.
        """
        model = huggingface.TransformersPipeline(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            task="text-generation",
            compute_pool_for_log=None,
        )

        # With OPENAI_CHAT_WITH_PARAMS_SIGNATURE, messages is the only input column.
        # Parameters are passed via InputSpec.
        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a master writer.",
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Write a haiku about the majesty of cats.",
                                },
                            ],
                        },
                    ],
                }
            ]
        )

        validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
            expected_phrases=[],
            test_case=self,
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        input_df = self.session.create_dataframe(x_df)

        # Pass inference parameters via InputSpec
        input_spec = InputSpec(
            params={
                "temperature": 0.0,
                "max_completion_tokens": 4,
                "stop": ["."],
                "n": 1,
            }
        )

        self._test_registry_batch_inference(
            model=model,
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            X=input_df,
            input_spec=input_spec,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                gpu_requests="1",
            ),
            prediction_assert_fn=validator,
            inference_engine_options={
                "engine": inference_engine.InferenceEngine.VLLM,
                "engine_args_override": ["--gpu-memory-utilization=0.9", "--max-model-len=1024"],
            },
            assert_container_count=3,
        )


if __name__ == "__main__":
    absltest.main()
