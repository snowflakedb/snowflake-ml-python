"""Test multi-modality vLLM batch inference with external S3-backed stage for unstructured file input."""

import os
import tempfile
import uuid
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model.batch import JobSpec, OutputSpec
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface
from snowflake.ml.model.openai_signatures import OPENAI_CHAT_SIGNATURE
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryMultiModalityVLLMExternalStageInteg(
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase
):
    """Test multi-modality vLLM batch inference with external S3-backed stage.

    NOTE: This test requires a pre-existing external S3-backed stage. The S3 bucket and
    external stage were manually created and configured with appropriate IAM permissions.
    Currently, the external stage is only set up in preprod8 and prod3 environments.
    This test is quarantined in all other environments.

    External stage setup documentation:
    https://docs.google.com/document/d/16SLJc2kpiOZgt7lspAw-uLxTQ4VvFs3LFajN_ATvX1E/edit?tab=t.0
    """

    _EXTERNAL_STAGE = "BATCH_INFERENCE_INTEGRATION_TESTS.PUBLIC.EXTERNAL_S3_STAGE"

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name

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

    @contextmanager
    def _external_stage_path(self, subdir: str) -> Generator[str, None, None]:
        """Context manager that provides an external stage path and cleans up on exit."""
        stage_location = f"@{self._EXTERNAL_STAGE}/{subdir}/"
        try:
            yield stage_location
        finally:
            self.session.sql(f"REMOVE {stage_location}").collect()

    def _construct_input(self, messages: list[list[dict]]) -> pd.DataFrame:
        import json

        schema = [
            "MESSAGES",
            "TEMPERATURE",
            "MAX_COMPLETION_TOKENS",
            "STOP",
            "N",
            "STREAM",
            "TOP_P",
            "FREQUENCY_PENALTY",
            "PRESENCE_PENALTY",
        ]

        data = [(json.dumps(m), 0.9, 250, None, 1, False, 0.9, 0.2, 0.1) for m in messages]

        return self.session.create_dataframe(data, schema=schema)

    def test_image_and_video_with_vllm_external_stage(self) -> None:
        """Test image+video vLLM inference with files served from an external S3 stage."""
        if not self._has_image_override():
            self.skipTest(
                "Validator asserts the column-alignment fix in ray_inference_job.py; requires image override "
                "so the ray_orchestrator image carries that fix."
            )
        model = huggingface.TransformersPipeline(model="Qwen/Qwen2-VL-2B-Instruct", task="image-text-to-text")

        test_subdir = f"batch_inference_test/{uuid.uuid4()}"

        (
            job_name,
            output_stage_location,
            internal_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        with self._external_stage_path(test_subdir) as external_stage_location:
            image_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
            video_file_path = "tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi"
            for file in [image_file_path, video_file_path]:
                self.session.sql(
                    f"PUT 'file://{file}' {internal_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                ).collect()
            self.session.sql(f"COPY FILES INTO {external_stage_location} FROM {internal_stage_location}").collect()

            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert on cats and kitchens."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe the cat breed and what is happening in the kitchen.",
                            },
                            {"type": "image_url", "image_url": {"url": f"{external_stage_location}cat.jpeg"}},
                            {
                                "type": "video_url",
                                "video_url": {"url": f"{external_stage_location}cutting_in_kitchen.avi"},
                            },
                        ],
                    },
                ],
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are an expert on cats."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please describe both cat images."},
                            {"type": "image_url", "image_url": {"url": f"{external_stage_location}cat.jpeg"}},
                            {"type": "image_url", "image_url": {"url": f"{external_stage_location}cat.jpeg"}},
                        ],
                    },
                ],
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert on kitchen activities."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please describe what is happening in both videos."},
                            {
                                "type": "video_url",
                                "video_url": {"url": f"{external_stage_location}cutting_in_kitchen.avi"},
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": f"{external_stage_location}cutting_in_kitchen.avi"},
                            },
                        ],
                    },
                ],
            ]
            input_df = self._construct_input(messages)

            output_validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
                expected_phrases=["cat"],
                test_case=self,
            )

            self._test_registry_batch_inference(
                model=model,
                X=input_df,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(job_name=job_name, replicas=1, gpu_requests="1"),
                options={"cuda_version": "12.4"},
                signatures=OPENAI_CHAT_SIGNATURE,
                compute_pool="SYSTEM_COMPUTE_POOL_GPU",
                inference_engine_options={
                    "engine": InferenceEngine.VLLM,
                    "engine_args_override": [
                        "--max-model-len=18048",
                        "--gpu-memory-utilization=0.8",
                    ],
                },
                prediction_assert_fn=output_validator,
                assert_container_count=3,  # main, vllm engine, proxy
            )

    def test_image_and_video_with_vllm_mixed_stages(self) -> None:
        """Test image+video vLLM inference with rows spanning external and internal stages."""
        if not self._has_image_override():
            self.skipTest(
                "Validator asserts the column-alignment fix in ray_inference_job.py; requires image override "
                "so the ray_orchestrator image carries that fix."
            )
        model = huggingface.TransformersPipeline(model="Qwen/Qwen2-VL-2B-Instruct", task="image-text-to-text")

        test_subdir = f"batch_inference_test/{uuid.uuid4()}"

        (
            job_name,
            output_stage_location,
            internal_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        with self._external_stage_path(test_subdir) as external_stage_location:
            image_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
            video_file_path = "tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi"
            for file in [image_file_path, video_file_path]:
                self.session.sql(
                    f"PUT 'file://{file}' {internal_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                ).collect()
            self.session.sql(f"COPY FILES INTO {external_stage_location} FROM {internal_stage_location}").collect()

            messages = [
                # Row 1: image from external, video from internal
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert on cats and kitchens."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe the cat breed and what is happening in the kitchen.",
                            },
                            {"type": "image_url", "image_url": {"url": f"{external_stage_location}cat.jpeg"}},
                            {
                                "type": "video_url",
                                "video_url": {"url": f"{internal_stage_location}cutting_in_kitchen.avi"},
                            },
                        ],
                    },
                ],
                # Row 2: both images from external
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are an expert on cats."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please describe both cat images."},
                            {"type": "image_url", "image_url": {"url": f"{external_stage_location}cat.jpeg"}},
                            {"type": "image_url", "image_url": {"url": f"{external_stage_location}cat.jpeg"}},
                        ],
                    },
                ],
                # Row 3: both videos from internal
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert on kitchen activities."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please describe what is happening in both videos."},
                            {
                                "type": "video_url",
                                "video_url": {"url": f"{internal_stage_location}cutting_in_kitchen.avi"},
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": f"{internal_stage_location}cutting_in_kitchen.avi"},
                            },
                        ],
                    },
                ],
            ]
            input_df = self._construct_input(messages)

            output_validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
                expected_phrases=["cat"],
                test_case=self,
            )

            self._test_registry_batch_inference(
                model=model,
                X=input_df,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(job_name=job_name, replicas=1, gpu_requests="1"),
                options={"cuda_version": "12.4"},
                signatures=OPENAI_CHAT_SIGNATURE,
                compute_pool="SYSTEM_COMPUTE_POOL_GPU",
                inference_engine_options={
                    "engine": InferenceEngine.VLLM,
                    "engine_args_override": [
                        "--max-model-len=18048",
                        "--gpu-memory-utilization=0.8",
                    ],
                },
                prediction_assert_fn=output_validator,
                assert_container_count=3,  # main, vllm engine, proxy
            )


if __name__ == "__main__":
    absltest.main()
