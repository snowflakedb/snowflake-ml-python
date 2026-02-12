import os
import tempfile

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import JobSpec, OutputSpec
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface
from snowflake.ml.model.openai_signatures import OPENAI_CHAT_SIGNATURE
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryMultiModalityVLLMHuggingFacePipelineBatchInferenceInteg(
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase
):
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

    def test_audio_with_vllm(self) -> None:
        # from transformers import pipeline

        # TODO: update the task to "audio-to-text"
        model = huggingface.TransformersPipeline(
            model="Qwen/Qwen2-Audio-7B-Instruct", task="text-generation", trust_remote_code=True
        )

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        audio_file_path = "tests/integ/snowflake/ml/test_data/batman_audio.mp3"
        self.session.sql(
            f"PUT 'file://{audio_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert audio transcriber."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the following audio file."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"{input_files_stage_location}batman_audio.mp3",
                                "format": "mp3",
                            },
                        },
                    ],
                },
            ],
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert audio transcriber."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the following audio files and concatenate the content."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"{input_files_stage_location}batman_audio.mp3",
                                "format": "mp3",
                            },
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"{input_files_stage_location}batman_audio.mp3",
                                "format": "mp3",
                            },
                        },
                    ],
                },
            ],
        ]
        input_df = self._construct_input(messages)

        # The audio file contains Batman's "darkness" speech, expect transcription-related content
        output_validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
            expected_phrases=["dark", "light"],  # Keywords from the Batman audio
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
                    "--max-model-len=5000",
                    "--gpu-memory-utilization=0.9",
                ],
            },
            prediction_assert_fn=output_validator,
            assert_container_count=3,  # main, vllm engine, proxy
        )

    def test_image_and_video_with_vllm(self) -> None:
        # TODO: change to correct task type
        model = huggingface.TransformersPipeline(model="Qwen/Qwen2-VL-2B-Instruct", task="text-generation")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        image_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        video_file_path = "tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi"
        for file in [image_file_path, video_file_path]:
            self.session.sql(
                f"PUT 'file://{file}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()

        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert on cats and kitchens."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the cat breed and what is happening in the kitchen."},
                        {"type": "image_url", "image_url": {"url": f"{input_files_stage_location}cat.jpeg"}},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"{input_files_stage_location}cutting_in_kitchen.avi"},
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
                        {"type": "image_url", "image_url": {"url": f"{input_files_stage_location}cat.jpeg"}},
                        {"type": "image_url", "image_url": {"url": f"{input_files_stage_location}cat.jpeg"}},
                    ],
                },
            ],
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert on kitchen activities."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe what is happening in both videos."},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"{input_files_stage_location}cutting_in_kitchen.avi"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": f"{input_files_stage_location}cutting_in_kitchen.avi"},
                        },
                    ],
                },
            ],
        ]
        input_df = self._construct_input(messages)

        # Verify output contains expected phrases related to the input content
        output_validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
            expected_phrases=["cat", "onion"],  # Keywords related to the image and video content
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
                    "--gpu-memory-utilization=0.9",
                ],
            },
            prediction_assert_fn=output_validator,
            assert_container_count=3,  # main, vllm engine, proxy
        )

    def test_image_and_video_with_vllm_multi_replica(self) -> None:
        # TODO: change to correct task type
        model = huggingface.TransformersPipeline(model="Qwen/Qwen2-VL-2B-Instruct", task="text-generation")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        image_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        video_file_path = "tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi"
        for file in [image_file_path, video_file_path]:
            self.session.sql(
                f"PUT 'file://{file}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()

        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert on cats and kitchens."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the cat breed and what is happening in the kitchen."},
                        {"type": "image_url", "image_url": {"url": f"{input_files_stage_location}cat.jpeg"}},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"{input_files_stage_location}cutting_in_kitchen.avi"},
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
                        {"type": "image_url", "image_url": {"url": f"{input_files_stage_location}cat.jpeg"}},
                        {"type": "image_url", "image_url": {"url": f"{input_files_stage_location}cat.jpeg"}},
                    ],
                },
            ],
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert on kitchen activities."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe what is happening in both videos."},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"{input_files_stage_location}cutting_in_kitchen.avi"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": f"{input_files_stage_location}cutting_in_kitchen.avi"},
                        },
                    ],
                },
            ],
            [
                {"role": "system", "content": [{"type": "text", "text": "You are master writer."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Write a haiku about the majesty of cats."},
                    ],
                },
            ],
        ]
        input_df = self._construct_input(messages)

        # Verify output contains expected phrases related to the input content
        output_validator = registry_batch_inference_test_base.create_openai_chat_completion_output_validator(
            expected_phrases=["cat", "onion"],  # Keywords related to the image and video content
            test_case=self,
        )

        # TODO: verify column names in output and content
        # output should contain the phrases "kitchen" and "cat".
        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=3, gpu_requests="1", memory_requests="16Gi"),
            options={"cuda_version": "12.4"},
            signatures=OPENAI_CHAT_SIGNATURE,
            inference_engine_options={
                "engine": InferenceEngine.VLLM,
                "engine_args_override": [
                    "--max-model-len=18048",
                    "--gpu-memory-utilization=0.9",
                ],
            },
            prediction_assert_fn=output_validator,
            assert_container_count=3,  # main, vllm engine, proxy
        )

    def test_mljob_get_logs_on_vllm(self) -> None:

        model = huggingface.TransformersPipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B",
        )

        (
            job_name,
            output_stage_location,
            _,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        # job will fail because the input does not match the expected input schema
        dummy_df = self.session.create_dataframe([("Hello world",)], schema=["TEXT"])
        job = self._test_registry_batch_inference(
            model=model,
            X=dummy_df,
            options={"cuda_version": "12.4"},
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, gpu_requests="1"),
            compute_pool="SYSTEM_COMPUTE_POOL_GPU",
            signatures=OPENAI_CHAT_SIGNATURE,
            inference_engine_options={
                "engine": InferenceEngine.VLLM,
                "engine_args_override": [
                    "--max-model-len=5000",
                    "--gpu-memory-utilization=0.9",
                ],
            },
            blocking=False,
        )

        job.wait()
        self.assertEqual(job.status, "FAILED")
        job.get_logs()


if __name__ == "__main__":
    absltest.main()
