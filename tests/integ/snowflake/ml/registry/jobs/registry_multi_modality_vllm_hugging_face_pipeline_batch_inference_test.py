import json
import os
import tempfile
from typing import Callable

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import JobSpec, OutputSpec
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface
from snowflake.ml.model.openai_signatures import OPENAI_CHAT_SIGNATURE
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


def _create_output_validator(
    expected_phrases: list[str], test_case: absltest.TestCase
) -> Callable[[pd.DataFrame], None]:
    """Create a validator function that checks if output contains expected phrases.

    Args:
        expected_phrases: List of phrases that should appear in the output (case-insensitive).
        test_case: The test case instance for assertions.

    Returns:
        A validation function that takes a DataFrame and asserts expected phrases are present.
    """

    def validator(output_df: pd.DataFrame) -> None:
        # Extract content from the 'id' column which contains the choices array
        # Structure: id -> list of choices -> message -> content
        all_content = []

        # Look for 'id' or 'ID' column which contains the actual choices
        id_col = None
        for col in output_df.columns:
            if col.lower() == "id":
                id_col = col
                break

        if id_col is not None:
            for val in output_df[id_col]:
                if val is None:
                    continue

                # Parse JSON string if needed
                test_case.assertIsInstance(val, str)
                val = json.loads(val)

                # val should be a list of choice objects
                test_case.assertIsInstance(val, list)
                for choice in val:
                    test_case.assertIsInstance(choice, dict)
                    message = choice.get("message", {})
                    test_case.assertIsInstance(message, dict)
                    content = message.get("content", "")
                    test_case.assertIsInstance(content, str)
                    if content:
                        all_content.append(str(content))

        output_text = " ".join(all_content).lower()

        # If no content found, show helpful debug info
        if not output_text.strip():
            test_case.fail(f"No content found. Columns: {list(output_df.columns)}. DataFrame:\n{output_df.to_string()}")

        for phrase in expected_phrases:
            test_case.assertIn(
                phrase.lower(),
                output_text,
                f"Expected phrase '{phrase}' not found in output. Output text: {output_text[:1000]}...",
            )

    return validator


class TestRegistryMultiModalityVLLMHuggingFacePipelineBatchInferenceInteg(
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
        batch_inputs = []
        for message in messages:
            batch_inputs.append(
                {
                    "messages": message,
                    "temperature": 0.9,
                    "max_completion_tokens": 250,
                    "stop": None,
                    "n": 1,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.1,
                }
            )
        data = []
        for item in batch_inputs:
            row = (
                json.dumps(item["messages"]),
                item["temperature"],
                item["max_completion_tokens"],
                item["stop"],
                item["n"],
                item["stream"],
                item["top_p"],
                item["frequency_penalty"],
                item["presence_penalty"],
            )
            data.append(row)
        input_df = self.session.create_dataframe(
            data,
            schema=[
                "MESSAGES",
                "TEMPERATURE",
                "MAX_COMPLETION_TOKENS",
                "STOP",
                "N",
                "STREAM",
                "TOP_P",
                "FREQUENCY_PENALTY",
                "PRESENCE_PENALTY",
            ],
        )

        # The audio file contains Batman's "darkness" speech, expect transcription-related content
        output_validator = _create_output_validator(
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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{input_files_stage_location}cat.jpeg",
                            },
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"{input_files_stage_location}cutting_in_kitchen.avi",
                            },
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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{input_files_stage_location}cat.jpeg",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{input_files_stage_location}cat.jpeg",
                            },
                        },
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
                            "video_url": {
                                "url": f"{input_files_stage_location}cutting_in_kitchen.avi",
                            },
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"{input_files_stage_location}cutting_in_kitchen.avi",
                            },
                        },
                    ],
                },
            ],
        ]
        batch_inputs = []
        for message in messages:
            batch_inputs.append(
                {
                    "messages": message,
                    "temperature": 0.9,
                    "max_completion_tokens": 250,
                    "stop": None,
                    "n": 1,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.1,
                }
            )
        data = []
        for item in batch_inputs:
            row = (
                json.dumps(item["messages"]),
                item["temperature"],
                item["max_completion_tokens"],
                item["stop"],
                item["n"],
                item["stream"],
                item["top_p"],
                item["frequency_penalty"],
                item["presence_penalty"],
            )
            data.append(row)
        input_df = self.session.create_dataframe(
            data,
            schema=[
                "MESSAGES",
                "TEMPERATURE",
                "MAX_COMPLETION_TOKENS",
                "STOP",
                "N",
                "STREAM",
                "TOP_P",
                "FREQUENCY_PENALTY",
                "PRESENCE_PENALTY",
            ],
        )

        # Verify output contains expected phrases related to the input content
        output_validator = _create_output_validator(
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


if __name__ == "__main__":
    absltest.main()
