import os
import tempfile
from typing import Any

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import InputSpec, JobSpec, OutputSpec, custom_model
from snowflake.ml.model._client.model import batch_inference_specs
from snowflake.ml.model.model_signature import core
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

Messages = list[dict[str, Any]]


class CustomVisualModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        import os

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        super().__init__(context)

        os.environ["HF_HUB_OFFLINE"] = "1"
        self.model = AutoModelForImageTextToText.from_pretrained(
            context.path("model_dir"),
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(context.path("model_dir"))

    def _pre_process(self, messages: list[Messages]) -> list[dict[str, Any]]:
        res = []
        for message in messages:
            inputs = self.processor.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            res.append(inputs)
        return res

    def _construct_single_message(
        self,
        video1: bytes | None,
        video2: bytes | None,
        image1: bytes | None,
        image2: bytes | None,
        text: str | None,
    ) -> tuple[Messages, list[tempfile._TemporaryFileWrapper]]:
        message_content = []
        temp_files = []

        def write_temp_file(suffix: str, data: bytes) -> tempfile._TemporaryFileWrapper:
            nonlocal temp_files

            temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
            temp_file.write(data)
            temp_file.flush()
            temp_files.append(temp_file)
            return temp_file

        if video1:
            temp_video1 = write_temp_file(suffix=".avi", data=video1)
            message_content.append(
                {"type": "video", "video": temp_video1.name},
            )
        if video2:
            temp_video2 = write_temp_file(suffix=".avi", data=video2)
            message_content.append(
                {"type": "video", "video": temp_video2.name},
            )
        if image1:
            temp_image1 = write_temp_file(suffix=".jpeg", data=image1)
            message_content.append(
                {"type": "image", "image": temp_image1.name},
            )
        if image2:
            temp_image2 = write_temp_file(suffix=".jpeg", data=image2)
            message_content.append(
                {"type": "image", "image": temp_image2.name},
            )
        if text:
            message_content.append(
                {"type": "text", "text": text},
            )

        return [
            {
                "role": "user",
                "content": message_content,
            }
        ], temp_files

    def _construct_messages(
        self,
        video1_bytes_list: list[bytes | None],
        video2_bytes_list: list[bytes | None],
        image1_bytes_list: list[bytes | None],
        image2_bytes_list: list[bytes | None],
        text_list: list[str | None],
    ) -> tuple[list[Messages], list[tempfile._TemporaryFileWrapper]]:
        messages = []
        temp_files = []
        for video1_bytes, video2_bytes, image1_bytes, image2_bytes, texts in zip(
            video1_bytes_list,
            video2_bytes_list,
            image1_bytes_list,
            image2_bytes_list,
            text_list,
        ):
            message, temp_files_single = self._construct_single_message(
                video1=video1_bytes, video2=video2_bytes, image1=image1_bytes, image2=image2_bytes, text=texts
            )
            messages.append(message)
            temp_files.extend(temp_files_single)

        return messages, temp_files

    @custom_model.inference_api
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        video1_bytes_list: list[bytes | None] = df["video1"].tolist()
        video2_bytes_list: list[bytes | None] = df["video2"].tolist()
        image1_bytes_list: list[bytes | None] = df["image1"].tolist()
        image2_bytes_list: list[bytes | None] = df["image2"].tolist()
        text_list: list[str | None] = df["prompt"].tolist()

        messages, temp_files = self._construct_messages(
            video1_bytes_list=video1_bytes_list,
            video2_bytes_list=video2_bytes_list,
            image1_bytes_list=image1_bytes_list,
            image2_bytes_list=image2_bytes_list,
            text_list=text_list,
        )
        messages = self._pre_process(messages)
        generated_ids_list = [self.model.generate(**message, max_new_tokens=256) for message in messages]

        for temp_file in temp_files:
            temp_file.close()

        generated_ids_trimmed_list = [
            [out_ids[len(in_ids) :] for in_ids, out_ids in zip(message.input_ids, generated_ids)]  # pyright: ignore
            for message, generated_ids in zip(messages, generated_ids_list)
        ]

        output_text_list = [
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for generated_ids_trimmed in generated_ids_trimmed_list
        ]

        return pd.DataFrame(output_text_list, columns=["text"])  # pyright: ignore

    @classmethod
    def signature(cls) -> core.ModelSignature:
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="video1", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="video2", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="image1", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="image2", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="prompt", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="text", dtype=core.DataType.STRING),
            ],
        )


class TestRegistryMultiModalityHuggingFacePipelineBatchInferenceInteg(
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase
):
    def setUp(self) -> None:
        super().setUp()
        # TODO: this is temporary since batch inference server image not released yet
        if not self._with_image_override():
            self.skipTest("Skipping multi modality tests: image override environment variables not set.")

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

    def test_image_classification(self) -> None:
        from transformers import pipeline

        model = pipeline(task="image-classification", model="google/vit-base-patch16-224")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
        ]
        column_names = ["images"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            }
        }

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
        )

    def test_automatic_speech_recognition(self) -> None:
        model = huggingface.TransformersPipeline(
            model="openai/whisper-small", task="automatic-speech-recognition", compute_pool_for_log=None
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

        data = [[f"{input_files_stage_location}batman_audio.mp3"]] * 3
        column_names = ["audio"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "AUDIO": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            }
        }

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(column_handling=column_handling),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=1, gpu_requests="1"),
            options={"cuda_version": "12.4"},
        )

    def test_video_classification(self) -> None:
        model = huggingface.TransformersPipeline(
            model="nateraw/videomae-base-finetuned-ucf101-subset",
            task="video-classification",
            compute_pool_for_log=None,
        )

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        video_file_path = "tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi"
        self.session.sql(
            f"PUT 'file://{video_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [[f"{input_files_stage_location}cutting_in_kitchen.avi"]] * 3
        column_names = ["video"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "VIDEO": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            }
        }

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(column_handling=column_handling),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=3),
            pip_requirements=["decord", "av", "pillow"],
        )

    # TODO: need to install num2words in the bazel conda environment to make this test work
    @absltest.skip("Skipping custom visual model test as its not working yet")
    def test_custom_visual_model(self) -> None:
        from huggingface_hub import snapshot_download

        path = snapshot_download("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", cache_dir=self.cache_dir.name)
        model = CustomVisualModel(custom_model.ModelContext(artifacts={"model_dir": path}))

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()
        video1_file_path = "tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi"
        for file in [video1_file_path]:
            self.session.sql(
                f"PUT 'file://{file}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()

        # TODO: add more cases with multiple videos and images
        data = [
            [f"{input_files_stage_location}cutting_in_kitchen.avi", None, None, None, "Describe the video content."]
        ]
        column_names = ["video1", "video2", "image1", "image2", "prompt"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "VIDEO1": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            },
            "VIDEO2": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            },
            "IMAGE1": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            },
            "IMAGE2": {
                "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
            },
        }

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            input_spec=InputSpec(column_handling=column_handling),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=[
                "transformers",
                "accelerate==1.12.0",
                "torch==2.9.1",
                "torchvision==0.24.1",
                "pillow",
                "av==16.0.1",
                "num2words",
            ],
        )


if __name__ == "__main__":
    absltest.main()
