import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model.batch import (
    FileEncoding,
    InputFormat,
    InputSpec,
    JobSpec,
    OutputSpec,
)
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryMultiModalityAudioVideoBatchInferenceInteg(
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

    # TODO: figure out why this test is failing
    @absltest.skip("Skipping test")
    def test_automatic_speech_recognition(self) -> None:
        from transformers import pipeline

        model = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")

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
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
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
        from snowflake.ml.model.models.huggingface import TransformersPipeline

        model = TransformersPipeline(task="video-classification", model="nateraw/videomae-base-finetuned-ucf101-subset")

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
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
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


if __name__ == "__main__":
    absltest.main()
