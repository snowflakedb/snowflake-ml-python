import os
import tempfile

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import InputSpec, JobSpec, OutputSpec, custom_model
from snowflake.ml.model._client.model import batch_inference_specs
from snowflake.ml.model.model_signature import core
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class CustomTranscriber(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        self.model = self.context.model_ref("my_model")

        # Clear deprecated forced_decoder_ids to fix compatibility with newer transformers versions
        # See: https://github.com/huggingface/transformers/issues/25186
        underlying_model = self.model.model
        if hasattr(underlying_model, "model") and hasattr(underlying_model.model, "generation_config"):
            underlying_model.model.generation_config.forced_decoder_ids = None

    @custom_model.inference_api
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        import base64

        audio_b64_list = df["audio"].tolist()
        audio_bytes_list = [base64.b64decode(audio_b64) for audio_b64 in audio_b64_list]
        temp_res = [self.model(audio_bytes) for audio_bytes in audio_bytes_list]
        return pd.DataFrame({"outputs": temp_res})

    @classmethod
    def signature(cls) -> core.ModelSignature:
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="audio", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureGroupSpec(
                    name="outputs",
                    specs=[
                        core.FeatureSpec(name="text", dtype=core.DataType.STRING),
                        core.FeatureGroupSpec(
                            name="chunks",
                            specs=[
                                core.FeatureSpec(name="timestamp", dtype=core.DataType.DOUBLE, shape=(2,)),
                                core.FeatureSpec(name="text", dtype=core.DataType.STRING),
                            ],
                            shape=(-1,),
                        ),
                    ],
                ),
            ],
        )


class TestRegistryCustomMultiModalityBatchInferenceInteg(
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

    def test_custom_transcriber(self) -> None:
        from transformers import pipeline

        whisper = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
        model = CustomTranscriber(custom_model.ModelContext(models={"my_model": whisper}))

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
                "convert_to": batch_inference_specs.FileEncoding.BASE64,
            }
        }

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_GPU",
            signatures={"predict": CustomTranscriber.signature()},
            input_spec=InputSpec(column_handling=column_handling),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=1, gpu_requests="1"),
            options={"cuda_version": "12.4"},
            pip_requirements=["transformers==4.51.0"],
        )


if __name__ == "__main__":
    absltest.main()
