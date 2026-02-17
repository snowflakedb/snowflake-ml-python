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


class TestRegistryMultiModalityHuggingFacePipelineBatchInferenceInteg(
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
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
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

    def test_image_classification_with_list_stage_files_utility(self) -> None:
        from transformers import pipeline

        from snowflake.ml.utils.stage_file import list_stage_files

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

        input_df = list_stage_files(
            self.session,
            input_files_stage_location,
            column_name="images",
        )

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
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


if __name__ == "__main__":
    absltest.main()
