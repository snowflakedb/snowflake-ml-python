import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


@absltest.skip("Skipping multi-modality test for now")
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
            service_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_service_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}/cat.jpeg"],
        ]
        column_names = ["images"]
        input_spec = self.session.create_dataframe(data, schema=column_names)

        column_handling = {"IMAGES": {"encoding": batch_inference_specs.FileEncoding.RAW_BYTES}}

        self._test_registry_batch_inference(
            model=model,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            service_name=service_name,
            replicas=1,
            pip_requirements=["pillow"],
            column_handling=column_handling,
        )


if __name__ == "__main__":
    absltest.main()
