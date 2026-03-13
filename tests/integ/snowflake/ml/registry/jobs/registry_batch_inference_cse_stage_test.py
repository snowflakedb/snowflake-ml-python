"""Test batch inference with CSE (Client-Side Encryption / SNOWFLAKE_FULL) encrypted stage."""

import os
import tempfile

from absl.testing import absltest
from typing_extensions import override

from snowflake.ml.model.batch import (
    FileEncoding,
    InputFormat,
    InputSpec,
    JobSpec,
    OutputSpec,
)
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryBatchInferenceCSEStageInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test batch inference with CSE (SNOWFLAKE_FULL) encrypted stage."""

    def setUp(self) -> None:
        # TODO: this is temporary
        if not self._has_image_override():
            self.skipTest("CSE stage test requires image override to be set for now.")
        super().setUp()

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

    @override
    def _create_stage(self) -> None:
        """Create CSE (SNOWFLAKE_FULL) encrypted stage instead of SSE."""
        print("Creating CSE (SNOWFLAKE_FULL) encrypted stage for testing...")
        self._db_manager.create_stage(self._test_stage, sse_encrypted=False)

    def test_image_classification_cse_stage(self) -> None:
        """Test image classification with CSE encrypted stage."""
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
        ]
        column_names = ["images"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        job = self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            blocking=False,
        )

        job.wait()

        self.assertEqual(job.status, "DONE")


if __name__ == "__main__":
    absltest.main()
