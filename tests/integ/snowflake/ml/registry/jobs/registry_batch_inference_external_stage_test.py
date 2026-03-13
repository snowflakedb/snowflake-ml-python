"""Test batch inference with external S3-backed stage for unstructured file input."""

import os
import tempfile
import uuid
from contextlib import contextmanager
from typing import Generator

from absl.testing import absltest

from snowflake.ml.model.batch import (
    FileEncoding,
    InputFormat,
    InputSpec,
    JobSpec,
    OutputSpec,
)
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryBatchInferenceExternalStageInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test batch inference with external S3-backed stage.

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

    def test_image_classification_external_stage_fails(self) -> None:
        """Test that image classification with external S3-backed stage fails.

        External stages are not currently supported for batch inference file input.
        This test verifies the expected failure behavior.
        """
        from transformers import pipeline

        model = pipeline(task="image-classification", model="google/vit-base-patch16-224")

        # Generate unique subdirectory for this test run
        test_subdir = f"batch_inference_test/{uuid.uuid4()}"

        # Prepare output stage and get internal stage location
        (
            job_name,
            output_stage_location,
            internal_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        with self._external_stage_path(test_subdir) as external_stage_location:
            # Upload test image to internal stage first (PUT works on internal stages)
            cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
            self.session.sql(
                f"PUT 'file://{cat_file_path}' {internal_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()

            # Copy file from internal stage to external stage using COPY FILES
            self.session.sql(f"COPY FILES INTO {external_stage_location} FROM {internal_stage_location}").collect()

            # Create input DataFrame with external stage path
            data = [
                [f"{external_stage_location}cat.jpeg"],
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

            self.assertEqual(job.status, "FAILED")


if __name__ == "__main__":
    absltest.main()
