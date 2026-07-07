"""Test batch inference with external S3-backed stage for unstructured file input."""

import os
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager

from absl.testing import absltest

from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)


class TestExecuteInferenceJobServiceExternalStageInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
    """Test batch inference with external S3-backed stage.

    NOTE: This test requires a pre-existing external S3-backed stage. The S3 bucket and
    external stage were manually created and configured with appropriate IAM permissions.
    Currently, the external stage is only set up in preprod8 and prod3 environments.
    This test is quarantined in all other environments.
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

    def test_image_classification_external_stage(self) -> None:
        """Test image classification with an external S3-backed stage as file input."""
        from transformers import pipeline

        model = pipeline(task="image-classification", model="google/vit-base-patch16-224")

        test_subdir = f"batch_inference_test/{uuid.uuid4()}"

        (
            job_name,
            output_stage_location,
            internal_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        with self._external_stage_path(test_subdir) as external_stage_location:
            cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
            self.session.sql(
                f"PUT 'file://{cat_file_path}' {internal_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()

            self.session.sql(f"COPY FILES INTO {external_stage_location} FROM {internal_stage_location}").collect()

            data = [
                [f"{external_stage_location}cat.jpeg"],
            ]
            column_names = ["images"]
            input_df = self.session.create_dataframe(data, schema=column_names)

            column_handling = {
                "IMAGES": {
                    "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                    "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
                }
            }

            self._test_registry_execute_inference_job_service(
                model=model,
                X=input_df,
                compute_pool="SYSTEM_COMPUTE_POOL_CPU",
                output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
                input_spec=batch_inference_specs.Input(column_handling=column_handling),
                job_name=job_name,
                replicas=1,
                pip_requirements=["pillow"],
            )

    def test_image_classification_mixed_stages(self) -> None:
        """Test image classification with input rows spanning 1 external and 2 internal stages."""
        from transformers import pipeline

        model = pipeline(task="image-classification", model="google/vit-base-patch16-224")

        test_subdir = f"batch_inference_test/{uuid.uuid4()}"

        (
            job_name,
            output_stage_location,
            _,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        second_internal_stage = "TEST_STAGE_2"
        self._db_manager.create_stage(second_internal_stage, sse_encrypted=True)

        internal_stage_1_root = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/"
        internal_stage_2_root = f"@{self._test_db}.{self._test_schema}.{second_internal_stage}/"

        with self._external_stage_path(test_subdir) as external_stage_location:
            cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"

            self.session.sql(
                f"PUT 'file://{cat_file_path}' {internal_stage_1_root} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()
            self.session.sql(
                f"PUT 'file://{cat_file_path}' {internal_stage_2_root} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            ).collect()

            self.session.sql(f"COPY FILES INTO {external_stage_location} FROM {internal_stage_1_root}").collect()

            data = [
                [f"{external_stage_location}cat.jpeg"],
                [f"{internal_stage_1_root}cat.jpeg"],
                [f"{internal_stage_2_root}cat.jpeg"],
            ]
            column_names = ["images"]
            input_df = self.session.create_dataframe(data, schema=column_names)

            column_handling = {
                "IMAGES": {
                    "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                    "convert_to": batch_inference_specs.FileEncoding.RAW_BYTES,
                }
            }

            self._test_registry_execute_inference_job_service(
                model=model,
                X=input_df,
                compute_pool="SYSTEM_COMPUTE_POOL_CPU",
                output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
                input_spec=batch_inference_specs.Input(column_handling=column_handling),
                job_name=job_name,
                replicas=1,
                pip_requirements=["pillow"],
            )


if __name__ == "__main__":
    absltest.main()
