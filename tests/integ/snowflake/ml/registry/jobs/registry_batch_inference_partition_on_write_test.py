from typing import Any

from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.jobs import (
    registry_batch_inference_partitioned_test,
)


class TestBatchInferencePartitionedPartitionOnWriteInteg(
    registry_batch_inference_partitioned_test.TestBatchInferencePartitionedInteg
):
    """Same partitioned batch-inference cases with the write param on at session level.

    Runs only in Jenkins image-override jobs (``*_IMAGE_PATH`` / ``*_PATH`` env vars set).
    """

    def setUp(self) -> None:
        if not self._has_image_override():
            self.skipTest(
                "Skipping: ENABLE_BATCH_INFERENCE_PARTITION_ON_WRITE variant requires Jenkins image override."
            )
        super().setUp()
        try:
            self.session.sql(f"ALTER SESSION SET {self._PARTITION_ON_WRITE_PARAM} = true").collect()
        except Exception as e:
            self.skipTest(f"Failed to enable {self._PARTITION_ON_WRITE_PARAM}: {e}")
        platform_capabilities.PlatformCapabilities._instance = None
        self.registry = registry.Registry(self.session)
        caps = platform_capabilities.PlatformCapabilities.get_instance(self.session)
        self.assertTrue(caps.is_batch_inference_partition_on_write_enabled())

    def _deploy_batch_inference(self, *args: Any, **kwargs: Any) -> Any:
        batch_job = super()._deploy_batch_inference(*args, **kwargs)
        if kwargs.get("blocking", True) is False:
            return batch_job
        input_spec = kwargs.get("input_spec")
        partition_column = getattr(input_spec, "partition_column", None) if input_spec is not None else None
        if partition_column:
            self._assert_partition_directory_inference_logs(batch_job)
        return batch_job

    def _assert_partition_directory_inference_logs(self, batch_job: Any) -> None:
        """Fail if a keyed job ran groupby / repartition(1) instead of directory map_batches."""
        logs = batch_job.get_logs(limit=-1)
        msg = (
            f"Job {batch_job.id} did not take the partition-directory Ray path "
            f"(COPY PARTITION BY / LAYOUT=partitioned).\n\nJob logs:\n{logs}"
        )
        self.assertIn("Using partition-directory inference", logs, msg)
        self.assertIn("PartitionDirectoryModelActor", logs, msg)
        self.assertNotIn("Using groupby().map_groups()", logs, msg)
        self.assertNotIn("PARTITION BY 1 fallback", logs, msg)


if __name__ == "__main__":
    absltest.main()
