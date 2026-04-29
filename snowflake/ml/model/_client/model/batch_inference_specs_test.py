from absl.testing import absltest

from snowflake.ml.model._client.model import batch_inference_specs


class BatchInferenceSpecsTest(absltest.TestCase):
    def test_job_spec_name_and_prefix_exclusivity(self) -> None:
        """Test JobSpec raises error when both job_name and job_name_prefix are set."""
        with self.assertRaises(ValueError) as ctx:
            batch_inference_specs.JobSpec(
                job_name="CUSTOM_JOB",
                job_name_prefix="CUSTOM_PREFIX",
            )
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_output_spec_stage_exclusivity(self) -> None:
        """Test OutputSpec raises error when both or neither stage locations are set."""
        with self.assertRaises(ValueError):
            batch_inference_specs.OutputSpec(stage_location="@stage/path/", base_stage_location="@stage/base/")

        with self.assertRaises(ValueError):
            batch_inference_specs.OutputSpec()


if __name__ == "__main__":
    absltest.main()
