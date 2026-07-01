from absl.testing import absltest

from snowflake.ml.model import inference_engine as inference_engine_module
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

    def test_inference_job_output_requires_stage_location(self) -> None:
        with self.assertRaises(ValueError):
            batch_inference_specs.Output()  # type: ignore[call-arg]

    def test_inference_job_output_default_mode_is_error(self) -> None:
        spec = batch_inference_specs.Output(stage_location="@db.schema.stage/out/")
        self.assertEqual(spec.mode, batch_inference_specs.SaveMode.ERROR)

    def test_inference_job_input_defaults_are_none(self) -> None:
        spec = batch_inference_specs.Input()
        self.assertIsNone(spec.params)
        self.assertIsNone(spec.column_handling)
        self.assertIsNone(spec.partition_column)

    def test_inference_job_engine_options_accepts_enum(self) -> None:
        opts = batch_inference_specs.EngineOptions(
            engine=inference_engine_module.InferenceEngine.VLLM,
            engine_args_override=["--max-num-seqs=128"],
        )
        self.assertEqual(opts.engine, inference_engine_module.InferenceEngine.VLLM)
        self.assertEqual(opts.engine_args_override, ["--max-num-seqs=128"])

    def test_inference_job_resources_all_optional(self) -> None:
        spec = batch_inference_specs.Resources()
        self.assertIsNone(spec.cpu_requests)
        self.assertIsNone(spec.memory_requests)
        self.assertIsNone(spec.gpu_requests)

    def test_inference_job_image_build_default_force_rebuild_false(self) -> None:
        spec = batch_inference_specs.ImageBuild()
        self.assertFalse(spec.force_rebuild)


if __name__ == "__main__":
    absltest.main()
