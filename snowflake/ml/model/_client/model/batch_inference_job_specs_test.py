from typing import Any

from absl.testing import absltest

from snowflake.ml.model import inference_engine as inference_engine_module
from snowflake.ml.model._client.model import batch_inference_job_specs


class BatchInferenceJobSpecsTest(absltest.TestCase):
    def test_output_spec_requires_stage_location(self) -> None:
        with self.assertRaises(ValueError):
            batch_inference_job_specs.OutputSpec()  # type: ignore[call-arg]

    def test_output_spec_default_mode_is_error(self) -> None:
        spec = batch_inference_job_specs.OutputSpec(stage_location="@db.schema.stage/out/")
        self.assertEqual(spec.mode, batch_inference_job_specs.SaveMode.ERROR)

    def test_output_spec_accepts_mode_string(self) -> None:
        mode: Any = "overwrite"
        spec = batch_inference_job_specs.OutputSpec(stage_location="@db.schema.stage/out/", mode=mode)
        self.assertEqual(spec.mode, batch_inference_job_specs.SaveMode.OVERWRITE)

    def test_output_spec_rejects_invalid_mode(self) -> None:
        mode: Any = "nope"
        with self.assertRaises(ValueError):
            batch_inference_job_specs.OutputSpec(stage_location="@db.schema.stage/out/", mode=mode)

    def test_input_spec_defaults_are_none(self) -> None:
        spec = batch_inference_job_specs.InputSpec()
        self.assertIsNone(spec.params)
        self.assertIsNone(spec.column_handling)
        self.assertIsNone(spec.partition_column)

    def test_input_spec_column_handling_coerces_strings(self) -> None:
        column_handling: dict[str, Any] = {"IMG": {"input_format": "full_stage_path", "convert_to": "base64"}}
        spec = batch_inference_job_specs.InputSpec(column_handling=column_handling)
        self.assertEqual(
            spec.model_dump(mode="json", exclude_none=True)["column_handling"],
            {"IMG": {"input_format": "full_stage_path", "convert_to": "base64"}},
        )

    def test_input_spec_column_handling_rejects_invalid(self) -> None:
        column_handling: dict[str, Any] = {"IMG": {"input_format": "not_a_format", "convert_to": "base64"}}
        with self.assertRaises(ValueError):
            batch_inference_job_specs.InputSpec(column_handling=column_handling)

    def test_engine_options_accepts_enum(self) -> None:
        opts = batch_inference_job_specs.EngineOptions(
            engine=inference_engine_module.InferenceEngine.VLLM,
            engine_args_override=["--max-num-seqs=128"],
        )
        self.assertEqual(opts.engine, inference_engine_module.InferenceEngine.VLLM)
        self.assertEqual(opts.engine_args_override, ["--max-num-seqs=128"])

    def test_engine_options_accepts_case_insensitive_string(self) -> None:
        # Match V1 leniency: accept value or member name, any case, whitespace-stripped.
        values: tuple[Any, ...] = ("vllm", "VLLM", "Vllm", " vllm ")
        for value in values:
            self.assertEqual(
                batch_inference_job_specs.EngineOptions(engine=value).engine,
                inference_engine_module.InferenceEngine.VLLM,
            )
        python_generic: Any = "PYTHON_GENERIC"
        self.assertEqual(
            batch_inference_job_specs.EngineOptions(engine=python_generic).engine,
            inference_engine_module.InferenceEngine.PYTHON_GENERIC,
        )

    def test_engine_options_rejects_unknown_engine(self) -> None:
        engine: Any = "not-an-engine"
        with self.assertRaises(ValueError):
            batch_inference_job_specs.EngineOptions(engine=engine)

    def test_engine_options_serializes_engine_name_uppercase(self) -> None:
        opts = batch_inference_job_specs.EngineOptions(engine=inference_engine_module.InferenceEngine.VLLM)
        self.assertEqual(opts.model_dump(mode="json", exclude_none=True)["engine"], "VLLM")

    def test_engine_options_defaults_omit_engine(self) -> None:
        # engine=None must round-trip through both the coercion validator and the name serializer.
        opts = batch_inference_job_specs.EngineOptions()
        self.assertIsNone(opts.engine)
        self.assertIsNone(opts.engine_args_override)
        self.assertEqual(opts.model_dump(mode="json", exclude_none=True), {})

    def test_inference_spec_serializes_nested_engine_name(self) -> None:
        # The engine name serializer must fire when EngineOptions is nested in the job spec.
        spec = batch_inference_job_specs.InferenceSpec(
            engine_options=batch_inference_job_specs.EngineOptions(engine=inference_engine_module.InferenceEngine.VLLM)
        )
        self.assertEqual(
            spec.model_dump(mode="json", exclude_none=True),
            {"engine_options": {"engine": "VLLM"}},
        )

    def test_resources_spec_all_optional(self) -> None:
        spec = batch_inference_job_specs.ResourcesSpec()
        self.assertIsNone(spec.cpu_requests)
        self.assertIsNone(spec.memory_requests)
        self.assertIsNone(spec.gpu_requests)

    def test_inference_spec_omits_none_fields(self) -> None:
        spec = batch_inference_job_specs.InferenceSpec(num_workers=4)
        self.assertEqual(spec.model_dump(mode="json", exclude_none=True), {"num_workers": 4})

    def test_image_build_spec_default_force_rebuild_false(self) -> None:
        spec = batch_inference_job_specs.ImageBuildSpec()
        self.assertFalse(spec.force_rebuild)

    def test_all_specs_reject_unknown_fields(self) -> None:
        # extra="forbid" makes typos fail at construction rather than being silently dropped.
        cases = [
            (batch_inference_job_specs.InputSpec, {}),
            (batch_inference_job_specs.OutputSpec, {"stage_location": "@db.s.stage/out/"}),
            (batch_inference_job_specs.ResourcesSpec, {}),
            (batch_inference_job_specs.EngineOptions, {}),
            (batch_inference_job_specs.InferenceSpec, {}),
            (batch_inference_job_specs.ImageBuildSpec, {}),
        ]
        for spec_cls, kwargs in cases:
            with self.assertRaises(ValueError, msg=f"{spec_cls.__name__} should reject unknown fields"):
                spec_cls(**kwargs, definitely_not_a_field=1)


if __name__ == "__main__":
    absltest.main()
