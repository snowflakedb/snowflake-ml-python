from typing import Any, cast

import yaml
from absl.testing import absltest

from snowflake.ml.model import inference_engine as inference_engine_module
from snowflake.ml.model._client.model import batch_inference_specs
from snowflake.ml.model._client.service import inference_job_service_spec


class InferenceJobServiceSpecTest(absltest.TestCase):
    def _build(
        self,
        *,
        with_input: bool = False,
        with_resources: bool = False,
        with_inference: bool = False,
        with_image_build: bool = False,
    ) -> dict[str, Any]:
        builder = inference_job_service_spec.InferenceJobServiceSpec()
        if with_input:
            builder.add_input_spec(
                batch_inference_specs.Input(
                    params={"temperature": 0.7},
                    column_handling={
                        "image_col": {
                            "input_format": batch_inference_specs.InputFormat.FULL_STAGE_PATH,
                            "convert_to": batch_inference_specs.FileEncoding.BASE64,
                        }
                    },
                    partition_column="PART_COL",
                )
            )
        builder.add_output_spec(
            batch_inference_specs.Output(
                stage_location="@DB.SCHEMA.STAGE/out/",
                mode=batch_inference_specs.SaveMode.OVERWRITE,
            )
        )
        if with_resources:
            builder.add_resources_spec(
                batch_inference_specs.Resources(cpu_requests="2", memory_requests="8GiB", gpu_requests=None)
            )
        if with_inference:
            builder.add_inference_spec(
                batch_inference_specs.Inference(
                    num_workers=4,
                    max_batch_rows=2048,
                    engine_options=batch_inference_specs.EngineOptions(
                        engine=inference_engine_module.InferenceEngine.VLLM,
                        engine_args_override=["--max-num-seqs=128"],
                    ),
                )
            )
        if with_image_build:
            builder.add_image_build_spec(
                batch_inference_specs.ImageBuild(
                    image_repo="DB.SCHEMA.REPO",
                    force_rebuild=True,
                )
            )
        return cast(dict[str, Any], yaml.safe_load(builder.save()))

    def test_save_requires_output(self) -> None:
        builder = inference_job_service_spec.InferenceJobServiceSpec()
        with self.assertRaisesRegex(ValueError, "output spec is required"):
            builder.save()

    def test_minimal_body_only_has_output(self) -> None:
        body = self._build()
        self.assertEqual(set(body.keys()), {"output"})
        self.assertEqual(body["output"], {"stage_location": "@DB.SCHEMA.STAGE/out/", "mode": "overwrite"})

    def test_full_body_has_all_blocks_in_canonical_order(self) -> None:
        builder = inference_job_service_spec.InferenceJobServiceSpec()
        builder.add_image_build_spec(batch_inference_specs.ImageBuild(image_repo="DB.SCHEMA.REPO"))
        builder.add_inference_spec(batch_inference_specs.Inference(num_workers=2))
        builder.add_resources_spec(batch_inference_specs.Resources(cpu_requests="1"))
        builder.add_output_spec(batch_inference_specs.Output(stage_location="@stage/"))
        builder.add_input_spec(batch_inference_specs.Input(params={"k": "v"}))
        rendered = builder.save()
        self.assertLess(rendered.index("input"), rendered.index("output"))
        self.assertLess(rendered.index("output"), rendered.index("resources"))
        self.assertLess(rendered.index("resources"), rendered.index("inference"))
        self.assertLess(rendered.index("inference"), rendered.index("image_build"))

    def test_input_emits_raw_params_and_column_handling(self) -> None:
        body = self._build(with_input=True)
        self.assertEqual(body["input"]["params"], {"temperature": 0.7})
        self.assertEqual(
            body["input"]["column_handling"],
            {"image_col": {"input_format": "full_stage_path", "convert_to": "base64"}},
        )
        self.assertEqual(body["input"]["partition_column"], "PART_COL")

    def test_inference_engine_serializes_to_server_enum(self) -> None:
        body = self._build(with_inference=True)
        # The server engine enum is upper-case (DEFAULT / VLLM / PYTHON_GENERIC).
        self.assertEqual(body["inference"]["engine_options"]["engine"], "VLLM")
        self.assertEqual(body["inference"]["engine_options"]["engine_args_override"], ["--max-num-seqs=128"])

    def test_resources_omits_none_fields(self) -> None:
        body = self._build(with_resources=True)
        self.assertEqual(body["resources"], {"cpu_requests": "2", "memory_requests": "8GiB"})

    def test_image_build_emits_force_rebuild(self) -> None:
        body = self._build(with_image_build=True)
        self.assertEqual(body["image_build"], {"image_repo": "DB.SCHEMA.REPO", "force_rebuild": True})

    def test_clear_resets_state(self) -> None:
        builder = inference_job_service_spec.InferenceJobServiceSpec()
        builder.add_output_spec(batch_inference_specs.Output(stage_location="@stage/"))
        builder.add_input_spec(batch_inference_specs.Input(params={"a": 1}))
        builder.clear()
        with self.assertRaises(ValueError):
            builder.save()

    def test_no_unexpected_top_level_keys(self) -> None:
        body = self._build(with_input=True, with_resources=True, with_inference=True, with_image_build=True)
        self.assertEqual(
            set(body.keys()),
            {"input", "output", "resources", "inference", "image_build"},
        )


if __name__ == "__main__":
    absltest.main()
