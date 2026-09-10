import re
from collections import UserDict
from typing import Any, cast
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.model import target_platform
from snowflake.ml.model._client.model_spec import (
    legacy_model_spec,
    model_extension_spec,
    model_spec_parser,
)
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema


def _model_extension_spec() -> dict[str, Any]:
    return {
        "version": "2.0",
        "extension_field": {"retained": True},
        "model": {
            "type": "USER_MODEL",
            "framework": "CUSTOM_RUNTIME",
            "details": {
                "models": {
                    "primary": {
                        "model_type": "CUSTOM_RUNTIME",
                        "path": "artifacts/model",
                        "options": {"task": "forecasting", "custom_option": 7},
                        "custom_blob_field": "retained",
                    },
                    "secondary": {
                        "model_type": "OTHER_RUNTIME",
                        "options": {"task": "embedding"},
                    },
                },
                "custom_details_field": ["retained"],
            },
            "target_platforms": [target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value],
        },
        "serving": {
            "env": {
                "custom_env": {
                    "type": "python",
                    "python_version": "3.10",
                }
            },
            "functions": {
                "predict": {
                    "env": ["python_env"],
                    "type": "TABLE_FUNCTION",
                    "handler": "functions.predict.infer",
                    "signature": {"inputs": [{"name": "x"}], "outputs": [{"name": "y"}]},
                    "properties": {"is_partition": False, "case_sensitive": True},
                    "custom_function_field": "retained",
                },
                "forecast": {
                    "type": "TABLE_FUNCTION",
                    "handler": "functions.forecast.infer",
                    "signature": {"inputs": [], "outputs": []},
                },
            },
        },
    }


class ModelSpecParserTest(parameterized.TestCase):
    def test_v2_retains_nested_spec_and_common_accessors(self) -> None:
        raw_spec = _model_extension_spec()

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertIsInstance(parsed, model_extension_spec.ModelExtensionSpecV2)
        self.assertEqual(parsed.raw_spec, raw_spec)
        self.assertEqual(parsed.spec_version, "2.0")
        self.assertEqual(parsed.model_type, "custom_runtime")
        self.assertEqual(parsed.signatures["predict"], raw_spec["serving"]["functions"]["predict"]["signature"])
        self.assertEqual(parsed.model_blobs, raw_spec["model"]["details"]["models"])
        self.assertEqual(
            parsed.model_options,
            {
                "primary": {"task": "forecasting", "custom_option": 7},
                "secondary": {"task": "embedding"},
            },
        )
        self.assertEqual(parsed.model_tasks, {"primary": "forecasting", "secondary": "embedding"})
        self.assertTrue(parsed.supports_gpu)
        self.assertFalse(parsed.is_partitioned("predict"))
        self.assertFalse(parsed.is_partitioned("forecast"))
        self.assertTrue(parsed.is_partitioned("unknown"))
        self.assertEqual(parsed.method_options, {"predict": {"is_partition": False, "case_sensitive": True}})
        self.assertTrue(parsed.case_sensitive)

    def test_numeric_v2_version_is_canonicalized_without_changing_raw_spec(self) -> None:
        raw_spec = _model_extension_spec()
        raw_spec["version"] = 2.0

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertIsInstance(parsed, model_extension_spec.ModelExtensionSpecV2)
        self.assertEqual(parsed.spec_version, "2.0")
        self.assertEqual(parsed.raw_spec["version"], 2.0)

    def test_v2_accepts_mapping_input(self) -> None:
        raw_spec = UserDict(_model_extension_spec())

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertEqual(parsed.spec_version, "2.0")
        self.assertEqual(parsed.model_type, "custom_runtime")

    def test_v2_null_optional_fields_are_treated_as_absent(self) -> None:
        raw_spec = _model_extension_spec()
        raw_spec["model"]["target_platforms"] = None
        raw_spec["model"]["details"]["models"]["primary"]["options"] = None
        raw_spec["serving"]["env"] = None
        raw_spec["serving"]["functions"]["predict"]["properties"] = None

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertFalse(parsed.supports_gpu)
        self.assertEqual(parsed.model_options["primary"], {})
        self.assertIsNone(parsed.model_tasks["primary"])
        self.assertFalse(parsed.is_partitioned("predict"))
        self.assertEmpty(parsed.method_options)

    def test_v2_is_partition_omitted_is_false(self) -> None:
        raw_spec = _model_extension_spec()
        raw_spec["serving"]["functions"]["predict"]["properties"] = {"is_partition": True}

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertTrue(parsed.is_partitioned("predict"))
        self.assertFalse(parsed.is_partitioned("forecast"))
        self.assertTrue(parsed.is_partitioned("unknown"))

    def test_raw_spec_is_isolated_from_input_and_returned_copy_mutations(self) -> None:
        raw_spec = _model_extension_spec()
        parsed = model_spec_parser.parse_model_spec(raw_spec)

        raw_spec["model"]["framework"] = "MUTATED_INPUT"
        returned_raw_spec = cast(dict[str, Any], parsed.raw_spec)
        returned_raw_spec["model"]["framework"] = "MUTATED_COPY"

        self.assertEqual(parsed.model_type, "custom_runtime")
        self.assertEqual(parsed.raw_spec["model"]["framework"], "CUSTOM_RUNTIME")

    def test_v2_non_huggingface_type_and_spcs_gpu_without_cuda(self) -> None:
        raw_spec = _model_extension_spec()
        raw_spec["model"].pop("framework")
        raw_spec["model"]["details"]["models"]["primary"]["model_type"] = "acme_runtime"
        raw_spec["model"]["details"]["models"].pop("secondary")
        raw_spec["model"]["target_platforms"] = [
            target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value.lower()
        ]
        raw_spec["serving"]["functions"]["predict"].pop("properties")

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertEqual(parsed.model_type, "acme_runtime")
        self.assertTrue(parsed.supports_gpu)
        self.assertEmpty(parsed.method_options)
        self.assertFalse(parsed.case_sensitive)

    def test_warehouse_target_without_cuda_does_not_support_gpu(self) -> None:
        raw_spec = _model_extension_spec()
        raw_spec["model"]["target_platforms"] = [target_platform.TargetPlatform.WAREHOUSE.value]

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertFalse(parsed.supports_gpu)

    @mock.patch.object(model_meta.ModelMetadata, "_validate_model_metadata", autospec=True)
    def test_v2_bypasses_legacy_validation(self, validate_model_metadata: mock.Mock) -> None:
        parsed = model_spec_parser.parse_model_spec(_model_extension_spec())

        self.assertIsInstance(parsed, model_extension_spec.ModelExtensionSpecV2)
        validate_model_metadata.assert_not_called()

    @parameterized.named_parameters(  # type: ignore[misc]
        ("non_dict", [], "document must be a mapping"),
        ("missing_model", {"version": "2.0", "serving": {"functions": {}}}, "model section must be a mapping"),
        (
            "missing_serving",
            {"version": "2.0", "model": {"details": {"models": {}}}},
            "serving section must be a mapping",
        ),
        (
            "missing_details",
            {"version": "2.0", "model": {}, "serving": {"functions": {}}},
            "model details must be a mapping",
        ),
        (
            "missing_models",
            {"version": "2.0", "model": {"details": {}}, "serving": {"functions": {}}},
            "model blobs must be a mapping",
        ),
        (
            "missing_functions",
            {"version": "2.0", "model": {"details": {"models": {}}}, "serving": {}},
            "serving functions must be a mapping",
        ),
        (
            "non_dict_blob",
            {
                "version": "2.0",
                "model": {"details": {"models": {"primary": []}}},
                "serving": {"functions": {}},
            },
            "model blob 'primary' must be a mapping",
        ),
        (
            "non_dict_blob_options",
            {
                "version": "2.0",
                "model": {"details": {"models": {"primary": {"options": []}}}},
                "serving": {"functions": {}},
            },
            "model blob 'primary' options must be a mapping",
        ),
        (
            "non_list_target_platforms",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}, "target_platforms": "WAREHOUSE"},
                "serving": {"functions": {}},
            },
            "target platforms must be a list",
        ),
        (
            "non_dict_serving_environments",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}},
                "serving": {"env": [], "functions": {}},
            },
            "serving environments must be a mapping",
        ),
        (
            "non_dict_function",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}},
                "serving": {"functions": {"predict": []}},
            },
            "function 'predict' must be a mapping",
        ),
        (
            "missing_signature",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}},
                "serving": {"functions": {"predict": {}}},
            },
            "function 'predict' signature must be a mapping",
        ),
        (
            "non_dict_signature",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}},
                "serving": {"functions": {"predict": {"signature": []}}},
            },
            "function 'predict' signature must be a mapping",
        ),
        (
            "non_dict_function_properties",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}},
                "serving": {"functions": {"predict": {"signature": {}, "properties": []}}},
            },
            "function 'predict' properties must be a mapping",
        ),
        (
            "non_dict_client",
            {
                "version": "2.0",
                "model": {"details": {"models": {}}, "client": []},
                "serving": {"functions": {}},
            },
            "client must be a mapping",
        ),
    )
    def test_malformed_specs_raise_domain_errors(self, raw_spec: Any, expected_message: str) -> None:
        error_message = f"model specification: {expected_message}."
        with self.assertRaisesRegex(ValueError, f"^{re.escape(error_message)}$"):
            model_spec_parser.parse_model_spec(raw_spec)

    @mock.patch.object(model_meta.ModelMetadata, "_validate_model_metadata", autospec=True)
    def test_non_v2_version_uses_legacy_validation(self, validate_model_metadata: mock.Mock) -> None:
        raw_spec = _model_extension_spec()
        raw_spec["version"] = "3.0"
        validated = cast(
            model_meta_schema.ModelMetadataDict,
            {
                "version": "3.0",
                "model_type": "custom",
                "models": {},
                "signatures": {},
                "env": {},
            },
        )
        validate_model_metadata.return_value = validated

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        self.assertIsInstance(parsed, legacy_model_spec.LegacyModelSpec)
        validate_model_metadata.assert_called_once_with(raw_spec)

    @mock.patch.object(model_meta.ModelMetadata, "_validate_model_metadata", autospec=True)
    def test_legacy_uses_existing_validation(self, validate_model_metadata: mock.Mock) -> None:
        raw_spec = {"version": model_meta_schema.MODEL_METADATA_VERSION}
        validated = cast(
            model_meta_schema.ModelMetadataDict,
            {
                "version": model_meta_schema.MODEL_METADATA_VERSION,
                "creation_timestamp": "2026-01-01",
                "env": {"python_version": "3.9", "snowpark_ml_version": "1.0.12", "cuda_version": "11.8"},
                "model_type": "custom",
                "models": {
                    "primary": {
                        "name": "primary",
                        "model_type": "custom",
                        "path": "model",
                        "handler_version": model_meta_schema.MODEL_METADATA_VERSION,
                        "options": {"task": "classification"},
                    }
                },
                "name": "model",
                "signatures": {"predict": {"inputs": [], "outputs": []}},
                "min_snowpark_ml_version": "1.0.12",
                "task": "classification",
                "function_properties": {"predict": {"PARTITIONED": False}},
                "method_options": {"predict": {"strict_input_validation": True}},
                "case_sensitive": True,
            },
        )
        validate_model_metadata.return_value = validated

        parsed = model_spec_parser.parse_model_spec(raw_spec)

        validate_model_metadata.assert_called_once_with(raw_spec)
        self.assertIsInstance(parsed, legacy_model_spec.LegacyModelSpec)
        self.assertEqual(parsed.spec_version, model_meta_schema.MODEL_METADATA_VERSION)
        self.assertEqual(parsed.model_type, "custom")
        self.assertEqual(parsed.signatures, validated["signatures"])
        self.assertEqual(parsed.model_tasks, {"primary": "classification"})
        self.assertTrue(parsed.supports_gpu)
        self.assertFalse(parsed.is_partitioned("predict"))
        self.assertTrue(parsed.is_partitioned("unknown"))
        self.assertTrue(parsed.case_sensitive)

    @mock.patch.object(model_meta.ModelMetadata, "_validate_model_metadata", autospec=True)
    def test_legacy_null_optional_fields_are_treated_as_absent(self, validate_model_metadata: mock.Mock) -> None:
        validated = cast(
            model_meta_schema.ModelMetadataDict,
            {
                "version": model_meta_schema.MODEL_METADATA_VERSION,
                "creation_timestamp": "2026-01-01",
                "env": {"python_version": "3.9", "snowpark_ml_version": "1.0.12"},
                "model_type": "custom",
                "models": {
                    "primary": {
                        "name": "primary",
                        "model_type": "custom",
                        "path": "model",
                        "handler_version": model_meta_schema.MODEL_METADATA_VERSION,
                        "options": None,
                    }
                },
                "name": "model",
                "signatures": {},
                "min_snowpark_ml_version": "1.0.12",
                "task": "classification",
                "function_properties": None,
                "method_options": None,
            },
        )
        validate_model_metadata.return_value = validated

        parsed = model_spec_parser.parse_model_spec({"version": model_meta_schema.MODEL_METADATA_VERSION})

        self.assertEqual(parsed.model_options, {"primary": {}})
        self.assertEqual(parsed.model_tasks, {"primary": "classification"})
        self.assertTrue(parsed.is_partitioned("predict"))
        self.assertEmpty(parsed.method_options)


if __name__ == "__main__":
    absltest.main()
