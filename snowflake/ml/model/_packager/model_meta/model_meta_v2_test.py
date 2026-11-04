import os
import pathlib
import tempfile
from typing import Any
from unittest import mock

import yaml
from absl.testing import absltest

from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.model_spec import model_extension_spec
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_meta_schema,
    model_meta_v2,
)

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}

_PREDICT_SIGNATURE = {
    "inputs": [{"name": "input", "type": "FLOAT", "nullable": False}],
    "outputs": [{"name": "output", "type": "FLOAT", "nullable": False}],
    "params": [],
}


def _minimal_conda_yml() -> str:
    return """
name: snow-env
channels:
  - nodefaults
dependencies:
  - python=3.10
"""


def _write_runtime_env_files(runtime_dir: pathlib.Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "conda.yml").write_text(_minimal_conda_yml(), encoding="utf-8")
    (runtime_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")


def _write_env_files(model_dir: str, *, cpu: bool = False, gpu: bool = False) -> None:
    env_dir = pathlib.Path(model_dir) / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "conda.yml").write_text(_minimal_conda_yml(), encoding="utf-8")
    (env_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    if cpu:
        _write_runtime_env_files(pathlib.Path(model_dir) / "runtimes" / "cpu" / "env")
    if gpu:
        _write_runtime_env_files(pathlib.Path(model_dir) / "runtimes" / "gpu" / "env")


def _single_env_v2_spec() -> dict[str, Any]:
    return {
        "version": "2.0",
        "model": {
            "name": "LinearRegression",
            "type": "USER_MODEL",
            "framework": "SKLEARN",
            "task": "TABULAR_REGRESSION",
            "client": {
                "name": "snowflake-ml-python",
                "version": "1.6.0",
                "min_version": "1.0.12",
            },
            "details": {
                "models": {
                    "LinearRegression": {
                        "model_type": "SKLEARN",
                        "handler_version": "2023-12-01",
                        "path": "model.pkl",
                        "artifacts": {"weights": {"path": "weights.bin", "type": "file"}},
                        "options": {},
                    }
                }
            },
        },
        "serving": {
            "env": {
                "python_env": {
                    "type": "python",
                    "python_version": "3.10",
                    "snowpark_ml_version": "1.6.0",
                    "conda": "model/env/conda.yml",
                    "pip": "model/env/requirements.txt",
                    "imports": ["model/"],
                }
            },
            "functions": {
                "predict": {
                    "env": ["python_env"],
                    "type": "FUNCTION",
                    "handler": "functions.predict.infer",
                    "signature": _PREDICT_SIGNATURE,
                    "properties": {"case_sensitive": True, "max_batch_size": 64},
                }
            },
        },
        "user_data": {"owner": "ml-platform"},
    }


def _gpu_env_v2_spec() -> dict[str, Any]:
    spec = _single_env_v2_spec()
    spec["model"]["framework"] = "XGBOOST"
    spec["model"]["details"]["models"] = {
        "XgboostBooster": {
            "model_type": "XGBOOST",
            "handler_version": "2023-12-01",
            "path": "model.ubj",
            "artifacts": {},
            "options": {},
        }
    }
    spec["serving"]["env"]["gpu_env"] = {
        "type": "python",
        "python_version": "3.10",
        "snowpark_ml_version": "1.6.0",
        "cuda_version": "11.8",
        "conda": "model/runtimes/gpu/env/conda.yml",
        "pip": "model/runtimes/gpu/env/requirements.txt",
        "imports": ["model/"],
    }
    spec["serving"]["functions"]["predict"]["env"] = ["python_env", "gpu_env"]
    spec["serving"]["functions"]["predict_table"] = {
        "env": ["python_env"],
        "type": "TABLE_FUNCTION",
        "handler": "functions.predict_table.infer",
        "signature": _PREDICT_SIGNATURE,
        "properties": {"is_partition": True, "case_sensitive": False},
    }
    return spec


def _load_spec(spec: dict[str, Any], *, cpu: bool = False, gpu: bool = False) -> model_meta.ModelMetadataV2:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_env_files(tmpdir, cpu=cpu, gpu=gpu)
        with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f)
        loaded = model_meta.ModelMetadata.load(tmpdir)
    assert isinstance(loaded, model_meta.ModelMetadataV2)
    return loaded


class ModelMetaV2ValidateTest(absltest.TestCase):
    """Ensures the version 2.0 load path goes through ModelExtensionSpecV2._validate."""

    def test_load_invokes_extension_spec_validate(self) -> None:
        with mock.patch.object(
            model_extension_spec.ModelExtensionSpecV2,
            "_validate",
            wraps=model_extension_spec.ModelExtensionSpecV2._validate,
        ) as validate:
            _load_spec(_single_env_v2_spec())
            validate.assert_called_once()

    def test_load_surfaces_validate_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "serving section must be a mapping"):
            _load_spec(
                {
                    "version": "2.0",
                    "model": {
                        "name": "model",
                        "details": {"models": {"m": {"model_type": "sklearn", "path": "m.pkl"}}},
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "version must be 2.0"):
            model_meta_v2.parse(
                ".",
                {
                    "version": "1.0",
                    "model": {
                        "name": "model",
                        "details": {"models": {"m": {"model_type": "sklearn", "path": "m.pkl"}}},
                    },
                    "serving": {"functions": {"predict": {"type": "FUNCTION", "signature": _PREDICT_SIGNATURE}}},
                },
            )

        with self.assertRaisesRegex(ValueError, "function 'predict' signature must be a mapping"):
            _load_spec(
                {
                    "version": "2.0",
                    "model": {
                        "name": "model",
                        "details": {"models": {"m": {"model_type": "sklearn", "path": "m.pkl"}}},
                    },
                    "serving": {
                        "functions": {
                            "predict": {
                                "type": "FUNCTION",
                                "env": ["python_env"],
                                "handler": "functions.predict.infer",
                            }
                        }
                    },
                }
            )


class ModelMetadataV2ParseTest(absltest.TestCase):
    def test_single_env_strips_model_prefix_and_maps_fields(self) -> None:
        loaded = _load_spec(_single_env_v2_spec())

        self.assertEqual(loaded.original_metadata_version, "2.0")
        self.assertEqual(loaded.name, "LinearRegression")
        self.assertEqual(loaded.model_type, "sklearn")
        self.assertEqual(loaded.task, type_hints.Task.TABULAR_REGRESSION)
        self.assertEqual(loaded.min_snowpark_ml_version, "1.0.12")
        self.assertEqual(loaded.metadata, {"owner": "ml-platform"})
        self.assertIn("predict", loaded.signatures)
        self.assertEqual(loaded.models["LinearRegression"].model_type, "sklearn")
        self.assertEqual(loaded.models["LinearRegression"].artifacts, {"weights": "weights.bin"})
        self.assertEqual(str(loaded.env.conda_env_rel_path), "env/conda.yml")
        self.assertEqual(str(loaded.env.pip_requirements_rel_path), "env/requirements.txt")
        self.assertEqual(loaded.env.python_version, "3.10")
        self.assertIsNone(loaded._runtimes)
        self.assertEqual(set(loaded.runtimes.keys()), {"cpu"})
        self.assertEqual(loaded.method_options["predict"], {"case_sensitive": True, "max_batch_size": 64})
        self.assertEqual(loaded.function_type("predict"), "FUNCTION")
        self.assertTrue(loaded.case_sensitive)
        self.assertEqual(loaded.function_properties, {})

    def test_gpu_runtime_and_partition_properties(self) -> None:
        loaded = _load_spec(_gpu_env_v2_spec(), gpu=True)

        self.assertEqual(loaded.model_type, "xgboost")
        self.assertEqual(loaded.name, "XgboostBooster")
        self.assertIsNone(loaded._runtimes)
        self.assertEqual(set(loaded.runtimes.keys()), {"cpu", "gpu"})
        self.assertFalse(loaded.runtimes["cpu"].is_gpu)
        self.assertTrue(loaded.runtimes["gpu"].is_gpu)
        self.assertEqual(loaded.env.cuda_version, "11.8")
        self.assertEqual(loaded.runtimes["gpu"].runtime_env.cuda_version, "11.8")
        self.assertEqual(
            loaded.function_properties["predict_table"],
            {model_meta_schema.FunctionProperties.PARTITIONED.value: True},
        )
        self.assertEqual(loaded.method_options["predict"], {"case_sensitive": True, "max_batch_size": 64})
        self.assertEqual(loaded.method_options["predict_table"], {"case_sensitive": False})
        self.assertIsNone(loaded.case_sensitive)
        self.assertEqual(loaded.function_type("predict"), "FUNCTION")
        self.assertEqual(loaded.function_type("predict_table"), "TABLE_FUNCTION")

    def test_synthesizes_default_env_paths_when_omitted(self) -> None:
        spec = _single_env_v2_spec()
        del spec["serving"]["env"]["python_env"]["conda"]
        del spec["serving"]["env"]["python_env"]["pip"]

        loaded = _load_spec(spec)

        self.assertEqual(str(loaded.env.conda_env_rel_path), "env/conda.yml")
        self.assertEqual(str(loaded.env.pip_requirements_rel_path), "env/requirements.txt")

    def test_client_env_uses_packaged_paths_when_python_env_conda_differs(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["env"]["python_env"]["conda"] = "runtimes/python_runtime/env/conda.yml"
        del spec["serving"]["env"]["python_env"]["pip"]

        loaded = _load_spec(spec)

        assert loaded._packaged_env_dict is not None
        self.assertEqual(loaded._packaged_env_dict["conda"], "env/conda.yml")
        self.assertEqual(loaded._packaged_env_dict["pip"], "env/requirements.txt")
        self.assertIsNone(loaded._runtimes)

    def test_cpu_env_becomes_cpu_runtime(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["env"]["cpu_env"] = {
            "type": "python",
            "python_version": "3.10",
            "snowpark_ml_version": "1.6.0",
            "conda": "model/runtimes/cpu/env/conda.yml",
            "pip": "model/runtimes/cpu/env/requirements.txt",
            "imports": ["model/"],
        }

        loaded = _load_spec(spec, cpu=True)

        self.assertEqual(str(loaded.env.conda_env_rel_path), "env/conda.yml")
        self.assertIsNotNone(loaded._runtimes)
        self.assertEqual(set(loaded.runtimes.keys()), {"cpu"})
        self.assertEqual(loaded.runtimes["cpu"].imports, ["model/"])

    def test_cpu_and_gpu_envs_load_file_runtimes(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["env"]["cpu_env"] = {
            "type": "python",
            "python_version": "3.10",
            "snowpark_ml_version": "1.6.0",
            "conda": "model/runtimes/cpu/env/conda.yml",
            "pip": "model/runtimes/cpu/env/requirements.txt",
            "imports": ["model/"],
        }
        spec["serving"]["env"]["gpu_env"] = {
            "type": "python",
            "python_version": "3.10",
            "snowpark_ml_version": "1.6.0",
            "cuda_version": "11.8",
            "conda": "model/runtimes/gpu/env/conda.yml",
            "pip": "model/runtimes/gpu/env/requirements.txt",
            "imports": ["gpu/"],
        }

        loaded = _load_spec(spec, cpu=True, gpu=True)

        self.assertIsNotNone(loaded._runtimes)
        self.assertEqual(set(loaded.runtimes.keys()), {"cpu", "gpu"})
        self.assertEqual(loaded.runtimes["cpu"].imports, ["model/"])
        self.assertEqual(loaded.runtimes["gpu"].imports, ["gpu/"])
        self.assertFalse(loaded.runtimes["cpu"].is_gpu)
        self.assertTrue(loaded.runtimes["gpu"].is_gpu)
        self.assertEqual(str(loaded.env.conda_env_rel_path), "env/conda.yml")
        self.assertEqual(loaded.env.cuda_version, "11.8")

    def test_gpu_env_without_cuda_is_not_a_runtime(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["env"]["gpu_env"] = {
            "type": "python",
            "python_version": "3.10",
            "snowpark_ml_version": "1.6.0",
            "conda": "model/runtimes/gpu/env/conda.yml",
            "pip": "model/runtimes/gpu/env/requirements.txt",
        }

        loaded = _load_spec(spec)

        self.assertIsNone(loaded._runtimes)

    def test_framework_falls_back_to_blob_model_type(self) -> None:
        spec = _single_env_v2_spec()
        del spec["model"]["framework"]

        loaded = _load_spec(spec)

        self.assertEqual(loaded.model_type, "sklearn")

    def test_string_artifact_paths_preserved(self) -> None:
        spec = _single_env_v2_spec()
        spec["model"]["details"]["models"]["LinearRegression"]["artifacts"] = {"data": "artifacts/data.bin"}

        loaded = _load_spec(spec)

        self.assertEqual(loaded.models["LinearRegression"].artifacts, {"data": "artifacts/data.bin"})

    def test_explainability_promoted_to_top_level(self) -> None:
        spec = _single_env_v2_spec()
        spec["model"]["details"]["models"]["LinearRegression"]["explainability"] = {"algorithm": "shap"}

        loaded = _load_spec(spec)

        self.assertEqual(loaded.explain_algorithm, model_meta_schema.ModelExplainAlgorithm.SHAP)

    def test_non_string_user_data_ignored(self) -> None:
        spec = _single_env_v2_spec()
        spec["user_data"] = {"owner": "ml-platform", "count": 1}

        loaded = _load_spec(spec)

        self.assertIsNone(loaded.metadata)

    def test_empty_serving_env_synthesizes_defaults(self) -> None:
        spec = _single_env_v2_spec()
        del spec["serving"]["env"]

        loaded = _load_spec(spec)

        self.assertEqual(str(loaded.env.conda_env_rel_path), "env/conda.yml")
        self.assertEqual(str(loaded.env.pip_requirements_rel_path), "env/requirements.txt")
        self.assertIsNone(loaded._runtimes)

    def test_min_version_falls_back_when_client_omitted(self) -> None:
        spec = _single_env_v2_spec()
        del spec["model"]["client"]

        loaded = _load_spec(spec)

        self.assertEqual(loaded.min_snowpark_ml_version, model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION)

    def test_rejects_empty_models(self) -> None:
        spec = _single_env_v2_spec()
        spec["model"]["details"]["models"] = {}

        with self.assertRaisesRegex(ValueError, "model blobs must not be empty"):
            _load_spec(spec)

    def test_rejects_blob_missing_path(self) -> None:
        spec = _single_env_v2_spec()
        del spec["model"]["details"]["models"]["LinearRegression"]["path"]

        with self.assertRaisesRegex(ValueError, "missing path"):
            _load_spec(spec)

    def test_rejects_blob_missing_model_type(self) -> None:
        spec = _single_env_v2_spec()
        del spec["model"]["framework"]
        del spec["model"]["details"]["models"]["LinearRegression"]["model_type"]

        with self.assertRaisesRegex(ValueError, "model type is missing"):
            _load_spec(spec)

    def test_rejects_artifact_mapping_without_path(self) -> None:
        spec = _single_env_v2_spec()
        spec["model"]["details"]["models"]["LinearRegression"]["artifacts"] = {"weights": {"type": "file"}}

        with self.assertRaisesRegex(ValueError, "artifact 'weights' is missing path"):
            _load_spec(spec)

    def test_function_volatility_is_stored_on_method_options(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["functions"]["predict"]["volatility"] = "IMMUTABLE"

        loaded = _load_spec(spec)

        self.assertEqual(loaded.method_options["predict"]["volatility"], "IMMUTABLE")

    def test_rejects_function_missing_type(self) -> None:
        spec = _single_env_v2_spec()
        del spec["serving"]["functions"]["predict"]["type"]

        with self.assertRaisesRegex(ValueError, "function 'predict' type must be FUNCTION or TABLE_FUNCTION"):
            _load_spec(spec)

    def test_unknown_function_name_raises(self) -> None:
        loaded = _load_spec(_single_env_v2_spec())

        with self.assertRaises(KeyError):
            loaded.function_type("missing")

    def test_sample_input_file_path_is_preserved(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["functions"]["predict"]["signature"]["sample_input_file_path"] = "samples/input.parquet"

        loaded = _load_spec(spec)

        self.assertEqual(loaded.sample_input_file_paths["predict"], "samples/input.parquet")

    def test_original_metadata_version_comes_from_the_spec(self) -> None:
        spec = _single_env_v2_spec()
        spec["version"] = 2.0

        loaded = _load_spec(spec)

        self.assertEqual(loaded.original_metadata_version, "2.0")

    def test_rejects_min_version_newer_than_the_library(self) -> None:
        spec = _single_env_v2_spec()
        spec["model"]["client"]["min_version"] = "99.0.0"

        with self.assertRaisesRegex(RuntimeError, "The minimal version required to load the model is 99.0.0"):
            _load_spec(spec)

    def test_rejects_non_list_imports(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["env"]["cpu_env"] = {
            "type": "python",
            "python_version": "3.10",
            "conda": "model/runtimes/cpu/env/conda.yml",
            "imports": "model/",
        }

        with self.assertRaisesRegex(ValueError, "model specification: serving environment imports must be a list."):
            _load_spec(spec)

    def test_rejects_non_mapping_environment(self) -> None:
        spec = _single_env_v2_spec()
        spec["serving"]["env"]["cpu_env"] = "model/runtimes/cpu/env/conda.yml"

        with self.assertRaisesRegex(
            ValueError, "model specification: serving environment 'cpu_env' must be a mapping."
        ):
            _load_spec(spec)

    def test_to_model_dir_relpath_helpers(self) -> None:
        self.assertEqual(model_meta_v2._to_model_dir_relpath("model/env/conda.yml"), "env/conda.yml")
        self.assertEqual(model_meta_v2._to_model_dir_relpath("env/conda.yml"), "env/conda.yml")
        self.assertEqual(model_meta_v2._to_model_dir_relpath("./model/env/conda.yml"), "env/conda.yml")


class ModelMetadataLoadV2Test(absltest.TestCase):
    def test_load_v2_single_env_model_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_env_files(tmpdir)
            with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
                yaml.safe_dump(_single_env_v2_spec(), f)

            loaded = model_meta.ModelMetadata.load(tmpdir)

            self.assertEqual(loaded.original_metadata_version, "2.0")
            self.assertEqual(loaded.name, "LinearRegression")
            self.assertEqual(loaded.model_type, "sklearn")
            self.assertEqual(loaded.task, type_hints.Task.TABULAR_REGRESSION)
            self.assertIn("predict", loaded.signatures)
            self.assertEqual(str(loaded.env.conda_env_rel_path), "env/conda.yml")
            self.assertEqual(str(loaded.env.pip_requirements_rel_path), "env/requirements.txt")
            self.assertTrue(loaded.case_sensitive)
            self.assertEqual(loaded.method_options["predict"]["max_batch_size"], 64)
            self.assertEqual(loaded.function_type("predict"), "FUNCTION")
            # Single-env specs leave _runtimes unset; the property synthesizes a cpu runtime.
            self.assertIsNone(loaded._runtimes)
            self.assertEqual(set(loaded.runtimes.keys()), {"cpu"})

    def test_load_v2_gpu_and_partitioned_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_env_files(tmpdir, gpu=True)
            with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
                yaml.safe_dump(_gpu_env_v2_spec(), f)

            loaded = model_meta.ModelMetadata.load(tmpdir)

            self.assertEqual(loaded.original_metadata_version, "2.0")
            self.assertEqual(loaded.model_type, "xgboost")
            self.assertIsNone(loaded._runtimes)
            self.assertEqual(set(loaded.runtimes.keys()), {"cpu", "gpu"})
            self.assertTrue(loaded.runtimes["gpu"].is_gpu)
            self.assertEqual(
                loaded.function_properties["predict_table"],
                {model_meta_schema.FunctionProperties.PARTITIONED.value: True},
            )
            self.assertIsNone(loaded.case_sensitive)
            self.assertEqual(loaded.function_type("predict"), "FUNCTION")
            self.assertEqual(loaded.function_type("predict_table"), "TABLE_FUNCTION")

    def test_v2_save_is_rejected(self) -> None:
        loaded = _load_spec(_single_env_v2_spec())
        with self.assertRaisesRegex(RuntimeError, "Saving a version 2.0 model specification is not supported."):
            loaded.save("/tmp/unused")

    def test_load_legacy_path_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
                )

            loaded = model_meta.ModelMetadata.load(tmpdir)
            self.assertEqual(loaded.original_metadata_version, model_meta_schema.MODEL_METADATA_VERSION)
            self.assertEqual(loaded.name, "model1")
            self.assertEqual(loaded.model_type, "custom")


if __name__ == "__main__":
    absltest.main()
