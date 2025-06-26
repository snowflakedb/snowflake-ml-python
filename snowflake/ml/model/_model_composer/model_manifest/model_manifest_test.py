import os
import pathlib
import tempfile

import importlib_resources
import yaml
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_meta_schema,
)
from snowflake.ml.model._packager.model_runtime import model_runtime

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_1"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_2", shape=(-1,)),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_3", shape=(-1,)),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_4", shape=(-1,)),
        ],
        outputs=[
            model_signature.FeatureSpec(name="output_1", dtype=model_signature.DataType.FLOAT),
            model_signature.FeatureSpec(name="output_2", dtype=model_signature.DataType.FLOAT, shape=(2, 2)),
            model_signature.FeatureSpec(name="output_3", dtype=model_signature.DataType.FLOAT, shape=(2, 2)),
            model_signature.FeatureSpec(name="output_4", dtype=model_signature.DataType.FLOAT, shape=(2, 2)),
        ],
    )
}


_DUMMY_BLOB = model_blob_meta.ModelBlobMeta(
    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
)

_PACKAGING_REQUIREMENTS_TARGET_WITHOUT_SNOWML = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            model_meta._PACKAGING_REQUIREMENTS,
        )
    )
) + list(
    sorted(
        filter(
            lambda x: not any(
                dep in x for dep in model_runtime.PACKAGES_NOT_ALLOWED_IN_WAREHOUSE + model_meta._PACKAGING_REQUIREMENTS
            ),
            model_runtime._SNOWML_INFERENCE_ALTERNATIVE_DEPENDENCIES,
        ),
    )
)

_PACKAGING_REQUIREMENTS_TARGET_WITHOUT_SNOWML_RELAXED = list(
    sorted(
        map(
            lambda x: str(
                env_utils.relax_requirement_version(
                    env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))
                )
            ),
            model_meta._PACKAGING_REQUIREMENTS,
        )
    )
) + list(
    sorted(
        filter(
            lambda x: not any(
                dep in x for dep in model_runtime.PACKAGES_NOT_ALLOWED_IN_WAREHOUSE + model_meta._PACKAGING_REQUIREMENTS
            ),
            model_runtime._SNOWML_INFERENCE_ALTERNATIVE_DEPENDENCIES,
        ),
    )
)

_PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            model_meta._PACKAGING_REQUIREMENTS + [env_utils.SNOWPARK_ML_PKG_NAME],
        )
    )
)


_PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED = list(
    sorted(
        map(
            lambda x: str(
                env_utils.relax_requirement_version(
                    env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))
                )
            ),
            model_meta._PACKAGING_REQUIREMENTS + [env_utils.SNOWPARK_ML_PKG_NAME],
        )
    )
)


class ModelManifestTest(absltest.TestCase):
    def test_model_manifest_1(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                python_version="3.8",
                embed_local_ml_library=False,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            with self.assertWarnsRegex(UserWarning, "`relax_version` is not set and therefore defaulted to True."):
                mm.save(meta, pathlib.PurePosixPath("model"))
            with open(pathlib.Path(workspace, "runtimes", "python_runtime", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertDictEqual(
                    yaml.safe_load(f),
                    {
                        "channels": [env_utils.SNOWFLAKE_CONDA_CHANNEL_URL, "nodefaults"],
                        "dependencies": ["python==3.8.*"] + _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED,
                        "name": "snow-env",
                    },
                )
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_1.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_1_relax_version(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                python_version="3.8",
                embed_local_ml_library=False,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=type_hints.BaseModelSaveOption(
                    relax_version=False,
                ),
            )
            with open(pathlib.Path(workspace, "runtimes", "python_runtime", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertDictEqual(
                    yaml.safe_load(f),
                    {
                        "channels": [env_utils.SNOWFLAKE_CONDA_CHANNEL_URL, "nodefaults"],
                        "dependencies": ["python==3.8.*"] + _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML,
                        "name": "snow-env",
                    },
                )

    def test_model_manifest_2(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"__call__": _DUMMY_SIG["predict"]},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=type_hints.BaseModelSaveOption(
                    method_options={"__call__": type_hints.ModelMethodSaveOptions(max_batch_size=10)},
                    relax_version=False,
                ),
            )
            with open(pathlib.Path(workspace, "runtimes", "python_runtime", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertDictEqual(
                    yaml.safe_load(f),
                    {
                        "channels": [env_utils.SNOWFLAKE_CONDA_CHANNEL_URL, "nodefaults"],
                        "dependencies": ["python==3.8.*"] + _PACKAGING_REQUIREMENTS_TARGET_WITHOUT_SNOWML,
                        "name": "snow-env",
                    },
                )
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_2.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_2.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_2_relax_version(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"__call__": _DUMMY_SIG["predict"]},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=type_hints.BaseModelSaveOption(
                    method_options={"__call__": type_hints.ModelMethodSaveOptions(max_batch_size=10)},
                    relax_version=True,
                ),
            )
            with open(pathlib.Path(workspace, "runtimes", "python_runtime", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertDictEqual(
                    yaml.safe_load(f),
                    {
                        "channels": [env_utils.SNOWFLAKE_CONDA_CHANNEL_URL, "nodefaults"],
                        "dependencies": ["python==3.8.*"] + _PACKAGING_REQUIREMENTS_TARGET_WITHOUT_SNOWML_RELAXED,
                        "name": "snow-env",
                    },
                )

    def test_model_manifest_mix(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"], "__call__": _DUMMY_SIG["predict"]},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=type_hints.BaseModelSaveOption(
                    method_options={
                        "predict": type_hints.ModelMethodSaveOptions(case_sensitive=True),
                        "__call__": type_hints.ModelMethodSaveOptions(max_batch_size=10),
                    }
                ),
            )
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_3.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "runtimes", "python_runtime", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertDictEqual(
                    yaml.safe_load(f),
                    {
                        "channels": [env_utils.SNOWFLAKE_CONDA_CHANNEL_URL, "nodefaults"],
                        "dependencies": ["python==3.8.*"] + _PACKAGING_REQUIREMENTS_TARGET_WITHOUT_SNOWML_RELAXED,
                        "name": "snow-env",
                    },
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_2.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_bad(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"], "PREDICT": _DUMMY_SIG["predict"]},
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            with self.assertRaisesRegex(ValueError, "Found duplicate method named resolved as PREDICT in the model."):
                mm.save(
                    meta,
                    pathlib.PurePosixPath("model"),
                )

    def test_model_manifest_table_function(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=type_hints.BaseModelSaveOption(
                    method_options={"predict": type_hints.ModelMethodSaveOptions(function_type="TABLE_FUNCTION")}
                ),
            )
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_4.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_3.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_partitioned_function(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                function_properties={"predict": {model_meta_schema.FunctionProperties.PARTITIONED.value: True}},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=type_hints.BaseModelSaveOption(
                    method_options={"predict": type_hints.ModelMethodSaveOptions(function_type="TABLE_FUNCTION")}
                ),
            )
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_4.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_4.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_pip(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                pip_requirements=["xgboost==1.2.3"],
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            options: type_hints.ModelSaveOption = dict()
            mm.save(meta, pathlib.PurePosixPath("model"), options=options)
            self.assertFalse(options.get("relax_version", True))

            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_5.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_pip_relax_version(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                pip_requirements=["xgboost==1.2.3"],
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            with self.assertRaises(exceptions.SnowflakeMLException) as cm:
                mm.save(
                    meta,
                    pathlib.PurePosixPath("model"),
                    options=type_hints.BaseModelSaveOption(relax_version=True),
                )
            self.assertEqual(cm.exception.error_code, error_codes.INVALID_ARGUMENT)
            self.assertIn(
                "Setting `relax_version=True` is only allowed for models to be run in Warehouse with "
                "Snowflake Conda Channel dependencies",
                str(cm.exception),
            )

    def test_model_manifest_target_platforms(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                pip_requirements=["xgboost"],
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(meta, pathlib.PurePosixPath("model"), target_platforms=[type_hints.TargetPlatform.WAREHOUSE])
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_6.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_target_platforms_relax_version(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

                with self.assertRaises(exceptions.SnowflakeMLException) as cm:
                    mm.save(
                        meta,
                        pathlib.PurePosixPath("model"),
                        options=type_hints.BaseModelSaveOption(relax_version=True),
                        target_platforms=[type_hints.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
                    )
                self.assertEqual(cm.exception.error_code, error_codes.INVALID_ARGUMENT)
                self.assertIn(
                    "Setting `relax_version=True` is only allowed for models to be run in Warehouse with "
                    "Snowflake Conda Channel dependencies",
                    str(cm.exception),
                )

    def test_model_manifest_artifact_repo_map(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                pip_requirements=["xgboost"],
                artifact_repository_map={"pip": "db.sc.my_pypi_mirror"},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(meta, pathlib.PurePosixPath("model"))
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_8.yml")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_resource_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                pip_requirements=["xgboost"],
                resource_constraint={"architecture": "x86"},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm.save(meta, pathlib.PurePosixPath("model"))
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_9.yml")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_manifest_user_files(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"]},
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            with open(f"{tmpdir}/file1", "w") as f:
                path = os.path.abspath(f.name)
                f.write("user file")

            user_files = {
                "subdir1": [path],
                "subdir2/nested/dir": [path],
            }

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                user_files=user_files,
            )
            with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                        .joinpath("fixtures")
                        .joinpath("MANIFEST_7.yml")
                        .read_text()
                    ),
                    f.read(),
                )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump({}, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "Unable to get the version of the MANIFEST file."):
                mm.load()

        raw_input = {
            "manifest_version": "1.0",
            "runtimes": {
                "python_runtime": {
                    "language": "PYTHON",
                    "version": "3.8",
                    "imports": ["model", "runtimes/python_runtime/snowflake-ml-python.zip"],
                    "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                }
            },
            "methods": [
                {
                    "name": "predict",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "input", "type": "FLOAT"}],
                    "outputs": [{"type": "OBJECT"}],
                },
                {
                    "name": "__CALL__",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.__call__.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}],
                    "outputs": [{"type": "OBJECT"}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_input, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            self.assertDictEqual(raw_input, mm.load())

        raw_input["user_data"] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_input, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            self.assertDictEqual(raw_input, mm.load())

        raw_input["user_data"] = {"description": "Hello"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_input, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            self.assertDictEqual(raw_input, mm.load())


if __name__ == "__main__":
    absltest.main()
