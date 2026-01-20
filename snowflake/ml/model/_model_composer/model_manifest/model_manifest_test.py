import os
import pathlib
import tempfile

import importlib_resources
import yaml
from absl.testing import absltest, parameterized
from packaging import requirements

from snowflake.ml._internal import env_utils, platform_capabilities
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_meta_schema,
)
from snowflake.ml.model._packager.model_runtime import model_runtime
from snowflake.ml.model.volatility import Volatility

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


class ModelManifestTest(parameterized.TestCase):
    def test_model_manifest_1(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

            mm.save(meta, pathlib.PurePosixPath("model"), options={"relax_version": True})
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

    def test_model_manifest_2(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

    def test_model_manifest_mix(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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
                    },
                    relax_version=True,
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
                    options={"relax_version": True},
                )

    def test_model_manifest_table_function(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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
                    method_options={"predict": type_hints.ModelMethodSaveOptions(function_type="TABLE_FUNCTION")},
                    relax_version=True,
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
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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
                    method_options={"predict": type_hints.ModelMethodSaveOptions(function_type="TABLE_FUNCTION")},
                    relax_version=True,
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
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

            options: type_hints.ModelSaveOption = {"relax_version": False}
            mm.save(meta, pathlib.PurePosixPath("model"), options=options)

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

    def test_model_manifest_target_platforms(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options={"relax_version": True},
                target_platforms=[type_hints.TargetPlatform.WAREHOUSE],
            )
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

    def test_model_manifest_artifact_repo_map(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

            mm.save(meta, pathlib.PurePosixPath("model"), options={"relax_version": True})
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
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

            mm.save(meta, pathlib.PurePosixPath("model"), options={"relax_version": True})
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
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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
                options={"relax_version": True},
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

    @parameterized.parameters(  # type: ignore[misc]
        (Volatility.IMMUTABLE,),
        (Volatility.VOLATILE,),
    )
    def test_model_manifest_with_volatility(self, volatility: Volatility) -> None:
        """Test that ModelManifest.save() generates MANIFEST.yml with volatility field when specified."""
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True}
            ),
        ):
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

            # Test with explicit volatility options
            options = type_hints.BaseModelSaveOption(
                method_options={
                    "predict": type_hints.ModelMethodSaveOptions(volatility=volatility, function_type="FUNCTION")
                },
                relax_version=False,
            )

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=options,
            )

            # Verify MANIFEST.yml was created and contains volatility
            manifest_path = pathlib.Path(workspace, "MANIFEST.yml")
            self.assertTrue(manifest_path.exists())

            with open(manifest_path, encoding="utf-8") as f:
                manifest_content = yaml.safe_load(f)

            # Verify manifest structure
            self.assertIn("methods", manifest_content)
            self.assertEqual(len(manifest_content["methods"]), 1)

            method = manifest_content["methods"][0]
            self.assertIn("volatility", method)
            self.assertEqual(method["volatility"], volatility.name)
            self.assertEqual(method["name"], "PREDICT")
            self.assertEqual(method["type"], "FUNCTION")

    def test_model_manifest_with_no_volatility(self) -> None:
        """Test that ModelManifest.save() omits volatility when not specified and not set globally from reconciler."""
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True}
            ),
        ):
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

            options = type_hints.BaseModelSaveOption(
                relax_version=False,
            )

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=options,
            )

            # Load and verify the manifest
            with open(pathlib.Path(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                manifest_content = yaml.safe_load(f)

            method = manifest_content["methods"][0]
            self.assertNotIn("volatility", method)

    def test_model_manifest_with_global_volatility(self) -> None:
        """Test that ModelManifest.save() uses global volatility when not set at method level."""
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True}
            ),
        ):
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

            options = type_hints.BaseModelSaveOption(
                relax_version=False,
                volatility=Volatility.VOLATILE,
            )

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=options,
            )

            # Load and verify the manifest
            with open(pathlib.Path(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                manifest_content = yaml.safe_load(f)

            method = manifest_content["methods"][0]
            self.assertEqual(method.get("volatility"), Volatility.VOLATILE.name)

    def test_model_manifest_with_global_volatility_and_method_volatility(self) -> None:
        """Test that ModelManifest.save() uses method volatility over global volatility when not set at method level."""
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True}
            ),
        ):
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

            options = type_hints.BaseModelSaveOption(
                relax_version=False,
                volatility=Volatility.VOLATILE,
                method_options={
                    "predict": {
                        "volatility": Volatility.IMMUTABLE,
                    }
                },
            )

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=options,
            )

            # Load and verify the manifest
            with open(pathlib.Path(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                manifest_content = yaml.safe_load(f)

            method = manifest_content["methods"][0]
            self.assertEqual(method.get("volatility"), Volatility.IMMUTABLE.name)

    def test_model_manifest_without_set_volatility_from_manifest_parameter(self) -> None:
        """Test that ModelManifest.save() omits volatility when parameter in gs doesn't exist or not enabled"""
        for mock_feature in (
            {"dummy_feature": True},
            {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: False},
        ):
            with (
                tempfile.TemporaryDirectory() as workspace,
                tempfile.TemporaryDirectory() as tmpdir,
                platform_capabilities.PlatformCapabilities.mock_features(mock_feature),
            ):
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

                # Test with explicit volatility options
                options = type_hints.BaseModelSaveOption(
                    method_options={
                        "predict": type_hints.ModelMethodSaveOptions(
                            volatility=Volatility.IMMUTABLE, function_type="FUNCTION"
                        )
                    },
                    relax_version=False,
                )

                mm.save(
                    meta,
                    pathlib.PurePosixPath("model"),
                    options=options,
                )

                # Load and verify the manifest
                with open(pathlib.Path(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    manifest_content = yaml.safe_load(f)

                method = manifest_content["methods"][0]
                self.assertNotIn("volatility", method)

    def test_model_manifest_with_parameters(self) -> None:
        """Test that ModelManifest.save() generates MANIFEST.yml with parameters when capability is enabled."""
        sig_with_params = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT),
                ],
                params=[
                    model_signature.ParamSpec(
                        name="threshold", dtype=model_signature.DataType.FLOAT, default_value=0.5
                    ),
                    model_signature.ParamSpec(
                        name="max_iterations", dtype=model_signature.DataType.INT64, default_value=100
                    ),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=sig_with_params,
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            options = type_hints.BaseModelSaveOption(relax_version=False)

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=options,
            )

            # Load and verify the manifest
            with open(pathlib.Path(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                manifest_content = yaml.safe_load(f)

            self.assertEqual(len(manifest_content["methods"]), 1)
            method = manifest_content["methods"][0]
            self.assertIn("params", method)
            self.assertEqual(len(method["params"]), 2)
            self.assertEqual(method["params"][0]["name"], "THRESHOLD")
            self.assertEqual(method["params"][0]["type"], "FLOAT")
            self.assertEqual(method["params"][0]["default"], "0.5")
            self.assertEqual(method["params"][1]["name"], "MAX_ITERATIONS")
            self.assertEqual(method["params"][1]["type"], "BIGINT")
            self.assertEqual(method["params"][1]["default"], "100")

    def test_model_manifest_with_parameter_default_none(self) -> None:
        """Test that ModelManifest.save() handles parameters with default_value=None.

        When default_value is None, it should be written as "NULL" string in the MANIFEST
        so that Snowflake's SQL parser can interpret it as SQL NULL.
        """
        sig_with_none_default = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT),
                ],
                params=[
                    model_signature.ParamSpec(
                        name="optional_param", dtype=model_signature.DataType.STRING, default_value=None
                    ),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=sig_with_none_default,
                python_version="3.8",
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            options = type_hints.BaseModelSaveOption(relax_version=False)

            mm.save(
                meta,
                pathlib.PurePosixPath("model"),
                options=options,
            )

            # Load and verify the manifest
            with open(pathlib.Path(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                manifest_content = yaml.safe_load(f)

            method = manifest_content["methods"][0]
            self.assertIn("params", method)
            self.assertEqual(len(method["params"]), 1)
            self.assertEqual(method["params"][0]["name"], "OPTIONAL_PARAM")
            # None default_value is converted to "NULL" string for SQL parser compatibility
            self.assertEqual(method["params"][0]["default"], "NULL")


if __name__ == "__main__":
    absltest.main()
