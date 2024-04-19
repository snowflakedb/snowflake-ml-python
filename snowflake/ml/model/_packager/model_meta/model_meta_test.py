import os
import tempfile
from importlib import metadata as importlib_metadata

import yaml
from absl.testing import absltest
from packaging import requirements, version

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import model_blob_meta, model_meta

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}

_DUMMY_BLOB = model_blob_meta.ModelBlobMeta(
    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
)

_BASIC_DEPENDENCIES_TARGET = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            model_meta._PACKAGING_CORE_DEPENDENCIES,
        )
    )
)

_BASIC_DEPENDENCIES_TARGET_RELAXED = list(
    sorted(
        map(
            lambda x: str(
                env_utils.relax_requirement_version(
                    env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))
                )
            ),
            model_meta._PACKAGING_CORE_DEPENDENCIES,
        )
    )
)

_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            model_meta._PACKAGING_CORE_DEPENDENCIES + [env_utils.SNOWPARK_ML_PKG_NAME],
        )
    )
)

_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED = list(
    sorted(
        map(
            lambda x: str(
                env_utils.relax_requirement_version(
                    env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))
                )
            ),
            model_meta._PACKAGING_CORE_DEPENDENCIES + [env_utils.SNOWPARK_ML_PKG_NAME],
        )
    )
)

_PACKAGING_REQUIREMENTS_TARGET = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            model_meta._PACKAGING_REQUIREMENTS,
        )
    )
)

_PACKAGING_REQUIREMENTS_TARGET_RELAXED = list(
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


class ModelMetaEnvLegacyTest(absltest.TestCase):
    def test_model_meta_dependencies_no_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG, _legacy_save=True
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

    def test_model_meta_dependencies_no_packages_embedded_snowml_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                embed_local_ml_library=True,
                _legacy_save=True,
                relax_version=False,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)

    def test_model_meta_dependencies_no_packages_embedded_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                embed_local_ml_library=True,
                _legacy_save=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _BASIC_DEPENDENCIES_TARGET_RELAXED)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _BASIC_DEPENDENCIES_TARGET_RELAXED)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

    def test_model_meta_dependencies_dup_basic_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["cloudpickle"],
                relax_version=False,
                _legacy_save=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
            dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
            dep_target.append("cloudpickle")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_other_channel(self) -> None:
        with self.assertWarns(UserWarning):
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    conda_dependencies=["conda-forge::cloudpickle"],
                    relax_version=False,
                    _legacy_save=True,
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
                dep_target.append("conda-forge::cloudpickle")
                dep_target.sort()

                self.assertListEqual(meta.env.pip_requirements, [])
                self.assertListEqual(meta.env.conda_dependencies, dep_target)

                with self.assertWarns(UserWarning):
                    loaded_meta = model_meta.ModelMetadata.load(tmpdir)

                self.assertListEqual(loaded_meta.env.pip_requirements, [])
                self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_pip(self) -> None:
        with self.assertWarns(UserWarning):
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    pip_requirements=["cloudpickle"],
                    relax_version=False,
                    _legacy_save=True,
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
                dep_target.sort()

                self.assertListEqual(meta.env.pip_requirements, ["cloudpickle"])
                self.assertListEqual(meta.env.conda_dependencies, dep_target)

                with self.assertWarns(UserWarning):
                    loaded_meta = model_meta.ModelMetadata.load(tmpdir)

                self.assertListEqual(loaded_meta.env.pip_requirements, ["cloudpickle"])
                self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch==2.0.1"],
                _legacy_save=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.append("pytorch<3,>=2.0")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda_additional_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                _legacy_save=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.env.include_if_absent([model_env.ModelDependency("pytorch==2.0.1", "torch")])

            dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.append("pytorch<3,>=2.0")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                pip_requirements=["torch"],
                _legacy_save=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, ["torch"])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, ["torch"])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch"],
                pip_requirements=["torch"],
                _legacy_save=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.append("pytorch")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, ["torch"])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, ["torch"])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)


class ModelMetaEnvTest(absltest.TestCase):
    def test_model_meta_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG, relax_version=True
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

            with self.assertWarnsRegex(UserWarning, "`relax_version` is not set and therefore defaulted to True."):
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

    def test_model_meta_dependencies_no_relax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG, relax_version=False
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

    def test_model_meta_dependencies_no_packages_embedded_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                embed_local_ml_library=True,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_RELAXED)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_RELAXED)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

    def test_model_meta_dependencies_no_packages_embedded_snowml_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                embed_local_ml_library=True,
                relax_version=False,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

    def test_model_meta_dependencies_dup_basic_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["cloudpickle"],
                relax_version=False,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
            dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
            dep_target.append("cloudpickle")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_other_channel(self) -> None:
        with self.assertWarns(UserWarning):
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    conda_dependencies=["conda-forge::cloudpickle"],
                    relax_version=False,
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
                dep_target.append("conda-forge::cloudpickle")
                dep_target.sort()

                self.assertListEqual(meta.env.pip_requirements, [])
                self.assertListEqual(meta.env.conda_dependencies, dep_target)

                with self.assertWarns(UserWarning):
                    loaded_meta = model_meta.ModelMetadata.load(tmpdir)

                self.assertListEqual(loaded_meta.env.pip_requirements, [])
                self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_pip(self) -> None:
        with self.assertWarns(UserWarning):
            with tempfile.TemporaryDirectory() as tmpdir:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    pip_requirements=["cloudpickle"],
                    relax_version=False,
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
                dep_target.sort()

                self.assertListEqual(meta.env.pip_requirements, ["cloudpickle"])
                self.assertListEqual(meta.env.conda_dependencies, dep_target)

                with self.assertWarns(UserWarning):
                    loaded_meta = model_meta.ModelMetadata.load(tmpdir)

                self.assertListEqual(loaded_meta.env.pip_requirements, ["cloudpickle"])
                self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch==2.0.1"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.append("pytorch<3,>=2.0")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda_additional_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.env.include_if_absent([model_env.ModelDependency("pytorch==2.0.1", "torch")])

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.append("pytorch<3,>=2.0")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                pip_requirements=["torch"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, ["torch"])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, ["torch"])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_dependencies_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch"],
                pip_requirements=["torch"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML_RELAXED[:]
            dep_target.append("pytorch")
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, ["torch"])
            self.assertListEqual(meta.env.conda_dependencies, dep_target)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, ["torch"])
            self.assertListEqual(loaded_meta.env.conda_dependencies, dep_target)

    def test_model_meta_override_py_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG, python_version="2.7"
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertEqual(meta.env.python_version, "2.7")

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertEqual(loaded_meta.env.python_version, "2.7")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(version.InvalidVersion):
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG, python_version="a"
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

    def test_model_meta_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            saved_meta = meta

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertEqual(saved_meta.metadata, loaded_meta.metadata)

    def test_model_meta_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), encoding="utf-8") as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["version"]

            with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
                yaml.safe_dump(meta_yaml_data, f)

            with self.assertRaisesRegex(ValueError, "Unable to get the version of the metadata file."):
                model_meta.ModelMetadata.load(tmpdir)

    def test_model_meta_new_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), encoding="utf-8") as f:
                meta_yaml_data = yaml.safe_load(f)

            meta_yaml_data["random_field"] = "foo"

            with open(os.path.join(tmpdir, model_meta.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
                yaml.safe_dump(meta_yaml_data, f)

            model_meta.ModelMetadata.load(tmpdir)

    def test_model_meta_check_min_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                current_version = version.parse(snowml_env.VERSION)

                meta.min_snowpark_ml_version = (
                    f"{current_version.major}.{current_version.minor}.{current_version.micro+1}"
                )

            with self.assertRaisesRegex(RuntimeError, "The minimal version required to load the model is"):
                model_meta.ModelMetadata.load(tmpdir)

    def test_model_meta_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.env.cuda_version = "11.7"

            self.assertTrue("gpu" in meta.runtimes)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertEqual(loaded_meta.env.cuda_version, "11.7")
            self.assertTrue("gpu" in loaded_meta.runtimes)

            with self.assertRaisesRegex(ValueError, "Different CUDA version .+ and .+ found in the same model!"):
                loaded_meta.env.cuda_version = "12.0"

    def test_model_meta_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.env.include_if_absent([model_env.ModelDependency(requirement="pytorch", pip_name="torch")])
                self.assertListEqual(meta.env.pip_requirements, [])
                self.assertContainsSubset(["pytorch"], meta.env.conda_dependencies)

            self.assertContainsSubset(["pytorch"], meta.runtimes["cpu"].runtime_env.conda_dependencies)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)
            self.assertContainsSubset(["pytorch"], loaded_meta.runtimes["cpu"].runtime_env.conda_dependencies)

    def test_model_meta_runtimes_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.env.include_if_absent([model_env.ModelDependency(requirement="pytorch", pip_name="torch")])
                meta.env.cuda_version = "11.7"
                self.assertListEqual(meta.env.pip_requirements, [])
                self.assertContainsSubset(["pytorch"], meta.env.conda_dependencies)

            self.assertContainsSubset(["pytorch"], meta.runtimes["cpu"].runtime_env.conda_dependencies)
            self.assertContainsSubset(
                ["nvidia::cuda==11.7.*", "pytorch::pytorch", "pytorch::pytorch-cuda==11.7.*"],
                meta.runtimes["gpu"].runtime_env.conda_dependencies,
            )

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)
            self.assertContainsSubset(["pytorch"], loaded_meta.runtimes["cpu"].runtime_env.conda_dependencies)
            self.assertContainsSubset(
                ["nvidia::cuda==11.7.*", "pytorch::pytorch", "pytorch::pytorch-cuda==11.7.*"],
                loaded_meta.runtimes["gpu"].runtime_env.conda_dependencies,
            )


if __name__ == "__main__":
    absltest.main()
