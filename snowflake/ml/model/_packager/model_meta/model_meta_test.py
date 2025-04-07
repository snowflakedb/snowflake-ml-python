import os
import tempfile
from importlib import metadata as importlib_metadata

import yaml
from absl.testing import absltest
from packaging import requirements, version

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_meta_schema,
)

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

_PACKAGING_REQUIREMENTS_TARGET = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
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


class ModelMetaEnvTest(absltest.TestCase):
    def test_model_meta_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            self.assertListEqual(meta.env.pip_requirements, [])
            self.assertListEqual(meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML)
            self.assertEqual(meta.env.snowpark_ml_version, snowml_env.VERSION)

    def test_model_meta_dependencies_no_relax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
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
            self.assertListEqual(meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, [])
            self.assertListEqual(loaded_meta.env.conda_dependencies, _PACKAGING_REQUIREMENTS_TARGET)
            self.assertIsNotNone(meta.env._snowpark_ml_version.local)

    def test_model_meta_dependencies_no_packages_embedded_snowml_strict(self) -> None:
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
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"cloudpickle=={importlib_metadata.version('cloudpickle')}")
                dep_target.sort()

                self.assertListEqual(meta.env.pip_requirements, ["cloudpickle"] + dep_target)
                self.assertListEqual(meta.env.conda_dependencies, [])

                with self.assertWarns(UserWarning):
                    loaded_meta = model_meta.ModelMetadata.load(tmpdir)

                self.assertListEqual(loaded_meta.env.pip_requirements, ["cloudpickle"] + dep_target)
                self.assertListEqual(loaded_meta.env.conda_dependencies, [])

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

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
            dep_target.append("pytorch==2.0.1")
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

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
            dep_target.append("pytorch==2.0.1")
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

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
            dep_target.sort()

            self.assertListEqual(meta.env.pip_requirements, dep_target + ["torch"])
            self.assertListEqual(meta.env.conda_dependencies, [])

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)

            self.assertListEqual(loaded_meta.env.pip_requirements, dep_target + ["torch"])
            self.assertListEqual(loaded_meta.env.conda_dependencies, [])

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

            dep_target = _PACKAGING_REQUIREMENTS_TARGET_WITH_SNOWML[:]
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

            self.assertEqual(meta.explain_algorithm, None)

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

    def test_model_meta_model_specified_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.task = type_hints.Task.TABULAR_REGRESSION

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)
            self.assertEqual(loaded_meta.task, type_hints.Task.TABULAR_REGRESSION)

    def test_model_meta_explain_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                meta.task = type_hints.Task.TABULAR_REGRESSION
                meta.explain_algorithm = model_meta_schema.ModelExplainAlgorithm.SHAP

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)
            self.assertEqual(loaded_meta.task, type_hints.Task.TABULAR_REGRESSION)
            self.assertEqual(loaded_meta.explain_algorithm, model_meta_schema.ModelExplainAlgorithm.SHAP)

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
            with open(os.path.join(tmpdir, "runtimes", "cpu", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertListEqual(yaml.safe_load(f)["channels"], ["conda-forge", "nodefaults"])

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
            with open(os.path.join(tmpdir, "runtimes", "cpu", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertListEqual(yaml.safe_load(f)["channels"], ["conda-forge", "nodefaults"])
            self.assertContainsSubset(
                ["pytorch"],
                meta.runtimes["gpu"].runtime_env.conda_dependencies,
            )
            with open(os.path.join(tmpdir, "runtimes", "gpu", "env", "conda.yml"), encoding="utf-8") as f:
                self.assertListEqual(yaml.safe_load(f)["channels"], ["conda-forge", "nodefaults"])

            loaded_meta = model_meta.ModelMetadata.load(tmpdir)
            self.assertContainsSubset(["pytorch"], loaded_meta.runtimes["cpu"].runtime_env.conda_dependencies)
            self.assertContainsSubset(
                ["pytorch"],
                loaded_meta.runtimes["gpu"].runtime_env.conda_dependencies,
            )

    def test_model_meta_prefer_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                target_platforms=[type_hints.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

        self.assertTrue(meta.env.prefer_pip)


if __name__ == "__main__":
    absltest.main()
