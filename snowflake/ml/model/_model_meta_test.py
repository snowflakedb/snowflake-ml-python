import os
import tempfile
from importlib import metadata as importlib_metadata

import yaml
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml.model import _model_meta, model_signature

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}

_BASIC_DEPENDENCIES_TARGET = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            _model_meta._BASIC_DEPENDENCIES,
        )
    )
)

_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            _model_meta._BASIC_DEPENDENCIES + [env_utils._SNOWML_PKG_NAME],
        )
    )
)


class ModelMetaTest(absltest.TestCase):
    def test_model_meta_dependencies_no_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                self.assertListEqual(meta.pip_requirements, [])
                self.assertListEqual(meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML)
                self.assertFalse(hasattr(meta, "local_ml_library_version"))

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertListEqual(laoded_meta.pip_requirements, [])
                self.assertListEqual(laoded_meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML)
                self.assertFalse(hasattr(meta, "local_ml_library_version"))

    def test_model_meta_dependencies_no_packages_embeded_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                embed_local_ml_library=True,
            ) as meta:
                self.assertListEqual(meta.pip_requirements, [])
                self.assertListEqual(meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)
                self.assertTrue(hasattr(meta, "local_ml_library_version"))

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertListEqual(laoded_meta.pip_requirements, [])
                self.assertListEqual(laoded_meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)
                self.assertTrue(hasattr(meta, "local_ml_library_version"))

    def test_model_meta_dependencies_dup_basic_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pandas"],
            ) as meta:
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"pandas=={importlib_metadata.version('pandas')}")
                dep_target.append("pandas")
                dep_target.sort()

                self.assertListEqual(meta.pip_requirements, [])
                self.assertListEqual(meta.conda_dependencies, dep_target)

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertListEqual(laoded_meta.pip_requirements, [])
                self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_other_channel(self) -> None:
        with self.assertWarns(UserWarning):
            with tempfile.TemporaryDirectory() as tmpdir:
                with _model_meta._create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    conda_dependencies=["conda-forge::pandas"],
                ) as meta:
                    dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                    dep_target.remove(f"pandas=={importlib_metadata.version('pandas')}")
                    dep_target.append("conda-forge::pandas")
                    dep_target.sort()

                    self.assertListEqual(meta.pip_requirements, [])
                    self.assertListEqual(meta.conda_dependencies, dep_target)

                    meta_dict = meta.to_dict()

                    with self.assertWarns(UserWarning):
                        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                    self.assertListEqual(laoded_meta.pip_requirements, [])
                    self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_pip(self) -> None:
        with self.assertWarns(UserWarning):
            with tempfile.TemporaryDirectory() as tmpdir:
                with _model_meta._create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    pip_requirements=["pandas"],
                ) as meta:
                    dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                    dep_target.sort()

                    self.assertListEqual(meta.pip_requirements, ["pandas"])
                    self.assertListEqual(meta.conda_dependencies, dep_target)

                    meta_dict = meta.to_dict()

                    with self.assertWarns(UserWarning):
                        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                    self.assertListEqual(laoded_meta.pip_requirements, ["pandas"])
                    self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch"],
            ) as meta:
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.append("pytorch")
                dep_target.sort()

                self.assertListEqual(meta.pip_requirements, [])
                self.assertListEqual(meta.conda_dependencies, dep_target)

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertListEqual(laoded_meta.pip_requirements, [])
                self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                pip_requirements=["torch"],
            ) as meta:
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.sort()

                self.assertListEqual(meta.pip_requirements, ["torch"])
                self.assertListEqual(meta.conda_dependencies, dep_target)

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertListEqual(laoded_meta.pip_requirements, ["torch"])
                self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch"],
                pip_requirements=["torch"],
            ) as meta:
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.append("pytorch")
                dep_target.sort()

                self.assertListEqual(meta.pip_requirements, ["torch"])
                self.assertListEqual(meta.conda_dependencies, dep_target)

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertListEqual(laoded_meta.pip_requirements, ["torch"])
                self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_override_py_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG, python_version="2.7"
            ) as meta:
                self.assertEqual(meta.python_version, "2.7")

                meta_dict = meta.to_dict()

                laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

                self.assertEqual(laoded_meta.python_version, "2.7")

            with self.assertRaises(ValueError):
                meta = _model_meta.ModelMetadata(
                    name="model1", model_type="custom", signatures=_DUMMY_SIG, python_version="a"
                )

    def test_model_meta_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ) as meta:
                saved_meta = meta
            loaded_meta = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

            self.assertEqual(saved_meta.metadata, loaded_meta.metadata)
            self.assertDictEqual(saved_meta.to_dict(), loaded_meta.to_dict())

    def test_model_meta_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _model_meta._create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                metadata={"foo": "bar"},
            ):
                pass
            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MODEL_METADATA_FILE), encoding="utf-8") as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["version"]

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MODEL_METADATA_FILE), "w", encoding="utf-8") as f:
                yaml.safe_dump(meta_yaml_data, f)

            with self.assertRaises(NotImplementedError):
                _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)


if __name__ == "__main__":
    absltest.main()
