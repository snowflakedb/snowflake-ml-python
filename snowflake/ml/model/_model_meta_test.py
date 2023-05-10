import os
import tempfile
from importlib import metadata as importlib_metadata

import yaml
from absl.testing import absltest

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
    sorted(map(lambda x: f"{x}=={importlib_metadata.version(x)}", _model_meta._BASIC_DEPENDENCIES))
)


class ModelMetaTest(absltest.TestCase):
    def test_model_meta_dependencies_no_packages(self) -> None:
        meta = _model_meta.ModelMetadata(name="model1", model_type="custom", signatures=_DUMMY_SIG)
        self.assertListEqual(meta.pip_requirements, [])
        self.assertListEqual(meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)

        meta_dict = meta.to_dict()

        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, [])
        self.assertListEqual(laoded_meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)

    def test_model_meta_dependencies_dup_basic_dep(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas"]
        )
        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
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
            meta = _model_meta.ModelMetadata(
                name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["conda-forge::pandas"]
            )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
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
            meta = _model_meta.ModelMetadata(
                name="model1", model_type="custom", signatures=_DUMMY_SIG, pip_requirements=["pandas"]
            )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, ["pandas"])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        with self.assertWarns(UserWarning):
            laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, ["pandas"])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pytorch"]
        )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.append("pytorch")
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, [])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, [])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_pip(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, pip_requirements=["torch"]
        )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, ["torch"])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, ["torch"])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_both(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1",
            model_type="custom",
            signatures=_DUMMY_SIG,
            conda_dependencies=["pytorch"],
            pip_requirements=["torch"],
        )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.append("pytorch")
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, ["torch"])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, ["torch"])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_override_py_version(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, python_version="2.7"
        )

        self.assertEqual(meta.python_version, "2.7")

        meta_dict = meta.to_dict()

        laoded_meta = _model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertEqual(laoded_meta.python_version, "2.7")

        with self.assertRaises(ValueError):
            meta = _model_meta.ModelMetadata(
                name="model1", model_type="custom", signatures=_DUMMY_SIG, python_version="a"
            )

    def test_model_meta_metadata(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, metadata={"foo": "bar"}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_model_metadata(tmpdir)

            loaded_meta = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

            self.assertEqual(meta.metadata, loaded_meta.metadata)

    def test_model_meta_check(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, metadata={"foo": "bar"}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_model_metadata(tmpdir)

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MODEL_METADATA_FILE)) as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["version"]

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MODEL_METADATA_FILE), "w") as f:
                yaml.safe_dump(meta_yaml_data, f)

            with self.assertRaises(NotImplementedError):
                _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

    def test_manifest_check(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, metadata={"foo": "bar"}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_model_metadata(tmpdir)

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE)) as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["version"]

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE), "w") as f:
                yaml.safe_dump(meta_yaml_data, f)

            with self.assertRaises(NotImplementedError):
                _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_model_metadata(tmpdir)

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE)) as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["language"]

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE), "w") as f:
                yaml.safe_dump(meta_yaml_data, f)

            with self.assertRaises(NotImplementedError):
                _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_model_metadata(tmpdir)

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE)) as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["kind"]

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE), "w") as f:
                yaml.safe_dump(meta_yaml_data, f)

            with self.assertRaises(NotImplementedError):
                _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_model_metadata(tmpdir)

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE)) as f:
                meta_yaml_data = yaml.safe_load(f)

            del meta_yaml_data["env"]

            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE), "w") as f:
                yaml.safe_dump(meta_yaml_data, f)

            _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, _model_meta.ModelMetadata.MANIFEST_FILE), "w") as f:
                yaml.safe_dump(["a"], f)

            with self.assertRaises(ValueError):
                _ = _model_meta.ModelMetadata.load_model_metadata(tmpdir)


if __name__ == "__main__":
    absltest.main()
