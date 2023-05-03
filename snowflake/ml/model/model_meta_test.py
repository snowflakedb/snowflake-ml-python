from importlib import metadata as importlib_metadata

from absl.testing import absltest

from snowflake.ml.model import model_meta, model_signature

_DUMMY_SIG = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
    ],
    outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
)

_BASIC_DEPENDENCIES_TARGET = list(
    sorted(map(lambda x: f"defaults::{x}=={importlib_metadata.version(x)}", model_meta._BASIC_DEPENDENCIES))
)


class ModelMetaTest(absltest.TestCase):
    def test_model_meta_dependencies_no_packages(self) -> None:
        meta = model_meta.ModelMetadata(name="model1", model_type="custom", signature=_DUMMY_SIG)
        self.assertListEqual(meta.pip_requirements, [])
        self.assertListEqual(meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)

        meta_dict = meta.to_dict()

        laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, [])
        self.assertListEqual(laoded_meta.conda_dependencies, _BASIC_DEPENDENCIES_TARGET)

    def test_model_meta_dependencies_dup_basic_dep(self) -> None:
        meta = model_meta.ModelMetadata(
            name="model1", model_type="custom", signature=_DUMMY_SIG, conda_dependencies=["pandas"]
        )
        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.remove(f"defaults::pandas=={importlib_metadata.version('pandas')}")
        dep_target.append("defaults::pandas")
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, [])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, [])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_other_channel(self) -> None:
        with self.assertWarns(UserWarning):
            meta = model_meta.ModelMetadata(
                name="model1", model_type="custom", signature=_DUMMY_SIG, conda_dependencies=["conda-forge::pandas"]
            )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.remove(f"defaults::pandas=={importlib_metadata.version('pandas')}")
        dep_target.append("conda-forge::pandas")
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, [])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        with self.assertWarns(UserWarning):
            laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, [])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_dup_basic_dep_pip(self) -> None:
        with self.assertWarns(UserWarning):
            meta = model_meta.ModelMetadata(
                name="model1", model_type="custom", signature=_DUMMY_SIG, pip_requirements=["pandas"]
            )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, ["pandas"])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        with self.assertWarns(UserWarning):
            laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, ["pandas"])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_conda(self) -> None:
        meta = model_meta.ModelMetadata(
            name="model1", model_type="custom", signature=_DUMMY_SIG, conda_dependencies=["pytorch"]
        )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.append("defaults::pytorch")
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, [])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, [])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_pip(self) -> None:
        meta = model_meta.ModelMetadata(
            name="model1", model_type="custom", signature=_DUMMY_SIG, pip_requirements=["torch"]
        )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, ["torch"])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, ["torch"])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_dependencies_both(self) -> None:
        meta = model_meta.ModelMetadata(
            name="model1",
            model_type="custom",
            signature=_DUMMY_SIG,
            conda_dependencies=["pytorch"],
            pip_requirements=["torch"],
        )

        dep_target = _BASIC_DEPENDENCIES_TARGET[:]
        dep_target.append("defaults::pytorch")
        dep_target.sort()

        self.assertListEqual(meta.pip_requirements, ["torch"])
        self.assertListEqual(meta.conda_dependencies, dep_target)

        meta_dict = meta.to_dict()

        laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertListEqual(laoded_meta.pip_requirements, ["torch"])
        self.assertListEqual(laoded_meta.conda_dependencies, dep_target)

    def test_model_meta_override_py_version(self) -> None:
        meta = model_meta.ModelMetadata(name="model1", model_type="custom", signature=_DUMMY_SIG, python_version="2.7")

        self.assertEqual(meta.python_version, "2.7")

        meta_dict = meta.to_dict()

        laoded_meta = model_meta.ModelMetadata.from_dict(meta_dict)

        self.assertEqual(laoded_meta.python_version, "2.7")

        with self.assertRaises(ValueError):
            meta = model_meta.ModelMetadata(
                name="model1", model_type="custom", signature=_DUMMY_SIG, python_version="a"
            )


if __name__ == "__main__":
    absltest.main()
