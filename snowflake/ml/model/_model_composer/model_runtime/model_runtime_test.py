import os
import pathlib
import tempfile
from importlib import metadata as importlib_metadata
from unittest import mock

import yaml
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml.model import model_signature
from snowflake.ml.model._model_composer.model_runtime import model_runtime
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
            model_runtime._UDF_INFERENCE_DEPENDENCIES,
        )
    )
)

_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML = list(
    sorted(
        map(
            lambda x: str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x))),
            model_runtime._UDF_INFERENCE_DEPENDENCIES + [env_utils.SNOWPARK_ML_PKG_NAME],
        )
    )
)


class ModelRuntimeTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock.MagicMock()

    def test_model_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    returned_dict = mr.save(pathlib.Path(workspace))

                    self.assertDictEqual(
                        returned_dict,
                        {
                            "language": "PYTHON",
                            "version": meta.env.python_version,
                            "imports": ["model.zip"],
                            "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                        },
                    )
                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML, dependencies["dependencies"])

    def test_model_runtime_local_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=None):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    returned_dict = mr.save(pathlib.Path(workspace))

                    self.assertDictEqual(
                        returned_dict,
                        {
                            "language": "PYTHON",
                            "version": meta.env.python_version,
                            "imports": ["model.zip", "runtimes/python_runtime/snowflake-ml-python.zip"],
                            "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                        },
                    )
                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pandas"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"pandas=={importlib_metadata.version('pandas')}")
                dep_target.append("pandas")
                dep_target.sort()

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    _ = mr.save(pathlib.Path(workspace))

                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep_other_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["conda-forge::pandas"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"pandas=={importlib_metadata.version('pandas')}")
                dep_target.append("conda-forge::pandas")
                dep_target.sort()

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    _ = mr.save(pathlib.Path(workspace))

                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                pip_requirements=["pandas"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.remove(f"pandas=={importlib_metadata.version('pandas')}")
                dep_target.sort()

            with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                mr = model_runtime.ModelRuntime(
                    self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                )
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_conda_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.append("pytorch")
                dep_target.sort()

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    _ = mr.save(pathlib.Path(workspace))

                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_pip_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                pip_requirements=["torch"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.sort()

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    _ = mr.save(pathlib.Path(workspace))

                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_dep_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pytorch"],
                pip_requirements=["torch"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                dep_target = _BASIC_DEPENDENCIES_TARGET_WITH_SNOWML[:]
                dep_target.append("pytorch")
                dep_target.sort()

                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mr = model_runtime.ModelRuntime(
                        self.m_session, "python_runtime", meta, [pathlib.PurePosixPath("model.zip")]
                    )
                    _ = mr.save(pathlib.Path(workspace))

                    with open(os.path.join(workspace, "runtimes/python_runtime/env/conda.yml"), encoding="utf-8") as f:
                        dependencies = yaml.safe_load(f)

                    self.assertContainsSubset(dep_target, dependencies["dependencies"])


if __name__ == "__main__":
    absltest.main()
