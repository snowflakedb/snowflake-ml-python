import copy
import os
import pathlib
import tempfile
from unittest import mock

import yaml
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_runtime import model_runtime

_BASIC_DEPENDENCIES_TARGET_RELAXED = list(
    sorted(
        map(
            lambda x: str(env_utils.relax_requirement_version(requirements.Requirement(x))),
            model_runtime._SNOWML_INFERENCE_ALTERNATIVE_DEPENDENCIES,
        )
    )
)

_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED = [
    str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(env_utils.SNOWPARK_ML_PKG_NAME)))
]


class ModelRuntimeTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_env = model_env.ModelEnv()
        self.mock_to_use_released_snowml = mock.patch.object(
            env_utils,
            "get_matched_package_versions_in_information_schema_with_active_session",
            return_value={env_utils.SNOWPARK_ML_PKG_NAME: [""]},
        )
        self.mock_to_use_local_snowml = mock.patch.object(
            env_utils,
            "get_matched_package_versions_in_information_schema_with_active_session",
            return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
        )

    def test_model_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with self.mock_to_use_released_snowml:
                mr = model_runtime.ModelRuntime("cpu", self.m_env, [])
                returned_dict = mr.save(pathlib.Path(workspace))

                self.assertDictEqual(
                    returned_dict,
                    {
                        "imports": [],
                        "dependencies": {
                            "conda": "runtimes/cpu/env/conda.yml",
                            "pip": "runtimes/cpu/env/requirements.txt",
                        },
                    },
                )
                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED, dependencies["dependencies"])

    def test_model_runtime_with_import(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:

            with self.mock_to_use_released_snowml:
                mr = model_runtime.ModelRuntime("cpu", self.m_env, [pathlib.PurePosixPath("model.zip")])
                returned_dict = mr.save(pathlib.Path(workspace))

                self.assertDictEqual(
                    returned_dict,
                    {
                        "imports": ["model.zip"],
                        "dependencies": {
                            "conda": "runtimes/cpu/env/conda.yml",
                            "pip": "runtimes/cpu/env/requirements.txt",
                        },
                    },
                )
                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED, dependencies["dependencies"])

    def test_model_runtime_local_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with self.mock_to_use_local_snowml:
                mr = model_runtime.ModelRuntime(
                    "cpu",
                    self.m_env,
                )
                returned_dict = mr.save(pathlib.Path(workspace))

                self.assertDictEqual(
                    returned_dict,
                    {
                        "imports": ["runtimes/cpu/snowflake-ml-python.zip"],
                        "dependencies": {
                            "conda": "runtimes/cpu/env/conda.yml",
                            "pip": "runtimes/cpu/env/requirements.txt",
                        },
                    },
                )
                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_RELAXED, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.remove(next(filter(lambda x: x.startswith("packaging"), dep_target)))
            dep_target.append("packaging")
            dep_target.sort()

            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = dep_target

            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ):
                mr = model_runtime.ModelRuntime("cpu", m_env)
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep_other_channel(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:

            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.remove(next(filter(lambda x: x.startswith("packaging"), dep_target)))
            dep_target.append("conda-forge::packaging")
            dep_target.sort()
            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = dep_target

            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ):
                mr = model_runtime.ModelRuntime("cpu", m_env)
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep_pip(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.remove(next(filter(lambda x: x.startswith("packaging"), dep_target)))
            dep_target.sort()
            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = dep_target

            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ):
                mr = model_runtime.ModelRuntime("cpu", m_env)
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_conda_dep(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.append("pytorch")
            dep_target.sort()
            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = dep_target

            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ):
                mr = model_runtime.ModelRuntime("cpu", m_env)
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_pip_dep(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.sort()
            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = dep_target

            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ):
                mr = model_runtime.ModelRuntime("cpu", m_env)
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_dep_both(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.append("pytorch")
            dep_target.sort()
            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = dep_target

            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ):
                mr = model_runtime.ModelRuntime("cpu", m_env)
                _ = mr.save(pathlib.Path(workspace))

                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(dep_target, dependencies["dependencies"])

    def test_model_runtime_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = copy.deepcopy(self.m_env)
            m_env.conda_dependencies = ["pytorch"]
            m_env.cuda_version = "11.7"
            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema_with_active_session",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: [""]},
            ):
                mr = model_runtime.ModelRuntime("gpu", m_env, is_gpu=True)
                returned_dict = mr.save(pathlib.Path(workspace))

                self.assertDictEqual(
                    returned_dict,
                    {
                        "imports": [],
                        "dependencies": {
                            "conda": "runtimes/gpu/env/conda.yml",
                            "pip": "runtimes/gpu/env/requirements.txt",
                        },
                    },
                )
                with open(os.path.join(workspace, "runtimes/gpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(
                    ["nvidia::cuda==11.7.*", "pytorch::pytorch", "pytorch::pytorch-cuda==11.7.*"],
                    dependencies["dependencies"],
                )

    def test_model_runtime_check_conda(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_snowflake_conda_channel",
                return_value=[""],
            ):
                mr = model_runtime.ModelRuntime(
                    "cpu",
                    self.m_env,
                    server_availability_source="conda",
                )
                returned_dict = mr.save(pathlib.Path(workspace))

                self.assertDictEqual(
                    returned_dict,
                    {
                        "imports": [],
                        "dependencies": {
                            "conda": "runtimes/cpu/env/conda.yml",
                            "pip": "runtimes/cpu/env/requirements.txt",
                        },
                    },
                )
                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_WITH_SNOWML_RELAXED, dependencies["dependencies"])

    def test_model_runtime_local_snowml_check_conda(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with self.mock_to_use_local_snowml:
                mr = model_runtime.ModelRuntime("cpu", self.m_env, server_availability_source="conda")
                returned_dict = mr.save(pathlib.Path(workspace))

                self.assertDictEqual(
                    returned_dict,
                    {
                        "imports": ["runtimes/cpu/snowflake-ml-python.zip"],
                        "dependencies": {
                            "conda": "runtimes/cpu/env/conda.yml",
                            "pip": "runtimes/cpu/env/requirements.txt",
                        },
                    },
                )
                with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                    dependencies = yaml.safe_load(f)

                self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_RELAXED, dependencies["dependencies"])

    def test_model_runtime_load_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            mr = model_runtime.ModelRuntime("cpu", self.m_env, [pathlib.PurePosixPath("model.zip")])
            returned_dict = mr.save(pathlib.Path(workspace))

            loaded_mr = model_runtime.ModelRuntime.load(pathlib.Path(workspace), "cpu", self.m_env, returned_dict)

            self.assertDictEqual(loaded_mr.save(pathlib.Path(workspace)), returned_dict)


if __name__ == "__main__":
    absltest.main()
