import os
import pathlib
import tempfile

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

_BASIC_DEPENDENCIES_TARGET_RELAXED_WAREHOUSE = list(
    filter(
        lambda x: not any(dep in x for dep in model_runtime.PACKAGES_NOT_ALLOWED_IN_WAREHOUSE),
        _BASIC_DEPENDENCIES_TARGET_RELAXED,
    )
)


class ModelRuntimeTest(absltest.TestCase):
    def _check_deps(self, dep_target: list[str], deps: list[str]) -> None:
        # Dependencies without specifiers will have specifiers added according to local version. Check these separately.
        dep_target_no_specifier = []
        for d in dep_target:
            channel, _, req_str = d.rpartition("::")
            if not requirements.Requirement(req_str).specifier:
                dep_target_no_specifier.append(d)

        self.assertTrue(all([any([d2.startswith(d1) for d2 in deps]) for d1 in dep_target_no_specifier]))

        dep_target = [d for d in dep_target if d not in dep_target_no_specifier]
        self.assertContainsSubset(dep_target, deps)

    def test_model_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"

            mr = model_runtime.ModelRuntime("cpu", m_env, [])
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": [],
                    "dependencies": {
                        "conda": "runtimes/cpu/env/conda.yml",
                        "pip": "runtimes/cpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(["snowflake-ml-python==1.0.0"], dependencies["dependencies"])

    def test_model_runtime_with_channel_override(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"

            mr = model_runtime.ModelRuntime("cpu", m_env, [])
            returned_dict = mr.save(pathlib.Path(workspace), default_channel_override="conda-forge")

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": [],
                    "dependencies": {
                        "conda": "runtimes/cpu/env/conda.yml",
                        "pip": "runtimes/cpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(["snowflake-ml-python==1.0.0"], dependencies["dependencies"])
            self.assertEqual(["conda-forge", "nodefaults"], dependencies["channels"])

    def test_model_runtime_with_import(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"

            mr = model_runtime.ModelRuntime("cpu", m_env, ["model.zip"])
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": ["model.zip"],
                    "dependencies": {
                        "conda": "runtimes/cpu/env/conda.yml",
                        "pip": "runtimes/cpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(["snowflake-ml-python==1.0.0"], dependencies["dependencies"])

    def test_model_runtime_with_dir_import(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"

            mr = model_runtime.ModelRuntime("cpu", m_env, ["model/"])
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": ["model/"],
                    "dependencies": {
                        "conda": "runtimes/cpu/env/conda.yml",
                        "pip": "runtimes/cpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(["snowflake-ml-python==1.0.0"], dependencies["dependencies"])

    def test_model_runtime_local_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0+abcdef"

            mr = model_runtime.ModelRuntime(
                "cpu",
                m_env,
            )
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": ["runtimes/cpu/snowflake-ml-python.zip"],
                    "dependencies": {
                        "conda": "runtimes/cpu/env/conda.yml",
                        "pip": "runtimes/cpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_RELAXED, dependencies["dependencies"])

    def test_model_runtime_local_snowml_warehouse(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0+abcdef"

            mr = model_runtime.ModelRuntime(
                "cpu",
                m_env,
                is_warehouse=True,
            )
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": ["runtimes/cpu/snowflake-ml-python.zip"],
                    "dependencies": {
                        "conda": "runtimes/cpu/env/conda.yml",
                        "pip": "runtimes/cpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(_BASIC_DEPENDENCIES_TARGET_RELAXED_WAREHOUSE, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.remove(next(filter(lambda x: x.startswith("packaging"), dep_target)))
            dep_target.append("packaging")
            dep_target.sort()

            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = dep_target

            mr = model_runtime.ModelRuntime("cpu", m_env)
            _ = mr.save(pathlib.Path(workspace))

            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self._check_deps(dep_target, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep_other_channel(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:

            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.remove(next(filter(lambda x: x.startswith("packaging"), dep_target)))
            dep_target.append("conda-forge::packaging")
            dep_target.sort()
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = dep_target

            mr = model_runtime.ModelRuntime("cpu", m_env)
            _ = mr.save(pathlib.Path(workspace))

            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self._check_deps(dep_target, dependencies["dependencies"])

    def test_model_runtime_dup_basic_dep_pip(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.remove(next(filter(lambda x: x.startswith("packaging"), dep_target)))
            dep_target.sort()
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = dep_target

            mr = model_runtime.ModelRuntime("cpu", m_env)
            _ = mr.save(pathlib.Path(workspace))

            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self._check_deps(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_conda_dep(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.append("pytorch")
            dep_target.sort()

            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = dep_target

            mr = model_runtime.ModelRuntime("cpu", m_env)
            _ = mr.save(pathlib.Path(workspace))

            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self._check_deps(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_pip_dep(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.sort()
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = dep_target

            mr = model_runtime.ModelRuntime("cpu", m_env)
            _ = mr.save(pathlib.Path(workspace))

            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self._check_deps(dep_target, dependencies["dependencies"])

    def test_model_runtime_additional_dep_both(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            dep_target = _BASIC_DEPENDENCIES_TARGET_RELAXED[:]
            dep_target.append("pytorch")
            dep_target.sort()
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = dep_target

            mr = model_runtime.ModelRuntime("cpu", m_env)
            _ = mr.save(pathlib.Path(workspace))

            with open(os.path.join(workspace, "runtimes/cpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self._check_deps(dep_target, dependencies["dependencies"])

    def test_model_runtime_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = ["pytorch"]
            m_env.cuda_version = "11.7"

            mr = model_runtime.ModelRuntime("gpu", m_env, is_gpu=True)
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": [],
                    "dependencies": {
                        "conda": "runtimes/gpu/env/conda.yml",
                        "pip": "runtimes/gpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )
            with open(os.path.join(workspace, "runtimes/gpu/env/conda.yml"), encoding="utf-8") as f:
                dependencies = yaml.safe_load(f)

            self.assertContainsSubset(
                ["python==3.9.*", "pytorch", "snowflake-ml-python==1.0.0", "nvidia::cuda==11.7.*"],
                dependencies["dependencies"],
            )

    def test_artifact_repository(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = ["pytorch"]
            m_env.cuda_version = "11.7"

            mr = model_runtime.ModelRuntime("gpu", m_env, is_gpu=True)
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": [],
                    "dependencies": {
                        "conda": "runtimes/gpu/env/conda.yml",
                        "pip": "runtimes/gpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {},
                },
            )

            m_env.artifact_repository_map = {
                "my_channel": "db.sch.my_repo",
            }

            mr = model_runtime.ModelRuntime("gpu", m_env, is_gpu=True)
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": [],
                    "dependencies": {
                        "conda": "runtimes/gpu/env/conda.yml",
                        "pip": "runtimes/gpu/env/requirements.txt",
                        "artifact_repository_map": {
                            "my_channel": "db.sch.my_repo",
                        },
                    },
                    "resource_constraint": {},
                },
            )

    def test_resource_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"
            m_env.conda_dependencies = ["pytorch"]

            m_env.resource_constraint = {"architecture": "x86"}

            mr = model_runtime.ModelRuntime("gpu", m_env, is_gpu=True)
            returned_dict = mr.save(pathlib.Path(workspace))

            self.assertDictEqual(
                returned_dict,
                {
                    "imports": [],
                    "dependencies": {
                        "conda": "runtimes/gpu/env/conda.yml",
                        "pip": "runtimes/gpu/env/requirements.txt",
                        "artifact_repository_map": {},
                    },
                    "resource_constraint": {"architecture": "x86"},
                },
            )

    def test_model_runtime_load_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()
            m_env.snowpark_ml_version = "1.0.0"

            mr = model_runtime.ModelRuntime("cpu", m_env, ["model.zip"])
            returned_dict = mr.save(pathlib.Path(workspace))

            loaded_mr = model_runtime.ModelRuntime.load(pathlib.Path(workspace), "cpu", m_env, returned_dict)

            self.assertDictEqual(loaded_mr.save(pathlib.Path(workspace)), returned_dict)

    def test_model_runtime_load_from_file_dir_import(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            m_env = model_env.ModelEnv()

            mr = model_runtime.ModelRuntime("cpu", m_env, ["model/"])
            returned_dict = mr.save(pathlib.Path(workspace))

            loaded_mr = model_runtime.ModelRuntime.load(pathlib.Path(workspace), "cpu", m_env, returned_dict)

            self.assertDictEqual(loaded_mr.save(pathlib.Path(workspace)), returned_dict)


if __name__ == "__main__":
    absltest.main()
