import copy
import os
import pathlib
import tempfile
import warnings
from unittest import mock

import yaml
from absl.testing import absltest
from packaging import requirements, version

from snowflake.ml import version as snowml_version
from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env


class ModelEnvTest(absltest.TestCase):
    def test_empty_model_env(self) -> None:
        env = model_env.ModelEnv()
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, [])
        py_ver = version.parse(snowml_env.PYTHON_VERSION)
        self.assertEqual(env.python_version, f"{py_ver.major}.{py_ver.minor}")
        self.assertIsNone(env.cuda_version)
        self.assertEqual(env.snowpark_ml_version, snowml_version.VERSION)

    def test_conda_dependencies(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["package"]
        self.assertListEqual(env.conda_dependencies, ["package"])

        env.conda_dependencies = ["some_package"]
        self.assertListEqual(env.conda_dependencies, ["some_package"])

        env.conda_dependencies = ["some_package==1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["some_package==1.0.1"])

        env.conda_dependencies = ["some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["some_package<1.2,>=1.0.1"])

        env.conda_dependencies = ["channel::some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["channel::some_package<1.2,>=1.0.1"])

        with self.assertRaisesRegex(ValueError, "Invalid package requirement _some_package<1.2,>=1.0.1 found."):
            env.conda_dependencies = ["channel::_some_package<1.2,>=1.0.1"]

        env.conda_dependencies = ["::some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["some_package<1.2,>=1.0.1"])

        env.conda_dependencies = ["another==1.3", "channel::some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["another==1.3", "channel::some_package<1.2,>=1.0.1"])

    def test_pip_requirements(self) -> None:
        env = model_env.ModelEnv()
        env.pip_requirements = ["package"]
        self.assertListEqual(env.pip_requirements, ["package"])

        env.pip_requirements = ["some_package"]
        self.assertListEqual(env.pip_requirements, ["some_package"])

        env.pip_requirements = ["some_package==1.0.1"]
        self.assertListEqual(env.pip_requirements, ["some_package==1.0.1"])

        env.pip_requirements = ["some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.pip_requirements, ["some_package<1.2,>=1.0.1"])

        with self.assertRaisesRegex(ValueError, "Invalid package requirement channel::some_package<1.2,>=1.0.1 found."):
            env.pip_requirements = ["channel::some_package<1.2,>=1.0.1"]

    def test_python_version(self) -> None:
        env = model_env.ModelEnv()
        env.python_version = "3.9"
        self.assertEqual(env.python_version, "3.9")

        env.python_version = "3.9.16"
        self.assertEqual(env.python_version, "3.9")

        env.python_version = None  # type: ignore[assignment]
        self.assertEqual(env.python_version, "3.9")

    def test_cuda_version(self) -> None:
        env = model_env.ModelEnv()
        env.cuda_version = "11.2"
        self.assertEqual(env.cuda_version, "11.2")

        env.cuda_version = "11.2.1"
        self.assertEqual(env.cuda_version, "11.2")

        env.cuda_version = None
        self.assertEqual(env.cuda_version, "11.2")

        with self.assertRaisesRegex(ValueError, "Different CUDA version 11.2 and 12.1 found in the same model!"):
            env.cuda_version = "12.1"

    def test_snowpark_ml_version(self) -> None:
        env = model_env.ModelEnv()
        env.python_version = "3.9"
        self.assertEqual(env.python_version, "3.9")

        env.python_version = "3.9.16"
        self.assertEqual(env.python_version, "3.9")

        env.python_version = None  # type: ignore[assignment]
        self.assertEqual(env.python_version, "3.9")

    def test_include_if_absent(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.include_if_absent([model_env.ModelDependency(requirement="some-package", pip_name="some-package")])
        self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.include_if_absent([model_env.ModelDependency(requirement="some-package==1.0.2", pip_name="some-package")])
        self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.include_if_absent([model_env.ModelDependency(requirement="some-package>=1.0,<2", pip_name="some-package")])
        self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.include_if_absent(
            [model_env.ModelDependency(requirement="another-package>=1.0,<2", pip_name="some-package")]
        )
        self.assertListEqual(env.conda_dependencies, ["another-package<2,>=1.0", "some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["channel::some-package==1.0.1"]

        with self.assertWarnsRegex(UserWarning, "Basic dependency some-package specified from non-Snowflake channel."):
            env.include_if_absent([model_env.ModelDependency(requirement="some-package", pip_name="some-package")])
            self.assertListEqual(env.conda_dependencies, ["channel::some-package==1.0.1"])
            self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        with self.assertWarnsRegex(
            UserWarning,
            (
                "Dependencies specified from pip requirements."
                " This may prevent model deploying to Snowflake Warehouse."
            ),
        ):
            env.include_if_absent([model_env.ModelDependency(requirement="some-package", pip_name="some-package")])
            self.assertListEqual(env.conda_dependencies, [])
            self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["channel::some-package==1.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env.include_if_absent(
                [model_env.ModelDependency(requirement="channel::some-package", pip_name="some-package")]
            )
            self.assertListEqual(env.conda_dependencies, ["channel::some-package==1.0.1"])
            self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        env.include_if_absent([model_env.ModelDependency(requirement="channel::some-package", pip_name="some-package")])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

        env = model_env.ModelEnv(prefer_pip=True)
        env.include_if_absent([model_env.ModelDependency(requirement="channel::some-package", pip_name="some-package")])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["some-package"])

        env = model_env.ModelEnv(prefer_pip=True)
        env.conda_dependencies = ["channel::some-package==1.0.1"]
        env.include_if_absent(
            [model_env.ModelDependency(requirement="another-package>=1.0,<2", pip_name="some-package")]
        )
        self.assertListEqual(env.conda_dependencies, ["another-package<2,>=1.0", "channel::some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

    def test_include_if_absent_check_local(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = []

        env.include_if_absent(
            [model_env.ModelDependency(requirement="numpy", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(
            env.conda_dependencies,
            [str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("numpy")))],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = []

        env.include_if_absent(
            [model_env.ModelDependency(requirement="numpy>=1.0", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(
            env.conda_dependencies,
            [str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("numpy")))],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = []

        env.include_if_absent(
            [model_env.ModelDependency(requirement="numpy<1.0", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(
            env.conda_dependencies,
            ["numpy<1.0"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = []

        env.include_if_absent(
            [model_env.ModelDependency(requirement="invalid-package", pip_name="invalid-package")],
            check_local_version=True,
        )
        self.assertListEqual(
            env.conda_dependencies,
            ["invalid-package"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = []

        env.include_if_absent(
            [model_env.ModelDependency(requirement="pytorch", pip_name="torch")], check_local_version=True
        )
        self.assertListEqual(
            env.conda_dependencies,
            [
                "pytorch=="
                + list(
                    env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("torch")).specifier
                )[0].version,
            ],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["numpy==1.0.1"]

        env.include_if_absent(
            [model_env.ModelDependency(requirement="numpy", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(env.conda_dependencies, ["numpy==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["numpy==1.0.1"]

        env.include_if_absent(
            [model_env.ModelDependency(requirement="numpy==1.0.2", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(env.conda_dependencies, ["numpy==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["numpy==1.0.1"]

        env.include_if_absent(
            [model_env.ModelDependency(requirement="numpy>=1.0,<2", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(env.conda_dependencies, ["numpy==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["numpy==1.0.1"]

        env.include_if_absent(
            [model_env.ModelDependency(requirement="pytorch>=1.0", pip_name="torch")], check_local_version=True
        )
        self.assertListEqual(
            env.conda_dependencies,
            [
                "numpy==1.0.1",
                "pytorch=="
                + list(
                    env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("torch")).specifier
                )[0].version,
            ],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["channel::numpy==1.0.1"]

        with self.assertWarnsRegex(UserWarning, "Basic dependency numpy specified from non-Snowflake channel."):
            env.include_if_absent(
                [model_env.ModelDependency(requirement="numpy", pip_name="numpy")], check_local_version=True
            )
            self.assertListEqual(env.conda_dependencies, ["channel::numpy==1.0.1"])
            self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]

        with self.assertWarnsRegex(
            UserWarning,
            (
                "Dependencies specified from pip requirements."
                " This may prevent model deploying to Snowflake Warehouse."
            ),
        ):
            env.include_if_absent(
                [model_env.ModelDependency(requirement="numpy", pip_name="numpy")], check_local_version=True
            )
            self.assertListEqual(env.conda_dependencies, [])
            self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["channel::numpy==1.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env.include_if_absent(
                [model_env.ModelDependency(requirement="channel::numpy", pip_name="numpy")], check_local_version=True
            )
            self.assertListEqual(env.conda_dependencies, ["channel::numpy==1.0.1"])
            self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]

        env.include_if_absent(
            [model_env.ModelDependency(requirement="channel::numpy", pip_name="numpy")], check_local_version=True
        )
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

    def test_include_if_absent_pip(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env._include_if_absent_pip(["some-package"])
            self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])
            self.assertListEqual(env.pip_requirements, ["some-package"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env._include_if_absent_pip(["some-package"])
            self.assertListEqual(env.conda_dependencies, [])
            self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        env._include_if_absent_pip(["some-package==1.0.2"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        env._include_if_absent_pip(["some-package>=1.0,<2"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        env._include_if_absent_pip(["another-package>=1.0,<2"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["another-package<2,>=1.0", "some-package==1.0.1"])

    def test_include_if_absent_pip_check_local(self) -> None:
        env = model_env.ModelEnv()
        env._include_if_absent_pip(["numpy"], check_local_version=True)
        self.assertListEqual(
            env.conda_dependencies,
            [],
        )
        self.assertListEqual(
            env.pip_requirements,
            [str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("numpy")))],
        )

        env = model_env.ModelEnv()
        env._include_if_absent_pip(["numpy>=1.0"], check_local_version=True)
        self.assertListEqual(
            env.conda_dependencies,
            [],
        )
        self.assertListEqual(
            env.pip_requirements,
            [str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("numpy")))],
        )

        env = model_env.ModelEnv()
        env._include_if_absent_pip(["numpy<1.0"], check_local_version=True)
        self.assertListEqual(
            env.conda_dependencies,
            [],
        )
        self.assertListEqual(env.pip_requirements, ["numpy<1.0"])

        env = model_env.ModelEnv()
        env._include_if_absent_pip(["invalid-package"], check_local_version=True)
        self.assertListEqual(
            env.conda_dependencies,
            [],
        )
        self.assertListEqual(env.pip_requirements, ["invalid-package"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]

        env._include_if_absent_pip(["numpy"], check_local_version=True)
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]
        env._include_if_absent_pip(["numpy==1.0.2"], check_local_version=True)
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]
        env._include_if_absent_pip(["numpy>=1.0,<2"], check_local_version=True)
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]

        env._include_if_absent_pip(["torch>=1.0"], check_local_version=True)
        self.assertListEqual(
            env.conda_dependencies,
            [],
        )
        self.assertListEqual(
            env.pip_requirements,
            [
                "numpy==1.0.1",
                str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("torch"))),
            ],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["numpy==1.0.1"]
        env._include_if_absent_pip(["numpy>=1.0"], check_local_version=True)
        self.assertListEqual(env.conda_dependencies, ["numpy==1.0.1"])
        self.assertListEqual(
            env.pip_requirements,
            [str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("numpy")))],
        )

        env = model_env.ModelEnv()
        env.pip_requirements = ["numpy==1.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env._include_if_absent_pip(["numpy>=1.0"], check_local_version=True)
            self.assertListEqual(env.conda_dependencies, [])
            self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

    def test_remove_if_present_conda(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.remove_if_present_conda(["some-package"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.remove_if_present_conda(["some-package"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["some-package==1.0.1"]

        env.remove_if_present_conda(["another-package"])
        self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["another-package<2,>=1.0", "some-package==1.0.1"]

        env.remove_if_present_conda(["some-package", "another-package"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["another-package<2,>=1.0", "some-package==1.0.1"]

        env.remove_if_present_conda(["another-package"])
        self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["channel::some-package==1.0.1"]

        env.remove_if_present_conda(["some-package"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.pip_requirements = ["some-package==1.0.1"]

        env.remove_if_present_conda(["some-package"])
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

    def test_generate_conda_env_for_cuda(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["somepackage==1.0.0", "another_channel::another_package==1.0.0"]
        original_env = copy.deepcopy(env)
        env.generate_env_for_cuda()

        self.assertListEqual(env.conda_dependencies, original_env.conda_dependencies)
        self.assertListEqual(env.pip_requirements, original_env.pip_requirements)

        env = model_env.ModelEnv()
        env.conda_dependencies = ["somepackage==1.0.0", "another_channel::another_package==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "another_channel::another_package==1.0.0",
                "somepackage==1.0.0",
            ],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = [
            "somepackage==1.0.0",
            "another_channel::another_package==1.0.0",
        ]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "another_channel::another_package==1.0.0",
                "somepackage==1.0.0",
            ],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["pytorch==1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["pytorch>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch>=1.0.0", "pytorch::pytorch-cuda>=11.7"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["pytorch::pytorch-cuda>=11.7", "pytorch>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch::pytorch>=1.1.0", "pytorch::pytorch-cuda==11.7.*"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["pytorch::pytorch-cuda==11.7.*", "pytorch::pytorch>=1.1.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["conda-forge::pytorch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::pytorch==1.0.0"],
        )

        env = model_env.ModelEnv()
        env.pip_requirements = ["torch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [],
        )
        self.assertListEqual(env.pip_requirements, ["torch==1.0.0"])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["tensorflow==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["tensorflow-gpu==1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["tensorflow>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["tensorflow-gpu>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["tensorflow==1.0.0", "conda-forge::tensorflow-gpu==1.1.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::tensorflow-gpu==1.1.0"],
        )
        self.assertIn(env_utils.DEFAULT_CHANNEL_NAME, env._conda_dependencies)

        env = model_env.ModelEnv()
        env.pip_requirements = ["tensorflow==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["tensorflow-gpu==1.0.0"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["xgboost==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["py-xgboost-gpu==1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["xgboost>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["py-xgboost-gpu>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["xgboost>=1.0.0", "conda-forge::py-xgboost-gpu>=1.1.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::py-xgboost-gpu>=1.1.0"],
        )
        self.assertIn(env_utils.DEFAULT_CHANNEL_NAME, env._conda_dependencies)

        env = model_env.ModelEnv()
        env.conda_dependencies = ["conda-forge::xgboost>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["py-xgboost-gpu>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.pip_requirements = ["xgboost>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["py-xgboost-gpu>=1.0.0"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["transformers==1.0.0", "pytorch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "accelerate>=0.22.0",
                "pytorch==1.0.0",
                "scipy>=1.9",
                "transformers==1.0.0",
            ],
        )

        self.assertListEqual(env.pip_requirements, ["bitsandbytes>=0.41.0"])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["transformers==1.0.0", "scipy==1.0.0", "conda-forge::accelerate==1.0.0"]
        env.pip_requirements = ["bitsandbytes==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "conda-forge::accelerate==1.0.0",
                "scipy==1.0.0",
                "transformers==1.0.0",
            ],
        )

        self.assertListEqual(env.pip_requirements, ["bitsandbytes==1.0.0"])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["conda-forge::transformers==1.0.0", "conda-forge::accelerate==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "conda-forge::accelerate==1.0.0",
                "conda-forge::transformers==1.0.0",
                "scipy>=1.9",
            ],
        )

        self.assertListEqual(env.pip_requirements, ["bitsandbytes>=0.41.0"])

    def test_relax_version(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = [
            "somepackage==1.0.0,!=1.1",
            "random-package>=2.3",
            "another_channel::another-package==1.0",
        ]
        env.pip_requirements = ["pip-packages==3"]

        env.relax_version()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "another_channel::another-package<2,>=1.0",
                "random-package>=2.3",
                "somepackage!=1.1,<2,>=1.0",
            ],
        )

        self.assertListEqual(env.pip_requirements, ["pip-packages<4,>=3.0"])

    def test_artifact_repository(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["somepackage==1.0.0,!=1.1"]
        env.pip_requirements = ["pip-packages==3"]

        env.artifact_repository_map = {"channel": "db.sc.repo"}

        self.assertDictEqual(env.artifact_repository_map, {"channel": "db.sc.repo"})

    def test_resource_constraint(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["somepackage==1.0.0,!=1.1"]
        env.pip_requirements = ["pip-packages==3"]

        env.resource_constraint = {"architecture": "x86"}

        self.assertDictEqual(env.resource_constraint, {"architecture": "x86"})

    def test_load_from_conda_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file_path = pathlib.Path(os.path.join(tmpdir, "conda.yml"))
            with open(env_file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    stream=f,
                    data={
                        "name": "snow-env",
                        "channels": ["https://repo.anaconda.com/pkgs/snowflake", "apple", "nodefaults"],
                        "dependencies": [
                            "python=3.10",
                            "::numpy>=1.22.4",
                            "conda-forge::pytorch!=2.0",
                            {"pip": ["python-package", "numpy==1.22.4"]},
                        ],
                    },
                )

            env = model_env.ModelEnv()
            with self.assertWarnsRegex(
                UserWarning,
                (
                    "Found dependencies specified in the conda file from non-Snowflake channel."
                    " This may prevent model deploying to Snowflake Warehouse."
                ),
            ):
                env.load_from_conda_file(env_file_path)

            env = model_env.ModelEnv()
            with self.assertWarnsRegex(
                UserWarning,
                (
                    "Found additional conda channel apple specified in the conda file."
                    " This may prevent model deploying to Snowflake Warehouse."
                ),
            ):
                env.load_from_conda_file(env_file_path)

            env = model_env.ModelEnv()
            with self.assertWarnsRegex(
                UserWarning,
                (
                    "Found dependencies specified as pip requirements."
                    " This may prevent model deploying to Snowflake Warehouse."
                ),
            ):
                env.load_from_conda_file(env_file_path)

            self.assertListEqual(env.conda_dependencies, ["conda-forge::pytorch!=2.0", "numpy>=1.22.4"])
            self.assertIn("apple", env._conda_dependencies)
            self.assertListEqual(env.pip_requirements, ["python-package"])
            self.assertEqual(env.python_version, "3.10")

            env = model_env.ModelEnv()
            env.conda_dependencies = ["pandas==1.5.3"]
            env.pip_requirements = ["pip-only==3.0"]
            env.load_from_conda_file(env_file_path)

            self.assertListEqual(
                env.conda_dependencies, ["conda-forge::pytorch!=2.0", "numpy>=1.22.4", "pandas==1.5.3"]
            )
            self.assertIn("apple", env._conda_dependencies)
            self.assertListEqual(env.pip_requirements, ["pip-only==3.0", "python-package"])
            self.assertEqual(env.python_version, "3.10")

            env = model_env.ModelEnv()
            env.conda_dependencies = ["numpy==1.22.4"]
            env.load_from_conda_file(env_file_path)

            self.assertListEqual(env.conda_dependencies, ["conda-forge::pytorch!=2.0", "numpy==1.22.4"])
            self.assertIn("apple", env._conda_dependencies)
            self.assertListEqual(env.pip_requirements, ["python-package"])
            self.assertEqual(env.python_version, "3.10")

            env = model_env.ModelEnv()
            env.conda_dependencies = ["pytorch==2.1"]
            with self.assertWarnsRegex(
                UserWarning,
                "Dependency pytorch appeared in multiple channels as conda dependency. This may be unintentional.",
            ):
                env.load_from_conda_file(env_file_path)

            self.assertListEqual(env.conda_dependencies, ["numpy>=1.22.4", "pytorch==2.1"])
            self.assertIn("apple", env._conda_dependencies)
            self.assertListEqual(env.pip_requirements, ["python-package"])
            self.assertEqual(env.python_version, "3.10")

    def test_load_from_pip_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pip_file_path = pathlib.Path(os.path.join(tmpdir, "requirements.txt"))
            with open(pip_file_path, "w", encoding="utf-8") as f:
                f.writelines(["python-package\n", "numpy==1.22.4\n"])

            env = model_env.ModelEnv()
            with self.assertWarnsRegex(
                UserWarning,
                (
                    "Found dependencies specified as pip requirements."
                    " This may prevent model deploying to Snowflake Warehouse."
                ),
            ):
                env.load_from_pip_file(pip_file_path)

            self.assertListEqual(env.pip_requirements, ["numpy==1.22.4", "python-package"])

            env = model_env.ModelEnv()
            env.conda_dependencies = ["numpy>=1.22.4"]
            env.load_from_pip_file(pip_file_path)

            self.assertListEqual(env.pip_requirements, ["python-package"])

            env = model_env.ModelEnv()
            env.conda_dependencies = ["conda-forge::numpy>=1.22.4"]
            env.load_from_pip_file(pip_file_path)

            self.assertListEqual(env.pip_requirements, ["python-package"])

    def test_save_and_load(self) -> None:
        def check_env_equality(this: model_env.ModelEnv, that: model_env.ModelEnv) -> bool:
            return all(
                getattr(this, attr) == getattr(that, attr)
                for attr in [
                    "conda_env_rel_path",
                    "pip_requirements_rel_path",
                    "conda_dependencies",
                    "pip_requirements",
                    "python_version",
                    "cuda_version",
                    "snowpark_ml_version",
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            env = model_env.ModelEnv()
            saved_dict = env.save_as_dict(tmpdir_path)

            loaded_env = model_env.ModelEnv()
            loaded_env.load_from_dict(tmpdir_path, saved_dict)
            self.assertTrue(check_env_equality(env, loaded_env), "Loaded env object is different.")

            env = model_env.ModelEnv()
            env.conda_dependencies = ["another==1.3", "channel::some_package<1.2,>=1.0.1"]
            env.pip_requirements = ["pip-package<1.2,>=1.0.1"]
            env.python_version = "3.10.2"
            env.cuda_version = "11.7.1"
            env.snowpark_ml_version = "1.1.0"

            saved_dict = env.save_as_dict(tmpdir_path)

            self.assertDictEqual(
                saved_dict,
                {
                    "conda": "env/conda.yml",
                    "pip": "env/requirements.txt",
                    "python_version": "3.10",
                    "cuda_version": "11.7",
                    "snowpark_ml_version": "1.1.0",
                    "artifact_repository_map": {},
                    "resource_constraint": {},
                },
            )

            loaded_env = model_env.ModelEnv()
            loaded_env.load_from_dict(tmpdir_path, saved_dict)
            self.assertTrue(check_env_equality(env, loaded_env), "Loaded env object is different.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            env = model_env.ModelEnv()
            saved_dict = env.save_as_dict(tmpdir_path, default_channel_override="conda-forge", is_gpu=True)

            loaded_env = model_env.ModelEnv()
            loaded_env.load_from_dict(tmpdir_path, saved_dict)
            self.assertTrue(check_env_equality(env, loaded_env), "Loaded env object is different.")

            env = model_env.ModelEnv()
            env.conda_dependencies = ["another==1.3", "channel::some_package<1.2,>=1.0.1"]
            env.pip_requirements = ["pip-package<1.2,>=1.0.1"]
            env.python_version = "3.10.2"
            env.cuda_version = "11.7.1"
            env.snowpark_ml_version = "1.1.0"

            saved_dict = env.save_as_dict(tmpdir_path, default_channel_override="conda-forge", is_gpu=True)

            self.assertDictEqual(
                saved_dict,
                {
                    "conda": "env/conda.yml",
                    "pip": "env/requirements.txt",
                    "python_version": "3.10",
                    "cuda_version": "11.7",
                    "snowpark_ml_version": "1.1.0",
                    "artifact_repository_map": {},
                    "resource_constraint": {},
                },
            )

            with open(tmpdir_path / "env" / "conda.yml", encoding="utf-8") as f:
                conda_yml = yaml.safe_load(f)
                self.assertDictEqual(
                    conda_yml,
                    {
                        "channels": ["conda-forge", "channel", "nodefaults"],
                        "dependencies": [
                            "python==3.10.*",
                            "nvidia::cuda==11.7.*",
                            "another==1.3",
                            "channel::some_package<1.2,>=1.0.1",
                        ],
                        "name": "snow-env",
                    },
                )

            loaded_env = model_env.ModelEnv()
            loaded_env.load_from_dict(tmpdir_path, saved_dict)
            self.assertTrue(check_env_equality(env, loaded_env), "Loaded env object is different.")

    def test_validate_with_local_env(self) -> None:
        with mock.patch.object(
            env_utils, "validate_py_runtime_version"
        ) as mock_validate_py_runtime_version, mock.patch.object(
            env_utils, "validate_local_installed_version_of_pip_package"
        ) as mock_validate_local_installed_version_of_pip_package:
            env = model_env.ModelEnv()
            env.conda_dependencies = ["pytorch==1.3", "channel::some_package<1.2,>=1.0.1"]
            env.pip_requirements = ["pip-package<1.2,>=1.0.1"]
            env.python_version = "3.10.2"
            env.cuda_version = "11.7.1"
            env.snowpark_ml_version = "1.1.0"

            self.assertListEqual(env.validate_with_local_env(), [])
            mock_validate_py_runtime_version.assert_called_once_with("3.10.2")
            mock_validate_local_installed_version_of_pip_package.assert_has_calls(
                [
                    mock.call(requirements.Requirement("torch==1.3")),
                    mock.call(requirements.Requirement("some_package<1.2,>=1.0.1")),
                    mock.call(requirements.Requirement("pip-package<1.2,>=1.0.1")),
                ]
            )

        with mock.patch.object(
            env_utils, "validate_py_runtime_version", side_effect=env_utils.IncorrectLocalEnvironmentError()
        ) as mock_validate_py_runtime_version, mock.patch.object(
            env_utils,
            "validate_local_installed_version_of_pip_package",
            side_effect=env_utils.IncorrectLocalEnvironmentError(),
        ) as mock_validate_local_installed_version_of_pip_package:
            env = model_env.ModelEnv()
            env.conda_dependencies = ["pytorch==1.3", "channel::some_package<1.2,>=1.0.1"]
            env.pip_requirements = ["pip-package<1.2,>=1.0.1"]
            env.python_version = "3.10.2"
            env.cuda_version = "11.7.1"
            env.snowpark_ml_version = "1.1.0"

            self.assertLen(env.validate_with_local_env(), 4)
            mock_validate_py_runtime_version.assert_called_once_with("3.10.2")
            mock_validate_local_installed_version_of_pip_package.assert_has_calls(
                [
                    mock.call(requirements.Requirement("torch==1.3")),
                    mock.call(requirements.Requirement("some_package<1.2,>=1.0.1")),
                    mock.call(requirements.Requirement("pip-package<1.2,>=1.0.1")),
                ]
            )

        with mock.patch.object(
            env_utils, "validate_py_runtime_version"
        ) as mock_validate_py_runtime_version, mock.patch.object(
            env_utils, "validate_local_installed_version_of_pip_package"
        ) as mock_validate_local_installed_version_of_pip_package:
            env = model_env.ModelEnv()
            env.conda_dependencies = ["pytorch==1.3", "channel::some_package<1.2,>=1.0.1"]
            env.pip_requirements = ["pip-package<1.2,>=1.0.1"]
            env.python_version = "3.10.2"
            env.cuda_version = "11.7.1"
            env.snowpark_ml_version = f"{snowml_version.VERSION}+abcdef"

            self.assertListEqual(env.validate_with_local_env(check_snowpark_ml_version=True), [])
            mock_validate_py_runtime_version.assert_called_once_with("3.10.2")
            mock_validate_local_installed_version_of_pip_package.assert_has_calls(
                [
                    mock.call(requirements.Requirement("torch==1.3")),
                    mock.call(requirements.Requirement("some_package<1.2,>=1.0.1")),
                    mock.call(requirements.Requirement("pip-package<1.2,>=1.0.1")),
                ]
            )

        with mock.patch.object(
            env_utils, "validate_py_runtime_version", side_effect=env_utils.IncorrectLocalEnvironmentError()
        ) as mock_validate_py_runtime_version, mock.patch.object(
            env_utils,
            "validate_local_installed_version_of_pip_package",
            side_effect=env_utils.IncorrectLocalEnvironmentError(),
        ) as mock_validate_local_installed_version_of_pip_package:
            env = model_env.ModelEnv()
            env.conda_dependencies = ["pytorch==1.3", "channel::some_package<1.2,>=1.0.1"]
            env.pip_requirements = ["pip-package<1.2,>=1.0.1"]
            env.python_version = "3.10.2"
            env.cuda_version = "11.7.1"
            env.snowpark_ml_version = "0.0.0"

            self.assertLen(env.validate_with_local_env(check_snowpark_ml_version=True), 5)
            mock_validate_py_runtime_version.assert_called_once_with("3.10.2")
            mock_validate_local_installed_version_of_pip_package.assert_has_calls(
                [
                    mock.call(requirements.Requirement("torch==1.3")),
                    mock.call(requirements.Requirement("some_package<1.2,>=1.0.1")),
                    mock.call(requirements.Requirement("pip-package<1.2,>=1.0.1")),
                ]
            )

    def test_add_local_env_version(self) -> None:
        with mock.patch.object(
            env_utils, "get_local_installed_version_of_pip_package"
        ) as mock_get_local_installed_version_of_pip_package:
            mock_get_local_installed_version_of_pip_package.return_value = requirements.Requirement(
                "pip-package==1.6.2"
            )
            env = model_env.ModelEnv()
            env.conda_dependencies = ["pip_package"]
            env.pip_requirements = ["pip-package"]

            env2 = model_env.ModelEnv()
            env2.conda_dependencies = ["channel::pip-package"]

        self.assertEqual(env.pip_requirements, ["pip-package==1.6.2"])
        self.assertEqual(env.conda_dependencies, ["pip-package==1.6.2"])
        self.assertEqual(env2.conda_dependencies, ["channel::pip-package==1.6.2"])

    def test_warnings_with_target_platforms(self) -> None:
        env = model_env.ModelEnv(target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES])
        env.pip_requirements = ["some-package==1.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env.include_if_absent([model_env.ModelDependency(requirement="some-package", pip_name="some-package")])

        env_default = model_env.ModelEnv()
        env_default.pip_requirements = ["some-package==1.0.1"]

        with self.assertWarnsRegex(
            UserWarning, "Dependencies specified from pip requirements.*prevent model deploying to Snowflake Warehouse"
        ):
            env_default.include_if_absent(
                [model_env.ModelDependency(requirement="some-package", pip_name="some-package")]
            )

        env_warehouse = model_env.ModelEnv(target_platforms=[model_types.TargetPlatform.WAREHOUSE])
        env_warehouse.pip_requirements = ["some-package==1.0.1"]

        with self.assertWarnsRegex(
            UserWarning, "Dependencies specified from pip requirements.*prevent model deploying to Snowflake Warehouse"
        ):
            env_warehouse.include_if_absent(
                [model_env.ModelDependency(requirement="some-package", pip_name="some-package")]
            )

        env_both = model_env.ModelEnv(
            target_platforms=[
                model_types.TargetPlatform.WAREHOUSE,
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
            ]
        )
        env_both.pip_requirements = ["some-package==1.0.1"]

        with self.assertWarnsRegex(
            UserWarning, "Dependencies specified from pip requirements.*prevent model deploying to Snowflake Warehouse"
        ):
            env_both.include_if_absent([model_env.ModelDependency(requirement="some-package", pip_name="some-package")])


if __name__ == "__main__":
    absltest.main()
