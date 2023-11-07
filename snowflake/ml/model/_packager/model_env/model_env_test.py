import copy
import os
import pathlib
import tempfile

import yaml
from absl.testing import absltest
from packaging import requirements, version

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model._packager.model_env import model_env


class ModelEnvTest(absltest.TestCase):
    def test_empty_model_env(self) -> None:
        env = model_env.ModelEnv()
        self.assertListEqual(env.conda_dependencies, [])
        self.assertListEqual(env.pip_requirements, [])
        py_ver = version.parse(snowml_env.PYTHON_VERSION)
        self.assertEqual(env.python_version, f"{py_ver.major}.{py_ver.minor}")
        self.assertIsNone(env.cuda_version)
        self.assertEqual(env.snowpark_ml_version, snowml_env.VERSION)

    def test_conda_dependencies(self) -> None:
        env = model_env.ModelEnv()
        env.conda_dependencies = ["package"]
        self.assertListEqual(env.conda_dependencies, ["package"])

        env.conda_dependencies = ["some_package"]
        self.assertListEqual(env.conda_dependencies, ["some-package"])

        env.conda_dependencies = ["some_package==1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["some-package==1.0.1"])

        env.conda_dependencies = ["some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["some-package<1.2,>=1.0.1"])

        env.conda_dependencies = ["channel::some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["channel::some-package<1.2,>=1.0.1"])

        with self.assertRaisesRegex(ValueError, "Invalid package requirement _some_package<1.2,>=1.0.1 found."):
            env.conda_dependencies = ["channel::_some_package<1.2,>=1.0.1"]

        env.conda_dependencies = ["::some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["some-package<1.2,>=1.0.1"])

        env.conda_dependencies = ["another==1.3", "channel::some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.conda_dependencies, ["another==1.3", "channel::some-package<1.2,>=1.0.1"])

    def test_pip_requirements(self) -> None:
        env = model_env.ModelEnv()
        env.pip_requirements = ["package"]
        self.assertListEqual(env.pip_requirements, ["package"])

        env.pip_requirements = ["some_package"]
        self.assertListEqual(env.pip_requirements, ["some-package"])

        env.pip_requirements = ["some_package==1.0.1"]
        self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

        env.pip_requirements = ["some_package<1.2,>=1.0.1"]
        self.assertListEqual(env.pip_requirements, ["some-package<1.2,>=1.0.1"])

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
                "Basic dependency some-package specified from PIP requirements. "
                "This may prevent model deploying to Snowflake Warehouse."
            ),
        ):
            env.include_if_absent([model_env.ModelDependency(requirement="some-package", pip_name="some-package")])
            self.assertListEqual(env.conda_dependencies, [])
            self.assertListEqual(env.pip_requirements, ["some-package==1.0.1"])

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
            [str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("torch")))],
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
                str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("torch"))),
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
                "Basic dependency numpy specified from PIP requirements. "
                "This may prevent model deploying to Snowflake Warehouse."
            ),
        ):
            env.include_if_absent(
                [model_env.ModelDependency(requirement="numpy", pip_name="numpy")], check_local_version=True
            )
            self.assertListEqual(env.conda_dependencies, [])
            self.assertListEqual(env.pip_requirements, ["numpy==1.0.1"])

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
                "another_channel::another-package==1.0.0",
                "nvidia::cuda==11.7.*",
                "somepackage==1.0.0",
            ],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = [
            "nvidia::cuda>=11.7",
            "somepackage==1.0.0",
            "another_channel::another_package==1.0.0",
        ]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "another_channel::another-package==1.0.0",
                "nvidia::cuda>=11.7",
                "somepackage==1.0.0",
            ],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = [
            "nvidia::cuda==11.8.*",
            "somepackage==1.0.0",
            "another_channel::another_package==1.0.0",
        ]
        env.cuda_version = "11.7"
        with self.assertRaisesRegex(
            ValueError,
            "The CUDA requirement you specified in your conda dependencies or pip requirements is"
            " conflicting with CUDA version required. Please do not specify CUDA dependency using conda"
            " dependencies or pip requirements.",
        ):
            env.generate_env_for_cuda()

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["nvidia::cuda==11.7.*", "pytorch::pytorch-cuda==11.7.*", "pytorch::pytorch==1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["nvidia::cuda==11.7.*", "pytorch::pytorch-cuda==11.7.*", "pytorch::pytorch>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch>=1.0.0", "pytorch::pytorch-cuda>=11.7"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["nvidia::cuda==11.7.*", "pytorch::pytorch-cuda>=11.7", "pytorch::pytorch>=1.0.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch>=1.0.0", "pytorch::pytorch-cuda==11.8.*"]
        env.cuda_version = "11.7"

        with self.assertRaisesRegex(
            ValueError,
            "The Pytorch-CUDA requirement you specified in your conda dependencies or pip requirements is"
            " conflicting with CUDA version required. Please do not specify Pytorch-CUDA dependency using conda"
            " dependencies or pip requirements.",
        ):
            env.generate_env_for_cuda()

        env = model_env.ModelEnv()
        env.conda_dependencies = ["pytorch::pytorch>=1.1.0", "pytorch::pytorch-cuda==11.7.*"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["nvidia::cuda==11.7.*", "pytorch::pytorch-cuda==11.7.*", "pytorch::pytorch>=1.1.0"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["conda-forge::pytorch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["nvidia::cuda==11.7.*", "pytorch::pytorch-cuda==11.7.*", "pytorch::pytorch==1.0.0"],
        )
        self.assertIn("conda-forge", env._conda_dependencies)

        env = model_env.ModelEnv()
        env.pip_requirements = ["torch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["nvidia::cuda==11.7.*", "pytorch::pytorch-cuda==11.7.*", "pytorch::pytorch==1.0.0"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["tensorflow==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::tensorflow-gpu==1.0.0", "nvidia::cuda==11.7.*"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["tensorflow>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::tensorflow-gpu>=1.0.0", "nvidia::cuda==11.7.*"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["tensorflow==1.0.0", "conda-forge::tensorflow-gpu==1.1.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::tensorflow-gpu==1.1.0", "nvidia::cuda==11.7.*"],
        )
        self.assertIn(env_utils.DEFAULT_CHANNEL_NAME, env._conda_dependencies)

        env = model_env.ModelEnv()
        env.pip_requirements = ["tensorflow==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::tensorflow-gpu==1.0.0", "nvidia::cuda==11.7.*"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["xgboost==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::py-xgboost-gpu==1.0.0", "nvidia::cuda==11.7.*"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["xgboost>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::py-xgboost-gpu>=1.0.0", "nvidia::cuda==11.7.*"],
        )

        env = model_env.ModelEnv()
        env.conda_dependencies = ["xgboost>=1.0.0", "conda-forge::py-xgboost-gpu>=1.1.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::py-xgboost-gpu>=1.1.0", "nvidia::cuda==11.7.*"],
        )
        self.assertIn(env_utils.DEFAULT_CHANNEL_NAME, env._conda_dependencies)

        env = model_env.ModelEnv()
        env.conda_dependencies = ["conda-forge::xgboost>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::py-xgboost-gpu>=1.0.0", "nvidia::cuda==11.7.*"],
        )

        env = model_env.ModelEnv()
        env.pip_requirements = ["xgboost>=1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            ["conda-forge::py-xgboost-gpu>=1.0.0", "nvidia::cuda==11.7.*"],
        )
        self.assertListEqual(env.pip_requirements, [])

        env = model_env.ModelEnv()
        env.conda_dependencies = ["transformers==1.0.0", "pytorch==1.0.0"]
        env.cuda_version = "11.7"

        env.generate_env_for_cuda()

        self.assertListEqual(
            env.conda_dependencies,
            [
                "conda-forge::accelerate>=0.22.0",
                "nvidia::cuda==11.7.*",
                "pytorch::pytorch-cuda==11.7.*",
                "pytorch::pytorch==1.0.0",
                str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("scipy"))),
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
                "nvidia::cuda==11.7.*",
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
                "nvidia::cuda==11.7.*",
                str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("scipy"))),
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
                },
            )

            loaded_env = model_env.ModelEnv()
            loaded_env.load_from_dict(tmpdir_path, saved_dict)
            self.assertTrue(check_env_equality(env, loaded_env), "Loaded env object is different.")


if __name__ == "__main__":
    absltest.main()
