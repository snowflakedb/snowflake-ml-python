import os

import yaml
from absl.testing import absltest

from snowflake.ml.model import _env


class EnvTest(absltest.TestCase):
    def test_validate_dependencies(self) -> None:
        _env._validate_dependencies(
            [
                "python_package==1.0.1",
                "python_package!=1.0.1",
                "python_package>=1.0.1",
                "python_package<=1.0.1",
                "python_package>1.0.1",
                "python_package<1.0.1",
                "python_package~=1.0.1",
            ]
        )
        with self.assertRaises(ValueError):
            _env._validate_dependencies(["python_package=1.0.1"])
        with self.assertRaises(ValueError):
            _env._validate_dependencies(["_python_package==1.0.1"])

    def test_add_basic_dependencies_if_not_exists(self) -> None:
        dependencies_list_1_mod = _env._add_basic_dependencies_if_not_exists(_env._BASIC_DEPENDENCIES.copy())
        self.assertListEqual(_env._BASIC_DEPENDENCIES, dependencies_list_1_mod)

    def test_generate_conda_env_file_snowflake_only(self) -> None:
        tmpdir = self.create_tempdir()
        _env._generate_conda_env_file(tmpdir.full_path, ["numpy"])
        with open(os.path.join(tmpdir, _env._CONDA_ENV_FILE_NAME)) as f:
            conda_env = yaml.safe_load(stream=f)
            self.assertListEqual(conda_env["channels"], [_env._SNOWFLAKE_CONDA_CHANNEL_URL])

    def test_generate_conda_env_file_additional_channel(self) -> None:
        tmpdir = self.create_tempdir()
        # This is a known version that available in conda-forge but not in snowflake channel.
        _env._generate_conda_env_file(tmpdir.full_path, ["numpy==1.22.4"])
        with open(os.path.join(tmpdir, _env._CONDA_ENV_FILE_NAME)) as f:
            conda_env = yaml.safe_load(stream=f)
            self.assertDictEqual(
                conda_env,
                {
                    "name": "snow-env",
                    "dependencies": [f"python={_env.PYTHON_VERSION}", "numpy==1.22.4"],
                    "channels": [_env._SNOWFLAKE_CONDA_CHANNEL_URL, "conda-forge"],
                },
            )

    def test_generate_conda_env_file_pip(self) -> None:
        tmpdir = self.create_tempdir()
        _env._generate_conda_env_file(tmpdir.full_path, ["this_is_a_random_pip_package==2.5.68"])
        with open(os.path.join(tmpdir, _env._CONDA_ENV_FILE_NAME)) as f:
            conda_env = yaml.safe_load(stream=f)
            self.assertDictEqual(
                conda_env,
                {
                    "name": "snow-env",
                    "dependencies": [
                        f"python={_env.PYTHON_VERSION}",
                        "pip",
                        {"pip": ["this_is_a_random_pip_package==2.5.68"]},
                    ],
                },
            )


if __name__ == "__main__":
    absltest.main()
