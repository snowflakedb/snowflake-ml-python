import os
import typing as t
from sys import version_info as pyver

import yaml

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"


_conda_header = """\
name: snowml-env
channels:
  - conda-forge
"""

_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"


def _generate_conda_env_file(dir_path: str, pip_deps: t.List[str]) -> None:
    path = os.path.join(dir_path, _CONDA_ENV_FILE_NAME)
    env = yaml.safe_load(_conda_header)
    env["dependencies"] = [f"python={PYTHON_VERSION}"]
    env["dependencies"] = ["pip"]
    env["dependencies"].append({"pip": pip_deps})
    with open(path, "w") as f:
        yaml.safe_dump(env, stream=f, default_flow_style=False)


def _generate_requirements_file(dir_path: str, pip_deps: t.List[str]) -> None:
    requirements = "\n".join(pip_deps)
    path = os.path.join(dir_path, _REQUIREMENTS_FILE_NAME)
    with open(path, "w") as out:
        out.write(requirements)


# TODO: Add validations
def _validate(pip_deps: t.List[str]) -> None:
    pass


def generate_env_files(dir_path: str, pip_deps: t.List[str]) -> None:
    _validate(pip_deps)
    _generate_conda_env_file(dir_path, pip_deps)
    _generate_requirements_file(dir_path, pip_deps)
