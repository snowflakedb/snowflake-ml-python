import importlib.metadata
import os
from sys import version_info as py_ver
from typing import Any, Dict, List

import yaml
from packaging.requirements import InvalidRequirement, Requirement

from snowflake.ml.model import _utils

PYTHON_VERSION: str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_SNOWFLAKE_CONDA_CHANNEL_URL = "https://repo.anaconda.com/pkgs/snowflake"
_BASIC_DEPENDENCIES = [
    "conda-libmamba-solver",
    "pandas",
    "pyyaml",
    "typing_extensions",
    "cloudpickle",
    "anyio",
    "snowflake-snowpark-python",
]


def _generate_conda_env_file(dir_path: str, deps: List[str]) -> None:
    """Generate conda.yaml file given a list of dependencies after validation. It will try to resolve all
        dependencies using snowflake conda channel first, then try to use both snowflake and conda-forge if failed.
        If any attempt using conda resolver success, it pins the versions of all dependencies that explicitly
        provided, writes them into yaml file as well as the conda channel be used. If conda resolve failed
        anyway, it treats all dependencies as from pip.

    Args:
        dir_path: Path to the directory where conda.yaml file should be written.
        deps: List of dependencies string after validated.
    """
    path = os.path.join(dir_path, _CONDA_ENV_FILE_NAME)
    env: Dict[str, Any] = dict()
    env["name"] = "snow-env"
    env["dependencies"] = [f"python={PYTHON_VERSION}"]
    original_dependencies = [Requirement(dep).name for dep in deps]
    channels = [_SNOWFLAKE_CONDA_CHANNEL_URL, "conda-forge"]
    resolved_by_conda = False

    for i in range(1, len(channels) + 1):
        channels_used = channels[:i]
        resolved_dependencies = _utils._resolve_dependencies(deps, channels=channels_used)

        if resolved_dependencies is not None:
            env["channels"] = channels_used
            env["dependencies"].extend(
                [
                    f"{pkg_name}=={pkg_version}"
                    for (pkg_name, pkg_version) in resolved_dependencies
                    if pkg_name in original_dependencies
                ]
            )
            resolved_by_conda = True
            break

    if not resolved_by_conda:
        env["dependencies"].append("pip")
        env["dependencies"].append({"pip": deps})

    with open(path, "w") as f:
        yaml.safe_dump(env, stream=f, default_flow_style=False)


def _generate_requirements_file(dir_path: str, pip_deps: List[str]) -> None:
    """Generate Python requirements.txt file in the given directory path.

    Args:
        dir_path: Path to the directory where requirements.txt file should be written.
        pip_deps: List of dependencies string after validated.
    """
    requirements = "\n".join(pip_deps)
    path = os.path.join(dir_path, _REQUIREMENTS_FILE_NAME)
    with open(path, "w") as out:
        out.write(requirements)


def _add_basic_dependencies_if_not_exists(pip_deps: List[str]) -> List[str]:
    """Add some basic dependencies into dependencies list if they are not there.

    Args:
        pip_deps: User provided dependencies list.

    Returns:
        List: Final dependencies list.
    """
    for basic_dep in _BASIC_DEPENDENCIES:
        if all([basic_dep != Requirement(dep).name for dep in pip_deps]):
            try:
                pip_deps.append(f"{basic_dep}=={importlib.metadata.version(basic_dep)}")
            except importlib.metadata.PackageNotFoundError:
                pip_deps.append(basic_dep)
    return pip_deps


def _validate_dependencies(pip_deps: List[str]) -> None:
    """Validate user provided dependencies string.

    Args:
        pip_deps: User provided list of dependencies string.

    Raises:
        ValueError: Raised when confronting an invalid dependency string from user.
    """
    for dep_str in pip_deps:
        try:
            _ = Requirement(dep_str)
        except InvalidRequirement:
            raise ValueError(f"Invalid dependency string {dep_str}")


def generate_env_files(dir_path: str, pip_deps: List[str]) -> None:
    _validate_dependencies(pip_deps)
    pip_deps = _add_basic_dependencies_if_not_exists(pip_deps)
    _generate_conda_env_file(dir_path, pip_deps)
    _generate_requirements_file(dir_path, pip_deps)
