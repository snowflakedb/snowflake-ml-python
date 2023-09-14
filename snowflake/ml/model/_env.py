import os
import warnings
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import yaml
from packaging import requirements, version

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)

_CONDA_ENV_FILE_NAME = "conda.yaml"
_SNOWFLAKE_CONDA_CHANNEL_URL = "https://repo.anaconda.com/pkgs/snowflake"
_NODEFAULTS = "nodefaults"
_REQUIREMENTS_FILE_NAME = "requirements.txt"


def save_conda_env_file(
    dir_path: str,
    deps: DefaultDict[str, List[requirements.Requirement]],
    python_version: Optional[str] = snowml_env.PYTHON_VERSION,
) -> str:
    """Generate conda.yaml file given a dict of dependencies after validation.

    Args:
        dir_path: Path to the directory where conda.yaml file should be written.
        deps: Dict of conda dependencies after validated.
        python_version: A string 'major.minor.patchlevel' showing python version relate to model. Default to current.

    Returns:
        The path to conda env file.
    """
    path = os.path.join(dir_path, _CONDA_ENV_FILE_NAME)
    env: Dict[str, Any] = dict()
    env["name"] = "snow-env"
    # Get all channels in the dependencies, ordered by the number of the packages which belongs to
    channels = list(dict(sorted(deps.items(), key=lambda item: len(item[1]), reverse=True)).keys())
    if env_utils.DEFAULT_CHANNEL_NAME in channels:
        channels.remove(env_utils.DEFAULT_CHANNEL_NAME)
    env["channels"] = [_SNOWFLAKE_CONDA_CHANNEL_URL] + channels + [_NODEFAULTS]
    env["dependencies"] = [f"python=={python_version}"]
    for chan, reqs in deps.items():
        env["dependencies"].extend([f"{chan}::{str(req)}" if chan else str(req) for req in reqs])

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(env, stream=f, default_flow_style=False)

    return path


def save_requirements_file(dir_path: str, pip_deps: List[requirements.Requirement]) -> str:
    """Generate Python requirements.txt file in the given directory path.

    Args:
        dir_path: Path to the directory where requirements.txt file should be written.
        pip_deps: List of dependencies string after validated.

    Returns:
        The path to pip requirements file.
    """
    requirements = "\n".join(map(str, pip_deps))
    path = os.path.join(dir_path, _REQUIREMENTS_FILE_NAME)
    with open(path, "w", encoding="utf-8") as out:
        out.write(requirements)

    return path


def load_conda_env_file(path: str) -> Tuple[DefaultDict[str, List[requirements.Requirement]], Optional[str]]:
    """Read conda.yaml file to get n a dict of dependencies after validation.

    Args:
        path: Path to conda.yaml.

    Returns:
        A tuple of Dict of conda dependencies after validated and a string 'major.minor.patchlevel' of python version.
    """
    with open(path, encoding="utf-8") as f:
        env = yaml.safe_load(stream=f)

    assert isinstance(env, dict)

    deps = []

    python_version = None

    channels = env["channels"]
    channels.remove(_SNOWFLAKE_CONDA_CHANNEL_URL)
    channels.remove(_NODEFAULTS)

    for dep in env["dependencies"]:
        if isinstance(dep, str):
            ver = env_utils.parse_python_version_string(dep)
            # ver is None: not python, ver is "": python w/o specifier, ver is str: python w/ specifier
            if ver is not None:
                if ver:
                    python_version = ver
            else:
                deps.append(dep)

    conda_dep_dict = env_utils.validate_conda_dependency_string_list(deps)

    if len(channels) > 0:
        for channel in channels:
            if channel not in conda_dep_dict:
                conda_dep_dict[channel] = []

    return conda_dep_dict, python_version


def load_requirements_file(path: str) -> List[requirements.Requirement]:
    """Load Python requirements.txt file from the given directory path.

    Args:
        path: Path to the requirements.txt file.

    Returns:
        List of dependencies string after validated.
    """
    with open(path, encoding="utf-8") as f:
        reqs = f.readlines()

    return env_utils.validate_pip_requirement_string_list(reqs)


def validate_py_runtime_version(provided_py_version_str: str) -> None:
    if provided_py_version_str != snowml_env.PYTHON_VERSION:
        provided_py_version = version.parse(provided_py_version_str)
        current_py_version = version.parse(snowml_env.PYTHON_VERSION)
        if (
            provided_py_version.major != current_py_version.major
            or provided_py_version.minor != current_py_version.minor
        ):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.LOCAL_ENVIRONMENT_ERROR,
                original_exception=RuntimeError(
                    f"Unable to load model which is saved with Python {provided_py_version_str} "
                    f"while current Python version is {snowml_env.PYTHON_VERSION}. "
                    "To load model metadata only, set meta_only to True."
                ),
            )
        warnings.warn(
            (
                f"Model is saved with Python {provided_py_version_str} "
                f"while current Python version is {snowml_env.PYTHON_VERSION}. "
                "There might be some issues when using loaded model."
            ),
            category=RuntimeWarning,
        )
