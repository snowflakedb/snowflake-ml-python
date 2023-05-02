import os
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import yaml
from packaging import requirements

from snowflake.ml._internal import env as snowml_env, env_utils

_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"


def save_conda_env_file(
    dir_path: str,
    deps: DefaultDict[str, List[requirements.Requirement]],
    python_version: Optional[str] = snowml_env.PYTHON_VERSION,
) -> None:
    """Generate conda.yaml file given a dict of dependencies after validation.

    Args:
        dir_path: Path to the directory where conda.yaml file should be written.
        deps: Dict of conda dependencies after validated.
        python_version: A string 'major.minor.patchlevel' showing python version relate to model. Default to current.
    """
    path = os.path.join(dir_path, _CONDA_ENV_FILE_NAME)
    env: Dict[str, Any] = dict()
    env["name"] = "snow-env"
    env["dependencies"] = [f"python={python_version}"]
    for channel, reqs in deps.items():
        env["dependencies"].extend([f"{channel}::{req}" for req in reqs])

    with open(path, "w") as f:
        yaml.safe_dump(env, stream=f, default_flow_style=False)


def save_requirements_file(dir_path: str, pip_deps: List[requirements.Requirement]) -> None:
    """Generate Python requirements.txt file in the given directory path.

    Args:
        dir_path: Path to the directory where requirements.txt file should be written.
        pip_deps: List of dependencies string after validated.
    """
    requirements = "\n".join(map(str, pip_deps))
    path = os.path.join(dir_path, _REQUIREMENTS_FILE_NAME)
    with open(path, "w") as out:
        out.write(requirements)


def load_conda_env_file(dir_path: str) -> Tuple[DefaultDict[str, List[requirements.Requirement]], Optional[str]]:
    """Read conda.yaml file to get n a dict of dependencies after validation.

    Args:
        dir_path: Path to the directory where conda.yaml file should be written.

    Returns:
        A tuple of Dict of conda dependencies after validated and a string 'major.minor.patchlevel' of python version.
    """
    path = os.path.join(dir_path, _CONDA_ENV_FILE_NAME)
    with open(path) as f:
        env = yaml.safe_load(stream=f)

    assert isinstance(env, dict)

    deps = []

    python_version = None

    for dep in env["dependencies"]:
        if isinstance(dep, str):
            if dep.startswith("python="):
                hd, _, ver = dep.partition("=")
                assert hd == "python"
                python_version = ver
            else:
                deps.append(dep)

    return env_utils.validate_conda_dependency_string_list(deps), python_version


def load_requirements_file(dir_path: str) -> List[requirements.Requirement]:
    """Load Python requirements.txt file from the given directory path.

    Args:
        dir_path: Path to the directory where requirements.txt file should be written.

    Returns:
        List of dependencies string after validated.
    """
    path = os.path.join(dir_path, _REQUIREMENTS_FILE_NAME)
    with open(path) as f:
        reqs = f.readlines()

    return env_utils.validate_pip_requirement_string_list(reqs)
