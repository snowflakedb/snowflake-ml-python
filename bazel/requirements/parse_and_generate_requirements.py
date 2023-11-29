import argparse
import collections
import contextlib
import copy
import functools
import itertools
import json
import os
import platform
import sys
from typing import (
    Generator,
    List,
    Literal,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TypedDict,
    Union,
    cast,
)

import jsonschema
from conda_libmamba_solver import solver
from packaging import requirements as packaging_requirements
from ruamel.yaml import YAML

SNOWFLAKE_CONDA_CHANNEL = "https://repo.anaconda.com/pkgs/snowflake"

yaml = YAML()
yaml.default_flow_style = False
yaml.map_indent = 2  # type: ignore[assignment]
yaml.sequence_dash_offset = 2
yaml.sequence_indent = 4  # type: ignore[assignment]
yaml.width = 120  # type: ignore[assignment]


class RequirementInfo(TypedDict, total=False):
    """This reflect the requirements.schema.json file."""

    name: str
    name_pypi: str
    name_conda: str
    dev_version: str
    dev_version_pypi: str
    dev_version_conda: str
    from_channel: str
    version_requirements: str
    version_requirements_pypi: str
    version_requirements_conda: str
    require_gpu: bool
    requirements_extra_tags: Sequence[str]
    tags: Sequence[str]


def filter_by_tag(
    req_info: RequirementInfo,
    field: Literal["tags", "requirements_extra_tags"],
    tag_filter: Optional[str] = None,
) -> bool:
    """Filter the requirement by whether given tag filter appears in the given field in the requirement information.
    The field is an array.

    Args:
        req_info: requirement information.
        field: field to filter the tag from.
        tag_filter: tag to filter the requirement. Defaults to None.

    Returns:
        True if tag_filter is None, or in the array of given field in presented.
    """
    return tag_filter is None or tag_filter in req_info.get(field, [])


def filter_by_extras(req_info: RequirementInfo, extras: bool, no_extras: bool) -> bool:
    """Filter the requirements by whether it contains extras.

    Args:
        req_info: requirement information.
        extras: if set to True, only filter those requirements are extras.
        no_extras: if set to True, only filter those requirements are not extras.

    Returns:
        True, for all requirements if extras and no_extras are both False;
        or for all extras requirements if extras is True;
        or for all non-extras requirements if no_extras is True.
    """
    return (
        (not extras and not no_extras)
        or (extras and len(req_info.get("requirements_extra_tags", [])) > 0)
        or (no_extras and len(req_info.get("requirements_extra_tags", [])) == 0)
    )


def get_req_name(req_info: RequirementInfo, env: Literal["conda", "pip", "conda-only", "pip-only"]) -> Optional[str]:
    """Get the name of the requirement in the given env.
    For each env, env specific name will be chosen, if not presented, common name will be chosen.

    Args:
        req_info: requirement information.
        env: environment indicator, choose from conda and pip.


    Raises:
        ValueError: Illegal env argument.

    Returns:
        The name of the requirement, if not presented, return None.
    """
    if env == "conda":
        return req_info.get("name_conda", req_info.get("name", None))
    elif env == "conda-only":
        if "name_pypi" in req_info or "name" in req_info:
            return None
        return req_info.get("name_conda", None)
    elif env == "pip":
        return req_info.get("name_pypi", req_info.get("name", None))
    elif env == "pip-only":
        if "name_conda" in req_info or "name" in req_info:
            return None
        return req_info.get("name_pypi", None)
    else:
        raise ValueError("Unreachable")


def generate_dev_pinned_string(
    req_info: RequirementInfo, env: Literal["conda", "pip", "conda-only", "pip-only"], has_gpu: bool = False
) -> Optional[str]:
    """Get the pinned version for dev environment of the requirement in the given env.
    For each env, env specific pinned version will be chosen, if not presented, common pinned version will be chosen.

    Args:
        req_info: requirement information.
        env: environment indicator, choose from conda and pip.
        has_gpu: If the environment has GPU, present to filter require required GPU package.

    Raises:
        ValueError: Illegal env argument.
        ValueError: No pinned dev version exists, which is not allowed.

    Returns:
        If the name is None, return None.
        Otherwise, return name==x.y.z format string, showing the pinned version in the dev environment.
    """
    name = get_req_name(req_info, env)
    if name is None:
        return None
    if not has_gpu and req_info.get("require_gpu", False):
        return None
    if env.startswith("conda"):
        version = req_info.get("dev_version_conda", req_info.get("dev_version", None))
        if version is None:
            raise ValueError("No pinned version exists.")
        if env == "conda-only":
            if "dev_version_conda" in req_info or "dev_version" in req_info:
                return None
        from_channel = req_info.get("from_channel", None)
        if version == "":
            version_str = ""
        else:
            version_str = f"=={version}"
        if from_channel:
            return f"{from_channel}::{name}{version_str}"
        return f"{name}{version_str}"
    elif env.startswith("pip"):
        version = req_info.get("dev_version_pypi", req_info.get("dev_version", None))
        if version is None:
            raise ValueError("No pinned version exists.")
        if env == "pip-only":
            if "dev_version_conda" in req_info or "dev_version" in req_info:
                return None
        if version == "":
            version_str = ""
        else:
            version_str = f"=={version}"
        return f"{name}{version_str}"
    else:
        raise ValueError("Unreachable")


def generate_user_requirements_string(req_info: RequirementInfo, env: Literal["conda", "pip"]) -> Optional[str]:
    """Get the user requirements version specifier string of the requirement in the given env.
    For each env, env specific user requirements version will be chosen, if not presented, common one will be chosen.

    Args:
        req_info: requirement information.
        env: environment indicator, choose from conda and pip.

    Raises:
        ValueError: Illegal env argument.

    Returns:
        If the name is None, return None.
        If no user requirements version, return the package name.
        Otherwise, return PEP-508 compatible format string, showing requirements when users install SnowML.
    """
    name = get_req_name(req_info, env)
    if name is None:
        return None
    if env == "conda":
        specifiers = req_info.get("version_requirements_conda", req_info.get("version_requirements", None))
    elif env == "pip":
        specifiers = req_info.get("version_requirements_pypi", req_info.get("version_requirements", None))
    else:
        raise ValueError("Unreachable")
    if specifiers is None:
        return None
    return f"{name}{specifiers}"


def validate_dev_version_and_user_requirements(req_info: RequirementInfo, env: Literal["conda", "pip"]) -> None:
    """Validate dev version and the user requirements version of the requirement in the given env.
    Check if dev version is within the user requirements version.

    Args:
        req_info: requirement information.
        env: environment indicator, choose from conda and pip.

    Raises:
        ValueError: Illegal env argument.
        ValueError: No pinned dev version exists, which is not allowed.
        ValueError: Pinned dev version does not exist in user requirements.
    """
    user_requirements_string = generate_user_requirements_string(req_info, env)
    if user_requirements_string is None:
        return
    if env == "conda":
        version = req_info.get("dev_version_conda", req_info.get("dev_version", None))
    elif env == "pip":
        version = req_info.get("dev_version_pypi", req_info.get("dev_version", None))
    else:
        raise ValueError("Unreachable")
    req = packaging_requirements.Requirement(user_requirements_string)
    if version is None:
        raise ValueError("No pinned version exists.")
    if not req.specifier.contains(version):
        raise ValueError(
            f"Pinned dev version {version} does not exist in user requirements {user_requirements_string}."
        )
    return


def resolve_conda_environment(specs: Sequence[str], channels: Sequence[str]) -> None:
    """Use conda api to check if given packages are resolvable in given channels.

    Args:
        specs: Packages to be installed.
        channels: Anaconda channels (name or url) where conda should search into.

    Raises:
        ValueError: Raised when the resolving result is empty.

    """

    @contextlib.contextmanager
    def _block_print() -> Generator[None, None, None]:
        _original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        yield
        sys.stdout.close()
        sys.stdout = _original_stdout

    with _block_print():
        conda_solver = solver.LibMambaSolver(
            "snowml-dev", channels=channels, specs_to_add=list(specs) + [f"python=={platform.python_version()}"]
        )
        solve_result = conda_solver.solve_final_state()
        if solve_result is None:
            raise ValueError("Unable to resolve the environment.")


def fold_extras_tags(extras_tags: Set[str], req_info: RequirementInfo) -> Set[str]:
    """Left-fold style function to get all extras tags in all requirements.

    Args:
        extras_tags: A set containing all existing extras tags.
        req_info: requirement information.

    Returns:
        Updated set with tags in the requirement information added.
    """
    for extras_tag in req_info.get("requirements_extra_tags", []):
        extras_tags.add(extras_tag)
    return extras_tags


def fold_channel(channels: Set[str], req_info: RequirementInfo) -> Set[str]:
    """Left-fold style function to get all channels in all requirements.

    Args:
        channels: A set containing all existing extras channels.
        req_info: requirement information.

    Returns:
        Updated set with channels in the requirement information added.
    """
    channel = req_info.get("from_channel", None)
    if channel:
        channels.add(channel)
    return channels


def generate_requirements(
    req_file_path: str,
    schema_file_path: str,
    mode: str,
    format: Optional[str],
    snowflake_channel_only: bool,
    tag_filter: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    with open(schema_file_path, encoding="utf-8") as f:
        schema = json.load(f)
    with open(req_file_path, encoding="utf-8") as f:
        requirements = yaml.load(f)

    jsonschema.validate(requirements, schema=schema)

    requirements = cast(Sequence[RequirementInfo], requirements)
    requirements = list(filter(lambda req_info: filter_by_tag(req_info, "tags", tag_filter), requirements))

    for req_info in requirements:
        validate_dev_version_and_user_requirements(req_info, "pip")
        validate_dev_version_and_user_requirements(req_info, "conda")

    reqs_pypi = list(filter(None, map(lambda req_info: get_req_name(req_info, "pip"), requirements)))
    reqs_conda = list(filter(None, map(lambda req_info: get_req_name(req_info, "conda"), requirements)))
    if len(reqs_pypi) != len(set(reqs_pypi)) or len(reqs_conda) != len(set(reqs_conda)):
        # TODO: Remove this after snowpandas is released and is no longer a CI internal requirement.
        counter = collections.Counter(reqs_pypi)
        duplicates = {item for item, count in counter.items() if count > 1}
        if duplicates and duplicates != {"snowflake-snowpark-python"}:
            raise ValueError(f"Duplicate Requirements: {duplicates}")
    channels_to_use = [SNOWFLAKE_CONDA_CHANNEL, "nodefaults"]

    if mode == "dev_gpu_version":
        pytorch_req = next(filter(lambda req: get_req_name(req, "conda") == "pytorch", requirements), None)
        if pytorch_req:
            pytorch_req["from_channel"] = "pytorch"
            # TODO(halu): Central place for supported CUDA version.
            # To integrate with cuda util.
            cuda_req = RequirementInfo(
                name_conda="cuda",
                dev_version_conda="11.7.*",
                from_channel="nvidia",
            )
            pytorch_cuda_req = RequirementInfo(name_conda="pytorch-cuda", dev_version="11.7.*", from_channel="pytorch")
            requirements.extend([cuda_req, pytorch_cuda_req])

    snowflake_only_env = list(
        sorted(
            filter(
                None,
                map(
                    lambda req_info: generate_dev_pinned_string(req_info, "conda", has_gpu=(mode == "dev_gpu_version")),
                    filter(
                        lambda req_info: req_info.get("from_channel", SNOWFLAKE_CONDA_CHANNEL)
                        == SNOWFLAKE_CONDA_CHANNEL,
                        requirements,
                    ),
                ),
            )
        )
    )
    extended_env_conda = list(
        sorted(
            filter(
                None,
                map(
                    lambda req_info: generate_dev_pinned_string(req_info, "conda", has_gpu=(mode == "dev_gpu_version")),
                    requirements,
                ),
            )
        )
    )

    extended_env: List[Union[str, MutableMapping[str, Sequence[str]]]] = copy.deepcopy(
        extended_env_conda  # type: ignore[arg-type]
    )
    # Relative order needs to be maintained here without sorting.
    # For external pip-only packages, we want to it able to access pypi.org index,
    # while for internal pip-only packages, nexus is the only viable index.
    # Relative order is here to prevent nexus index overriding public index.
    pip_only_reqs = list(
        filter(
            None,
            map(
                lambda req_info: generate_dev_pinned_string(req_info, "pip-only", has_gpu=(mode == "dev_gpu_version")),
                requirements,
            ),
        )
    )
    if pip_only_reqs:
        extended_env.extend(["pip", {"pip": pip_only_reqs}])

    if (mode, format) == ("validate", None):
        resolve_conda_environment(snowflake_only_env, channels=channels_to_use)
        resolve_conda_environment(extended_env_conda, channels=channels_to_use)
    elif (mode, format) == ("dev_version", "text"):
        results = list(
            sorted(
                map(
                    lambda s: s + "\n",
                    filter(
                        None,
                        map(
                            lambda req_info: generate_dev_pinned_string(
                                req_info, "pip", has_gpu=(mode == "dev_gpu_version")
                            ),
                            requirements,
                        ),
                    ),
                )
            )
        )
        sys.stdout.writelines(results)
    elif (mode, format) == ("dev_version", "python"):
        sys.stdout.write(f"REQUIREMENTS = {json.dumps(snowflake_only_env, indent=4)}\n")
    elif (mode, format) == ("version_requirements", "bzl"):
        extras_requirements = list(filter(lambda req_info: filter_by_extras(req_info, True, False), requirements))
        extras_results: MutableMapping[str, Sequence[str]] = {}
        all_extras_tags: Set[str] = set()
        all_extras_tags = functools.reduce(fold_extras_tags, requirements, all_extras_tags)
        for extras_tag in sorted(list(all_extras_tags)):
            requirements_with_tag = list(
                filter(
                    lambda req_info: filter_by_tag(req_info, "requirements_extra_tags", extras_tag),
                    extras_requirements,
                )
            )
            extras_results[extras_tag] = list(
                sorted(
                    filter(
                        None,
                        map(
                            lambda req_info: generate_user_requirements_string(req_info, "pip"),
                            requirements_with_tag,
                        ),
                    )
                )
            )
        extras_results["all"] = sorted(list(set(itertools.chain(*extras_results.values()))))
        extras_results = {k: extras_results[k] for k in sorted(extras_results)}
        results = list(
            sorted(
                filter(
                    None,
                    map(
                        lambda req_info: generate_user_requirements_string(req_info, "pip"),
                        filter(lambda req_info: filter_by_extras(req_info, False, True), requirements),
                    ),
                )
            )
        )
        sys.stdout.write(
            "EXTRA_REQUIREMENTS = {extra_requirements}\n\nREQUIREMENTS = {requirements}\n".format(
                extra_requirements=json.dumps(extras_results, indent=4), requirements=json.dumps(results, indent=4)
            )
        )
    elif (mode, format) == ("version_requirements", "python"):
        results = list(
            sorted(
                filter(None, map(lambda req_info: generate_user_requirements_string(req_info, "conda"), requirements)),
            )
        )
        sys.stdout.writelines(f"REQUIREMENTS = {repr(results)}\n")
    elif (mode, format) == ("dev_version", "conda_env") or (mode, format) == ("dev_gpu_version", "conda_env"):
        env_result = {
            "channels": channels_to_use,
            "dependencies": snowflake_only_env if snowflake_channel_only else extended_env,
        }
        yaml.dump(env_result, sys.stdout)
    elif (mode, format) == ("version_requirements", "conda_meta"):
        if version is None:
            raise ValueError("Version must be specified when generate conda meta.")
        run_results = list(
            sorted(
                filter(
                    None,
                    map(
                        lambda req_info: generate_user_requirements_string(req_info, "conda"),
                        filter(lambda req_info: filter_by_extras(req_info, False, True), requirements),
                    ),
                )
            )
        )
        run_constrained_results = list(
            sorted(
                filter(
                    None,
                    map(
                        lambda req_info: generate_user_requirements_string(req_info, "conda"),
                        filter(lambda req_info: filter_by_extras(req_info, True, False), requirements),
                    ),
                )
            )
        )
        meta_result = {
            "package": {"version": version},
            "requirements": {"run": run_results, "run_constrained": run_constrained_results},
        }
        yaml.dump(meta_result, sys.stdout)
    else:
        raise ValueError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requirement_file", help="Path to the requirement.yaml file", type=str)
    parser.add_argument("--schema", type=str, help="Path to the json schema file.", required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev_version", "dev_gpu_version", "version_requirements", "version_requirements_extras", "validate"],
        help="Define the mode when specifying the requirements.",
        required=True,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "bzl", "python", "conda_env", "conda_meta"],
        help="Define the output format.",
    )
    parser.add_argument("--filter_by_tag", type=str, default=None, help="Filter the result by tags.")
    parser.add_argument("--version", type=str, default=None, help="Filter the result by tags.")
    parser.add_argument(
        "--snowflake_channel_only",
        action="store_true",
        default=False,
        help="Flag to set if only output dependencies in Snowflake Anaconda Channel.",
    )
    args = parser.parse_args()

    VALID_SETTINGS = [
        ("validate", None, False),  # Validate the environment
        ("dev_version", "text", False),  # requirements.txt
        ("dev_version", "python", True),  # sproc test dependencies list
        ("version_requirements", "bzl", False),  # wheel rule requirements
        ("version_requirements", "python", False),  # model deployment core dependencies list
        ("dev_version", "conda_env", False),  # dev conda-env.yml file
        ("dev_gpu_version", "conda_env", False),  # dev conda-gpu-env.yml file
        ("dev_version", "conda_env", True),  # dev conda-env-snowflake.yml file
        ("version_requirements", "conda_meta", False),  # conda build recipe metadata file
    ]

    if (args.mode, args.format, args.snowflake_channel_only) not in VALID_SETTINGS:
        raise ValueError("Invalid config combination found.")

    generate_requirements(
        args.requirement_file,
        args.schema,
        args.mode,
        args.format,
        args.snowflake_channel_only,
        args.filter_by_tag,
        args.version,
    )


if __name__ == "__main__":
    main()
