import argparse
import collections
import copy
import functools
import itertools
import json
import sys
from typing import (
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
import toml
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


def filter_by_extras(req_info: RequirementInfo, mode: Literal["no_extras", "extras_only"]) -> bool:
    """Filter the requirements by whether it contains extras.

    Args:
        req_info: requirement information.
        mode: mode to filter the requirement. If no_extras, only requirements without extras will be returned.
            If extras_only, only requirements with extras will be returned.

    Returns:
        True if the requirement is in the mode.
    """
    return (mode == "no_extras" and len(req_info.get("requirements_extra_tags", [])) == 0) or (
        mode == "extras_only" and len(req_info.get("requirements_extra_tags", [])) > 0
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
    req_info: RequirementInfo, env: Literal["conda", "pip", "conda-only", "pip-only"]
) -> Optional[str]:
    """Get the pinned version for dev environment of the requirement in the given env.
    For each env, env specific pinned version will be chosen, if not presented, common pinned version will be chosen.

    Args:
        req_info: requirement information.
        env: environment indicator, choose from conda and pip.

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
    pyproject_file_path: str,
    mode: str,
    format: Optional[str],
    extras_filter: Optional[List[str]] = None,
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

    if extras_filter is None:
        requirements = requirements
    elif extras_filter == ["no_extras"]:
        requirements = list(filter(lambda req_info: filter_by_extras(req_info, "no_extras"), requirements))
    else:
        requirements = list(filter(lambda req_info: filter_by_extras(req_info, "no_extras"), requirements)) + list(
            filter(
                lambda req_info: any(
                    filter_by_tag(req_info, "requirements_extra_tags", extra) for extra in extras_filter
                ),
                requirements,
            )
        )

    extended_env_conda = list(
        sorted(
            filter(
                None,
                map(
                    lambda req_info: generate_dev_pinned_string(req_info, "conda"),
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
                lambda req_info: generate_dev_pinned_string(req_info, "pip-only"),
                requirements,
            ),
        )
    )

    if pip_only_reqs:
        extended_env.extend(["pip", {"pip": pip_only_reqs}])

    if (mode, format) == ("dev_version", "text"):
        results = list(
            sorted(
                map(
                    lambda s: s + "\n",
                    filter(
                        None,
                        map(
                            lambda req_info: generate_dev_pinned_string(req_info, "pip"),
                            requirements,
                        ),
                    ),
                )
            )
        )
        sys.stdout.writelines(results)
    elif (mode, format) == ("version_requirements", "python"):
        reqs = list(
            sorted(
                filter(
                    None,
                    map(
                        lambda req_info: generate_user_requirements_string(req_info, "conda"),
                        filter(lambda req_info: filter_by_extras(req_info, "no_extras"), requirements),
                    ),
                ),
            )
        )
        sys.stdout.write(f"REQUIREMENTS = {repr(reqs)}\n")
    elif (mode, format) == ("version_requirements", "toml"):
        extras_requirements = list(filter(lambda req_info: filter_by_extras(req_info, "extras_only"), requirements))
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
                        filter(lambda req_info: filter_by_extras(req_info, "no_extras"), requirements),
                    ),
                )
            )
        )
        with open(pyproject_file_path, encoding="utf-8") as f:
            pyproject_config = toml.load(f)
        pyproject_config["project"]["dependencies"] = results
        pyproject_config["project"]["optional-dependencies"] = extras_results
        toml.dump(pyproject_config, sys.stdout)
    elif (mode, format) == ("dev_version", "conda_env"):
        env_result = {
            "channels": channels_to_use,
            "dependencies": extended_env,
        }
        yaml.dump(env_result, sys.stdout)
    elif (mode, format) == ("version_requirements", "conda_env"):
        results = list(
            sorted(
                filter(
                    None,
                    map(
                        lambda req_info: generate_user_requirements_string(req_info, "conda"),
                        filter(lambda req_info: filter_by_extras(req_info, "extras_only"), requirements),
                    ),
                )
            )
        )
        env_result = {
            "channels": channels_to_use,
            "dependencies": results,
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
                        filter(lambda req_info: filter_by_extras(req_info, "no_extras"), requirements),
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
                        filter(lambda req_info: filter_by_extras(req_info, "extras_only"), requirements),
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
    parser.add_argument("--pyproject-template", type=str, help="PyProject template file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev_version", "version_requirements"],
        help="Define the mode when specifying the requirements.",
        required=True,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "toml", "python", "conda_env", "conda_meta"],
        help="Define the output format.",
    )
    parser.add_argument("--filter_by_tag", type=str, default=None, help="Filter the result by tags.")
    parser.add_argument("--filter_by_extras", type=str, default=None, help="Filter the result by extras.")
    parser.add_argument("--version", type=str, default=None, help="Filter the result by tags.")
    args = parser.parse_args()

    VALID_SETTINGS = [
        ("dev_version", "text"),  # requirements.txt
        ("version_requirements", "python"),  # sproc test dependencies list
        ("version_requirements", "toml"),  # wheel rule requirements
        ("dev_version", "conda_env"),  # dev conda-env.yml file
        ("version_requirements", "conda_env"),  # build and test conda-env.yml file
        ("version_requirements", "conda_meta"),  # conda build recipe metadata file
    ]

    if (args.mode, args.format) not in VALID_SETTINGS:
        raise ValueError("Invalid config combination found.")

    filter_by_extras: Optional[List[str]] = None
    if args.filter_by_extras:
        filter_by_extras = args.filter_by_extras.split(",")

    generate_requirements(
        args.requirement_file,
        args.schema,
        args.pyproject_template,
        args.mode,
        args.format,
        extras_filter=filter_by_extras,
        tag_filter=args.filter_by_tag,
        version=args.version,
    )


if __name__ == "__main__":
    main()
