import collections
import copy
import pathlib
import re
import textwrap
import warnings
from enum import Enum
from importlib import metadata as importlib_metadata
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import yaml
from packaging import requirements, specifiers, version

import snowflake.connector
from snowflake.ml._internal import env as snowml_env
from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import context, exceptions, session
from snowflake.snowpark._internal import utils as snowpark_utils


class CONDA_OS(Enum):
    LINUX_64 = "linux-64"
    LINUX_AARCH64 = "linux-aarch64"
    OSX_64 = "osx-64"
    OSX_ARM64 = "osx-arm64"
    WIN_64 = "win-64"
    NO_ARCH = "noarch"


_NODEFAULTS = "nodefaults"
_SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE: Dict[str, List[version.Version]] = {}
_SNOWFLAKE_CONDA_PACKAGE_CACHE: Dict[str, List[version.Version]] = {}
_SUPPORTED_PACKAGE_SPEC_OPS = ["==", ">=", "<=", ">", "<"]

DEFAULT_CHANNEL_NAME = ""
SNOWML_SPROC_ENV = "IN_SNOWML_SPROC"
SNOWPARK_ML_PKG_NAME = "snowflake-ml-python"
SNOWFLAKE_CONDA_CHANNEL_URL = "https://repo.anaconda.com/pkgs/snowflake"


def _validate_pip_requirement_string(req_str: str) -> requirements.Requirement:
    """Validate the input pip requirement string according to PEP 508.

    Args:
        req_str: The string contains the pip requirement specification.

    Raises:
        ValueError: Raised when python is specified as a dependency.
        ValueError: Raised when invalid requirement string confronted.
        ValueError: Raised when requirement containing marker record confronted.

    Returns:
        A requirements.Requirement object containing the requirement information.
    """
    try:
        r = requirements.Requirement(req_str)

        if r.name == "python":
            raise ValueError("Don't specify python as a dependency, use python version argument instead.")
    except requirements.InvalidRequirement:
        raise ValueError(f"Invalid package requirement {req_str} found.")

    return r


def _validate_conda_dependency_string(dep_str: str) -> Tuple[str, requirements.Requirement]:
    """Validate conda dependency string like `pytorch == 1.12.1` or `conda-forge::transformer` and split the channel
        name before the double colon and requirement specification after that.

    Args:
        dep_str: The string contains the conda dependency specification.

    Raises:
        ValueError: Raised when invalid conda dependency containing extras record confronted.
        ValueError: Raised when invalid conda dependency containing url record confronted.
        ValueError: Raised when conda dependency operator ~= which is not supported by conda.

    Returns:
        A tuple containing the conda channel name and requirement.Requirement object showing requirement information.
    """
    channel_str, _, requirement_str = dep_str.rpartition("::")
    r = _validate_pip_requirement_string(requirement_str)
    if channel_str != "pip":
        if r.marker:
            raise ValueError("Markers is not supported in conda dependency.")
        if r.extras:
            raise ValueError("Extras is not supported in conda dependency.")
        if r.url:
            raise ValueError("Url is not supported in conda dependency.")
        for specifier in r.specifier:
            if specifier.operator == "~=":
                raise ValueError("Operator ~= is not supported in conda dependency.")
    return (channel_str, r)


def _check_if_requirement_same(req_a: requirements.Requirement, req_b: requirements.Requirement) -> bool:
    """Check if two requirements are the same package.

    Args:
        req_a: The first requirement.
        req_b: The second requirement.

    Returns:
        Whether two requirements should and need to be merged.
    """
    return req_a.name == req_b.name


class DuplicateDependencyError(Exception):
    ...


class DuplicateDependencyInMultipleChannelsError(Exception):
    ...


def append_requirement_list(req_list: List[requirements.Requirement], p_req: requirements.Requirement) -> None:
    """Append a requirement to an existing requirement list. If need and able to merge, merge it, otherwise, append it.

    Args:
        req_list: The target requirement list.
        p_req: The requirement to append.

    Raises:
        DuplicateDependencyError: Raised when the same package being added.
    """
    for req in req_list:
        if _check_if_requirement_same(p_req, req):
            raise DuplicateDependencyError(
                f"Found duplicate dependencies in pip requirements: {str(req)} and {str(p_req)}."
            )
    req_list.append(p_req)


def append_conda_dependency(
    conda_chan_deps: DefaultDict[str, List[requirements.Requirement]], p_chan_dep: Tuple[str, requirements.Requirement]
) -> None:
    """Append a conda dependency to an existing conda dependencies dict, if not existed in any channel.
        To avoid making unnecessary modification to dict, we check the existence first, then try to merge, then append,
        if still needed.

    Args:
        conda_chan_deps: The target dependencies dict.
        p_chan_dep: The tuple of channel and dependency to append.

    Raises:
        DuplicateDependencyError: Raised when the same package required from one channel.
        DuplicateDependencyInMultipleChannelsError: Raised when the same package required from different channels.
    """
    p_channel, p_req = p_chan_dep
    for chan_channel, chan_req_list in conda_chan_deps.items():
        for chan_req in chan_req_list:
            if _check_if_requirement_same(p_req, chan_req):
                if chan_channel != p_channel:
                    raise DuplicateDependencyInMultipleChannelsError(
                        "Found duplicate dependencies: "
                        + f"{str(chan_req)} in channel {chan_channel} and {str(p_req)} in channel {p_channel}."
                    )
                else:
                    raise DuplicateDependencyError(
                        f"Found duplicate dependencies in channel {chan_channel}: {str(chan_req)} and {str(p_req)}."
                    )
    conda_chan_deps[p_channel].append(p_req)


def validate_pip_requirement_string_list(req_str_list: List[str]) -> List[requirements.Requirement]:
    """Validate the a list of pip requirement string according to PEP 508.

    Args:
        req_str_list: The list of string contains the pip requirement specification.

    Returns:
        A requirements.Requirement list containing the requirement information.
    """
    seen_pip_requirement_list: List[requirements.Requirement] = []
    for req_str in req_str_list:
        append_requirement_list(seen_pip_requirement_list, _validate_pip_requirement_string(req_str=req_str))

    return seen_pip_requirement_list


def validate_conda_dependency_string_list(dep_str_list: List[str]) -> DefaultDict[str, List[requirements.Requirement]]:
    """Validate a list of conda dependency string, find any duplicate package across different channel and create a dict
        to represent the whole dependencies.

    Args:
        dep_str_list: The list of string contains the conda dependency specification.

    Returns:
        A dict mapping from the channel name to the list of requirements from that channel.
    """
    validated_conda_dependency_list = list(map(_validate_conda_dependency_string, dep_str_list))
    ret_conda_dependency_dict: DefaultDict[str, List[requirements.Requirement]] = collections.defaultdict(list)
    for p_channel, p_req in validated_conda_dependency_list:
        append_conda_dependency(ret_conda_dependency_dict, (p_channel, p_req))

    return ret_conda_dependency_dict


def get_local_installed_version_of_pip_package(pip_req: requirements.Requirement) -> requirements.Requirement:
    """Get the local installed version of a given pip package requirement.
        If the package is locally installed, and the local version meet the specifier of the requirements, return a new
        requirement specifier that pins the version.
        If the local version does not meet the specifier of the requirements, a warn will be omitted and returns
        the original package requirement.
        If the package is not locally installed or not found, the original package requirement is returned.

    Args:
        pip_req: A requirements.Requirement object showing the requirement.

    Returns:
        A requirements.Requirement object that might have version pinned to local installed version.
    """
    try:
        local_dist = importlib_metadata.distribution(pip_req.name)
        local_dist_version = local_dist.version
    except importlib_metadata.PackageNotFoundError:
        if pip_req.name == SNOWPARK_ML_PKG_NAME:
            local_dist_version = snowml_env.VERSION
        else:
            return pip_req
    new_pip_req = copy.deepcopy(pip_req)
    new_pip_req.specifier = specifiers.SpecifierSet(specifiers=f"=={version.parse(local_dist_version).base_version}")
    if not pip_req.specifier.contains(local_dist_version):
        warnings.warn(
            f"Package requirement {str(pip_req)} specified, while version {local_dist_version} is installed. "
            "Local version will be ignored to conform to package requirement.",
            stacklevel=2,
            category=UserWarning,
        )
        return pip_req
    return new_pip_req


class IncorrectLocalEnvironmentError(Exception):
    ...


def validate_local_installed_version_of_pip_package(pip_req: requirements.Requirement) -> None:
    """Validate if the package is locally installed, and the local version meet the specifier of the requirements.

    Args:
        pip_req: A requirements.Requirement object showing the requirement.

    Raises:
        IncorrectLocalEnvironmentError: Raised when cannot find the local installation of the requested package.
        IncorrectLocalEnvironmentError: Raised when the local installed version cannot meet the requirement.
    """
    try:
        local_dist = importlib_metadata.distribution(pip_req.name)
        local_dist_version = version.parse(local_dist.version)
    except importlib_metadata.PackageNotFoundError:
        raise IncorrectLocalEnvironmentError(f"Cannot find the local installation of the requested package {pip_req}.")

    if not pip_req.specifier.contains(local_dist_version):
        raise IncorrectLocalEnvironmentError(
            f"The local installed version {local_dist_version} cannot meet the requirement {pip_req}."
        )


CONDA_PKG_NAME_TO_PYPI_MAP = {"pytorch": "torch"}


def try_convert_conda_requirement_to_pip(conda_req: requirements.Requirement) -> requirements.Requirement:
    """Return a new requirements.Requirement object whose name has been attempted to convert to name in pypi from conda.

    Args:
        conda_req: A requirements.Requirement object showing the requirement in conda.

    Returns:
        A new requirements.Requirement object showing the requirement in pypi.
    """
    pip_req = copy.deepcopy(conda_req)
    pip_req.name = CONDA_PKG_NAME_TO_PYPI_MAP.get(conda_req.name, conda_req.name)
    return pip_req


def validate_py_runtime_version(provided_py_version_str: str) -> None:
    """Validate the provided python version string with python version in current runtime.
        If the major or minor is different, errors out.

    Args:
        provided_py_version_str: the provided python version string.

    Raises:
        IncorrectLocalEnvironmentError: Raised when the provided python version has different major or minor.
    """
    if provided_py_version_str != snowml_env.PYTHON_VERSION:
        provided_py_version = version.parse(provided_py_version_str)
        current_py_version = version.parse(snowml_env.PYTHON_VERSION)
        if (
            provided_py_version.major != current_py_version.major
            or provided_py_version.minor != current_py_version.minor
        ):
            raise IncorrectLocalEnvironmentError(
                f"Requested python version is {provided_py_version_str} "
                f"while current Python version is {snowml_env.PYTHON_VERSION}. "
            )


def get_package_spec_with_supported_ops_only(req: requirements.Requirement) -> requirements.Requirement:
    """Get the package spec with supported ops only including ==, >=, <=, > and <

    Args:
        req: A requirements.Requirement object showing the requirement.

    Returns:
        A requirements.Requirement object with supported ops only
    """
    new_req = copy.deepcopy(req)
    new_req.specifier = specifiers.SpecifierSet(
        specifiers=",".join([str(spec) for spec in req.specifier if spec.operator in _SUPPORTED_PACKAGE_SPEC_OPS])
    )
    return new_req


def relax_requirement_version(req: requirements.Requirement) -> requirements.Requirement:
    """Relax version specifier from a requirement. It detects any ==x.y.z in specifiers and replaced with
    >=x.y, <(x+1)

    Args:
        req: The requirement that version specifier to be removed.

    Returns:
        A new requirement object after relaxations.
    """
    new_req = copy.deepcopy(req)
    relaxed_specifier_set = set()
    for spec in new_req.specifier._specs:
        if spec.operator != "==":
            relaxed_specifier_set.add(spec)
            continue
        pinned_version = None
        try:
            pinned_version = version.parse(spec.version)
        except version.InvalidVersion:
            # For the case that the version string has * like 1.2.*
            relaxed_specifier_set.add(spec)
            continue
        assert pinned_version is not None
        relaxed_specifier_set.add(specifiers.Specifier(f">={pinned_version.major}.{pinned_version.minor}"))
        relaxed_specifier_set.add(specifiers.Specifier(f"<{pinned_version.major + 1}"))
    new_req.specifier._specs = frozenset(relaxed_specifier_set)
    return new_req


def get_matched_package_versions_in_snowflake_conda_channel(
    req: requirements.Requirement,
    python_version: str = snowml_env.PYTHON_VERSION,
    conda_os: CONDA_OS = CONDA_OS.LINUX_64,
) -> List[version.Version]:
    """Search the snowflake anaconda channel for packages that matches the specifier. Note that this will be the
    source of truth for checking whether a package indeed exists in Snowflake conda channel.

    Given that a package comes in different architectures, we only check for the Linux x86_64 architecture and assume
    the package exists in other architectures. If such an assumption does not hold true for a certain package, the
    caller should specify the architecture to search for.

    Args:
        req: Requirement specifier.
        python_version: A string of python version where model is run.
        conda_os: Specified platform to search availability of the package.

    Returns:
        List of package versions that meet the requirement specifier.
    """
    # Move the retryable_http import here as when UDF import this file, it won't have the "requests" dependency.
    from snowflake.ml._internal.utils import retryable_http

    assert not snowpark_utils.is_in_stored_procedure()  # type: ignore[no-untyped-call]

    url = f"{SNOWFLAKE_CONDA_CHANNEL_URL}/{conda_os.value}/repodata.json"

    if req.name not in _SNOWFLAKE_CONDA_PACKAGE_CACHE:
        try:
            http_client = retryable_http.get_http_client()
            parsed_python_version = version.Version(python_version)
            python_version_build_str = f"py{parsed_python_version.major}{parsed_python_version.minor}"
            repodata = http_client.get(url).json()
            assert isinstance(repodata, dict)
            packages_info = repodata["packages"]
            assert isinstance(packages_info, dict)
            version_list = [
                version.parse(package_info["version"])
                for package_info in packages_info.values()
                if package_info["name"] == req.name and python_version_build_str in package_info["build"]
            ]
            _SNOWFLAKE_CONDA_PACKAGE_CACHE[req.name] = version_list
        except Exception:
            pass

    matched_versions = list(req.specifier.filter(set(_SNOWFLAKE_CONDA_PACKAGE_CACHE.get(req.name, []))))
    return matched_versions


def get_matched_package_versions_in_information_schema_with_active_session(
    reqs: List[requirements.Requirement], python_version: str
) -> Dict[str, List[version.Version]]:
    try:
        session = context.get_active_session()
    except exceptions.SnowparkSessionException:
        return {}
    return get_matched_package_versions_in_information_schema(session, reqs, python_version)


def get_matched_package_versions_in_information_schema(
    session: session.Session, reqs: List[requirements.Requirement], python_version: str
) -> Dict[str, List[version.Version]]:
    """Look up the information_schema table to check if a package with the specified specifier exists in the Snowflake
    Conda channel. Note that this is not the source of truth due to the potential delay caused by a package that might
    exist in the information_schema table but has not yet become available in the Snowflake Conda channel.

    Args:
        session: Snowflake connection session.
        reqs: List of requirement specifiers.
        python_version: A string of python version where model is run.

    Returns:
        A Dict, whose key is the package name, and value is a list of versions match the requirements.
    """
    ret_dict: Dict[str, List[version.Version]] = {}
    reqs_to_request: List[requirements.Requirement] = []
    for req in reqs:
        if req.name in _SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE:
            available_versions = list(
                sorted(req.specifier.filter(set(_SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE.get(req.name, []))))
            )
            ret_dict[req.name] = available_versions
        else:
            reqs_to_request.append(req)

    if reqs_to_request:
        pkg_names_str = " OR ".join(
            f"package_name = '{req_name}'" for req_name in sorted(req.name for req in reqs_to_request)
        )

        parsed_python_version = version.Version(python_version)
        sql = textwrap.dedent(
            f"""
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE ({pkg_names_str})
            AND language = 'python'
            AND (runtime_version = '{parsed_python_version.major}.{parsed_python_version.minor}'
                OR runtime_version is null);
            """
        )

        try:
            result = (
                query_result_checker.SqlResultValidator(
                    session=session,
                    query=sql,
                )
                .has_column("VERSION")
                .has_dimensions(expected_rows=None, expected_cols=2)
                .validate()
            )
            for row in result:
                req_name = row["PACKAGE_NAME"]
                req_ver = version.parse(row["VERSION"])
                cached_req_ver_list = _SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE.get(req_name, [])
                cached_req_ver_list.append(req_ver)
                _SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE[req_name] = cached_req_ver_list
        except snowflake.connector.DataError:
            return ret_dict
    for req in reqs_to_request:
        available_versions = list(
            sorted(req.specifier.filter(set(_SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE.get(req.name, []))))
        )
        ret_dict[req.name] = available_versions
    return ret_dict


def save_conda_env_file(
    path: pathlib.Path,
    conda_chan_deps: DefaultDict[str, List[requirements.Requirement]],
    python_version: str,
    default_channel_override: str = SNOWFLAKE_CONDA_CHANNEL_URL,
) -> None:
    """Generate conda.yml file given a dict of dependencies after validation.
    The channels part of conda.yml file will contains Snowflake Anaconda Channel, nodefaults and all channel names
    in keys of the dict, ordered by the number of the packages which belongs to.
    The dependencies part of conda.yml file will contains requirements specifications. If the requirements is in the
    value list whose key is DEFAULT_CHANNEL_NAME, then the channel won't be specified explicitly. Otherwise, it will be
    specified explicitly via {channel}::{requirement} format.

    Args:
        path: Path to the conda.yml file.
        conda_chan_deps: Dict of conda dependencies after validated.
        python_version: A string 'major.minor' showing python version relate to model.
        default_channel_override: The default channel to be put in the first place of the channels section.
    """
    assert path.suffix in [".yml", ".yaml"], "Conda environment file should have extension of yml or yaml."
    path.parent.mkdir(parents=True, exist_ok=True)
    env: Dict[str, Any] = dict()
    env["name"] = "snow-env"
    # Get all channels in the dependencies, ordered by the number of the packages which belongs to and put into
    # channels section.
    channels = list(dict(sorted(conda_chan_deps.items(), key=lambda item: len(item[1]), reverse=True)).keys())
    if DEFAULT_CHANNEL_NAME in channels:
        channels.remove(DEFAULT_CHANNEL_NAME)

    if default_channel_override in channels:
        channels.remove(default_channel_override)

    env["channels"] = [default_channel_override] + channels + [_NODEFAULTS]
    env["dependencies"] = [f"python=={python_version}.*"]
    for chan, reqs in conda_chan_deps.items():
        env["dependencies"].extend(
            [f"{chan}::{str(req)}" if chan != DEFAULT_CHANNEL_NAME else str(req) for req in reqs]
        )

    with open(path, "w", encoding="utf-8") as f:
        yaml.SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[method-assign]
        yaml.safe_dump(env, stream=f, default_flow_style=False)


def save_requirements_file(path: pathlib.Path, pip_deps: List[requirements.Requirement]) -> None:
    """Generate Python requirements.txt file in the given directory path.

    Args:
        path: Path to the requirements.txt file.
        pip_deps: List of dependencies string after validated.
    """
    assert path.suffix in [".txt"], "PIP requirement file should have extension of txt."
    path.parent.mkdir(parents=True, exist_ok=True)
    requirements = "\n".join(map(str, pip_deps))
    with open(path, "w", encoding="utf-8") as out:
        out.write(requirements)


def load_conda_env_file(
    path: pathlib.Path,
) -> Tuple[DefaultDict[str, List[requirements.Requirement]], Optional[List[requirements.Requirement]], Optional[str]]:
    """Read conda.yml file to get a dict of dependencies after validation.
    The channels part of conda.yml file will be processed with following rules:
    1. If it is Snowflake Anaconda Channel, ignore as it is default.
    2. If it is nodefaults channel (meta-channel), ignore as we don't need it.
    3. If it is a channel where its name does not appear in the dependencies section, added into the dict as key where
        value is an empty list.
    4. If it is a channel where its name appears in the dependencies section as {channel}::{requirements}, remove it
        as when parsing the requirements part it will be added into the dict as a key.

    The dependencies part of conda.yml file will be processed as follows.
    If the requirements has no channel specified, it will be stored in the bucket of DEFAULT_CHANNEL_NAME
    If the requirements is specified explicitly via {channel}::{requirement} format, it will be stored into
        corresponding bucket in the dict.
    If the requirement is a dict whose key is "pip", it will be parsed as pip requirements.
    If the requirement has the name "python", it will be parsed as python version requirement.

    Args:
        path: Path to conda.yml.

    Raises:
        ValueError: Raised when the requirement has an unsupported or unknown type.

    Returns:
        A tuple of Dict of conda dependencies after validated, optional pip requirements if exist
        and a string 'major.minor.patchlevel' of python version.
    """
    if not path.exists():
        return collections.defaultdict(list), None, None

    with open(path, encoding="utf-8") as f:
        env = yaml.safe_load(stream=f)

    assert isinstance(env, dict)

    deps = []
    pip_deps = []

    python_version = None

    channels = env.get("channels", [])
    if len(channels) >= 1:
        channels = channels[1:]  # Skip the first channel which is the default channel
    if _NODEFAULTS in channels:
        channels.remove(_NODEFAULTS)

    for dep in env["dependencies"]:
        if isinstance(dep, str):
            ver = parse_python_version_string(dep)
            # ver is None: not python
            # ver is "": python w/o specifier
            # ver is str: python w/ specifier
            if ver:
                python_version = ver
            elif ver is None:
                deps.append(dep)
        elif isinstance(dep, dict) and "pip" in dep:
            pip_deps.extend(dep["pip"])
        else:
            raise ValueError(f"Unable to parse the conda.yml file, confronting unsupported type of requirement {dep}")

    conda_dep_dict = validate_conda_dependency_string_list(deps)
    pip_deps_list = validate_pip_requirement_string_list(pip_deps)

    for channel in channels:
        if channel not in conda_dep_dict:
            conda_dep_dict[channel] = []

    return conda_dep_dict, pip_deps_list if pip_deps_list else None, python_version


def load_requirements_file(path: pathlib.Path) -> List[requirements.Requirement]:
    """Load Python requirements.txt file from the given directory path.

    Args:
        path: Path to the requirements.txt file.

    Returns:
        List of dependencies string after validated.
    """
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        reqs = f.readlines()

    return validate_pip_requirement_string_list(reqs)


# We have to use re to support MLFlow generated python string, which use = rather than ==
PYTHON_VERSION_PATTERN = re.compile(r"python(?:(?P<op>=|==|>|<|>=|<=|~=|===)(?P<ver>\d(?:\.\d+)+))?(\.*)?")


def parse_python_version_string(dep: str) -> Optional[str]:
    if dep.startswith("python"):
        m = PYTHON_VERSION_PATTERN.match(dep)
        if m is None:
            return None
        op = m.group("op")
        if op and (op != "=" and op != "=="):
            raise ValueError("Unsupported operator for python version specifier.")
        ver = m.group("ver")
        if ver:
            return ver
        else:
            # "python" only, no specifier
            return ""
    return None


def _find_conda_dep_spec(
    conda_chan_deps: DefaultDict[str, List[requirements.Requirement]], pkg_name: str
) -> Optional[Tuple[str, requirements.Requirement]]:
    for channel in conda_chan_deps:
        spec = next(filter(lambda req: req.name == pkg_name, conda_chan_deps[channel]), None)
        if spec:
            return channel, spec
    return None


def _find_pip_req_spec(pip_reqs: List[requirements.Requirement], pkg_name: str) -> Optional[requirements.Requirement]:
    spec = next(filter(lambda req: req.name == pkg_name, pip_reqs), None)
    return spec


def find_dep_spec(
    conda_chan_deps: DefaultDict[str, List[requirements.Requirement]],
    pip_reqs: List[requirements.Requirement],
    conda_pkg_name: str,
    pip_pkg_name: Optional[str] = None,
    remove_spec: bool = False,
) -> Optional[requirements.Requirement]:
    if pip_pkg_name is None:
        pip_pkg_name = conda_pkg_name
    spec_conda = _find_conda_dep_spec(conda_chan_deps, conda_pkg_name)
    if spec_conda:
        channel, spec = spec_conda
        if remove_spec:
            conda_chan_deps[channel].remove(spec)
        return spec
    else:
        spec_pip = _find_pip_req_spec(pip_reqs, pip_pkg_name)
        if spec_pip:
            if remove_spec:
                pip_reqs.remove(spec_pip)
            return spec_pip
    return None
