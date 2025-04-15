import sys
import warnings
from typing import Optional, Union

from packaging.version import Version

from snowflake.ml._internal import telemetry
from snowflake.snowpark import AsyncJob, Row, Session
from snowflake.snowpark._internal import utils as snowpark_utils

cache: dict[str, Optional[str]] = {}

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "utils"

_RUNTIME_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

_relax_version: bool = False


def is_relaxed() -> bool:
    return _relax_version


def get_valid_pkg_versions_supported_in_snowflake_conda_channel(
    pkg_versions: list[str], session: Session, subproject: Optional[str] = None
) -> list[str]:
    if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
        return pkg_versions
    else:
        return _get_valid_pkg_versions_supported_in_snowflake_conda_channel_async(pkg_versions, session, subproject)


def _get_valid_pkg_versions_supported_in_snowflake_conda_channel_async(
    pkg_versions: list[str], session: Session, subproject: Optional[str] = None
) -> list[str]:
    pkg_version_async_job_list: list[tuple[str, AsyncJob]] = []
    for pkg_version in pkg_versions:
        if pkg_version not in cache:
            # Execute pkg version queries asynchronously.
            async_job = _query_pkg_version_supported_in_snowflake_conda_channel(
                pkg_version=pkg_version, session=session, block=False, subproject=subproject
            )
            if isinstance(async_job, list):
                raise RuntimeError(
                    "Async job was expected, executed query was returned. Please contact Snowflake support."
                )

            pkg_version_async_job_list.append((pkg_version, async_job))

    # Populate the cache.
    for pkg_version, pkg_version_async_job in pkg_version_async_job_list:
        pkg_version_list = pkg_version_async_job.result()
        assert isinstance(pkg_version_list, list)
        try:
            cache[pkg_version] = pkg_version_list[0]["VERSION"]
        except IndexError:
            cache[pkg_version] = None

    pkg_version_conda_list = _get_conda_packages_and_emit_warnings(pkg_versions)

    return pkg_version_conda_list


def _query_pkg_version_supported_in_snowflake_conda_channel(
    pkg_version: str, session: Session, block: bool, subproject: Optional[str] = None
) -> Union[AsyncJob, list[Row]]:
    tokens = pkg_version.split("==")
    if len(tokens) != 2:
        raise RuntimeError(
            "Expected package name and versions to specified in format "
            f"'<pkg_name>==<version>', but found {pkg_version}"
        )
    pkg_name = tokens[0]
    version = Version(tokens[1])
    major_version, minor_version, micro_version = version.major, version.minor, version.micro

    # relax version control - only major_version.minor_version.* will be enforced.
    # the result would be ordered by, the version that closest to user's version, and the latest.
    sql = f"""
        SELECT PACKAGE_NAME, VERSION, LANGUAGE
        FROM (
            SELECT *,
            SUBSTRING(VERSION, LEN(VERSION) - CHARINDEX('.', REVERSE(VERSION)) + 2, LEN(VERSION)) as micro_version
            FROM information_schema.packages
            WHERE package_name = '{pkg_name}'
            AND version LIKE '{major_version}.{minor_version}.%'
            ORDER BY abs({micro_version}-micro_version), -micro_version
        )
        """
    result_df = session.sql(sql)

    # TODO(snandamuri): Move this filter into main SQL query after BCR 7.19 is completed.
    if "RUNTIME_VERSION" in result_df.columns:
        result_df = result_df.filter(f"RUNTIME_VERSION = {_RUNTIME_VERSION}")

    pkg_version_list_or_async_job = result_df.collect(
        statement_params=telemetry.get_statement_params(_PROJECT, subproject or _SUBPROJECT),
        block=block,
    )

    return pkg_version_list_or_async_job


def _get_conda_packages_and_emit_warnings(pkg_versions: list[str]) -> list[str]:
    pkg_version_conda_list: list[str] = []
    pkg_version_warning_list: list[list[str]] = []
    for pkg_version in pkg_versions:
        try:
            conda_pkg_version = cache[pkg_version]
        except KeyError:
            conda_pkg_version = None

        if conda_pkg_version is None:
            if _relax_version:
                pkg_version_warning_list.append([pkg_version, _RUNTIME_VERSION])
            else:
                raise RuntimeError(
                    f"Package {pkg_version} is not supported in snowflake conda channel for "
                    f"python runtime {_RUNTIME_VERSION}."
                )
        else:
            tokens = pkg_version.split("==")
            pkg_name = tokens[0]
            pkg_version_conda_list.append(f"{pkg_name}=={conda_pkg_version}")

    if pkg_version_warning_list:
        warnings.warn(
            f"Package {', '.join([pkg[0] for pkg in pkg_version_warning_list])} is not supported "
            f"in snowflake conda channel for python runtime "
            f"{', '.join([pkg[1] for pkg in pkg_version_warning_list])}.",
            stacklevel=1,
        )

    return pkg_version_conda_list
