import inspect
import sys
from typing import Dict, List, Optional

from snowflake.ml._internal import telemetry
from snowflake.snowpark import DataFrame, Session

cache: Dict[str, bool] = {}

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "utils"

_RUNTIME_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


def validate_pkg_versions_supported_in_snowflake_conda_channel(
    pkg_versions: List[str], session: Session, subproject: Optional[str] = None
) -> None:
    for pkg_version in pkg_versions:
        if not _validate_pkg_version_supported_in_snowflake_conda_channel(
            pkg_version=pkg_version, session=session, subproject=subproject
        ):
            raise RuntimeError(
                f"Package {pkg_version} is not supported in snowflake conda channel for "
                f"python runtime {_RUNTIME_VERSION}."
            )


def _validate_pkg_version_supported_in_snowflake_conda_channel(
    pkg_version: str, session: Session, subproject: Optional[str] = None
) -> bool:
    if pkg_version not in cache:
        tokens = pkg_version.split("==")
        if len(tokens) != 2:
            raise RuntimeError(
                "Expected package name and versions to specified in format "
                f"'<pkg_name>==<version>', but found {pkg_version}"
            )
        pkg_name = tokens[0]
        version = tokens[1]

        sql = f"""SELECT *
                    FROM information_schema.packages
                    WHERE package_name = '{pkg_name}'
                    AND version = '{version}'"""
        result_df = session.sql(sql)

        # TODO(snandamuri): Move this filter into main SQL query after BCR 7.19 is completed.
        if "RUNTIME_VERSION" in result_df.columns:
            result_df = result_df.filter(f"RUNTIME_VERSION = {_RUNTIME_VERSION}")

        num_rows = result_df.count(
            statement_params=telemetry.get_function_usage_statement_params(
                project=_PROJECT,
                subproject=subproject or _SUBPROJECT,
                function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe()),
                api_calls=[DataFrame.count],
            )
        )
        cache[pkg_version] = num_rows >= 1
    return cache[pkg_version]
