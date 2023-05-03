from typing import List, Optional

from snowflake import connector
from snowflake.ml._internal.utils import query_result_checker
from snowflake.ml.utils import telemetry
from snowflake.snowpark import Session

cache = {}

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "utils"


def validate_pkg_versions_supported_in_snowflake_conda_channel(
    pkg_versions: List[str], session: Session, subproject: Optional[str] = None
) -> None:
    for pkg_version in pkg_versions:
        if not _validate_pkg_version_supported_in_snowflake_conda_channel(
            pkg_version=pkg_version, session=session, subproject=subproject
        ):
            raise RuntimeError(f"Package {pkg_version} is not supported in snowflake conda channel.")


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
                    AND version = '{version}';"""
        result = session.sql(sql).collect(
            statement_params=telemetry.get_function_usage_statement_params(
                project=_PROJECT, subproject=subproject or _SUBPROJECT
            )
        )
        try:
            cache[pkg_version] = query_result_checker.result_dimension_matcher(
                expected_rows=1, expected_cols=3, result=result, sql=sql
            )
        except connector.DataError:
            cache[pkg_version] = False

    return cache[pkg_version]
