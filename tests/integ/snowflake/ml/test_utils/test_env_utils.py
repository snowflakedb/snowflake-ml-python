import functools
import textwrap
from typing import List

from packaging import requirements, version

import snowflake.connector
from snowflake.ml._internal import env, env_utils
from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import session


def get_current_snowflake_version(session: session.Session) -> version.Version:
    res = session.sql("SELECT CURRENT_VERSION() AS CURRENT_VERSION").collect()[0]
    version_str = res.CURRENT_VERSION
    assert isinstance(version_str, str)

    version_str = "+".join(version_str.split())
    return version.parse(version_str)


@functools.lru_cache
def get_package_versions_in_server(
    session: session.Session,
    package_req_str: str,
    python_version: str = env.PYTHON_VERSION,
) -> List[version.Version]:
    package_req = requirements.Requirement(package_req_str)
    parsed_python_version = version.Version(python_version)
    sql = textwrap.dedent(
        f"""
        SELECT PACKAGE_NAME, VERSION
        FROM information_schema.packages
        WHERE package_name = '{package_req.name}'
        AND language = 'python'
        AND runtime_version = '{parsed_python_version.major}.{parsed_python_version.minor}';
        """
    )

    version_list = []
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
            req_ver = version.parse(row["VERSION"])
            version_list.append(req_ver)
    except snowflake.connector.DataError:
        return []
    available_version_list = list(package_req.specifier.filter(version_list))
    return available_version_list


@functools.lru_cache
def get_latest_package_version_spec_in_server(
    session: session.Session,
    package_req_str: str,
    python_version: str = env.PYTHON_VERSION,
) -> str:
    package_req = requirements.Requirement(package_req_str)
    available_version_list = get_package_versions_in_server(session, package_req_str, python_version)
    if len(available_version_list) == 0:
        return str(package_req)
    return f"{package_req.name}=={max(available_version_list)}"


@functools.lru_cache
def get_latest_package_version_spec_in_conda(package_req_str: str, python_version: str = env.PYTHON_VERSION) -> str:
    package_req = requirements.Requirement(package_req_str)
    available_version_list = env_utils.get_matched_package_versions_in_snowflake_conda_channel(
        req=requirements.Requirement(package_req_str), python_version=python_version
    )
    if len(available_version_list) == 0:
        return str(package_req)
    return f"{package_req.name}=={max(available_version_list)}"
