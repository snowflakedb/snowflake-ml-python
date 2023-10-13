import functools
import textwrap

import requests
from packaging import version

import snowflake.connector
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import session


@functools.lru_cache
def get_latest_package_versions_in_server(
    session: session.Session, package_name: str, python_version: str = env.PYTHON_VERSION
) -> str:
    parsed_python_version = version.Version(python_version)
    sql = textwrap.dedent(
        f"""
        SELECT PACKAGE_NAME, VERSION
        FROM information_schema.packages
        WHERE package_name = '{package_name}'
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
        return package_name
    if len(version_list) == 0:
        return package_name
    return f"{package_name}=={max(version_list)}"


@functools.lru_cache
def get_latest_package_versions_in_conda(package_name: str, python_version: str = env.PYTHON_VERSION) -> str:
    repodata_url = "https://repo.anaconda.com/pkgs/snowflake/linux-64/repodata.json"

    parsed_python_version = version.Version(python_version)
    python_version_build_str = f"py{parsed_python_version.major}{parsed_python_version.minor}"

    max_retry = 3

    exc_list = []

    while max_retry > 0:
        try:
            version_list = []
            repodata = requests.get(repodata_url).json()
            assert isinstance(repodata, dict)
            packages_info = repodata["packages"]
            assert isinstance(packages_info, dict)
            for package_info in packages_info.values():
                if package_info["name"] == package_name and python_version_build_str in package_info["build"]:
                    version_list.append(version.parse(package_info["version"]))
            return f"{package_name}=={str(max(version_list))}"
        except Exception as e:
            max_retry -= 1
            exc_list.append(e)

    raise RuntimeError(
        f"Failed to get latest version of package {package_name} in Snowflake Anaconda Channel. "
        + "Exceptions are "
        + ", ".join(map(str, exc_list))
    )
