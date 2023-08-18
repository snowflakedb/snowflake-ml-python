#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import functools
import importlib
import textwrap

from packaging import version

import snowflake.connector
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import session


@functools.lru_cache
def is_in_pip_env() -> bool:
    try:
        importlib.import_module("conda")
        return False
    except ModuleNotFoundError:
        return True


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
