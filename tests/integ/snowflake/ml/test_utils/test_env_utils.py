import functools
import os
from typing import Any, Mapping, Union

from packaging import requirements, version

from snowflake.ml._internal import env, env_utils
from snowflake.ml._internal.utils import connection_params, snowflake_env
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.snowpark import session
from snowflake.snowpark._internal import utils as snowpark_utils


def _ensure_session_has_warehouse(sess: session.Session, login_opts: Mapping[str, Any]) -> None:
    """Activate a warehouse when the connector left the session without one (common if omitted from snowsql config)."""
    if sess.get_current_warehouse():
        return
    wh = os.getenv("SNOWFLAKE_WAREHOUSE", "").strip()
    if not wh:
        raw = login_opts.get("warehouse")
        if isinstance(raw, str):
            wh = raw.strip()
        elif raw is not None:
            wh = str(raw).strip()
    if not wh:
        raise RuntimeError(
            'Snowpark session has no active warehouse. Add `warehouse = "..."` to the active '
            "[connections.*] section in ~/.snowsql/config, or set environment variable "
            "SNOWFLAKE_WAREHOUSE before running tests."
        )
    sess.sql(f"USE WAREHOUSE {SqlIdentifier(wh)}").collect()


def get_available_session() -> session.Session:
    if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
        return session._get_active_session()
    opts: dict[str, Union[str, bytes]] = dict(connection_params.SnowflakeLoginOptions())
    sess = session.Session.builder.configs(opts).create()
    _ensure_session_has_warehouse(sess, opts)
    return sess


@functools.lru_cache
def get_current_snowflake_version() -> version.Version:
    with get_available_session() as sess:
        return snowflake_env.get_current_snowflake_version(sess)


@functools.lru_cache
def get_current_snowflake_cloud_type() -> snowflake_env.SnowflakeCloudType:
    region = get_current_snowflake_region()
    return region["cloud"]


@functools.lru_cache
def get_current_snowflake_region() -> snowflake_env.SnowflakeRegion:
    with get_available_session() as sess:
        return snowflake_env.get_regions(sess)[snowflake_env.get_current_region_id(sess)]


@functools.lru_cache
def get_latest_package_version_spec_in_server(
    sess: session.Session,
    package_req_str: str,
    python_version: str = env.PYTHON_VERSION,
) -> str:
    package_req = requirements.Requirement(package_req_str)
    available_version_list = env_utils.get_matched_package_versions_in_information_schema(
        sess, [package_req], python_version
    ).get(package_req.name, [])
    if len(available_version_list) == 0:
        return str(package_req)
    return f"{package_req.name}=={max(available_version_list)}"
