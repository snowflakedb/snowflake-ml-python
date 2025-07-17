import functools

from packaging import requirements, version

from snowflake.ml._internal import env, env_utils
from snowflake.ml._internal.utils import connection_params, snowflake_env
from snowflake.snowpark import session
from snowflake.snowpark._internal import utils as snowpark_utils


def get_available_session() -> session.Session:
    return (
        session._get_active_session()
        if snowpark_utils.is_in_stored_procedure()  # type: ignore[no-untyped-call] #
        else session.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
    )


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
