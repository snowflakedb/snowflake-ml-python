import functools

import requests


@functools.lru_cache
def get_snowpark_ml_released_versions() -> list[str]:
    releases_url = "https://api.github.com/repos/snowflakedb/snowflake-ml-python/releases"
    releases_resp = requests.get(releases_url).json()
    return [rel["tag_name"] for rel in releases_resp]
