import functools

import requests


@functools.lru_cache
def get_snowpark_ml_released_versions() -> list[str]:
    pypi_url = "https://pypi.org/pypi/snowflake-ml-python/json"
    pypi_resp = requests.get(pypi_url).json()
    return list(pypi_resp["releases"].keys())
