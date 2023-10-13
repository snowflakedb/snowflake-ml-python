import http

import requests
from requests import adapters
from urllib3.util import retry


def get_http_client() -> requests.Session:
    # Set up a retry policy for requests
    retry_strategy = retry.Retry(
        total=3,  # total number of retries
        backoff_factor=0.1,  # 100ms initial delay
        status_forcelist=[
            http.HTTPStatus.TOO_MANY_REQUESTS,
            http.HTTPStatus.INTERNAL_SERVER_ERROR,
            http.HTTPStatus.BAD_GATEWAY,
            http.HTTPStatus.SERVICE_UNAVAILABLE,
            http.HTTPStatus.GATEWAY_TIMEOUT,
        ],  # retry on these status codes
    )

    adapter = adapters.HTTPAdapter(max_retries=retry_strategy)
    req_session = requests.Session()
    req_session.mount("https://", adapter)
    req_session.mount("http://", adapter)

    return req_session
