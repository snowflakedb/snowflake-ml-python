import http

import requests
from requests import adapters
from urllib3.util import retry


def get_http_client(total_retries: int = 5, backoff_factor: float = 0.1) -> requests.Session:
    """Construct retryable http client.

    Args:
        total_retries: Total number of retries to allow.
        backoff_factor: A backoff factor to apply between attempts after the second try. Time to sleep is calculated by
            {backoff factor} * (2 ** ({number of previous retries})). For example, with default retries of 5 and backoff
            factor set to 0.1, each subsequent retry will sleep [0.2s, 0.4s, 0.8s, 1.6s, 3.2s] respectively.

    Returns:
        requests.Session object.

    """

    retry_strategy = retry.Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
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
