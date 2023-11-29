import http
import json
import logging
import time
from typing import Any, Callable, Dict, FrozenSet, Optional
from urllib.parse import urlparse, urlunparse

import requests

from snowflake import snowpark
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import retryable_http, session_token_manager

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY_SECONDS = 1
_RETRYABLE_HTTP_CODE = frozenset([http.HTTPStatus.UNAUTHORIZED])


def retry_on_error(
    http_call_function: Callable[..., requests.Response],
    retryable_http_code: FrozenSet[http.HTTPStatus] = _RETRYABLE_HTTP_CODE,
) -> Callable[..., requests.Response]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        retry_delay_seconds = _RETRY_DELAY_SECONDS
        for attempt in range(1, _MAX_RETRIES + 1):
            resp = http_call_function(*args, **kwargs)
            if resp.status_code in retryable_http_code:
                logger.warning(
                    f"Received {resp.status_code} status code. Retrying " f"(attempt {attempt}/{_MAX_RETRIES})..."
                )
                time.sleep(retry_delay_seconds)
                retry_delay_seconds *= 2  # Increase the retry delay exponentially
                if attempt < _MAX_RETRIES:
                    assert isinstance(args[0], ImageRegistryHttpClient)
                    args[0]._fetch_bearer_token()
            else:
                return resp

            if attempt == _MAX_RETRIES:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
                    original_exception=RuntimeError(
                        f"Failed to authenticate to registry after max retries {attempt} \n"
                        f"Status {resp.status_code},"
                        f"{str(resp.text)}"
                    ),
                )

    return wrapper


class ImageRegistryHttpClient:
    """
    An image registry HTTP client utilizes a retryable HTTP client underneath. Its primary function is to facilitate
    re-authentication with the image registry by obtaining a new GS token, which is then used to acquire a new bearer
    token for subsequent HTTP request authentication.

    Ideally you should not use this client directly. Please use ImageRegistryClient for image registry-specific
    operations. For general use of a retryable HTTP client, consider using the "retryable_http" module.
    """

    def __init__(self, *, session: snowpark.Session, repo_url: str) -> None:
        self._repo_url = repo_url
        self._session_token_manager = session_token_manager.SessionTokenManager(session)
        self._retryable_http = retryable_http.get_http_client()
        self._bearer_token = ""

    def _with_bearer_token_header(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if not self._bearer_token:
            self._fetch_bearer_token()
        assert self._bearer_token
        new_headers = {} if not headers else headers.copy()
        new_headers["Authorization"] = f"Bearer {self._bearer_token}"
        return new_headers

    def _fetch_bearer_token(self) -> None:
        resp = self._login()
        self._bearer_token = str(json.loads(resp.text)["token"])

    def _login(self) -> requests.Response:
        """Log in to image registry. repo_url is expected to set when _login function is invoked.

        Returns:
            Bearer token when login succeeded.
        """
        parsed_url = urlparse(self._repo_url)
        scheme = parsed_url.scheme
        host = parsed_url.netloc

        login_path = "/login"  # Construct the login path
        url_tuple = (scheme, host, login_path, "", "", "")
        login_url = urlunparse(url_tuple)

        base64_encoded_token = self._session_token_manager.get_base64_encoded_token()
        return self._retryable_http.get(login_url, headers={"Authorization": f"Basic {base64_encoded_token}"})

    @retry_on_error
    def head(self, api_url: str, *, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        return self._retryable_http.head(api_url, headers=self._with_bearer_token_header(headers))

    @retry_on_error
    def get(self, api_url: str, *, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        return self._retryable_http.get(api_url, headers=self._with_bearer_token_header(headers))

    @retry_on_error
    def put(self, api_url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> requests.Response:
        return self._retryable_http.put(api_url, headers=self._with_bearer_token_header(headers), **kwargs)

    @retry_on_error
    def post(self, api_url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> requests.Response:
        return self._retryable_http.post(api_url, headers=self._with_bearer_token_header(headers), **kwargs)

    @retry_on_error
    def patch(self, api_url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> requests.Response:
        return self._retryable_http.patch(api_url, headers=self._with_bearer_token_header(headers), **kwargs)
