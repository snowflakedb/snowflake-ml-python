import http
import logging
from datetime import timedelta
from typing import Optional

import requests
from cryptography.hazmat.primitives.asymmetric import types
from requests import auth

from snowflake.ml._internal.utils import jwt_generator

logger = logging.getLogger(__name__)
_JWT_TOKEN_CACHE: dict[str, dict[int, str]] = {}


def get_jwt_token_generator(
    account: str,
    user: str,
    private_key: types.PRIVATE_KEY_TYPES,
    lifetime: Optional[timedelta] = None,
    renewal_delay: Optional[timedelta] = None,
) -> jwt_generator.JWTGenerator:
    return jwt_generator.JWTGenerator(account, user, private_key, lifetime=lifetime, renewal_delay=renewal_delay)


def _get_snowflake_token_by_jwt(
    jwt_token_generator: jwt_generator.JWTGenerator,
    account: Optional[str] = None,
    role: Optional[str] = None,
    endpoint: Optional[str] = None,
    snowflake_account_url: Optional[str] = None,
) -> str:
    scope_role = f"session:role:{role}" if role is not None else None
    scope = " ".join(filter(None, [scope_role, endpoint]))
    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "scope": scope or None,
        "assertion": jwt_token_generator.get_token(),
    }
    account = account or jwt_token_generator.account
    url = f"https://{account}.snowflakecomputing.com/oauth/token"
    if snowflake_account_url:
        url = f"{snowflake_account_url}/oauth/token"

    cache_key = hash(frozenset(data.items()))
    if url in _JWT_TOKEN_CACHE:
        if cache_key in _JWT_TOKEN_CACHE[url]:
            return _JWT_TOKEN_CACHE[url][cache_key]
    else:
        _JWT_TOKEN_CACHE[url] = {}

    response = requests.post(url, data=data)
    if response.status_code != http.HTTPStatus.OK:
        raise RuntimeError(f"Failed to get snowflake token: {response.status_code} {response.content!r}")
    auth_token = response.text
    _JWT_TOKEN_CACHE[url][cache_key] = auth_token
    return auth_token


class SnowflakeJWTTokenAuth(auth.AuthBase):
    def __init__(
        self,
        jwt_token_generator: jwt_generator.JWTGenerator,
        account: Optional[str] = None,
        role: Optional[str] = None,
        endpoint: Optional[str] = None,
        snowflake_account_url: Optional[str] = None,
    ) -> None:
        self.snowflake_token = _get_snowflake_token_by_jwt(
            jwt_token_generator, account, role, endpoint, snowflake_account_url
        )

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers["Authorization"] = f'Snowflake Token="{self.snowflake_token}"'
        return r


class SnowflakePATAuth(auth.AuthBase):
    """Authentication using Snowflake Programmatic Access Token (PAT)."""

    def __init__(self, pat_token: str) -> None:
        """Initialize with a PAT token.

        Args:
            pat_token: The programmatic access token string
        """
        self.pat_token = pat_token

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers["Authorization"] = f'Snowflake Token="{self.pat_token}"'
        return r
