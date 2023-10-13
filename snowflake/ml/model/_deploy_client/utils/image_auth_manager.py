import dataclasses
from abc import abstractmethod
from typing import Optional

from snowflake.ml._internal.utils import retryable_http

http = retryable_http.get_http_client()


class AuthManager:
    """
    An interface to support fetching tokens for registries.
    This can be subclassed to provide custom implementations.
    """

    @abstractmethod
    def get_auth_token(self, spcs_token: Optional[str] = None) -> str:
        """Returns a bearer token for the registry. Use the spcs registry cred to authenticate.
        The details of generation and lifetime management/caching are left to the
        implementation.

        Args:
            spcs_token: session token from SPCS image registry.

        """
        pass


@dataclasses.dataclass
class SnowflakeAuthManager(AuthManager):
    """
    Implements authentication against Snowflake image registry.
    """

    registry_host: str

    def get_auth_token(self, spcs_token: Optional[str] = None) -> str:
        """Get bearer token by authenticating to registry with the given spcs session token.

        Args:
            spcs_token: session token from SPCS image registry.

        Returns:
            str: A bearer token from Docker API.

        """
        assert spcs_token is not None
        resp = http.get(url=f"https://{self.registry_host}/login", headers={"authorization": f"Basic {spcs_token}"})
        assert resp.status_code == 200, f"login failed with code {resp.status_code} and message {str(resp.content)}"
        return str(resp.json()["token"])
