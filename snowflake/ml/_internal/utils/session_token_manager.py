import base64
import json
from typing import TypedDict

from snowflake import snowpark


class SessionToken(TypedDict):
    token: str
    expires_in: str


class SessionTokenManager:
    def __init__(self, session: snowpark.Session) -> None:
        self._session = session

    def get_session_token(self) -> SessionToken:
        """
        This function retrieves the session token from Snowpark session object.

        Returns:
            The session token string value.
        """
        ctx = self._session._conn._conn
        assert ctx._rest, "SnowflakeRestful is not set in session"
        token_data = ctx._rest._token_request("ISSUE")
        session_token = token_data["data"]["sessionToken"]
        validity_in_seconds = token_data["data"]["validityInSecondsST"]
        assert session_token, "session_token is not obtained successfully from the session object"
        assert validity_in_seconds, "validityInSecondsST is not obtained successfully from the session object"
        return {"token": session_token, "expires_in": validity_in_seconds}

    def get_base64_encoded_token(self, username: str = "0sessiontoken") -> str:
        """This function returns the base64 encoded username:password, which is compatible with registry, such as
        SnowService image registry, that uses Docker credential helper. In this case, password will be session token.

        Args:
            username: username for authentication.

        Returns:
            base64 encoded credential string.

        """
        credentials = f"{username}:{json.dumps(self.get_session_token())}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        return encoded_credentials
