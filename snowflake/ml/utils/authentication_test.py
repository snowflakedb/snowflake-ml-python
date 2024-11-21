from unittest import mock

import requests
from absl.testing import absltest

from snowflake.ml.utils import authentication


class AuthenticationTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_jwt_generator = mock.MagicMock()
        self.m_jwt_generator.get_token.return_value = "jwt_token"
        authentication._JWT_TOKEN_CACHE = {}

    def test_get_jwt_token_default_account(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator)
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_error(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 404
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            with self.assertRaisesRegex(RuntimeError, "Failed to get snowflake token"):
                authentication._get_snowflake_token_by_jwt(self.m_jwt_generator)
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_overridden_account(self) -> None:
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator, account="account")
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_role(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator, role="role")
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": "session:role:role",
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_endpoint(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator, endpoint="endpoint")
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": "endpoint",
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_role_and_endpoint(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator, role="role", endpoint="endpoint")
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": "session:role:role endpoint",
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_account_url(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(
                self.m_jwt_generator, snowflake_account_url="https://account.url"
            )
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.url/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )

    def test_get_jwt_token_cache_hit(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator)
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator)
            self.assertEqual(token, "auth_token")
            m_post.assert_not_called()

    def test_get_jwt_token_cache_miss(self) -> None:
        self.m_jwt_generator.account = "account"
        m_response = mock.MagicMock()
        m_response.status_code = 200
        m_response.text = "auth_token"
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator)
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(self.m_jwt_generator, role="role")
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.snowflakecomputing.com/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": "session:role:role",
                    "assertion": "jwt_token",
                },
            )
        with mock.patch.object(requests, "post", return_value=m_response) as m_post:
            token = authentication._get_snowflake_token_by_jwt(
                self.m_jwt_generator, snowflake_account_url="https://account.url"
            )
            self.assertEqual(token, "auth_token")
            m_post.assert_called_once_with(
                "https://account.url/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "scope": None,
                    "assertion": "jwt_token",
                },
            )


if __name__ == "__main__":
    absltest.main()
