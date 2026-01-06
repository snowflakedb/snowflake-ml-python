from unittest import mock

from absl.testing.absltest import TestCase, main

import snowflake.ml._internal.utils.url as url


class UrlTest(TestCase):
    def test_get_snowflake_url_basic(self) -> None:
        """Tests basic URL construction with default scheme (https)."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        # When scheme attribute doesn't exist, should default to https
        del mock_session.connection.scheme

        result = url.get_snowflake_url(mock_session, "/some/path")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com/some/path")

    def test_get_snowflake_url_with_custom_scheme(self) -> None:
        """Tests URL construction with custom scheme."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "http"

        result = url.get_snowflake_url(mock_session, "/some/path")
        self.assertEqual(result, "http://myaccount.snowflakecomputing.com/some/path")

    def test_get_snowflake_url_with_all_parameters(self) -> None:
        """Tests URL construction with all optional parameters."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(
            mock_session,
            "/path/to/resource",
            params="param1",
            query="key=value&foo=bar",
            fragment="section1",
        )
        expected = "https://myaccount.snowflakecomputing.com/path/to/resource;param1?key=value&foo=bar#section1"
        self.assertEqual(result, expected)

    def test_get_snowflake_url_with_query_only(self) -> None:
        """Tests URL construction with only query parameters."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "/api/endpoint", query="token=abc123")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com/api/endpoint?token=abc123")

    def test_get_snowflake_url_with_fragment_only(self) -> None:
        """Tests URL construction with only fragment."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "/docs/page", fragment="introduction")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com/docs/page#introduction")

    def test_get_snowflake_url_empty_path(self) -> None:
        """Tests URL construction with empty path."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com")

    def test_get_snowflake_url_root_path(self) -> None:
        """Tests URL construction with root path."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "/")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com/")

    def test_get_snowflake_url_different_host(self) -> None:
        """Tests URL construction with different host formats."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "account.region.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "/path")
        self.assertEqual(result, "https://account.region.snowflakecomputing.com/path")

    def test_get_snowflake_url_with_port(self) -> None:
        """Tests URL construction when host includes port."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com:8080"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "/api")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com:8080/api")

    def test_get_snowflake_url_nested_path(self) -> None:
        """Tests URL construction with deeply nested path."""
        mock_session = mock.MagicMock()
        mock_session.connection.host = "myaccount.snowflakecomputing.com"
        mock_session.connection.scheme = "https"

        result = url.get_snowflake_url(mock_session, "/api/v2/models/deployments/list")
        self.assertEqual(result, "https://myaccount.snowflakecomputing.com/api/v2/models/deployments/list")


if __name__ == "__main__":
    main()
