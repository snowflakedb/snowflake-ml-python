import configparser
import os
import tempfile

import connection_params
from absl.testing import absltest


class SnowflakeLoginOptionsTest(absltest.TestCase):  # # type: ignore
    """Testing SnowflakeLoginOptions() function."""

    def setUp(self) -> None:
        """Creates a named login file to be used by other tests."""
        # Default login file content for test.
        self._login_file_toml = tempfile.NamedTemporaryFile(suffix=".config")
        self._login_file_toml.write(
            bytes(
                """
                [connections.foo]
                accountname = "foo_account"
                username = "foo_user"
                password = "foo_password"
                dbname = "foo_db"
                schemaname = "foo_schema"
                warehousename = "foo_wh"
                port = 8085

                [connections.bar]
                accountname = "bar_account"
                username = "bar_user"
                password = "bar_password"
                dbname = "bar_db"
                schemaname = "bar_schema"
                warehousename = "bar_wh"
                """,
                "utf-8",
            )
        )
        self._login_file_toml.flush()
        # Default dict generated from the above content.
        self._connection_dict_from_toml_foo = {
            "account": "foo_account",
            "database": "foo_db",
            "port": "8085",
            "password": "foo_password",
            "schema": "foo_schema",
            "user": "foo_user",
            "warehouse": "foo_wh",
        }

        # Default login file content for test.
        self._invalid_login_file_toml = tempfile.NamedTemporaryFile(suffix=".config")
        self._invalid_login_file_toml.write(
            bytes(
                """
                [connections.foo]
                accountname = "foo_account"
                username = "foo_user"
                password = "foo_password"
                dbname  foo_db
                schemaname = "foo_schema"
                warehousename = "foo_wh"
                port = 8085

                [connections.bar]
                accountname = "bar_account"
                username = "bar_user"
                password = "bar_password"
                dbname = "bar_db"
                schemaname = "bar_schema"
                warehousename = "bar_wh"
                """,
                "utf-8",
            )
        )
        self._invalid_login_file_toml.flush()

        # Default dict set in environment
        self._default_env_variable_dict = {
            "SNOWFLAKE_ACCOUNT": "admin2",
            "SNOWFLAKE_USER": "admin2",
            "SNOWFLAKE_DATABASE": "db2",
            "SNOWFLAKE_HOST": "admin2.snowflakecomputing.com",
            "SNOWFLAKE_SCHEMA": "public",
            "SNOWFLAKE_PASSWORD": "test",
            "SNOWFLAKE_WAREHOUSE": "env_warehouse",
        }

        # Connection params from above env variables.
        self._connection_dict_from_env = {
            "account": "admin2",
            "database": "db2",
            "host": "admin2.snowflakecomputing.com",
            "password": "test",
            "schema": "public",
            "user": "admin2",
            "warehouse": "env_warehouse",
        }

        # Default token file
        self._token_file = tempfile.NamedTemporaryFile()
        self._token_file.write(b"login_file_token")
        self._token_file.flush()

    def testReadInvalidSnowSQLConfigFile(self) -> None:
        """Tests if given snowsql config file is invalid, it raises exception."""
        with self.assertRaises(configparser.ParsingError):
            connection_params.SnowflakeLoginOptions("connections.foo", login_file=self._invalid_login_file_toml.name)

    def testReadSnowSQLConfigFile(self) -> None:
        """Tests if given snowsql config file is read correctly."""
        params = connection_params.SnowflakeLoginOptions("connections.foo", login_file=self._login_file_toml.name)
        self.assertEqual(params, self._connection_dict_from_toml_foo)

    def testReadFromEnv(self) -> None:
        """Tests if params are read from environment correctly."""
        connection_params._DEFAULT_CONNECTION_FILE = "/does/not/exist"
        with absltest.mock.patch.dict(os.environ, self._default_env_variable_dict):
            params = connection_params.SnowflakeLoginOptions()
            self.assertEqual(params, self._connection_dict_from_env)

    def testTokenOverrideUserPasswordAsWellAsTokenFile(self) -> None:
        """Tests if token overrides user/password & token_file from environment."""
        connection_params._DEFAULT_CONNECTION_FILE = "/does/not/exist"
        env_vars = self._default_env_variable_dict
        env_vars["SNOWFLAKE_TOKEN"] = "env_token"
        env_vars["SNOWFLAKE_TOKEN_FILE"] = self._token_file.name
        with absltest.mock.patch.dict(os.environ, env_vars):
            params = connection_params.SnowflakeLoginOptions()
            expected = self._connection_dict_from_env
            del expected["user"]
            del expected["password"]
            expected["token"] = "env_token"
            expected["authenticator"] = "oauth"
            self.assertEqual(params, expected)

    def testTokenFileOverrideEnvUserPassword(self) -> None:
        """Tests if token file overrides user/password from environment."""
        connection_params._DEFAULT_CONNECTION_FILE = "/does/not/exist"
        env_vars = self._default_env_variable_dict
        env_vars["SNOWFLAKE_TOKEN_FILE"] = self._token_file.name
        with absltest.mock.patch.dict(os.environ, self._default_env_variable_dict):
            params = connection_params.SnowflakeLoginOptions()
            expected = self._connection_dict_from_env
            del expected["user"]
            del expected["password"]
            expected["token"] = "login_file_token"
            expected["authenticator"] = "oauth"
            self.assertEqual(params, expected)

    @absltest.mock.patch.dict(  # type: ignore
        os.environ,
        {"SNOWFLAKE_ACCOUNT": "", "SNOWFLAKE_TOKEN": "env_token"},
        clear=True,
    )
    def testTokenFileOverridesLoginFile(self) -> None:
        """Tests if token overrides user/password from file."""
        connection_params._DEFAULT_CONNECTION_FILE = self._login_file_toml.name
        params = connection_params.SnowflakeLoginOptions("foo")
        expected = self._connection_dict_from_toml_foo
        del expected["user"]
        del expected["password"]
        expected["token"] = "env_token"
        expected["authenticator"] = "oauth"
        self.assertEqual(params, expected)

    def testAllOptionalFieldsMissing(self) -> None:
        """Tests if ommitting all optional fields parses correctly."""
        self._minimal_login_file = tempfile.NamedTemporaryFile(suffix=".config")
        self._minimal_login_file.write(
            bytes(
                """
                [connections]
                accountname = "snowflake"
                user = "admin"
                password = "test"
                """,
                "utf-8",
            )
        )
        self._minimal_login_file.flush()
        connection_params._DEFAULT_CONNECTION_FILE = self._minimal_login_file.name
        params = connection_params.SnowflakeLoginOptions()
        expected = {
            "account": "snowflake",
            "user": "admin",
            "password": "test",
        }
        self.assertEqual(params, expected)

    def testExternalBrowser(self) -> None:
        """Tests that using external browser authentication is correctly passed on."""
        self._minimal_login_file = tempfile.NamedTemporaryFile(suffix=".json")
        self._minimal_login_file.write(
            bytes(
                """
                [connections]
                accountname = "snowflake"
                user = "admin"
                authenticator = "externalbrowser"
                """,
                "utf-8",
            )
        )
        self._minimal_login_file.flush()
        connection_params._DEFAULT_CONNECTION_FILE = self._minimal_login_file.name
        params = connection_params.SnowflakeLoginOptions()
        expected = {
            "account": "snowflake",
            "user": "admin",
            "authenticator": "externalbrowser",
        }
        self.assertEqual(params, expected)


if __name__ == "__main__":
    absltest.main()
