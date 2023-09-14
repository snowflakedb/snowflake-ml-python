import configparser
import os
import tempfile
from typing import Optional

import connection_params
from absl.testing import absltest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


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
            # Default
            "protocol": "https",
            "ssl": "on",
            "token_file": "/snowflake/session/token",
        }

        # Default token file
        self._token_file = tempfile.NamedTemporaryFile()
        self._token_file.write(b"login_file_token")
        self._token_file.flush()

    @staticmethod
    def genPrivateRsaKey(key_password: Optional[bytes] = None) -> bytes:
        "Generate a new RSA private key and return."
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        encryption_algorithm: serialization.KeySerializationEncryption = serialization.NoEncryption()
        if key_password:
            encryption_algorithm = serialization.BestAvailableEncryption(key_password)
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm,
        )

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
            # TODO - SUMIT
            self.assertEqual(params, self._connection_dict_from_env)

    def testOptionalEmptyEnvVarRemoved(self) -> None:
        """Tests that empty optional env variables are skipped."""
        connection_params._DEFAULT_CONNECTION_FILE = "/does/not/exist"
        env_vars = self._default_env_variable_dict
        env_vars["SNOWFLAKE_TOKEN_FILE"] = self._token_file.name
        env_vars["SNOWFLAKE_USER"] = ""  # Optional field empty env var => will not come in result
        del env_vars["SNOWFLAKE_PASSWORD"]  # Removing env var for password => will not come in result
        with absltest.mock.patch.dict(os.environ, env_vars, clear=True):
            params = connection_params.SnowflakeLoginOptions()
            expected = self._connection_dict_from_env
            del expected["user"]
            del expected["password"]  # No env var => will not come in result
            expected["token_file"] = self._token_file.name
            expected["token"] = "login_file_token"
            expected["authenticator"] = "oauth"
            self.assertEqual(params, expected)

    def testAllOptionalFieldsMissing(self) -> None:
        """Tests if omitting all optional fields parses correctly."""
        minimal_login_file = tempfile.NamedTemporaryFile(suffix=".config")
        minimal_login_file.write(
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
        minimal_login_file.flush()
        connection_params._DEFAULT_CONNECTION_FILE = minimal_login_file.name
        params = connection_params.SnowflakeLoginOptions()
        expected = {
            "account": "snowflake",
            "user": "admin",
            "password": "test",
        }
        self.assertEqual(params, expected)

    def testExternalBrowser(self) -> None:
        """Tests that using external browser authentication is correctly passed on."""
        minimal_login_file = tempfile.NamedTemporaryFile(suffix=".json")
        minimal_login_file.write(
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
        minimal_login_file.flush()
        connection_params._DEFAULT_CONNECTION_FILE = minimal_login_file.name
        params = connection_params.SnowflakeLoginOptions()
        expected = {
            "account": "snowflake",
            "user": "admin",
            "authenticator": "externalbrowser",
        }
        self.assertEqual(params, expected)

    def testUnencryptedPrivateKeyPath(self) -> None:
        """Tests unencrypted private key path populates private key."""
        unencrypted_pem_private_key = self.genPrivateRsaKey()
        private_key_path = tempfile.NamedTemporaryFile(suffix=".pk8")
        private_key_path.write(unencrypted_pem_private_key)
        private_key_path.flush()
        minimal_login_file = tempfile.NamedTemporaryFile(suffix=".config")
        minimal_login_file.write(
            bytes(
                """
                [connections]
                accountname = "snowflake"
                user = "admin"
                private_key_path = "{private_key_path}"
                """.format(
                    private_key_path=private_key_path.name
                ),
                "utf-8",
            )
        )
        minimal_login_file.flush()

        connection_params._DEFAULT_CONNECTION_FILE = minimal_login_file.name
        params = connection_params.SnowflakeLoginOptions()

        # Check private_key is set and not empty - aka deserialization worked
        self.assertNotEqual(params["private_key"], "")
        # No need to validate the value. So resetting it.
        del params["private_key"]

        expected = {
            "account": "snowflake",
            "user": "admin",
            "private_key_path": private_key_path.name,  # We do not remove it, connect() does not use it
        }
        self.assertEqual(params, expected)

    def testUnencryptedPrivateKeyPathWithEmptyEnvPassword(self) -> None:
        """Tests unencrypted private key path populates private key where empty env var for passphrase."""
        unencrypted_pem_private_key = self.genPrivateRsaKey()
        private_key_path = tempfile.NamedTemporaryFile(suffix=".pk8")
        private_key_path.write(unencrypted_pem_private_key)
        private_key_path.flush()
        minimal_login_file = tempfile.NamedTemporaryFile(suffix=".config")
        minimal_login_file.write(
            bytes(
                """
                [connections]
                accountname = "snowflake"
                user = "admin"
                private_key_path = "{private_key_path}"
                """.format(
                    private_key_path=private_key_path.name
                ),
                "utf-8",
            )
        )
        minimal_login_file.flush()

        connection_params._DEFAULT_CONNECTION_FILE = minimal_login_file.name
        with absltest.mock.patch.dict(os.environ, {"SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": ""}):
            params = connection_params.SnowflakeLoginOptions()

        # Check private_key is set and not empty - aka deserialization worked
        self.assertNotEqual(params["private_key"], "")
        # No need to validate the value. So resetting it.
        del params["private_key"]

        expected = {
            "account": "snowflake",
            "user": "admin",
            "private_key_path": private_key_path.name,  # We do not remove it, connect() does not use it
        }
        self.assertEqual(params, expected)

    def testEncryptedPrivateKeyPath(self) -> None:
        """Tests unencrypted private key path populates private key."""
        key_password = "foo"
        unencrypted_pem_private_key = self.genPrivateRsaKey(bytes(key_password, "utf-8"))
        private_key_path = tempfile.NamedTemporaryFile(suffix=".pk8")
        private_key_path.write(unencrypted_pem_private_key)
        private_key_path.flush()
        minimal_login_file = tempfile.NamedTemporaryFile(suffix=".config")
        minimal_login_file.write(
            bytes(
                """
                [connections]
                accountname = "snowflake"
                user = "admin"
                private_key_path = "{private_key_path}"
                """.format(
                    private_key_path=private_key_path.name
                ),
                "utf-8",
            )
        )
        minimal_login_file.flush()

        connection_params._DEFAULT_CONNECTION_FILE = minimal_login_file.name
        with absltest.mock.patch.dict(os.environ, {"SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": key_password}):
            params = connection_params.SnowflakeLoginOptions()

        # Check private_key is set and not empty - aka deserialization worked
        self.assertNotEqual(params["private_key"], "")
        # No need to validate the value. So resetting it.
        del params["private_key"]

        expected = {
            "account": "snowflake",
            "user": "admin",
            "private_key_path": private_key_path.name,  # We do not remove it, connect() does not use it
        }
        self.assertEqual(params, expected)


if __name__ == "__main__":
    absltest.main()
