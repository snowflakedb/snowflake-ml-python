import configparser
import logging
import os
from typing import Optional, Union

from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from snowflake import snowpark

logger = logging.getLogger(__name__)

_DEFAULT_CONNECTION_FILE = "~/.snowsql/config"


def _read_token(token_file: str = "") -> str:
    """
    Reads token from environment or file provided.

    First tries to read the token from environment variable
    (`SNOWFLAKE_TOKEN`) followed by the token file.
    Both the options are tried out in SnowServices.

    Args:
        token_file: File from which token needs to be read. Optional.

    Returns:
        the token.
    """
    token = os.getenv("SNOWFLAKE_TOKEN", "")
    if token:
        return token
    if token_file and os.path.exists(token_file):
        with open(token_file) as f:
            token = f.read()
    return token


_ENCRYPTED_PKCS8_PK_HEADER = b"-----BEGIN ENCRYPTED PRIVATE KEY-----"
_UNENCRYPTED_PKCS8_PK_HEADER = b"-----BEGIN PRIVATE KEY-----"


def _load_pem_to_der(private_key_path: str) -> bytes:
    """Given a private key file path (in PEM format), decode key data into DER format."""
    with open(private_key_path, "rb") as f:
        private_key_pem = f.read()
    private_key_passphrase: Optional[str] = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", None)

    # Only PKCS#8 format key will be accepted. However, openssl
    # transparently handle PKCS#8 and PKCS#1 format (by some fallback
    # logic) and their is no function to distinguish between them. By
    # reading openssl source code, apparently they also relies on header
    # to determine if give bytes is PKCS#8 format or not
    if not private_key_pem.startswith(_ENCRYPTED_PKCS8_PK_HEADER) and not private_key_pem.startswith(
        _UNENCRYPTED_PKCS8_PK_HEADER
    ):
        raise Exception("Private key provided is not in PKCS#8 format. Please use correct format.")

    if private_key_pem.startswith(_ENCRYPTED_PKCS8_PK_HEADER) and private_key_passphrase is None:
        raise Exception(
            "Private key is encrypted but passphrase could not be found. "
            "Please set SNOWFLAKE_PRIVATE_KEY_PASSPHRASE env variable."
        )

    if private_key_pem.startswith(_UNENCRYPTED_PKCS8_PK_HEADER):
        private_key_passphrase = None

    private_key = serialization.load_pem_private_key(
        private_key_pem,
        str.encode(private_key_passphrase) if private_key_passphrase is not None else private_key_passphrase,
        backends.default_backend(),
    )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _connection_properties_from_env() -> dict[str, str]:
    """Returns a dict with all possible login related env variables."""
    sf_conn_prop = {
        # Mandatory fields
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        # With a default value
        "token_file": os.getenv("SNOWFLAKE_TOKEN_FILE", "/snowflake/session/token"),
        "ssl": os.getenv("SNOWFLAKE_SSL", "on"),
        "protocol": os.getenv("SNOWFLAKE_PROTOCOL", "https"),
    }
    # With empty default value
    for key, env_var in {
        "user": "SNOWFLAKE_USER",
        "authenticator": "SNOWFLAKE_AUTHENTICATOR",
        "password": "SNOWFLAKE_PASSWORD",
        "host": "SNOWFLAKE_HOST",
        "port": "SNOWFLAKE_PORT",
        "schema": "SNOWFLAKE_SCHEMA",
        "warehouse": "SNOWFLAKE_WAREHOUSE",
        "private_key_path": "SNOWFLAKE_PRIVATE_KEY_PATH",
    }.items():
        value = os.getenv(env_var, "")
        if value:
            sf_conn_prop[key] = value
    return sf_conn_prop


def _load_from_snowsql_config_file(connection_name: str, login_file: str = "") -> dict[str, str]:
    """Loads the dictionary from snowsql config file."""
    snowsql_config_file = login_file if login_file else os.path.expanduser(_DEFAULT_CONNECTION_FILE)
    if not os.path.exists(snowsql_config_file):
        logger.error(f"Connection name given but snowsql config file is not found at: {snowsql_config_file}")
        raise Exception("Snowflake SnowSQL config not found.")

    config = configparser.ConfigParser(inline_comment_prefixes="#")

    snowflake_connection_name = os.getenv("SNOWFLAKE_CONNECTION_NAME")
    if snowflake_connection_name is not None:
        connection_name = snowflake_connection_name

    if connection_name:
        if not connection_name.startswith("connections."):
            connection_name = "connections." + connection_name
    else:
        # See https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings
        connection_name = "connections"

    logger.info(f"Reading {snowsql_config_file} for connection parameters defined as {connection_name}")
    config.read(snowsql_config_file)
    conn_params = dict(config[connection_name])
    # Remap names to appropriate args in Python Connector API
    # Note: "dbname" should become "database"
    conn_params = {k.replace("name", ""): v.strip('"') for k, v in conn_params.items()}
    if "db" in conn_params:
        conn_params["database"] = conn_params["db"]
        del conn_params["db"]
    return conn_params


@snowpark._internal.utils.deprecated(version="1.8.5")
def SnowflakeLoginOptions(connection_name: str = "", login_file: Optional[str] = None) -> dict[str, Union[str, bytes]]:
    """Returns a dict that can be used directly into snowflake python connector or Snowpark session config.

    NOTE: Token/Auth information is sideloaded in all cases above, if provided in following order:
      1. If SNOWFLAKE_TOKEN is defined in the environment, it will be used.
      2. If SNOWFLAKE_TOKEN_FILE is defined in the environment and file matching the value found, content of the file
         will be used.

    If token is found, username, password will be reset and 'authenticator' will be set to 'oauth'.

    Python Connector:
    >> ctx = snowflake.connector.connect(**(SnowflakeLoginOptions()))

    Snowpark Session:
    >> session = Session.builder.configs(SnowflakeLoginOptions()).create()

    Usage Note:
      Ideally one should have a snowsql config file. Read more here:
      https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings

      If snowsql config file does not exist, it tries auth from env variables.

    Args:
        connection_name: Name of the connection to look for inside the config file. If environment variable
            SNOWFLAKE_CONNECTION_NAME is provided, it will override the input connection_name.
        login_file: If provided, this is used as config file instead of default one (_DEFAULT_CONNECTION_FILE).

    Returns:
        A dict with connection parameters.

    Raises:
        Exception: if none of config file and environment variable are present.
    """
    conn_prop: dict[str, Union[str, bytes]] = {}
    login_file = login_file or os.path.expanduser(_DEFAULT_CONNECTION_FILE)
    # If login file exists, use this exclusively.
    if os.path.exists(login_file):
        conn_prop = {**(_load_from_snowsql_config_file(connection_name, login_file))}
    else:
        # If environment exists for SNOWFLAKE_ACCOUNT, assume everything
        # comes from environment. Mixing it not allowed.
        account = os.getenv("SNOWFLAKE_ACCOUNT", "")
        if account:
            conn_prop = {**_connection_properties_from_env()}
        else:
            raise Exception("Snowflake credential is neither set in env nor a login file was provided.")

    # Token, if specified, is always side-loaded in all cases.
    token = _read_token(str(conn_prop["token_file"]) if "token_file" in conn_prop else "")
    if token:
        conn_prop["token"] = token
        if "authenticator" not in conn_prop or conn_prop["authenticator"]:
            conn_prop["authenticator"] = "oauth"
    elif "private_key_path" in conn_prop and "private_key" not in conn_prop:
        conn_prop["private_key"] = _load_pem_to_der(str(conn_prop["private_key_path"]))

    if "ssl" in conn_prop and conn_prop["ssl"].lower() == "off":
        conn_prop["protocol"] = "http"

    return conn_prop
