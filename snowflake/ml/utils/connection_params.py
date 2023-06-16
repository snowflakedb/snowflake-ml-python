import configparser
import os
from typing import Dict, Optional

from absl import logging

from snowflake import snowpark

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


def _connection_properties_from_env() -> Dict[str, str]:
    """Returns a dict with all possible login related env variables."""
    sf_conn_prop = {
        # Mandatory fields
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.getenv("SNOWFLAKE_USER", ""),
        "database": os.environ["SNOWFLAKE_DATABASE"],
        # With empty default value
        "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", ""),
        "password": os.getenv("SNOWFLAKE_PASSWORD", ""),
        "token_file": os.getenv("SNOWFLAKE_TOKEN_FILE", "/snowflake/session/token"),
        "host": os.getenv("SNOWFLAKE_HOST", ""),
        "port": os.getenv("SNOWFLAKE_PORT", ""),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "basic"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", ""),
        "ssl": os.getenv("SNOWFLAKE_SSL", "on"),
    }
    return sf_conn_prop


def _load_from_snowsql_config_file(connection_name: str, login_file: str = "") -> Dict[str, str]:
    """Loads the dictionary from snowsql config file."""
    snowsql_config_file = login_file if login_file else os.path.expanduser(_DEFAULT_CONNECTION_FILE)
    if not os.path.exists(snowsql_config_file):
        logging.error(f"Connection name given but snowsql config file is not found at: {snowsql_config_file}")
        raise Exception("Snowflake SnowSQL config not found.")

    config = configparser.ConfigParser(inline_comment_prefixes="#")

    if connection_name:
        if not connection_name.startswith("connections."):
            connection_name = "connections." + connection_name
    else:
        # See https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings
        connection_name = "connections"

    logging.info(f"Reading {snowsql_config_file} for connection parameters defined as {connection_name}")
    config.read(snowsql_config_file)
    conn_params = dict(config[connection_name])
    # Remap names to appropriate args in Python Connector API
    # Note: "dbname" should become "database"
    conn_params = {k.replace("name", ""): v.strip('"') for k, v in conn_params.items()}
    if "db" in conn_params:
        conn_params["database"] = conn_params["db"]
        del conn_params["db"]
    return conn_params


@snowpark._internal.utils.private_preview(version="0.2.0")
def SnowflakeLoginOptions(connection_name: str = "", login_file: Optional[str] = None) -> Dict[str, str]:
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
      Ideally one should have a snoqsql config file. Read more here:
      https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings

    Args:
        connection_name: Name of the connection to look for inside the config file. If `connection_name` is NOT given,
            it tries auth from env variables.
        login_file: If provided, this is used as config file instead of default one (_DEFAULT_CONNECTION_FILE).

    Returns:
        A dict with connection parameters.

    Raises:
        Exception: if none of config file and environment variable are present.
    """
    conn_prop = {}
    login_file = login_file or os.path.expanduser(_DEFAULT_CONNECTION_FILE)
    # If login file exists, use this exclusively.
    if os.path.exists(login_file):
        conn_prop = _load_from_snowsql_config_file(connection_name, login_file)
    else:
        # If environment exists for SNOWFLAKE_ACCOUNT, assume everything
        # comes from environment. Mixing it not allowed.
        account = os.getenv("SNOWFLAKE_ACCOUNT", "")
        if account:
            conn_prop = _connection_properties_from_env()
        else:
            raise Exception("Snowflake credential is neither set in env nor a login file was provided.")

    # Token, if specified, is always side-loaded in all cases.
    conn_prop["token"] = _read_token(conn_prop["token_file"] if "token_file" in conn_prop else "")
    data = {
        "account": conn_prop["account"],
    }
    for field in ["database", "schema", "warehouse", "host", "port", "role", "session_parameters"]:
        if field in conn_prop and conn_prop[field]:
            data[field] = conn_prop[field]

    if "authenticator" in conn_prop and conn_prop["authenticator"] == "externalbrowser":
        data["authenticator"] = conn_prop["authenticator"]
        data["user"] = conn_prop["user"]
    elif conn_prop["token"]:
        data["token"] = conn_prop["token"]
        data["authenticator"] = "oauth"
    else:
        data["user"] = conn_prop["user"]
        data["password"] = conn_prop["password"]

    if "ssl" in conn_prop and conn_prop["ssl"].lower() == "off":
        data["protocol"] = "http"

    return data
