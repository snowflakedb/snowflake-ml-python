import posixpath
from typing import Optional
from urllib.parse import ParseResult, urlparse, urlunparse

_LOCAL_URI_SCHEMES = ["", "file"]
_HTTP_URI_SCHEMES = ["http", "https"]
_SNOWFLAKE_STAGE_URI_SCHEMES = ["sfc", "sfstage"]


def is_local_uri(uri: str) -> bool:
    """Returns true if the URI has a scheme that indicates a local file."""
    return urlparse(uri).scheme in _LOCAL_URI_SCHEMES


def is_http_uri(uri: str) -> bool:
    """Returns true if the URI has a scheme that indicates a web (http,https) address."""
    return urlparse(uri).scheme in _HTTP_URI_SCHEMES


def is_snowflake_stage_uri(uri: str) -> bool:
    """Returns true if the URI is a scheme that indicates a Snowflake stage location."""
    return urlparse(uri).scheme in _SNOWFLAKE_STAGE_URI_SCHEMES


def get_snowflake_stage_path_from_uri(uri: str) -> Optional[str]:
    """Returns the stage path pointed by the URI.

    Args:
        uri: URI for which stage file is needed.

    Returns:
        The Snowflake stage location encoded by the given URI. Returns None if the URI is not pointing to a Snowflake
            stage.
    """
    if not is_snowflake_stage_uri(uri):
        return None
    uri_components = urlparse(uri)
    # posixpath.join will drop other components if any of arguments is absolute path.
    # The path we get is actually absolute (starting with '/'), however, since we concat them to stage location,
    # it should not.
    return posixpath.normpath(
        posixpath.join(posixpath.normpath(uri_components.netloc), posixpath.normpath(uri_components.path.lstrip("/")))
    )


def get_uri_scheme(uri: str) -> str:
    """Returns the scheme for the given URI."""
    return urlparse(uri).scheme


def get_uri_from_snowflake_stage_path(path: str) -> str:
    """Generates a URI from Snowflake stage path."""
    clean_path = path.replace('"', "").replace("'", "").lstrip("@")
    return urlunparse(
        ParseResult(
            scheme=_SNOWFLAKE_STAGE_URI_SCHEMES[0],
            netloc="",
            path=clean_path,
            params="",
            query="",
            fragment="",
        )
    )
