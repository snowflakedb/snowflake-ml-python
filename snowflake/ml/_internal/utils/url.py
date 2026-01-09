from urllib.parse import urlunparse

from snowflake import snowpark as snowpark

JOB_URL_PREFIX = "#/compute/job/"
SERVICE_URL_PREFIX = "#/compute/service/"


def get_snowflake_url(
    session: snowpark.Session,
    url_path: str,
    params: str = "",
    query: str = "",
    fragment: str = "",
) -> str:
    """Construct a Snowflake URL from session connection details and URL components.

    Args:
        session: The Snowpark session containing connection details.
        url_path: The path component of the URL (e.g., "/compute/job/123").
        params: Optional parameters for the URL (RFC 1808). Defaults to "".
        query: Optional query string for the URL. Defaults to "".
        fragment: Optional fragment identifier for the URL (e.g., "#section"). Defaults to "".

    Returns:
        A fully constructed Snowflake URL string with scheme, host, and specified components.
    """
    scheme = "https"
    if hasattr(session.connection, "scheme"):
        scheme = session.connection.scheme
    host = session.connection.host

    return urlunparse(
        (
            scheme,
            host,
            url_path,
            params,
            query,
            fragment,
        )
    )
