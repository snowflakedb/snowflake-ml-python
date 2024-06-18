import json
from typing import Iterator, Optional, Union, cast
from urllib.parse import urljoin, urlparse

import requests

from snowflake import snowpark
from snowflake.cortex._sse_client import SSEClient
from snowflake.snowpark import context, functions

CORTEX_FUNCTIONS_TELEMETRY_PROJECT = "CortexFunctions"


class SSEParseException(Exception):
    """This exception is raised when an invalid server sent event is received from the server."""

    pass


class SnowflakeAuthenticationException(Exception):
    """This exception is raised when the session object does not have session.connection.rest.token attribute."""

    pass


# Calls a sql function, handling both immediate (e.g. python types) and batch
# (e.g. snowpark column and literal type modes).
def call_sql_function(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, snowpark.Column],
) -> Union[str, snowpark.Column]:
    handle_as_column = False
    for arg in args:
        if isinstance(arg, snowpark.Column):
            handle_as_column = True

    if handle_as_column:
        return cast(Union[str, snowpark.Column], _call_sql_function_column(function, *args))
    return cast(
        Union[str, snowpark.Column],
        _call_sql_function_immediate(function, session, *args),
    )


def _call_sql_function_column(function: str, *args: Union[str, snowpark.Column]) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def _call_sql_function_immediate(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, snowpark.Column],
) -> str:
    if session is None:
        session = context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )

    lit_args = []
    for arg in args:
        lit_args.append(functions.lit(arg))

    empty_df = session.create_dataframe([snowpark.Row()])
    df = empty_df.select(functions.builtin(function)(*lit_args))
    return cast(str, df.collect()[0][0])


def call_rest_function(
    function: str,
    model: Union[str, snowpark.Column],
    prompt: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
    stream: bool = False,
) -> requests.Response:
    if session is None:
        session = context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )

    if not hasattr(session.connection.rest, "token"):
        raise SnowflakeAuthenticationException("Snowflake session error: REST token missing.")

    if session.connection.rest.token is None or session.connection.rest.token == "":  # type: ignore[union-attr]
        raise SnowflakeAuthenticationException("Snowflake session error: REST token is empty.")

    url = urljoin(session.connection.host, f"api/v2/cortex/inference/{function}")
    if urlparse(url).scheme == "":
        url = "https://" + url
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Snowflake Token="{session.connection.rest.token}"',  # type: ignore[union-attr]
        "Accept": "application/json, text/event-stream",
    }

    data = {
        "model": model,
        "messages": [{"content": prompt}],
        "stream": stream,
    }

    response = requests.post(
        url,
        json=data,
        headers=headers,
        stream=stream,
    )
    response.raise_for_status()
    return response


def process_rest_response(response: requests.Response, stream: bool = False) -> Union[str, Iterator[str]]:
    if not stream:
        try:
            message = response.json()["choices"][0]["message"]
            output = str(message.get("content", ""))
            return output
        except (KeyError, IndexError) as e:
            raise SSEParseException("Failed to parse streamed response.") from e
    else:
        return _return_gen(response)


def _return_gen(response: requests.Response) -> Iterator[str]:
    client = SSEClient(response)
    for event in client.events():
        response_loaded = json.loads(event.data)
        try:
            delta = response_loaded["choices"][0]["delta"]
            output = str(delta.get("content", ""))
            yield output
        except (KeyError, IndexError) as e:
            raise SSEParseException("Failed to parse streamed response.") from e
