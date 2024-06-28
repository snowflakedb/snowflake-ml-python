import json
from typing import Dict, Iterator, Optional, Tuple, TypedDict, Union, cast
from urllib.parse import urljoin, urlparse

import requests

from snowflake import snowpark
from snowflake.cortex._sse_client import SSEClient
from snowflake.snowpark import context, functions

CORTEX_FUNCTIONS_TELEMETRY_PROJECT = "CortexFunctions"


class CompleteOptions(TypedDict):
    # Options configuring a snowflake.cortex.Complete call
    max_tokens: int  # Sets the maximum number of output tokens in the response. Small values can result in truncated
    # responses.
    temperature: float  # A value from 0 to 1 (inclusive) that controls the randomness of the output of the language
    # model. A higher temperature (for example, 0.7) results in more diverse and random output, while a lower
    # temperature (such as 0.2) makes the output more deterministic and focused.
    topP: float  # A value from 0 to 1 (inclusive) that controls the randomness and diversity of the language model,
    # generally used as an alternative to temperature. The difference is that top_p restricts the set of possible tokens
    # that the model outputs, while temperature influences which tokens are chosen at each step.


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
    *args: Union[str, snowpark.Column, Dict[str, Union[int, float]]],
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


def _call_sql_function_column(
    function: str, *args: Union[str, snowpark.Column, Dict[str, Union[int, float]]]
) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def _call_sql_function_immediate(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, snowpark.Column, Dict[str, Union[int, float]]],
) -> str:
    if session is None:
        session = context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )

    options_present = check_for_dict_in_args(args)
    lit_args = []
    for arg in args:
        lit_args.append(functions.lit(arg))

    if options_present:  # https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
        name, content, options = args[0], args[1], args[2]
        lit_args = [
            cast(snowpark.Column, name),
            cast(snowpark.Column, [{"role": "user", "content": content}]),
            cast(snowpark.Column, options),
        ]

    empty_df = session.create_dataframe([snowpark.Row()])
    df = empty_df.select(functions.builtin(function)(*lit_args))
    return cast(str, df.collect()[0][0])


def call_rest_function(
    function: str,
    model: Union[str, snowpark.Column],
    prompt: Union[str, snowpark.Column],
    options: Optional[CompleteOptions] = None,
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

    if options:
        data = {
            **data,
            **options,  # type: ignore[dict-item]
        }  # dict | dict operation is for Python >= 3.9

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


def check_for_dict_in_args(args: Tuple[Union[str, snowpark.Column, Dict[str, Union[int, float]]], ...]) -> bool:
    options_present = False
    for arg in args:
        if isinstance(arg, dict):  # looking for options dict
            options_present = True
    return options_present
