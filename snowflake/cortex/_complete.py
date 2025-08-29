import json
import logging
import time
import typing
from io import BytesIO
from typing import Any, Callable, Iterator, Optional, TypedDict, Union, cast
from urllib.parse import urlunparse

import requests
from snowflake.core.rest import RESTResponse
from typing_extensions import NotRequired, deprecated

from snowflake import snowpark
from snowflake.cortex._sse_client import SSEClient
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    SnowflakeAuthenticationException,
    SnowflakeConfigurationException,
)
from snowflake.ml._internal import telemetry
from snowflake.snowpark import context, functions
from snowflake.snowpark._internal.utils import is_in_stored_procedure

logger = logging.getLogger(__name__)
_REST_COMPLETE_URL = "/api/v2/cortex/inference:complete"


class ResponseFormat(TypedDict):
    """Represents an object describing response format config for structured-output mode"""

    type: str
    """The response format type (e.g. "json")"""
    schema: dict[str, Any]
    """The schema defining the structure of the response. For json it should be a valid json schema object"""


class ConversationMessage(TypedDict):
    """Represents an conversation interaction."""

    role: str
    """The role of the participant. For example, "user" or "assistant"."""

    content: str
    """The content of the message."""


class CompleteOptions(TypedDict):
    """Options configuring a snowflake.cortex.Complete call."""

    max_tokens: NotRequired[int]
    """ Sets the maximum number of output tokens in the response. Small values can result in
    truncated responses. """
    temperature: NotRequired[float]
    """ A value from 0 to 1 (inclusive) that controls the randomness of the output of the language
    model. A higher temperature (for example, 0.7) results in more diverse and random output, while a lower
    temperature (such as 0.2) makes the output more deterministic and focused. """

    top_p: NotRequired[float]
    """ A value from 0 to 1 (inclusive) that controls the randomness and diversity of the language model,
    generally used as an alternative to temperature. The difference is that top_p restricts the set of possible tokens
    that the model outputs, while temperature influences which tokens are chosen at each step. """

    guardrails: NotRequired[bool]
    """ A boolean value that controls whether Cortex Guard filters unsafe or harmful responses
    from the language model. """

    response_format: NotRequired[ResponseFormat]
    """ An object describing response format config for structured-output mode """


class ResponseParseException(Exception):
    """This exception is raised when the server response cannot be parsed."""


class MidStreamException(Exception):
    """The SSE (Server-sent Event) stream can contain error messages in the middle of the stream,
    using the “error” event type. This exception is raised when there is such a mid-stream error.
    """

    def __init__(
        self,
        reason: typing.Optional[str] = None,
        http_resp: typing.Optional["RESTResponse"] = None,
        request_id: typing.Optional[str] = None,
    ) -> None:
        message = ""
        if reason is not None:
            message = reason
        if http_resp:
            message = f"Error in stream (HTTP Response: {http_resp.status}) - {http_resp.reason}"
        if request_id is not None and request_id != "":
            # add request_id to error message
            message += f" (Request ID: {request_id})"
        super().__init__(message)


class GuardrailsOptions(TypedDict):
    enabled: bool
    """A boolean value that controls whether Cortex Guard filters unsafe or harmful responses
    from the language model."""

    response_when_unsafe: str
    """The response to return when the language model generates unsafe or harmful content."""


_MAX_RETRY_SECONDS = 30


def retry(func: Callable[..., requests.Response]) -> Callable[..., requests.Response]:
    def inner(*args: Any, **kwargs: Any) -> requests.Response:
        deadline = cast(Optional[float], kwargs["deadline"])
        kwargs = {key: value for key, value in kwargs.items() if key != "deadline"}
        expRetrySeconds = 0.5
        while True:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError()
            response = func(*args, **kwargs)
            if response.status_code >= 200 and response.status_code < 300:
                return response
            retry_status_codes = [429, 503, 504]
            if response.status_code not in retry_status_codes:
                response.raise_for_status()
            logger.debug(f"request failed with status code {response.status_code}, retrying")

            # Formula: delay(i) = max(RetryAfterHeader, min(2^i, _MAX_RETRY_SECONDS)).
            expRetrySeconds = min(2 * expRetrySeconds, _MAX_RETRY_SECONDS)
            retrySeconds = expRetrySeconds
            retryAfterHeader = response.headers.get("retry-after")
            if retryAfterHeader is not None:
                retrySeconds = max(retrySeconds, int(retryAfterHeader))
            logger.debug(f"sleeping for {retrySeconds}s before retrying")
            time.sleep(retrySeconds)

    return inner


def _make_common_request_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    return headers


def _get_request_id(resp: dict[str, Any]) -> Optional[Any]:
    request_id = None
    if "headers" in resp:
        for key, value in resp["headers"].items():
            # Note: There is some whitespace in the headers making it not possible
            # to directly index the header reliably.
            if key.strip().lower() == "x-snowflake-request-id":
                request_id = value
                break
    return request_id


def _validate_response_format_object(options: CompleteOptions) -> None:
    """Validate the response format object for structured-output mode.

    More details can be found in:
    docs.snowflake.com/en/user-guide/snowflake-cortex/complete-structured-outputs#using-complete-structured-outputs

    Args:
        options: The complete options object.

    Raises:
        ValueError: If the response format object is invalid or missing required fields.
    """
    if options is not None and options.get("response_format") is not None:
        options_obj = options.get("response_format")
        if not isinstance(options_obj, dict):
            raise ValueError("'response_format' should be an object")
        if options_obj.get("type") is None:
            raise ValueError("'type' cannot be empty for 'response_format' object")
        if not isinstance(options_obj.get("type"), str):
            raise ValueError("'type' needs to be a str for 'response_format' object")
        if options_obj.get("schema") is None:
            raise ValueError("'schema' cannot be empty for 'response_format' object")
        if not isinstance(options_obj.get("schema"), dict):
            raise ValueError("'schema' needs to be a dict for 'response_format' object")


def _make_request_body(
    model: str,
    prompt: Union[str, list[ConversationMessage]],
    options: Optional[CompleteOptions] = None,
) -> dict[str, Any]:
    data = {
        "model": model,
        "stream": True,
    }
    if isinstance(prompt, list):
        data["messages"] = prompt
    else:
        data["messages"] = [{"content": prompt}]

    if options:
        if "max_tokens" in options:
            data["max_tokens"] = options["max_tokens"]
            data["max_output_tokens"] = options["max_tokens"]
        if "temperature" in options:
            data["temperature"] = options["temperature"]
        if "top_p" in options:
            data["top_p"] = options["top_p"]
        if "guardrails" in options and options["guardrails"]:
            guardrails_options: GuardrailsOptions = {
                "enabled": True,
                "response_when_unsafe": "Response filtered by Cortex Guard",
            }
            data["guardrails"] = guardrails_options
        if "response_format" in options:
            data["response_format"] = options["response_format"]

    return data


# XP endpoint returns a dict response which needs to be converted to a format which can
# be consumed by the SSEClient. This method does that.
def _xp_dict_to_response(raw_resp: dict[str, Any]) -> requests.Response:

    response = requests.Response()
    response.status_code = int(raw_resp["status"])
    response.headers = raw_resp["headers"]

    request_id = _get_request_id(raw_resp)

    data = raw_resp["content"]
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        raise ValueError(f"Request failed (request id: {request_id})")
    if response.status_code < 200 or response.status_code >= 300:
        if "message" not in data:
            raise ValueError(f"Request failed (request id: {request_id})")
        message = data["message"]
        raise ValueError(f"Request failed: {message} (request id: {request_id})")

    # Convert the dictionary to a string format that resembles the SSE event format
    # For example, if the dict is {'event': 'message', 'data': 'your data'}, it should be formatted like this:
    sse_format_data = ""
    for event in data:
        event_type = event.get("event", "message")
        event_data = event.get("data", "")
        event_data = json.dumps(event_data)
        sse_format_data += f"event: {event_type}\ndata: {event_data}\n\n"  # Add each event with new lines

    response.raw = BytesIO(sse_format_data.encode("utf-8"))
    return response


@retry
def _call_complete_xp(
    snow_api_xp_request_handler: Optional[Callable[..., dict[str, Any]]],
    model: str,
    prompt: Union[str, list[ConversationMessage]],
    options: Optional[CompleteOptions] = None,
    deadline: Optional[float] = None,
) -> requests.Response:
    headers = _make_common_request_headers()
    body = _make_request_body(model, prompt, options)
    assert snow_api_xp_request_handler is not None
    raw_resp = snow_api_xp_request_handler("POST", _REST_COMPLETE_URL, {}, headers, body, {}, deadline)
    return _xp_dict_to_response(raw_resp)


@retry
def _call_complete_rest(
    model: str,
    prompt: Union[str, list[ConversationMessage]],
    options: Optional[CompleteOptions] = None,
    session: Optional[snowpark.Session] = None,
) -> requests.Response:
    session = session or context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )

    if session.connection.host is None or session.connection.host == "":
        raise SnowflakeConfigurationException("Snowflake connection configuration does not specify 'host'")

    if session.connection.rest is None or not hasattr(session.connection.rest, "token"):
        raise SnowflakeAuthenticationException("Snowflake session error: REST token missing.")

    if session.connection.rest.token is None or session.connection.rest.token == "":
        raise SnowflakeAuthenticationException("Snowflake session error: REST token is empty.")

    scheme = "https"
    if hasattr(session.connection, "scheme"):
        scheme = session.connection.scheme
    url = urlunparse((scheme, session.connection.host, _REST_COMPLETE_URL, "", "", ""))

    headers = _make_common_request_headers()
    headers["Authorization"] = f'Snowflake Token="{session.connection.rest.token}"'

    body = _make_request_body(model, prompt, options)
    logger.debug(f"making POST request to {url} (model={model})")
    return requests.post(
        url,
        json=body,
        headers=headers,
        stream=True,
    )


def _return_stream_response(
    response: requests.Response,
    deadline: Optional[float],
    session: Optional[snowpark.Session] = None,
) -> Iterator[str]:
    request_id = _get_request_id(dict(response.headers))
    client = SSEClient(response)
    for event in client.events():
        if deadline is not None and time.time() > deadline:
            raise TimeoutError()
        try:
            parsed_resp = json.loads(event.data)
        except json.JSONDecodeError:
            raise ResponseParseException("Server response cannot be parsed")
        try:
            yield parsed_resp["choices"][0]["delta"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError):
            # For the sake of evolution of the output format,
            # ignore stream messages that don't match the expected format.

            # This is the case of midstream errors which were introduced specifically for structured output.
            # TODO: discuss during code review
            if parsed_resp.get("error"):
                error_info = parsed_resp["error"]
                raise MidStreamException(reason=str(error_info), request_id=request_id)
            else:
                pass


def _complete_call_sql_function_snowpark(
    function: str, *args: Union[str, snowpark.Column, CompleteOptions]
) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def _complete_non_streaming_immediate(
    snow_api_xp_request_handler: Optional[Callable[..., dict[str, Any]]],
    model: str,
    prompt: Union[str, list[ConversationMessage]],
    options: Optional[CompleteOptions],
    session: Optional[snowpark.Session] = None,
    deadline: Optional[float] = None,
) -> str:
    response = _complete_rest(
        snow_api_xp_request_handler=snow_api_xp_request_handler,
        model=model,
        prompt=prompt,
        options=options,
        session=session,
        deadline=deadline,
    )
    return "".join(response)


def _complete_non_streaming_impl(
    snow_api_xp_request_handler: Optional[Callable[..., dict[str, Any]]],
    function: str,
    model: Union[str, snowpark.Column],
    prompt: Union[str, list[ConversationMessage], snowpark.Column],
    options: Optional[Union[CompleteOptions, snowpark.Column]],
    session: Optional[snowpark.Session] = None,
    deadline: Optional[float] = None,
) -> Union[str, snowpark.Column]:
    if isinstance(prompt, snowpark.Column):
        if options is not None:
            return _complete_call_sql_function_snowpark(function, model, prompt, options)
        else:
            return _complete_call_sql_function_snowpark(function, model, prompt)
    if isinstance(model, snowpark.Column):
        raise ValueError("'model' cannot be a snowpark.Column when 'prompt' is a string.")
    if isinstance(options, snowpark.Column):
        raise ValueError("'options' cannot be a snowpark.Column when 'prompt' is a string.")
    if options and not isinstance(options, snowpark.Column):
        _validate_response_format_object(options)
    return _complete_non_streaming_immediate(
        snow_api_xp_request_handler=snow_api_xp_request_handler,
        model=model,
        prompt=prompt,
        options=options,
        session=session,
        deadline=deadline,
    )


def _complete_rest(
    snow_api_xp_request_handler: Optional[Callable[..., dict[str, Any]]],
    model: str,
    prompt: Union[str, list[ConversationMessage]],
    options: Optional[CompleteOptions] = None,
    session: Optional[snowpark.Session] = None,
    deadline: Optional[float] = None,
) -> Iterator[str]:
    if options:
        _validate_response_format_object(options)
    if snow_api_xp_request_handler is not None:
        response = _call_complete_xp(
            snow_api_xp_request_handler=snow_api_xp_request_handler,
            model=model,
            prompt=prompt,
            options=options,
            deadline=deadline,
        )
    else:
        response = _call_complete_rest(model=model, prompt=prompt, options=options, session=session, deadline=deadline)
    assert response.status_code >= 200 and response.status_code < 300
    return _return_stream_response(response, deadline, session)


def _complete_impl(
    model: Union[str, snowpark.Column],
    prompt: Union[str, list[ConversationMessage], snowpark.Column],
    snow_api_xp_request_handler: Optional[Callable[..., dict[str, Any]]] = None,
    function: str = "snowflake.cortex.complete",
    options: Optional[CompleteOptions] = None,
    session: Optional[snowpark.Session] = None,
    stream: bool = False,
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
) -> Union[str, Iterator[str], snowpark.Column]:
    if timeout is not None and deadline is not None:
        raise ValueError('only one of "timeout" and "deadline" must be set')
    if timeout is not None:
        deadline = time.time() + timeout
    if stream:
        if not isinstance(model, str):
            raise ValueError("in REST mode, 'model' must be a string")
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError("in REST mode, 'prompt' must be a string or a list of ConversationMessage")
        return _complete_rest(
            snow_api_xp_request_handler=snow_api_xp_request_handler,
            model=model,
            prompt=prompt,
            options=options,
            session=session,
            deadline=deadline,
        )
    return _complete_non_streaming_impl(
        snow_api_xp_request_handler=snow_api_xp_request_handler,
        function=function,
        model=model,
        prompt=prompt,
        options=options,
        session=session,
        deadline=deadline,
    )


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def complete(
    model: Union[str, snowpark.Column],
    prompt: Union[str, list[ConversationMessage], snowpark.Column],
    *,
    options: Optional[CompleteOptions] = None,
    session: Optional[snowpark.Session] = None,
    stream: bool = False,
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
) -> Union[str, Iterator[str], snowpark.Column]:
    """Complete calls into the LLM inference service to perform completion.

    Args:
        model: A Column of strings representing model types.
        prompt: A Column of prompts to send to the LLM.
        options: A instance of snowflake.cortex.CompleteOptions
        session: The snowpark session to use. Will be inferred by context if not specified.
        stream (bool): Enables streaming. When enabled, a generator function is returned that provides the streaming
            output as it is received. Each update is a string containing the new text content since the previous update.
        timeout (float): Timeout in seconds to retry failed REST requests.
        deadline (float): Time in seconds since the epoch (as returned by time.time()) to retry failed REST requests.

    Raises:
        ValueError: incorrect argument.

    Returns:
        A column of string responses.
    """

    # Set the XP snow api function, if available.
    snow_api_xp_request_handler = None
    if is_in_stored_procedure():  # type: ignore[no-untyped-call]
        import _snowflake

        snow_api_xp_request_handler = _snowflake.send_snow_api_request

    try:
        return _complete_impl(
            model,
            prompt,
            snow_api_xp_request_handler=snow_api_xp_request_handler,
            options=options,
            session=session,
            stream=stream,
            timeout=timeout,
            deadline=deadline,
        )
    except ValueError as err:
        raise err


Complete = deprecated("Complete() is deprecated and will be removed in a future release. Use complete() instead")(
    telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(complete)
)
