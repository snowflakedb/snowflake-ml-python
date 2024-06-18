from typing import Iterator, Optional, Union

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    call_rest_function,
    call_sql_function,
    process_rest_response,
)
from snowflake.ml._internal import telemetry


@snowpark._internal.utils.experimental(version="1.0.12")
@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def Complete(
    model: Union[str, snowpark.Column],
    prompt: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
    use_rest_api_experimental: bool = False,
    stream: bool = False,
) -> Union[str, Iterator[str], snowpark.Column]:
    """Complete calls into the LLM inference service to perform completion.

    Args:
        model: A Column of strings representing model types.
        prompt: A Column of prompts to send to the LLM.
        session: The snowpark session to use. Will be inferred by context if not specified.
        use_rest_api_experimental (bool): Toggles between the use of SQL and REST implementation. This feature is
            experimental and can be removed at any time.
        stream (bool): Enables streaming. When enabled, a generator function is returned that provides the streaming
            output as it is received. Each update is a string containing the new text content since the previous update.
            The use of streaming requires the experimental use_rest_api_experimental flag to be enabled.

    Raises:
        ValueError: If `stream` is set to True and `use_rest_api_experimental` is set to False.

    Returns:
        A column of string responses.
    """
    if stream is True and use_rest_api_experimental is False:
        raise ValueError("If stream is set to True use_rest_api_experimental must also be set to True")
    if use_rest_api_experimental:
        response = call_rest_function("complete", model, prompt, session=session, stream=stream)
        return process_rest_response(response)
    return _complete_impl("snowflake.cortex.complete", model, prompt, session=session)


def _complete_impl(
    function: str,
    model: Union[str, snowpark.Column],
    prompt: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return call_sql_function(function, session, model, prompt)
