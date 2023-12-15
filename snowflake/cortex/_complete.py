from typing import Optional, Union

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@snowpark._internal.utils.experimental(version="1.0.12")
@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def Complete(
    model: Union[str, snowpark.Column], prompt: Union[str, snowpark.Column], session: Optional[snowpark.Session] = None
) -> Union[str, snowpark.Column]:
    """Complete calls into the LLM inference service to perform completion.

    Args:
        model: A Column of strings representing model types.
        prompt: A Column of prompts to send to the LLM.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of string responses.
    """

    return _complete_impl("snowflake.cortex.complete", model, prompt, session=session)


def _complete_impl(
    function: str,
    model: Union[str, snowpark.Column],
    prompt: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return call_sql_function(function, session, model, prompt)
