from typing import Optional, Union

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@snowpark._internal.utils.experimental(version="1.0.12")
@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def Summarize(
    text: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """Summarize calls into the LLM inference service to summarize the input text.

    Args:
        text: A Column of strings to summarize.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of string summaries.
    """

    return _summarize_impl("snowflake.cortex.summarize", text, session=session)


def _summarize_impl(
    function: str,
    text: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return call_sql_function(function, session, text)
