from typing import cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    call_sql_function,
    warn_cortex_deprecated,
)
from snowflake.ml._internal import telemetry

_AI_SUMMARIZE_FUNCTION_NAME = "AI_SUMMARIZE"


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def summarize(
    text: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    """Calls into the LLM inference service to summarize the input text.

    Args:
        text: A Column of strings to summarize.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of string summaries.
    """
    warn_cortex_deprecated("summarize", "ai_complete")
    return _summarize_impl(_AI_SUMMARIZE_FUNCTION_NAME, text, session=session)


def _summarize_impl(
    function: str,
    text: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    return cast(str | snowpark.Column, call_sql_function(function, session, text))


Summarize = deprecated("Summarize() is deprecated and will be removed in a future release. Use summarize() instead")(
    telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(summarize)
)
