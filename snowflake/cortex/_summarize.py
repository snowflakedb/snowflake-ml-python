from typing import Optional, Union, cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def summarize(
    text: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """Calls into the LLM inference service to summarize the input text.

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
    return cast(Union[str, snowpark.Column], call_sql_function(function, session, text))


Summarize = deprecated("Summarize() is deprecated and will be removed in a future release. Use summarize() instead")(
    telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(summarize)
)
