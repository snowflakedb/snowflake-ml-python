from typing import Optional, Union, cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def sentiment(
    text: Union[str, snowpark.Column], session: Optional[snowpark.Session] = None
) -> Union[float, snowpark.Column]:
    """Calls into the LLM inference service to perform sentiment analysis on the input text.

    Args:
        text: A Column of text strings to send to the LLM.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of floats. 1 represents positive sentiment, -1 represents negative sentiment.
    """
    return _sentiment_impl("snowflake.cortex.sentiment", text, session=session)


def _sentiment_impl(
    function: str, text: Union[str, snowpark.Column], session: Optional[snowpark.Session] = None
) -> Union[float, snowpark.Column]:

    output = call_sql_function(function, session, text)
    if isinstance(output, snowpark.Column):
        return output
    return float(cast(str, output))


Sentiment = deprecated("Sentiment() is deprecated and will be removed in a future release. Use sentiment() instead")(
    telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(sentiment)
)
