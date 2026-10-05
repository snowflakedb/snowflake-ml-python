from typing import cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    SnowflakeAuthenticationException,
    warn_cortex_deprecated,
)
from snowflake.ml._internal import telemetry
from snowflake.snowpark import context, functions, types

_AI_COMPLETE_FUNCTION_NAME = "AI_COMPLETE"

# AI_SENTIMENT returns a categorical label (positive/negative/neutral/mixed), not the legacy FLOAT in
# [-1, 1] that SNOWFLAKE.CORTEX.SENTIMENT produced. To keep the numeric return contract we score the
# text with AI_COMPLETE and a constrained prompt instead. The score is model-derived, so exact values
# differ from the legacy sentiment model.
_SENTIMENT_MODEL = "llama3.1-8b"
_SENTIMENT_PROMPT_PREFIX = (
    "Rate the sentiment of the following text from -1.0 (very negative) to 1.0 (very positive). "
    "Respond with only the score. Text: "
)
_SENTIMENT_MODEL_PARAMETERS = {"temperature": 0}
_SENTIMENT_RESPONSE_FORMAT = {
    "type": "json",
    "schema": {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
}


def _sentiment_score_column(text: snowpark.Column) -> snowpark.Column:
    """Build the AI_COMPLETE sentiment-score expression, clamped to the legacy [-1, 1] range."""
    prompt = functions.concat(functions.lit(_SENTIMENT_PROMPT_PREFIX), text)
    completion = functions.builtin(_AI_COMPLETE_FUNCTION_NAME)(
        functions.lit(_SENTIMENT_MODEL),
        prompt,
        _SENTIMENT_MODEL_PARAMETERS,
        _SENTIMENT_RESPONSE_FORMAT,
    )
    score = functions.get(functions.parse_json(completion), functions.lit("score")).cast(types.FloatType())
    return functions.least(functions.lit(1.0), functions.greatest(functions.lit(-1.0), score))


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def sentiment(text: str | snowpark.Column, session: snowpark.Session | None = None) -> float | snowpark.Column:
    """Calls into the LLM inference service to perform sentiment analysis on the input text.

    Args:
        text: A Column of text strings to send to the LLM.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of floats. 1 represents positive sentiment, -1 represents negative sentiment.

    Raises:
        SnowflakeAuthenticationException: if no session is provided and none is active.
    """
    warn_cortex_deprecated("sentiment", "ai_complete")
    if isinstance(text, snowpark.Column):
        return _sentiment_score_column(text)

    session = session or context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )
    score_column = _sentiment_score_column(functions.lit(text))
    output = session.create_dataframe([snowpark.Row()]).select(score_column).collect()[0][0]
    return float(cast(str, output))


Sentiment = deprecated("Sentiment() is deprecated and will be removed in a future release. Use sentiment() instead")(
    telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(sentiment)
)
