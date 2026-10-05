from typing import cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    call_sql_function,
    warn_cortex_deprecated,
)
from snowflake.ml._internal import telemetry

_AI_EMBED_FUNCTION_NAME = "AI_EMBED"


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def embed_text_1024(
    model: str | snowpark.Column,
    text: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> list[float] | snowpark.Column:
    """Calls into the LLM inference service to embed the text.

    Args:
        model: A Column of strings representing the model to use for embedding. The value
               of the strings must be within the SUPPORTED_MODELS list.
        text: A Column of strings representing input text.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of vectors containing embeddings.
    """
    warn_cortex_deprecated("embed_text_1024", "ai_embed")
    return _embed_text_1024_impl(_AI_EMBED_FUNCTION_NAME, model, text, session=session)


def _embed_text_1024_impl(
    function: str,
    model: str | snowpark.Column,
    text: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> list[float] | snowpark.Column:
    return cast(list[float] | snowpark.Column, call_sql_function(function, session, model, text))


EmbedText1024 = deprecated(
    "EmbedText1024() is deprecated and will be removed in a future release. Use embed_text_1024() instead"
)(telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(embed_text_1024))
