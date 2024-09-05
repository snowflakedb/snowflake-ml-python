from typing import Optional, Union, List

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    SnowflakeConfigurationException,
    call_sql_function,
)
from snowflake.ml._internal import telemetry


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def EmbedText768(
    model: Union[str, snowpark.Column],
    text: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """TextEmbed calls into the LLM inference service to embed the text.

    Args:
        model: A Column of strings representing the model to use for embedding. The value
               of the strings must be within the SUPPORTED_MODELS list.
        text: A Column of strings representing input text.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of vectors containing embeddings.
    """

    return _embed_text_768_impl(
        "snowflake.cortex.embed_text_768", model, text, session=session
    )


def _embed_text_768_impl(
    function: str,
    model: Union[str, snowpark.Column],
    text: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return call_sql_function(function, session, model, text)
