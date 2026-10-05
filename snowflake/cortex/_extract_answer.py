from typing import Any, cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    SnowflakeAuthenticationException,
    warn_cortex_deprecated,
)
from snowflake.ml._internal import telemetry
from snowflake.snowpark import context, functions

_AI_EXTRACT_FUNCTION_NAME = "AI_EXTRACT"
_ANSWER_KEY = "answer"


def _legacy_extract_answer_column(
    from_text: str | snowpark.Column,
    response_format: dict[str, Any] | snowpark.Column,
) -> snowpark.Column:
    """Call AI_EXTRACT and reshape it to the legacy EXTRACT_ANSWER [{"answer": ..., "score": ...}] ARRAY."""
    # AI_EXTRACT returns the score only when the `scores` argument is TRUE, and `scores` is a named-only
    # argument that the Snowpark builtin wrapper can't pass positionally -- inject it as a raw SQL fragment.
    # AI_EXTRACT(<text>, <response_format>, scores => TRUE) returns:
    #   {"response": {"answer": ...}, "scoring": {"scores": {"answer": {"score": ...}}}}
    extracted = functions.builtin(_AI_EXTRACT_FUNCTION_NAME)(
        from_text, response_format, functions.sql_expr("scores => TRUE")
    )
    # Subscript access (not functions.get with a bare string, which Snowpark treats as a column
    # identifier) extracts fields by literal key.
    answer = extracted["response"][_ANSWER_KEY]
    score = extracted["scoring"]["scores"][_ANSWER_KEY]["score"]
    return functions.array_construct(
        functions.object_construct(functions.lit(_ANSWER_KEY), answer, functions.lit("score"), score)
    )


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def extract_answer(
    from_text: str | snowpark.Column,
    question: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    """Calls into the LLM inference service to extract an answer from within specified text.

    Args:
        from_text: A Column of strings representing input text.
        question: A Column of strings representing a question to ask against from_text.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of arrays, each [{"answer": ..., "score": ...}], matching legacy EXTRACT_ANSWER.

    Raises:
        SnowflakeAuthenticationException: if no session is provided and none is active.
    """
    warn_cortex_deprecated("extract_answer", "ai_extract")
    if isinstance(question, snowpark.Column):
        response_format: dict[str, Any] | snowpark.Column = functions.object_construct(
            functions.lit(_ANSWER_KEY), question
        )
    else:
        response_format = {_ANSWER_KEY: question}

    column = _legacy_extract_answer_column(from_text, response_format)
    if isinstance(from_text, snowpark.Column) or isinstance(question, snowpark.Column):
        return column

    # Immediate (literal) path: evaluate the expression and return the collected ARRAY as a JSON string,
    # exactly as the legacy EXTRACT_ANSWER immediate path did.
    session = session or context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )
    return cast(str, session.create_dataframe([snowpark.Row()]).select(column).collect()[0][0])


ExtractAnswer = deprecated(
    "ExtractAnswer() is deprecated and will be removed in a future release. Use extract_answer() instead"
)(telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(extract_answer))
