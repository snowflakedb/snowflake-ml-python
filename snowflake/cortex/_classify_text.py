import json
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

_AI_CLASSIFY_FUNCTION_NAME = "AI_CLASSIFY"
# AI_CLASSIFY requires its config argument to be a constant object literal. A Python dict is rendered by
# Snowpark as PARSE_JSON('...')::OBJECT, which the server rejects ("... is not a constant"), so emit the
# config as a SQL object constant via sql_expr -- this mirrors snowflake.snowpark.functions.ai_classify.
# Legacy CLASSIFY_TEXT was always single-label, so set output_mode explicitly to keep that contract.
_AI_CLASSIFY_SINGLE_LABEL_CONFIG_SQL = "{'output_mode': 'single'}"


def _adapt_ai_classify_result(result: Any) -> str:
    """Map an AI_CLASSIFY result to the legacy CLASSIFY_TEXT {"label": ...} JSON string."""
    # Used for non-Column results (e.g. a dict/JSON string): reproduce the legacy single-label object shape.
    parsed: Any = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result
    label: Any = None
    if isinstance(parsed, dict):
        labels = parsed.get("labels")
        if isinstance(labels, list) and labels:
            label = labels[0]
        elif parsed.get("label") is not None:
            label = parsed.get("label")
    return json.dumps({"label": label})


def _ai_classify_label_column(
    str_input: str | snowpark.Column,
    categories: list[str] | snowpark.Column,
) -> snowpark.Column:
    """Call AI_CLASSIFY in single-label mode and re-wrap to the legacy CLASSIFY_TEXT {"label": ...} OBJECT."""
    result = functions.builtin(_AI_CLASSIFY_FUNCTION_NAME)(
        str_input, categories, functions.sql_expr(_AI_CLASSIFY_SINGLE_LABEL_CONFIG_SQL)
    )
    # AI_CLASSIFY returns {"labels": [...]}; collapse to the legacy single {"label": ...}. Subscript access
    # (not functions.get with a bare string, which Snowpark treats as a column identifier) uses literal keys.
    label = functions.coalesce(result["labels"][0], result["label"])
    return functions.object_construct(functions.lit("label"), label)


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def classify_text(
    str_input: str | snowpark.Column,
    categories: list[str] | snowpark.Column,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    """Use the LLM inference service to classify the INPUT text into one of the target CATEGORIES.

    Args:
        str_input: A Column of strings to classify.
        categories: A list of candidate categories to classify the INPUT text into.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of classification responses (legacy CLASSIFY_TEXT {"label": ...} shape).

    Raises:
        SnowflakeAuthenticationException: if no session is provided and none is active.
    """
    warn_cortex_deprecated("classify_text", "ai_classify")
    column = _ai_classify_label_column(str_input, categories)
    if isinstance(str_input, snowpark.Column) or isinstance(categories, snowpark.Column):
        return column

    # Immediate (literal) path: evaluate and return the collected OBJECT as a JSON string, exactly as the
    # legacy CLASSIFY_TEXT immediate path did.
    session = session or context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )
    return cast(str, session.create_dataframe([snowpark.Row()]).select(column).collect()[0][0])


ClassifyText = deprecated(
    "ClassifyText() is deprecated and will be removed in a future release. Please use classify_text() instead."
)(
    telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    )(classify_text)
)
