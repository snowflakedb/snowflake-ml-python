import json
from typing import Any, cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    call_sql_function,
    warn_cortex_deprecated,
)
from snowflake.ml._internal import telemetry
from snowflake.snowpark import functions

_AI_CLASSIFY_FUNCTION_NAME = "AI_CLASSIFY"
# Legacy CLASSIFY_TEXT always returned exactly one label. AI_CLASSIFY defaults to single-label mode, but
# we set output_mode explicitly so the single-label contract can't drift and multi-label results can
# never leak through.
_AI_CLASSIFY_SINGLE_LABEL_CONFIG = {"output_mode": "single"}


def _adapt_ai_classify_result(result: Any) -> str:
    """Map an AI_CLASSIFY result to the legacy CLASSIFY_TEXT {"label": ...} JSON string."""
    # Legacy classify_text returned the raw CLASSIFY_TEXT OBJECT, which the immediate (non-Column) path
    # collects as a JSON string such as '{"label": "positive"}'. We reproduce that exact shape so
    # existing callers that json.loads the result and read ["label"] keep working.
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
        A column of classification responses.
    """
    warn_cortex_deprecated("classify_text", "ai_classify")
    result = _classify_text_impl(
        _AI_CLASSIFY_FUNCTION_NAME, str_input, categories, _AI_CLASSIFY_SINGLE_LABEL_CONFIG, session=session
    )
    if isinstance(result, snowpark.Column):
        # AI_CLASSIFY returns {"labels": [...]}; re-wrap as the legacy CLASSIFY_TEXT {"label": ...} OBJECT
        # so downstream SQL that reads col["label"] is unchanged. Subscript access (not functions.get with
        # a bare string, which Snowpark treats as a column identifier) extracts the field by literal key.
        label = functions.coalesce(result["labels"][0], result["label"])
        return functions.object_construct(functions.lit("label"), label)
    return _adapt_ai_classify_result(result)


def _classify_text_impl(
    function: str,
    str_input: str | snowpark.Column,
    categories: list[str] | snowpark.Column,
    config: dict[str, Any] | None = None,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    args: list[Any] = [str_input, categories]
    if config is not None:
        args.append(config)
    return cast(str | snowpark.Column, call_sql_function(function, session, *args))


ClassifyText = deprecated(
    "ClassifyText() is deprecated and will be removed in a future release. Please use classify_text() instead."
)(
    telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    )(classify_text)
)
