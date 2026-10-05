from typing import cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    call_sql_function,
    warn_cortex_deprecated,
)
from snowflake.ml._internal import telemetry

_AI_TRANSLATE_FUNCTION_NAME = "AI_TRANSLATE"


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def translate(
    text: str | snowpark.Column,
    from_language: str | snowpark.Column,
    to_language: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    """Calls into the LLM inference service to perform translation.

    Args:
        text: A Column of strings to translate.
        from_language: A Column of input languages.
        to_language: A Column of output languages.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of string translations.
    """
    warn_cortex_deprecated("translate", "ai_translate")
    return _translate_impl(_AI_TRANSLATE_FUNCTION_NAME, text, from_language, to_language, session=session)


def _translate_impl(
    function: str,
    text: str | snowpark.Column,
    from_language: str | snowpark.Column,
    to_language: str | snowpark.Column,
    session: snowpark.Session | None = None,
) -> str | snowpark.Column:
    return cast(str | snowpark.Column, call_sql_function(function, session, text, from_language, to_language))


Translate = deprecated("Translate() is deprecated and will be removed in a future release. Use translate() instead")(
    telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(translate)
)
