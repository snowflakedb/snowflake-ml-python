from typing import Optional, Union

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@snowpark._internal.utils.experimental(version="1.0.12")
@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def Translate(
    text: Union[str, snowpark.Column],
    from_language: Union[str, snowpark.Column],
    to_language: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """Translate calls into the LLM inference service to perform translation.

    Args:
        text: A Column of strings to translate.
        from_language: A Column of input languages.
        to_language: A Column of output languages.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of string translations.
    """

    return _translate_impl("snowflake.cortex.translate", text, from_language, to_language, session=session)


def _translate_impl(
    function: str,
    text: Union[str, snowpark.Column],
    from_language: Union[str, snowpark.Column],
    to_language: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return call_sql_function(function, session, text, from_language, to_language)
