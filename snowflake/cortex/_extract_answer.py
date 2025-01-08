from typing import Optional, Union, cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def extract_answer(
    from_text: Union[str, snowpark.Column],
    question: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """Calls into the LLM inference service to extract an answer from within specified text.

    Args:
        from_text: A Column of strings representing input text.
        question: A Column of strings representing a question to ask against from_text.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of strings containing answers.
    """

    return _extract_answer_impl("snowflake.cortex.extract_answer", from_text, question, session=session)


def _extract_answer_impl(
    function: str,
    from_text: Union[str, snowpark.Column],
    question: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return cast(Union[str, snowpark.Column], call_sql_function(function, session, from_text, question))


ExtractAnswer = deprecated(
    "ExtractAnswer() is deprecated and will be removed in a future release. Use extract_answer() instead"
)(telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT)(extract_answer))
