from typing import Optional, Union

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@snowpark._internal.utils.experimental(version="1.0.12")
@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def ExtractAnswer(
    from_text: Union[str, snowpark.Column],
    question: Union[str, snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """ExtractAnswer calls into the LLM inference service to extract an answer from within specified text.

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
    return call_sql_function(function, session, from_text, question)
