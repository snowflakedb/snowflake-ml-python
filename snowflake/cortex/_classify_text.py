from typing import Optional, Union, cast

from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT, call_sql_function
from snowflake.ml._internal import telemetry


@telemetry.send_api_usage_telemetry(
    project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
)
def classify_text(
    str_input: Union[str, snowpark.Column],
    categories: Union[list[str], snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    """Use the LLM inference service to classify the INPUT text into one of the target CATEGORIES.

    Args:
        str_input: A Column of strings to classify.
        categories: A list of candidate categories to classify the INPUT text into.
        session: The snowpark session to use. Will be inferred by context if not specified.

    Returns:
        A column of classification responses.
    """

    return _classify_text_impl("snowflake.cortex.classify_text", str_input, categories, session=session)


def _classify_text_impl(
    function: str,
    str_input: Union[str, snowpark.Column],
    categories: Union[list[str], snowpark.Column],
    session: Optional[snowpark.Session] = None,
) -> Union[str, snowpark.Column]:
    return cast(Union[str, snowpark.Column], call_sql_function(function, session, str_input, categories))


ClassifyText = deprecated(
    "ClassifyText() is deprecated and will be removed in a future release. Please use classify_text() instead."
)(
    telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    )(classify_text)
)
