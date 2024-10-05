from typing import Dict, List, Optional, Union, cast

from snowflake import snowpark
from snowflake.snowpark import context, functions

CORTEX_FUNCTIONS_TELEMETRY_PROJECT = "CortexFunctions"


class SnowflakeAuthenticationException(Exception):
    """This exception is raised when there is an issue with Snowflake's configuration."""

    pass


class SnowflakeConfigurationException(Exception):
    """This exception is raised when there is an issue with Snowflake's configuration."""

    pass


# Calls a sql function, handling both immediate (e.g. python types) and batch
# (e.g. snowpark column and literal type modes).
def call_sql_function(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, List[str], snowpark.Column, Dict[str, Union[int, float]]],
) -> Union[str, List[float], snowpark.Column]:
    handle_as_column = False

    for arg in args:
        if isinstance(arg, snowpark.Column):
            handle_as_column = True

    if handle_as_column:
        return cast(Union[str, List[float], snowpark.Column], _call_sql_function_column(function, *args))
    return cast(
        Union[str, List[float], snowpark.Column],
        _call_sql_function_immediate(function, session, *args),
    )


def _call_sql_function_column(
    function: str, *args: Union[str, List[str], snowpark.Column, Dict[str, Union[int, float]]]
) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def _call_sql_function_immediate(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, List[str], snowpark.Column, Dict[str, Union[int, float]]],
) -> Union[str, List[float]]:
    session = session or context.get_active_session()
    if session is None:
        raise SnowflakeAuthenticationException(
            """Session required. Provide the session through a session=... argument or ensure an active session is
            available in your environment."""
        )

    lit_args = []
    for arg in args:
        lit_args.append(functions.lit(arg))

    empty_df = session.create_dataframe([snowpark.Row()])
    df = empty_df.select(functions.builtin(function)(*lit_args))
    return cast(str, df.collect()[0][0])
