from typing import Any, Optional, Union, cast

from snowflake import snowpark
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import formatting
from snowflake.snowpark import context, functions

CORTEX_FUNCTIONS_TELEMETRY_PROJECT = "CortexFunctions"


class SnowflakeAuthenticationException(Exception):
    """This exception is raised when there is an issue with Snowflake's configuration."""


class SnowflakeConfigurationException(Exception):
    """This exception is raised when there is an issue with Snowflake's configuration."""


# Calls a sql function, handling both immediate (e.g. python types) and batch
# (e.g. snowpark column and literal type modes).
def call_sql_function(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, list[str], snowpark.Column, dict[str, Union[int, float]]],
) -> Union[str, list[float], snowpark.Column]:
    handle_as_column = False

    for arg in args:
        if isinstance(arg, snowpark.Column):
            handle_as_column = True

    if handle_as_column:
        return cast(Union[str, list[float], snowpark.Column], _call_sql_function_column(function, *args))
    return cast(
        Union[str, list[float], snowpark.Column],
        _call_sql_function_immediate(function, session, *args),
    )


def _call_sql_function_column(
    function: str, *args: Union[str, list[str], snowpark.Column, dict[str, Union[int, float]]]
) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def _call_sql_function_immediate(
    function: str,
    session: Optional[snowpark.Session],
    *args: Union[str, list[str], snowpark.Column, dict[str, Union[int, float]]],
) -> Union[str, list[float]]:
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


def call_sql_function_literals(function: str, session: Optional[snowpark.Session], *args: Any) -> str:
    r"""Call a SQL function with only literal arguments.

    This is useful for calling system functions.

    Args:
        function: The name of the function to be called.
        session: The Snowpark session to use.
        *args: The list of arguments

    Returns:
        String value that corresponds the the first cell in the dataframe.

    Raises:
        SnowflakeMLException: If no session is given and no active session exists.
    """
    if session is None:
        session = context.get_active_session()
    if session is None:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_SNOWPARK_SESSION,
        )

    function_arguments = ",".join(["NULL" if arg is None else formatting.format_value_for_select(arg) for arg in args])
    return cast(str, session.sql(f"SELECT {function}({function_arguments})").collect()[0][0])
