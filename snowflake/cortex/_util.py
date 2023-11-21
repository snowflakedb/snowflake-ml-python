from typing import Optional, Union, cast

from snowflake import snowpark
from snowflake.snowpark import context, functions

CORTEX_FUNCTIONS_TELEMETRY_PROJECT = "CortexFunctions"


# Calls a sql function, handling both immediate (e.g. python types) and batch
# (e.g. snowpark column and literal type modes).
def call_sql_function(
    function: str, session: Optional[snowpark.Session], *args: Union[str, snowpark.Column]
) -> Union[str, snowpark.Column]:
    handle_as_column = False
    for arg in args:
        if isinstance(arg, snowpark.Column):
            handle_as_column = True

    if handle_as_column:
        return cast(Union[str, snowpark.Column], call_sql_function_column(function, *args))
    return cast(Union[str, snowpark.Column], call_sql_function_immediate(function, session, *args))


def call_sql_function_column(function: str, *args: Union[str, snowpark.Column]) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def call_sql_function_immediate(
    function: str, session: Optional[snowpark.Session], *args: Union[str, snowpark.Column]
) -> str:
    if session is None:
        session = context.get_active_session()
    if session is None:
        raise Exception("No session available in the current context nor specified as an argument.")

    lit_args = []
    for arg in args:
        lit_args.append(functions.lit(arg))

    empty_df = session.create_dataframe([snowpark.Row()])
    df = empty_df.select(functions.builtin(function)(*lit_args))
    return cast(str, df.collect()[0][0])
