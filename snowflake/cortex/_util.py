import warnings
from typing import Any, Union, cast

from snowflake import snowpark
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import formatting
from snowflake.snowpark import context, functions

CORTEX_FUNCTIONS_TELEMETRY_PROJECT = "CortexFunctions"


_SNOWPARK_FUNCTIONS_DOC_URL = (
    "https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/"
    "snowpark/api/snowflake.snowpark.functions.{replacement}"
)
_MIGRATION_GUIDE_URL = "https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql-migrate-legacy-functions"

# Functions that have already emitted their deprecation warning this process. Guarding on this keeps a
# repeated call (e.g. inside a loop) from spamming the console -- each deprecated function warns once.
_emitted_deprecations: set[str] = set()


def warn_cortex_deprecated(name: str, replacement: str) -> None:
    """Emit the deprecation warning for the named deprecated snowflake.cortex function, once per process.

    Args:
        name: The deprecated snowflake.cortex function name (e.g. "classify_text").
        replacement: The snowflake.snowpark.functions replacement to point callers at (e.g. "ai_classify").
    """
    # Warn on use (not on package import) so callers of functions that are NOT deprecated -- e.g.
    # Finetune -- are never warned.
    if name in _emitted_deprecations:
        return
    _emitted_deprecations.add(name)
    warnings.warn(
        f"snowflake.cortex.{name} is deprecated and will be removed in a future release. "
        f"Its backend is now redirected to Snowflake AI Functions. "
        f"Use snowflake.snowpark.functions.{replacement} instead. "
        f"API: {_SNOWPARK_FUNCTIONS_DOC_URL.format(replacement=replacement)} "
        f"Migration guide: {_MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=3,
    )


class SnowflakeAuthenticationException(Exception):
    """This exception is raised when there is an issue with Snowflake's configuration."""


class SnowflakeConfigurationException(Exception):
    """This exception is raised when there is an issue with Snowflake's configuration."""


# Calls a sql function, handling both immediate (e.g. python types) and batch
# (e.g. snowpark column and literal type modes).
def call_sql_function(
    function: str,
    session: snowpark.Session | None,
    *args: str | list[str] | snowpark.Column | dict[str, int | float],
) -> str | list[float] | snowpark.Column:
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
    function: str, *args: str | list[str] | snowpark.Column | dict[str, int | float]
) -> snowpark.Column:
    return cast(snowpark.Column, functions.builtin(function)(*args))


def _call_sql_function_immediate(
    function: str,
    session: snowpark.Session | None,
    *args: str | list[str] | snowpark.Column | dict[str, int | float],
) -> str | list[float]:
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


def call_sql_function_literals(function: str, session: snowpark.Session | None, *args: Any) -> str:
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
