from enum import Enum
from typing import Any, Optional

from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.snowpark import session

MAX_IDENTIFIER_LENGTH = 255


class SnowflakeDbObjectType(Enum):
    TABLE = "TABLE"
    WAREHOUSE = "WAREHOUSE"


def db_object_exists(
    session: session.Session,
    object_type: SnowflakeDbObjectType,
    object_name: sql_identifier.SqlIdentifier,
    *,
    database_name: Optional[sql_identifier.SqlIdentifier] = None,
    schema_name: Optional[sql_identifier.SqlIdentifier] = None,
    statement_params: Optional[dict[str, Any]] = None,
) -> bool:
    """Check if object exists in database.

    Args:
        session: Active Snowpark Session.
        object_type: Type of object to search for.
        object_name: Name of object to search for.
        database_name: Optional database name to search in. Only used if both schema is also provided.
        schema_name: Optional schema to search in.
        statement_params: Optional set of statement_params to include with queries.

    Returns:
        boolean indicating whether object exists.
    """
    optional_in_clause = ""
    if database_name and schema_name:
        optional_in_clause = f" IN {database_name}.{schema_name}"

    result = (
        query_result_checker.SqlResultValidator(
            session,
            f"""SHOW {object_type.value}S LIKE '{object_name}'{optional_in_clause}""",
            statement_params=statement_params,
        )
        .has_column("name", allow_empty=True)  # TODO: Check this is actually what is returned from server
        .validate()
    )
    return len(result) == 1
