from typing import Any, Dict, List, Tuple

from snowflake import snowpark
from snowflake.ml._internal.utils import formatting, query_result_checker

"""Table_manager is a set of utils that helps create tables.

TODO: We should make table manager a class and then put the following functions as public methods.
      Class constructor should take the session. Potentially db, schema as well.
"""


def get_fully_qualified_schema_name(database_name: str, schema_name: str) -> str:
    return f"{database_name}.{schema_name}"


def get_fully_qualified_table_name(database_name: str, schema_name: str, table_name: str) -> str:
    return f"{get_fully_qualified_schema_name(database_name, schema_name)}.{table_name}"


def create_single_registry_table(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    table_name: str,
    table_schema: List[Tuple[str, str]],
    statement_params: Dict[str, Any],
) -> str:
    """Creates a single table for registry and returns the fully qualified name of the table.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        table_name: Name of the target table.
        table_schema: A list of pair of strings, each pair denotes `(<column name>, <column type>)`.
        statement_params: Function usage statement parameters used in sql query executions.

    Returns:
        A string which is the name of the created table.

    Raises:
        RuntimeError: If table creation failed.
    """
    fully_qualified_table_name = get_fully_qualified_table_name(database_name, schema_name, table_name)
    table_schema_string = ", ".join([f"{k} {v}" for k, v in table_schema])
    try:
        session.sql(f"CREATE TABLE IF NOT EXISTS {fully_qualified_table_name} ({table_schema_string})").collect(
            statement_params=statement_params
        )
    except Exception as e:
        raise RuntimeError(f"Registry table {fully_qualified_table_name} creation failed due to {e}")

    return fully_qualified_table_name


def insert_table_entry(session: snowpark.Session, table: str, columns: Dict[str, Any]) -> List[snowpark.Row]:
    """Insert an entry into an internal Model Registry table.

    Args:
        session: Snowpark session object to communicate with Snowflake.
        table: Fully qualified name of the table to insert into.
        columns: Key-value pairs of columns and values to be inserted into the table.

    Returns:
        Result of the operation as returned by the Snowpark session (snowpark.DataFrame).

    Raises:
        RuntimeError: If entry insertion failed.
    """
    sorted_columns = sorted(columns.items())
    try:
        sql = "INSERT INTO {table} ( {columns} ) SELECT {values}".format(
            table=table,
            columns=",".join([x[0] for x in sorted_columns]),
            values=",".join([formatting.format_value_for_select(x[1]) for x in sorted_columns]),
        )
        return query_result_checker.SqlResultValidator(session, sql).insertion_success(expected_num_rows=1).validate()
    except Exception as e:
        raise RuntimeError(f"Table {table} entry {columns} insertion failed due to {e}")


def validate_table_exist(session: snowpark.Session, table: str, qualified_schema_name: str) -> bool:
    """Check if the given table exists in the target schema.

    Note:
        In case the table doesn't exist, a DataError will be raised by SqlResultValidator.

    Args:
        session: Snowpark session object to communicate with Snowflake.
        table: Name of the target table as an identifier.
        qualified_schema_name: Fully qualidied schema name where the target table is expected to exist.

    Returns:
        A boolean stands for whether the target table already exists.
    """
    tables = session.sql(f"SHOW TABLES LIKE '{table}' IN {qualified_schema_name}").collect()
    return len(tables) == 1
