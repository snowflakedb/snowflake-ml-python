"""SQL string factories for feature store object operations.

No IO — no SQL execution, no HTTP calls, no file writes.
All functions accept connection context (database, schema) as plain string
arguments and return SQL strings for the CLI to execute.
"""

from __future__ import annotations


def state_queries(database: str, schema: str) -> dict[str, str]:
    """Return SQL strings for fetching applied state.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        Dict with keys: ``show_ofts``, ``show_tables``,
        ``show_dynamic_tables`` (used to recover the offline-DT ``text``
        column so :func:`state.fetch_applied_state` can re-derive the
        BatchFV ``sources[0].table`` binding the deployed SPECIFICATION
        JSON loses on the FROM SPECIFICATION round-trip),
        ``describe_specification_template`` (with ``{name}`` placeholder
        for use with ``.format(name=...)``).
    """
    location = f"{database}.{schema}"
    return {
        "show_ofts": f"SHOW ONLINE FEATURE TABLES IN SCHEMA {location}",
        "show_tables": f"SHOW TABLES LIKE '%' IN SCHEMA {location}",
        "show_dynamic_tables": dynamic_tables_query(database, schema),
        "describe_specification_template": (
            f'DESCRIBE ONLINE FEATURE TABLE "{database}"."{schema}"."{{name}}" ' "TYPE = SPECIFICATION"
        ),
    }


def dynamic_tables_query(database: str, schema: str) -> str:
    """Return the ``SHOW DYNAMIC TABLES IN SCHEMA <db>.<schema>`` SQL string.

    The declarative state-fetcher uses the offline DT's ``text`` column —
    which carries the original ``CREATE DYNAMIC TABLE … AS SELECT * FROM
    <db>.<schema>.<table>`` body — to recover the BatchFV source-table
    binding that the deployed ``DESCRIBE ONLINE FEATURE TABLE … TYPE =
    SPECIFICATION`` JSON drops (it always returns ``sources: []`` for
    BatchFVs because snowml-core's FROM SPECIFICATION serializer encodes
    the source binding into the DT's SELECT instead of the spec
    payload).  Issuing one schema-wide ``SHOW DYNAMIC TABLES`` call per
    state fetch keeps the overhead bounded (a single network round trip
    regardless of how many OFTs are deployed) and avoids leaking SQL
    construction into ``state.py`` or ``manager.py``.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        SQL string to list every Dynamic Table in the schema along
        with its ``text`` column (the ``CREATE DYNAMIC TABLE`` body).
    """
    return f"SHOW DYNAMIC TABLES IN SCHEMA {database}.{schema}"


def list_query(database: str, schema: str) -> str:
    """Return the SHOW ONLINE FEATURE TABLES SQL string.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        SQL string to list all online feature tables in the schema.
    """
    return f"SHOW ONLINE FEATURE TABLES IN SCHEMA {database}.{schema}"


def describe_query(name: str, database: str, schema: str) -> str:
    """Return the SHOW ONLINE FEATURE TABLES LIKE SQL string for a named object.

    Args:
        name: Online feature table name to look up.
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        SQL string to show the named online feature table.
    """
    location = f"{database}.{schema}"
    return f"SHOW ONLINE FEATURE TABLES LIKE '{name}' IN SCHEMA {location}"


def describe_columns_query(name: str, database: str, schema: str) -> str:
    """Return the DESCRIBE ONLINE FEATURE TABLE SQL string for a named object.

    Args:
        name: Online feature table name.
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        SQL string to describe the columns of the named online feature table.
    """
    return f'DESCRIBE ONLINE FEATURE TABLE "{database}"."{schema}"."{name}"'


def drop_queries(names: list[str], database: str, schema: str) -> list[str]:
    """Return DROP SQL strings for the named online feature tables.

    Args:
        names: List of online feature table names to drop.
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        List of DROP ONLINE FEATURE TABLE IF EXISTS SQL strings, one per name.
    """
    return [f'DROP ONLINE FEATURE TABLE IF EXISTS "{database}"."{schema}"."{name}"' for name in names]


def describe_specification_query(database: str, schema: str, name: str) -> str:
    """Return the ``DESCRIBE ... TYPE = SPECIFICATION`` SQL string.

    This is the public, stable Snowflake call that returns the original spec
    JSON used to create the Online Feature Table.  It replaces the previous
    structural-fingerprint approach for state inspection by exposing UDF
    code, source schemas, aggregation windows, target lag, and other
    properties that are otherwise invisible after creation.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.
        name: Online feature table name.

    Returns:
        SQL string of the form
        ``DESCRIBE ONLINE FEATURE TABLE "db"."schema"."name" TYPE = SPECIFICATION``.
    """
    return f'DESCRIBE ONLINE FEATURE TABLE "{database}"."{schema}"."{name}" ' "TYPE = SPECIFICATION"


def list_state_queries(database: str, schema: str) -> dict[str, str]:
    """Return the SQL set used by ``snow feature list``.

    The CLI runs each query, parses the rows, and passes them to
    :func:`decl.api.enrich_list_results` to produce a multi-kind table
    output (FeatureView / Entity / Datasource).

    Entity rows are *not* fetched via SQL anymore; the CLI calls
    :func:`decl.api.fetch_entity_rows` (which delegates to the imperative
    ``FeatureStore.list_entities()``) instead.  The previous
    ``show_entities`` key — and the standalone ``list_entities_query``
    factory — were removed in the entity round-trip migration to
    eliminate a duplicated copy of the entity-tag query shape.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        Dict with keys:

        - ``show_ofts``: enumerates Online Feature Tables.
        - ``describe_specification_template``: per-OFT spec retrieval; format
          with ``.format(name=oft_name)``.
    """
    location = f"{database}.{schema}"
    return {
        "show_ofts": f"SHOW ONLINE FEATURE TABLES IN SCHEMA {location}",
        "describe_specification_template": (
            f'DESCRIBE ONLINE FEATURE TABLE "{database}"."{schema}"."{{name}}" ' "TYPE = SPECIFICATION"
        ),
    }
