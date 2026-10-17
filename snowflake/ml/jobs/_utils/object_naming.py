from pathlib import PurePosixPath

from snowflake import snowpark
from snowflake.ml._internal.utils import identifier


def resolve_database_schema(session: snowpark.Session, database: str | None, schema: str | None) -> tuple[str, str]:
    """Resolve the database and schema an object will be created in, falling back to session context.

    Args:
        session: The Snowpark session whose context supplies any unspecified part.
        database: The requested database, or None to use the session's current database.
        schema: The requested schema, or None to use the session's current schema.

    Returns:
        The resolved (database, schema) pair, each normalized to a valid identifier.

    Raises:
        ValueError: If either part is unspecified and unavailable from the session context.
    """
    database = database or session.get_current_database()
    schema = schema or session.get_current_schema()

    if database is None:
        raise ValueError("Database must be specified either in the session context or as a parameter.")
    if schema is None:
        raise ValueError("Schema must be specified either in the session context or as a parameter.")

    return identifier.resolve_identifier(database), identifier.resolve_identifier(schema)


def build_object_stage_path(stage_name: str, object_name: str) -> PurePosixPath:
    """Build the stage path an object's payload is uploaded to.

    Args:
        stage_name: The target stage, optionally qualified and optionally with a leading ``@`` and
            a subdirectory, e.g. ``@DB.SCHEMA.MY_STAGE/subdir``.
        object_name: The name of the object, used as the leaf directory under the stage.

    Returns:
        The stage path, e.g. ``@DB.SCHEMA.MY_STAGE/subdir/my_object``.
    """
    stage_path_parts = identifier.parse_snowflake_stage_path(stage_name.lstrip("@"))
    stage_name_prefix = f"@{'.'.join(filter(None, stage_path_parts[:3]))}"
    return PurePosixPath(f"{stage_name_prefix}{stage_path_parts[-1].rstrip('/')}/{object_name}")
