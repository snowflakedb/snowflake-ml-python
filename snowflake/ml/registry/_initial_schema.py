from typing import Any, Dict, List, Tuple

from snowflake import snowpark
from snowflake.ml._internal.utils import identifier, query_result_checker, table_manager

# THIS FILE CONTAINS INITIAL REGISTRY SCHEMA.
# !!!!!!! WARNING !!!!!!!
# Please do not modify initial schema and use schema evolution mechanism in SchemaVersionManager to change the schema.
# If you are touching this file, make sure you understand what you are doing.

_INITIAL_VERSION: int = 0

_MODELS_TABLE_NAME: str = "_SYSTEM_REGISTRY_MODELS"
_METADATA_TABLE_NAME: str = "_SYSTEM_REGISTRY_METADATA"
_DEPLOYMENT_TABLE_NAME: str = "_SYSTEM_REGISTRY_DEPLOYMENTS"
_ARTIFACT_TABLE_NAME: str = "_SYSTEM_REGISTRY_ARTIFACTS"

_INITIAL_REGISTRY_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("CREATION_CONTEXT", "VARCHAR"),
    ("CREATION_ENVIRONMENT_SPEC", "OBJECT"),
    ("CREATION_ROLE", "VARCHAR"),
    ("CREATION_TIME", "TIMESTAMP_TZ"),
    ("ID", "VARCHAR PRIMARY KEY RELY"),
    ("INPUT_SPEC", "OBJECT"),
    ("NAME", "VARCHAR"),
    ("OUTPUT_SPEC", "OBJECT"),
    ("RUNTIME_ENVIRONMENT_SPEC", "OBJECT"),
    ("TRAINING_DATASET_ID", "VARCHAR"),
    ("TYPE", "VARCHAR"),
    ("URI", "VARCHAR"),
    ("VERSION", "VARCHAR"),
]

_INITIAL_METADATA_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("ATTRIBUTE_NAME", "VARCHAR"),
    ("EVENT_ID", "VARCHAR UNIQUE NOT NULL"),
    ("EVENT_TIMESTAMP", "TIMESTAMP_TZ"),
    ("MODEL_ID", "VARCHAR FOREIGN KEY REFERENCES {registry_table_name}(ID) RELY"),
    ("OPERATION", "VARCHAR"),
    ("ROLE", "VARCHAR"),
    ("SEQUENCE_ID", "BIGINT AUTOINCREMENT START 0 INCREMENT 1 PRIMARY KEY"),
    ("VALUE", "OBJECT"),
]

_INITIAL_DEPLOYMENTS_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("CREATION_TIME", "TIMESTAMP_TZ"),
    ("MODEL_ID", "VARCHAR FOREIGN KEY REFERENCES {registry_table_name}(ID) RELY"),
    ("DEPLOYMENT_NAME", "VARCHAR"),
    ("OPTIONS", "VARIANT"),
    ("TARGET_PLATFORM", "VARCHAR"),
    ("ROLE", "VARCHAR"),
    ("STAGE_PATH", "VARCHAR"),
    ("SIGNATURE", "VARIANT"),
    ("TARGET_METHOD", "VARCHAR"),
]

_INITIAL_ARTIFACT_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("ID", "VARCHAR"),
    ("TYPE", "VARCHAR"),
    ("NAME", "VARCHAR"),
    ("VERSION", "VARCHAR"),
    ("CREATION_ROLE", "VARCHAR"),
    ("CREATION_TIME", "TIMESTAMP_TZ"),
    ("ARTIFACT_SPEC", "OBJECT"),
    # Below is out-of-line constraints of Snowflake table.
    # See https://docs.snowflake.com/en/sql-reference/sql/create-table
    ("PRIMARY KEY", "(ID, TYPE) RELY"),
]

_INITIAL_TABLE_SCHEMAS = {
    _MODELS_TABLE_NAME: _INITIAL_REGISTRY_TABLE_SCHEMA,
    _METADATA_TABLE_NAME: _INITIAL_METADATA_TABLE_SCHEMA,
    _DEPLOYMENT_TABLE_NAME: _INITIAL_DEPLOYMENTS_TABLE_SCHEMA,
    _ARTIFACT_TABLE_NAME: _INITIAL_ARTIFACT_TABLE_SCHEMA,
}


def create_initial_registry_tables(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    statement_params: Dict[str, Any],
) -> None:
    """Creates initial set of tables for registry. This is the legacy schema from which schema evolution is supported.

    Args:
        session: Active session to create tables.
        database_name: Name of database in which tables will be created.
        schema_name: Name of schema in which tables will be created.
        statement_params: Statement parameters for telemetry tracking.
    """
    model_table_full_path = table_manager.get_fully_qualified_table_name(database_name, schema_name, _MODELS_TABLE_NAME)

    for table_name, schema_template in _INITIAL_TABLE_SCHEMAS.items():
        table_schema = [(k, v.format(registry_table_name=model_table_full_path)) for k, v in schema_template]
        table_manager.create_single_table(
            session=session,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_schema=table_schema,
            statement_params=statement_params,
        )


def check_access(session: snowpark.Session, database_name: str, schema_name: str) -> None:
    """Check that the required tables exist and are accessible by the current role.

    Args:
        session: Active session to execution SQL queries.
        database_name: Name of database where schema tables live.
        schema_name: Name of schema where schema tables live.
    """
    query_result_checker.SqlResultValidator(
        session,
        query=f"SHOW DATABASES LIKE '{identifier.get_unescaped_names(database_name)}'",
    ).has_dimensions(expected_rows=1).validate()

    query_result_checker.SqlResultValidator(
        session,
        query=f"SHOW SCHEMAS LIKE '{identifier.get_unescaped_names(schema_name)}' IN DATABASE {database_name}",
    ).has_dimensions(expected_rows=1).validate()

    full_qualified_schema_name = table_manager.get_fully_qualified_schema_name(database_name, schema_name)

    table_manager.validate_table_exist(
        session,
        identifier.get_unescaped_names(_MODELS_TABLE_NAME),
        full_qualified_schema_name,
    )
    table_manager.validate_table_exist(
        session,
        identifier.get_unescaped_names(_METADATA_TABLE_NAME),
        full_qualified_schema_name,
    )
    table_manager.validate_table_exist(
        session,
        identifier.get_unescaped_names(_DEPLOYMENT_TABLE_NAME),
        full_qualified_schema_name,
    )

    # TODO(zzhu): Also check validity of views.
