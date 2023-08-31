import enum
from typing import Any, Dict, Optional, cast

from snowflake import connector, snowpark
from snowflake.ml._internal.utils import formatting, identifier, table_manager
from snowflake.ml.registry import _schema


# Set of allowed artifact types.
class ArtifactType(enum.Enum):
    TESTTYPE = "TESTTYPE"  # A placeholder type just for unit test
    TRAINING_DATASET = "TRAINING_DATASET"


# Default name of the artifact table
_ARTIFACT_TABLE_NAME: str = identifier.get_inferred_name("_SYSTEM_REGISTRY_ARTIFACTS")


def create_ml_artifact_table(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    statement_params: Dict[str, Any],
) -> None:
    """Create the ml artifact table to store immutable properties of various artifacts.

    This artifact table will follow a predefined schema detailed in `_ARTIFACT_TABLE_SCHEMA` from `_schema.py`.

    Note:
        The artifact table uses (ID + TYPE) as its compound primary key, hence, it needs an out-of-line private key.

    Args:
        session: Snowpark session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        statement_params: Function usage statement parameters used in sql query executions.
    """
    table_manager.create_single_registry_table(
        session=session,
        database_name=database_name,
        schema_name=schema_name,
        table_name=_ARTIFACT_TABLE_NAME,
        table_schema=_schema._ARTIFACT_TABLE_SCHEMA,
        statement_params=statement_params,
    )


def if_artifact_table_exists(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
) -> bool:
    """
    Verify the existence of the artifact table.

    Args:
        session: Snowpark session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.

    Returns:
        bool: True if the artifact table exists, False otherwise.
    """
    qualified_schema_name = table_manager.get_fully_qualified_schema_name(database_name, schema_name)
    return table_manager.validate_table_exist(session, _ARTIFACT_TABLE_NAME, qualified_schema_name)


def if_artifact_exists(
    session: snowpark.Session, database_name: str, schema_name: str, artifact_id: str, artifact_type: ArtifactType
) -> bool:
    """Validate if a specific artifact record exists in the artifact table.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        artifact_id: Unique identifier of the target artifact.
        artifact_type: Type of the target artifact

    Returns:
        bool: True if the artifact exists, False otherwise.
    """
    selected_artifact = _get_artifact(session, database_name, schema_name, artifact_id, artifact_type).collect()

    assert (
        len(selected_artifact) < 2
    ), f"Multiple records found for the specified artifact (ID: {artifact_id}, TYPE: {artifact_type.name})!"

    return len(selected_artifact) == 1


def add_artifact(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    artifact_id: str,
    artifact_type: ArtifactType,
    artifact_name: str,
    artifact_version: Optional[str],
    artifact_spec: Dict[str, Any],
) -> None:
    """
    Insert a new artifact record into the designated artifact table.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        artifact_id: Unique identifier for the artifact.
        artifact_type: Type of the artifact.
        artifact_name: Name of the artifact.
        artifact_version: Version of the artifact if applicable.
        artifact_spec: Specifications related to the artifact.

    Raises:
        TypeError: If the given artifact type isn't valid.
        DataError: If the given artifact already exists in the database.
    """
    if not isinstance(artifact_type, ArtifactType):
        raise TypeError(f"{artifact_type} isn't a recognized artifact type.")

    if if_artifact_exists(session, database_name, schema_name, artifact_id, artifact_type):
        raise connector.DataError(
            f"artifact with ID {artifact_id} and TYPE {artifact_type.name} already exists. Unable to add the artifact."
        )

    fully_qualified_table_name = table_manager.get_fully_qualified_table_name(
        database_name, schema_name, _ARTIFACT_TABLE_NAME
    )

    new_artifact = {
        "ID": artifact_id,
        "TYPE": artifact_type.name,
        "NAME": artifact_name,
        "VERSION": artifact_version,
        "CREATION_ROLE": session.get_current_role(),
        "CREATION_TIME": formatting.SqlStr("CURRENT_TIMESTAMP()"),
        "ARTIFACT_SPEC": artifact_spec,
    }

    # TODO: Consider updating the METADATA table for artifact history tracking as well.
    table_manager.insert_table_entry(session, fully_qualified_table_name, new_artifact)


def delete_artifact(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    artifact_id: str,
    artifact_type: ArtifactType,
    error_if_not_exist: bool = False,
) -> None:
    """
    Remove an artifact record from the designated artifact table.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        artifact_id: Unique identifier for the artifact to be deleted.
        artifact_type: Type of the artifact to be deleted.
        error_if_not_exist: Whether to raise errors if the target entry doesn't exist. Default to be false.

    Raises:
        DataError: If error_if_not_exist is true and the artifact doesn't exist in the database.
        RuntimeError: If the artifact deletion failed.
    """
    if error_if_not_exist and not if_artifact_exists(session, database_name, schema_name, artifact_id, artifact_type):
        raise connector.DataError(
            f"Artifact with ID '{artifact_id}' and TYPE '{artifact_type.name}' doesn't exist. Deletion not possible."
        )

    fully_qualified_table_name = table_manager.get_fully_qualified_table_name(
        database_name, schema_name, _ARTIFACT_TABLE_NAME
    )

    delete_query = f"DELETE FROM {fully_qualified_table_name} WHERE ID='{artifact_id}' AND TYPE='{artifact_type.name}'"

    # TODO: Consider updating the METADATA table for artifact history tracking as well.
    try:
        session.sql(delete_query).collect()
    except Exception as e:
        raise RuntimeError(f"Delete ML artifact (ID: {artifact_id}, TYPE: {artifact_type.name}) failed due to {e}")


def _get_artifact(
    session: snowpark.Session, database_name: str, schema_name: str, artifact_id: str, artifact_type: ArtifactType
) -> snowpark.DataFrame:
    """Retrieve the Snowpark dataframe of the artifact matching the provided artifact id and type.

    Given that ID and TYPE act as a compound primary key for the artifact table, the resulting dataframe should have,
    at most, one row.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        artifact_id: Unique identifier of the target artifact.
        artifact_type: Type of the target artifact

    Returns:
        A Snowpark dataframe representing the artifacts that match the given constraints.

    WARNING:
        The returned DataFrame is writable and shouldn't be made accessible to users.
    """
    artifacts = session.sql(
        "SELECT * FROM "
        f"{table_manager.get_fully_qualified_table_name(database_name, schema_name, _ARTIFACT_TABLE_NAME)}"
    )
    target_artifact = artifacts.filter(snowpark.Column("ID") == artifact_id).filter(
        snowpark.Column("TYPE") == artifact_type.name
    )
    return cast(snowpark.DataFrame, target_artifact)
