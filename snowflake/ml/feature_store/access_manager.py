from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils.query_result_checker import SqlResultValidator
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_store import (
    _FEATURE_STORE_OBJECT_TAG,
    _FEATURE_VIEW_METADATA_TAG,
    CreationMode,
    FeatureStore,
)
from snowflake.snowpark import Session, exceptions

_PROJECT = "FeatureStore"
_ALL_OBJECTS = "@ALL_OBJECTS"  # Special flag to mark "all+future" grants


class _FeatureStoreRole(Enum):
    NONE = 0  # For testing purposes
    CONSUMER = 1
    PRODUCER = 2


@dataclass(frozen=True)
class _Privilege:
    object_type: str
    object_name: str
    privileges: List[str]
    scope: Optional[str] = None


@dataclass(frozen=True)
class _SessionInfo:
    database: SqlIdentifier
    schema: SqlIdentifier
    warehouse: SqlIdentifier


# Lists of permissions as tuples of (OBJECT_TYPE, [PRIVILEGES, ...])
_PRE_INIT_PRIVILEGES: Dict[_FeatureStoreRole, List[_Privilege]] = {
    _FeatureStoreRole.PRODUCER: [
        _Privilege(
            "SCHEMA",
            "{database}.{schema}",
            [
                "CREATE TABLE",  # Necessary for testing only (create mock table)
                "CREATE DYNAMIC TABLE",
                "CREATE TAG",
                "CREATE VIEW",
                "CREATE TASK",
                # FIXME(dhung): Dataset RBAC won't be ready until 8.17 release
                # "CREATE DATASET",
            ],
        ),
        _Privilege("DYNAMIC TABLE", _ALL_OBJECTS, ["OPERATE"], "SCHEMA {database}.{schema}"),
        _Privilege("TASK", _ALL_OBJECTS, ["OPERATE"], "SCHEMA {database}.{schema}"),
    ],
    _FeatureStoreRole.CONSUMER: [
        _Privilege("DATABASE", "{database}", ["USAGE"]),
        _Privilege("SCHEMA", "{database}.{schema}", ["USAGE"]),
        _Privilege("DYNAMIC TABLE", _ALL_OBJECTS, ["SELECT", "MONITOR"], "SCHEMA {database}.{schema}"),
        _Privilege("VIEW", _ALL_OBJECTS, ["SELECT", "REFERENCES"], "SCHEMA {database}.{schema}"),
        # FIXME(dhung): Dataset RBAC won't be ready until 8.17 release
        # _Privilege("DATASETS", _ALL_OBJECTS, ["USAGE"], "SCHEMA {database}.{schema}"),
        # User should decide whether they want to grant warehouse usage to CONSUMER
        # _Privilege("WAREHOUSE", "{warehouse}", ["USAGE"]),
    ],
    _FeatureStoreRole.NONE: [],
}

_POST_INIT_PRIVILEGES: Dict[_FeatureStoreRole, List[_Privilege]] = {
    _FeatureStoreRole.PRODUCER: [
        _Privilege("TAG", f"{{database}}.{{schema}}.{_FEATURE_VIEW_METADATA_TAG}", ["APPLY"]),
        _Privilege("TAG", f"{{database}}.{{schema}}.{_FEATURE_STORE_OBJECT_TAG}", ["APPLY"]),
    ],
    _FeatureStoreRole.CONSUMER: [],
    _FeatureStoreRole.NONE: [],
}


def _grant_privileges(
    session: Session, role_name: str, privileges: List[_Privilege], session_info: _SessionInfo
) -> None:
    session_info_dict = asdict(session_info)
    for p in privileges:
        if p.object_name == _ALL_OBJECTS:
            # Ensure obj is plural
            obj = p.object_type.upper()
            if not obj.endswith("S"):
                obj += "S"
            grant_objects = [f"{prefix} {obj}" for prefix in ("FUTURE", "ALL")]
        else:
            grant_objects = [f"{p.object_type} {p.object_name.format(**session_info_dict)}"]
        for grant_object in grant_objects:
            query = f"GRANT {','.join(p.privileges)} ON {grant_object}"
            if p.scope:
                query += f" IN {p.scope.format(**session_info_dict)}"
            query += f" TO ROLE {role_name}"
            session.sql(query).collect()


def _configure_pre_init_privileges(
    session: Session,
    session_info: _SessionInfo,
    producer_role: str = "SNOWML_FEATURE_STORE_PRODUCER_RL",
    consumer_role: str = "SNOWML_FEATURE_STORE_CONSUMER_RL",
) -> None:
    """
    Configure Feature Store role privileges. Must be run with ACCOUNTADMIN
    or a role with `MANAGE GRANTS` privilege.

    See https://docs.snowflake.com/en/sql-reference/sql/grant-privilege for more information
    about privilege grants in Snowflake.

    Args:
        session: Snowpark Session to interact with Snowflake backend.
        session_info: Session info like database and schema for the FeatureStore instance.
        producer_role: Name of producer role to be configured.
        consumer_role: Name of consumer role to be configured.
    """

    # Create schema if not already exists
    (create_rst,) = (
        SqlResultValidator(
            session,
            f"CREATE SCHEMA IF NOT EXISTS {session_info.database}.{session_info.schema}",
        )
        .has_dimensions(expected_rows=1)
        .has_column("status")
        .validate()
    )
    schema_created = create_rst["status"].endswith("successfully created.")

    # Grant privileges to roles
    _grant_privileges(session, producer_role, _PRE_INIT_PRIVILEGES[_FeatureStoreRole.PRODUCER], session_info)
    _grant_privileges(session, consumer_role, _PRE_INIT_PRIVILEGES[_FeatureStoreRole.CONSUMER], session_info)

    # Pass schema ownership from admin to PRODUCER
    if schema_created:
        session.sql(
            f"GRANT OWNERSHIP ON SCHEMA {session_info.database}.{session_info.schema} TO ROLE {producer_role}"
        ).collect()


def _configure_post_init_privileges(
    session: Session,
    session_info: _SessionInfo,
    producer_role: str = "FS_PRODUCER",
    consumer_role: str = "FS_CONSUMER",
) -> None:
    _grant_privileges(session, producer_role, _POST_INIT_PRIVILEGES[_FeatureStoreRole.PRODUCER], session_info)
    _grant_privileges(session, consumer_role, _POST_INIT_PRIVILEGES[_FeatureStoreRole.CONSUMER], session_info)


def _configure_role_hierarchy(
    session: Session,
    producer_role: str,
    consumer_role: str,
) -> None:
    """
    Create Feature Store roles and configure role hierarchy hierarchy. Must be run with
    ACCOUNTADMIN or a role with `CREATE ROLE` privilege.

    See https://docs.snowflake.com/en/sql-reference/sql/grant-privilege for more information
    about privilege grants in Snowflake.

    Args:
        session: Snowpark Session to interact with Snowflake backend.
        producer_role: Name of producer role to be configured.
        consumer_role: Name of consumer role to be configured.
    """
    producer_role = SqlIdentifier(producer_role)
    consumer_role = SqlIdentifier(consumer_role)

    # Create the necessary roles
    session.sql(f"CREATE ROLE IF NOT EXISTS {producer_role}").collect()
    session.sql(f"CREATE ROLE IF NOT EXISTS {consumer_role}").collect()

    # Build role hierarchy
    session.sql(f"GRANT ROLE {consumer_role} TO ROLE {producer_role}").collect()
    session.sql(f"GRANT ROLE {producer_role} TO ROLE SYSADMIN").collect()
    session.sql(f"GRANT ROLE {producer_role} TO ROLE {session.get_current_role()}").collect()


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def setup_feature_store(
    session: Session,
    database: str,
    schema: str,
    warehouse: str,
    producer_role: str = "FS_PRODUCER",
    consumer_role: str = "FS_CONSUMER",
) -> FeatureStore:
    """
    Sets up a new Feature Store including role/privilege setup. Must be run with ACCOUNTADMIN
    or a role with `MANAGE GRANTS` and `CREATE ROLE` privileges.

    See https://docs.snowflake.com/en/sql-reference/sql/grant-privilege for more information
    about privilege grants in Snowflake.

    Args:
        session: Snowpark Session to interact with Snowflake backend.
        database: Database to create the FeatureStore instance.
        schema: Schema to create the FeatureStore instance.
        warehouse: Default warehouse for Feature Store compute.
        producer_role: Name of producer role to be configured.
        consumer_role: Name of consumer role to be configured.

    Returns:
        Feature Store instance.

    Raises:
        exceptions.SnowparkSQLException: Insufficient privileges.
    """

    database = SqlIdentifier(database)
    schema = SqlIdentifier(schema)
    warehouse = SqlIdentifier(warehouse)
    session_info = _SessionInfo(
        SqlIdentifier(database),
        SqlIdentifier(schema),
        SqlIdentifier(warehouse),
    )

    try:
        _configure_role_hierarchy(session, producer_role=producer_role, consumer_role=consumer_role)
    except exceptions.SnowparkSQLException:
        # Error can be safely ignored if roles already exist and hierarchy is already built
        for role in (producer_role, consumer_role):
            # Ensure roles already exist
            if session.sql(f"SHOW ROLES LIKE '{role}' STARTS WITH '{role}'").count() == 0:
                raise
        # Ensure hierarchy already configured
        consumer_grants = session.sql(f"SHOW GRANTS ON ROLE {consumer_role}").collect()
        if not any(r["granted_to"] == "ROLE" and r["grantee_name"] == producer_role for r in consumer_grants):
            raise

    # Do any pre-FeatureStore.__init__() privilege setup
    _configure_pre_init_privileges(session, session_info, producer_role, consumer_role)

    # Use PRODUCER role to create and operate new Feature Store
    current_role = session.get_current_role()
    assert current_role is not None  # to make mypy happy
    try:
        session.use_role(producer_role)
        fs = FeatureStore(session, database, schema, warehouse, creation_mode=CreationMode.CREATE_IF_NOT_EXIST)
    finally:
        session.use_role(current_role)

    # Do any post-FeatureStore.__init__() privilege setup
    _configure_post_init_privileges(session, session_info, producer_role, consumer_role)

    return fs
