from enum import Enum
from typing import Dict, List

from snowflake.ml.feature_store.feature_store import (
    _FEATURE_STORE_OBJECT_TAG,
    _FEATURE_VIEW_METADATA_TAG,
    FeatureStore,
)
from snowflake.snowpark import Session, exceptions


class FeatureStoreRole(Enum):
    NONE = 0
    CONSUMER = 1
    PRODUCER = 2
    ADMIN = 9


# Lists of permissions as tuples of (OBJECT_TYPE, [PRIVILEGES, ...])
_PRIVILEGE_LEVELS: Dict[FeatureStoreRole, Dict[str, List[str]]] = {
    FeatureStoreRole.ADMIN: {
        "database {database}": ["CREATE SCHEMA"],
        f"tag {{database}}.{{schema}}.{_FEATURE_VIEW_METADATA_TAG}": ["OWNERSHIP"],
        f"tag {{database}}.{{schema}}.{_FEATURE_STORE_OBJECT_TAG}": ["OWNERSHIP"],
        "schema {database}.{schema}": ["OWNERSHIP"],
    },
    FeatureStoreRole.PRODUCER: {
        "schema {database}.{schema}": [
            "CREATE TABLE",  # Necessary for testing only (create mock table)
            "CREATE DYNAMIC TABLE",
            "CREATE TAG",
            "CREATE VIEW",
            # FIXME(dhung): Dataset RBAC won't be ready until 8.17 release
            # "CREATE DATASET",
        ],
        f"tag {{database}}.{{schema}}.{_FEATURE_VIEW_METADATA_TAG}": ["APPLY"],
        f"tag {{database}}.{{schema}}.{_FEATURE_STORE_OBJECT_TAG}": ["APPLY"],
        # TODO: The below privileges should be granted on a per-resource level
        #       between producers (e.g. PRODUCER_A grants PRODUCER_B operate access
        #       to FEATURE_VIEW_0, but not FEATURE_VIEW_1)
        "future dynamic tables in schema {database}.{schema}": ["OPERATE"],
        "all dynamic tables in schema {database}.{schema}": ["OPERATE"],
        "future tasks in schema {database}.{schema}": ["OPERATE"],
        "all tasks in schema {database}.{schema}": ["OPERATE"],
    },
    FeatureStoreRole.CONSUMER: {
        # "warehouse {warehouse}": ["USAGE"],
        "database {database}": ["USAGE"],
        "schema {database}.{schema}": ["USAGE"],
        "future dynamic tables in schema {database}.{schema}": [
            "SELECT",
            "MONITOR",
        ],
        "all dynamic tables in schema {database}.{schema}": [
            "SELECT",
            "MONITOR",
        ],
        "future views in schema {database}.{schema}": [
            "SELECT",
            "REFERENCES",
        ],
        "all views in schema {database}.{schema}": [
            "SELECT",
            "REFERENCES",
        ],
        # FIXME(dhung): Dataset RBAC won't be ready until 8.17 release
        # "future datasets in schema {database}.{schema}": ["USAGE"],
        # "all datasets in schema {database}.{schema}": ["USAGE"],
    },
    FeatureStoreRole.NONE: {},
}


def configure_roles(
    feature_store: FeatureStore,
    admin_role_name: str = "FS_ADMIN",
    producer_role_name: str = "FS_PRODUCER",
    consumer_role_name: str = "FS_CONSUMER",
) -> None:
    session = feature_store._session
    session_info = {
        "account": session.get_current_account(),
        "database": feature_store._config.database,
        "schema": feature_store._config.schema,
        "warehouse": session.get_current_warehouse(),
    }

    def _grant_privileges(session: Session, role_name: str, access_level: FeatureStoreRole) -> None:
        for scope, privilege_list in _PRIVILEGE_LEVELS[access_level].items():
            session.sql(
                f"grant {','.join(privilege_list)} on {scope.format(**session_info)} to role {role_name}"
            ).collect()

    # Try ensuring roles exist. If fail (no CREATE ROLE privilege), just continue
    try:
        session.sql(f"create role if not exists {admin_role_name}").collect()
        session.sql(f"create role if not exists {producer_role_name}").collect()
        session.sql(f"create role if not exists {consumer_role_name}").collect()
    except exceptions.SnowparkSQLException:
        pass

    # Grant privileges to roles
    _grant_privileges(session, admin_role_name, FeatureStoreRole.ADMIN)
    _grant_privileges(session, producer_role_name, FeatureStoreRole.PRODUCER)
    _grant_privileges(session, consumer_role_name, FeatureStoreRole.CONSUMER)

    # Build role hierarchy
    # session.sql(f"grant role {consumer_role_name} to role {producer_role_name}").collect()
    # session.sql(f"grant role {producer_role_name} to role {admin_role_name}").collect()
    # session.sql(f"grant role {admin_role_name} to role {session.get_current_role()}").collect()
