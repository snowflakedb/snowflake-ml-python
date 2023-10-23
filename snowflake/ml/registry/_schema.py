from typing import Dict, List, Tuple, Type

from snowflake.ml.registry import _initial_schema, _schema_upgrade_plans

# BUMP THIS VERSION WHENEVER YOU CHANGE ANY SCHEMA TABLES.
# ALSO UPDATE SCHEMA UPGRADE PLANS.
_CURRENT_SCHEMA_VERSION = 3

_REGISTRY_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("CREATION_CONTEXT", "VARCHAR"),
    ("CREATION_ENVIRONMENT_SPEC", "OBJECT"),
    ("CREATION_ROLE", "VARCHAR"),
    ("CREATION_TIME", "TIMESTAMP_TZ"),
    ("ID", "VARCHAR PRIMARY KEY RELY"),
    ("INPUT_SPEC", "OBJECT"),
    ("NAME", "VARCHAR"),
    ("OUTPUT_SPEC", "OBJECT"),
    ("RUNTIME_ENVIRONMENT_SPEC", "OBJECT"),
    ("ARTIFACT_IDS", "ARRAY"),
    ("TYPE", "VARCHAR"),
    ("URI", "VARCHAR"),
    ("VERSION", "VARCHAR"),
]

_METADATA_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("ATTRIBUTE_NAME", "VARCHAR"),
    ("EVENT_ID", "VARCHAR UNIQUE NOT NULL"),
    ("EVENT_TIMESTAMP", "TIMESTAMP_TZ"),
    ("MODEL_ID", "VARCHAR FOREIGN KEY REFERENCES {registry_table_name}(ID) RELY"),
    ("OPERATION", "VARCHAR"),
    ("ROLE", "VARCHAR"),
    ("SEQUENCE_ID", "BIGINT AUTOINCREMENT START 0 INCREMENT 1 PRIMARY KEY"),
    ("VALUE", "OBJECT"),
]

_DEPLOYMENTS_TABLE_SCHEMA: List[Tuple[str, str]] = [
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

_ARTIFACT_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("ID", "VARCHAR"),
    ("TYPE", "VARCHAR"),
    ("NAME", "VARCHAR"),
    ("VERSION", "VARCHAR"),
    ("CREATION_ROLE", "VARCHAR"),
    ("CREATION_TIME", "TIMESTAMP_TZ"),
    ("ARTIFACT_SPEC", "VARCHAR"),
    # Below is out-of-line constraints of Snowflake table.
    # See https://docs.snowflake.com/en/sql-reference/sql/create-table
    ("PRIMARY KEY", "(ID, TYPE) RELY"),
]

# Note, one can add/remove tables from this tuple as well. As long as correct schema update process is followed.
# In case of a new table, they should not be defined in _initial_schema.
_CURRENT_TABLE_SCHEMAS = {
    _initial_schema._MODELS_TABLE_NAME: _REGISTRY_TABLE_SCHEMA,
    _initial_schema._METADATA_TABLE_NAME: _METADATA_TABLE_SCHEMA,
    _initial_schema._DEPLOYMENT_TABLE_NAME: _DEPLOYMENTS_TABLE_SCHEMA,
    _initial_schema._ARTIFACT_TABLE_NAME: _ARTIFACT_TABLE_SCHEMA,
}


_SCHEMA_UPGRADE_PLANS: Dict[int, Type[_schema_upgrade_plans.BaseSchemaUpgradePlans]] = {
    # Currently _CURRENT_SCHEMA_VERSION == _initial_schema._INITIAL_VERSION, so no entry.
    # But if schema evolves it must contain:
    #   Key = a version number
    #   Value = a subclass of _schema_upgrades.BaseSchemaUpgrade
    # NOTE, all version from _INITIAL_VERSION + 1 till _CURRENT_SCHEMA_VERSION must exists.
    1: _schema_upgrade_plans.AddTrainingDatasetIdIfNotExists,
    2: _schema_upgrade_plans.ReplaceTrainingDatasetIdWithArtifactIds,
    3: _schema_upgrade_plans.ChangeArtifactSpecFromObjectToVarchar,
}

assert len(_SCHEMA_UPGRADE_PLANS) == _CURRENT_SCHEMA_VERSION - _initial_schema._INITIAL_VERSION
