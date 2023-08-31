from typing import List, Tuple

# TODO(amauser): Move this scheme and registry creation in general into a server-side implementation.
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
    ("TRAINING_DATASET_ID", "VARCHAR"),
    ("TYPE", "VARCHAR"),
    ("URI", "VARCHAR"),
    ("VERSION", "VARCHAR"),
]

# TODO(amauser): Generalize attribute to any column reference.
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
    ("ARTIFACT_SPEC", "OBJECT"),
    # Below is out-of-line constraints of Snowflake table.
    # See https://docs.snowflake.com/en/sql-reference/sql/create-table
    ("PRIMARY KEY", "(ID, TYPE) RELY"),
]
