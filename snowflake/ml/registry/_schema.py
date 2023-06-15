from typing import Dict

# TODO(amauser): Move this scheme and registry creation in general into a server-side implementation.
_REGISTRY_TABLE_SCHEMA: Dict[str, str] = {
    "CREATION_CONTEXT": "VARCHAR",
    "CREATION_ENVIRONMENT_SPEC": "OBJECT",
    "CREATION_ROLE": "VARCHAR",
    "CREATION_TIME": "TIMESTAMP_TZ",
    "ID": "VARCHAR PRIMARY KEY RELY",
    "INPUT_SPEC": "OBJECT",
    "NAME": "VARCHAR",
    "OUTPUT_SPEC": "OBJECT",
    "RUNTIME_ENVIRONMENT_SPEC": "OBJECT",
    "TYPE": "VARCHAR",
    "URI": "VARCHAR",
    "VERSION": "VARCHAR",
}

_METADATA_TABLE_SCHEMA: Dict[str, str] = {
    # TODO(amauser): Generalize attribute to any column reference.
    "ATTRIBUTE_NAME": "VARCHAR",
    "EVENT_ID": "VARCHAR UNIQUE NOT NULL",
    "EVENT_TIMESTAMP": "TIMESTAMP_TZ",
    "MODEL_ID": "VARCHAR FOREIGN KEY REFERENCES {registry_table_name}(ID) RELY",
    "OPERATION": "VARCHAR",
    "ROLE": "VARCHAR",
    "SEQUENCE_ID": "BIGINT AUTOINCREMENT START 0 INCREMENT 1 PRIMARY KEY",
    "VALUE": "OBJECT",
}

_DEPLOYMENTS_TABLE_SCHEMA: Dict[str, str] = {
    "CREATION_TIME": "TIMESTAMP_TZ",
    "MODEL_ID": "VARCHAR FOREIGN KEY REFERENCES {registry_table_name}(ID) RELY",
    "DEPLOYMENT_NAME": "VARCHAR",
    "OPTIONS": "VARIANT",
    "TARGET_PLATFORM": "VARCHAR",
    "ROLE": "VARCHAR",
    "STAGE_PATH": "VARCHAR",
    "SIGNATURE": "VARIANT",
    "TARGET_METHOD": "VARCHAR",
}
