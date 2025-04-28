import json
from typing import Any, Optional, TypedDict

from typing_extensions import NotRequired

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import (
    model as model_sql,
    model_version as model_version_sql,
)
from snowflake.snowpark import session

MODEL_VERSION_METADATA_SCHEMA_VERSION = "2024-01-01"


class ModelVersionMetadataSchema(TypedDict):
    metrics: NotRequired[dict[str, Any]]


class MetadataOperator:
    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._model_client = model_sql.ModelSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._model_version_client = model_version_sql.ModelVersionSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, MetadataOperator):
            return False
        return (
            self._model_client == __value._model_client and self._model_version_client == __value._model_version_client
        )

    @staticmethod
    def _parse(metadata_dict: dict[str, Any]) -> ModelVersionMetadataSchema:
        loaded_metadata_schema_version = metadata_dict.get("snowpark_ml_schema_version", None)
        if loaded_metadata_schema_version is None:
            return ModelVersionMetadataSchema(metrics={})
        elif (
            not isinstance(loaded_metadata_schema_version, str)
            or loaded_metadata_schema_version != MODEL_VERSION_METADATA_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported model metadata schema version {loaded_metadata_schema_version} confronted.")
        loaded_metrics = metadata_dict.get("metrics", {})
        if not isinstance(loaded_metrics, dict):
            raise ValueError(f"Metrics in the metadata is expected to be a dictionary, getting {loaded_metrics}")
        return ModelVersionMetadataSchema(metrics=loaded_metrics)

    def _get_current_metadata_dict(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        version_info_list = self._model_client.show_versions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        metadata_str = version_info_list[0][self._model_client.MODEL_VERSION_METADATA_COL_NAME]
        if not metadata_str:
            return {}
        res = json.loads(metadata_str)
        if not isinstance(res, dict):
            raise ValueError(f"Metadata is expected to be a dictionary, getting {res}")
        return res

    def load(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> ModelVersionMetadataSchema:
        metadata_dict = self._get_current_metadata_dict(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        return MetadataOperator._parse(metadata_dict)

    def save(
        self,
        metadata: ModelVersionMetadataSchema,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        metadata_dict = self._get_current_metadata_dict(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        metadata_dict.update({**metadata, "snowpark_ml_schema_version": MODEL_VERSION_METADATA_SCHEMA_VERSION})
        self._model_version_client.set_metadata(
            metadata_dict,
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
