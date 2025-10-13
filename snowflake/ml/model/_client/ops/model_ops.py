import enum
import json
import os
import pathlib
import tempfile
import warnings
from typing import Any, Literal, Optional, TypedDict, Union, cast, overload

import yaml

from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import formatting, identifier, sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.ops import metadata_ops
from snowflake.ml.model._client.sql import (
    model as model_sql,
    model_version as model_version_sql,
    service as service_sql,
    stage as stage_sql,
    tag as tag_sql,
)
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import (
    model_manifest,
    model_manifest_schema,
)
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema
from snowflake.ml.model._packager.model_runtime import model_runtime
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import dataframe, row, session
from snowflake.snowpark._internal import utils as snowpark_utils


# An enum class to represent Create Or Alter Model SQL command.
class ModelAction(enum.Enum):
    CREATE = "CREATE"
    ALTER = "ALTER"


class ServiceInfo(TypedDict):
    name: str
    status: str
    inference_endpoint: Optional[str]


class ModelOperator:
    INFERENCE_SERVICE_ENDPOINT_NAME = "inference"
    INGRESS_ENDPOINT_URL_SUFFIX = "snowflakecomputing.app"
    # app-service-privatelink might not contain "snowflakecomputing" in the url - using the minimum required substring
    PRIVATELINK_INGRESS_ENDPOINT_URL_SUBSTRING = "privatelink.snowflake"

    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        # Ideally, we should only keep session object inside the client, however, some components other than client
        # are requiring session object like ModelComposer and SnowparkDataFrameHandler. We currently cannot refractor
        # them all but we should try to avoid use the _session object here unless no other choice.
        self._session = session
        self._stage_client = stage_sql.StageSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
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
        self._tag_client = tag_sql.ModuleTagSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._service_client = service_sql.ServiceSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._metadata_ops = metadata_ops.MetadataOperator(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelOperator):
            return False
        return (
            self._stage_client == __value._stage_client
            and self._model_client == __value._model_client
            and self._model_version_client == __value._model_version_client
        )

    def prepare_model_temp_stage_path(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        statement_params: Optional[dict[str, Any]] = None,
    ) -> str:
        stage_name = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        )
        self._stage_client.create_tmp_stage(
            database_name=database_name,
            schema_name=schema_name,
            stage_name=stage_name,
            statement_params=statement_params,
        )
        return f"@{self._stage_client.fully_qualified_object_name(database_name, schema_name, stage_name)}/model"

    def get_model_version_stage_path(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
    ) -> str:
        return (
            f"snow://model/{self._stage_client.fully_qualified_object_name(database_name, schema_name, model_name)}"
            f"/versions/{version_name}/"
        )

    def get_model_action_from_model_name_and_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> ModelAction:
        if self.validate_existence(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            statement_params=statement_params,
        ):
            if self.validate_existence(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            ):
                raise ValueError(
                    "Model "
                    f"{self._model_version_client.fully_qualified_object_name(database_name, schema_name, model_name)}"
                    f" version {version_name} already existed."
                )
            else:
                return ModelAction.ALTER
        else:
            return ModelAction.CREATE

    def add_or_create_live_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        model_action = self.get_model_action_from_model_name_and_version(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        if model_action == ModelAction.CREATE:
            self._model_version_client.create_live_version(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        elif model_action == ModelAction.ALTER:
            self._model_version_client.add_live_version(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            raise AssertionError(f"The model_action is {model_action}. Expected CREATE or ALTER.")

    def create_from_stage(
        self,
        composed_model: model_composer.ModelComposer,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
        use_live_commit: Optional[bool] = False,
    ) -> None:

        if use_live_commit:
            # if the model version is live, we can only commit the version
            self._model_version_client.commit_version(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            stage_path = str(composed_model.stage_path)
            # if the model version is not live,
            # find whether the model exists and whether the version exists
            # and then decide whether to create or alter the model
            model_action = self.get_model_action_from_model_name_and_version(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
            if model_action == ModelAction.ALTER:
                self._model_version_client.add_version_from_stage(
                    database_name=database_name,
                    schema_name=schema_name,
                    stage_path=stage_path,
                    model_name=model_name,
                    version_name=version_name,
                    statement_params=statement_params,
                )
            elif model_action == ModelAction.CREATE:
                self._model_version_client.create_from_stage(
                    database_name=database_name,
                    schema_name=schema_name,
                    stage_path=stage_path,
                    model_name=model_name,
                    version_name=version_name,
                    statement_params=statement_params,
                )
            else:
                raise AssertionError(f"The model_action is {model_action}. Expected CREATE or ALTER.")

    def create_from_model_version(
        self,
        *,
        source_database_name: Optional[sql_identifier.SqlIdentifier],
        source_schema_name: Optional[sql_identifier.SqlIdentifier],
        source_model_name: sql_identifier.SqlIdentifier,
        source_version_name: sql_identifier.SqlIdentifier,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        model_exists: bool,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        if model_exists:
            return self._model_version_client.add_version_from_model_version(
                source_database_name=source_database_name,
                source_schema_name=source_schema_name,
                source_model_name=source_model_name,
                source_version_name=source_version_name,
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            return self._model_version_client.create_from_model_version(
                source_database_name=source_database_name,
                source_schema_name=source_schema_name,
                source_model_name=source_model_name,
                source_version_name=source_version_name,
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )

    def show_models_or_versions(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        if model_name:
            return self._model_client.show_versions(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                validate_result=False,
                statement_params=statement_params,
            )
        else:
            return self._model_client.show_models(
                database_name=database_name,
                schema_name=schema_name,
                validate_result=False,
                statement_params=statement_params,
            )

    def list_models_or_versions(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[sql_identifier.SqlIdentifier]:
        res = self.show_models_or_versions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            statement_params=statement_params,
        )
        if model_name:
            col_name = self._model_client.MODEL_VERSION_NAME_COL_NAME
        else:
            col_name = self._model_client.MODEL_NAME_COL_NAME
        return [sql_identifier.SqlIdentifier(row[col_name], case_sensitive=True) for row in res]

    def validate_existence(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> bool:
        if version_name:
            res = self._model_client.show_versions(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                validate_result=False,
                statement_params=statement_params,
            )
        else:
            res = self._model_client.show_models(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                validate_result=False,
                statement_params=statement_params,
            )
        return len(res) == 1

    def get_comment(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> str:
        if version_name:
            res = self._model_client.show_versions(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
            col_name = self._model_client.MODEL_VERSION_COMMENT_COL_NAME
        else:
            res = self._model_client.show_models(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                statement_params=statement_params,
            )
            col_name = self._model_client.MODEL_COMMENT_COL_NAME
        return cast(str, res[0][col_name])

    def set_comment(
        self,
        *,
        comment: str,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        if version_name:
            self._model_version_client.set_comment(
                comment=comment,
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            self._model_client.set_comment(
                comment=comment,
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                statement_params=statement_params,
            )

    def set_alias(
        self,
        *,
        alias_name: sql_identifier.SqlIdentifier,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self._model_version_client.set_alias(
            alias_name=alias_name,
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

    def unset_alias(
        self,
        *,
        version_or_alias_name: sql_identifier.SqlIdentifier,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self._model_version_client.unset_alias(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_or_alias_name=version_or_alias_name,
            statement_params=statement_params,
        )

    def set_default_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.validate_existence(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        ):
            raise ValueError(f"You cannot set version {version_name} as default version as it does not exist.")
        self._model_version_client.set_default_version(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

    def get_default_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> sql_identifier.SqlIdentifier:
        res = self._model_client.show_models(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            statement_params=statement_params,
        )[0]
        return sql_identifier.SqlIdentifier(
            res[self._model_client.MODEL_DEFAULT_VERSION_NAME_COL_NAME], case_sensitive=True
        )

    def get_version_by_alias(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        alias_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> Optional[sql_identifier.SqlIdentifier]:
        res = self._model_client.show_versions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            statement_params=statement_params,
        )
        for r in res:
            if alias_name in r[self._model_client.MODEL_VERSION_ALIASES_COL_NAME]:
                return sql_identifier.SqlIdentifier(
                    r[self._model_client.MODEL_VERSION_NAME_COL_NAME], case_sensitive=True
                )
        return None

    def get_tag_value(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        tag_database_name: Optional[sql_identifier.SqlIdentifier],
        tag_schema_name: Optional[sql_identifier.SqlIdentifier],
        tag_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        r = self._tag_client.get_tag_value(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            tag_database_name=tag_database_name,
            tag_schema_name=tag_schema_name,
            tag_name=tag_name,
            statement_params=statement_params,
        )
        value = r.TAG_VALUE
        if value is None:
            return value
        return str(value)

    def show_tags(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        tags_info = self._tag_client.get_tag_list(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            statement_params=statement_params,
        )
        res: dict[str, str] = {
            identifier.get_schema_level_object_identifier(
                sql_identifier.SqlIdentifier(r.TAG_DATABASE, case_sensitive=True),
                sql_identifier.SqlIdentifier(r.TAG_SCHEMA, case_sensitive=True),
                sql_identifier.SqlIdentifier(r.TAG_NAME, case_sensitive=True),
            ): str(r.TAG_VALUE)
            for r in tags_info
        }
        return res

    def set_tag(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        tag_database_name: Optional[sql_identifier.SqlIdentifier],
        tag_schema_name: Optional[sql_identifier.SqlIdentifier],
        tag_name: sql_identifier.SqlIdentifier,
        tag_value: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self._tag_client.set_tag_on_model(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            tag_database_name=tag_database_name,
            tag_schema_name=tag_schema_name,
            tag_name=tag_name,
            tag_value=tag_value,
            statement_params=statement_params,
        )

    def unset_tag(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        tag_database_name: Optional[sql_identifier.SqlIdentifier],
        tag_schema_name: Optional[sql_identifier.SqlIdentifier],
        tag_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self._tag_client.unset_tag_on_model(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            tag_database_name=tag_database_name,
            tag_schema_name=tag_schema_name,
            tag_name=tag_name,
            statement_params=statement_params,
        )

    def _is_privatelink_connection(self) -> bool:
        """Detect if the current session is using a privatelink connection."""
        try:
            host = self._session.connection.host
            return ModelOperator.PRIVATELINK_INGRESS_ENDPOINT_URL_SUBSTRING in host
        except AttributeError:
            return False

    def _extract_and_validate_ingress_url(self, res_row: "row.Row") -> Optional[str]:
        """Extract and validate ingress URL from endpoint row."""
        url_value = res_row[self._service_client.MODEL_INFERENCE_SERVICE_ENDPOINT_INGRESS_URL_COL_NAME]
        if url_value is None:
            return None
        url_str = str(url_value)
        return url_str if url_str.endswith(ModelOperator.INGRESS_ENDPOINT_URL_SUFFIX) else None

    def _extract_and_validate_privatelink_url(self, res_row: "row.Row") -> Optional[str]:
        """Extract and validate privatelink ingress URL from endpoint row."""
        # Check if the privatelink_ingress_url column exists
        col_name = self._service_client.MODEL_INFERENCE_SERVICE_ENDPOINT_PRIVATELINK_INGRESS_URL_COL_NAME
        if col_name not in res_row:
            # Column doesn't exist in query result for non-Business Critical accounts
            return None

        url_value = res_row[col_name]
        if url_value is None:
            return None
        url_str = str(url_value)
        return url_str if ModelOperator.PRIVATELINK_INGRESS_ENDPOINT_URL_SUBSTRING in url_str else None

    def show_services(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[ServiceInfo]:
        res = self._model_client.show_versions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        service_col_name = self._model_client.MODEL_VERSION_INFERENCE_SERVICES_COL_NAME
        if service_col_name not in res[0]:
            # User need to opt into BCR 2024_08
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.OPT_IN_REQUIRED,
                original_exception=RuntimeError(
                    "Please opt in to BCR Bundle 2024_08 ("
                    "https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_08_bundle)."
                ),
            )

        json_array = json.loads(res[0][service_col_name])
        # TODO(sdas): Figure out a better way to filter out MODEL_BUILD_ services server side.
        fully_qualified_service_names = [str(service) for service in json_array if "MODEL_BUILD_" not in service]

        result: list[ServiceInfo] = []
        is_privatelink_connection = self._is_privatelink_connection()

        for fully_qualified_service_name in fully_qualified_service_names:
            inference_endpoint: Optional[str] = None
            db, schema, service_name = sql_identifier.parse_fully_qualified_name(fully_qualified_service_name)
            statuses = self._service_client.get_service_container_statuses(
                database_name=db, schema_name=schema, service_name=service_name, statement_params=statement_params
            )
            if len(statuses) == 0:
                return result

            service_status = statuses[0].service_status
            for res_row in self._service_client.show_endpoints(
                database_name=db, schema_name=schema, service_name=service_name, statement_params=statement_params
            ):
                if (
                    res_row[self._service_client.MODEL_INFERENCE_SERVICE_ENDPOINT_NAME_COL_NAME]
                    != self.INFERENCE_SERVICE_ENDPOINT_NAME
                ):
                    continue

                ingress_url = self._extract_and_validate_ingress_url(res_row)
                privatelink_ingress_url = self._extract_and_validate_privatelink_url(res_row)

                if is_privatelink_connection and privatelink_ingress_url is not None:
                    inference_endpoint = privatelink_ingress_url
                else:
                    inference_endpoint = ingress_url

            result.append(
                ServiceInfo(
                    name=fully_qualified_service_name,
                    status=service_status.value,
                    inference_endpoint=inference_endpoint,
                )
            )

        return result

    def delete_service(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        service_database_name: Optional[sql_identifier.SqlIdentifier],
        service_schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        services = self.show_services(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

        # Fall back to the model's database and schema.
        # database_name or schema_name are set if the model is created or get using fully qualified name
        # Otherwise, the model's database and schema are same as registry's database and schema, which are set in the
        # self._model_client.

        service_database_name = service_database_name or database_name or self._model_client._database_name
        service_schema_name = service_schema_name or schema_name or self._model_client._schema_name
        fully_qualified_service_name = sql_identifier.get_fully_qualified_name(
            service_database_name, service_schema_name, service_name
        )

        for service_info in services:
            if service_info["name"] == fully_qualified_service_name:
                self._service_client.drop_service(
                    database_name=service_database_name,
                    schema_name=service_schema_name,
                    service_name=service_name,
                    statement_params=statement_params,
                )
                return
        raise ValueError(
            f"Service '{fully_qualified_service_name}' does not exist "
            "or unauthorized or not associated with this model version."
        )

    def get_model_version_manifest(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> model_manifest_schema.ModelManifestDict:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._model_version_client.get_file(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                file_path=pathlib.PurePosixPath(model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH),
                target_path=pathlib.Path(tmpdir),
                statement_params=statement_params,
            )
            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))
            return mm.load()

    @staticmethod
    def _match_model_spec_with_sql_functions(
        sql_functions_names: list[sql_identifier.SqlIdentifier], target_methods: list[str]
    ) -> dict[sql_identifier.SqlIdentifier, str]:
        res: dict[sql_identifier.SqlIdentifier, str] = {}

        for target_method in target_methods:
            # Here we need to find the SQL function corresponding to the Python function.
            # If the python function name is `abc`, then SQL function name can be `ABC` or `"abc"`.
            # We will try to match`"abc"` first, then `ABC`.
            # The reason why is because, if we have two python methods whose names are `abc` and `aBc`.
            # At most 1 of them can be `ABC`, so if we check `"abc"` or `"aBc"` first we could resolve them correctly.
            function_name = sql_identifier.SqlIdentifier(target_method, case_sensitive=True)
            if function_name not in sql_functions_names:
                function_name = sql_identifier.SqlIdentifier(target_method)
                assert (
                    function_name in sql_functions_names
                ), f"Unable to match {target_method} in {sql_functions_names}."
            res[function_name] = target_method
        return res

    def _fetch_model_spec(
        self,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> model_meta_schema.ModelMetadataDict:
        raw_model_spec_res = self._model_client.show_versions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            check_model_details=True,
            statement_params={**(statement_params or {}), "SHOW_MODEL_DETAILS_IN_SHOW_VERSIONS_IN_MODEL": True},
        )[0][self._model_client.MODEL_VERSION_MODEL_SPEC_COL_NAME]
        model_spec_dict = yaml.safe_load(raw_model_spec_res)
        model_spec = model_meta.ModelMetadata._validate_model_metadata(model_spec_dict)
        return model_spec

    def get_model_task(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> type_hints.Task:
        model_version = self._model_client.show_versions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            validate_result=True,
            statement_params=statement_params,
        )[0]

        model_attributes = json.loads(model_version.model_attributes)
        task_val = model_attributes.get("task", type_hints.Task.UNKNOWN.value)
        return type_hints.Task(task_val)

    def get_functions(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[model_manifest_schema.ModelFunctionInfo]:
        model_spec = self._fetch_model_spec(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        show_functions_res = self._model_version_client.show_functions(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )
        function_names_and_types = []
        for r in show_functions_res:
            function_name = sql_identifier.SqlIdentifier(
                r[self._model_version_client.FUNCTION_NAME_COL_NAME], case_sensitive=True
            )

            function_type = model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value
            try:
                return_type = r[self._model_version_client.FUNCTION_RETURN_TYPE_COL_NAME]
            except KeyError:
                pass
            else:
                if "TABLE" in return_type:
                    function_type = model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value

            function_names_and_types.append((function_name, function_type))

        if not function_names_and_types:
            # If function_names_and_types is not populated, there are currently
            # no warehouse functions for the model version. In order to do inference
            # we must populate the functions so the mapping can be constructed.
            model_manifest = self.get_model_version_manifest(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
            for method in model_manifest["methods"]:
                function_names_and_types.append((sql_identifier.SqlIdentifier(method["name"]), method["type"]))

        signatures = model_spec["signatures"]
        function_names = [name for name, _ in function_names_and_types]
        function_name_mapping = ModelOperator._match_model_spec_with_sql_functions(
            function_names, list(signatures.keys())
        )

        model_func_info = []

        for function_name, function_type in function_names_and_types:

            target_method = function_name_mapping[function_name]

            is_partitioned = False
            if function_type == model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value:
                # better to set default True here because worse case it will be slow but not error out
                is_partitioned = (
                    (
                        model_spec["function_properties"]
                        .get(target_method, {})
                        .get(model_meta_schema.FunctionProperties.PARTITIONED.value, True)
                    )
                    if "function_properties" in model_spec
                    else True
                )

            model_func_info.append(
                model_manifest_schema.ModelFunctionInfo(
                    name=function_name.identifier(),
                    target_method=target_method,
                    target_method_function_type=function_type,
                    signature=model_signature.ModelSignature.from_dict(signatures[target_method]),
                    is_partitioned=is_partitioned,
                )
            )

        return model_func_info

    @overload
    def invoke_method(
        self,
        *,
        method_name: sql_identifier.SqlIdentifier,
        method_function_type: str,
        signature: model_signature.ModelSignature,
        X: Union[type_hints.SupportedDataType, dataframe.DataFrame],
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        strict_input_validation: bool = False,
        partition_column: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, str]] = None,
        is_partitioned: Optional[bool] = None,
        explain_case_sensitive: bool = False,
    ) -> Union[type_hints.SupportedDataType, dataframe.DataFrame]:
        ...

    @overload
    def invoke_method(
        self,
        *,
        method_name: sql_identifier.SqlIdentifier,
        signature: model_signature.ModelSignature,
        X: Union[type_hints.SupportedDataType, dataframe.DataFrame],
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        strict_input_validation: bool = False,
        statement_params: Optional[dict[str, str]] = None,
        explain_case_sensitive: bool = False,
    ) -> Union[type_hints.SupportedDataType, dataframe.DataFrame]:
        ...

    def invoke_method(
        self,
        *,
        method_name: sql_identifier.SqlIdentifier,
        method_function_type: Optional[str] = None,
        signature: model_signature.ModelSignature,
        X: Union[type_hints.SupportedDataType, dataframe.DataFrame],
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        service_name: Optional[sql_identifier.SqlIdentifier] = None,
        strict_input_validation: bool = False,
        partition_column: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, str]] = None,
        is_partitioned: Optional[bool] = None,
        explain_case_sensitive: bool = False,
    ) -> Union[type_hints.SupportedDataType, dataframe.DataFrame]:
        identifier_rule = model_signature.SnowparkIdentifierRule.INFERRED

        # Validate and prepare input
        if not isinstance(X, dataframe.DataFrame):
            keep_order = True
            output_with_input_features = False
            df = model_signature._convert_and_validate_local_data(X, signature.inputs, strict=strict_input_validation)
            s_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
                self._session, df, keep_order=keep_order, features=signature.inputs, statement_params=statement_params
            )
        else:
            keep_order = False
            output_with_input_features = True
            identifier_rule = model_signature._validate_snowpark_data(
                X, signature.inputs, strict=strict_input_validation
            )
            s_df = X

        original_cols = s_df.columns

        # Compose input and output names
        input_args = []
        quoted_identifiers_ignore_case = (
            snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                self._session, statement_params
            )
        )

        for input_feature in signature.inputs:
            col_name = identifier_rule.get_sql_identifier_from_feature(input_feature.name)
            if quoted_identifiers_ignore_case:
                col_name = sql_identifier.SqlIdentifier(input_feature.name.upper(), case_sensitive=True)
            input_args.append(col_name)

        returns = []
        for output_feature in signature.outputs:
            output_name = identifier_rule.get_sql_identifier_from_feature(output_feature.name)
            returns.append((output_feature.name, output_feature.as_snowpark_type(), output_name))
            # Avoid removing output cols when output_with_input_features is False
            if output_name in original_cols:
                original_cols.remove(output_name)

        if service_name:
            df_res = self._service_client.invoke_function_method(
                method_name=method_name,
                input_df=s_df,
                input_args=input_args,
                returns=returns,
                database_name=database_name,
                schema_name=schema_name,
                service_name=service_name,
                statement_params=statement_params,
            )
        else:
            assert model_name is not None
            assert version_name is not None
            if method_function_type == model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value:
                df_res = self._model_version_client.invoke_function_method(
                    method_name=method_name,
                    input_df=s_df,
                    input_args=input_args,
                    returns=returns,
                    database_name=database_name,
                    schema_name=schema_name,
                    model_name=model_name,
                    version_name=version_name,
                    statement_params=statement_params,
                )
            elif method_function_type == model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value:
                df_res = self._model_version_client.invoke_table_function_method(
                    method_name=method_name,
                    input_df=s_df,
                    input_args=input_args,
                    partition_column=partition_column,
                    returns=returns,
                    database_name=database_name,
                    schema_name=schema_name,
                    model_name=model_name,
                    version_name=version_name,
                    statement_params=statement_params,
                    is_partitioned=is_partitioned or False,
                    explain_case_sensitive=explain_case_sensitive,
                )

        if keep_order:
            # if it's a partitioned table function, _ID will be null and we won't be able to sort.
            if df_res.select(snowpark_handler._KEEP_ORDER_COL_NAME).limit(1).collect()[0][0] is None:
                warnings.warn(
                    formatting.unwrap(
                        """
                        When invoking partitioned inference methods, ordering of rows in output dataframe will differ
                        from that of input dataframe.
                        """
                    ),
                    category=UserWarning,
                    stacklevel=1,
                )
            else:
                df_res = df_res.sort(
                    snowpark_handler._KEEP_ORDER_COL_NAME,
                    ascending=True,
                )

        if not output_with_input_features:
            cols_to_drop = original_cols
            if partition_column is not None:
                # don't drop partition column
                cols_to_drop.remove(partition_column.identifier())
            df_res = df_res.drop(*cols_to_drop)

        # Get final result
        if not isinstance(X, dataframe.DataFrame):
            return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(
                df_res, features=signature.outputs, statement_params=statement_params
            )
        else:
            return df_res

    def delete_model_or_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        if version_name:
            self._model_version_client.drop_version(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            self._model_client.drop_model(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                statement_params=statement_params,
            )

    def rename(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        new_model_db: Optional[sql_identifier.SqlIdentifier],
        new_model_schema: Optional[sql_identifier.SqlIdentifier],
        new_model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self._model_client.rename(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            new_model_db=new_model_db,
            new_model_schema=new_model_schema,
            new_model_name=new_model_name,
            statement_params=statement_params,
        )

    # Map indicating in different modes, the path to list and download.
    # The boolean value indicates if it is a directory,
    MODEL_FILE_DOWNLOAD_PATTERN = {
        "minimal": {
            pathlib.PurePosixPath(model_composer.ModelComposer.MODEL_DIR_REL_PATH)
            / model_meta.MODEL_METADATA_FILE: False,
            pathlib.PurePosixPath(model_composer.ModelComposer.MODEL_DIR_REL_PATH) / model_env._DEFAULT_ENV_DIR: True,
            pathlib.PurePosixPath(model_composer.ModelComposer.MODEL_DIR_REL_PATH)
            / model_runtime.ModelRuntime.RUNTIME_DIR_REL_PATH: True,
        },
        "model": {pathlib.PurePosixPath(model_composer.ModelComposer.MODEL_DIR_REL_PATH): True},
        "full": {pathlib.PurePosixPath(os.curdir): True},
    }

    def download_files(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        target_path: pathlib.Path,
        mode: Literal["full", "model", "minimal"] = "model",
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        for remote_rel_path, is_dir in self.MODEL_FILE_DOWNLOAD_PATTERN[mode].items():
            list_file_res = self._model_version_client.list_file(
                database_name=database_name,
                schema_name=schema_name,
                model_name=model_name,
                version_name=version_name,
                file_path=remote_rel_path,
                is_dir=is_dir,
                statement_params=statement_params,
            )
            file_list = [
                pathlib.PurePosixPath(*pathlib.PurePosixPath(row.name).parts[2:])  # versions/<version_name>/...
                for row in list_file_res
            ]
            for stage_file_path in file_list:
                local_file_dir = target_path / stage_file_path.parent
                local_file_dir.mkdir(parents=True, exist_ok=True)
                self._model_version_client.get_file(
                    database_name=database_name,
                    schema_name=schema_name,
                    model_name=model_name,
                    version_name=version_name,
                    file_path=stage_file_path,
                    target_path=local_file_dir,
                    statement_params=statement_params,
                )
