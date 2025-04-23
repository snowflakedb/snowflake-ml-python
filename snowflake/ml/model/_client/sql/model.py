from typing import Any, Optional

from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import row


class ModelSQLClient(_base._BaseSQLClient):
    MODEL_NAME_COL_NAME = "name"
    MODEL_COMMENT_COL_NAME = "comment"
    MODEL_DEFAULT_VERSION_NAME_COL_NAME = "default_version_name"

    MODEL_VERSION_NAME_COL_NAME = "name"
    MODEL_VERSION_COMMENT_COL_NAME = "comment"
    MODEL_VERSION_METADATA_COL_NAME = "metadata"
    MODEL_VERSION_MODEL_SPEC_COL_NAME = "model_spec"
    MODEL_VERSION_ALIASES_COL_NAME = "aliases"
    MODEL_VERSION_INFERENCE_SERVICES_COL_NAME = "inference_services"

    def show_models(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        validate_result: bool = True,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        actual_database_name = database_name or self._database_name
        actual_schema_name = schema_name or self._schema_name
        fully_qualified_schema_name = ".".join([actual_database_name.identifier(), actual_schema_name.identifier()])
        like_sql = ""
        if model_name:
            like_sql = f" LIKE '{model_name.resolved()}'"

        res = (
            query_result_checker.SqlResultValidator(
                self._session,
                f"SHOW MODELS{like_sql} IN SCHEMA {fully_qualified_schema_name}",
                statement_params=statement_params,
            )
            .has_column(ModelSQLClient.MODEL_NAME_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_COMMENT_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_DEFAULT_VERSION_NAME_COL_NAME, allow_empty=True)
        )
        if validate_result and model_name:
            res = res.has_dimensions(expected_rows=1)

        return res.validate()

    def show_versions(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        validate_result: bool = True,
        check_model_details: bool = False,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        like_sql = ""
        if version_name:
            like_sql = f" LIKE '{version_name.resolved()}'"

        res = (
            query_result_checker.SqlResultValidator(
                self._session,
                (
                    f"SHOW VERSIONS{like_sql} IN "
                    f"MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                ),
                statement_params=statement_params,
            )
            .has_column(ModelSQLClient.MODEL_VERSION_NAME_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_VERSION_COMMENT_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_VERSION_METADATA_COL_NAME, allow_empty=True)
        )
        if validate_result and version_name:
            res = res.has_dimensions(expected_rows=1)
        if check_model_details:
            res = res.has_column(ModelSQLClient.MODEL_VERSION_MODEL_SPEC_COL_NAME, allow_empty=True)

        return res.validate()

    def set_comment(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        comment: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"COMMENT ON MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" IS $${comment}$$"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def drop_model(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"DROP MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

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
        # Use registry's database and schema if a non fully qualified new model name is provided.
        new_fully_qualified_name = self.fully_qualified_object_name(new_model_db, new_model_schema, new_model_name)
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" RENAME TO {new_fully_qualified_name}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()
