from typing import Any

from snowflake import connector
from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import row

# SHOW VERSIONS can return 0 rows immediately after CREATE/COMMIT while catalog metadata catches up.
_EMPTY_VERSION_RESULT_EXPECTED = "Expected 1 rows"
_EMPTY_VERSION_RESULT_FOUND = "found: 0 rows"


def _is_empty_version_result(exc: BaseException) -> bool:
    if not isinstance(exc, connector.DataError):
        return False
    message = str(exc)
    return _EMPTY_VERSION_RESULT_EXPECTED in message and _EMPTY_VERSION_RESULT_FOUND in message


class ModelSQLClient(_base._BaseSQLClient):
    MODEL_NAME_COL_NAME = "name"
    MODEL_COMMENT_COL_NAME = "comment"
    MODEL_DEFAULT_VERSION_NAME_COL_NAME = "default_version_name"
    MODEL_OWNER_COL_NAME = "owner"

    MODEL_VERSION_NAME_COL_NAME = "name"
    MODEL_VERSION_COMMENT_COL_NAME = "comment"
    MODEL_VERSION_METADATA_COL_NAME = "metadata"
    MODEL_VERSION_MODEL_SPEC_COL_NAME = "model_spec"
    MODEL_VERSION_ALIASES_COL_NAME = "aliases"
    MODEL_VERSION_RUNNABLE_IN_COL_NAME = "runnable_in"
    MODEL_VERSION_INFERENCE_SERVICES_COL_NAME = "inference_services"

    def show_models(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier | None,
        schema_name: sql_identifier.SqlIdentifier | None,
        model_name: sql_identifier.SqlIdentifier | None = None,
        validate_result: bool = True,
        statement_params: dict[str, Any] | None = None,
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
        database_name: sql_identifier.SqlIdentifier | None,
        schema_name: sql_identifier.SqlIdentifier | None,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier | None = None,
        validate_result: bool = True,
        check_model_details: bool = False,
        statement_params: dict[str, Any] | None = None,
        retry: bool | None = False,
    ) -> list[row.Row]:
        like_sql = ""
        if version_name:
            like_sql = f" LIKE '{version_name.resolved()}'"

        def _execute() -> list[row.Row]:
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

        if retry:
            import retrying

            versions: list[row.Row] = retrying.retry(
                retry_on_exception=_is_empty_version_result,
                stop_max_attempt_number=5,
                wait_exponential_multiplier=100,
                wait_exponential_max=10000,
            )(_execute)()
            return versions
        return _execute()

    def set_comment(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier | None,
        schema_name: sql_identifier.SqlIdentifier | None,
        model_name: sql_identifier.SqlIdentifier,
        comment: str,
        statement_params: dict[str, Any] | None = None,
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
        database_name: sql_identifier.SqlIdentifier | None,
        schema_name: sql_identifier.SqlIdentifier | None,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: dict[str, Any] | None = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"DROP MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def rename(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier | None,
        schema_name: sql_identifier.SqlIdentifier | None,
        model_name: sql_identifier.SqlIdentifier,
        new_model_db: sql_identifier.SqlIdentifier | None,
        new_model_schema: sql_identifier.SqlIdentifier | None,
        new_model_name: sql_identifier.SqlIdentifier,
        statement_params: dict[str, Any] | None = None,
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
