from typing import Any, Dict, List, Optional

from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    sql_identifier,
)
from snowflake.snowpark import row, session


class ModelSQLClient:
    MODEL_NAME_COL_NAME = "name"
    MODEL_COMMENT_COL_NAME = "comment"
    MODEL_DEFAULT_VERSION_NAME_COL_NAME = "default_version_name"

    MODEL_VERSION_NAME_COL_NAME = "name"
    MODEL_VERSION_COMMENT_COL_NAME = "comment"
    MODEL_VERSION_METADATA_COL_NAME = "metadata"
    MODEL_VERSION_USER_DATA_COL_NAME = "user_data"

    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._session = session
        self._database_name = database_name
        self._schema_name = schema_name

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelSQLClient):
            return False
        return self._database_name == __value._database_name and self._schema_name == __value._schema_name

    def fully_qualified_model_name(self, model_name: sql_identifier.SqlIdentifier) -> str:
        return identifier.get_schema_level_object_identifier(
            self._database_name.identifier(), self._schema_name.identifier(), model_name.identifier()
        )

    def show_models(
        self,
        *,
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        validate_result: bool = True,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[row.Row]:
        fully_qualified_schema_name = ".".join([self._database_name.identifier(), self._schema_name.identifier()])
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
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        validate_result: bool = True,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[row.Row]:
        like_sql = ""
        if version_name:
            like_sql = f" LIKE '{version_name.resolved()}'"

        res = (
            query_result_checker.SqlResultValidator(
                self._session,
                f"SHOW VERSIONS{like_sql} IN MODEL {self.fully_qualified_model_name(model_name)}",
                statement_params=statement_params,
            )
            .has_column(ModelSQLClient.MODEL_VERSION_NAME_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_VERSION_COMMENT_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_VERSION_METADATA_COL_NAME, allow_empty=True)
            .has_column(ModelSQLClient.MODEL_VERSION_USER_DATA_COL_NAME, allow_empty=True)
        )
        if validate_result and version_name:
            res = res.has_dimensions(expected_rows=1)

        return res.validate()

    def set_comment(
        self,
        *,
        comment: str,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"COMMENT ON MODEL {self.fully_qualified_model_name(model_name)} IS $${comment}$$",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def drop_model(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"DROP MODEL {self.fully_qualified_model_name(model_name)}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()
