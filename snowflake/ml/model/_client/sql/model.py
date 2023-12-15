from typing import Any, Dict, List, Optional

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.snowpark import row, session


class ModelSQLClient:
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
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[row.Row]:
        fully_qualified_schema_name = ".".join([self._database_name.identifier(), self._schema_name.identifier()])
        like_sql = ""
        if model_name:
            like_sql = f" LIKE '{model_name.resolved()}'"
        res = self._session.sql(f"SHOW MODELS{like_sql} IN SCHEMA {fully_qualified_schema_name}")

        return res.collect(statement_params=statement_params)

    def show_versions(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[row.Row]:
        like_sql = ""
        if version_name:
            like_sql = f" LIKE '{version_name.resolved()}'"
        res = self._session.sql(f"SHOW VERSIONS{like_sql} IN MODEL {self.fully_qualified_model_name(model_name)}")

        return res.collect(statement_params=statement_params)

    def set_comment(
        self,
        *,
        comment: str,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        comment_sql = f"COMMENT ON MODEL {self.fully_qualified_model_name(model_name)} IS $${comment}$$"
        self._session.sql(comment_sql).collect(statement_params=statement_params)

    def drop_model(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._session.sql(f"DROP MODEL {self.fully_qualified_model_name(model_name)}").collect(
            statement_params=statement_params
        )
