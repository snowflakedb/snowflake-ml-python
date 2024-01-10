from typing import Any, Dict, List, Optional

from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    sql_identifier,
)
from snowflake.snowpark import row, session


class ModuleTagSQLClient:
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
        if not isinstance(__value, ModuleTagSQLClient):
            return False
        return self._database_name == __value._database_name and self._schema_name == __value._schema_name

    def fully_qualified_module_name(
        self,
        module_name: sql_identifier.SqlIdentifier,
    ) -> str:
        return identifier.get_schema_level_object_identifier(
            self._database_name.identifier(), self._schema_name.identifier(), module_name.identifier()
        )

    def set_tag_on_model(
        self,
        model_name: sql_identifier.SqlIdentifier,
        *,
        tag_database_name: sql_identifier.SqlIdentifier,
        tag_schema_name: sql_identifier.SqlIdentifier,
        tag_name: sql_identifier.SqlIdentifier,
        tag_value: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        fq_model_name = self.fully_qualified_module_name(model_name)
        fq_tag_name = identifier.get_schema_level_object_identifier(
            tag_database_name.identifier(), tag_schema_name.identifier(), tag_name.identifier()
        )
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER MODEL {fq_model_name} SET TAG {fq_tag_name} = $${tag_value}$$",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def unset_tag_on_model(
        self,
        model_name: sql_identifier.SqlIdentifier,
        *,
        tag_database_name: sql_identifier.SqlIdentifier,
        tag_schema_name: sql_identifier.SqlIdentifier,
        tag_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        fq_model_name = self.fully_qualified_module_name(model_name)
        fq_tag_name = identifier.get_schema_level_object_identifier(
            tag_database_name.identifier(), tag_schema_name.identifier(), tag_name.identifier()
        )
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER MODEL {fq_model_name} UNSET TAG {fq_tag_name}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def get_tag_value(
        self,
        module_name: sql_identifier.SqlIdentifier,
        *,
        tag_database_name: sql_identifier.SqlIdentifier,
        tag_schema_name: sql_identifier.SqlIdentifier,
        tag_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> row.Row:
        fq_module_name = self.fully_qualified_module_name(module_name)
        fq_tag_name = identifier.get_schema_level_object_identifier(
            tag_database_name.identifier(), tag_schema_name.identifier(), tag_name.identifier()
        )
        return (
            query_result_checker.SqlResultValidator(
                self._session,
                f"SELECT SYSTEM$GET_TAG($${fq_tag_name}$$, $${fq_module_name}$$, 'MODULE') AS TAG_VALUE",
                statement_params=statement_params,
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_column("TAG_VALUE")
            .validate()[0]
        )

    def get_tag_list(
        self,
        module_name: sql_identifier.SqlIdentifier,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[row.Row]:
        fq_module_name = self.fully_qualified_module_name(module_name)
        return (
            query_result_checker.SqlResultValidator(
                self._session,
                f"""SELECT TAG_DATABASE, TAG_SCHEMA, TAG_NAME, TAG_VALUE
FROM TABLE({self._database_name.identifier()}.INFORMATION_SCHEMA.TAG_REFERENCES($${fq_module_name}$$, 'MODULE'))""",
                statement_params=statement_params,
            )
            .has_column("TAG_DATABASE", allow_empty=True)
            .has_column("TAG_SCHEMA", allow_empty=True)
            .has_column("TAG_NAME", allow_empty=True)
            .has_column("TAG_VALUE", allow_empty=True)
            .validate()
        )
