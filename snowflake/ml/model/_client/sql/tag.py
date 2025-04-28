from typing import Any, Optional

from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import row


class ModuleTagSQLClient(_base._BaseSQLClient):
    def set_tag_on_model(
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
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        fq_tag_name = self.fully_qualified_object_name(tag_database_name, tag_schema_name, tag_name)
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER MODEL {fq_model_name} SET TAG {fq_tag_name} = $${tag_value}$$",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def unset_tag_on_model(
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
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        fq_tag_name = self.fully_qualified_object_name(tag_database_name, tag_schema_name, tag_name)
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER MODEL {fq_model_name} UNSET TAG {fq_tag_name}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

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
    ) -> row.Row:
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        fq_tag_name = self.fully_qualified_object_name(tag_database_name, tag_schema_name, tag_name)
        return (
            query_result_checker.SqlResultValidator(
                self._session,
                f"SELECT SYSTEM$GET_TAG($${fq_tag_name}$$, $${fq_model_name}$$, 'MODULE') AS TAG_VALUE",
                statement_params=statement_params,
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_column("TAG_VALUE")
            .validate()[0]
        )

    def get_tag_list(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        actual_database_name = database_name or self._database_name
        return (
            query_result_checker.SqlResultValidator(
                self._session,
                f"""SELECT TAG_DATABASE, TAG_SCHEMA, TAG_NAME, TAG_VALUE
FROM TABLE({actual_database_name.identifier()}.INFORMATION_SCHEMA.TAG_REFERENCES($${fq_model_name}$$, 'MODULE'))""",
                statement_params=statement_params,
            )
            .has_column("TAG_DATABASE", allow_empty=True)
            .has_column("TAG_SCHEMA", allow_empty=True)
            .has_column("TAG_NAME", allow_empty=True)
            .has_column("TAG_VALUE", allow_empty=True)
            .validate()
        )
