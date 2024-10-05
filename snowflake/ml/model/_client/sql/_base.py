from typing import Optional

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.snowpark import session
from snowflake.snowpark._internal import utils as snowpark_utils


class _BaseSQLClient:
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
        if not isinstance(__value, _BaseSQLClient):
            return False
        return self._database_name == __value._database_name and self._schema_name == __value._schema_name

    def fully_qualified_object_name(
        self,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        object_name: sql_identifier.SqlIdentifier,
    ) -> str:
        actual_database_name = database_name or self._database_name
        actual_schema_name = schema_name or self._schema_name
        return identifier.get_schema_level_object_identifier(
            actual_database_name.identifier(), actual_schema_name.identifier(), object_name.identifier()
        )

    @staticmethod
    def get_tmp_name_with_prefix(prefix: str) -> str:
        return f"{prefix}_{snowpark_utils.generate_random_alphanumeric().upper()}"
