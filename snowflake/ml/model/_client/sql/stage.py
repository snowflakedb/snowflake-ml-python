from typing import Any, Dict, Optional

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.snowpark import session


class StageSQLClient:
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

    def fully_qualified_stage_name(
        self,
        stage_name: sql_identifier.SqlIdentifier,
    ) -> str:
        return identifier.get_schema_level_object_identifier(
            self._database_name.identifier(), self._schema_name.identifier(), stage_name.identifier()
        )

    def create_sse_stage(
        self,
        *,
        stage_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._session.sql(
            f"CREATE TEMPORARY STAGE {self.fully_qualified_stage_name(stage_name)} ENCRYPTION = (TYPE= 'SNOWFLAKE_SSE')"
        ).collect(statement_params=statement_params)
