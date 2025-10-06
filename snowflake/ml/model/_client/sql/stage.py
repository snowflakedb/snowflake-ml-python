from typing import Any, Optional

from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import Row


class StageSQLClient(_base._BaseSQLClient):
    def create_tmp_stage(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        stage_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> str:
        fq_stage_name = self.fully_qualified_object_name(database_name, schema_name, stage_name)
        query_result_checker.SqlResultValidator(
            self._session,
            f"CREATE SCOPED TEMPORARY STAGE {fq_stage_name}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

        return fq_stage_name

    def list_stage(self, stage_name: str) -> list[Row]:
        try:
            list_results = self._session.sql(f"LIST {stage_name}").collect()
        except Exception as e:
            raise RuntimeError(f"Failed to check stage location '{stage_name}': {e}")
        return list_results
