from typing import Any, Optional, Sequence

from snowflake import snowpark
from snowflake.snowpark import Row
from snowflake.snowpark._internal import utils
from snowflake.snowpark._internal.analyzer import snowflake_plan
from snowflake.snowpark._internal.utils import is_in_stored_procedure


def result_set_to_rows(session: snowpark.Session, result: dict[str, Any]) -> list[Row]:
    metadata = session._conn._cursor.description
    result_set = result["data"]
    return utils.result_set_to_rows(result_set, metadata)


@snowflake_plan.SnowflakePlan.Decorator.wrap_exception  # type: ignore[misc]
def run_query(session: snowpark.Session, query_text: str, params: Optional[Sequence[Any]] = None) -> list[Row]:
    kwargs: dict[str, Any] = {"query": query_text, "params": params}
    if not is_in_stored_procedure():  # type: ignore[no-untyped-call]
        kwargs["_force_qmark_paramstyle"] = True
    result = session._conn.run_query(**kwargs)
    if not isinstance(result, dict) or "data" not in result:
        raise ValueError(f"Unprocessable result: {result}")
    return result_set_to_rows(session, result)
