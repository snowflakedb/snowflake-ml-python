import json
from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.connector import errors
from snowflake.ml import jobs
from snowflake.ml.jobs import job
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark.row import Row


class JobTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("target_instances=2", {"data": [("_", "_", "_", "_", "_", "_", "_", "_", "_", 2)]}, 2),
    )
    def test_get_target_instances_positive(self, sql_result: dict[str, Any], expected_result: int) -> None:
        def sql_side_effect(query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return sql_result
            return None

        mock_session = MagicMock()
        mock_session._conn.run_query.side_effect = sql_side_effect
        target_instances = job._get_target_instances(mock_session, "jobs_DB.jobs_schema.test_id")
        self.assertEqual(target_instances, expected_result)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("target instance is 1", 1, {"data": [("_", "_", "_", "_", None, "_", "_", "_", None)]}, 0),
        (
            "start_time and instance_id are not None",
            2,
            {
                "data": [
                    ("_", "_", "_", "_", 0, "_", "_", "_", "2025-01-01"),
                    ("_", "_", "_", "_", 1, "_", "_", "_", "2025-01-01"),
                ]
            },
            0,
        ),
    )
    def test_get_head_instance_id_positive(
        self, target_instances: int, sql_result: list[Row], expected_result: int
    ) -> None:
        def sql_side_effect(query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return {"data": [("_", "_", "_", "_", "_", "_", "_", "_", "_", target_instances)]}
            elif query_str.startswith("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER"):
                return sql_result

        mock_session = MagicMock()
        mock_session._conn.run_query.side_effect = sql_side_effect
        head_instance_id = job._get_head_instance_id(mock_session, "jobs_DB.jobs_schema.test_id")
        self.assertEqual(head_instance_id, expected_result)

    @parameterized.named_parameters(  # type: ignore[misc]
        (
            "target_instances > len(rows)",
            {"data": [("_", "_", "_", "_", "_", "_", "_", "_", "_", 2)]},
            {"data": [("_", "_", "_", "_", 1, "_", "_", "_", "2025-01-01")]},
            RuntimeError,
        ),
        (
            "start_time or instance_id is None",
            {"data": [("_", "_", "_", "_", "_", "_", "_", "_", "_", 2)]},
            {
                "data": [
                    ("_", "_", "_", "_", None, "_", "_", "_", None),
                    ("_", "_", "_", "_", 1, "_", "_", "_", "2025-01-01"),
                ]
            },
            RuntimeError,
        ),
    )
    def test_get_head_instance_id_negative(
        self,
        target_instances: dict[str, tuple[Any]],
        sql_result: dict[str, tuple[Any]],
        expected_error: type[Exception],
    ) -> None:
        def sql_side_effect(query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return target_instances
            elif query_str.startswith("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER"):
                return sql_result

        mock_session = MagicMock()
        mock_session._conn.run_query.side_effect = sql_side_effect
        with self.assertRaises(expected_error):
            job._get_head_instance_id(mock_session, "jobs_DB.jobs_schema.test_id")

    def test_get_logs_negative(self) -> None:
        mock_session = MagicMock()

        def sql_side_effect(query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return ({"data": [("_", "_", "_", "_", "_", "_", "_", "_", "_", 2)]},)
            else:
                raise errors.ProgrammingError("Waiting to start, Container Status: PENDING")

        mock_session._conn.run_query.side_effect = sql_side_effect
        job = jobs.MLJob[None]("jobs_DB.jobs_schema.test_id", session=mock_session)
        with self.assertLogs("root", level="WARNING") as cm:
            job.get_logs()
            self.assertIn("Waiting for container to start. Logs will be shown when available.", cm.output[0])

    def test_get_logs_from_event_table(self) -> None:
        def sql_side_effect(query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return {"data": [("_", "_", "_", "_", "_", "_", "_", "_", "_", 2)]}
            elif query_str.startswith("SELECT VALUE FROM snowflake.telemetry.events_view"):
                return {"data": [(json.dumps("test_log_0"),), (json.dumps("test_log_1"),), (json.dumps("test_log_2"),)]}

            elif query_str.startswith("SHOW SERVICE INSTANCES"):
                raise errors.ProgrammingError("does not exist")
            elif query_str.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
                raise errors.ProgrammingError("Unable to get container status for instance id: 0. Available instances")

        mock_session = MagicMock()
        mock_session._conn.run_query.side_effect = sql_side_effect
        with patch(
            "snowflake.ml.jobs.job._get_logs_spcs",
            side_effect=sp_exceptions.SnowparkSQLException("Unknown user-defined table function", sql_error_code=2143),
        ):
            job = jobs.MLJob[None]("test_db.test_schema.test_id", session=mock_session)
            test_logs = ["test_log_0", "test_log_1", "test_log_2"]
            self.assertEqual(job.get_logs(), "\n".join(test_logs))


if __name__ == "__main__":
    absltest.main()
