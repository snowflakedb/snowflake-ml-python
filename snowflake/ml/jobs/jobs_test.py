import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from snowflake.ml.jobs import job
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark.row import Row


class JobTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("target_instances=2", [Row(target_instances=2)], 2),
    )
    def test_get_target_instances_positive(self, sql_result: list[Row], expected_result: int) -> None:
        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = sql_result
            return mock_result

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect
        target_instances = job._get_target_instances(mock_session, "jobs_DB.jobs_schema.test_id")
        self.assertEqual(target_instances, expected_result)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("target instance is 1", 1, [Row(start_time=None, instance_id=None)], 0),
        (
            "start_time and instance_id are not None",
            2,
            [Row(start_time="2025-01-01", instance_id=0), Row(start_time="2025-01-01", instance_id=1)],
            0,
        ),
    )
    def test_get_head_instance_id_positive(
        self, target_instances: int, sql_result: list[Row], expected_result: int
    ) -> None:
        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = [
                    Row(target_instances=target_instances),
                ]
            elif query_str.startswith("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER"):
                mock_result.collect.return_value = sql_result
            return mock_result

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect
        head_instance_id = job._get_head_instance_id(mock_session, "jobs_DB.jobs_schema.test_id")
        self.assertEqual(head_instance_id, expected_result)

    @parameterized.named_parameters(  # type: ignore[misc]
        (
            "target_instances > len(rows)",
            [Row(target_instances=2)],
            [Row(start_time="2025-01-01", instance_id=1)],
            RuntimeError,
        ),
        (
            "start_time or instance_id is None",
            [Row(target_instances=2)],
            [Row(start_time=None, instance_id=None), Row(start_time="2025-01-01", instance_id=1)],
            RuntimeError,
        ),
    )
    def test_get_head_instance_id_negative(
        self, target_instances: list[Row], sql_result: list[Row], expected_error: type[Exception]
    ) -> None:
        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = target_instances
            elif query_str.startswith("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER"):
                mock_result.collect.return_value = sql_result
            return mock_result

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect
        with self.assertRaises(expected_error):
            job._get_head_instance_id(mock_session, "jobs_DB.jobs_schema.test_id")

    def test_get_logs_negative(self) -> None:
        mock_session = MagicMock()

        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = [Row(target_instances=2)]
            else:
                raise sp_exceptions.SnowparkSQLException("Waiting to start, Container Status: PENDING")
            return mock_result

        mock_session.sql.side_effect = sql_side_effect
        job = jobs.MLJob[None]("jobs_DB.jobs_schema.test_id", session=mock_session)
        with self.assertLogs("root", level="WARNING") as cm:
            job.get_logs()
            self.assertIn("Waiting for container to start. Logs will be shown when available.", cm.output[0])

    def test_get_logs_from_event_table(self) -> None:
        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = [
                    Row(target_instances=2),
                ]
            elif query_str.startswith("SELECT VALUE FROM snowflake.telemetry.events_view"):
                mock_result.collect.return_value = [
                    Row(VALUE=json.dumps("test_log_0")),
                    Row(VALUE=json.dumps("test_log_1")),
                    Row(VALUE=json.dumps("test_log_2")),
                ]
            elif query_str.startswith("SHOW SERVICE INSTANCES"):
                mock_result.collect.side_effect = sp_exceptions.SnowparkSQLException("does not exist")
            elif query_str.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
                mock_result.collect.side_effect = sp_exceptions.SnowparkSQLException(
                    "Unable to get container status for instance id: 0. Available instances"
                )

            return mock_result

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect
        with patch(
            "snowflake.ml.jobs.job._get_logs_spcs",
            side_effect=sp_exceptions.SnowparkSQLException("Unknown user-defined table function", sql_error_code=2143),
        ):
            job = jobs.MLJob[None]("test_db.test_schema.test_id", session=mock_session)
            test_logs = ["test_log_0", "test_log_1", "test_log_2"]
            self.assertEqual(job.get_logs(), "\n".join(test_logs))

    def test_get_logs_from_spcs_interface(self) -> None:
        def sql_side_effect(query_str: str, params: Optional[list[Any]] = None) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = [
                    Row(target_instances=2),
                ]
            elif query_str.startswith("SELECT LOG FROM table"):
                mock_result.collect.return_value = [
                    Row(LOG="test_log_0"),
                    Row(LOG="test_log_1"),
                    Row(LOG="test_log_2"),
                ]
            elif query_str.startswith("SHOW SERVICE INSTANCES"):
                mock_result.collect.side_effect = sp_exceptions.SnowparkSQLException("does not exist")
            elif query_str.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
                mock_result.collect.side_effect = sp_exceptions.SnowparkSQLException(
                    "Unable to get container status for instance id: 0. Available instances"
                )

            return mock_result

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect
        job = jobs.MLJob[None]("test_db.test_schema.test_id", session=mock_session)
        test_logs = ["test_log_0", "test_log_1", "test_log_2"]
        self.assertEqual(job.get_logs(), "\n".join(test_logs))


if __name__ == "__main__":
    absltest.main()
