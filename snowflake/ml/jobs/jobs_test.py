from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from snowflake.ml.jobs import job
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark.row import Row


class JobTest(parameterized.TestCase):
    def test_get_instance_negative(self) -> None:
        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()

            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                mock_result.collect.return_value = [
                    Row(instance_id=None),
                ]
            elif query_str.startswith("SHOW SERVICE INSTANCES"):
                mock_result.collect.return_value = [Row(start_time=None, instance_id=None)]
            else:
                raise ValueError(f"Unexpected SQL: {query_str}")

            return mock_result

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect
        with patch("snowflake.ml.jobs.job._get_target_instances") as mock_get_target_instances:
            mock_get_target_instances.return_value = 2
            with self.assertRaisesRegex(
                RuntimeError,
                "Couldn’t retrieve head instance due to missing instances.",
            ):
                job._get_head_instance_id(mock_session, "test_db.test_schema.test_id")

    def test_get_logs_negative(self) -> None:
        mock_session = MagicMock()
        mock_session.sql.side_effect = sp_exceptions.SnowparkSQLException("Waiting to start, Container Status: PENDING")
        job = jobs.MLJob[None]("jobs_DB.jobs_schema.test_id", session=mock_session)
        with self.assertLogs("root", level="WARNING") as cm:
            job.get_logs()
            self.assertIn("Waiting for container to start. Logs will be shown when available.", cm.output[0])

    def test_get_logs_fallback(self) -> None:
        def sql_side_effect(query_str: str, params: list[Any]) -> MagicMock:
            mock_result = MagicMock()
            if query_str.startswith("SELECT VALUE FROM snowflake.telemetry.events_view"):
                mock_result.collect.return_value = [
                    Row(VALUE="test_log_0"),
                    Row(VALUE="test_log_1"),
                    Row(VALUE="test_log_2"),
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
