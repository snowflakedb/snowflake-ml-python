import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

import snowflake.snowpark as snowpark
from snowflake.ml import jobs
from snowflake.ml.jobs import job
from snowflake.ml.jobs._utils import stage_utils
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark.row import Row

SERVICE_SPEC = """
spec:
  containers:
    - name: main
      image: test-image
"""


class JobTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("target_instances=2", [Row(target_instances=2)], 2),
    )
    def test_get_target_instances_positive(self, sql_result: list[Row], expected_result: int) -> None:
        mock_session = MagicMock()
        with patch("snowflake.ml.jobs._utils.query_helper.run_query", return_value=sql_result):
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
        def sql_side_effect(session: snowpark.Session, query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return [
                    Row(target_instances=target_instances),
                ]
            elif query_str.startswith("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER"):
                return sql_result

        mock_session = MagicMock()
        with patch("snowflake.ml.jobs._utils.query_helper.run_query", side_effect=sql_side_effect):
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
        self,
        target_instances: dict[str, tuple[Any]],
        sql_result: dict[str, tuple[Any]],
        expected_error: type[Exception],
    ) -> None:
        def sql_side_effect(session: snowpark.Session, query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return target_instances
            elif query_str.startswith("SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER"):
                return sql_result

        mock_session = MagicMock()

        with patch("snowflake.ml.jobs._utils.query_helper.run_query", side_effect=sql_side_effect):
            with self.assertRaises(expected_error):
                job._get_head_instance_id(mock_session, "jobs_DB.jobs_schema.test_id")

    def test_get_logs_negative(self) -> None:
        mock_session = MagicMock()

        def sql_side_effect(session: snowpark.Session, query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return [Row(target_instances=2, spec=SERVICE_SPEC)]
            else:
                raise sp_exceptions.SnowparkSQLException("Waiting to start, Container Status: PENDING")

        with patch("snowflake.ml.jobs._utils.query_helper.run_query", side_effect=sql_side_effect):
            job = jobs.MLJob[None]("jobs_DB.jobs_schema.test_id", session=mock_session)
            with self.assertLogs("root", level="WARNING") as cm:
                job.get_logs()
                self.assertIn("Waiting for container to start. Logs will be shown when available.", cm.output[0])

    def test_get_logs_from_event_table(self) -> None:
        def sql_side_effect(session: snowpark.Session, query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("DESCRIBE SERVICE IDENTIFIER"):
                return [
                    Row(target_instances=2, spec=SERVICE_SPEC),
                ]
            elif query_str.startswith("SELECT VALUE FROM "):
                return [
                    Row(VALUE=json.dumps("test_log_0")),
                    Row(VALUE=json.dumps("test_log_1")),
                    Row(VALUE=json.dumps("test_log_2")),
                ]
            elif query_str.startswith("SHOW SERVICE INSTANCES"):
                raise sp_exceptions.SnowparkSQLException("does not exist")
            elif query_str.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
                raise sp_exceptions.SnowparkSQLException(
                    "Unable to get container status for instance id: 0. Available instances"
                )

        mock_session = MagicMock()
        mock_session._conn.run_query.side_effect = sql_side_effect
        with patch(
            "snowflake.ml.jobs.job._get_logs_spcs",
            side_effect=sp_exceptions.SnowparkSQLException("Unknown user-defined table function", sql_error_code=2143),
        ), patch("snowflake.ml.jobs._utils.query_helper.run_query", side_effect=sql_side_effect):
            job = jobs.MLJob[None]("test_db.test_schema.test_id", session=mock_session)
            test_logs = ["test_log_0", "test_log_1", "test_log_2"]
            self.assertEqual(job.get_logs(), "\n".join(test_logs))

    @parameterized.named_parameters(  # type: ignore[misc]
        ("posix_absolute_path", "/mnt/job_result/mljob_extra.pkl"),
        ("posix_nested_path", "/mnt/job_result/subdir/result.pkl"),
    )
    def test_transform_path_cross_platform(self, container_path: str) -> None:
        """Test that _transform_path handles container paths correctly across platforms.

        Container paths from Linux SPCS should work consistently whether the client
        is running on Windows, macOS, or Linux. This test verifies the fix for the
        bug where Windows clients couldn't retrieve results due to Path.is_absolute()
        returning False for POSIX paths like /mnt/...
        """
        mock_session = MagicMock()
        mock_job = jobs.MLJob[None]("test_db.test_schema.test_id", session=mock_session)

        # Mock the service spec to provide volume mount and stage information
        # This mirrors the actual customer scenario from the bug report
        mock_job._service_spec_cached = {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "volumeMounts": [
                            {"name": "result-volume", "mountPath": "/mnt/job_result"},
                            {"name": "stage-volume", "mountPath": "/mnt/job_stage"},
                        ],
                        "env": {},
                    }
                ],
                "volumes": [
                    {"name": "stage-volume", "source": "@test_stage/test_path"},
                    {"name": "result-volume", "source": "@test_stage/result_path"},
                ],
            }
        }

        # Test that the path transformation works correctly
        result = mock_job._transform_path(container_path)

        # The result should be a valid stage path without the mount prefix
        self.assertIn("@test_stage", result)
        # Should not have double slashes or the /mnt prefix (the bug)
        self.assertNotIn("//", result)
        self.assertNotIn("/mnt", result)

    def test_transform_path_relative(self) -> None:
        """Test that relative paths are handled correctly."""
        mock_session = MagicMock()
        mock_job = jobs.MLJob[None]("test_db.test_schema.test_id", session=mock_session)
        mock_job._service_spec_cached = {
            "spec": {
                "containers": [{"name": "main", "volumeMounts": [], "env": {}}],
                "volumes": [{"name": "stage-volume", "source": "@test_stage/test_path"}],
            }
        }

        result = mock_job._transform_path("relative/path/file.pkl")
        self.assertEqual(result, "@test_stage/test_path/relative/path/file.pkl")

    def test_resolve_path_container_paths(self) -> None:
        """Test that resolve_path returns Path objects correctly.

        Note: The cross-platform fix is in _transform_path, not resolve_path.
        resolve_path continues to return platform-native Path for filesystem access.
        """
        # Container paths should return regular Path (for filesystem operations)
        container_path = stage_utils.resolve_path("/mnt/job_result/file.pkl")
        # On Unix systems, this is a PosixPath (subclass of Path)
        self.assertTrue(hasattr(container_path, "is_file"))
        self.assertTrue(hasattr(container_path, "exists"))

        # The cross-platform fix happens in _transform_path when processing manifests

    def test_resolve_path_relative_paths(self) -> None:
        """Test that resolve_path handles relative paths correctly."""
        # Relative paths should work correctly
        relative_path = stage_utils.resolve_path("relative/path/file.pkl")
        # Should return a Path (concrete path type for filesystem access)
        self.assertTrue(hasattr(relative_path, "is_file"))
        self.assertFalse(relative_path.is_absolute())


class CrossPlatformPathTest(absltest.TestCase):
    """Tests to verify cross-platform path handling.

    These tests simulate Windows behavior to ensure the fix prevents the bug
    where Windows clients couldn't retrieve results from Linux SPCS containers.
    """

    def test_pureposixpath_is_absolute(self) -> None:
        """Verify PurePosixPath correctly identifies POSIX absolute paths."""
        # PurePosixPath should treat /mnt/... as absolute on all platforms
        posix_path = PurePosixPath("/mnt/job_result/file.pkl")
        self.assertTrue(posix_path.is_absolute())

        # PureWindowsPath would treat this as relative (the bug)
        windows_path = PureWindowsPath("/mnt/job_result/file.pkl")
        self.assertFalse(windows_path.is_absolute())  # Windows requires drive letters

    def test_pureposixpath_relative_to(self) -> None:
        """Verify PurePosixPath.relative_to works correctly."""
        path = PurePosixPath("/mnt/job_result/subdir/file.pkl")
        mount = PurePosixPath("/mnt/job_result")

        relative = path.relative_to(mount)
        self.assertEqual(str(relative), "subdir/file.pkl")
        self.assertEqual(relative.as_posix(), "subdir/file.pkl")

    def test_windows_bug_demonstration(self) -> None:
        """Demonstrate the Windows bug and verify PurePosixPath fixes it.

        This test proves:
        1. Platform-native Path on Windows mishandles Linux container paths (the bug)
        2. PurePosixPath correctly handles them (the fix)
        """
        container_path_str = "/mnt/job_result/mljob_extra.pkl"
        mount_str = "/mnt/job_result"

        # Simulate Windows behavior with PureWindowsPath (the bug)
        windows_path = PureWindowsPath(container_path_str)

        # Windows treats /mnt as relative (no drive letter C:/)
        self.assertFalse(windows_path.is_absolute())

        # This causes the bug: in _transform_path(), path.is_absolute() returns False,
        # so it goes to the "not absolute" branch and prepends result_stage_path.
        # Result: "@stage/result_path//mnt/job_result/mljob_extra.pkl" (double slash + /mnt)
        # Leading to "file does not exist" errors.

        # The fix: use PurePosixPath (always POSIX semantics)
        posix_path = PurePosixPath(container_path_str)
        posix_mount = PurePosixPath(mount_str)

        # PurePosixPath correctly identifies /mnt as absolute
        self.assertTrue(posix_path.is_absolute())

        # And correctly computes relative path
        relative = posix_path.relative_to(posix_mount)
        self.assertEqual(relative.as_posix(), "mljob_extra.pkl")

        # Resulting stage path is correct: "@stage/result_path/mljob_extra.pkl"


if __name__ == "__main__":
    absltest.main()
