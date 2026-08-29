import itertools
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

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

    def test_get_submitted_instance_count_reads_replicas(self) -> None:
        row = Row(PARAMETERS=json.dumps({"ASYNC": True, "REPLICAS": 3}))
        with patch.object(job, "_get_service_info_spcs", return_value=row):
            self.assertEqual(job._get_submitted_instance_count(MagicMock(), "jobs_DB.jobs_schema.test_id"), 3)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("other parameters present", {"ASYNC": True}),
        ("empty parameters", {}),
    )
    def test_get_submitted_instance_count_defaults_to_one_when_replicas_omitted(
        self, parameters: dict[str, Any]
    ) -> None:
        row = Row(PARAMETERS=json.dumps(parameters))
        with patch.object(job, "_get_service_info_spcs", return_value=row):
            self.assertEqual(job._get_submitted_instance_count(MagicMock(), "jobs_DB.jobs_schema.test_id"), 1)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("malformed PARAMETERS", Row(PARAMETERS="not-json")),
        ("non-object PARAMETERS", Row(PARAMETERS=json.dumps([]))),
        ("null REPLICAS", Row(PARAMETERS=json.dumps({"REPLICAS": None}))),
        ("non-numeric REPLICAS", Row(PARAMETERS=json.dumps({"REPLICAS": "many"}))),
        ("non-positive REPLICAS", Row(PARAMETERS=json.dumps({"REPLICAS": 0}))),
    )
    def test_get_submitted_instance_count_raises_when_unreadable(self, row: Row) -> None:
        with patch.object(job, "_get_service_info_spcs", return_value=row), self.assertRaises(RuntimeError):
            job._get_submitted_instance_count(MagicMock(), "jobs_DB.jobs_schema.test_id")

    def test_get_submitted_instance_count_raises_when_history_unavailable(self) -> None:
        err = sp_exceptions.SnowparkSQLException("job history unavailable", sql_error_code=2003)
        with patch.object(job, "_get_service_info_spcs", side_effect=err), self.assertRaises(RuntimeError):
            job._get_submitted_instance_count(MagicMock(), "jobs_DB.jobs_schema.test_id")

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


class DistributedResultReduceTest(parameterized.TestCase):
    """Unit tests for the distributed-result reduce (job.py module functions)."""

    @parameterized.named_parameters(  # type: ignore[misc]
        # No failure -> None.
        ("no_failure", [0], {0: {"ended_at": 1.0}}, {0: 0}, None),
        # Two failures, both with records: pick the smaller ended_at. This is the only
        # cross-instance timestamp compare; a lower instance_id must NOT win if it ended later.
        (
            "orders_by_earliest_ended_at",
            [0, 1],
            {0: {"ended_at": 5.0}, 1: {"ended_at": 2.0}},
            {0: 1, 1: 1},
            1,
        ),
        # All failed instances are lost (no record) -> deterministic lowest-ID fallback.
        (
            "all_lost_falls_back_to_lowest_id",
            [1, 0],
            {0: None, 1: None},
            {0: None, 1: None},
            0,
        ),
    )
    def test_earliest_failed_instance(
        self,
        instance_ids: list[int],
        records: dict[int, Any],
        exit_codes: dict[int, Any],
        expected: Any,
    ) -> None:
        self.assertEqual(job._earliest_failed_instance(instance_ids, records, exit_codes), expected)

    def test_reduce_all_success(self) -> None:
        records = {0: {"exit_code": 0, "ended_at": 1.0}, 1: {"exit_code": 0, "ended_at": 2.0}}
        with patch.object(job, "_read_all_records_with_retry", return_value=records), patch.object(
            job, "_load_instance0_value_or_none", return_value="v0"
        ):
            dr = job._reduce_distributed_result(MagicMock(), "@stage/r", path_transform=lambda p: p, target_instances=2)
        self.assertTrue(dr.success)
        self.assertEqual(dr.exit_codes, {0: 0, 1: 0})
        self.assertIsNone(dr.failed_instance)
        self.assertEqual(dr.return_value, "v0")

    def test_reduce_failure_and_lost(self) -> None:
        # instance 0 ok, 1 failed (has record), 2 lost (no record): covers failure + lost +
        # earliest-failed preferring the record-bearing instance over the lost one.
        records: dict[int, Any] = {
            0: {"exit_code": 0, "ended_at": 1.0},
            1: {"exit_code": 1, "ended_at": 2.0},
            2: None,
        }
        with patch.object(job, "_read_all_records_with_retry", return_value=records) as mock_read_records:
            dr = job._reduce_distributed_result(MagicMock(), "@stage/r", path_transform=lambda p: p, target_instances=3)
        self.assertFalse(dr.success)
        self.assertEqual(dr.exit_codes, {0: 0, 1: 1, 2: None})  # None = lost
        self.assertEqual(dr.failed_instance, 1)  # has a record -> ranks before the lost one
        self.assertIsNone(dr.return_value)
        self.assertEqual(mock_read_records.call_args.args[2], {0, 1, 2})

    def test_rebuild_failure_exception_malformed_record_does_not_raise(self) -> None:
        # Runs inside a `raise ... from` position, so a malformed exc dict (missing keys) must
        # degrade to a returned exception, never raise (which would replace DistributedJobError).
        with patch.object(job, "_read_instance_record", return_value={"exit_code": 1, "exc": {"message": "x"}}):
            rebuilt = job._rebuild_failure_exception(MagicMock(), "@stage/r", 0)
        self.assertIsInstance(rebuilt, BaseException)

    def test_retry_timeout_marks_lost(self) -> None:
        with patch.object(job, "_read_instance_record", return_value=None), patch(
            "snowflake.ml.jobs.job.time"
        ) as mock_time:
            # First monotonic() sets the deadline; each later call jumps far past it, so the loop
            # times out on its next check no matter how many times monotonic() is called.
            mock_time.monotonic.side_effect = itertools.count(0, 1000)
            records = job._read_all_records_with_retry(MagicMock(), "@stage/r", {0})
        self.assertEqual(records, {0: None})  # still missing after timeout -> lost


class DistributedResultApiTest(parameterized.TestCase):
    """Unit tests for MLJob.distributed_result() (result() is untouched classic behavior).

    Cover the accessor: gating, return-on-success, raise-on-failure, and the wait/count/reduce path
    (submitted-count read and the min<target warning). The reduce itself is covered by
    DistributedResultReduceTest. Tests either seed ``_distributed_result`` directly or mock the
    count/reduce, so no network is needed.
    """

    def _make_job(self, distributed: bool) -> job.MLJob[None]:
        j: job.MLJob[None] = job.MLJob(
            "db.schema.jid",
            service_spec={"spec": {"containers": [{"name": "main", "image": "img"}]}},
            session=MagicMock(),
        )
        # Pre-seed the cached_property so it isn't computed from the (mock) container spec.
        j.__dict__["_has_distributed_result"] = distributed
        return j

    def test_distributed_result_returns_object_on_success(self) -> None:
        j = self._make_job(distributed=True)
        dr = jobs.DistributedResult(success=True, exit_codes={0: 0, 1: 0}, failed_instance=None, return_value="v0")
        j._distributed_result = dr
        self.assertIs(j.distributed_result(), dr)

    def test_distributed_result_raises_distributed_job_error_on_failure(self) -> None:
        j = self._make_job(distributed=True)
        dr = jobs.DistributedResult(success=False, exit_codes={0: 0, 1: 1}, failed_instance=1, return_value=None)
        j._distributed_result = dr
        with patch.object(job.MLJob, "_result_path", new_callable=PropertyMock, return_value="@stage/r"), patch.object(
            job, "_rebuild_failure_exception", return_value=ValueError("root cause")
        ):
            with self.assertRaises(jobs.DistributedJobError) as ctx:
                j.distributed_result()
        self.assertIs(ctx.exception.result, dr)  # aggregate is carried on the raised error
        self.assertIsInstance(ctx.exception.__cause__, ValueError)  # earliest failure as cause

    def test_distributed_result_raises_on_non_distributed_job(self) -> None:
        j = self._make_job(distributed=False)
        with self.assertRaises(NotImplementedError):
            j.distributed_result()

    def test_distributed_result_warns_when_min_lt_target(self) -> None:
        j = self._make_job(distributed=True)
        j.__dict__["min_instances"] = 2
        dr = jobs.DistributedResult(success=True, exit_codes={0: 0}, failed_instance=None, return_value=None)
        with patch.object(j, "wait"), patch.object(job, "_get_submitted_instance_count", return_value=3), patch.object(
            job.MLJob, "_result_path", new_callable=PropertyMock, return_value="@stage/r"
        ), patch.object(job, "_reduce_distributed_result", return_value=dr), self.assertLogs(
            job.logger, level="WARNING"
        ) as logs:
            self.assertIs(j.distributed_result(), dr)
        self.assertTrue(any("min_instances" in line for line in logs.output))

    def test_has_distributed_result_deleted_job_no_keyerror(self) -> None:
        # A deleted job's _container_spec is {}; the gate must not KeyError on a missing "env".
        j: job.MLJob[None] = job.MLJob("db.schema.jid", session=MagicMock())
        with patch.object(job.MLJob, "_container_spec", new_callable=PropertyMock, return_value={}):
            self.assertFalse(j._has_distributed_result)


if __name__ == "__main__":
    absltest.main()
