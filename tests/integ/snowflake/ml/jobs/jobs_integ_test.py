import inspect
import itertools
import json
import os
import pathlib
import re
import tempfile
import textwrap
import time
from typing import Any, Callable, Optional, cast
from unittest import mock
from uuid import uuid4

import cloudpickle as cp
from absl.testing import absltest, parameterized
from packaging import version

from snowflake import snowpark
from snowflake.connector import errors
from snowflake.ml import jobs
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import job as jd
from snowflake.ml.jobs._utils import constants
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as sp_exceptions
from tests.integ.snowflake.ml.jobs import test_constants
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

INVALID_IDENTIFIERS = [
    "has'quote",
    "quote', 0, 'main'); drop table foo; select system$get_service_logs('job_id'",
]


def dummy_function() -> None:
    print("hello world")


@absltest.skipIf(
    (region := test_env_utils.get_current_snowflake_region()) is None
    or region["cloud"] not in test_constants._SUPPORTED_CLOUDS,
    "Test only for SPCS supported clouds",
)
class JobManagerTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)
        cls.dbm.cleanup_schemas(prefix=test_constants._TEST_SCHEMA, expire_days=1)
        cls.db = cls.session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            cls.compute_pool = cls.dbm.create_compute_pool(
                test_constants._TEST_COMPUTE_POOL, sql_client.CreationMode(if_not_exists=True), max_nodes=5
            )
        except sp_exceptions.SnowparkSQLException:
            if not cls.dbm.show_compute_pools(test_constants._TEST_COMPUTE_POOL).count() > 0:
                raise cls.failureException(
                    f"Compute pool {test_constants._TEST_COMPUTE_POOL} not available and could not be created"
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbm.drop_schema(cls.schema, if_exists=True)
        cls.session.close()
        super().tearDownClass()

    def _submit_func_as_file(self, func: Callable[[], None], **kwargs: Any) -> jobs.MLJob[None]:
        # Insert default kwargs
        default_kwargs = dict(
            compute_pool=self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
        )
        kwargs = {**default_kwargs, **kwargs}

        func_source = inspect.getsource(func)
        payload_str = textwrap.dedent(func_source) + "\n" + func.__name__ + "()\n"
        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            temp_file.write(payload_str.encode("utf-8"))
            temp_file.flush()
            job: jobs.MLJob[None] = jobs.submit_file(
                temp_file.name,
                **kwargs,
            )
            return job

    def test_async_job_parameter(self) -> None:
        try:
            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE").collect()
        except sp_exceptions.SnowparkSQLException:
            self.skipTest("Unable to toggle SPCS Async Jobs parameter. Skipping test.")

        try:
            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = FALSE").collect()
            with self.assertRaisesRegex(RuntimeError, "ENABLE_SNOWSERVICES_ASYNC_JOBS"):
                self._submit_func_as_file(dummy_function)

            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE").collect()
            job = self._submit_func_as_file(dummy_function)
            self.assertIsNotNone(job)
        finally:
            # Make sure we re-enable the parameter even if this test fails
            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE").collect()

    def test_list_jobs(self) -> None:
        # Use a separate schema for this test
        original_schema = self.session.get_current_schema()
        temp_schema = self.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            # Should be empty initially
            self.assertEmpty(jobs.list_jobs(session=self.session))

            # Submit a job
            job = self._submit_func_as_file(dummy_function)

            # Validate list jobs output
            jobs_df = jobs.list_jobs(session=self.session)
            self.assertEqual(1, jobs_df.shape[0])
            self.assertSequenceEqual(
                [
                    "name",
                    "status",
                    "message",
                    "database_name",
                    "schema_name",
                    "owner",
                    "compute_pool",
                    "target_instances",
                    "created_time",
                    "completed_time",
                ],
                list(jobs_df.columns),
            )
            self.assertEqual(job.name, jobs_df.iloc[0]["name"])

            # Loading job ID shouldn't generate any additional jobs in backend
            loaded_job = jobs.get_job(job.id, session=self.session)
            loaded_job.status  # Trigger status check
            self.assertEqual(1, len(jobs.list_jobs(session=self.session)))

            # Test different scopes
            scopes = [
                (None, None),
                (self.db, None),
                (None, temp_schema),
                (self.db, temp_schema),
            ]
            for database, schema in scopes:
                with self.subTest(f"scope={database}.{schema}"):
                    self.assertGreater(len(jobs.list_jobs(database=database, schema=schema, session=self.session)), 0)

            # Submit a second job to test different limits
            job2 = self._submit_func_as_file(dummy_function)
            limits = [1, 2, 5, 10]
            for limit in limits:
                with self.subTest(f"limit={limit}"):
                    self.assertBetween(
                        len(jobs.list_jobs(limit=limit, schema=temp_schema, session=self.session)),
                        min(limit, 2),
                        limit,
                    )
            self.assertEqual(2, len(jobs.list_jobs(limit=0, schema=temp_schema, session=self.session)))
            self.assertEqual(2, len(jobs.list_jobs(limit=-1, schema=temp_schema, session=self.session)))
            self.assertEqual(2, len(jobs.list_jobs(limit=-10, schema=temp_schema, session=self.session)))

            # Delete the job
            jobs.delete_job(job.id, session=self.session)
            jobs.delete_job(job2.id, session=self.session)
            self.assertEqual(0, len(jobs.list_jobs(session=self.session)))
        finally:
            self.dbm.drop_schema(temp_schema, if_exists=True)
            self.session.use_schema(original_schema)

    @parameterized.parameters(  # type: ignore[misc]
        {"database": '"not_exist_db"'},
        {"schema": '"not_exist_schema"'},
        {"database": '"not_exist_db"', "schema": '"not_exist_schema"'},
    )
    def test_list_jobs_negative(self, **kwargs: Any) -> None:
        enable_job_history_spcs = False
        row = self.session.sql("show parameters like 'SNOWSERVICES_ENABLE_SPCS_JOB_HISTORY' in account;").collect()
        if row and row[0]["value"] == "true":
            row = self.session.sql("show parameters like 'ENABLE_SPCS_SCHEMA_IN_SNOWFLAKE_SHARE';").collect()
            if row and row[0]["value"] == "true":
                enable_job_history_spcs = True

        if enable_job_history_spcs:
            self.assertEmpty(jobs.list_jobs(**kwargs, session=self.session))

        else:
            with self.assertRaises(sp_exceptions.SnowparkSQLException):
                jobs.list_jobs(**kwargs, session=self.session)

    def test_get_job_positive(self):
        # Submit a job
        job = self._submit_func_as_file(dummy_function)

        test_cases = [
            job.name,
            f"{self.schema}.{job.name}",
            f"{self.db}.{self.schema}.{job.name}",
        ]
        for id in test_cases:
            with self.subTest(f"id={id}"):
                load_job = jobs.get_job(id, session=self.session)
                self.assertIsNotNone(load_job)
                job_db, job_schema, _ = identifier.parse_schema_level_object_identifier(load_job.id)
                self.assertEqual(job_db, identifier.resolve_identifier(self.db))
                self.assertEqual(job_schema, identifier.resolve_identifier(self.schema))

    def test_get_job_negative(self) -> None:
        # Invalid job ID (not a valid identifier)
        for id in INVALID_IDENTIFIERS:
            with self.assertRaisesRegex(ValueError, "Invalid job ID", msg=f"id={id}"):
                jobs.get_job(id, session=self.session)

        nonexistent_job_ids = [
            f"{self.db}.non_existent_schema.nonexistent_job_id",
            f"{self.db}.{self.schema}.nonexistent_job_id",
            "nonexistent_job_id",
        ]
        for id in nonexistent_job_ids:
            with self.subTest(f"id={id}"):
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    jobs.get_job(id, session=self.session)

    def test_delete_job_negative(self) -> None:
        nonexistent_job_ids = [
            f"{self.db}.non_existent_schema.nonexistent_job_id",
            f"{self.db}.{self.schema}.nonexistent_job_id",
            "nonexistent_job_id",
            *INVALID_IDENTIFIERS,
        ]
        for id in nonexistent_job_ids:
            with self.subTest(f"id={id}"):
                job = jobs.MLJob[None](id, session=self.session)
                with self.assertRaises(ValueError, msg=f"id={id}"):
                    jobs.delete_job(job.id, session=self.session)
                with self.assertRaises(errors.ProgrammingError, msg=f"id={id}"):
                    jobs.delete_job(job, session=self.session)

    def test_get_status_negative(self) -> None:
        nonexistent_job_ids = [
            f"{self.db}.non_existent_schema.nonexistent_job_id",
            f"{self.db}.{self.schema}.nonexistent_job_id",
            "nonexistent_job_id",
            *INVALID_IDENTIFIERS,
        ]
        for id in nonexistent_job_ids:
            with self.subTest(f"id={id}"):
                job = jobs.MLJob[None](id, session=self.session)
                with self.assertRaises(errors.ProgrammingError, msg=f"id={id}"):
                    job.status

    def test_get_logs(self) -> None:
        # Submit a job
        job = self._submit_func_as_file(dummy_function)
        job.wait()
        self.assertIsInstance(job.get_logs(), str)
        self.assertIsInstance(job.get_logs(as_list=True), list)

        # Validate full job logs
        self.assertIn("hello world", job.get_logs(verbose=True))
        self.assertIn("ray", job.get_logs(verbose=True))
        self.assertIn(constants.LOG_START_MSG, job.get_logs(verbose=True))
        self.assertIn(constants.LOG_END_MSG, job.get_logs(verbose=True))

        # Check job for non-verbose mode
        self.assertIn("hello world", job.get_logs(verbose=False))
        self.assertNotIn("ray", job.get_logs(verbose=False))
        self.assertNotIn(constants.LOG_START_MSG, job.get_logs(verbose=False))
        self.assertNotIn(constants.LOG_END_MSG, job.get_logs(verbose=False))

    def test_job_wait(self) -> None:
        # Status check adds some latency
        max_backoff = constants.JOB_POLL_MAX_DELAY_SECONDS
        try:
            # Speed up polling for testing
            constants.JOB_POLL_MAX_DELAY_SECONDS = 0.1  # type: ignore[assignment]
            fudge_factor = 0.5

            # Create a dummy job
            job = jobs.MLJob[None]("dummy_job_id", session=self.session)
            with mock.patch("snowflake.ml.jobs.job._get_status", return_value="RUNNING") as mock_get_status:
                # Test waiting with timeout=0
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    job.wait(timeout=0)
                self.assertLess(time.monotonic() - start, fudge_factor)
                mock_get_status.assert_called_once()

                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    job.wait(timeout=1)
                self.assertBetween(time.monotonic() - start, 1, 1 + fudge_factor)

            with mock.patch("snowflake.ml.jobs.job._get_status", return_value="DONE") as mock_get_status:
                # Test waiting on a completed job with different timeouts
                start = time.monotonic()
                self.assertEqual(job.wait(timeout=0), "DONE")
                self.assertEqual(job.wait(timeout=-10), "DONE")
                self.assertEqual(job.wait(timeout=+10), "DONE")
                self.assertLess(time.monotonic() - start, fudge_factor)
        finally:
            constants.JOB_POLL_MAX_DELAY_SECONDS = max_backoff

    def test_job_execution(self) -> None:
        payload = TestAsset("src/main.py")

        # Create a job
        job = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo", "--delay", "1"],
            session=self.session,
        )

        # Check job while it's running
        self.assertIn(job.status, {"PENDING", "RUNNING"})
        # FIXME: SPCS PENDING -> RUNNING transition is very slow, often
        #        still shows as PENDING even after the job has started.
        #        At time of writing, observed latency was about 8 seconds.
        # while job.status == "PENDING":
        #     time.sleep(0.5)
        # self.assertEqual(job.status, "RUNNING")
        # self.assertIn("Job start", job.get_logs())

        # Wait for job to finish
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        self.assertEqual(job.status, "DONE")
        self.assertIn("Job complete", job.get_logs())
        self.assertIsNone(job.result())

        # Test loading job by ID
        loaded_job = jobs.get_job(job.id, session=self.session)
        self.assertEqual(loaded_job.status, "DONE")
        self.assertIn("Job start", loaded_job.get_logs())
        self.assertIn("Job complete", loaded_job.get_logs())
        self.assertIsNone(loaded_job.result())

    def test_job_execution_metrics(self) -> None:
        payload = TestAsset("src/main.py")

        # Create a job with metrics disabled
        job1 = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo", "--delay", "10"],
            enable_metrics=False,
            session=self.session,
        )

        # Create a job with metrics enabled
        job2 = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo", "--delay", "10"],
            enable_metrics=True,
            session=self.session,
        )

        # Wait for job to finish
        job1.wait()
        job2.wait()
        self.assertEqual(job1.status, "DONE")
        self.assertEqual(job2.status, "DONE")

        # Skip event table validation to avoid unpredictably slow Event Table latency
        validate_metrics_events = False
        if validate_metrics_events:
            # Retrieve event table name
            event_table_name: str = self.session.sql(
                f"SHOW PARAMETERS LIKE 'event_table' IN DATABASE {self.session.get_current_database()}"
            ).collect()[0]["value"]
            if event_table_name.lower() == "snowflake.telemetry.events":
                # Use EVENTS_VIEW if using default event table
                event_table_name = "SNOWFLAKE.TELEMETRY.EVENTS_VIEW"

            # Retrieve and validate metrics for the job
            # Use a retry loop to wait for the metrics to be published since Event Table has some delay
            query = textwrap.dedent(
                f"""
                SELECT * FROM {event_table_name}
                    WHERE TIMESTAMP > DATEADD('minute', -5, CURRENT_TIMESTAMP())
                        AND RESOURCE_ATTRIBUTES:"snow.service.name" = '{{job_id}}'
                        AND RECORD_TYPE = 'METRIC'
                """
            )
            max_delay = 60  # Max delay in seconds
            retry_interval = 1  # Retry delay in seconds
            num_tries = ((max_delay - 1) // retry_interval) + 1
            for _ in range(num_tries):
                if self.session.sql(query.format(job_id=job2.id)).count() > 0:
                    break
                time.sleep(retry_interval)
            self.assertEqual(self.session.sql(query.format(job_id=job1.id)).count(), 0, job1.id)
            self.assertGreater(self.session.sql(query.format(job_id=job2.id)).count(), 0, job2.id)

    def test_job_pickling(self) -> None:
        """Dedicated test for MLJob pickling and unpickling functionality."""
        payload = TestAsset("src/main.py")

        # Create a job and wait for completion
        job = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo", "--delay", "1"],
            session=self.session,
        )

        self.assertEqual(job.wait(), "DONE", job.get_logs())

        # Get initial state for comparison
        original_id = job.id
        original_status = job.status
        original_logs = job.get_logs()
        original_result = job.result()
        original_target_instances = job.target_instances

        pickled_data = cp.dumps(job)
        self.assertIsInstance(pickled_data, bytes)
        self.assertGreater(len(pickled_data), 0)
        unpickled_job: jobs.MLJob[None] = cp.loads(pickled_data)
        self.assertIsInstance(unpickled_job, jobs.MLJob)

        # Verify session validation - should get same session back
        self.assertIs(unpickled_job._session, self.session)

        # Verify job identity and basic properties are preserved
        self.assertEqual(unpickled_job.id, original_id)
        self.assertEqual(unpickled_job.name, job.name)

        # Verify all MLJob functionality works on unpickled object
        self.assertEqual(unpickled_job.status, original_status)
        self.assertEqual(unpickled_job.target_instances, original_target_instances)
        self.assertEqual(unpickled_job.get_logs(), original_logs)
        self.assertEqual(unpickled_job.result(), original_result)

        # Test that unpickled job can handle multiple operations
        self.assertIn("Job start", unpickled_job.get_logs())
        self.assertIn("Job complete", unpickled_job.get_logs())

        # Verify cached properties work correctly
        _ = unpickled_job.min_instances  # Should not raise an error

        # Test that session-dependent operations work
        self.assertIsNotNone(unpickled_job._compute_pool)

        # Verify the job can be pickled again (round-trip test)
        second_pickle = cp.dumps(unpickled_job)
        second_unpickled = cp.loads(second_pickle)
        self.assertEqual(second_unpickled.id, original_id)
        self.assertEqual(second_unpickled.status, original_status)

        # Test session validation - should fail with different session context
        with mock.patch.object(self.session, "get_current_account", return_value="DIFFERENT_ACCOUNT"):
            with self.assertRaisesRegex(RuntimeError, "No active Snowpark session available"):
                cp.loads(pickled_data)

    # TODO(SNOW-1911482): Enable test for Python 3.11+
    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )  # type: ignore[misc]
    def test_job_decorator(self) -> None:
        @jobs.remote(self.compute_pool, stage_name="@payload_stage/subdir", session=self.session)
        def decojob_fn(arg1: str, arg2: int, arg3: Optional[Any] = None) -> dict[str, Any]:
            from datetime import datetime

            print(f"{datetime.now()}\t[{arg1}, {arg2}+1={arg2+1}, arg3={arg3}] Job complete", flush=True)
            return {"arg1": arg1, "arg2": arg2, "result": 100}

        class MyDataClass:
            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

            def __str__(self) -> str:
                return f"MyDataClass({self.x}, {self.y})"

        # Define parameter combinations to test
        params: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            (("Positional Arg", 5), {}),
            (("Positional Arg",), {"arg2": 5}),
            (tuple(), {"arg1": "Named Arg", "arg2": 5}),
            (("Positional Arg", 5, {"key": "value"}), {}),
            (("Positional Arg", 5, MyDataClass(1, 2)), {}),
            (("Positional Arg", 5), {"arg3": MyDataClass(1, 2)}),
        ]

        # Kick off jobs in parallel
        job_list: list[jobs.MLJob[Any]] = []
        for i in range(len(params)):
            args, kwargs = params[i]
            job = decojob_fn(*args, **kwargs)

            # Check job while it's running
            self.assertIn(job.status, {"PENDING", "RUNNING"})
            job_list.append(job)

        # Check job completions
        for param, job in zip(params, job_list):
            with self.subTest(param):
                self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs())
                self.assertIn("Job complete", job_logs)

                args, kwargs = param
                for arg in args:
                    self.assertIn(str(arg), job_logs)
                for k, v in kwargs.items():
                    self.assertIn(str(v), job_logs, f"key={k}")

                job_result = cast(dict[str, Any], job.result())
                self.assertIsInstance(job.result(), dict)
                self.assertEqual(job_result.get("result"), 100)

                loaded_job = jobs.get_job(job.id, session=self.session)
                self.assertEqual(loaded_job.status, "DONE")
                self.assertDictEqual(loaded_job.result(), job_result)

    # TODO(SNOW-1911482): Enable test for Python 3.11+
    @absltest.skipIf(  # type: ignore[misc]
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_job_decorator_negative_result(self) -> None:
        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def func_no_return() -> None:
            pass

        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def func_with_error() -> None:
            raise NotImplementedError("This function is expected to fail")

        # Run jobs in parallel for speed
        job1 = func_no_return()
        job2 = func_with_error()

        # Job 1 should succeed but return None
        self.assertEqual(job1.wait(), "DONE", job1.get_logs())
        self.assertIsNone(job1.result())

        # Should be able to retrieve Job 1's exception by job ID
        job1_loaded = jobs.get_job(job1.id, session=self.session)
        self.assertIsNone(job1_loaded.result())

        # Job 2 should fail
        self.assertEqual(job2.wait(), "FAILED", job2.get_logs())
        with self.assertRaisesRegex(RuntimeError, "Job execution failed") as job2_cm:
            job2.result()
        self.assertIsNotNone(getattr(job2_cm.exception, "__cause__", None))
        self.assertIsInstance(job2_cm.exception.__cause__, NotImplementedError)
        self.assertEqual(str(job2_cm.exception.__cause__), "This function is expected to fail")

        # Should be able to retrieve Job 2's exception by job ID
        job2_loaded = jobs.get_job(job2.id, session=self.session)
        with self.assertRaisesRegex(RuntimeError, "Job execution failed") as job2_loaded_cm:
            job2_loaded.result()
        self.assertIsNotNone(getattr(job2_loaded_cm.exception, "__cause__", None))
        self.assertIsInstance(job2_loaded_cm.exception.__cause__, NotImplementedError)
        self.assertEqual(str(job2_loaded_cm.exception.__cause__), "This function is expected to fail")

    def test_job_runtime_api(self) -> None:
        # Submit this function via file to avoid pickling issues
        # TODO: Test this via job decorator as well
        def runtime_func() -> None:
            from common_utils import common_util as mlrs_util

            from snowflake.ml.data.data_connector import DataConnector
            from snowflake.snowpark.context import get_active_session

            # Validate simple data ingestion
            session = get_active_session()
            num_rows = 100
            df = session.sql(
                f"SELECT uniform(1, 1000, random()) as random_val FROM table(generator(rowcount => {num_rows}))"
            )
            dc = DataConnector.from_dataframe(df)
            assert "Ray" in type(dc._ingestor).__name__, type(dc._ingestor).__qualname__
            assert len(dc.to_pandas()) == num_rows, len(dc.to_pandas())

            # Validate mlruntimes utils
            assert mlrs_util.get_num_ray_nodes() > 0

            # Print success message which will be checked in the test
            print("Runtime API test success")

        job = self._submit_func_as_file(runtime_func)
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs())
        self.assertIn("Runtime API test success", job_logs)

        # TODO: Add test for DataConnector serialization/deserialization

    def test_multinode_job_basic(self) -> None:
        job_from_file = self._submit_func_as_file(dummy_function, target_instances=2)

        self.assertEqual(job_from_file.target_instances, 2)
        self.assertEqual(job_from_file.min_instances, 2)  # min_instances defaults to target_instances
        self.assertEqual(job_from_file.wait(), "DONE", file_job_logs := job_from_file.get_logs())
        self.assertIn("hello world", file_job_logs)

        @jobs.remote(self.compute_pool, stage_name="payload_stage", target_instances=2, session=self.session)
        def dummy_remote() -> None:
            print("hello world")

        # TODO(SNOW-1911482): Enable test for Python 3.11+
        if version.Version(env.PYTHON_VERSION) < version.Version("3.11"):
            job_from_func = dummy_remote()
            self.assertEqual(job_from_func.wait(), "DONE", file_job_logs := job_from_func.get_logs())
            self.assertIn("hello world", file_job_logs)

    def test_multinode_job_ray_task(self) -> None:
        def ray_workload() -> int:
            import socket

            import ray

            @ray.remote(scheduling_strategy="SPREAD")
            def compute_heavy(n):
                # a quick CPU‐bound toy workload
                a, b = 0, 1
                for _ in range(n):
                    a, b = b, a + b
                # report which node we ran on
                return socket.gethostname()

            ray.init(address="auto", ignore_reinit_error=True)
            hosts = [compute_heavy.remote(50_000) for _ in range(10)]
            unique_hosts = set(ray.get(hosts))
            assert len(unique_hosts) >= 2, f"Expected at least 2 unique hosts, get: {unique_hosts}"
            print("test succeeded")

        job = self._submit_func_as_file(ray_workload, target_instances=2, min_instances=2)
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        self.assertTrue("test succeeded" in job.get_logs())

    def test_multinode_job_wait_for_min_instances(self) -> None:
        def get_cluster_size() -> None:
            from common_utils import common_util as mlrs_util

            print("num_nodes:", mlrs_util.get_num_ray_nodes())

        job = self._submit_func_as_file(get_cluster_size, target_instances=2, min_instances=2)
        self.assertEqual(job.target_instances, 2)
        self.assertEqual(job.min_instances, 2)
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        self.assertEqual(re.match(r"num_nodes: (\d+)", job.get_logs(verbose=False)).group(1), "2")

        # Check verbose log to ensure min_instances was checked
        self.assertTrue("Minimum instance requirement met: 2 instances available" in job.get_logs(verbose=True))

    def test_min_instances_exceeding_max_nodes(self) -> None:
        compute_pool_info = self.dbm.show_compute_pools(self.compute_pool).collect()
        self.assertTrue(compute_pool_info, f"Could not find compute pool {self.compute_pool}")
        max_nodes = int(compute_pool_info[0]["max_nodes"])

        # Calculate a min_instances value that exceeds max_nodes
        min_instances = max_nodes + 1
        # Set target_instances to be greater than min_instances to pass the first validation
        target_instances = min_instances + 1

        # Attempt to submit a job with min_instances exceeding max_nodes
        with self.assertRaisesRegex(ValueError, "min_instances .* exceeds the max_nodes"):
            self._submit_func_as_file(
                dummy_function,
                min_instances=min_instances,
                target_instances=target_instances,
            )

    def test_job_with_pip_requirements(self) -> None:
        rows = self.session.sql("SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI%'").collect()
        if not rows:
            self.fail("No PyPI EAI found in environment.")
        pypi_eais = [r["name"] for r in rows]

        payload = TestAsset("src/main.py")

        # Create a job with invalid requirements
        job_bad_requirement = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo"],
            pip_requirements=["nonexistent_package"],
            external_access_integrations=pypi_eais,
            session=self.session,
        )

        # Create a job with valid requirements but no EAI
        job_no_eai = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo"],
            pip_requirements=["tabulate"],
            external_access_integrations=None,
            session=self.session,
        )

        # Create a job with valid requirements and EAI for PyPI access
        job = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            args=["foo"],
            pip_requirements=["tabulate"],
            external_access_integrations=pypi_eais,
            session=self.session,
        )

        # Create a job with a requirement that conflicts with runtime env
        job_dep_conflict = jobs.submit_file(
            TestAsset("src/check_numpy.py").path,
            self.compute_pool,
            stage_name="payload_stage",
            pip_requirements=["numpy==1.23"],
            external_access_integrations=pypi_eais,
            session=self.session,
        )

        # Job with bad requirement should fail due to installation error
        self.assertEqual(job_bad_requirement.wait(), "FAILED")
        self.assertIn(
            "No matching distribution found for nonexistent_package", job_bad_requirement.get_logs(verbose=True)
        )

        # Job with no EAI should fail due to network access error
        self.assertEqual(job_no_eai.wait(), "FAILED")
        self.assertIn("No matching distribution found for tabulate", job_no_eai.get_logs(verbose=True))

        # Job with valid requirements and EAI should succeed
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs(verbose=True))
        self.assertIn("Successfully installed tabulate", job_logs)
        self.assertIn("[foo] Job complete", job_logs)

        # Job with conflicting requirement should prefer the user specified package
        self.assertEqual(
            job_dep_conflict.wait(), "DONE", job_dep_conflict_logs := job_dep_conflict.get_logs(verbose=True)
        )
        self.assertRegex(job_dep_conflict_logs, r"you have numpy 1\.23\.\d+ which is incompatible")
        self.assertIn("Numpy version: 1.23", job_dep_conflict_logs)

    @parameterized.parameters(  # type: ignore[misc]
        "cloudpickle~=2.0",
        "cloudpickle~=3.0",
    )
    def test_job_cached_packages(self, package: str) -> None:
        """Test that cached packages are available without requiring network access."""

        job = self._submit_func_as_file(dummy_function, pip_requirements=[package])
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs(verbose=True))
        self.assertTrue(
            package in job_logs or "Successfully installed" in job_logs,
            "Didn't find expected package log:" + job_logs,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {
            "env_vars": {
                "ENV_VAR": "VALUE1",
            },
            "expected_string": "VALUE1",
        },
        {
            "spec_overrides": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "env": {
                                "ENV_VAR": "OVERRIDE_VALUE",
                            },
                        }
                    ]
                },
                "capabilities": {"securityContext": {"executeAsCaller": True}},
            },
            "expected_string": "OVERRIDE_VALUE",
        },
    )
    def test_job_with_spec_overrides(self, *, expected_string: str, **kwargs: Any) -> None:
        # Create a job
        payload = TestAsset("src/")
        job = jobs.submit_directory(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            entrypoint="secondary.py",
            session=self.session,
            **kwargs,
        )

        # Wait for job to finish
        job.wait()
        self.assertEqual(job.status, "DONE", job.get_logs())
        self.assertIn(expected_string, job.get_logs())

    def test_submit_job_fully_qualified_name(self):
        temp_schema = self.dbm.create_random_schema(prefix=f"{test_constants._TEST_SCHEMA}_EXT")
        self.dbm.use_schema(self.schema)  # Stay on default schema

        test_cases = [
            (None, None),
            (None, self.schema),
            (None, temp_schema),
            (self.db, self.schema),
            (self.db, temp_schema),
        ]
        try:
            for database, schema in test_cases:
                with self.subTest(database=database, schema=schema):
                    job = self._submit_func_as_file(
                        dummy_function,
                        database=database,
                        schema=schema,
                    )
                    job_database, job_schema, job_name = identifier.parse_schema_level_object_identifier(job.id)

                    # job_database/job_schema should not be wrapped in quotes unless absolutely necessary
                    self.assertEqual(job_database, identifier.resolve_identifier(self.db))
                    self.assertEqual(job_schema, identifier.resolve_identifier(schema or self.schema))

                    if schema == temp_schema:
                        with self.assertRaisesRegex(ValueError, "does not exist"):
                            jobs.get_job(job_name, session=self.session)
                    else:
                        self.assertIsNotNone(jobs.get_job(job_name, session=self.session))
        finally:
            self.dbm.drop_schema(temp_schema, if_exists=True)

    def test_submit_job_negative(self):
        test_cases = [
            ("not_valid_database", self.schema, errors.ProgrammingError, "does not exist"),
            (self.db, "not_valid_schema", errors.ProgrammingError, "does not exist"),
            (self.db, None, ValueError, "Schema must be specified if database is specified."),
        ]
        for database, schema, expected_exception, expected_regex in test_cases:
            with self.subTest(database=database, schema=schema):
                with self.assertRaisesRegex(expected_exception, expected_regex):
                    _ = self._submit_func_as_file(
                        dummy_function,
                        database=database,
                        schema=schema,
                    )

        kwargs = {
            "compute_pool": self.compute_pool,
            "stage_name": "payload_stage",
            "query_warehouse": self.session.get_current_warehouse(),
            "database": self.db,
            "schema": self.schema,
        }
        for k, v in itertools.product(kwargs.keys(), INVALID_IDENTIFIERS):
            with self.subTest(f"{k}={v}"):
                invalid_kwargs = kwargs.copy()
                invalid_kwargs[k] = v
                with self.assertRaisesRegex(
                    (ValueError, errors.ProgrammingError), re.compile(re.escape(v), re.IGNORECASE)
                ):
                    _ = self._submit_func_as_file(
                        dummy_function,
                        **invalid_kwargs,
                    )

    @absltest.skipIf(  # type: ignore[misc]
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_remote_with_session_positive(self):
        @jobs.remote(self.compute_pool, stage_name="@payload_stage", session=self.session)
        def test_session_as_first_positional(arg1: snowpark.Session, arg2: str, arg3: str):
            print(f"database: {arg1.get_current_database()}")
            print(f"hello {arg2}, {arg3}")

        test_cases = [
            (
                "test_session_as_first_positional('test1', 'test2')",
                test_session_as_first_positional(self.session, "test1", "test2"),
                True,
            ),
        ]
        for test_case, func, hasSession in test_cases:
            with self.subTest(f"func={test_case}"):
                job = func
                self.assertEqual(job.wait(), "DONE", job.get_logs())
                if hasSession:
                    self.assertIn(self.session.get_current_database(), job.get_logs())

    @parameterized.parameters(  # type: ignore[misc]
        (f"TMP_{uuid4().hex.upper()}", True, "SNOWFLAKE_FULL"),
        (f"TEST_{uuid4().hex.upper()}", False, "SNOWFLAKE_SSE"),
    )
    def test_submit_job_from_stage(
        self,
        stage_name: str,
        temporary: bool,
        encryption: str,
    ):
        """
        currently there are no commands supporting copy files from or to user stage(@~)
        only cover these two cases
        1. temporary stage
        2. session stage

        """
        self.session.sql(
            f"CREATE {'TEMPORARY' if temporary else ''} STAGE {stage_name} ENCRYPTION = (TYPE = {repr(encryption)});"
        ).collect()
        upload_files = TestAsset("src")
        for path in {
            p.parent.joinpath(f"*{p.suffix}") if p.suffix else p
            for p in upload_files.path.resolve().rglob("*")
            if p.is_file()
        }:
            self.session.file.put(
                str(path),
                pathlib.Path(stage_name).joinpath(path.parent.relative_to(upload_files.path)).as_posix(),
                overwrite=True,
                auto_compress=False,
            )

        test_cases = [
            (f"@{stage_name}/", f"@{stage_name}/subdir/sub_main.py"),
            (f"@{stage_name}/subdir", f"@{stage_name}/subdir/sub_main.py"),
        ]
        for source, entrypoint in test_cases:
            with self.subTest(source=source, entrypoint=entrypoint):
                job = jobs.submit_from_stage(
                    source=source,
                    entrypoint=entrypoint,
                    compute_pool=self.compute_pool,
                    stage_name="payload_stage",
                    args=["foo", "--delay", "1"],
                    session=self.session,
                )

                self.assertEqual(job.wait(), "DONE", job.get_logs())

    @parameterized.parameters(
        [
            {"method": "job_object"},
            {"method": "job_id"},
        ]
    )
    def test_delete_job_stage_cleanup(self, method: str) -> None:
        """Test that deleting a job cleans up the stage files."""
        original_schema = self.session.get_current_schema()
        temp_schema = self.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            job = self._submit_func_as_file(dummy_function)
            self.assertEqual(1, len(jobs.list_jobs(session=self.session)))
            stage_path = job._stage_path
            stage_files = self.session.sql(f"LIST '{stage_path}'").collect()
            self.assertGreater(len(stage_files), 0, "Stage should contain uploaded job files")

            # Verify we can find expected files (startup.sh, requirements.txt, etc.)
            file_names = {row["name"].split("/")[-1] for row in stage_files}
            expected_files = {"startup.sh", "mljob_launcher.py"}
            for expected_file in expected_files:
                self.assertIn(expected_file, file_names, f"Expected file {expected_file} not found in stage")

            jobs.delete_job(job.id if method == "job_id" else job, session=self.session)
            remaining_files = self.session.sql(f"LIST '{stage_path}'").collect()
            self.assertEqual(len(remaining_files), 0, "Stage files should be cleaned up after job deletion")
            self.assertEqual(0, len(jobs.list_jobs(session=self.session)))
        finally:
            self.dbm.drop_schema(temp_schema, if_exists=True)
            self.session.use_schema(original_schema)

    def test_get_logs_fallback(self) -> None:
        real_run_query = self.session._conn.run_query

        def sql_side_effect(query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
                raise errors.ProgrammingError("unable to get logs")
            return real_run_query(query_str, *args, **kwargs)

        try:
            self.session.sql("ALTER SESSION SET ENABLE_SPCS_NESTED_FUNCTIONS = False").collect()
        except sp_exceptions.SnowparkSQLException:
            self.skipTest("Unable to disable SPCS persistent logs parameter. Skipping test.")
        job_event_table = self._submit_func_as_file(dummy_function)

        try:
            self.session.sql("ALTER SESSION SET ENABLE_SPCS_NESTED_FUNCTIONS = TRUE").collect()
        except sp_exceptions.SnowparkSQLException:
            self.skipTest("Unable to control the SPCS persistent logs parameter. Skipping test.")

        # creat two separate jobs with different function service, one is enabled SPCS persistent logs, the other is not
        job_spcs = self._submit_func_as_file(dummy_function)

        self.assertEqual(job_event_table.wait(), "DONE", job_event_table.get_logs())
        self.assertEqual(job_spcs.wait(), "DONE", job_event_table.get_logs())
        # Wait for event table ingest to complete
        # check if the event table ingest is complete for job_event_table
        max_wait = 300
        interval = 30
        elapsed = 0
        database, schema, id = identifier.parse_schema_level_object_identifier(job_event_table.id)
        event_table_logs = []
        while elapsed <= max_wait:
            event_table_logs = jd._get_service_log_from_event_table(
                self.session, id, database=database, schema=schema, instance_id=0
            )
            if len(event_table_logs) > 0:
                break
            time.sleep(interval)
            elapsed += interval
        else:
            raise TimeoutError("Event table ingest did not complete in 5 minutes")

        spcs_logs = []
        # check if the event table ingest is complete for job_spcs
        while elapsed <= max_wait:
            spcs_logs = jd._get_logs_spcs(self.session, job_spcs.id, instance_id=0, container_name="main")
            if len(spcs_logs) > 0:
                break
            time.sleep(interval)
            elapsed += interval
        else:
            raise TimeoutError("Event table ingest did not complete in 5 minutes")

        with mock.patch.object(self.session._conn, "run_query", side_effect=sql_side_effect):
            # check the fallback logic
            # fallback to event table if SPCS logs are not available
            with self.assertLogs(level="DEBUG") as cm:
                original_logs = job_event_table.get_logs(verbose=True)
                self.assertEqual(original_logs, os.linesep.join(json.loads(row[0]) for row in event_table_logs))
            self.assertTrue(any("falling back to event table" in line for line in cm.output))
            # check the SPCS logs
            with self.assertLogs(level="DEBUG") as cm:
                original_logs = job_spcs.get_logs(verbose=True)
                self.assertEqual(original_logs, os.linesep.join(row[0] for row in spcs_logs))
            self.assertFalse(any("falling back to event table" in line for line in cm.output))

    def test_file_indentation_tabs(self) -> None:
        payload = TestAsset("src/test_tabs_indentation.py")

        # Create a job
        job = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
        )
        self.assertEqual(job.wait(), "DONE", job.get_logs())

    def test_cancel_job(self) -> None:
        """Test cancelling a long running job."""
        try:
            self.session.sql("ALTER SESSION SET SNOWSERVICES_ENABLE_SPCS_JOB_CANCELLATION = TRUE").collect()
            self.session.sql("ALTER SESSION SET ENABLE_ENTITY_FACADE_SYSTEM_FUNCTIONS = TRUE").collect()
        except sp_exceptions.SnowparkSQLException:
            self.skipTest("Unable to control the SPCS job cancellation parameter. Skipping test.")

        def long_running_function() -> None:
            import time

            time.sleep(300)

        job = self._submit_func_as_file(long_running_function)
        self.assertIn(job.status, ["PENDING", "RUNNING"])
        job.cancel()
        final_status = job.wait(timeout=20)
        self.assertEqual(final_status, "CANCELLED")

    def test_cancel_nonexistent_job(self) -> None:
        """Test cancelling a job that doesn't exist."""
        try:
            self.session.sql("ALTER SESSION SET SNOWSERVICES_ENABLE_SPCS_JOB_CANCELLATION = TRUE").collect()
            self.session.sql("ALTER SESSION SET ENABLE_ENTITY_FACADE_SYSTEM_FUNCTIONS = TRUE").collect()
        except sp_exceptions.SnowparkSQLException:
            self.skipTest("Unable to control the SPCS job cancellation parameter. Skipping test.")

        nonexistent_job_ids = [
            f"{self.db}.non_existent_schema.nonexistent_job_id",
            f"{self.db}.{self.schema}.NONEXISTENT_JOB_ID",
            "nonexistent_job_id",
            *INVALID_IDENTIFIERS,
        ]
        for id in nonexistent_job_ids:
            with self.subTest(f"id={id}"):
                job = jobs.MLJob[None](id, session=self.session)
                with self.assertRaises(RuntimeError, msg=f"id={id}"):
                    job.cancel()


if __name__ == "__main__":
    absltest.main()
