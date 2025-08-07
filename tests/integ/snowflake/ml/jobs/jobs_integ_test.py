import inspect
import itertools
import os
import pathlib
import re
import sys
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
from snowflake.ml import jobs
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import job as jd
from snowflake.ml.jobs._utils import (
    constants,
    payload_utils,
    query_helper,
    spec_utils,
    types,
)
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as sp_exceptions, functions as F
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
        payload_str = textwrap.dedent(func_source) + "\n\n__return__ = " + func.__name__ + "()\n"
        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            temp_file.write(payload_str.encode("utf-8"))
            temp_file.flush()
            job: jobs.MLJob[None] = jobs.submit_file(
                temp_file.name,
                **kwargs,
            )
            return job

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
        self.assertEmpty(jobs.list_jobs(**kwargs, session=self.session))

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
                with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
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
                with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
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

    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )  # type: ignore[misc]
    def test_job_pickling(self) -> None:
        """Dedicated test for MLJob pickling and unpickling functionality."""
        payload = TestAsset("src/main.py")

        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def check_job_status(job: jobs.MLJob[Any]) -> str:
            return job.status

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

        # Test that we can pickle jobs into remote functions
        job_status_remote = check_job_status(job)
        self.assertEqual(job.wait(), job_status_remote.result(), job_status_remote.get_logs())

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
        with mock.patch("snowflake.snowpark.session._get_active_sessions", return_value=set()):
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
    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )  # type: ignore[misc]
    def test_job_execution_in_stored_procedure(self) -> None:
        jobs_import_src = os.path.dirname(jobs.__file__)

        @jobs.remote(self.compute_pool, stage_name="payload_stage")
        def job_fn() -> None:
            print("Hello from remote function!")

        @F.sproc(
            session=self.session,
            packages=["snowflake-snowpark-python", "snowflake-ml-python"],
            imports=[
                (jobs_import_src, "snowflake.ml.jobs"),
            ],
        )
        def job_sproc(session: snowpark.Session) -> None:
            job = job_fn()
            assert job.wait() == "DONE", f"Job {job.id} failed. Logs:\n{job.get_logs()}"
            return job.get_logs()

        result = job_sproc()
        self.assertEqual("Hello from remote function!", result)

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

    @absltest.skipIf(  # type: ignore[misc]
        not version.Version(env.PYTHON_VERSION).public.startswith("3.10."),
        "Decorator test only works for Python 3.10 to pickle compatibility",
    )
    def test_job_data_connector(self) -> None:
        from snowflake.ml._internal.utils import mixins
        from snowflake.ml.data import data_connector
        from snowflake.ml.data._internal import arrow_ingestor

        num_rows = 100

        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def runtime_func(dc: data_connector.DataConnector) -> data_connector.DataConnector:
            # TODO(SNOW-2182155): Enable this once headless backend receives updated SnowML with unpickle support
            # assert "Ray" in type(dc._ingestor).__name__, type(dc._ingestor).__qualname__
            assert len(dc.to_pandas()) == num_rows, len(dc.to_pandas())
            return dc

        df = self.session.sql(
            f"SELECT uniform(1, 1000, random()) as random_val FROM table(generator(rowcount => {num_rows}))"
        )
        dc = data_connector.DataConnector.from_dataframe(df)
        self.assertIsInstance(dc._ingestor, arrow_ingestor.ArrowIngestor)

        # TODO(SNOW-2182155): Remove this once headless backend receives updated SnowML with unpickle support
        #       Register key modules to be picklable by value to avoid version desync in this test
        cp.register_pickle_by_value(mixins)
        cp.register_pickle_by_value(arrow_ingestor)
        try:
            job = runtime_func(dc)
        finally:
            cp.unregister_pickle_by_value(mixins)
            cp.unregister_pickle_by_value(arrow_ingestor)

        self.assertEqual(job.wait(), "DONE", job.get_logs())
        dc_unpickled = job.result()
        self.assertIsInstance(dc_unpickled, data_connector.DataConnector)
        self.assertIsInstance(dc_unpickled._ingestor, arrow_ingestor.ArrowIngestor)
        self.assertEqual(dc.to_pandas().shape, dc_unpickled.to_pandas().shape)

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
            assert (
                len(unique_hosts) >= 2
            ), f"Expected at least 2 unique hosts, get: {unique_hosts}, hosts: {ray.get(hosts)}"
            print("test succeeded")

        job = self._submit_func_as_file(ray_workload, target_instances=2, min_instances=2)
        self.assertEqual(job.wait(), "DONE", f"job {job.id} logs: {job.get_logs(verbose=True)}")
        self.assertTrue("test succeeded" in job.get_logs())

    def test_multinode_job_wait_for_instances(self) -> None:
        def get_cluster_size() -> int:
            from common_utils import common_util as mlrs_util

            num_nodes = mlrs_util.get_num_ray_nodes()
            print("num_nodes:", num_nodes)

        # Verify min_instances met
        job1 = self._submit_func_as_file(get_cluster_size, target_instances=3, min_instances=2)
        self.assertEqual(job1.target_instances, 3)
        self.assertEqual(job1.min_instances, 2)
        self.assertEqual(job1.wait(), "DONE", job1.get_logs(verbose=True))
        self.assertIsNotNone(
            match_group := re.search(r"num_nodes: (\d+)", concise_logs := job1.get_logs(verbose=False)),
            concise_logs,
        )
        self.assertBetween(int(match_group.group(1)), 2, 3, match_group.groups())

        # Check verbose log to ensure min_instances was checked
        self.assertIn("instance requirement met", job1.get_logs(verbose=True))

        # Verify min_wait is respected
        job2 = self._submit_func_as_file(
            get_cluster_size,
            target_instances=2,
            min_instances=1,
            env_vars={"MLRS_INSTANCES_MIN_WAIT": 720},
        )
        self.assertEqual(job2.target_instances, 2)
        self.assertEqual(job2.min_instances, 1)
        self.assertEqual(job2.wait(), "DONE", job2.get_logs(verbose=True))
        self.assertIsNotNone(
            match_group := re.search(r"num_nodes: (\d+)", concise_logs := job2.get_logs(verbose=False)),
            concise_logs,
        )
        self.assertEqual(int(match_group.group(1)), 2)

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
        self.assertRegex(job_logs, r"Successfully installed\s+.*\btabulate[-\d\.]*\b")
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
            ("not_valid_database", self.schema, sp_exceptions.SnowparkSQLException, "does not exist"),
            (self.db, "not_valid_schema", sp_exceptions.SnowparkSQLException, "does not exist"),
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
                    (ValueError, sp_exceptions.SnowparkSQLException), re.compile(re.escape(v), re.IGNORECASE)
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
        payload_utils.upload_payloads(
            self.session, pathlib.PurePath(stage_name), types.PayloadSpec(upload_files.path, None)
        )
        test_cases = [
            (f"@{stage_name}/", f"@{stage_name}/subdir/sub_main.py", "DONE"),
            (f"@{stage_name}/subdir", f"@{stage_name}/subdir/sub_main.py", "DONE"),
            (f"@{stage_name}/subdir", "sub_main.py", "DONE"),
            (f"@{stage_name}/subdir", "non_exist_file.py", "FAILED"),
        ]
        for source, entrypoint, expected_status in test_cases:
            with self.subTest(source=source, entrypoint=entrypoint):
                job = jobs.submit_from_stage(
                    source=source,
                    entrypoint=entrypoint,
                    compute_pool=self.compute_pool,
                    stage_name="payload_stage",
                    args=["foo", "--delay", "1"],
                    session=self.session,
                )

                self.assertEqual(job.wait(), expected_status, job.get_logs())

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
        real_run_query = query_helper.run_query

        def sql_side_effect(session: snowpark.Session, query_str: str, *args: Any, **kwargs: Any) -> Any:
            if query_str.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
                raise sp_exceptions.SnowparkSQLException("unable to get logs")
            return real_run_query(session, query_str, *args, **kwargs)

        job = self._submit_func_as_file(dummy_function)
        self.assertEqual(job.wait(), "DONE", job.get_logs())

        # Wait for logs to be ingested into event table
        max_wait = 300
        interval = 30
        elapsed = 0
        spcs_logs = []
        while elapsed <= max_wait:
            spcs_logs = jd._get_logs_spcs(self.session, job.id, instance_id=0, container_name="main")
            if len(spcs_logs) > 0 and "hello world" in os.linesep.join(row[0] for row in spcs_logs):
                break
            time.sleep(interval)
            elapsed += interval
        else:
            raise TimeoutError(f"Event table ingest did not complete in {max_wait} seconds")

        with mock.patch("snowflake.ml.jobs._utils.query_helper.run_query", side_effect=sql_side_effect):
            # check the fallback logic
            # fallback to event table if SPCS logs are not available
            with (
                mock.patch(
                    "snowflake.ml.jobs.job._get_logs_spcs",
                    side_effect=sp_exceptions.SnowparkSQLException("spcs logs not available", sql_error_code=2143),
                ),
                self.assertLogs(level="DEBUG") as cm,
            ):
                self.assertIn("hello world", job.get_logs(verbose=True))
                self.assertTrue(any("falling back to event table" in line for line in cm.output))
            # check the SPCS logs
            with (
                mock.patch(
                    "snowflake.ml.jobs.job._get_service_log_from_event_table",
                    side_effect=NotImplementedError("should be unreachable"),
                ),
                self.assertLogs(level="DEBUG") as cm,
            ):
                self.assertIn("hello world", job.get_logs(verbose=True))
                self.assertFalse(any("falling back to event table" in line for line in cm.output))

    def test_file_indentation_tabs(self) -> None:
        import tempfile

        code = textwrap.dedent(
            """\
        def greet():
        \tprint("Hello")
        \tprint("World")


        if __name__ == "__main__":
        \tgreet()
        """
        )

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=True) as tmp:
            tmp.write(code)
            job = jobs.submit_file(
                tmp.name,
                self.compute_pool,
                stage_name="payload_stage",
                session=self.session,
            )
            self.assertEqual(job.wait(), "DONE", job.get_logs())

    def test_cancel_job(self) -> None:
        """Test cancelling a long running job."""

        def long_running_function() -> None:
            import time

            time.sleep(300)

        job = self._submit_func_as_file(long_running_function)
        self.assertIn(job.status, ["PENDING", "RUNNING"])
        job.cancel()
        try:
            job.wait(timeout=20)
        except TimeoutError:
            print("Job did not cancel within timeout", job.status, job.get_logs())
        finally:
            self.assertIn(job.status, ["CANCELLED", "CANCELLING"])

    def test_cancel_nonexistent_job(self) -> None:
        """Test cancelling a job that doesn't exist."""
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

    def test_multinode_job_orders(self) -> None:
        """Test that the job orders are correct for a multinode job."""
        job = self._submit_func_as_file(dummy_function, target_instances=2)
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        # Step 1: Show service instances in service
        rows = query_helper.run_query(self.session, "SHOW SERVICE INSTANCES IN SERVICE IDENTIFIER(?)", params=[job.id])
        self.assertEqual(len(rows), 2, "Expected 2 service instances for target_instances=2")

        # Step 2: Sort them by start-time
        sorted_instances = sorted(rows, key=lambda x: (x["start_time"], int(x["instance_id"])))

        # Step 3: Check instance with id 0 starts first
        first_instance = sorted_instances[0]
        self.assertEqual(
            int(first_instance["instance_id"]),
            0,
            f"Expected instance 0 to start first, but instance {first_instance['instance_id']} started first",
        )

    @parameterized.parameters(  # type: ignore[misc]
        ("src", "src/entry.py", [(TestAsset("src/subdir/utils").path.as_posix(), "src.subdir.utils")]),
        ("src", "src/nine.py", [(TestAsset("src/subdir/utils").path.as_posix(), "subdir.utils")]),
        ("src/subdir2", "src/subdir2/eight.py", [(TestAsset("src/subdir3/").path.as_posix(), "subdir3")]),
    )
    def test_submit_with_additional_payloads_local(
        self, source: str, entrypoint: str, additional_payloads: list[tuple[str, str]]
    ) -> None:
        job1 = jobs.submit_directory(
            TestAsset(source).path,
            self.compute_pool,
            entrypoint=TestAsset(entrypoint).path,
            stage_name="payload_stage",
            session=self.session,
            additional_payloads=additional_payloads,
        )
        self.assertEqual(job1.wait(), "DONE", job1.get_logs())

        job2 = jobs.submit_file(
            TestAsset(entrypoint).path,
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            additional_payloads=additional_payloads,
        )
        self.assertEqual(job2.wait(), "DONE", job2.get_logs())

    def test_submit_with_additional_payloads_stage(self) -> None:
        stage_path = f"{self.session.get_session_stage()}/{str(uuid4())}"
        upload_files = TestAsset("src")

        payload_utils.upload_payloads(
            self.session, pathlib.PurePath(stage_path), types.PayloadSpec(upload_files.path, None)
        )

        test_cases = [
            (f"{stage_path}/", f"{stage_path}/entry.py", [(f"{stage_path}/subdir/utils", "src.subdir.utils")]),
            (f"{stage_path}", f"{stage_path}/nine.py", [(f"{stage_path}/subdir/utils", "subdir.utils")]),
        ]
        for source, entrypoint, additional_payloads in test_cases:
            with self.subTest(source=source, entrypoint=entrypoint, additional_payloads=additional_payloads):
                job = jobs.submit_from_stage(
                    source=source,
                    entrypoint=entrypoint,
                    compute_pool=self.compute_pool,
                    stage_name="payload_stage",
                    session=self.session,
                    additional_payloads=additional_payloads,
                )
                self.assertEqual(job.wait(), "DONE", job.get_logs())

    def test_submit_directory_with_constants(self) -> None:
        job = jobs.submit_directory(
            TestAsset("src/subdir4").path,
            self.compute_pool,
            stage_name="payload_stage",
            entrypoint="main.py",
            session=self.session,
        )
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        self.assertIn("This is something entirely different", job.get_logs())

    def test_requirements_non_overwrite(self) -> None:
        rows = self.session.sql("SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI%'").collect()
        if not rows:
            self.fail("No PyPI EAI found in environment.")
        pypi_eais = [r["name"] for r in rows]
        job = jobs.submit_directory(
            TestAsset("src/subdir5").path,
            self.compute_pool,
            entrypoint="main.py",
            stage_name="payload_stage",
            external_access_integrations=pypi_eais,
            session=self.session,
        )
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        self.assertIn("Numpy version: 1.23", job.get_logs())
        self.assertIn(f"Cloudpickle version: {version.parse(cp.__version__).major}.", job.get_logs())

    def test_submit_with_hidden_files(self) -> None:
        job = jobs.submit_directory(
            TestAsset("src/subdir6").path,
            self.compute_pool,
            entrypoint="main.py",
            stage_name="payload_stage",
            session=self.session,
        )
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        self.assertIn("This is a secret message stored in a hidden YAML file", job.get_logs())
        self.assertIn("This is the content of a hidden file with no extension", job.get_logs())

    def test_job_with_different_python_version(self) -> None:
        target_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        resources = spec_utils._get_node_resources(self.session, self.compute_pool)
        hardware = "GPU" if resources.gpu > 0 else "CPU"
        try:
            expected_runtime_image = spec_utils._get_runtime_image(self.session, hardware)
        except Exception:
            expected_runtime_image = None

        with mock.patch.dict(os.environ, {constants.ENABLE_IMAGE_VERSION_ENV_VAR: "True"}):
            job = jobs.submit_file(
                TestAsset("src/check_python.py").path,
                self.compute_pool,
                stage_name="payload_stage",
                session=self.session,
            )
            self.assertEqual(job.wait(), "DONE", job.get_logs())
            if expected_runtime_image:
                self.assertIn(
                    target_version,
                    job.get_logs(),
                    f"Expected Python {target_version} when matching runtime available: {expected_runtime_image}",
                )
            else:
                self.assertIn(
                    "3.10",
                    job.get_logs(),
                    "Expected fallback to default Python version when no matching runtime available",
                )


if __name__ == "__main__":
    absltest.main()
