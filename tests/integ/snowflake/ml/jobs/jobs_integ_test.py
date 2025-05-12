import inspect
import itertools
import re
import tempfile
import textwrap
import time
from typing import Any, Callable, Optional, cast
from unittest import mock
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml import jobs
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import manager as jm
from snowflake.ml.jobs._utils import constants
from snowflake.ml.utils import sql_client
from snowflake.snowpark import Row, exceptions as sp_exceptions
from snowflake.snowpark.exceptions import SnowparkSQLException
from tests.integ.snowflake.ml.jobs import test_constants
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

INVALID_IDENTIFIERS = [
    "has'quote",
    "quote', 0, 'main'); drop table foo; select system$get_service_logs('job_id'",
]


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

    def test_async_job_parameter(self) -> None:
        try:
            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE").collect()
        except sp_exceptions.SnowparkSQLException:
            self.skipTest("Unable to toggle SPCS Async Jobs parameter. Skipping test.")

        try:
            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = FALSE").collect()
            with self.assertRaisesRegex(RuntimeError, "ENABLE_SNOWSERVICES_ASYNC_JOBS"):
                jm._submit_job(
                    lambda: print("hello world"), self.compute_pool, stage_name="payload_stage", session=self.session
                )

            self.session.sql("ALTER SESSION SET ENABLE_SNOWSERVICES_ASYNC_JOBS = TRUE").collect()
            job = jm._submit_job(
                lambda: print("hello world"), self.compute_pool, stage_name="payload_stage", session=self.session
            )
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
            self.assertEmpty(jobs.list_jobs(session=self.session).collect())

            # Submit a job
            job = jm._submit_job(
                lambda: print("hello world"), self.compute_pool, stage_name="payload_stage", session=self.session
            )

            # Validate list jobs output
            jobs_df = jobs.list_jobs(session=self.session)
            self.assertEqual(1, jobs_df.count())
            self.assertSequenceEqual(
                ['"name"', '"owner"', '"status"', '"created_on"', '"compute_pool"'], jobs_df.columns
            )
            self.assertSequenceEqual(
                ["name", "owner", "status", "created_on", "compute_pool"], list(jobs_df.to_pandas().columns)
            )
            self.assertEqual(job.name, jobs_df.collect()[0]["name"])

            # Loading job ID shouldn't generate any additional jobs in backend
            loaded_job = jobs.get_job(job.id, session=self.session)
            loaded_job.status  # Trigger status check
            self.assertEqual(1, jobs.list_jobs(session=self.session).count())

            # Test different scopes
            scopes = [
                "account",
                "database",
                f"database {self.db}",
                "schema",
                f"schema {temp_schema}",
                f"schema {self.db}.{temp_schema}",
                f"compute pool {self.compute_pool}",
            ]
            for scope in scopes:
                with self.subTest(f"scope={scope}"):
                    self.assertGreater(jobs.list_jobs(scope=scope, session=self.session).count(), 0)

            # Submit a second job to test different limits
            job2 = jm._submit_job(
                lambda: print("hello world"), self.compute_pool, stage_name="payload_stage", session=self.session
            )
            limits = [1, 2, 5, 10]
            for limit in limits:
                with self.subTest(f"limit={limit}"):
                    self.assertBetween(
                        jobs.list_jobs(limit=limit, scope="schema", session=self.session).count(), min(limit, 2), limit
                    )
            self.assertEqual(2, jobs.list_jobs(limit=0, scope="schema", session=self.session).count())
            self.assertEqual(2, jobs.list_jobs(limit=-1, scope="schema", session=self.session).count())
            self.assertEqual(2, jobs.list_jobs(limit=-10, scope="schema", session=self.session).count())

            # Delete the job
            jobs.delete_job(job.id, session=self.session)
            jobs.delete_job(job2.id, session=self.session)
            self.assertEqual(0, jobs.list_jobs(session=self.session).count())
        finally:
            self.dbm.drop_schema(temp_schema, if_exists=True)
            self.session.use_schema(original_schema)

    @parameterized.parameters(  # type: ignore[misc]
        {"scope": "invalid_scope"},
        {"scope": "database not_exist_db"},
        {"scope": "schema not_exist_schema"},
        {"scope": "schema not_exist_db.not_exist_schema"},
        {"scope": "compute_pool not_exist_pool"},
    )
    def test_list_jobs_negative(self, **kwargs: Any) -> None:
        with self.assertRaises(sp_exceptions.SnowparkSQLException):
            jobs.list_jobs(**kwargs, session=self.session)

    def test_get_head_node_negative(self):
        mock_session = MagicMock()
        mock_session.sql.return_value.collect.return_value = [
            Row(instance_id=1),
            Row(instance_id=2),
        ]
        with patch("snowflake.ml.jobs.job._get_num_instances") as mock_get_num_instances:
            mock_get_num_instances.return_value = 3
            job = jobs.MLJob[None](f"{self.db}.{self.schema}.test_id", session=mock_session)
            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to retrieve job logs. "
                "Logs may be inaccessible due to job expiration and can be retrieved from Event Table instead.",
            ):
                job.get_logs()

    def test_get_instance_negative(self):
        def sql_side_effect(query_str, params):
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
        with patch("snowflake.ml.jobs.job._get_num_instances") as mock_get_num_instances:
            mock_get_num_instances.return_value = 1
            job = jobs.MLJob[None](f"{self.db}.{self.schema}.test_id", session=mock_session)
            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to retrieve job logs. "
                "Logs may be inaccessible due to job expiration and can be retrieved from Event Table instead.",
            ):
                job.get_logs()

    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_get_job_positive(self):
        # Submit a job
        job = jm._submit_job(
            lambda: print("hello world"), self.compute_pool, stage_name="payload_stage", session=self.session
        )

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
                with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
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
        job = jm._submit_job(
            lambda: print("hello world"), self.compute_pool, stage_name="payload_stage", session=self.session
        )

        self.assertIsInstance(job.get_logs(), str)
        self.assertIsInstance(job.get_logs(as_list=True), list)

    def test_get_logs_negative(self) -> None:
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
                    job.get_logs()

        mock_session = MagicMock()
        mock_session.sql.side_effect = sp_exceptions.SnowparkSQLException("Waiting to start, Container Status: PENDING")
        job = jobs.MLJob[None](f"{self.db}.{self.schema}.test_id", session=mock_session)
        with self.assertRaises(sp_exceptions.SnowparkSQLException):
            self.assertEqual(
                job.get_logs(instance_id=0),
                "Warning: Waiting for container to start. Logs will be shown when available.",
                job.get_logs(),
            )

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
            import ray

            from snowflake.ml.data.data_connector import DataConnector
            from snowflake.snowpark.context import get_active_session

            # Will throw a ConnectionError if Ray is not initialized
            ray.init(address="auto")

            # Validate simple data ingestion
            session = get_active_session()
            num_rows = 100
            df = session.sql(
                f"SELECT uniform(1, 1000, random()) as random_val FROM table(generator(rowcount => {num_rows}))"
            )
            dc = DataConnector.from_dataframe(df)
            assert "Ray" in type(dc._ingestor).__name__, type(dc._ingestor).__qualname__
            assert len(dc.to_pandas()) == num_rows, len(dc.to_pandas())

            # Print success message which will be checked in the test
            print("Runtime API test success")

        job = self._submit_func_as_file(runtime_func, num_instances=1)
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs())
        self.assertIn("Runtime API test success", job_logs)

        # TODO: Add test for DataConnector serialization/deserialization

    def test_multinode_job_basic(self) -> None:
        def hello_world() -> None:
            print("Hello world")

        job_from_file = self._submit_func_as_file(hello_world, num_instances=2)
        job_from_func = jobs.remote(
            self.compute_pool, stage_name="payload_stage", num_instances=2, session=self.session
        )(hello_world)()

        self.assertEqual(job_from_file.wait(), "DONE", file_job_logs := job_from_file.get_logs())
        self.assertIn("Hello world", file_job_logs)

        self.assertEqual(job_from_func.wait(), "DONE", file_job_logs := job_from_func.get_logs())
        self.assertIn("Hello world", file_job_logs)

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
        self.assertIn("No matching distribution found for nonexistent_package", job_bad_requirement.get_logs())

        # Job with no EAI should fail due to network access error
        self.assertEqual(job_no_eai.wait(), "FAILED")
        self.assertIn("No matching distribution found for tabulate", job_no_eai.get_logs())

        # Job with valid requirements and EAI should succeed
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs())
        self.assertIn("Successfully installed tabulate", job_logs)
        self.assertIn("[foo] Job complete", job_logs)

        # Job with conflicting requirement should prefer the user specified package
        self.assertEqual(job_dep_conflict.wait(), "DONE", job_dep_conflict_logs := job_dep_conflict.get_logs())
        self.assertRegex(job_dep_conflict_logs, r"you have numpy 1\.23\.\d+ which is incompatible")
        self.assertIn("Numpy version: 1.23", job_dep_conflict_logs)

    @parameterized.parameters(  # type: ignore[misc]
        "cloudpickle~=2.0",
        "cloudpickle~=3.0",
    )
    def test_job_cached_packages(self, package: str) -> None:
        """Test that cached packages are available without requiring network access."""

        def dummy_func():
            print("Dummy function executed successfully")

        job = self._submit_func_as_file(dummy_func, pip_requirements=[package])
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs())
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

    def _submit_func_as_file(self, func: Callable[[], None], **kwargs: Any) -> jobs.MLJob[None]:
        func_source = inspect.getsource(func)
        payload_str = textwrap.dedent(func_source) + "\n" + func.__name__ + "()\n"
        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            temp_file.write(payload_str.encode("utf-8"))
            temp_file.flush()
            job: jobs.MLJob[None] = jobs.submit_file(
                temp_file.name,
                self.compute_pool,
                stage_name="payload_stage",
                session=self.session,
                **kwargs,
            )
            return job

    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
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
                    job = jm._submit_job(
                        lambda: print("hello world"),
                        self.compute_pool,
                        stage_name="payload_stage",
                        database=database,
                        schema=schema,
                        session=self.session,
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
            ("not_valid_database", self.schema, SnowparkSQLException, "does not exist"),
            (self.db, "not_valid_schema", SnowparkSQLException, "does not exist"),
            (self.db, None, ValueError, "Schema must be specified if database is specified."),
        ]
        for database, schema, expected_exception, expected_regex in test_cases:
            with self.subTest(database=database, schema=schema):
                with self.assertRaisesRegex(expected_exception, expected_regex):
                    _ = jm._submit_job(
                        lambda: print("hello world"),
                        self.compute_pool,
                        stage_name="payload_stage",
                        database=database,
                        schema=schema,
                        session=self.session,
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
                    _ = jm._submit_job(
                        lambda: print("hello world"),
                        **invalid_kwargs,
                        session=self.session,
                    )


if __name__ == "__main__":
    absltest.main()
