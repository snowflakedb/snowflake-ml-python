import inspect
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml import jobs
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.jobs import manager as jm
from snowflake.ml.jobs._utils import constants
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as sp_exceptions
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

_TEST_COMPUTE_POOL = "E2E_TEST_POOL"
_TEST_SCHEMA = "ML_JOB_TEST_SCHEMA"
_SUPPORTED_CLOUDS = {
    snowflake_env.SnowflakeCloudType.AWS,
    snowflake_env.SnowflakeCloudType.AZURE,
}
INVALID_JOB_IDS = [
    "has'quote",
    "quote', 0, 'main'); drop table foo; select system$get_service_logs('job_id'",
]


@absltest.skipIf(
    (region := test_env_utils.get_current_snowflake_region()) is None or region["cloud"] not in _SUPPORTED_CLOUDS,
    "Test only for SPCS supported clouds",
)
class JobManagerTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)
        cls.dbm.cleanup_schemas(prefix=_TEST_SCHEMA, expire_days=1)
        cls.db = cls.session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(prefix=_TEST_SCHEMA)
        try:
            cls.compute_pool = cls.dbm.create_compute_pool(
                _TEST_COMPUTE_POOL, sql_client.CreationMode(if_not_exists=True), max_nodes=5
            )
        except sp_exceptions.SnowparkSQLException:
            if not cls.dbm.show_compute_pools(_TEST_COMPUTE_POOL).count() > 0:
                raise cls.failureException(f"Compute pool {_TEST_COMPUTE_POOL} not available and could not be created")

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
        temp_schema = self.dbm.create_random_schema(prefix=_TEST_SCHEMA)
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
            self.assertSequenceEqual(['"id"', '"owner"', '"status"', '"created_on"', '"compute_pool"'], jobs_df.columns)
            self.assertSequenceEqual(
                ["id", "owner", "status", "created_on", "compute_pool"], list(jobs_df.to_pandas().columns)
            )
            self.assertEqual(job.id, jobs_df.collect()[0]["id"])

            # Loading job ID shouldn't generate any additional jobs in backend
            loaded_job = jobs.get_job(job.id, session=self.session)
            loaded_job.wait()
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
    def test_list_jobs_negative(self, **kwargs: str) -> None:
        with self.assertRaises(sp_exceptions.SnowparkSQLException):
            jobs.list_jobs(**kwargs, session=self.session)

    def test_get_job_negative(self) -> None:
        # Invalid job ID (not a valid identifier)
        for id in INVALID_JOB_IDS:
            with self.assertRaisesRegex(ValueError, "Invalid job ID", msg=f"id={id}"):
                jobs.get_job(id, session=self.session)

        # Non-existent job ID
        with self.assertRaisesRegex(ValueError, "does not exist"):
            jobs.get_job("nonexistent_job_id", session=self.session)

    def test_delete_job_negative(self) -> None:
        for id in INVALID_JOB_IDS + ["nonexistent_job_id"]:
            job = jobs.MLJob(id, session=self.session)
            with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
                jobs.delete_job(job.id, session=self.session)
            with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
                jobs.delete_job(job, session=self.session)

    def test_get_status_negative(self) -> None:
        for id in INVALID_JOB_IDS + ["nonexistent_job_id"]:
            job = jobs.MLJob(id, session=self.session)
            with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
                job.status

    def test_get_logs_negative(self) -> None:
        for id in INVALID_JOB_IDS + ["nonexistent_job_id"]:
            job = jobs.MLJob(id, session=self.session)
            with self.assertRaises(sp_exceptions.SnowparkSQLException, msg=f"id={id}"):
                job.get_logs()

    def test_job_wait(self) -> None:
        # Status check adds some latency
        max_backoff = constants.JOB_POLL_MAX_DELAY_SECONDS
        try:
            # Speed up polling for testing
            constants.JOB_POLL_MAX_DELAY_SECONDS = 0.1  # type: ignore[assignment]
            fudge_factor = 0.2

            # Create a dummy job
            job = jobs.MLJob("dummy_job_id", session=self.session)
            with mock.patch("snowflake.ml.jobs.job._get_status", return_value="RUNNING") as mock_get_status:
                # Test waiting with timeout=0
                start = time.perf_counter()
                with self.assertRaises(TimeoutError):
                    job.wait(timeout=0)
                self.assertLess(time.perf_counter() - start, fudge_factor)
                mock_get_status.assert_called_once()

                start = time.perf_counter()
                with self.assertRaises(TimeoutError):
                    job.wait(timeout=1)
                self.assertBetween(time.perf_counter() - start, 1, 1 + fudge_factor)

            with mock.patch("snowflake.ml.jobs.job._get_status", return_value="DONE") as mock_get_status:
                # Test waiting on a completed job with different timeouts
                start = time.perf_counter()
                self.assertEqual(job.wait(timeout=0), "DONE")
                self.assertEqual(job.wait(timeout=-10), "DONE")
                self.assertEqual(job.wait(timeout=+10), "DONE")
                self.assertLess(time.perf_counter() - start, fudge_factor)
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
        rst = job.wait()
        self.assertEqual(rst, "DONE")
        self.assertEqual(job.status, "DONE")
        self.assertIn("Job complete", job.get_logs())

        # Test loading job by ID
        loaded_job = jobs.get_job(job.id, session=self.session)
        self.assertEqual(loaded_job.status, "DONE")
        self.assertIn("Job start", loaded_job.get_logs())
        self.assertIn("Job complete", loaded_job.get_logs())

    # TODO(SNOW-1911482): Enable test for Python 3.11+
    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_job_decorator(self) -> None:
        @jobs.remote(self.compute_pool, "payload_stage", session=self.session)  # type: ignore[misc]
        def decojob_fn(arg1: str, arg2: int, arg3: Optional[Any] = None) -> None:
            from datetime import datetime

            print(f"{datetime.now()}\t[{arg1}, {arg2}+1={arg2+1}, arg3={arg3}] Job complete", flush=True)

        class MyDataClass:
            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

            def __str__(self) -> str:
                return f"MyDataClass({self.x}, {self.y})"

        # Define parameter combinations to test
        params: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
            (("Positional Arg", 5), {}),
            (("Positional Arg",), {"arg2": 5}),
            (tuple(), {"arg1": "Named Arg", "arg2": 5}),
            (("Positional Arg", 5, {"key": "value"}), {}),
            (("Positional Arg", 5, MyDataClass(1, 2)), {}),
            (("Positional Arg", 5), {"arg3": MyDataClass(1, 2)}),
        ]

        # Kick off jobs in parallel
        job_list: List[jobs.MLJob] = []
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

    def test_job_runtime_api(self) -> None:
        # Submit this function via file to avoid pickling issues
        # TODO: Test this via job decorator as well
        def runtime_func() -> None:
            # NOTE: This Runtime module path may change in the future
            import ray

            from snowflake.ml.data.data_connector import DataConnector
            from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
            from snowflake.snowpark import Session

            # Will throw a ConnectionError if Ray is not initialized
            ray.init(address="auto")

            # Validate simple data ingestion
            session = Session.builder.configs(SnowflakeLoginOptions()).create()
            num_rows = 100
            df = session.sql(
                f"SELECT uniform(1, 1000, random()) as random_val FROM table(generator(rowcount => {num_rows}))"
            )
            dc = DataConnector.from_dataframe(df)
            assert "Ray" in type(dc._ingestor).__name__, type(dc._ingestor).__qualname__
            assert len(dc.to_pandas()) == num_rows, len(dc.to_pandas())

            # Print success message which will be checked in the test
            print("Runtime API test success")

        job = self._submit_func_as_file(runtime_func)
        self.assertEqual(job.wait(), "DONE", job_logs := job.get_logs())
        self.assertIn("Runtime API test success", job_logs)

        # TODO: Add test for DataConnector serialization/deserialization

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

        # Job with bad requirement should fail due to installation error
        self.assertEqual(job_bad_requirement.wait(), "FAILED")
        self.assertIn("No matching distribution found for nonexistent_package", job_bad_requirement.get_logs())

        # Job with no EAI should fail due to network access error
        self.assertEqual(job_no_eai.wait(), "FAILED")
        self.assertIn("No matching distribution found for tabulate", job_no_eai.get_logs())

        # Job with valid requirements and EAI should succeed
        self.assertEqual(job.wait(), "DONE")
        job_logs = job.get_logs()
        self.assertIn("Successfully installed tabulate", job_logs)
        self.assertIn("[foo] Job complete", job_logs)

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

    def _submit_func_as_file(self, func: Callable[[], None]) -> jobs.MLJob:
        func_source = inspect.getsource(func)
        payload_str = textwrap.dedent(func_source) + "\n" + func.__name__ + "()\n"
        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            temp_file.write(payload_str.encode("utf-8"))
            temp_file.flush()
            return jobs.submit_file(
                temp_file.name,
                self.compute_pool,
                stage_name="payload_stage",
                session=self.session,
            )


if __name__ == "__main__":
    absltest.main()
