import os
import time
from concurrent import futures
from typing import Any
from unittest import mock

import numpy as np
from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from snowflake.ml._internal.utils import identifier, mixins
from snowflake.ml.jobs._utils import arg_protocol
from snowflake.snowpark import functions as F, session as snowpark
from tests.integ.snowflake.ml.jobs import test_constants
from tests.integ.snowflake.ml.jobs.job_test_base import JobTestBase
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset


@jobs.remote(compute_pool=test_constants._TEST_COMPUTE_POOL, stage_name="payload_stage")
def job_fn_lazy_registration() -> str:
    return "hello world"


class JobDefinitionsTest(JobTestBase):
    def test_delete_job_definition_stage_cleanup(self) -> None:
        """Test that deleting a job definition cleans up the stage files."""
        original_schema = self.session.get_current_schema()
        temp_schema = self.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            job_definition = self._register_definition()
            job_definition()
            self.assertEqual(1, len(jobs.list_jobs(session=self.session)))
            stage_path = job_definition.stage_name
            stage_files = self.session.sql(f"LIST '{stage_path}'").collect()
            self.assertGreater(len(stage_files), 0, "Stage should contain uploaded job files")

            # Verify we can find expected files (startup.sh, requirements.txt, etc.)
            file_names = {row["name"].split("/")[-1] for row in stage_files}
            expected_files = {"startup.sh", "mljob_launcher.py"}
            for expected_file in expected_files:
                self.assertIn(expected_file, file_names, f"Expected file {expected_file} not found in stage")

            job_definition.delete()
            remaining_files = self.session.sql(f"LIST '{stage_path}'").collect()
            self.assertEqual(len(remaining_files), 0, "Stage files should be cleaned up after job deletion")
        finally:
            self.dbm.drop_schema(temp_schema, if_exists=True)
            self.session.use_schema(original_schema)

    def _register_definition(self, **overrides: Any) -> Any:
        payload = TestAsset("src/main.py")
        return jobs.MLJobDefinition.register(
            payload.path,
            compute_pool=self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            **overrides,
        )

    def test_job_definition_multiple_invocations(self) -> None:
        job_def = self._register_definition(overwrite=False)
        try:
            first_job = job_def("foo", "--delay", "1")
            second_job = job_def("foo", "--delay", "1")
            self.assertEqual(first_job.wait(), "DONE", first_job.get_logs(verbose=True))
            self.assertEqual(second_job.wait(), "DONE", second_job.get_logs(verbose=True))

            jobs_df = jobs.list_jobs(session=self.session)
            _, _, definition_name = identifier.parse_schema_level_object_identifier(job_def.job_definition_id)
            self.assertGreaterEqual(
                len(jobs_df[jobs_df["name"].str.lower().str.startswith(definition_name.lower() + "_")]), 2
            )
        finally:
            job_def.delete()

    @parameterized.parameters(  # type: ignore[misc]
        (True,),
        (False,),
    )
    def test_decorator_with_runtime_args(self, import_utils: bool) -> None:
        self.session.sql("ALTER SESSION SET ENABLE_ML_JOB_DEFINITIONS = true").collect()

        @jobs.remote(
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            imports=[(os.path.dirname(jobs.__file__), "snowflake.ml.jobs")] if import_utils else [],
        )
        def job_fn(arg1: str, arg2: int = 1) -> str:
            return f"Hello from remote function! {arg1} {arg2}"

        test_cases: list[tuple[tuple[Any, ...], dict[str, Any], str]] = [
            (("foo",), {"arg2": None}, "foo None"),
            (("bar",), {"arg2": 2}, "bar 2"),
            (("baz",), {}, "baz 1"),
        ]
        for args, kwargs, expected_val in test_cases:
            with self.subTest(args=args, kwargs=kwargs):
                job = job_fn(*args, **kwargs)
                self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
                self.assertIn(job.name.lower(), job._result_path.lower())
                self.assertEqual(
                    job.result(), f"Hello from remote function! {expected_val}", job.get_logs(verbose=True)
                )

    @parameterized.parameters(  # type: ignore[misc]
        (True,),
        (False,),
    )
    def test_decorator_with_runtime_args_stage_payload(self, import_utils: bool) -> None:
        self.session.sql("ALTER SESSION SET ENABLE_ML_JOB_DEFINITIONS = true").collect()

        @jobs.remote(
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            imports=[(os.path.dirname(jobs.__file__), "snowflake.ml.jobs")] if import_utils else [],
        )
        def job_fn(arg1: str, arg2: int = 1) -> str:
            return f"Hello from remote function! {arg1} {arg2}"

        test_cases: list[tuple[tuple[Any, ...], dict[str, Any], str]] = [
            (("foo",), {"arg2": None}, "foo None"),
            (("baz",), {}, "baz 1"),
        ]
        with mock.patch("snowflake.ml.jobs._interop.utils._MAX_INLINE_SIZE", 0):
            for args, kwargs, expected_val in test_cases:
                with self.subTest(args=args, kwargs=kwargs):
                    job = job_fn(*args, **kwargs)
                    self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
                    self.assertIn(job.name.lower(), job._result_path.lower())
                    self.assertEqual(
                        job.result(), f"Hello from remote function! {expected_val}", job.get_logs(verbose=True)
                    )

    @parameterized.parameters(  # type: ignore[misc]
        (True,),
        (False,),
    )
    def test_decorator_with_default_runtime_args_negative(self, import_utils: bool) -> None:
        @jobs.remote(
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            imports=[(os.path.dirname(jobs.__file__), "snowflake.ml.jobs")] if import_utils else [],
        )
        def job_fn(arg1: str, arg2: int) -> str:
            return f"Hello from remote function! {arg1} {arg2}"

        job = job_fn("foo")
        self.assertEqual(job.wait(), "FAILED", job.get_logs(verbose=True))
        self.assertIn("missing 1 required positional argument", job.get_logs(verbose=True))

    def test_arg_protocol_cli(self) -> None:
        job_def = self._register_definition(
            default_args=["--delay", "1", "--flag", "NotNoneValue"], arg_protocol=arg_protocol.ArgProtocol.CLI
        )
        test_cases: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            (("foo",), {"delay": 3, "flag": None}),
            (("hello world",), {"delay": 2}),
        ]
        for args, kwargs in test_cases:
            with self.subTest(args=args, kwargs=kwargs):
                job = job_def(*args, **kwargs)
                self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
                self.assertIn(
                    f"arg1: {args[0]}, delay: {float(kwargs.get('delay', 1) or 1)}", job.get_logs(verbose=True)
                )
                if "flag" in kwargs and kwargs["flag"] is None:
                    self.assertIn("flag is None", job.get_logs(verbose=True))
                else:
                    self.assertNotIn("flag is None", job.get_logs(verbose=True))

    @parameterized.parameters(  # type: ignore[misc]
        (True,),
        (False,),
    )
    def test_job_arguments_load_complex_object_compatibility(self, import_utils: bool) -> None:
        class CustomData:
            def __init__(self, name: str, values: list) -> None:
                self.name = name
                self.values = values

            def summary(self) -> str:
                return f"{self.name}: {sum(self.values)}"

        @jobs.remote(
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            imports=[(os.path.dirname(jobs.__file__), "snowflake.ml.jobs")] if import_utils else [],
        )
        def job_fn(data: CustomData, arr: np.ndarray, multiplier: int = 2) -> str:
            result = data.summary()
            arr_sum = arr.sum() * multiplier
            return f"{result}, array_sum={arr_sum}"

        custom_obj = CustomData("test_data", [1, 2, 3, 4, 5])
        numpy_arr = np.array([10, 20, 30])

        job = job_fn(custom_obj, numpy_arr, multiplier=3)
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        # CustomData sum = 15, numpy sum = 60 * 3 = 180
        self.assertIn("test_data: 15, array_sum=180", job.result(), job.get_logs(verbose=True))

    def test_job_definition_concurrent_invocations(self) -> None:
        job_def = self._register_definition()
        try:
            with futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures_list = [executor.submit(job_def, "foo", "--delay", "1") for _ in range(2)]
            jobs_submitted = [future.result() for future in futures_list]

            for job in jobs_submitted:
                self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        finally:
            job_def.delete()

    def test_job_definition_synchronous(self) -> None:
        job_def = jobs.MLJobDefinition.register(
            TestAsset("src/subdir5/main.py").path,
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
        )
        sql = job_def.to_sql()
        job_id = self.session.sql(sql).collect()[0][0]
        job = jobs.get_job(job_id, session=self.session)
        is_async_job = self.session.sql(f"describe service {job_id}").collect()[0]["is_async_job"]
        self.assertTrue(
            job.wait() == "DONE" or not is_async_job,
            job.get_logs(verbose=True),
        )

    def test_job_with_task_execution(self) -> None:
        from snowflake.core import Root
        from snowflake.core.task import Task

        job_def = self._register_definition()
        sql = job_def.to_sql(job_args=["foo", "--delay", "1"], use_async=False)
        task_name = "TEST_TASK"

        root = Root(self.session)
        root.databases[self.db].schemas[self.schema].tasks.create(Task(name=task_name, definition=sql))
        task_ref = root.databases[self.db].schemas[self.schema].tasks[task_name]
        task_ref.execute()
        max_retries = 60
        status = None
        try:
            for _ in range(max_retries):
                time.sleep(5)
                history = self.session.sql(
                    f"SELECT state FROM TABLE(information_schema.task_history(task_name=>'{task_name}')) "
                    "ORDER BY scheduled_time DESC LIMIT 1"
                ).collect()

                if history:
                    status = history[0][0]
                    if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                        break
            self.assertEqual(status, "SUCCEEDED")
        finally:
            task_ref.drop(if_exists=True)

    def test_job_decorater_lazy_registration(self) -> None:
        job_def = job_fn_lazy_registration
        self.assertFalse(job_def._is_registered)
        job = job_def()
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        self.assertIn(job.name.lower(), job._result_path.lower())
        self.assertEqual(job.result(), "hello world", job.get_logs(verbose=True))

    @parameterized.parameters(  # type: ignore[misc]
        "owner",
        "caller",
    )
    def test_job_definition_in_stored_procedure(self, sproc_rights: str) -> None:
        @F.sproc(
            session=self.session,
            packages=["snowflake-snowpark-python", "snowflake-ml-python"],
            imports=[
                (os.path.dirname(jobs.__file__), "snowflake.ml.jobs"),
                (os.path.dirname(mixins.__file__), "snowflake.ml._internal.utils"),
            ],
            execute_as=sproc_rights,
        )
        def job_sproc(session: snowpark.Session, compute_pool: str) -> str:
            import snowflake.ml.jobs as jobs

            @jobs.remote(compute_pool, stage_name="payload_stage")
            def job_fn() -> None:
                print("Hello from remote function!")

            job = job_fn()
            if job.wait() != "DONE":
                raise RuntimeError(f"Job {job.id} failed. Logs:\n{job.get_logs(verbose=True)}")
            return job.get_logs()

        result = job_sproc(self.session, test_constants._TEST_COMPUTE_POOL)
        self.assertEqual("Hello from remote function!", result)


if __name__ == "__main__":
    absltest.main()
