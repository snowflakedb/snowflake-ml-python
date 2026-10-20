import os
import pathlib
import tempfile
import textwrap
from typing import Any

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml import jobs
from snowflake.ml._internal.utils import identifier, mixins
from snowflake.ml.jobs._utils import runtime_env_utils
from snowflake.snowpark import exceptions as sp_exceptions, functions as F
from tests.integ.snowflake.ml.jobs.job_test_base import JobTestBase
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset


class CodeBundleIntegTest(JobTestBase):
    """Integration tests for the code bundle APIs.

    Each test checks several behaviors and starts its jobs together where they are independent, since
    starting a job dominates the run time.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.result_stage = cls.dbm.create_stage("code_bundle_result_stage", sse_encrypted=True)

    def _register_bundle(self, *, result_stage: str | None = None, **overrides: Any) -> jobs.MLCodeBundle:
        payload = TestAsset("src/main.py")
        default_kwargs: dict[str, Any] = dict(
            stage_name="payload_stage",
            session=self.session,
        )
        kwargs = {**default_kwargs, **overrides}
        return jobs.register_code_bundle(
            payload.path,
            self.compute_pool,
            result_stage or self.result_stage,
            **kwargs,
        )

    def _register_directory_bundle(self, source_dir: str, name: str, **overrides: Any) -> jobs.MLCodeBundle:
        return jobs.register_code_bundle(
            source_dir,
            self.compute_pool,
            self.result_stage,
            stage_name="payload_stage",
            name=name,
            session=self.session,
            **overrides,
        )

    def _write_files(self, root: str, files: dict[str, str]) -> None:
        for relative_path, body in files.items():
            path = pathlib.Path(root, relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(body))

    def _count_code_bundles(self, short_name: str, schema: str | None = None) -> int:
        scope = f" IN SCHEMA {self.db}.{schema}" if schema else ""
        return len(self.session.sql(f"SHOW CODE BUNDLES LIKE '{short_name}'{scope}").collect())

    def test_register_replace_and_reregister(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir:
            self._write_files(
                source_dir, {"main.py": 'print("main.py ran")\n', "removed.py": 'print("removed.py ran")\n'}
            )
            first = self._register_directory_bundle(source_dir, "cb_lifecycle_test")
            self.assertEqual(self._count_code_bundles("cb_lifecycle_test"), 1)

            pathlib.Path(source_dir, "removed.py").unlink()
            bundle = self._register_directory_bundle(source_dir, "cb_lifecycle_test")

        with self.subTest("replace keeps a single bundle"):
            self.assertEqual(first.name, bundle.name)
            self.assertEqual(self._count_code_bundles("cb_lifecycle_test"), 1)

        with self.subTest("name inferred from the source"):
            inferred = self._register_bundle()
            _, _, inferred_name = inferred.name.rpartition(".")
            self.assertTrue(inferred_name.strip('"').upper().startswith("MAIN_"), inferred.name)
            self.assertEqual(self._count_code_bundles(inferred_name.strip('"')), 1)

        job = jobs.register_job_from_code_bundle(bundle, entrypoint="main.py")
        with self.subTest("a file removed from the source is not in the re-registered bundle"):
            with self.assertRaises(sp_exceptions.SnowparkSQLException):
                jobs.register_job_from_code_bundle(bundle, entrypoint="removed.py")
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        self.assertIn("main.py ran", job.get_logs())

    def test_run_by_handle_full_name_and_short_name(self) -> None:
        bundle = self._register_bundle(name="cb_run_test")
        started = {
            "handle": jobs.register_job_from_code_bundle(
                bundle, entrypoint="main.py", args=["by_handle", "--delay", "1"]
            ),
            "full_name": jobs.register_job_from_code_bundle(
                bundle.name, entrypoint="main.py", args=["by_full_name"], session=self.session
            ),
            "short_name": jobs.register_job_from_code_bundle(
                "cb_run_test", entrypoint="main.py", args=["by_short_name"], session=self.session
            ),
        }
        self.assertEqual(len({job.id for job in started.values()}), 3, "each run should start its own job")
        for how, job in started.items():
            with self.subTest(how):
                self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
                self.assertIn(f"arg1: by_{how}", job.get_logs())
                self.assertEqual(jobs.get_job(job.id, session=self.session).status, "DONE")
                stage_path = job._stage_path
                assert stage_path is not None
                self.assertTrue(stage_path.startswith("snow://"), stage_path)
                self.assertIsNone(job.result())

    def test_execution_errors(self) -> None:
        bundle = self._register_bundle(name="cb_invalid_execution_test")
        for label, code_bundle, entrypoint, expected_error in (
            ("missing_bundle", f"{self.db}.{self.schema}.cb_does_not_exist", "main.py", "does not exist"),
            ("non_python_entrypoint", bundle.name, "main.sh", "must be of types .py"),
            ("missing_entrypoint", bundle.name, "does_not_exist.py", "does_not_exist.py. does not exist"),
        ):
            with self.subTest(label), self.assertRaisesRegex(sp_exceptions.SnowparkSQLException, expected_error):
                jobs.register_job_from_code_bundle(code_bundle, entrypoint=entrypoint, session=self.session)

    def test_directory_source(self) -> None:
        with tempfile.TemporaryDirectory() as project_dir, tempfile.TemporaryDirectory() as library_dir:
            self._write_files(
                project_dir,
                {
                    "helper.py": """
                        def greet():
                            return "hello from helper"
                    """,
                    "pkg/__init__.py": "",
                    "pkg/math_ops.py": """
                        def add(a, b):
                            return a + b
                    """,
                    "entry.py": """
                        import extra_lib
                        import helper
                        from pkg import math_ops

                        print(helper.greet())
                        print(f"add={math_ops.add(2, 3)}")
                        print(f"extra_lib={extra_lib.VALUE}")
                    """,
                    "subdir/sub_main.py": """
                        print("sub_main ran")
                    """,
                    "fail.py": """
                        raise RuntimeError("intentional code bundle failure")
                    """,
                },
            )
            self._write_files(library_dir, {"extra_lib/__init__.py": 'VALUE = "imported"\n'})
            bundle = self._register_directory_bundle(
                project_dir, "cb_directory_test", imports=[(os.path.join(library_dir, "extra_lib"), "extra_lib")]
            )

        expected = {
            "entry.py": ("DONE", ("hello from helper", "add=5", "extra_lib=imported")),
            "subdir/sub_main.py": ("DONE", ("sub_main ran",)),
            "fail.py": ("FAILED", ("intentional code bundle failure",)),
        }
        started = {
            entrypoint: jobs.register_job_from_code_bundle(bundle, entrypoint=entrypoint) for entrypoint in expected
        }
        for entrypoint, job in started.items():
            status, log_lines = expected[entrypoint]
            with self.subTest(entrypoint):
                self.assertEqual(job.wait(), status, job.get_logs(verbose=True))
                logs = job.get_logs()
                for line in log_lines:
                    self.assertIn(line, logs)

    def test_stage_source(self) -> None:
        self.session.sql("CREATE STAGE IF NOT EXISTS payload_stage ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')").collect()
        self.session.file.put(
            str(TestAsset("src/main.py").path), "@payload_stage/cb_stage_src/", auto_compress=False, overwrite=True
        )
        bundle = jobs.register_code_bundle(
            "@payload_stage/cb_stage_src/",
            self.compute_pool,
            self.result_stage,
            stage_name="payload_stage",
            name="cb_stage_source_test",
            session=self.session,
        )
        job = jobs.register_job_from_code_bundle(bundle, entrypoint="main.py", args=["from_stage"])
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        self.assertIn("arg1: from_stage", job.get_logs())

    def test_another_schema(self) -> None:
        other_schema = self.dbm.create_schema(f"{self.schema}_OTHER")
        # CREATE SCHEMA makes the new schema current; keep the session in the test schema.
        self.session.use_schema(self.schema)
        try:
            result_stage = f"@{self.db}.{self.schema}.{self.result_stage}"
            bundles = {
                "cb_fqn_test": self._register_bundle(
                    result_stage=result_stage, name=f"{self.db}.{other_schema}.cb_fqn_test"
                ),
                "cb_kwargs_test": self._register_bundle(
                    result_stage=result_stage, name="cb_kwargs_test", database=self.db, schema=other_schema
                ),
            }
            started = {
                name: jobs.register_job_from_code_bundle(bundle, entrypoint="main.py", args=[name])
                for name, bundle in bundles.items()
            }
            for name, bundle in bundles.items():
                with self.subTest(name):
                    expected_name = identifier.get_schema_level_object_identifier(
                        identifier.resolve_identifier(self.db),
                        identifier.resolve_identifier(other_schema),
                        identifier.resolve_identifier(name),
                    )
                    self.assertEqual(bundle.name, expected_name)
                    self.assertEqual(self._count_code_bundles(name, other_schema), 1)
                    self.assertEqual(self._count_code_bundles(name, self.schema), 0)
                    payload_files = self.session.sql(f"LIST @{self.db}.{other_schema}.payload_stage/{name}/").collect()
                    self.assertNotEqual(payload_files, [], "payload should be uploaded to the bundle's own schema")
                    job = started[name]
                    self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        finally:
            self.dbm.drop_schema(other_schema, if_exists=True)
            self.session.use_schema(self.schema)

    def test_runtime_image_tag_and_full_url(self) -> None:
        image_url = runtime_env_utils.resolve_runtime_image(self.session, self.compute_pool)
        image_tag = image_url.rsplit(":", 1)[1]
        started = {
            label: jobs.register_job_from_code_bundle(
                self._register_bundle(name=f"cb_image_{label}_test", runtime_environment=runtime_environment),
                entrypoint="main.py",
                args=[label],
            )
            for label, runtime_environment in (("tag", image_tag), ("full_url", image_url))
        }
        for label, job in started.items():
            with self.subTest(runtime_environment=label):
                self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
                self.assertTrue(
                    job._container_spec["image"].lower().endswith(image_url.lower()),
                    f"Job ran {job._container_spec['image']}, expected {image_url}",
                )

    def test_stored_procedure(self) -> None:
        # Only caller's rights is covered: in an owner's-rights procedure the server cannot resolve the
        # caller's identity and rejects code bundle execution.
        self.session.sql("CREATE STAGE IF NOT EXISTS payload_stage ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')").collect()
        source_stage_path = "@payload_stage/sproc_src/"
        self.session.file.put(
            str(TestAsset("src/main.py").path), source_stage_path, auto_compress=False, overwrite=True
        )

        @F.sproc(
            session=self.session,
            packages=["snowflake-snowpark-python", "snowflake-ml-python"],
            imports=[
                (os.path.dirname(jobs.__file__), "snowflake.ml.jobs"),
                (os.path.dirname(mixins.__file__), "snowflake.ml._internal.utils"),
            ],
            execute_as="caller",
        )
        def code_bundle_sproc(session: snowpark.Session, compute_pool: str, source: str, result_stage: str) -> str:
            from snowflake.ml import jobs as sproc_jobs

            bundle = sproc_jobs.register_code_bundle(
                source, compute_pool, result_stage, stage_name="payload_stage", name="cb_sproc_test", session=session
            )
            job = sproc_jobs.register_job_from_code_bundle(bundle, entrypoint="main.py", args=["sproc"])
            if job.wait() != "DONE":
                raise RuntimeError(f"Job {job.id} failed. Logs:\n{job.get_logs(verbose=True)}")
            return job.get_logs()

        result = code_bundle_sproc(self.session, self.compute_pool, source_stage_path, self.result_stage)
        self.assertIn("arg1: sproc", result)

    def test_results_and_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir:
            self._write_files(
                source_dir,
                {
                    "entry.py": """
                        import os
                        import sys

                        __return__ = {"arg": sys.argv[1], "env": os.environ.get("CODE_BUNDLE_TEST_VAR")}
                    """
                },
            )
            bundle = self._register_directory_bundle(
                source_dir, "cb_result_test", env_vars={"CODE_BUNDLE_TEST_VAR": "expected_value"}
            )

        # Every run of a bundle writes the same result file, so runs are read one at a time.
        for arg in ("first", "second"):
            with self.subTest(arg):
                job = jobs.register_job_from_code_bundle(bundle, entrypoint="entry.py", args=[arg])
                self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
                self.assertEqual(job.result(), {"arg": arg, "env": "expected_value"})

    def test_delete_job_then_delete_code_bundle(self) -> None:
        bundle = self._register_bundle(name="cb_delete_test")
        payload_stage_path = bundle._stage_path
        self.assertIsNotNone(payload_stage_path)
        self.assertNotEqual(self.session.sql(f"LIST {payload_stage_path}/").collect(), [])

        first_job = jobs.register_job_from_code_bundle(bundle, entrypoint="main.py", args=["run_one"])
        self.assertEqual(first_job.wait(), "DONE", first_job.get_logs(verbose=True))
        jobs.delete_job(first_job)
        with self.subTest("delete_job keeps the bundle runnable"):
            second_job = jobs.register_job_from_code_bundle(bundle, entrypoint="main.py", args=["run_two"])
            self.assertEqual(second_job.wait(), "DONE", second_job.get_logs(verbose=True))

        jobs.delete_code_bundle(bundle)
        with self.subTest("delete_code_bundle drops the bundle and its payload"):
            self.assertEqual(self._count_code_bundles("cb_delete_test"), 0)
            self.assertEqual(self.session.sql(f"LIST {payload_stage_path}/").collect(), [])


if __name__ == "__main__":
    absltest.main()
