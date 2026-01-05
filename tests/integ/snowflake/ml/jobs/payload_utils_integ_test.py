import pathlib
from typing import Any, Callable, Optional, Union
from uuid import uuid4

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import constants, payload_utils, types
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

_TEST_SCHEMA = "ML_JOB_TEST_SCHEMA"

_TEST_STAGE = "TEST_STAGE"


def function_with_pos_arg(a: str, b: int) -> None:
    print(a, b + 1)


def function_with_pos_arg_free_vars(a: str, b: int) -> None:
    print(a, b + 1, _TEST_SCHEMA)


def function_with_pos_arg_modules(a: str, b: int) -> None:
    print(a, b + 1, uuid4())


def function_with_opt_arg(a: str, b: int, c: float = 0.0) -> None:
    print(a, b + 1, c * 2)


def function_with_kw_arg(a: str, b: int, c: float = 0.0, *, named_arg: bool, opt_named: str = "undefined") -> None:
    print(a, b + 1, c * 2, named_arg, "optional: " + opt_named)


def function_with_unpacking_kw_arg(a: str, b: int, c: float = 0.0, *, named_arg: bool, **kwargs: Any) -> None:
    print(a, b + 1, c * 2, named_arg, kwargs)


class PayloadUtilsTests(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)
        cls.dbm.cleanup_schemas(prefix=_TEST_SCHEMA, expire_days=1)
        cls.db = cls.session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(prefix=_TEST_SCHEMA)
        cls.stage = cls.dbm.create_stage(stage_name=_TEST_STAGE, schema_name=cls.schema, db_name=cls.db)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbm.drop_schema(cls.schema, if_exists=True)
        cls.session.close()
        super().tearDownClass()

    @parameterized.named_parameters(  # type: ignore[misc]
        ("payload_not_exist", TestAsset("not-exist"), TestAsset("src/main.py"), FileNotFoundError),
        ("entrypoint_not_exist", TestAsset("src"), TestAsset("src/not-exist"), FileNotFoundError),
        ("entrypoint_null", TestAsset("src"), None, ValueError),
        ("both_not_exist", TestAsset("not-exist"), TestAsset("not-exist"), FileNotFoundError),
        ("dir_as_entrypoint_absolute", TestAsset("src"), TestAsset("src/subdir"), FileNotFoundError),
        ("dir_as_entrypoint_relative", TestAsset("src"), TestAsset("subdir", resolve_path=False), FileNotFoundError),
        ("entrypoint_outside_payload1", TestAsset("src/subdir"), TestAsset("src/main.py"), ValueError),
        ("entrypoint_outside_payload2", TestAsset("src/subdir"), TestAsset("src/subdir2/some_file.py"), ValueError),
        ("callable_payload_unpacking_kw_arg", function_with_unpacking_kw_arg, None, NotImplementedError),
    )
    def test_upload_payload_negative(
        self,
        source: Union[TestAsset, Callable[..., Any]],
        entrypoint: Optional[TestAsset],
        error_type: type[Exception] = ValueError,
    ) -> None:
        payload = payload_utils.JobPayload(
            pathlib.Path(source.path) if isinstance(source, TestAsset) else source,
            pathlib.Path(entrypoint.path) if isinstance(entrypoint, TestAsset) else entrypoint,
        )
        with self.assertRaises(error_type):
            payload.upload(self.session, self.session.get_session_stage())

    @parameterized.parameters(  # type: ignore[misc]
        # Payload == entrypoint
        (TestAsset("src/main.py"), TestAsset("src/main.py"), "/mnt/job_stage/app/main.py", 1),
        (TestAsset("src/main.py"), None, "/mnt/job_stage/app/main.py", 1),
        # Entrypoint as relative path inside payload directory
        (TestAsset("src"), TestAsset("main.py", resolve_path=False), "/mnt/job_stage/app/main.py", 28),
        (
            TestAsset("src"),
            TestAsset("subdir/sub_main.py", resolve_path=False),
            "/mnt/job_stage/app/subdir/sub_main.py",
            28,
        ),
        (TestAsset("src/subdir"), TestAsset("sub_main.py", resolve_path=False), "/mnt/job_stage/app/sub_main.py", 2),
        # Entrypoint as absolute path
        (TestAsset("src"), TestAsset("src/main.py"), "/mnt/job_stage/app/main.py", 28),
        (TestAsset("src"), TestAsset("src/subdir/sub_main.py"), "/mnt/job_stage/app/subdir/sub_main.py", 28),
        (TestAsset("src/subdir"), TestAsset("src/subdir/sub_main.py"), "/mnt/job_stage/app/sub_main.py", 2),
        # Function as payload
        (function_with_pos_arg, pathlib.Path("function_payload.py"), "/mnt/job_stage/app/function_payload.py", 1),
        (function_with_pos_arg, None, f"/mnt/job_stage/app/{constants.DEFAULT_ENTRYPOINT_PATH}", 1),
        (function_with_pos_arg_free_vars, None, f"/mnt/job_stage/app/{constants.DEFAULT_ENTRYPOINT_PATH}", 1),
        (function_with_pos_arg_modules, None, f"/mnt/job_stage/app/{constants.DEFAULT_ENTRYPOINT_PATH}", 1),
    )
    def test_upload_payload(
        self,
        source: Union[TestAsset, Callable[..., Any]],
        entrypoint: Optional[Union[TestAsset, pathlib.Path]],
        expected_entrypoint: str,
        expected_file_count: int,
    ) -> None:
        stage_path = f"{self.session.get_session_stage()}/{str(uuid4())}"

        payload = payload_utils.JobPayload(
            pathlib.Path(source.path) if isinstance(source, TestAsset) else source,
            pathlib.Path(entrypoint.path) if isinstance(entrypoint, TestAsset) else entrypoint,
        )
        uploaded_payload = payload.upload(self.session, stage_path)

        actual_entrypoint = next(
            item for item in reversed(uploaded_payload.entrypoint) if isinstance(item, pathlib.PurePath)
        )
        self.assertEqual(actual_entrypoint.as_posix(), expected_entrypoint)
        self.assertEqual(
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").count(),
            expected_file_count,
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").collect(),
        )

    @parameterized.parameters(  # type: ignore[misc]
        (
            TestAsset("src/main.py"),
            f"@{_TEST_STAGE}/main.py",
            f"@{_TEST_STAGE}/main.py",
            "/mnt/job_stage/app/main.py",
            1,
        ),
        (TestAsset("src/main.py"), f"@{_TEST_STAGE}/main.py", None, "/mnt/job_stage/app/main.py", 1),
        (TestAsset("src"), f"@{_TEST_STAGE}/main.py", None, "/mnt/job_stage/app/main.py", 28),
        (TestAsset("src"), f"@{_TEST_STAGE}/", f"@{_TEST_STAGE}/main.py", "/mnt/job_stage/app/main.py", 28),
        (
            TestAsset("src"),
            f"@{_TEST_STAGE}/",
            f"@{_TEST_STAGE}/subdir/sub_main.py",
            "/mnt/job_stage/app/subdir/sub_main.py",
            28,
        ),
        (
            TestAsset("src"),
            f"@{_TEST_STAGE}/subdir",
            f"@{_TEST_STAGE}/subdir/sub_main.py",
            "/mnt/job_stage/app/sub_main.py",
            2,
        ),
    )
    def test_copy_payload_positive(
        self,
        upload_files: TestAsset,
        source: str,
        entrypoint: Optional[str],
        expected_entrypoint: str,
        expected_file_count: int,
    ) -> None:
        stage_path = f"{self.session.get_session_stage()}/{str(uuid4())}"
        payload_utils.upload_payloads(
            self.session, pathlib.Path(_TEST_STAGE), types.PayloadSpec(upload_files.path, None)
        )
        payload = payload_utils.JobPayload(
            source=source,
            entrypoint=entrypoint,
        )

        uploaded_payload = payload.upload(self.session, stage_path)

        actual_entrypoint = next(
            item for item in reversed(uploaded_payload.entrypoint) if isinstance(item, pathlib.PurePath)
        )

        self.assertEqual(actual_entrypoint.as_posix(), expected_entrypoint)
        self.assertEqual(
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").count(),
            expected_file_count,
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").collect(),
        )

    @parameterized.parameters(  # type: ignore[misc]
        (
            TestAsset("src/entry.py"),
            TestAsset("src/entry.py"),
            [(TestAsset("src/subdir/utils").path.as_posix(), "src.subdir.utils")],
            "/mnt/job_stage/app/entry.py",
            2,
        ),
        (
            TestAsset("src/subdir2/eight.py"),
            TestAsset("src/subdir2/eight.py"),
            [(TestAsset("src/subdir3").path.as_posix(), None)],
            "/mnt/job_stage/app/eight.py",
            2,
        ),
        (
            TestAsset("src/secondary.py"),
            TestAsset("src/secondary.py"),
            [(TestAsset("src/main.py").path.as_posix(), "main")],
            "/mnt/job_stage/app/secondary.py",
            2,
        ),
    )
    def test_upload_payload_additional_imports_local(
        self,
        source: Union[TestAsset, Callable[..., Any]],
        entrypoint: Optional[Union[TestAsset, pathlib.Path]],
        imports: list[Union[str, tuple[str, str]]],
        expected_entrypoint: str,
        expected_file_count: int,
    ) -> None:

        stage_path = f"{self.session.get_session_stage()}/{str(uuid4())}"

        payload = payload_utils.JobPayload(
            pathlib.Path(source.path) if isinstance(source, TestAsset) else source,
            pathlib.Path(entrypoint.path) if isinstance(entrypoint, TestAsset) else entrypoint,
            imports=imports,
        )
        uploaded_payload = payload.upload(self.session, stage_path)
        actual_entrypoint = next(
            item for item in reversed(uploaded_payload.entrypoint) if isinstance(item, pathlib.PurePath)
        )
        self.assertEqual(actual_entrypoint.as_posix(), expected_entrypoint)
        self.assertEqual(
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").count(),
            expected_file_count,
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").collect(),
        )

    @parameterized.parameters(  # type: ignore[misc]
        {
            "source": f"@{_TEST_STAGE}/",
            "entrypoint": f"@{_TEST_STAGE}/subdir/sub_main.py",
            "imports": [(f"@{_TEST_STAGE}/subdir/utils/tool.py", "subdir.utils.tool")],
            "expected_entrypoint": "/mnt/job_stage/app/subdir/sub_main.py",
            "expected_file_count": 29,
        },
        {
            "source": f"@{_TEST_STAGE}/subdir",
            "entrypoint": f"@{_TEST_STAGE}/subdir/sub_main.py",
            "imports": [(f"@{_TEST_STAGE}/subdir2/eight.py", "subdir2.eight")],
            "expected_entrypoint": "/mnt/job_stage/app/sub_main.py",
            "expected_file_count": 3,
        },
    )
    def test_upload_payload_imports_stage(
        self,
        source: str,
        entrypoint: str,
        imports: list,
        expected_entrypoint: str,
        expected_file_count: int,
    ) -> None:
        upload_files = TestAsset("src")
        payload_utils.upload_payloads(
            self.session, pathlib.Path(_TEST_STAGE), types.PayloadSpec(upload_files.path, None)
        )

        stage_path = f"{self.session.get_session_stage()}/{str(uuid4())}"
        payload = payload_utils.JobPayload(
            source=source,
            entrypoint=entrypoint,
            imports=imports,
        )

        uploaded_payload = payload.upload(self.session, stage_path)
        actual_entrypoint = next(
            item for item in reversed(uploaded_payload.entrypoint) if isinstance(item, pathlib.PurePath)
        )
        self.assertEqual(actual_entrypoint.as_posix(), expected_entrypoint)
        self.assertEqual(
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").count(),
            expected_file_count,
            self.session.sql(f"LIST {stage_path}/{constants.APP_STAGE_SUBPATH}").collect(),
        )


if __name__ == "__main__":
    absltest.main()
