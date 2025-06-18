import pathlib
from typing import Any, Callable, Optional, Union
from uuid import uuid4

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import constants, payload_utils
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
        (TestAsset("src/main.py"), TestAsset("src/main.py"), "main.py", 1),
        (TestAsset("src/main.py"), None, "main.py", 1),
        # Entrypoint as relative path inside payload directory
        (TestAsset("src"), TestAsset("main.py", resolve_path=False), "main.py", 7),
        (TestAsset("src"), TestAsset("subdir/sub_main.py", resolve_path=False), "subdir/sub_main.py", 7),
        (TestAsset("src/subdir"), TestAsset("sub_main.py", resolve_path=False), "sub_main.py", 1),
        # Entrypoint as absolute path
        (TestAsset("src"), TestAsset("src/main.py"), "main.py", 7),
        (TestAsset("src"), TestAsset("src/subdir/sub_main.py"), "subdir/sub_main.py", 7),
        (TestAsset("src/subdir"), TestAsset("src/subdir/sub_main.py"), "sub_main.py", 1),
        # Function as payload
        (function_with_pos_arg, pathlib.Path("function_payload.py"), "function_payload.py", 1),
        (function_with_pos_arg, None, str(constants.DEFAULT_ENTRYPOINT_PATH), 1),
        (function_with_pos_arg_free_vars, None, str(constants.DEFAULT_ENTRYPOINT_PATH), 1),
        (function_with_pos_arg_modules, None, str(constants.DEFAULT_ENTRYPOINT_PATH), 1),
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

        system_files_count = 6  # startup.sh and 5 files in scripts/ directory
        expected_file_count = expected_file_count + system_files_count
        if callable(source):
            expected_file_count += 1  # requirements.txt file for callable payload

        actual_entrypoint = next(
            item for item in reversed(uploaded_payload.entrypoint) if isinstance(item, pathlib.PurePath)
        )
        self.assertEqual(actual_entrypoint.as_posix(), expected_entrypoint)
        self.assertEqual(self.session.sql(f"LIST {stage_path}").count(), expected_file_count)

    @parameterized.parameters(
        (TestAsset("src/main.py"), f"@{_TEST_STAGE}/main.py", f"@{_TEST_STAGE}/main.py", "main.py", 1),
        (TestAsset("src/main.py"), f"@{_TEST_STAGE}/main.py", None, "main.py", 1),
        (TestAsset("src"), f"@{_TEST_STAGE}/main.py", None, "main.py", 7),
        (TestAsset("src"), f"@{_TEST_STAGE}/", f"@{_TEST_STAGE}/main.py", "main.py", 7),
        (TestAsset("src"), f"@{_TEST_STAGE}/", f"@{_TEST_STAGE}/subdir/sub_main.py", "subdir/sub_main.py", 7),
        (TestAsset("src"), f"@{_TEST_STAGE}/subdir", f"@{_TEST_STAGE}/subdir/sub_main.py", "sub_main.py", 1),
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
        if upload_files.path.is_dir():
            for path in {
                p.parent.joinpath(f"*{p.suffix}") if p.suffix else p
                for p in upload_files.path.resolve().rglob("*")
                if p.is_file()
            }:
                self.session.file.put(
                    str(path),
                    pathlib.Path(_TEST_STAGE).joinpath(path.parent.relative_to(upload_files.path)).as_posix(),
                    overwrite=True,
                    auto_compress=False,
                )
        else:
            self.session.file.put(
                str(upload_files.path.resolve()),
                f"{_TEST_STAGE}",
                overwrite=True,
                auto_compress=False,
            )
        payload = payload_utils.JobPayload(
            source=source,
            entrypoint=entrypoint,
        )

        uploaded_payload = payload.upload(self.session, stage_path)

        system_files_count = 6
        expected_file_count = expected_file_count + system_files_count
        if callable(source):
            expected_file_count += 1

        actual_entrypoint = next(
            item for item in reversed(uploaded_payload.entrypoint) if isinstance(item, pathlib.PurePath)
        )

        self.assertEqual(actual_entrypoint.as_posix(), expected_entrypoint)
        self.assertEqual(self.session.sql(f"LIST {stage_path}").count(), expected_file_count)


if __name__ == "__main__":
    absltest.main()
