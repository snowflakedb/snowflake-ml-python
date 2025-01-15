import pathlib
from typing import Any, Callable, Optional, Type, Union
from uuid import uuid4

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import constants, payload_utils
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

_TEST_SCHEMA = "ML_JOB_TEST_SCHEMA"


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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbm.drop_schema(cls.schema, if_exists=True)
        cls.session.close()
        super().tearDownClass()

    @parameterized.named_parameters(
        ("payload_not_exist", TestAsset("not-exist"), TestAsset("src/main.py"), FileNotFoundError),
        ("entrypoint_not_exist", TestAsset("src"), TestAsset("not-exist"), FileNotFoundError),
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
        source: Union[TestAsset, Callable],
        entrypoint: Optional[TestAsset],
        error_type: Type[Exception] = ValueError,
    ) -> None:
        payload = payload_utils.JobPayload(
            pathlib.Path(source.path) if isinstance(source, TestAsset) else source,
            pathlib.Path(entrypoint.path) if isinstance(entrypoint, TestAsset) else entrypoint,
        )
        with self.assertRaises(error_type):
            payload.upload(self.session, self.session.get_session_stage())

    @parameterized.parameters(
        # Payload == entrypoint
        (TestAsset("src/main.py"), TestAsset("src/main.py"), "main.py", 1),
        (TestAsset("src/main.py"), None, "main.py", 1),
        # Entrypoint as relative path inside payload directory
        (TestAsset("src"), TestAsset("main.py", resolve_path=False), "main.py", 5),
        (TestAsset("src"), TestAsset("subdir/sub_main.py", resolve_path=False), "subdir/sub_main.py", 5),
        (TestAsset("src/subdir"), TestAsset("sub_main.py", resolve_path=False), "sub_main.py", 1),
        # Entrypoint as absolute path
        (TestAsset("src"), TestAsset("src/main.py"), "main.py", 5),
        (TestAsset("src"), TestAsset("src/subdir/sub_main.py"), "subdir/sub_main.py", 5),
        (TestAsset("src/subdir"), TestAsset("src/subdir/sub_main.py"), "sub_main.py", 1),
        # Function as payload
        (function_with_pos_arg, pathlib.Path("function_payload.py"), "function_payload.py", 1),
        (function_with_pos_arg, None, str(constants.DEFAULT_ENTRYPOINT_PATH), 1),
        (function_with_pos_arg_free_vars, None, str(constants.DEFAULT_ENTRYPOINT_PATH), 1),
        (function_with_pos_arg_modules, None, str(constants.DEFAULT_ENTRYPOINT_PATH), 1),
    )
    def test_upload_payload(
        self,
        source: Union[TestAsset, Callable],
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

        # NOTE: Add 1 for the launch script
        expected_file_count = expected_file_count + 1

        self.assertEqual(str(uploaded_payload.entrypoint[-1]), expected_entrypoint)
        self.assertEqual(self.session.sql(f"LIST {stage_path}").count(), expected_file_count)


if __name__ == "__main__":
    absltest.main()
