import contextlib
from typing import Generator, Optional, Type

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions


@contextlib.contextmanager
def assert_snowml_exceptions(
    test_case: absltest.TestCase,
    *,
    expected_error_code: Optional[str] = None,
    expected_original_error_type: Optional[Type[Exception]] = None,
    expected_regex: str = "",
) -> Generator[None, None, None]:
    with test_case.assertRaisesRegex(exceptions.SnowflakeMLException, expected_regex) as exc:
        yield
        if expected_error_code:
            test_case.assertEqual(exc.exception.error_code, expected_error_code)
        if expected_original_error_type:
            test_case.assertIsInstance(exc.exception.original_exception, expected_original_error_type)
