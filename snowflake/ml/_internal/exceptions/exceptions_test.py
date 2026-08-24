from absl.testing import absltest, parameterized

from snowflake.ml._internal.exceptions import error_codes, exceptions


class ExceptionsTest(parameterized.TestCase):
    """Testing exceptions."""

    def test_message(self) -> None:
        message = "Error message."
        error_code_message = f"({error_codes.INTERNAL_TEST}) {message}"
        expected_exception = Exception(error_code_message)
        actual_exception = exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_TEST, original_exception=Exception(message)
        )

        self.assertEqual(repr(expected_exception), str(actual_exception))

    def test_original_exception_requiring_multiple_args_is_preserved(self) -> None:
        class MultiArgError(Exception):
            def __init__(self, statement: str, params: dict[str, str]) -> None:
                super().__init__(f"{statement} {params}")

        original_exception = MultiArgError("SELECT 1", {})
        actual_exception = exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_TEST, original_exception=original_exception
        )

        self.assertIs(original_exception, actual_exception.original_exception)
        self.assertEqual(repr(original_exception), str(actual_exception))


if __name__ == "__main__":
    absltest.main()
