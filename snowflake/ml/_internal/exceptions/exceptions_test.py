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


if __name__ == "__main__":
    absltest.main()
