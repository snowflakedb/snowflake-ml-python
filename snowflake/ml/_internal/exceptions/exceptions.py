class SnowflakeMLException(Exception):
    """Base Snowflake ML exception class"""

    def __init__(
        self,
        error_code: str,
        original_exception: Exception,
        suppress_source_trace: bool = False,
    ) -> None:
        """
        Args:
            error_code: Error code.
            original_exception: Original exception. This is the exception raised to users by telemetry.
            suppress_source_trace: Suppress source stacktrace.

        Attributes:
            error_code: Error code.
            original_exception: Original exception with an error code in its message.
            suppress_source_trace: Suppress source stacktrace.

        Raises:
            ValueError: Null error_code or original_exception.

        Examples:
            raise exceptions.SnowflakeMLException(error_code=ERROR_CODE, original_exception=ValueError("Message."))

            Internal error:
                SnowflakeMLException("ValueError('(ERROR_CODE) Message.')")

            User error info:
                ValueError: (ERROR_CODE) Message.
        """
        if not (error_code and original_exception):
            raise ValueError("Must provide non-empty error_code and original_exception.")

        self.error_code = error_code
        self.original_exception = type(original_exception)(f"({self.error_code}) {str(original_exception)}")
        self.suppress_source_trace = suppress_source_trace
        self._pretty_msg = repr(self.original_exception)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._pretty_msg!r})"

    def __str__(self) -> str:
        return self._pretty_msg
