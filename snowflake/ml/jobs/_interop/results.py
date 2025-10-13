from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ExecutionResult:
    """
    A result of a job execution.

    Args:
        success: Whether the execution was successful.
        value: The value of the execution.
    """

    success: bool
    value: Any

    def get_value(self, wrap_exceptions: bool = True) -> Any:
        if not self.success:
            assert isinstance(self.value, BaseException), "Unexpected non-exception value for failed result"
            self._raise_exception(self.value, wrap_exceptions)
        return self.value

    def _raise_exception(self, exception: BaseException, wrap_exceptions: bool) -> None:
        if wrap_exceptions:
            raise RuntimeError(f"Job execution failed with error: {exception!r}") from exception
        else:
            raise exception


@dataclass(frozen=True)
class LoadedExecutionResult(ExecutionResult):
    """
    A result of a job execution that has been loaded from a file.
    """

    load_error: Optional[Exception] = None
    result_metadata: Optional[dict[str, Any]] = None

    def get_value(self, wrap_exceptions: bool = True) -> Any:
        if not self.success:
            # Raise the original exception if available, otherwise raise the load error
            ex = self.value
            if not isinstance(ex, BaseException):
                ex = RuntimeError(f"Unknown error {ex or ''}")
                ex.__cause__ = self.load_error
            self._raise_exception(ex, wrap_exceptions)
        else:
            if self.load_error:
                raise ValueError("Job execution succeeded but result retrieval failed") from self.load_error
            return self.value
