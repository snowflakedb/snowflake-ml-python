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


@dataclass(frozen=True)
class DistributedResult:
    """Aggregated result of a multi-node job that has no single result-bearing instance.

    Results are reduced over the submitted target instance range.

    Args:
        success: True iff all declared instances exited 0.
        exit_codes: instance_id -> exit code; None = lost (killed before writing a record).
        failed_instance: Earliest-failing instance id; None on success.
        return_value: The run's Python return value on success — persisted by instance 0, if the
            entrypoint produced one — else None (subprocess entrypoints, lost instances, and
            failures have no return value).
    """

    success: bool
    exit_codes: dict[int, Optional[int]]
    failed_instance: Optional[int] = None
    return_value: Any = None


class DistributedJobError(RuntimeError):
    """Raised by ``MLJob.distributed_result()`` when a multi-node job did not fully succeed.

    Carries the full :class:`DistributedResult` (``.result``) so callers can inspect per-instance
    ``exit_codes`` and ``failed_instance``. The earliest-failing instance's reconstructed
    exception is attached by the raising site as ``__cause__`` (via ``raise ... from``), so
    the original error and its remote traceback surface on an uncaught raise.
    """

    def __init__(self, message: str, result: Optional[DistributedResult] = None) -> None:
        super().__init__(message)
        self.result = result

    @classmethod
    def from_result(cls, result: DistributedResult) -> "DistributedJobError":
        num_succeeded = sum(1 for code in result.exit_codes.values() if code == 0)
        num_failed = len(result.exit_codes) - num_succeeded
        return cls(
            f"Distributed job failed: {num_failed}/{len(result.exit_codes)} instances did not exit 0 "
            f"(earliest failed instance: {result.failed_instance}); see .result.exit_codes for details.",
            result=result,
        )
