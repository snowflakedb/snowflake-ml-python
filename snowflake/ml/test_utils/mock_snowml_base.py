from __future__ import annotations  # for return self methods

import functools
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypeVar

from snowflake.ml.utils import formatting

_STATEMENT_PARAMS = "statement_params"


@dataclass
class MockOperation:
    """Mock operation on DataFrames/Sessions with expected arguments and results."""

    operation: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any

    # Meta arguments to control checking of members.
    check_args: bool = True
    check_kwargs: bool = True
    # Checking statement_params is off by default for now since it is mainly for validating telemetry.
    check_statement_params: bool = False

    def __repr__(self) -> str:
        return f"{self.operation}(*args={self.args}, **kwargs={self.kwargs}) -> {self.result}"


MockSnowMLBaseSubclassType = TypeVar("MockSnowMLBaseSubclassType", bound="MockSnowMLBase")


class MockSnowMLBase:
    """Base class for mocking snowpark DataFrame and Session."""

    def __init__(self, check_call_sequence_completion: bool = True) -> None:
        self._check_call_sequence_completion = check_call_sequence_completion
        self._call_sequence: list[MockOperation] = []
        self._call_sequence_index = -1

    def _get_class_name(self) -> str:
        return self.__class__.__name__.replace("Mock", "")

    def _generic_operation_wrapper(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper for any expected operation performed on the class"""
        operation = kwargs["_MOCK_OPERATION"]
        del kwargs["_MOCK_OPERATION"]
        return self._check_operation(operation=operation, args=args, kwargs=kwargs).result

    def _check_operation(
        self: MockSnowMLBaseSubclassType,
        operation: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        check_args: bool = True,
        check_kwargs: bool = True,
        check_statement_params: bool = True,
    ) -> MockOperation:
        """Compare an incoming operation against the expected operation sequence."""

        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}

        def formatCall(operation: str, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> str:
            return f"{self._get_class_name()}.{operation}(*args={args}, **kwargs={kwargs})"

        def formatFailureInfo(
            operation: str, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None, expected: MockOperation
        ) -> str:
            return (
                formatting.unwrap(
                    f"""Expected call: {self._get_class_name()}.{expected}
                    Actual   call: {formatCall(operation, args, kwargs)}""",
                    keep_newlines=True,
                )
                + "\n\n"
                + self.format_operations()
            )

        self._call_sequence_index += 1
        assert (
            len(self._call_sequence) > self._call_sequence_index
        ), f"Unexpected call for operation {formatCall(operation, args, kwargs)}. No further calls were expected."
        mo = self._call_sequence[self._call_sequence_index]
        assert mo.operation == operation, (
            formatting.unwrap(
                f""""Operation {self._get_class_name()}.{operation} does not match expected
                     operation {self._get_class_name()}.{mo.operation}."""
            )
            + formatFailureInfo(operation, args, kwargs, mo)
        )

        if check_args and mo.check_args:
            assert args.__eq__(mo.args), (
                f"non-keyword arguments {args} do not match expected non-keyword arguments {mo.args}."
                + formatFailureInfo(operation, args, kwargs, mo)
            )

        # Statement params are used for telemetry only so we handle their matching separately. If
        # check_statement_params is set, we will only validate the subset of entries given in the spected operation.
        # Any additional entries in statement_params will be ignored. If the expected statement_params are empty,
        # we will only validate the presence of statement_params in the call.
        if check_statement_params and mo.check_statement_params:
            assert _STATEMENT_PARAMS in kwargs, (
                f"check_statement_params is true but {_STATEMENT_PARAMS} is missing from actual operation call.\n"
                + formatFailureInfo(operation, args, kwargs, mo)
            )
            assert _STATEMENT_PARAMS in mo.kwargs, (
                f"check_statement_params is true but {_STATEMENT_PARAMS} is missing from expected operation call.\n"
                + formatFailureInfo(operation, args, kwargs, mo)
            )
            mismatched_statement_params = []
            statement_params = kwargs[_STATEMENT_PARAMS]
            mo_statement_params = mo.kwargs[_STATEMENT_PARAMS]
            for k, v in mo_statement_params.items():
                if k not in statement_params or statement_params[k] != mo_statement_params[k]:
                    mismatched_statement_params.append((k, v))
            assert len(mismatched_statement_params) == 0, "Mismatch in statement_params.\n" + formatFailureInfo(
                operation, args, kwargs, mo
            )

            # Remove kwargs for the rest of the comparisons of kwargs.
            del kwargs[_STATEMENT_PARAMS]
            del mo.kwargs[_STATEMENT_PARAMS]
        else:
            # Ignore statement params if check_statement_params is False in either the actual call or the expected
            # call.
            if _STATEMENT_PARAMS in kwargs:
                del kwargs[_STATEMENT_PARAMS]
            if _STATEMENT_PARAMS in mo.kwargs:
                del mo.kwargs[_STATEMENT_PARAMS]

        if check_kwargs and mo.check_kwargs:
            # TODO(amauser): Give a more detailed report on what happened with statement_params.
            assert (
                kwargs == mo.kwargs
            ), f"keyword arguments {kwargs} do not match expected keyword arguments {mo.kwargs}." + formatFailureInfo(
                operation, args, kwargs, mo
            )

        return mo

    def add_operation(
        self: MockSnowMLBaseSubclassType,
        operation: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        result: Any = None,
        check_args: bool = True,
        check_kwargs: bool = True,
        # TODO(amauser): Set this to True after updating the tests.
        check_statement_params: bool = False,
    ) -> MockSnowMLBaseSubclassType:
        """Add an expected operation to the MockDataFrame."""

        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}
        result = result if result is not None else self

        self._call_sequence.append(
            # Checking statement_params is off by default for now since it is mainly for validating telemetry.
            MockOperation(operation, args, kwargs, result, check_args, check_kwargs, check_statement_params)
        )

        # Dynamically create handlers for functions calls that are not explicitly defined.
        if not hasattr(self, operation):
            setattr(self, operation, functools.partial(self._generic_operation_wrapper, _MOCK_OPERATION=operation))
        return self

    # needed for mypy to realise that this class has dynamic attributes.
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    # needed for mypy to realise that this class has dynamic attributes.
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def get_operations(self) -> list[MockOperation]:
        """Returns the list of operations currently expected."""
        return self._call_sequence

    def format_operations(self, indent: int = 0) -> str:
        """Returns a formatted multi-line string with the expected operations, highlighting the current position.

        Args:
            indent (int): Number of spaces to indent the lines.

        Returns:
            A formatted multi-line string.
        """
        result = ""
        base_indent = 8
        for i in range(len(self._call_sequence)):
            prefix = " " * indent
            prefix += "-" * (base_indent - 2) + "> " if i == self._call_sequence_index else " " * base_indent
            result += f"{prefix}[{id(self):08x}-{i:02d}] {self._call_sequence[i]}\n"
        return result

    def finalize(self) -> None:
        """Test that all expected operations were completed."""
        if self._check_call_sequence_completion:
            if len(self._call_sequence) > self._call_sequence_index + 1:
                raise AssertionError(
                    formatting.unwrap(
                        f"""Only completed {self._call_sequence_index + 1} operations out of {len(self._call_sequence)}.
                        Expected operations:""",
                    )
                    + "\n"
                    + self.format_operations()
                )
            elif len(self._call_sequence) < self._call_sequence_index + 1:
                raise AssertionError(
                    formatting.unwrap(
                        f"""There were {self._call_sequence_index + 1} operations while we expected only
                        {len(self._call_sequence)} operations. Expected operations:""",
                    )
                    + "\n"
                    + self.format_operations()
                )

    def __enter__(self: MockSnowMLBaseSubclassType) -> MockSnowMLBaseSubclassType:
        """Helper method called when entering a `with MockDataFrame(...)` context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Helper method when exiting a `with MockDataFrame(...)` context."""
        self.finalize()
