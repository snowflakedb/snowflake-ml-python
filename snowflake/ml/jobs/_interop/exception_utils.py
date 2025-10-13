import builtins
import functools
import importlib
import re
import sys
import traceback
from collections import namedtuple
from types import TracebackType
from typing import Any, Callable, Optional, cast

_TRACEBACK_ENTRY_PATTERN = re.compile(
    r'File "(?P<filename>[^"]+)", line (?P<lineno>\d+), in (?P<name>[^\n]+)(?:\n(?!^\s*File)^\s*(?P<line>[^\n]+))?\n',
    flags=re.MULTILINE,
)
_REMOTE_ERROR_ATTR_NAME = "_remote_error"

RemoteErrorInfo = namedtuple("RemoteErrorInfo", ["exc_type", "exc_msg", "exc_tb"])


class RemoteError(RuntimeError):
    """Base exception for errors from remote execution environment which could not be reconstructed locally."""


def build_exception(type_str: str, message: str, traceback: str, original_repr: Optional[str] = None) -> BaseException:
    """Build an exception from metadata, attaching remote error info."""
    if not original_repr:
        original_repr = f"{type_str}('{message}')"
    try:
        ex = reconstruct_exception(type_str=type_str, message=message)
    except Exception as e:
        # Fallback to a generic error type if reconstruction fails
        ex = RemoteError(original_repr)
        ex.__cause__ = e
    return attach_remote_error_info(ex, type_str, message, traceback)


def reconstruct_exception(type_str: str, message: str) -> BaseException:
    """Best effort reconstruction of an exception from metadata."""
    try:
        type_split = type_str.rsplit(".", 1)
        if len(type_split) == 1:
            module = builtins
        else:
            module = importlib.import_module(type_split[0])
        exc_type = getattr(module, type_split[-1])
    except (ImportError, AttributeError):
        raise ModuleNotFoundError(
            f"Unrecognized exception type '{type_str}', likely due to a missing or unavailable package"
        ) from None

    if not issubclass(exc_type, BaseException):
        raise TypeError(f"Imported type {type_str} is not a known exception type, possibly due to a name conflict")
    return cast(BaseException, exc_type(message))


def attach_remote_error_info(ex: BaseException, exc_type: str, exc_msg: str, traceback_str: str) -> BaseException:
    """
    Attach a string-formatted traceback to an exception.

    When the exception is raised and not caught, it will display the original traceback.
    When caught, it behaves like a regular exception without showing the traceback.

    Args:
        ex: The exception object to modify
        exc_type: The original exception type name
        exc_msg: The original exception message
        traceback_str: String representation of the traceback

    Returns:
        An exception object with the original traceback information
    """
    # Store the traceback information
    exc_type = exc_type.rsplit(".", 1)[-1]  # Remove module path
    setattr(ex, _REMOTE_ERROR_ATTR_NAME, RemoteErrorInfo(exc_type=exc_type, exc_msg=exc_msg, exc_tb=traceback_str))
    return ex


def retrieve_remote_error_info(ex: Optional[BaseException]) -> Optional[RemoteErrorInfo]:
    """
    Retrieve the string-formatted traceback from an exception if it exists.

    Args:
        ex: The exception to retrieve the traceback from

    Returns:
        The remote error tuple if it exists, None otherwise
    """
    if not ex:
        return None
    return getattr(ex, _REMOTE_ERROR_ATTR_NAME, None)


# ###############################################################################
# ------------------------------- !!! NOTE !!! -------------------------------- #
# ###############################################################################
# Job execution results (including uncaught exceptions) are serialized to file(s)
# in mljob_launcher.py. When the job is executed remotely, the serialized results
# are fetched and deserialized in the local environment. If the result contains
# an exception the original traceback is reconstructed and displayed to the user.
#
# It's currently impossible to recreate the original traceback object, so the
# following overrides are necessary to attach and display the deserialized
# traceback during exception handling.
#
# The following code implements the necessary overrides including sys.excepthook
# modifications and IPython traceback formatting. The hooks are applied on init
# and will be active for the duration of the process. The hooks are designed to
# self-uninstall in the event of an error in case of future compatibility issues.
# ###############################################################################


def _revert_func_wrapper(
    patched_func: Callable[..., Any],
    original_func: Callable[..., Any],
    uninstall_func: Callable[[], None],
) -> Callable[..., Any]:
    """
    Create a wrapper function that uninstalls the original function if an error occurs during execution.

    This wrapper provides a fallback mechanism where if the patched function fails, it will:
    1. Uninstall the patched function using the provided uninstall_func, reverting back to using the original function
    2. Re-execute the current call using the original (unpatched) function with the same arguments

    Args:
        patched_func: The patched function to call.
        original_func: The original function to call if patched_func fails.
        uninstall_func: The function to call to uninstall the patched function.

    Returns:
        A wrapped function that calls patched_func and uninstalls on failure.
    """

    @functools.wraps(patched_func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return patched_func(*args, **kwargs)
        except Exception:
            # Uninstall and revert to original on failure
            uninstall_func()
            return original_func(*args, **kwargs)

    return wrapped


def _install_sys_excepthook() -> None:
    """
    Install a custom sys.excepthook to handle remote exception tracebacks.

    sys.excepthook is the global hook that Python calls when an unhandled exception occurs.
    By default it prints the exception type, message and traceback to stderr.

    We override sys.excepthook to intercept exceptions that contain our special RemoteErrorInfo
    attribute. These exceptions come from deserialized remote execution results and contain
    the original traceback information from where they occurred.

    When such an exception is detected, we format and display the original remote traceback
    instead of the local one, which provides better debugging context by showing where the
    error actually happened during remote execution.

    The custom hook maintains proper exception chaining for both __cause__ (from raise from)
    and __context__ (from implicit exception chaining).
    """
    # Attach the custom excepthook for standard Python scripts if not already attached
    if not hasattr(sys, "_original_excepthook"):
        original_excepthook = sys.excepthook

        def custom_excepthook(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_tb: Optional[TracebackType],
            *,
            seen_exc_ids: Optional[set[int]] = None,
        ) -> None:
            if seen_exc_ids is None:
                seen_exc_ids = set()
            seen_exc_ids.add(id(exc_value))

            cause = getattr(exc_value, "__cause__", None)
            context = getattr(exc_value, "__context__", None)
            if cause:
                # Handle cause-chained exceptions
                custom_excepthook(type(cause), cause, cause.__traceback__, seen_exc_ids=seen_exc_ids)
                print(  # noqa: T201
                    "\nThe above exception was the direct cause of the following exception:\n", file=sys.stderr
                )
            elif context and not getattr(exc_value, "__suppress_context__", False):
                # Handle context-chained exceptions
                # Only process context if it's different from cause to avoid double printing
                custom_excepthook(type(context), context, context.__traceback__, seen_exc_ids=seen_exc_ids)
                print(  # noqa: T201
                    "\nDuring handling of the above exception, another exception occurred:\n", file=sys.stderr
                )

            if (remote_err := retrieve_remote_error_info(exc_value)) and isinstance(remote_err, RemoteErrorInfo):
                # Display stored traceback for deserialized exceptions
                print("Traceback (from remote execution):", file=sys.stderr)  # noqa: T201
                print(remote_err.exc_tb, end="", file=sys.stderr)  # noqa: T201
                print(f"{remote_err.exc_type}: {remote_err.exc_msg}", file=sys.stderr)  # noqa: T201
            else:
                # Fall back to the original excepthook
                traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr, chain=False)

        sys._original_excepthook = original_excepthook  # type: ignore[attr-defined]
        sys.excepthook = _revert_func_wrapper(custom_excepthook, original_excepthook, _uninstall_sys_excepthook)


def _uninstall_sys_excepthook() -> None:
    """
    Restore the original excepthook for the current process.

    This is useful when we want to revert to the default behavior after installing a custom excepthook.
    """
    if hasattr(sys, "_original_excepthook"):
        sys.excepthook = sys._original_excepthook
        del sys._original_excepthook


def _install_ipython_hook() -> bool:
    """Install IPython-specific exception handling hook to improve remote error reporting.

    This function enhances IPython's error formatting capabilities by intercepting and customizing
    how remote execution errors are displayed. It modifies two key IPython traceback formatters:

    1. VerboseTB.format_exception_as_a_whole: Customizes the full traceback formatting for remote
        errors by:
        - Adding a "(from remote execution)" header instead of "(most recent call last)"
        - Properly formatting the remote traceback entries
        - Maintaining original behavior for non-remote errors

    2. ListTB.structured_traceback: Modifies the structured traceback output by:
        - Parsing and formatting remote tracebacks appropriately
        - Adding remote execution context to the output
        - Preserving original functionality for local errors

    The modifications are needed because IPython's default error handling doesn't properly display
    remote execution errors that occur in Snowpark/Snowflake operations. The custom formatters
    ensure that error messages from remote executions are properly captured, formatted and displayed
    with the correct context and traceback information.

    Returns:
         bool: True if IPython hooks were successfully installed, False if IPython is not available
         or not in an IPython environment.

    Note:
         This function maintains the ability to revert changes through _uninstall_ipython_hook by
         storing original implementations before applying modifications.
    """
    try:
        from IPython.core.getipython import get_ipython
        from IPython.core.ultratb import ListTB, VerboseTB

        if get_ipython() is None:
            return False
    except ImportError:
        return False

    def parse_traceback_str(traceback_str: str) -> list[tuple[str, int, str, str]]:
        return [
            (m.group("filename"), int(m.group("lineno")), m.group("name"), m.group("line"))
            for m in re.finditer(_TRACEBACK_ENTRY_PATTERN, traceback_str)
        ]

    if not hasattr(VerboseTB, "_original_format_exception_as_a_whole"):
        original_format_exception_as_a_whole = VerboseTB.format_exception_as_a_whole

        def custom_format_exception_as_a_whole(
            self: VerboseTB,
            etype: type[BaseException],
            evalue: Optional[BaseException],
            etb: Optional[TracebackType],
            number_of_lines_of_context: int,
            tb_offset: Optional[int],
            **kwargs: Any,
        ) -> list[list[str]]:
            if (remote_err := retrieve_remote_error_info(evalue)) and isinstance(remote_err, RemoteErrorInfo):
                # Implementation forked from IPython.core.ultratb.VerboseTB.format_exception_as_a_whole
                head = self.prepare_header(remote_err.exc_type, long_version=False).replace(
                    "(most recent call last)",
                    "(from remote execution)",
                )

                frames = ListTB._format_list(
                    self,
                    parse_traceback_str(remote_err.exc_tb),
                )
                formatted_exception = self.format_exception(remote_err.exc_type, remote_err.exc_msg)

                return [[head] + frames + formatted_exception]
            return original_format_exception_as_a_whole(  # type: ignore[no-any-return]
                self,
                etype=etype,
                evalue=evalue,
                etb=etb,
                number_of_lines_of_context=number_of_lines_of_context,
                tb_offset=tb_offset,
                **kwargs,
            )

        VerboseTB._original_format_exception_as_a_whole = original_format_exception_as_a_whole
        VerboseTB.format_exception_as_a_whole = _revert_func_wrapper(
            custom_format_exception_as_a_whole, original_format_exception_as_a_whole, _uninstall_ipython_hook
        )

    if not hasattr(ListTB, "_original_structured_traceback"):
        original_structured_traceback = ListTB.structured_traceback

        def structured_traceback(
            self: ListTB,
            etype: type,
            evalue: Optional[BaseException],
            etb: Optional[TracebackType],
            tb_offset: Optional[int] = None,
            **kwargs: Any,
        ) -> list[str]:
            if (remote_err := retrieve_remote_error_info(evalue)) and isinstance(remote_err, RemoteErrorInfo):
                tb_list = [
                    (m.group("filename"), m.group("lineno"), m.group("name"), m.group("line"))
                    for m in re.finditer(_TRACEBACK_ENTRY_PATTERN, remote_err.exc_tb or "")
                ]
                out_list = original_structured_traceback(self, etype, evalue, tb_list, tb_offset, **kwargs)
                if out_list:
                    out_list[0] = out_list[0].replace(
                        "(most recent call last)",
                        "(from remote execution)",
                    )
                return cast(list[str], out_list)
            return original_structured_traceback(  # type: ignore[no-any-return]
                self, etype, evalue, etb, tb_offset, **kwargs
            )

        ListTB._original_structured_traceback = original_structured_traceback
        ListTB.structured_traceback = _revert_func_wrapper(
            structured_traceback, original_structured_traceback, _uninstall_ipython_hook
        )

    return True


def _uninstall_ipython_hook() -> None:
    """
    Restore the original IPython traceback formatting if it was modified.

    This is useful when we want to revert to the default behavior after installing a custom hook.
    """
    try:
        from IPython.core.ultratb import ListTB, VerboseTB

        if hasattr(VerboseTB, "_original_format_exception_as_a_whole"):
            VerboseTB.format_exception_as_a_whole = VerboseTB._original_format_exception_as_a_whole
            del VerboseTB._original_format_exception_as_a_whole

        if hasattr(ListTB, "_original_structured_traceback"):
            ListTB.structured_traceback = ListTB._original_structured_traceback
            del ListTB._original_structured_traceback
    except ImportError:
        pass


def install_exception_display_hooks() -> None:
    """Install custom exception display hooks for improved remote error reporting.

    This function should be called once during package initialization to set up
    enhanced error handling for remote job execution errors. The hooks will:

    - Display original remote tracebacks instead of local deserialization traces
    - Work in both standard Python and IPython/Jupyter environments
    - Safely fall back to original behavior if errors occur

    Note: This function is idempotent and safe to call multiple times.
    """
    if not _install_ipython_hook():
        _install_sys_excepthook()
