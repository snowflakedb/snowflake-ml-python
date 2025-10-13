import sys
from types import TracebackType
from typing import Any, Optional
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._interop import exception_utils
from snowflake.snowpark import exceptions as sp_exceptions


class ComplexError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code

    def __repr__(self) -> str:
        return f"ComplexError(message={self.args[0]!r}, code={self.code})"


class TestExceptionUtils(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Ensure hooks are installed for testing
        exception_utils.install_exception_display_hooks()

    @parameterized.named_parameters(  # type: ignore[misc]
        ("value_error", "ValueError", "test error message", "traceback info", None, ValueError("test error message")),
        (
            "not_implemented_error",
            "NotImplementedError",
            "test error message",
            "traceback info",
            None,
            NotImplementedError("test error message"),
        ),
        (
            "snowpark_error",
            "snowflake.snowpark.exceptions.SnowparkSQLException",
            "test error message",
            "traceback info",
            None,
            sp_exceptions.SnowparkSQLException("test error message"),
        ),
        (
            "not_exist_error",
            "NonExistentError",
            "custom error",
            "traceback info",
            None,
            exception_utils.RemoteError("NonExistentError('custom error')"),
        ),
        (
            "complex_ctor",
            "__main__.ComplexError",
            "custom error",
            "traceback info",
            repr(ComplexError("Execution failed with error: custom error", 100)),
            exception_utils.RemoteError(repr(ComplexError("Execution failed with error: custom error", 100))),
        ),
        (
            "custom_repr_NonExistentError",
            "NonExistentError",
            "custom error",
            "traceback info",
            "NonExistentError with custom repr: custom error",
            exception_utils.RemoteError("NonExistentError with custom repr: custom error"),
        ),
        (
            "custom_repr_complex_ctor",
            "__main__.ComplexError",
            "custom error",
            "traceback info",
            "ComplexError with custom repr: custom error",
            exception_utils.RemoteError("ComplexError with custom repr: custom error"),
        ),
    )
    def test_build_exception(
        self,
        exc_type: str,
        exc_msg: str,
        exc_tb: str,
        exc_repr: Optional[str],
        expected: BaseException,
    ) -> None:
        exc_value = exception_utils.build_exception(
            type_str=exc_type,
            message=exc_msg,
            traceback=exc_tb,
            original_repr=exc_repr,
        )
        self.assertEqual(type(exc_value), type(expected))
        self.assertEqual(str(exc_value), str(expected))
        self.assertEqual(
            exception_utils.retrieve_remote_error_info(exc_value),
            exception_utils.RemoteErrorInfo(exc_type.rsplit(".", 1)[-1], exc_msg, exc_tb),
        )

    def test_attach_and_retrieve_traceback(self) -> None:
        """Test attaching and retrieving a traceback from an exception."""
        exc = ValueError("test error")
        exception_utils.attach_remote_error_info(exc, type(exc).__name__, str(exc), "sample traceback")

        # Test retrieval
        remote_error = exception_utils.retrieve_remote_error_info(exc)
        assert remote_error is not None
        self.assertEqual(remote_error.exc_type, "ValueError")
        self.assertEqual(remote_error.exc_msg, "test error")
        self.assertEqual(remote_error.exc_tb, "sample traceback")

        # Test retrieval on exception without traceback
        exc2 = RuntimeError("no traceback")
        self.assertIsNone(exception_utils.retrieve_remote_error_info(exc2))

    def test_excepthook_installation(self) -> None:
        """Test that the excepthook is installed correctly."""
        # Since _attach_excepthook is called on import, sys.excepthook should
        # already be our custom handler. We can verify it's not the original.
        self.assertIsNotNone(getattr(sys, "_original_excepthook", None))
        self.assertNotEqual(sys.excepthook, sys._original_excepthook)  # type: ignore[attr-defined]

    def test_uninstall_sys_excepthook(self) -> None:
        """Test that the system excepthook is properly uninstalled."""
        # First, ensure we have our custom excepthook installed
        original_excepthook = sys._original_excepthook  # type: ignore[attr-defined]
        custom_excepthook = sys.excepthook

        # Uninstall our custom excepthook
        exception_utils._uninstall_sys_excepthook()

        # Verify the original excepthook is restored
        self.assertEqual(sys.excepthook, original_excepthook)
        self.assertFalse(hasattr(sys, "_original_excepthook"))

        # Restore the testing environment
        sys._original_excepthook = original_excepthook  # type: ignore[attr-defined]
        sys.excepthook = custom_excepthook

    def test_uninstall_ipython_hook(self) -> None:
        """Test that the IPython hooks are properly uninstalled."""
        try:
            import IPython
        except ImportError:
            self.skipTest("IPython not available")

        # Mock IPython modules
        with mock.patch.dict(
            "sys.modules",
            {
                "IPython": mock.MagicMock(),
                "IPython.get_ipython": mock.MagicMock(),
                "IPython.core": mock.MagicMock(),
                "IPython.core.ultratb": mock.MagicMock(),
            },
        ):

            # Create mock objects for VerboseTB and ListTB
            mock_verbose_tb = mock.MagicMock()
            mock_list_tb = mock.MagicMock()

            # Setup the original methods that we'll restore
            original_format_exception = mock.MagicMock()
            original_structured_traceback = mock.MagicMock()

            # Setup our mock objects with the attributes we expect
            mock_verbose_tb._original_format_exception_as_a_whole = original_format_exception
            mock_list_tb._original_structured_traceback = original_structured_traceback

            # Get reference to IPython module
            IPython.core.ultratb.VerboseTB = mock_verbose_tb
            IPython.core.ultratb.ListTB = mock_list_tb

            # Call the uninstall function
            exception_utils._uninstall_ipython_hook()

            # Verify that the original methods were restored
            self.assertEqual(mock_verbose_tb.format_exception_as_a_whole, original_format_exception)
            self.assertEqual(mock_list_tb.structured_traceback, original_structured_traceback)

            # Verify that the _original_* attributes were removed
            self.assertFalse(hasattr(mock_verbose_tb, "_original_format_exception_as_a_whole"))
            self.assertFalse(hasattr(mock_list_tb, "_original_structured_traceback"))

    def test_revert_func_wrapper(self) -> None:
        """Test that the revert_func_wrapper properly reverts to original function on error."""
        patched_func = mock.MagicMock(side_effect=Exception("Test error"))
        original_func = mock.MagicMock(return_value="Original result")
        uninstall_func = mock.MagicMock()

        # Create the wrapped function
        wrapped_func = exception_utils._revert_func_wrapper(patched_func, original_func, uninstall_func)

        # Call the wrapped function
        result = wrapped_func("arg1", kwarg1="value1")

        # Verify behavior
        patched_func.assert_called_once_with("arg1", kwarg1="value1")
        uninstall_func.assert_called_once()
        original_func.assert_called_once_with("arg1", kwarg1="value1")
        self.assertEqual(result, "Original result")

    def test_excepthook_fallback_on_error(self) -> None:
        """Test that the system excepthook properly falls back to original if custom hook fails."""
        # Save original state
        original_excepthook = getattr(sys, "_original_excepthook", None)
        custom_excepthook = sys.excepthook

        # Create a mock that tracks if it was called
        mock_original_excepthook = mock.MagicMock()

        # Setup a failing custom excepthook
        def failing_custom_excepthook(
            exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType, **kwargs: Any
        ) -> None:
            raise RuntimeError("Hook failed!")

        # Install our test hooks
        sys._original_excepthook = mock_original_excepthook  # type: ignore[attr-defined]
        sys.excepthook = exception_utils._revert_func_wrapper(
            failing_custom_excepthook, mock_original_excepthook, exception_utils._uninstall_sys_excepthook
        )

        # Trigger the excepthook with an exception
        test_exception = ValueError("Test exception")
        sys.excepthook(ValueError, test_exception, None)

        # Verify the original excepthook was called as fallback
        mock_original_excepthook.assert_called_once()

        # Verify the hook was uninstalled (reset to original)
        self.assertEqual(sys.excepthook, mock_original_excepthook)
        self.assertFalse(hasattr(sys, "_original_excepthook"))

        # Restore original state for other tests
        if original_excepthook:
            sys._original_excepthook = original_excepthook  # type: ignore[attr-defined]
            sys.excepthook = custom_excepthook

    def test_ipython_hook_fallback(self) -> None:
        """Test that IPython hooks properly fall back to original methods if custom hooks fail."""
        try:
            import IPython
        except ImportError:
            self.skipTest("IPython not available")

        with mock.patch.dict(
            "sys.modules",
            {"IPython": mock.MagicMock(), "IPython.core": mock.MagicMock(), "IPython.core.ultratb": mock.MagicMock()},
        ):
            mock_verbose_tb = mock.MagicMock()
            mock_list_tb = mock.MagicMock()

            # Create original methods that will be called if wrapper fails
            original_format_exception = mock.MagicMock(return_value="Original format result")
            original_structured_traceback = mock.MagicMock(return_value="Original traceback result")

            # Create failing custom methods
            failing_format_exception = mock.MagicMock(side_effect=RuntimeError("Formatter failed!"))
            failing_structured_traceback = mock.MagicMock(side_effect=RuntimeError("Traceback formatter failed!"))

            # Setup the class mocks with original methods saved
            mock_verbose_tb._original_format_exception_as_a_whole = original_format_exception
            mock_verbose_tb.format_exception_as_a_whole = exception_utils._revert_func_wrapper(
                failing_format_exception, original_format_exception, exception_utils._uninstall_ipython_hook
            )

            mock_list_tb._original_structured_traceback = original_structured_traceback
            mock_list_tb.structured_traceback = exception_utils._revert_func_wrapper(
                failing_structured_traceback, original_structured_traceback, exception_utils._uninstall_ipython_hook
            )

            # Assign to IPython mock
            IPython.core.ultratb.VerboseTB = mock_verbose_tb
            IPython.core.ultratb.ListTB = mock_list_tb

            # Test VerboseTB formatter fallback
            result1 = mock_verbose_tb.format_exception_as_a_whole("args")
            failing_format_exception.assert_called_once_with("args")
            original_format_exception.assert_called_once_with("args")
            self.assertEqual(result1, "Original format result")

            # VerboseTB should be reset to original
            self.assertEqual(mock_verbose_tb.format_exception_as_a_whole, original_format_exception)
            self.assertFalse(hasattr(mock_verbose_tb, "_original_format_exception_as_a_whole"))

            # Reset for second test
            mock_verbose_tb._original_format_exception_as_a_whole = original_format_exception
            mock_verbose_tb.format_exception_as_a_whole = exception_utils._revert_func_wrapper(
                failing_format_exception, original_format_exception, exception_utils._uninstall_ipython_hook
            )

            # Test ListTB structured_traceback fallback
            result2 = mock_list_tb.structured_traceback("args")
            failing_structured_traceback.assert_called_once_with("args")
            original_structured_traceback.assert_called_once_with("args")
            self.assertEqual(result2, "Original traceback result")

            # ListTB should be reset to original
            self.assertEqual(mock_list_tb.structured_traceback, original_structured_traceback)
            self.assertFalse(hasattr(mock_list_tb, "_original_structured_traceback"))


if __name__ == "__main__":
    absltest.main()
