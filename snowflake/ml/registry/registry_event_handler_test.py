"""Tests for RegistryEventHandler and related event handling functionality."""

import os
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from snowflake.ml.registry.registry import RegistryEventHandler, _NullStatusContext


@contextmanager
def _mock_streamlit_not_available() -> Generator[None, None, None]:
    """Context manager that mocks streamlit import to raise ImportError."""
    original_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "streamlit":
            raise ImportError("No module named 'streamlit'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        yield


class TestNullStatusContext(absltest.TestCase):
    """Test the _NullStatusContext fallback implementation."""

    def test_context_manager_protocol(self) -> None:
        """Test that _NullStatusContext implements context manager protocol correctly."""
        context = _NullStatusContext("Test operation")

        # Test __enter__ returns self
        with context as ctx:
            self.assertIs(ctx, context)

    def test_context_manager_logs_start_message(self) -> None:
        """Test that entering the context manager logs the start message."""
        with self.assertLogs(level="INFO") as log:
            with _NullStatusContext("Test operation"):
                pass

            # Should log the starting message
            self.assertEqual(len(log.output), 1)
            self.assertIn("Starting: Test operation", log.output[0])

    def test_update_method_logs_correctly(self) -> None:
        """Test the update method logs messages with correct format."""
        context = _NullStatusContext("Test operation")

        with self.assertLogs(level="INFO") as log:
            with context as ctx:
                ctx.update("Processing data", state="running")
                ctx.update("Complete!", state="complete", expanded=False)

            # Should have start message plus 2 update messages
            self.assertEqual(len(log.output), 3)
            self.assertIn("Starting: Test operation", log.output[0])
            self.assertIn("Status update: Processing data (state: running)", log.output[1])
            self.assertIn("Status update: Complete! (state: complete)", log.output[2])


class TestRegistryEventHandler(absltest.TestCase):
    """Test the RegistryEventHandler class."""

    def test_streamlit_not_available(self) -> None:
        """Test behavior when streamlit is not available."""
        with _mock_streamlit_not_available():
            handler = RegistryEventHandler()
            self.assertIsNone(handler._streamlit)

    def test_streamlit_available_but_not_running(self) -> None:
        """Test behavior when streamlit is available but not running."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = False
        with patch("builtins.__import__", return_value=mock_st):
            handler = RegistryEventHandler()
            self.assertIsNone(handler._streamlit)

    def test_streamlit_available_and_running(self) -> None:
        """Test behavior when streamlit is available and running."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        with patch("builtins.__import__", return_value=mock_st):
            handler = RegistryEventHandler()
            self.assertEqual(handler._streamlit, mock_st)

    def test_streamlit_disabled_by_env_var(self) -> None:
        """Test that streamlit can be disabled via environment variable."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        with patch("builtins.__import__", return_value=mock_st):
            with patch.dict(os.environ, {"USE_STREAMLIT_WIDGETS": "0"}):
                handler = RegistryEventHandler()
                self.assertIsNone(handler._streamlit)

    def test_update_with_streamlit(self) -> None:
        """Test update method when streamlit is available."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        with patch("builtins.__import__", return_value=mock_st):
            handler = RegistryEventHandler()
            handler.update("Test message")
            mock_st.write.assert_called_once_with("Test message")

    def test_update_without_streamlit(self) -> None:
        """Test update method when streamlit is not available."""
        with _mock_streamlit_not_available():
            handler = RegistryEventHandler()
            # Should not raise an exception
            handler.update("Test message")

    def test_status_with_streamlit(self) -> None:
        """Test status method when streamlit is available."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        mock_status = MagicMock()
        mock_st.status.return_value = mock_status

        with patch("builtins.__import__", return_value=mock_st):
            handler = RegistryEventHandler()
            result = handler.status("Test status", state="running", expanded=True)

            mock_st.status.assert_called_once_with("Test status", state="running", expanded=True)
            self.assertEqual(result, mock_status)

    def test_status_context_with_logging_fallback(self) -> None:
        """Test that status context with logging fallback works correctly."""
        with _mock_streamlit_not_available():
            handler = RegistryEventHandler()

            with self.assertLogs(level="INFO") as log:
                with handler.status("Test operation") as status:
                    status.update("Progress update")

                # Should have start message and update message
                self.assertEqual(len(log.output), 2)
                self.assertIn("Starting: Test operation", log.output[0])
                self.assertIn("Status update: Progress update (state: running)", log.output[1])


class TestEventHandlerProtocol(absltest.TestCase):
    """Test EventHandler protocol compliance."""

    def test_registry_event_handler_satisfies_protocol(self) -> None:
        """Test that RegistryEventHandler satisfies EventHandler protocol."""
        with _mock_streamlit_not_available():
            handler = RegistryEventHandler()

            # Check that it has the required method
            self.assertTrue(hasattr(handler, "update"))
            self.assertTrue(callable(handler.update))

            # Verify method signature accepts string
            handler.update("Test")  # Should not raise

    def test_custom_event_handler_can_satisfy_protocol(self) -> None:
        """Test that custom implementations can satisfy the event handler interface."""

        class CustomEventHandler:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def update(self, message: str) -> None:
                self.messages.append(message)

        # Should satisfy the interface
        handler = CustomEventHandler()
        handler.update("Test message")
        self.assertEqual(handler.messages, ["Test message"])


if __name__ == "__main__":
    absltest.main()
