"""Tests for ModelEventHandler and related event handling functionality."""

import os
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from snowflake.ml.model.event_handler import ModelEventHandler


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


class TestModelEventHandler(absltest.TestCase):
    """Test the ModelEventHandler class."""

    def test_streamlit_not_available(self) -> None:
        """Test behavior when streamlit is not available."""
        with _mock_streamlit_not_available():
            handler = ModelEventHandler()
            self.assertIsNone(handler._streamlit)
            self.assertIsNotNone(handler._tqdm)  # tqdm should always be available

    def test_streamlit_available_but_not_running(self) -> None:
        """Test behavior when streamlit is available but not running."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = False
        with patch("builtins.__import__", return_value=mock_st):
            handler = ModelEventHandler()
            self.assertIsNone(handler._streamlit)
            self.assertIsNotNone(handler._tqdm)  # tqdm should always be available

    def test_streamlit_available_and_running(self) -> None:
        """Test behavior when streamlit is available and running."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        with patch("builtins.__import__", return_value=mock_st):
            handler = ModelEventHandler()
            self.assertEqual(handler._streamlit, mock_st)
            self.assertIsNotNone(handler._tqdm)  # tqdm should always be available

    def test_streamlit_disabled_by_env_var(self) -> None:
        """Test that streamlit can be disabled via environment variable."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        with patch("builtins.__import__", return_value=mock_st):
            with patch.dict(os.environ, {"USE_STREAMLIT_WIDGETS": "0"}):
                handler = ModelEventHandler()
                self.assertIsNone(handler._streamlit)
                self.assertIsNotNone(handler._tqdm)  # tqdm should always be available

    def test_update_with_streamlit(self) -> None:
        """Test update method when streamlit is available."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        with patch("builtins.__import__", return_value=mock_st):
            handler = ModelEventHandler()
            handler.update("Test message")
            mock_st.write.assert_called_once_with("Test message")

    def test_update_without_streamlit(self) -> None:
        """Test update method when streamlit is not available (falls back to tqdm)."""
        with _mock_streamlit_not_available():
            # Mock tqdm.write method
            with patch("tqdm.tqdm.write") as mock_tqdm_write:
                handler = ModelEventHandler()
                # Should not raise an exception and should use tqdm.write
                handler.update("Test message")
                mock_tqdm_write.assert_called_once_with("Test message")

    def test_status_with_streamlit(self) -> None:
        """Test status method when streamlit is available."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        mock_status = MagicMock()
        mock_st.status.return_value = mock_status

        with patch("builtins.__import__", return_value=mock_st):
            handler = ModelEventHandler()

            # Test that the status method returns a context manager
            with handler.status("Test status", state="running", expanded=True):
                pass

            # The streamlit status should be called when entering the context
            mock_st.status.assert_called_once_with("Test status", state="running", expanded=True)

    def test_status_context_with_tqdm_fallback(self) -> None:
        """Test that status context falls back to tqdm when streamlit is not available."""

        with _mock_streamlit_not_available():
            handler = ModelEventHandler()

            # Test that we get a tqdm context manager
            with handler.status("Test operation", total=2) as status:
                status.update("Progress update")
                status.increment()

            # Verify tqdm was used
            self.assertIsNone(handler._streamlit)
            self.assertIsNotNone(handler._tqdm)


class TestEventHandlerProtocol(absltest.TestCase):
    """Test EventHandler protocol compliance."""

    def test_model_event_handler_satisfies_protocol(self) -> None:
        """Test that ModelEventHandler satisfies EventHandler protocol."""
        with _mock_streamlit_not_available():
            handler = ModelEventHandler()

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

    def test_status_with_block_false_returns_noop(self) -> None:
        """Test that status method returns a no-op context manager when block=False."""
        with _mock_streamlit_not_available():
            handler = ModelEventHandler()

            # Test that it returns a no-op context manager
            with handler.status("Test status", block=False) as status:
                # Should be a no-op context
                self.assertEqual(type(status).__name__, "_NoOpStatusContext")

                # These methods should be callable but do nothing
                status.update("Some update")
                status.increment()

                # Should not raise any exceptions

    def test_status_with_block_true_returns_active_context(self) -> None:
        """Test that status method returns an active context manager when block=True."""
        with _mock_streamlit_not_available():
            handler = ModelEventHandler()

            # Test that it returns an active context manager (should be tqdm since streamlit is mocked out)
            with handler.status("Test status", block=True) as status:
                # Should be a tqdm context
                self.assertEqual(type(status).__name__, "_TqdmStatusContext")

    def test_status_with_block_true_and_streamlit_returns_streamlit_context(self) -> None:
        """Test that status method returns streamlit context when streamlit is available and block=True."""
        mock_st = MagicMock()
        mock_st.runtime.exists.return_value = True
        mock_status = MagicMock()
        mock_st.status.return_value = mock_status

        with patch("builtins.__import__", return_value=mock_st):
            handler = ModelEventHandler()

            # Test that it returns a streamlit context manager
            with handler.status("Test status", block=True) as status:
                # Should be a streamlit context
                self.assertEqual(type(status).__name__, "_StreamlitStatusContext")

    def test_status_block_parameter_default_true(self) -> None:
        """Test that the block parameter defaults to True."""
        with _mock_streamlit_not_available():
            handler = ModelEventHandler()

            # Test that default behavior is equivalent to block=True
            with handler.status("Test status") as status_default:
                with handler.status("Test status", block=True) as status_explicit:
                    # Both should be the same type (tqdm context)
                    self.assertEqual(type(status_default).__name__, type(status_explicit).__name__)


if __name__ == "__main__":
    absltest.main()
