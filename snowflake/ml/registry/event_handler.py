import logging
import os
from typing import Any


class _NullStatusContext:
    """A fallback context manager that logs status updates."""

    def __init__(self, label: str) -> None:
        self._label = label

    def __enter__(self) -> "_NullStatusContext":
        logging.info(f"Starting: {self._label}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def update(self, label: str, *, state: str = "running", expanded: bool = True) -> None:
        """Update the status by logging the message."""
        logging.info(f"Status update: {label} (state: {state})")


class RegistryEventHandler:
    """Event handler for registry operations with streamlit-aware status updates."""

    def __init__(self) -> None:
        try:
            import streamlit as st

            if not st.runtime.exists():
                self._streamlit = None
            else:
                self._streamlit = st
            USE_STREAMLIT_WIDGETS = os.getenv("USE_STREAMLIT_WIDGETS", "1") == "1"
            if not USE_STREAMLIT_WIDGETS:
                self._streamlit = None
        except ImportError:
            self._streamlit = None

    def update(self, message: str) -> None:
        """Write a message using streamlit if available, otherwise do nothing."""
        if self._streamlit is not None:
            self._streamlit.write(message)

    def status(self, label: str, *, state: str = "running", expanded: bool = True) -> Any:
        """Context manager that provides status updates with optional enhanced display capabilities."""
        if self._streamlit is None:
            return _NullStatusContext(label)
        else:
            return self._streamlit.status(label, state=state, expanded=expanded)
