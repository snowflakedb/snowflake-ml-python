import os
import sys
from typing import Any, Optional


class _TqdmStatusContext:
    """A tqdm-based context manager for status updates."""

    def __init__(self, label: str, tqdm_module: Any, total: Optional[int] = None) -> None:
        self._label = label
        self._tqdm = tqdm_module
        self._total = total or 1

    def __enter__(self) -> "_TqdmStatusContext":
        self._progress_bar = self._tqdm.tqdm(desc=self._label, file=sys.stdout, total=self._total, leave=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._progress_bar.close()

    def update(self, label: str, *, state: str = "running", expanded: bool = True) -> None:
        """Update the status by updating the tqdm description."""
        if state == "complete":
            self._progress_bar.update(self._progress_bar.total - self._progress_bar.n)
            self._progress_bar.set_description(label)
        elif state == "error":
            # For error state, use the label as-is and mark with ERROR prefix
            # Don't update progress bar position for errors - leave it where it was
            self._progress_bar.set_description(f"❌ ERROR: {label}")
        else:
            combined_desc = f"{self._label}: {label}" if label != self._label else self._label
            self._progress_bar.set_description(combined_desc)

    def increment(self) -> None:
        """Increment the progress bar."""
        self._progress_bar.update(1)

    def complete(self) -> None:
        """Complete the progress bar to full state."""
        if self._total:
            remaining = self._total - self._progress_bar.n
            if remaining > 0:
                self._progress_bar.update(remaining)


class _StreamlitStatusContext:
    """A streamlit-based context manager for status updates with progress bar support."""

    def __init__(self, label: str, streamlit_module: Any, total: Optional[int] = None) -> None:
        self._label = label
        self._streamlit = streamlit_module
        self._total = total
        self._current = 0
        self._current_label = label
        self._progress_bar = None

    def __enter__(self) -> "_StreamlitStatusContext":
        self._status_container = self._streamlit.status(self._label, state="running", expanded=True)
        if self._total is not None:
            with self._status_container:
                self._progress_bar = self._streamlit.progress(0, text=f"0/{self._total}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Only update to complete if there was no exception
        if exc_type is None:
            self._status_container.update(state="complete")

    def update(self, label: str, *, state: str = "running", expanded: bool = True) -> None:
        """Update the status label."""
        if state == "complete" or state == "error":
            # For completion/error, use the message as-is and update main status
            self._status_container.update(label=label, state=state, expanded=expanded)
            self._current_label = label

            # For error state, update progress bar text but preserve position
            if state == "error" and self._total is not None and self._progress_bar is not None:
                self._progress_bar.progress(
                    self._current / self._total if self._total > 0 else 0,
                    text=f"ERROR - ({self._current}/{self._total})",
                )
        else:
            combined_label = f"{self._label}: {label}" if label != self._label else self._label
            self._status_container.update(label=combined_label, state=state, expanded=expanded)
            self._current_label = label
            if self._total is not None and self._progress_bar is not None:
                progress_value = self._current / self._total if self._total > 0 else 0
                self._progress_bar.progress(progress_value, text=f"({self._current}/{self._total})")

    def increment(self) -> None:
        """Increment the progress."""
        if self._total is not None:
            self._current = min(self._current + 1, self._total)
            if self._progress_bar is not None:
                progress_value = self._current / self._total if self._total > 0 else 0
                self._progress_bar.progress(progress_value, text=f"({self._current}/{self._total})")

    def complete(self) -> None:
        """Complete the progress bar to full state."""
        if self._total is not None:
            self._current = self._total
            if self._progress_bar is not None:
                self._progress_bar.progress(1.0, text=f"({self._current}/{self._total})")


class _NoOpStatusContext:
    """A no-op context manager for when status updates should be disabled."""

    def __init__(self, label: str) -> None:
        self._label = label

    def __enter__(self) -> "_NoOpStatusContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def update(self, label: str, *, state: str = "running", expanded: bool = True) -> None:
        """No-op update method."""
        pass

    def increment(self) -> None:
        """No-op increment method."""
        pass

    def complete(self) -> None:
        """No-op complete method."""
        pass


class ModelEventHandler:
    """Event handler for model operations with streamlit-aware status updates."""

    def __init__(self) -> None:
        self._streamlit = None

        # Try streamlit first
        try:
            import streamlit as st

            if st.runtime.exists():
                USE_STREAMLIT_WIDGETS = os.getenv("USE_STREAMLIT_WIDGETS", "1") == "1"
                if USE_STREAMLIT_WIDGETS:
                    self._streamlit = st
        except ImportError:
            pass

        import tqdm

        self._tqdm = tqdm

    def update(self, message: str) -> None:
        """Write a message using streamlit if available, otherwise use tqdm."""
        if self._streamlit is not None:
            self._streamlit.write(message)
        else:
            self._tqdm.tqdm.write(message)

    def status(
        self,
        label: str,
        *,
        state: str = "running",
        expanded: bool = True,
        total: Optional[int] = None,
        block: bool = True,
    ) -> Any:
        """Context manager that provides status updates with optional enhanced display capabilities.

        Args:
            label: The status label
            state: The initial state ("running", "complete", "error")
            expanded: Whether to show expanded view (streamlit only)
            total: Total number of steps for progress tracking (optional)
            block: Whether to show progress updates (no-op if False)

        Returns:
            Status context (Streamlit, Tqdm, or NoOp based on availability and block parameter)
        """
        if not block:
            return _NoOpStatusContext(label)

        if self._streamlit is not None:
            return _StreamlitStatusContext(label, self._streamlit, total)
        else:
            return _TqdmStatusContext(label, self._tqdm, total)
