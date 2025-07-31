"""Mock utilities for progress status testing."""

from unittest import mock


def create_mock_progress_status() -> mock.MagicMock:
    """Create a mock progress status that implements the ProgressStatus protocol."""
    mock_progress = mock.MagicMock()
    mock_progress.update = mock.MagicMock()
    mock_progress.increment = mock.MagicMock()
    mock_progress.complete = mock.MagicMock()
    return mock_progress
