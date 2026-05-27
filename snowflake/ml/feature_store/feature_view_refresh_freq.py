"""Classification helpers for feature view ``refresh_freq`` / dynamic table lag strings."""

from __future__ import annotations

from typing import Optional

from pytimeparse.timeparse import timeparse


def _is_cron_refresh_freq(refresh_freq: Optional[str]) -> bool:
    """Return True iff ``refresh_freq`` is a CRON expression.

    A CRON expression is anything that is neither ``None`` / ``"DOWNSTREAM"``
    nor a duration parseable by ``pytimeparse`` (e.g. ``"5 minutes"``,
    ``"1h"``).  ``"DOWNSTREAM"`` is matched case-insensitively so callers
    don't need to normalise.  Centralising this check keeps the
    cron-vs-duration logic from drifting across the FV registration, update,
    and rollup paths.

    Args:
        refresh_freq: The refresh frequency string to classify, or ``None``.

    Returns:
        ``True`` if ``refresh_freq`` is a CRON expression; ``False`` if it is
        ``None``, ``"DOWNSTREAM"`` (case-insensitive), or a duration string.
    """
    if refresh_freq is None or refresh_freq.upper() == "DOWNSTREAM":
        return False
    return timeparse(refresh_freq) is None
