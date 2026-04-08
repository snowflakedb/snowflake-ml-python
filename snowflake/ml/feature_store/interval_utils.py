"""Interval parsing and conversion utilities.

General-purpose functions for parsing interval strings (e.g., ``"1h"``,
``"30 minutes"``, ``"0 seconds"``) into structured representations and
converting them to seconds.  Used by aggregation specs, spec builder,
tile SQL generator, and feature views.
"""

from __future__ import annotations

import re

LIFETIME_WINDOW = "lifetime"

_INTERVAL_PATTERN = re.compile(
    r"^\s*(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\s*$",
    re.IGNORECASE,
)

_INTERVAL_UNIT_MAP = {
    "s": "SECOND",
    "sec": "SECOND",
    "secs": "SECOND",
    "second": "SECOND",
    "seconds": "SECOND",
    "m": "MINUTE",
    "min": "MINUTE",
    "mins": "MINUTE",
    "minute": "MINUTE",
    "minutes": "MINUTE",
    "h": "HOUR",
    "hr": "HOUR",
    "hrs": "HOUR",
    "hour": "HOUR",
    "hours": "HOUR",
    "d": "DAY",
    "day": "DAY",
    "days": "DAY",
}

_SECONDS_PER_UNIT = {
    "SECOND": 1,
    "MINUTE": 60,
    "HOUR": 3600,
    "DAY": 86400,
}


def is_lifetime_window(window: str) -> bool:
    """Check if a window string represents a lifetime aggregation.

    Args:
        window: The window string to check.

    Returns:
        True if the window is "lifetime", False otherwise.
    """
    return window.lower().strip() == LIFETIME_WINDOW


def parse_interval(interval: str) -> tuple[int, str]:
    """Parse an interval string into (value, unit) tuple.

    This function is intentionally agnostic about whether zero is valid —
    callers that require positive intervals (e.g., aggregation windows)
    should enforce that constraint themselves.

    Args:
        interval: Interval string like ``"1h"``, ``"24 hours"``, ``"0 seconds"``.
            ``"lifetime"`` is NOT a valid interval — use
            :func:`is_lifetime_window` first.

    Returns:
        Tuple of (numeric_value, snowflake_unit) e.g. ``(1, "HOUR")``.

    Raises:
        ValueError: If the interval format is invalid or the value is negative.
    """
    if is_lifetime_window(interval):
        raise ValueError(f"'{interval}' is not a numeric interval.")

    match = _INTERVAL_PATTERN.match(interval)
    if not match:
        raise ValueError(
            f"Invalid interval format: '{interval}'. "
            f"Expected format: '<number> <unit>' where unit is one of: "
            f"seconds, minutes, hours, days (or abbreviations s, m, h, d)"
        )

    value = int(match.group(1))
    unit_str = match.group(2).lower()
    unit = _INTERVAL_UNIT_MAP[unit_str]

    if value < 0:
        raise ValueError(f"Interval value must be non-negative, got: {value}")

    return value, unit


def interval_to_seconds(interval: str) -> int:
    """Convert an interval string to total seconds.

    Args:
        interval: Interval string like ``"1h"``, ``"24 hours"``.
            ``"lifetime"`` returns ``-1`` as a sentinel value.

    Returns:
        Total seconds represented by the interval, or ``-1`` for lifetime.
    """
    if is_lifetime_window(interval):
        return -1
    value, unit = parse_interval(interval)
    return value * _SECONDS_PER_UNIT[unit]


def format_interval_for_snowflake(interval: str) -> str:
    """Format an interval string for use in Snowflake SQL.

    Args:
        interval: Interval string like ``"1h"``, ``"24 hours"``.

    Returns:
        Snowflake unit string like ``"HOUR"`` or ``"DAY"``.
    """
    _, unit = parse_interval(interval)
    return unit
