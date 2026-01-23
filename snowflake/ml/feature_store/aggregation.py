"""Aggregation types and specifications for tile-based feature views.

This module provides the building blocks for defining time-series aggregations
that are computed using a tile-based approach for efficiency and correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AggregationType(Enum):
    """Supported aggregation functions for tiled feature views.

    These aggregation types are classified into categories:
    - Simple aggregations (SUM, COUNT, AVG, MIN, MAX, STD, VAR): Stored as scalar partial results in tiles
    - Sketch aggregations (APPROX_COUNT_DISTINCT, APPROX_PERCENTILE): Stored as mergeable state in tiles
    - List aggregations (LAST_N, LAST_DISTINCT_N, FIRST_N, FIRST_DISTINCT_N): Stored as arrays in tiles
    """

    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    APPROX_COUNT_DISTINCT = "approx_count_distinct"
    APPROX_PERCENTILE = "approx_percentile"
    LAST_N = "last_n"
    LAST_DISTINCT_N = "last_distinct_n"
    FIRST_N = "first_n"
    FIRST_DISTINCT_N = "first_distinct_n"

    def is_simple(self) -> bool:
        """Check if this is a simple aggregation (scalar result per tile).

        Simple aggregations include both basic aggregates (SUM, COUNT, etc.)
        and sketch-based aggregates (APPROX_COUNT_DISTINCT, APPROX_PERCENTILE)
        because they all produce a single value per entity per tile boundary.

        Returns:
            True if this is a simple aggregation type, False otherwise.
        """
        return self in (
            AggregationType.SUM,
            AggregationType.COUNT,
            AggregationType.AVG,
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.STD,
            AggregationType.VAR,
            AggregationType.APPROX_COUNT_DISTINCT,
            AggregationType.APPROX_PERCENTILE,
        )

    def is_list(self) -> bool:
        """Check if this is a list aggregation (array result per tile)."""
        return self in (
            AggregationType.LAST_N,
            AggregationType.LAST_DISTINCT_N,
            AggregationType.FIRST_N,
            AggregationType.FIRST_DISTINCT_N,
        )

    def is_sketch(self) -> bool:
        """Check if this is a sketch-based aggregation (HLL, T-Digest)."""
        return self in (
            AggregationType.APPROX_COUNT_DISTINCT,
            AggregationType.APPROX_PERCENTILE,
        )


# Special window value for lifetime aggregations
LIFETIME_WINDOW = "lifetime"

# Regex pattern for interval parsing: "1h", "24 hours", "30 minutes", etc.
_INTERVAL_PATTERN = re.compile(
    r"^\s*(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\s*$",
    re.IGNORECASE,
)

# Mapping of interval unit aliases to canonical Snowflake units
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

# Seconds per unit for calculating window sizes
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

    Args:
        interval: Interval string like "1h", "24 hours", "30 minutes".
            Note: "lifetime" is NOT a valid interval - use is_lifetime_window() first.

    Returns:
        Tuple of (numeric_value, snowflake_unit) e.g. (1, "HOUR").

    Raises:
        ValueError: If the interval format is invalid.
    """
    if is_lifetime_window(interval):
        raise ValueError(
            f"'{interval}' is not a numeric interval. " f"Use is_lifetime_window() to check for lifetime windows."
        )

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

    if value <= 0:
        raise ValueError(f"Interval value must be positive, got: {value}")

    return value, unit


def interval_to_seconds(interval: str) -> int:
    """Convert an interval string to total seconds.

    Args:
        interval: Interval string like "1h", "24 hours".
            Note: "lifetime" is NOT supported - returns -1 as sentinel.

    Returns:
        Total seconds represented by the interval, or -1 for lifetime.
    """
    if is_lifetime_window(interval):
        return -1  # Sentinel value for lifetime
    value, unit = parse_interval(interval)
    return value * _SECONDS_PER_UNIT[unit]


def format_interval_for_snowflake(interval: str) -> str:
    """Format an interval string for use in Snowflake SQL.

    Args:
        interval: Interval string like "1h", "24 hours".

    Returns:
        Formatted string like "HOUR" or "DAY" for use with DATEADD/TIME_SLICE.
    """
    _, unit = parse_interval(interval)
    return unit


@dataclass(frozen=True)
class AggregationSpec:
    """Internal representation of an aggregation specification.

    This is the serializable form that gets stored in metadata and used
    for SQL generation. Users interact with the Feature class instead.

    Attributes:
        function: The aggregation function type.
        source_column: The column to aggregate.
        window: The lookback window for the aggregation (e.g., "24h", "7d").
        output_column: The name of the output column.
        offset: Offset to shift the window into the past (e.g., "1d" means [t-window-1d, t-1d]).
        params: Additional parameters (e.g., {"n": 10} for LAST_N).
    """

    function: AggregationType
    source_column: str
    window: str
    output_column: str
    offset: str = "0"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the aggregation spec after initialization."""
        # Validate window format (allow "lifetime" as special case)
        if not is_lifetime_window(self.window):
            try:
                interval_to_seconds(self.window)
            except ValueError as e:
                raise ValueError(f"Invalid window for aggregation '{self.output_column}': {e}") from e

        # Validate offset format (if not "0")
        # Note: offset is not allowed with lifetime windows
        if self.offset != "0":
            if is_lifetime_window(self.window):
                raise ValueError(
                    f"Offset is not supported with lifetime windows for aggregation '{self.output_column}'"
                )
            try:
                offset_seconds = interval_to_seconds(self.offset)
                if offset_seconds < 0:
                    raise ValueError("Offset must be non-negative")
            except ValueError as e:
                raise ValueError(f"Invalid offset for aggregation '{self.output_column}': {e}") from e

        # Validate params for list aggregations
        if self.function.is_list():
            if "n" not in self.params:
                raise ValueError(
                    f"Parameter 'n' is required for {self.function.value} aggregation " f"'{self.output_column}'"
                )
            n = self.params["n"]
            if not isinstance(n, int) or n <= 0:
                raise ValueError(
                    f"Parameter 'n' must be a positive integer for aggregation " f"'{self.output_column}', got: {n}"
                )

        # Validate params for approx_percentile
        if self.function == AggregationType.APPROX_PERCENTILE:
            if "percentile" not in self.params:
                raise ValueError(
                    f"Parameter 'percentile' is required for approx_percentile aggregation '{self.output_column}'"
                )
            percentile = self.params["percentile"]
            if not isinstance(percentile, (int, float)) or not (0.0 <= percentile <= 1.0):
                raise ValueError(
                    f"Parameter 'percentile' must be a float between 0.0 and 1.0 for aggregation "
                    f"'{self.output_column}', got: {percentile}"
                )

        # Validate lifetime window support
        # Only simple scalar aggregations support lifetime (O(1) via cumulative columns)
        if is_lifetime_window(self.window):
            supported_lifetime_types = (
                AggregationType.SUM,
                AggregationType.COUNT,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.STD,
                AggregationType.VAR,
            )
            if self.function not in supported_lifetime_types:
                supported_names = ", ".join(t.value.upper() for t in supported_lifetime_types)
                raise ValueError(
                    f"Lifetime window is not supported for {self.function.value} aggregation "
                    f"'{self.output_column}'. Lifetime is only supported for: {supported_names}."
                )

    def is_lifetime(self) -> bool:
        """Check if this aggregation has a lifetime window.

        Returns:
            True if the window is "lifetime", False otherwise.
        """
        return is_lifetime_window(self.window)

    def get_window_seconds(self) -> int:
        """Get the window size in seconds.

        Returns:
            Total seconds for the window, or -1 for lifetime windows.
        """
        return interval_to_seconds(self.window)

    def get_offset_seconds(self) -> int:
        """Get the offset in seconds."""
        return interval_to_seconds(self.offset) if self.offset != "0" else 0

    def get_cumulative_column_name(self, partial_type: str) -> str:
        """Get the cumulative column name for lifetime aggregations.

        Similar to get_tile_column_name but with _CUM_ prefix instead of _PARTIAL_.

        Args:
            partial_type: One of "SUM", "COUNT", "SUM_SQ", "HLL", "TDIGEST", "MIN", "MAX", "FIRST".

        Returns:
            Column name used in the tile table (prefixed with _CUM_).
        """
        return f"_CUM_{partial_type}_{self.source_column.upper()}"

    def get_tile_column_name(self, partial_type: str) -> str:
        """Get the internal tile column name for a base partial aggregate.

        Aggregations are computed from base partials:
        - _PARTIAL_SUM_{col}: SUM(col) - used by SUM, AVG, STD, VAR
        - _PARTIAL_COUNT_{col}: COUNT(col) - used by COUNT, AVG, STD, VAR
        - _PARTIAL_SUM_SQ_{col}: SUM(col*col) - used by STD, VAR
        - _PARTIAL_HLL_{col}: HLL state - used by APPROX_COUNT_DISTINCT
        - _PARTIAL_TDIGEST_{col}: T-Digest state - used by APPROX_PERCENTILE

        This allows sharing columns across aggregation types on the same column.

        Args:
            partial_type: One of "SUM", "COUNT", "SUM_SQ", "HLL", "TDIGEST", "LAST", "FIRST".

        Returns:
            Column name used in the tile table (prefixed with _PARTIAL_).
        """
        return f"_PARTIAL_{partial_type}_{self.source_column.upper()}"

    def get_sql_column_name(self) -> str:
        """Get the output column name formatted for SQL.

        Returns:
            Column name ready for use in SQL. Case-sensitive names are stored
            with quotes (e.g., '"My_Col"'), case-insensitive names are uppercase.
        """
        return self.output_column

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "function": self.function.value,
            "source_column": self.source_column,
            "window": self.window,
            "output_column": self.output_column,
            "offset": self.offset,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregationSpec:
        """Create an AggregationSpec from a dictionary."""
        return cls(
            function=AggregationType(data["function"]),
            source_column=data["source_column"],
            window=data["window"],
            output_column=data["output_column"],
            offset=data.get("offset", "0"),
            params=data.get("params", {}),
        )
