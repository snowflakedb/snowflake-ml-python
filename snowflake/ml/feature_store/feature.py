"""User-facing Feature builder for defining feature view features.

This module provides the Feature class for defining features in a FeatureView,
including both aggregated features (for tiled feature views) and non-aggregated
fields (for standard feature views).
"""

from __future__ import annotations

from typing import Any, Optional

from snowflake.ml.feature_store.aggregation import (
    AggregationSpec,
    AggregationType,
    is_lifetime_window,
)


class Feature:
    """Fluent builder for defining features in a FeatureView.

    This class provides a user-friendly API for defining features with
    time-series aggregations. It supports method chaining for setting
    options like aliases.

    Example::

        >>> from snowflake.ml.feature_store import Feature
        >>>
        >>> # Define features with factory methods
        >>> amount_sum = Feature.sum("amount", "24h").alias("total_amount_24h")
        >>> recent_pages = Feature.last_n("page_id", "1h", n=10).alias("recent_pages")
        >>> txn_count = Feature.count("transaction_id", "7d")
    """

    def __init__(
        self,
        function: AggregationType,
        column: str,
        window: str,
        offset: str = "0",
        **params: Any,
    ) -> None:
        """Create a Feature with the specified aggregation.

        Args:
            function: The aggregation function type.
            column: The source column to aggregate.
            window: The lookback window (e.g., "24h", "7d").
            offset: Offset to shift window into past (e.g., "1d" = [t-window-1d, t-1d]).
                Must be a multiple of feature_granularity. Default is "0" (no offset).
            **params: Additional parameters for the aggregation.
        """
        self._function = function
        self._column = column
        self._window = window
        self._offset = offset
        self._params = params
        self._alias: Optional[str] = None

    def alias(self, name: str, case_sensitive: bool = False) -> Feature:
        """Set the output column name for this feature.

        Args:
            name: The output column name.
            case_sensitive: If True, preserve the exact case of the name (will be quoted in SQL).
                If False (default), the name will be converted to uppercase (Snowflake default).

        Returns:
            Self for method chaining.
        """
        # Store with quotes if case-sensitive, uppercase if case-insensitive
        self._alias = f'"{name}"' if case_sensitive else name.upper()
        return self

    def to_spec(self) -> AggregationSpec:
        """Convert to an AggregationSpec for internal use.

        Returns:
            The AggregationSpec representation.
        """
        output_column = self._alias if self._alias else self._default_output_name()
        return AggregationSpec(
            function=self._function,
            source_column=self._column,
            window=self._window,
            output_column=output_column,
            offset=self._offset,
            params=self._params,
        )

    def _default_output_name(self) -> str:
        """Generate a default output column name."""
        if is_lifetime_window(self._window):
            window_suffix = "lifetime"
        else:
            window_suffix = self._window.replace(" ", "").lower()
        base_name = f"{self._column}_{self._function.value}_{window_suffix}"
        return base_name.upper()

    # Factory methods for creating features

    @classmethod
    def sum(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create a SUM aggregation feature.

        Args:
            column: The column to sum.
            window: The lookback window (e.g., "24h").
            offset: Offset to shift window into past (e.g., "7d" = previous week).
                Default is "0" (no offset).

        Returns:
            A Feature configured for SUM aggregation.

        Example::

            >>> amount_sum = Feature.sum("amount", "24h").alias("total_amount")
            >>> prev_week_sum = Feature.sum("amount", "7d", offset="7d").alias("prev_week")
        """
        return cls(AggregationType.SUM, column, window, offset)

    @classmethod
    def count(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create a COUNT aggregation feature.

        Args:
            column: The column to count.
            window: The lookback window (e.g., "7d").
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for COUNT aggregation.

        Example::

            >>> txn_count = Feature.count("transaction_id", "7d")
        """
        return cls(AggregationType.COUNT, column, window, offset)

    @classmethod
    def avg(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create an AVG aggregation feature.

        Args:
            column: The column to average.
            window: The lookback window.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for AVG aggregation.

        Example::

            >>> avg_amount = Feature.avg("amount", "24h")
        """
        return cls(AggregationType.AVG, column, window, offset)

    @classmethod
    def min(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create a MIN aggregation feature.

        Args:
            column: The column to find minimum of.
            window: The lookback window (e.g., "24h").
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for MIN aggregation.

        Example::

            >>> min_price = Feature.min("price", "24h")
        """
        return cls(AggregationType.MIN, column, window, offset)

    @classmethod
    def max(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create a MAX aggregation feature.

        Args:
            column: The column to find maximum of.
            window: The lookback window (e.g., "24h").
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for MAX aggregation.

        Example::

            >>> max_price = Feature.max("price", "24h")
        """
        return cls(AggregationType.MAX, column, window, offset)

    @classmethod
    def std(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create a STD (standard deviation) aggregation feature.

        Args:
            column: The column to compute standard deviation for.
            window: The lookback window.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for STD aggregation.

        Example::

            >>> price_std = Feature.std("price", "24h")
        """
        return cls(AggregationType.STD, column, window, offset)

    @classmethod
    def var(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create a VAR (variance) aggregation feature.

        Args:
            column: The column to compute variance for.
            window: The lookback window.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for VAR aggregation.

        Example::

            >>> price_var = Feature.var("price", "24h")
        """
        return cls(AggregationType.VAR, column, window, offset)

    @classmethod
    def last_n(cls, column: str, window: str, *, n: int, offset: str = "0") -> Feature:
        """Create a LAST_N aggregation feature.

        Collects the N most recent values within the window.

        Args:
            column: The column to collect values from.
            window: The lookback window.
            n: Number of values to collect.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for LAST_N aggregation.

        Example::

            >>> recent_pages = Feature.last_n("page_id", "1h", n=10)
        """
        return cls(AggregationType.LAST_N, column, window, offset, n=n)

    @classmethod
    def last_distinct_n(cls, column: str, window: str, *, n: int, offset: str = "0") -> Feature:
        """Create a LAST_DISTINCT_N aggregation feature.

        Collects the N most recent distinct values within the window.

        Args:
            column: The column to collect values from.
            window: The lookback window.
            n: Number of distinct values to collect.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for LAST_DISTINCT_N aggregation.

        Example::

            >>> recent_categories = Feature.last_distinct_n("category", "24h", n=5)
        """
        return cls(AggregationType.LAST_DISTINCT_N, column, window, offset, n=n)

    @classmethod
    def first_n(cls, column: str, window: str, *, n: int, offset: str = "0") -> Feature:
        """Create a FIRST_N aggregation feature.

        Collects the N oldest values within the window.

        Args:
            column: The column to collect values from.
            window: The lookback window.
            n: Number of values to collect.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for FIRST_N aggregation.

        Example::

            >>> first_pages = Feature.first_n("page_id", "1h", n=10)
        """
        return cls(AggregationType.FIRST_N, column, window, offset, n=n)

    @classmethod
    def first_distinct_n(cls, column: str, window: str, *, n: int, offset: str = "0") -> Feature:
        """Create a FIRST_DISTINCT_N aggregation feature.

        Collects the N oldest distinct values within the window.

        Args:
            column: The column to collect values from.
            window: The lookback window.
            n: Number of distinct values to collect.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for FIRST_DISTINCT_N aggregation.

        Example::

            >>> first_categories = Feature.first_distinct_n("category", "24h", n=5)
        """
        return cls(AggregationType.FIRST_DISTINCT_N, column, window, offset, n=n)

    @classmethod
    def approx_count_distinct(cls, column: str, window: str, offset: str = "0") -> Feature:
        """Create an APPROX_COUNT_DISTINCT aggregation feature.

        Estimates the number of distinct values using HyperLogLog algorithm.
        This is approximate but highly efficient for large datasets.

        Args:
            column: The column to count distinct values.
            window: The lookback window.
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for APPROX_COUNT_DISTINCT aggregation.

        Example::

            >>> unique_users = Feature.approx_count_distinct("user_id", "24h")
        """
        return cls(AggregationType.APPROX_COUNT_DISTINCT, column, window, offset)

    @classmethod
    def approx_percentile(cls, column: str, window: str, *, percentile: float = 0.5, offset: str = "0") -> Feature:
        """Create an APPROX_PERCENTILE aggregation feature.

        Estimates the specified percentile using T-Digest algorithm.
        This is approximate but highly efficient for large datasets.

        Args:
            column: The column to compute percentile for.
            window: The lookback window.
            percentile: The percentile to estimate (0.0 to 1.0). Default is 0.5 (median).
            offset: Offset to shift window into past. Default is "0" (no offset).

        Returns:
            A Feature configured for APPROX_PERCENTILE aggregation.

        Example::

            >>> median_amount = Feature.approx_percentile("amount", "24h", percentile=0.5)
            >>> p95_latency = Feature.approx_percentile("latency", "1h", percentile=0.95)
        """
        return cls(AggregationType.APPROX_PERCENTILE, column, window, offset, percentile=percentile)

    def __repr__(self) -> str:
        alias_str = f", alias='{self._alias}'" if self._alias else ""
        params_str = f", params={self._params}" if self._params else ""
        offset_str = f", offset='{self._offset}'" if self._offset != "0" else ""
        return (
            f"Feature({self._function.value}, column='{self._column}', "
            f"window='{self._window}'{offset_str}{params_str}{alias_str})"
        )
