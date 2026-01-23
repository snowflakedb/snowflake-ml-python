"""SQL generators for tile-based aggregations.

This module provides SQL generation for:
1. TilingSqlGenerator: Creates the DT query that computes partial aggregations (tiles)
2. MergingSqlGenerator: Creates the CTEs for merging tiles during dataset generation
"""

from __future__ import annotations

from snowflake.ml.feature_store.aggregation import (
    AggregationSpec,
    AggregationType,
    interval_to_seconds,
    parse_interval,
)

# Maximum number of elements to store in array columns to avoid 128MB limit
# Assuming ~1KB per value, 100,000 values ≈ 100MB (leaving buffer)
_MAX_ARRAY_ELEMENTS = 100000


class TilingSqlGenerator:
    """Generates SQL for creating tile Dynamic Tables.

    The tiling query:
    1. Computes TIME_SLICE to bucket rows into tiles
    2. Computes partial aggregations per (join_keys, tile_start)
    3. For simple aggregations: stores SUM/COUNT as scalars
    4. For list aggregations: stores pre-sorted arrays (ARRAY_AGG with ORDER BY)
    """

    def __init__(
        self,
        source_query: str,
        join_keys: list[str],
        timestamp_col: str,
        feature_granularity: str,
        features: list[AggregationSpec],
    ) -> None:
        """Initialize the TilingSqlGenerator.

        Args:
            source_query: The source query providing raw event data.
            join_keys: List of column names used as join keys (from entities).
            timestamp_col: The timestamp column name.
            feature_granularity: The tile interval (e.g., "1h", "1d").
            features: List of aggregation specifications.
        """
        self._source_query = source_query
        self._join_keys = join_keys
        self._timestamp_col = timestamp_col
        self._feature_granularity = feature_granularity
        self._features = features

        # Parse interval for SQL generation
        self._interval_value, self._interval_unit = parse_interval(feature_granularity)

        # Track if we have any lifetime features (need cumulative columns)
        self._has_lifetime_features = any(f.is_lifetime() for f in features)

    def generate(self) -> str:
        """Generate the complete tiling SQL query.

        Returns:
            SQL query for creating the tile Dynamic Table.
        """
        tile_columns = self._generate_tile_columns()
        # Join keys and timestamp_col are already properly formatted by SqlIdentifier
        join_keys_str = ", ".join(self._join_keys)
        ts_col = self._timestamp_col

        if not self._has_lifetime_features:
            # Simple case: no lifetime features, just partial aggregations
            query = f"""
SELECT
    {join_keys_str},
    TIME_SLICE({ts_col}, {self._interval_value}, '{self._interval_unit}', 'START') AS TILE_START,
    {', '.join(tile_columns)}
FROM ({self._source_query})
GROUP BY {join_keys_str}, TILE_START
"""
        else:
            # With lifetime features: need cumulative columns via window functions
            # Structure: SELECT *, cumulative_columns FROM (SELECT partial_columns GROUP BY)
            cumulative_columns = self._generate_cumulative_columns()

            query = f"""
SELECT
    base.*,
    {', '.join(cumulative_columns)}
FROM (
    SELECT
        {join_keys_str},
        TIME_SLICE({ts_col}, {self._interval_value}, '{self._interval_unit}', 'START') AS TILE_START,
        {', '.join(tile_columns)}
    FROM ({self._source_query})
    GROUP BY {join_keys_str}, TILE_START
) base
"""
        return query.strip()

    def _generate_tile_columns(self) -> list[str]:
        """Generate the tile column expressions for all features.

        All simple aggregations share base partial columns:
        - _PARTIAL_SUM_{col}: SUM(col) - used by SUM, AVG, STD, VAR
        - _PARTIAL_COUNT_{col}: COUNT(col) - used by COUNT, AVG, STD, VAR
        - _PARTIAL_SUM_SQ_{col}: SUM(col*col) - used by STD, VAR

        This allows maximum column reuse. For example, SUM(amount) + AVG(amount)
        only creates 2 columns, not 3.

        Returns:
            List of SQL column expressions for the tile table.
        """
        # Track unique tile columns by their full name to avoid duplicates
        seen_columns: set[str] = set()
        columns = []
        ts_col = self._timestamp_col

        for spec in self._features:
            src_col = spec.source_column

            if spec.function == AggregationType.SUM:
                # SUM needs _PARTIAL_SUM
                col_name = spec.get_tile_column_name("SUM")
                if col_name not in seen_columns:
                    columns.append(f"SUM({src_col}) AS {col_name}")
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.COUNT:
                # COUNT needs _PARTIAL_COUNT
                col_name = spec.get_tile_column_name("COUNT")
                if col_name not in seen_columns:
                    columns.append(f"COUNT({src_col}) AS {col_name}")
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.AVG:
                # AVG needs _PARTIAL_SUM and _PARTIAL_COUNT
                sum_col = spec.get_tile_column_name("SUM")
                count_col = spec.get_tile_column_name("COUNT")
                if sum_col not in seen_columns:
                    columns.append(f"SUM({src_col}) AS {sum_col}")
                    seen_columns.add(sum_col)
                if count_col not in seen_columns:
                    columns.append(f"COUNT({src_col}) AS {count_col}")
                    seen_columns.add(count_col)

            elif spec.function == AggregationType.MIN:
                # MIN needs _PARTIAL_MIN
                col_name = spec.get_tile_column_name("MIN")
                if col_name not in seen_columns:
                    columns.append(f"MIN({src_col}) AS {col_name}")
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.MAX:
                # MAX needs _PARTIAL_MAX
                col_name = spec.get_tile_column_name("MAX")
                if col_name not in seen_columns:
                    columns.append(f"MAX({src_col}) AS {col_name}")
                    seen_columns.add(col_name)

            elif spec.function in (AggregationType.STD, AggregationType.VAR):
                # STD/VAR need _PARTIAL_SUM, _PARTIAL_COUNT, and _PARTIAL_SUM_SQ
                sum_col = spec.get_tile_column_name("SUM")
                count_col = spec.get_tile_column_name("COUNT")
                sum_sq_col = spec.get_tile_column_name("SUM_SQ")
                if sum_col not in seen_columns:
                    columns.append(f"SUM({src_col}) AS {sum_col}")
                    seen_columns.add(sum_col)
                if count_col not in seen_columns:
                    columns.append(f"COUNT({src_col}) AS {count_col}")
                    seen_columns.add(count_col)
                if sum_sq_col not in seen_columns:
                    columns.append(f"SUM({src_col} * {src_col}) AS {sum_sq_col}")
                    seen_columns.add(sum_sq_col)

            elif spec.function == AggregationType.APPROX_COUNT_DISTINCT:
                # APPROX_COUNT_DISTINCT uses HLL (HyperLogLog) state
                col_name = spec.get_tile_column_name("HLL")
                if col_name not in seen_columns:
                    columns.append(f"HLL_EXPORT(HLL_ACCUMULATE({src_col})) AS {col_name}")
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.APPROX_PERCENTILE:
                # APPROX_PERCENTILE uses T-Digest state
                col_name = spec.get_tile_column_name("TDIGEST")
                if col_name not in seen_columns:
                    columns.append(f"APPROX_PERCENTILE_ACCUMULATE({src_col}) AS {col_name}")
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.LAST_N:
                col_name = spec.get_tile_column_name("LAST")
                if col_name not in seen_columns:
                    columns.append(
                        f"ARRAY_SLICE("
                        f"ARRAY_AGG({src_col}) WITHIN GROUP (ORDER BY {ts_col} DESC), "
                        f"0, {_MAX_ARRAY_ELEMENTS}) AS {col_name}"
                    )
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.LAST_DISTINCT_N:
                # Uses same tile column as LAST_N (dedup happens at merge time)
                col_name = spec.get_tile_column_name("LAST")
                if col_name not in seen_columns:
                    columns.append(
                        f"ARRAY_SLICE("
                        f"ARRAY_AGG({src_col}) WITHIN GROUP (ORDER BY {ts_col} DESC), "
                        f"0, {_MAX_ARRAY_ELEMENTS}) AS {col_name}"
                    )
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.FIRST_N:
                col_name = spec.get_tile_column_name("FIRST")
                if col_name not in seen_columns:
                    columns.append(
                        f"ARRAY_SLICE("
                        f"ARRAY_AGG({src_col}) WITHIN GROUP (ORDER BY {ts_col} ASC), "
                        f"0, {_MAX_ARRAY_ELEMENTS}) AS {col_name}"
                    )
                    seen_columns.add(col_name)

            elif spec.function == AggregationType.FIRST_DISTINCT_N:
                # Uses same tile column as FIRST_N (dedup happens at merge time)
                col_name = spec.get_tile_column_name("FIRST")
                if col_name not in seen_columns:
                    columns.append(
                        f"ARRAY_SLICE("
                        f"ARRAY_AGG({src_col}) WITHIN GROUP (ORDER BY {ts_col} ASC), "
                        f"0, {_MAX_ARRAY_ELEMENTS}) AS {col_name}"
                    )
                    seen_columns.add(col_name)

        return columns

    def _generate_cumulative_columns(self) -> list[str]:
        """Generate cumulative column expressions for lifetime aggregations.

        These columns use window functions to compute running totals per entity.
        They are computed over the partial columns from the inner GROUP BY.

        Returns:
            List of SQL column expressions for cumulative columns.
        """
        seen_columns: set[str] = set()
        columns = []
        join_keys_str = ", ".join(self._join_keys)

        # Only process lifetime features
        lifetime_features = [f for f in self._features if f.is_lifetime()]

        for spec in lifetime_features:
            if spec.function == AggregationType.SUM:
                # Cumulative SUM
                partial_col = spec.get_tile_column_name("SUM")
                cum_col = spec.get_cumulative_column_name("SUM")
                if cum_col not in seen_columns:
                    columns.append(
                        f"SUM({partial_col}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_col}"
                    )
                    seen_columns.add(cum_col)

            elif spec.function == AggregationType.COUNT:
                # Cumulative COUNT
                partial_col = spec.get_tile_column_name("COUNT")
                cum_col = spec.get_cumulative_column_name("COUNT")
                if cum_col not in seen_columns:
                    columns.append(
                        f"SUM({partial_col}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_col}"
                    )
                    seen_columns.add(cum_col)

            elif spec.function == AggregationType.AVG:
                # Cumulative AVG needs cumulative SUM and COUNT
                partial_sum = spec.get_tile_column_name("SUM")
                partial_count = spec.get_tile_column_name("COUNT")
                cum_sum = spec.get_cumulative_column_name("SUM")
                cum_count = spec.get_cumulative_column_name("COUNT")

                if cum_sum not in seen_columns:
                    columns.append(
                        f"SUM({partial_sum}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_sum}"
                    )
                    seen_columns.add(cum_sum)
                if cum_count not in seen_columns:
                    columns.append(
                        f"SUM({partial_count}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_count}"
                    )
                    seen_columns.add(cum_count)

            elif spec.function == AggregationType.MIN:
                # Cumulative MIN (running minimum)
                partial_col = spec.get_tile_column_name("MIN")
                cum_col = spec.get_cumulative_column_name("MIN")
                if cum_col not in seen_columns:
                    columns.append(
                        f"MIN({partial_col}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_col}"
                    )
                    seen_columns.add(cum_col)

            elif spec.function == AggregationType.MAX:
                # Cumulative MAX (running maximum)
                partial_col = spec.get_tile_column_name("MAX")
                cum_col = spec.get_cumulative_column_name("MAX")
                if cum_col not in seen_columns:
                    columns.append(
                        f"MAX({partial_col}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_col}"
                    )
                    seen_columns.add(cum_col)

            elif spec.function in (AggregationType.STD, AggregationType.VAR):
                # Cumulative STD/VAR needs cumulative SUM, COUNT, and SUM_SQ
                partial_sum = spec.get_tile_column_name("SUM")
                partial_count = spec.get_tile_column_name("COUNT")
                partial_sum_sq = spec.get_tile_column_name("SUM_SQ")
                cum_sum = spec.get_cumulative_column_name("SUM")
                cum_count = spec.get_cumulative_column_name("COUNT")
                cum_sum_sq = spec.get_cumulative_column_name("SUM_SQ")

                if cum_sum not in seen_columns:
                    columns.append(
                        f"SUM({partial_sum}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_sum}"
                    )
                    seen_columns.add(cum_sum)
                if cum_count not in seen_columns:
                    columns.append(
                        f"SUM({partial_count}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_count}"
                    )
                    seen_columns.add(cum_count)
                if cum_sum_sq not in seen_columns:
                    columns.append(
                        f"SUM({partial_sum_sq}) OVER ("
                        f"PARTITION BY {join_keys_str} "
                        f"ORDER BY TILE_START "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                        f") AS {cum_sum_sq}"
                    )
                    seen_columns.add(cum_sum_sq)

            # Note: APPROX_COUNT_DISTINCT (HLL_COMBINE) and APPROX_PERCENTILE (APPROX_PERCENTILE_COMBINE)
            # do NOT support cumulative window frames in Snowflake.
            # These will be handled at merge time by aggregating all tiles.

            # Note: FIRST_N, FIRST_DISTINCT_N, LAST_N, LAST_DISTINCT_N lifetime
            # are handled at merge time by scanning tiles, not via cumulative columns

        return columns


class MergingSqlGenerator:
    """Generates CTEs for merging tiles during dataset generation.

    The merging process:
    1. TILES_JOINED_FVi: Join tiles with spine, filtering by window and complete tiles only
    2. SIMPLE_MERGED_FVi: Aggregate simple features (SUM, COUNT, AVG)
    3. LIST_MERGED_FVi: Flatten and aggregate list features (LAST_N, etc.)
    4. FVi: Combine simple and list results
    """

    def __init__(
        self,
        tile_table: str,
        join_keys: list[str],
        timestamp_col: str,
        feature_granularity: str,
        features: list[AggregationSpec],
        spine_timestamp_col: str,
        fv_index: int,
    ) -> None:
        """Initialize the MergingSqlGenerator.

        Args:
            tile_table: Fully qualified name of the tile table.
            join_keys: List of join key column names.
            timestamp_col: The timestamp column from the feature view.
            feature_granularity: The tile interval.
            features: List of aggregation specifications.
            spine_timestamp_col: The timestamp column from the spine.
            fv_index: Index of this feature view (for CTE naming).
        """
        self._tile_table = tile_table
        self._join_keys = join_keys
        self._timestamp_col = timestamp_col
        self._feature_granularity = feature_granularity
        self._features = features
        self._spine_timestamp_col = spine_timestamp_col
        self._fv_index = fv_index

        # Separate lifetime from non-lifetime features
        self._lifetime_features = [f for f in features if f.is_lifetime()]
        self._non_lifetime_features = [f for f in features if not f.is_lifetime()]

        # Separate non-lifetime features by type (simple vs list)
        self._simple_features = [f for f in self._non_lifetime_features if f.function.is_simple()]
        self._list_features = [f for f in self._non_lifetime_features if f.function.is_list()]

        # Lifetime features are all simple (validation ensures only SUM, COUNT, AVG, MIN, MAX, STD, VAR)
        self._lifetime_simple_features = self._lifetime_features

        # Parse interval
        self._interval_value, self._interval_unit = parse_interval(feature_granularity)
        self._interval_seconds = interval_to_seconds(feature_granularity)

        # Calculate max window in tiles for filtering (only for non-lifetime features)
        if self._non_lifetime_features:
            # Max tiles needed is the max of (window + offset) across all non-lifetime features
            max_lookback_seconds = max(
                f.get_window_seconds() + f.get_offset_seconds() for f in self._non_lifetime_features
            )
            self._max_tiles_needed = (max_lookback_seconds + self._interval_seconds - 1) // self._interval_seconds
        else:
            self._max_tiles_needed = 0

    def generate_all_ctes(self) -> list[tuple[str, str]]:
        """Generate all CTEs needed for this feature view.

        The optimization flow:
        1. SPINE_BOUNDARY: Add truncated tile boundary to spine
        2. UNIQUE_BOUNDS: Get distinct (entity, boundary) pairs
        3. TILES_JOINED: Join tiles to unique boundaries (for non-lifetime features)
        4. SIMPLE_MERGED: Aggregate simple non-lifetime features per boundary
        5. LIST_MERGED: Aggregate list non-lifetime features per boundary
        6. LIFETIME_MERGED: ASOF JOIN for lifetime features (O(1) per boundary)
        7. LIFETIME_LIST_MERGED: Scan tiles for lifetime list features
        8. FV: Join back to spine to expand results

        Returns:
            List of (cte_name, cte_body) tuples.
        """
        ctes = []

        # CTE 1: Spine with tile boundary (for join-back)
        ctes.append(self._generate_spine_boundary_cte())

        # CTE 2: Unique boundaries (optimization - reduce aggregation work)
        ctes.append(self._generate_unique_boundaries_cte())

        # CTE 3: Join tiles with unique boundaries (only if we have non-lifetime features)
        if self._non_lifetime_features:
            ctes.append(self._generate_tiles_joined_cte())

        # CTE 4: Simple aggregations (if any non-lifetime)
        if self._simple_features:
            ctes.append(self._generate_simple_merged_cte())

        # CTE 5: List aggregations (if any non-lifetime)
        if self._list_features:
            ctes.append(self._generate_list_merged_cte())

        # CTE 6: Lifetime simple aggregations (using ASOF JOIN on cumulative columns)
        if self._lifetime_simple_features:
            ctes.append(self._generate_lifetime_merged_cte())

        # CTE 7: Combine all results and join back to spine
        ctes.append(self._generate_combined_cte())

        return ctes

    def _generate_spine_boundary_cte(self) -> tuple[str, str]:
        """Generate CTE that adds tile boundary to deduplicated spine.

        The tile boundary is the truncated timestamp that determines which
        complete tiles are visible. All spine rows with the same boundary
        will have identical feature values.

        Returns:
            Tuple of (cte_name, cte_body).
        """
        cte_name = f"SPINE_BOUNDARY_FV{self._fv_index}"

        # Quote column names to preserve case-sensitivity from spine dataframe
        # The spine_timestamp_col is passed as-is from the user (e.g., "query_ts")
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]
        quoted_spine_ts = f'"{self._spine_timestamp_col}"'

        # Select all columns plus the tile boundary
        all_cols = quoted_join_keys + [quoted_spine_ts]
        select_cols = ", ".join(all_cols)

        # DATE_TRUNC to tile granularity gives us the boundary
        # All timestamps in the same granularity window see the same complete tiles
        cte_body = f"""
    SELECT DISTINCT {select_cols},
           DATE_TRUNC('{self._interval_unit.lower()}', {quoted_spine_ts}) AS TILE_BOUNDARY
    FROM SPINE
"""
        return cte_name, cte_body.strip()

    def _generate_unique_boundaries_cte(self) -> tuple[str, str]:
        """Generate CTE with unique (entity, boundary) pairs.

        This is the key optimization: instead of computing features for each
        spine row, we compute once per unique boundary and join back.

        Returns:
            Tuple of (cte_name, cte_body).
        """
        cte_name = f"UNIQUE_BOUNDS_FV{self._fv_index}"

        # Quote column names to handle case-sensitivity
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]
        keys_str = ", ".join(quoted_join_keys)

        cte_body = f"""
    SELECT DISTINCT {keys_str}, TILE_BOUNDARY
    FROM SPINE_BOUNDARY_FV{self._fv_index}
"""
        return cte_name, cte_body.strip()

    def _generate_tiles_joined_cte(self) -> tuple[str, str]:
        """Generate the CTE that joins tiles with unique boundaries.

        This joins tiles to UNIQUE_BOUNDS (not full spine), which is much smaller
        when there are many spine rows per tile boundary.

        Returns:
            Tuple of (cte_name, cte_body) for the tiles joined CTE.
        """
        cte_name = f"TILES_JOINED_FV{self._fv_index}"

        # Quote column names for spine columns (case-sensitive)
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]

        # Tile table column names match the join keys (already SqlIdentifier-formatted)
        tile_join_keys = list(self._join_keys)
        tile_keys_str = ", ".join(tile_join_keys)

        # Join conditions: quoted spine columns to uppercase tile columns
        join_conditions = [f"UB.{qk} = TILES.{tk}" for qk, tk in zip(quoted_join_keys, tile_join_keys)]

        # Get all tile columns we need (deduplicated)
        tile_columns_set: set[str] = set()
        for spec in self._features:
            if spec.function == AggregationType.SUM:
                tile_columns_set.add(spec.get_tile_column_name("SUM"))
            elif spec.function == AggregationType.COUNT:
                tile_columns_set.add(spec.get_tile_column_name("COUNT"))
            elif spec.function == AggregationType.AVG:
                tile_columns_set.add(spec.get_tile_column_name("SUM"))
                tile_columns_set.add(spec.get_tile_column_name("COUNT"))
            elif spec.function == AggregationType.MIN:
                tile_columns_set.add(spec.get_tile_column_name("MIN"))
            elif spec.function == AggregationType.MAX:
                tile_columns_set.add(spec.get_tile_column_name("MAX"))
            elif spec.function in (AggregationType.STD, AggregationType.VAR):
                tile_columns_set.add(spec.get_tile_column_name("SUM"))
                tile_columns_set.add(spec.get_tile_column_name("COUNT"))
                tile_columns_set.add(spec.get_tile_column_name("SUM_SQ"))
            elif spec.function == AggregationType.APPROX_COUNT_DISTINCT:
                tile_columns_set.add(spec.get_tile_column_name("HLL"))
            elif spec.function == AggregationType.APPROX_PERCENTILE:
                tile_columns_set.add(spec.get_tile_column_name("TDIGEST"))
            elif spec.function in (AggregationType.LAST_N, AggregationType.LAST_DISTINCT_N):
                tile_columns_set.add(spec.get_tile_column_name("LAST"))
            elif spec.function in (AggregationType.FIRST_N, AggregationType.FIRST_DISTINCT_N):
                tile_columns_set.add(spec.get_tile_column_name("FIRST"))
        tile_columns = sorted(tile_columns_set)  # Sort for deterministic output

        tile_columns_str = ", ".join(f"TILES.{col}" for col in tile_columns)

        # Window filter: only include tiles within the max window and complete tiles
        # Complete tiles: tile_end <= tile_boundary (not spine timestamp)
        # tile_end = DATEADD(interval_unit, interval_value, tile_start)
        cte_body = f"""
    SELECT
        UB.*,
        TILES.TILE_START,
        {tile_columns_str}
    FROM UNIQUE_BOUNDS_FV{self._fv_index} UB
    LEFT JOIN (
        SELECT {tile_keys_str}, TILE_START, {', '.join(tile_columns)}
        FROM {self._tile_table}
    ) TILES
    ON {' AND '.join(join_conditions)}
    -- Window filter: tiles within max window from tile boundary
    AND TILES.TILE_START >= DATEADD(
        {self._interval_unit}, -{self._max_tiles_needed * self._interval_value}, UB.TILE_BOUNDARY
    )
    -- Complete tiles only: tile_end <= tile_boundary
    AND DATEADD({self._interval_unit}, {self._interval_value}, TILES.TILE_START) <= UB.TILE_BOUNDARY
"""
        return cte_name, cte_body.strip()

    def _get_tile_filter_condition(self, spec: AggregationSpec) -> str:
        """Generate the CASE WHEN condition for filtering tiles by window and offset.

        For a feature with window W and offset O, we want tiles where:
        - TILE_START >= TILE_BOUNDARY - W - O (start of window)
        - TILE_START < TILE_BOUNDARY - O (end of window, before offset)

        Args:
            spec: The aggregation specification.

        Returns:
            SQL condition string for use in CASE WHEN.
        """
        window_tiles = (spec.get_window_seconds() + self._interval_seconds - 1) // self._interval_seconds
        offset_tiles = spec.get_offset_seconds() // self._interval_seconds

        if offset_tiles == 0:
            # No offset: just filter by window start
            return (
                f"TILE_START >= DATEADD({self._interval_unit}, "
                f"-{window_tiles * self._interval_value}, TILE_BOUNDARY)"
            )
        else:
            # With offset: filter by both window start and end (shifted by offset)
            window_start_tiles = window_tiles + offset_tiles
            return (
                f"TILE_START >= DATEADD({self._interval_unit}, "
                f"-{window_start_tiles * self._interval_value}, TILE_BOUNDARY) "
                f"AND TILE_START < DATEADD({self._interval_unit}, "
                f"-{offset_tiles * self._interval_value}, TILE_BOUNDARY)"
            )

    def _get_list_tile_filter_condition(self, spec: AggregationSpec) -> str:
        """Generate filter condition for list aggregations (with table prefix).

        Similar to _get_tile_filter_condition but uses t. prefix for table references.

        Args:
            spec: The aggregation specification.

        Returns:
            SQL condition string for use in WHERE clause.
        """
        window_tiles = (spec.get_window_seconds() + self._interval_seconds - 1) // self._interval_seconds
        offset_tiles = spec.get_offset_seconds() // self._interval_seconds

        if offset_tiles == 0:
            # No offset: just filter by window start
            return (
                f"t.TILE_START >= DATEADD({self._interval_unit}, "
                f"-{window_tiles * self._interval_value}, t.TILE_BOUNDARY)"
            )
        else:
            # With offset: filter by both window start and end (shifted by offset)
            window_start_tiles = window_tiles + offset_tiles
            return (
                f"t.TILE_START >= DATEADD({self._interval_unit}, "
                f"-{window_start_tiles * self._interval_value}, t.TILE_BOUNDARY) "
                f"AND t.TILE_START < DATEADD({self._interval_unit}, "
                f"-{offset_tiles * self._interval_value}, t.TILE_BOUNDARY)"
            )

    def _generate_simple_merged_cte(self) -> tuple[str, str]:
        """Generate the CTE for simple aggregations (SUM, COUNT, AVG).

        Groups by entity keys + TILE_BOUNDARY (not spine timestamp) for efficiency.

        Returns:
            Tuple of (cte_name, cte_body) for the simple merged CTE.
        """
        cte_name = f"SIMPLE_MERGED_FV{self._fv_index}"

        # Quote column names for case-sensitivity (from UNIQUE_BOUNDS which inherits from spine)
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]
        # Group by entity keys + TILE_BOUNDARY (optimization)
        group_by_cols = quoted_join_keys + ["TILE_BOUNDARY"]
        group_by_str = ", ".join(group_by_cols)

        agg_columns = []
        for spec in self._simple_features:
            output_col = spec.get_sql_column_name()
            tile_filter = self._get_tile_filter_condition(spec)

            if spec.function == AggregationType.SUM:
                col_name = spec.get_tile_column_name("SUM")
                agg_columns.append(f"SUM(CASE WHEN {tile_filter} " f"THEN {col_name} ELSE 0 END) AS {output_col}")

            elif spec.function == AggregationType.COUNT:
                col_name = spec.get_tile_column_name("COUNT")
                agg_columns.append(f"SUM(CASE WHEN {tile_filter} " f"THEN {col_name} ELSE 0 END) AS {output_col}")

            elif spec.function == AggregationType.MIN:
                col_name = spec.get_tile_column_name("MIN")
                agg_columns.append(f"MIN(CASE WHEN {tile_filter} " f"THEN {col_name} ELSE NULL END) AS {output_col}")

            elif spec.function == AggregationType.MAX:
                col_name = spec.get_tile_column_name("MAX")
                agg_columns.append(f"MAX(CASE WHEN {tile_filter} " f"THEN {col_name} ELSE NULL END) AS {output_col}")

            elif spec.function == AggregationType.AVG:
                # AVG = SUM(partial_sums) / SUM(partial_counts)
                sum_col = spec.get_tile_column_name("SUM")
                count_col = spec.get_tile_column_name("COUNT")
                agg_columns.append(
                    f"CASE WHEN SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END) > 0 "
                    f"THEN SUM(CASE WHEN {tile_filter} "
                    f"THEN {sum_col} ELSE 0 END) / "
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END) "
                    f"ELSE NULL END AS {output_col}"
                )

            elif spec.function == AggregationType.VAR:
                # VAR = (SUM_SQ / COUNT) - (SUM / COUNT)^2
                # Using parallel variance formula
                # GREATEST(0, ...) clamps to non-negative to handle floating-point errors
                sum_col = spec.get_tile_column_name("SUM")
                count_col = spec.get_tile_column_name("COUNT")
                sum_sq_col = spec.get_tile_column_name("SUM_SQ")
                agg_columns.append(
                    f"CASE WHEN SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END) > 0 "
                    f"THEN GREATEST(0, ("
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {sum_sq_col} ELSE 0 END) / "
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END)"
                    f") - POWER("
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {sum_col} ELSE 0 END) / "
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END), 2)) "
                    f"ELSE NULL END AS {output_col}"
                )

            elif spec.function == AggregationType.STD:
                # STD = SQRT(VAR) = SQRT((SUM_SQ / COUNT) - (SUM / COUNT)^2)
                # GREATEST(0, ...) clamps variance to non-negative to handle floating-point errors
                # that can cause sqrt of tiny negative numbers like -4.54747e-13
                sum_col = spec.get_tile_column_name("SUM")
                count_col = spec.get_tile_column_name("COUNT")
                sum_sq_col = spec.get_tile_column_name("SUM_SQ")
                agg_columns.append(
                    f"CASE WHEN SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END) > 0 "
                    f"THEN SQRT(GREATEST(0, "
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {sum_sq_col} ELSE 0 END) / "
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END)"
                    f" - POWER("
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {sum_col} ELSE 0 END) / "
                    f"SUM(CASE WHEN {tile_filter} "
                    f"THEN {count_col} ELSE 0 END), 2))) "
                    f"ELSE NULL END AS {output_col}"
                )

            elif spec.function == AggregationType.APPROX_COUNT_DISTINCT:
                # Combine HLL states and estimate count
                # HLL_ESTIMATE(HLL_COMBINE(HLL_IMPORT(state))) gives the approximate count
                col_name = spec.get_tile_column_name("HLL")
                agg_columns.append(
                    f"HLL_ESTIMATE(HLL_COMBINE("
                    f"CASE WHEN {tile_filter} THEN HLL_IMPORT({col_name}) ELSE NULL END"
                    f")) AS {output_col}"
                )

            elif spec.function == AggregationType.APPROX_PERCENTILE:
                # Combine T-Digest states and estimate percentile
                # APPROX_PERCENTILE_ESTIMATE(APPROX_PERCENTILE_COMBINE(state), percentile)
                col_name = spec.get_tile_column_name("TDIGEST")
                percentile = spec.params.get("percentile", 0.5)
                agg_columns.append(
                    f"APPROX_PERCENTILE_ESTIMATE(APPROX_PERCENTILE_COMBINE("
                    f"CASE WHEN {tile_filter} THEN {col_name} ELSE NULL END"
                    f"), {percentile}) AS {output_col}"
                )

        cte_body = f"""
    SELECT
        {group_by_str},
        {', '.join(agg_columns)}
    FROM TILES_JOINED_FV{self._fv_index}
    GROUP BY {group_by_str}
"""
        return cte_name, cte_body.strip()

    def _generate_list_merged_cte(self) -> tuple[str, str]:
        """Generate the CTE for list aggregations using LATERAL FLATTEN."""
        cte_name = f"LIST_MERGED_FV{self._fv_index}"

        # Generate a more efficient single-pass CTE
        # Each list feature gets its own lateral flatten in the FROM clause
        cte_body = self._generate_list_cte_body()

        return cte_name, cte_body.strip()

    def _generate_lifetime_merged_cte(self) -> tuple[str, str]:
        """Generate the CTE for lifetime simple aggregations using ASOF JOIN.

        Uses ASOF JOIN on cumulative columns for O(1) lookup per boundary.
        This is much faster than aggregating all tiles from the beginning of time.

        Returns:
            Tuple of (cte_name, cte_body) for the lifetime merged CTE.
        """
        cte_name = f"LIFETIME_MERGED_FV{self._fv_index}"

        # Quote column names for case-sensitivity
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]
        group_by_cols = quoted_join_keys + ["TILE_BOUNDARY"]

        # Tile table column names match the join keys (already SqlIdentifier-formatted)
        tile_join_keys = list(self._join_keys)

        # Build ASOF JOIN condition: match the most recent tile before the boundary
        asof_match = "UB.TILE_BOUNDARY > TILES.TILE_START"
        join_conditions = [f"UB.{qk} = TILES.{tk}" for qk, tk in zip(quoted_join_keys, tile_join_keys)]
        join_conditions_str = " AND ".join(join_conditions)

        # Build select columns for lifetime features
        select_cols = []
        for spec in self._lifetime_simple_features:
            output_col = spec.get_sql_column_name()

            if spec.function == AggregationType.SUM:
                cum_col = spec.get_cumulative_column_name("SUM")
                select_cols.append(f"TILES.{cum_col} AS {output_col}")

            elif spec.function == AggregationType.COUNT:
                cum_col = spec.get_cumulative_column_name("COUNT")
                select_cols.append(f"TILES.{cum_col} AS {output_col}")

            elif spec.function == AggregationType.AVG:
                cum_sum = spec.get_cumulative_column_name("SUM")
                cum_count = spec.get_cumulative_column_name("COUNT")
                select_cols.append(
                    f"CASE WHEN TILES.{cum_count} > 0 "
                    f"THEN TILES.{cum_sum} / TILES.{cum_count} "
                    f"ELSE NULL END AS {output_col}"
                )

            elif spec.function == AggregationType.MIN:
                cum_col = spec.get_cumulative_column_name("MIN")
                select_cols.append(f"TILES.{cum_col} AS {output_col}")

            elif spec.function == AggregationType.MAX:
                cum_col = spec.get_cumulative_column_name("MAX")
                select_cols.append(f"TILES.{cum_col} AS {output_col}")

            elif spec.function == AggregationType.VAR:
                cum_sum = spec.get_cumulative_column_name("SUM")
                cum_count = spec.get_cumulative_column_name("COUNT")
                cum_sum_sq = spec.get_cumulative_column_name("SUM_SQ")
                select_cols.append(
                    f"CASE WHEN TILES.{cum_count} > 0 "
                    f"THEN GREATEST(0, "
                    f"TILES.{cum_sum_sq} / TILES.{cum_count} "
                    f"- POWER(TILES.{cum_sum} / TILES.{cum_count}, 2)) "
                    f"ELSE NULL END AS {output_col}"
                )

            elif spec.function == AggregationType.STD:
                cum_sum = spec.get_cumulative_column_name("SUM")
                cum_count = spec.get_cumulative_column_name("COUNT")
                cum_sum_sq = spec.get_cumulative_column_name("SUM_SQ")
                select_cols.append(
                    f"CASE WHEN TILES.{cum_count} > 0 "
                    f"THEN SQRT(GREATEST(0, "
                    f"TILES.{cum_sum_sq} / TILES.{cum_count} "
                    f"- POWER(TILES.{cum_sum} / TILES.{cum_count}, 2))) "
                    f"ELSE NULL END AS {output_col}"
                )

            elif spec.function == AggregationType.APPROX_COUNT_DISTINCT:
                cum_col = spec.get_cumulative_column_name("HLL")
                select_cols.append(f"HLL_ESTIMATE(HLL_IMPORT(TILES.{cum_col})) AS {output_col}")

            elif spec.function == AggregationType.APPROX_PERCENTILE:
                cum_col = spec.get_cumulative_column_name("TDIGEST")
                percentile = spec.params.get("percentile", 0.5)
                select_cols.append(f"APPROX_PERCENTILE_ESTIMATE(TILES.{cum_col}, {percentile}) AS {output_col}")

        select_cols_str = ", ".join(select_cols)

        # Qualify group by columns with UB alias to avoid ambiguity in ASOF JOIN
        qualified_group_cols = [f"UB.{col}" for col in group_by_cols]
        qualified_group_str = ", ".join(qualified_group_cols)

        cte_body = f"""
    SELECT
        {qualified_group_str},
        {select_cols_str}
    FROM UNIQUE_BOUNDS_FV{self._fv_index} UB
    ASOF JOIN {self._tile_table} TILES
      MATCH_CONDITION ({asof_match})
      ON {join_conditions_str}
"""
        return cte_name, cte_body.strip()

    def _generate_list_cte_body(self) -> str:
        """Generate the body of the list merged CTE.

        Groups by entity keys + TILE_BOUNDARY (not spine timestamp) for efficiency.

        Returns:
            SQL string for the list merged CTE body.
        """
        # Quote column names for case-sensitivity (from UNIQUE_BOUNDS which inherits from spine)
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]
        # Group by entity keys + TILE_BOUNDARY (optimization)
        group_by_cols = quoted_join_keys + ["TILE_BOUNDARY"]
        group_by_str = ", ".join(group_by_cols)
        select_group_cols = ", ".join(f"t.{col}" for col in group_by_cols)

        if not self._list_features:
            return f"SELECT {group_by_str} FROM TILES_JOINED_FV{self._fv_index} WHERE 1=0"

        # Build subqueries for each list feature
        feature_subqueries = []

        for _idx, spec in enumerate(self._list_features):
            # Determine the tile column name based on aggregation type
            if spec.function in (AggregationType.LAST_N, AggregationType.LAST_DISTINCT_N):
                col_name = spec.get_tile_column_name("LAST")
                order_clause = "t.TILE_START DESC, flat.INDEX ASC"
            else:  # FIRST_N, FIRST_DISTINCT_N
                col_name = spec.get_tile_column_name("FIRST")
                order_clause = "t.TILE_START ASC, flat.INDEX ASC"

            output_col = spec.get_sql_column_name()
            n_value = spec.params["n"]

            # Calculate tile filter condition for window and offset
            tile_filter = self._get_list_tile_filter_condition(spec)

            is_distinct = spec.function in (AggregationType.LAST_DISTINCT_N, AggregationType.FIRST_DISTINCT_N)

            if is_distinct:
                # For distinct, use QUALIFY to keep first occurrence of each value
                # Note: Inner query uses t. prefix, outer query uses bare column names
                subquery = f"""
    (SELECT {group_by_str},
            ARRAY_AGG(val) WITHIN GROUP (ORDER BY rn) AS {output_col}
     FROM (
         SELECT {select_group_cols}, flat.VALUE AS val,
                ROW_NUMBER() OVER (PARTITION BY {select_group_cols} ORDER BY {order_clause}) AS rn,
                ROW_NUMBER() OVER (PARTITION BY {select_group_cols}, flat.VALUE ORDER BY {order_clause}) AS dup_rn
         FROM TILES_JOINED_FV{self._fv_index} t,
         LATERAL FLATTEN(INPUT => t.{col_name}) flat
         WHERE {tile_filter}
         AND flat.VALUE IS NOT NULL
     ) ranked
     WHERE dup_rn = 1 AND rn <= {n_value}
     GROUP BY {group_by_str}
    )"""
            else:
                # Non-distinct: straightforward flatten and aggregate
                subquery = f"""
    (SELECT {select_group_cols},
            ARRAY_SLICE(
                ARRAY_AGG(flat.VALUE) WITHIN GROUP (ORDER BY {order_clause}),
                0, {n_value}
            ) AS {output_col}
     FROM TILES_JOINED_FV{self._fv_index} t,
     LATERAL FLATTEN(INPUT => t.{col_name}) flat
     WHERE {tile_filter}
     AND flat.VALUE IS NOT NULL
     GROUP BY {select_group_cols}
    )"""

            feature_subqueries.append((output_col, subquery))

        # Combine all subqueries with JOINs
        if len(feature_subqueries) == 1:
            return feature_subqueries[0][1]

        # Multiple list features: join them together
        first_name, first_query = feature_subqueries[0]
        result = "SELECT sq0.*, "
        for i, (name, _) in enumerate(feature_subqueries[1:], 1):
            result += f"sq{i}.{name}"
            if i < len(feature_subqueries) - 1:
                result += ", "

        result += f"\n    FROM {first_query} sq0"

        for i, (_name, query) in enumerate(feature_subqueries[1:], 1):
            join_cond = " AND ".join(f"sq0.{col} = sq{i}.{col}" for col in group_by_cols)
            result += f"\n    LEFT JOIN {query} sq{i}\n    ON {join_cond}"

        return result

    def _generate_combined_cte(self) -> tuple[str, str]:
        """Generate the final CTE that combines and expands results.

        This joins the merged results (grouped by entity + TILE_BOUNDARY) back to
        the original spine (SPINE_BOUNDARY) to expand to per-spine-row output.

        Returns:
            Tuple of (cte_name, cte_body) for the combined CTE.
        """
        cte_name = f"FV{self._fv_index:03d}"

        # Quote column names for case-sensitivity
        quoted_join_keys = [f'"{k}"' for k in self._join_keys]
        quoted_spine_ts = f'"{self._spine_timestamp_col}"'
        spine_output_cols = quoted_join_keys + [quoted_spine_ts]
        # Merged results are grouped by entity + TILE_BOUNDARY
        boundary_group_cols = quoted_join_keys + ["TILE_BOUNDARY"]

        # Select columns from spine (entity keys + original timestamp)
        select_cols = [f"s.{col}" for col in spine_output_cols]

        # Add simple feature columns (non-lifetime)
        for spec in self._simple_features:
            select_cols.append(f"simple.{spec.get_sql_column_name()}")

        # Add list feature columns (non-lifetime)
        for spec in self._list_features:
            select_cols.append(f"list_agg.{spec.get_sql_column_name()}")

        # Add lifetime simple feature columns
        for spec in self._lifetime_simple_features:
            select_cols.append(f"lifetime.{spec.get_sql_column_name()}")

        # Build FROM clause with all necessary JOINs
        from_clause = f"SPINE_BOUNDARY_FV{self._fv_index} s"
        joins = []

        # Join condition template
        def make_join_cond(alias: str) -> str:
            return " AND ".join(f"s.{col} = {alias}.{col}" for col in boundary_group_cols)

        # Add JOINs for each CTE that has features
        if self._simple_features:
            joins.append(f"LEFT JOIN SIMPLE_MERGED_FV{self._fv_index} simple ON {make_join_cond('simple')}")

        if self._list_features:
            joins.append(f"LEFT JOIN LIST_MERGED_FV{self._fv_index} list_agg ON {make_join_cond('list_agg')}")

        if self._lifetime_simple_features:
            joins.append(f"LEFT JOIN LIFETIME_MERGED_FV{self._fv_index} lifetime ON {make_join_cond('lifetime')}")

        if joins:
            cte_body = f"""
    SELECT {', '.join(select_cols)}
    FROM {from_clause}
    {chr(10).join('    ' + j for j in joins)}
"""
        else:
            # No features (shouldn't happen)
            cte_body = f"""
    SELECT DISTINCT {', '.join(spine_output_cols)}
    FROM SPINE
"""

        return cte_name, cte_body.strip()

    def get_output_columns(self) -> list[str]:
        """Get the list of output column names from this feature view."""
        return [spec.output_column for spec in self._features]
