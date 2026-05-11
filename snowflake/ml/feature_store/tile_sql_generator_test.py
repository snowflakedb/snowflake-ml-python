"""Unit tests for tile_sql_generator module."""

from absl.testing import absltest

from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
from snowflake.ml.feature_store.feature_view import _prepend_keys_specs
from snowflake.ml.feature_store.tile_sql_generator import (
    MergingSqlGenerator,
    RollupSqlGenerator,
    TilingSqlGenerator,
    _generate_cumulative_expressions,
)


class TilingSqlGeneratorTest(absltest.TestCase):
    """Unit tests for TilingSqlGenerator class."""

    def test_simple_sum_aggregation(self) -> None:
        """Test generating tiling SQL for SUM aggregation."""
        # Use uppercase identifiers to match real usage (SqlIdentifier normalizes to uppercase)
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        self.assertIn("TIME_SLICE", sql)
        self.assertIn("TILE_START", sql)
        self.assertIn("SUM(AMOUNT)", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("USER_ID", sql)

    def test_count_aggregation(self) -> None:
        """Test generating tiling SQL for COUNT aggregation."""
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="TRANSACTION_ID",
                window="7d",
                output_column="TXN_COUNT",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM transactions",
            join_keys=["USER_ID"],
            timestamp_col="TS",
            feature_granularity="1d",
            features=features,
        )
        sql = generator.generate()

        self.assertIn("COUNT(TRANSACTION_ID)", sql)

    def test_avg_aggregation(self) -> None:
        """Test generating tiling SQL for AVG aggregation (stores SUM and COUNT)."""
        features = [
            AggregationSpec(
                function=AggregationType.AVG,
                source_column="PRICE",
                window="24h",
                output_column="AVG_PRICE",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM orders",
            join_keys=["USER_ID"],
            timestamp_col="ORDER_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        # AVG stores both SUM and COUNT for proper merging
        self.assertIn("SUM(PRICE)", sql)
        self.assertIn("COUNT(PRICE)", sql)
        self.assertIn("_SUM", sql)
        self.assertIn("_COUNT", sql)

    def test_std_aggregation(self) -> None:
        """Test generating tiling SQL for STD aggregation (stores SUM, COUNT, and SUM_SQ)."""
        features = [
            AggregationSpec(
                function=AggregationType.STD,
                source_column="PRICE",
                window="24h",
                output_column="PRICE_STD",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM orders",
            join_keys=["USER_ID"],
            timestamp_col="ORDER_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        # STD stores SUM, COUNT, and SUM of squares for parallel algorithm
        self.assertIn("SUM(PRICE)", sql)
        self.assertIn("COUNT(PRICE)", sql)
        self.assertIn("PRICE * PRICE", sql)
        self.assertIn("_SUM_SQ", sql)

    def test_var_aggregation(self) -> None:
        """Test generating tiling SQL for VAR aggregation (stores SUM, COUNT, and SUM_SQ)."""
        features = [
            AggregationSpec(
                function=AggregationType.VAR,
                source_column="PRICE",
                window="24h",
                output_column="PRICE_VAR",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM orders",
            join_keys=["USER_ID"],
            timestamp_col="ORDER_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        # VAR stores SUM, COUNT, and SUM of squares for parallel algorithm
        self.assertIn("SUM(PRICE)", sql)
        self.assertIn("COUNT(PRICE)", sql)
        self.assertIn("PRICE * PRICE", sql)
        self.assertIn("_SUM_SQ", sql)

    def test_last_n_aggregation(self) -> None:
        """Test generating tiling SQL for LAST_N aggregation."""
        features = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="RECENT_PAGES",
                params={"n": 10},
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM page_views",
            join_keys=["USER_ID"],
            timestamp_col="VIEW_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        self.assertIn("ARRAY_AGG", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("DESC", sql)
        self.assertIn("ARRAY_SLICE", sql)
        # Companion timestamp array for rollup ordering
        self.assertIn("_PARTIAL_LAST_TS_PAGE_ID", sql)

    def test_first_n_aggregation(self) -> None:
        """Test generating tiling SQL for FIRST_N aggregation."""
        features = [
            AggregationSpec(
                function=AggregationType.FIRST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="FIRST_PAGES",
                params={"n": 10},
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM page_views",
            join_keys=["USER_ID"],
            timestamp_col="VIEW_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        self.assertIn("ARRAY_AGG", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("ASC", sql)
        # Companion timestamp array for rollup ordering
        self.assertIn("_PARTIAL_FIRST_TS_PAGE_ID", sql)

    def test_last_n_companion_ts_array(self) -> None:
        """Test that LAST_N tile generation includes a companion timestamp array."""
        features = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        # Value array
        self.assertIn("_PARTIAL_LAST_ORDER_VALUE", sql)
        # Companion TS array — aligned with value array for rollup ordering
        self.assertIn("_PARTIAL_LAST_TS_ORDER_VALUE", sql)
        # Both should use ARRAY_AGG with ORDER BY DESC
        self.assertEqual(sql.count("ORDER BY EVENT_TS DESC"), 2)

    def test_multiple_join_keys(self) -> None:
        """Test generating tiling SQL with multiple join keys."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID", "SESSION_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        self.assertIn("USER_ID", sql)
        self.assertIn("SESSION_ID", sql)

    def test_multiple_features(self) -> None:
        """Test generating tiling SQL with multiple features."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="TRANSACTION_ID",
                window="24h",
                output_column="TXN_COUNT",
            ),
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="RECENT_PAGES",
                params={"n": 5},
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID"],
            timestamp_col="TS",
            feature_granularity="1h",
            features=features,
        )
        sql = generator.generate()

        self.assertIn("SUM", sql)
        self.assertIn("COUNT", sql)
        self.assertIn("ARRAY_AGG", sql)


class MergingSqlGeneratorTest(absltest.TestCase):
    """Unit tests for MergingSqlGenerator class."""

    def test_simple_aggregation_ctes(self) -> None:
        """Test generating merging CTEs for simple aggregations."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        # Should have: SPINE_BOUNDARY, UNIQUE_BOUNDS, TILES_JOINED, SIMPLE_MERGED, FV
        cte_names = [cte[0] for cte in ctes]
        self.assertIn("SPINE_BOUNDARY_FV0", cte_names)
        self.assertIn("UNIQUE_BOUNDS_FV0", cte_names)
        self.assertIn("TILES_JOINED_FV0", cte_names)
        self.assertIn("SIMPLE_MERGED_FV0", cte_names)
        self.assertIn("FV000", cte_names)

    def test_list_aggregation_ctes(self) -> None:
        """Test generating merging CTEs for list aggregations."""
        features = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="RECENT_PAGES",
                params={"n": 10},
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        # Should have: SPINE_BOUNDARY, UNIQUE_BOUNDS, TILES_JOINED, LIST_MERGED, FV
        cte_names = [cte[0] for cte in ctes]
        self.assertIn("SPINE_BOUNDARY_FV0", cte_names)
        self.assertIn("UNIQUE_BOUNDS_FV0", cte_names)
        self.assertIn("TILES_JOINED_FV0", cte_names)
        self.assertIn("LIST_MERGED_FV0", cte_names)
        self.assertIn("FV000", cte_names)

    def test_mixed_aggregation_ctes(self) -> None:
        """Test generating merging CTEs for mixed aggregations."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="RECENT_PAGES",
                params={"n": 10},
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        # Should have all CTEs
        cte_names = [cte[0] for cte in ctes]
        self.assertIn("SPINE_BOUNDARY_FV0", cte_names)
        self.assertIn("UNIQUE_BOUNDS_FV0", cte_names)
        self.assertIn("TILES_JOINED_FV0", cte_names)
        self.assertIn("SIMPLE_MERGED_FV0", cte_names)
        self.assertIn("LIST_MERGED_FV0", cte_names)
        self.assertIn("FV000", cte_names)

    def test_tiles_joined_cte_content(self) -> None:
        """Test the content of TILES_JOINED CTE."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        tiles_joined_cte = next(cte for cte in ctes if cte[0] == "TILES_JOINED_FV0")
        cte_body = tiles_joined_cte[1]

        # Check for key elements - now joins to UNIQUE_BOUNDS instead of SPINE_DEDUP
        self.assertIn("UNIQUE_BOUNDS_FV0", cte_body)
        self.assertIn("LEFT JOIN", cte_body)
        self.assertIn("TILE_START", cte_body)
        self.assertIn("TILE_BOUNDARY", cte_body)
        self.assertIn("DATEADD", cte_body)
        # Complete tiles check
        self.assertIn("DATEADD(HOUR,", cte_body)

    def test_spine_boundary_cte_content(self) -> None:
        """Test the content of SPINE_BOUNDARY CTE."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        spine_boundary_cte = next(cte for cte in ctes if cte[0] == "SPINE_BOUNDARY_FV0")
        cte_body = spine_boundary_cte[1]

        # Check for DATE_TRUNC to create tile boundary
        self.assertIn("DATE_TRUNC", cte_body)
        self.assertIn("TILE_BOUNDARY", cte_body)
        self.assertIn("FROM SPINE", cte_body)
        self.assertIn("query_ts", cte_body)

    def test_unique_bounds_cte_content(self) -> None:
        """Test the content of UNIQUE_BOUNDS CTE."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        unique_bounds_cte = next(cte for cte in ctes if cte[0] == "UNIQUE_BOUNDS_FV0")
        cte_body = unique_bounds_cte[1]

        # Check for DISTINCT on entity keys and boundary
        self.assertIn("DISTINCT", cte_body)
        self.assertIn("USER_ID", cte_body)
        self.assertIn("TILE_BOUNDARY", cte_body)
        self.assertIn("FROM SPINE_BOUNDARY_FV0", cte_body)

    def test_get_output_columns(self) -> None:
        """Test get_output_columns method."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            ),
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="RECENT_PAGES",
                params={"n": 10},
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=0,
        )

        output_cols = generator.get_output_columns()
        self.assertEqual(output_cols, ["AMOUNT_SUM_24H", "RECENT_PAGES"])

    def test_different_fv_index(self) -> None:
        """Test CTE naming with different fv_index."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col="query_ts",
            fv_index=2,
        )
        ctes = generator.generate_all_ctes()

        cte_names = [cte[0] for cte in ctes]
        self.assertIn("SPINE_BOUNDARY_FV2", cte_names)
        self.assertIn("UNIQUE_BOUNDS_FV2", cte_names)
        self.assertIn("TILES_JOINED_FV2", cte_names)
        self.assertIn("SIMPLE_MERGED_FV2", cte_names)
        self.assertIn("FV002", cte_names)

    def test_unicode_join_keys_quoted_correctly(self) -> None:
        """Test that already-quoted Unicode join keys are not double-quoted."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = MergingSqlGenerator(
            tile_table="DB.SCHEMA.USER_TILES",
            join_keys=['"顧客ID"'],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=features,
            spine_timestamp_col='"記録時刻"',
            fv_index=0,
        )
        ctes = generator.generate_all_ctes()

        spine_boundary = next(cte for cte in ctes if cte[0] == "SPINE_BOUNDARY_FV0")
        unique_bounds = next(cte for cte in ctes if cte[0] == "UNIQUE_BOUNDS_FV0")
        combined = next(cte for cte in ctes if cte[0] == "FV000")

        for cte_body in [spine_boundary[1], unique_bounds[1], combined[1]]:
            self.assertIn('"顧客ID"', cte_body)
            self.assertNotIn('""顧客ID""', cte_body, "Double-quoted Unicode join key found")

        self.assertIn('"記録時刻"', spine_boundary[1])
        self.assertNotIn('""記録時刻""', spine_boundary[1], "Double-quoted Unicode timestamp found")


class RollupSqlGeneratorTest(absltest.TestCase):
    """Unit tests for RollupSqlGenerator class."""

    def test_simple_rollup_no_cte(self) -> None:
        """Test that simple aggregations use direct SELECT...GROUP BY (no CTE)."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="ORDER_TOTAL",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # Should NOT use CTEs for simple-only aggregations
        self.assertNotIn("WITH", sql)
        self.assertIn("SUM(t._PARTIAL_COUNT_VISITOR_ID)", sql)
        self.assertIn("SUM(t._PARTIAL_SUM_ORDER_VALUE)", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("m.SUBSCRIBER_ID", sql)

    def test_list_rollup_uses_cte_with_lateral_flatten(self) -> None:
        """Test that list aggregations use CTE + LATERAL FLATTEN for ordering."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # Should use CTE-based approach
        self.assertIn("WITH", sql)
        self.assertIn("base AS", sql)
        self.assertIn("list_rollup_0 AS", sql)
        # Should use LATERAL FLATTEN instead of ARRAY_UNION_AGG
        self.assertIn("LATERAL FLATTEN", sql)
        self.assertNotIn("ARRAY_UNION_AGG", sql)
        # Should order by companion TS column
        self.assertIn("_PARTIAL_LAST_TS_ORDER_VALUE", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("DESC", sql)

    def test_first_n_rollup_uses_asc_ordering(self) -> None:
        """Test that FIRST_N rollup uses ASC ordering."""
        specs = [
            AggregationSpec(
                function=AggregationType.FIRST_N,
                source_column="PRODUCT_ID",
                window="24h",
                output_column="FIRST_PRODUCTS",
                params={"n": 3},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertIn("LATERAL FLATTEN", sql)
        self.assertIn("_PARTIAL_FIRST_TS_PRODUCT_ID", sql)
        self.assertIn("ASC", sql)

    def test_mixed_simple_and_list_rollup(self) -> None:
        """Test rollup with both simple and list aggregations."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="ORDER_TOTAL",
            ),
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # Should use CTE approach (due to list features)
        self.assertIn("WITH", sql)
        self.assertIn("base AS", sql)
        self.assertIn("simple_rollup AS", sql)
        self.assertIn("list_rollup_0 AS", sql)
        # Simple aggs use GROUP BY in simple_rollup CTE
        self.assertIn("SUM(_PARTIAL_COUNT_VISITOR_ID)", sql)
        self.assertIn("SUM(_PARTIAL_SUM_ORDER_VALUE)", sql)
        # List aggs use LATERAL FLATTEN
        self.assertIn("LATERAL FLATTEN", sql)
        self.assertNotIn("ARRAY_UNION_AGG", sql)
        # Final SELECT joins simple_rollup with list_rollup
        self.assertIn("LEFT JOIN list_rollup_0", sql)

    def test_multiple_list_columns_rollup(self) -> None:
        """Test rollup with multiple distinct list features on different columns."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
            AggregationSpec(
                function=AggregationType.FIRST_N,
                source_column="PRODUCT_ID",
                window="24h",
                output_column="FIRST_PRODUCTS",
                params={"n": 3},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # Should have two separate list_rollup CTEs
        self.assertIn("list_rollup_0 AS", sql)
        self.assertIn("list_rollup_1 AS", sql)
        # Both should use LATERAL FLATTEN
        self.assertEqual(sql.count("LATERAL FLATTEN"), 2)
        # One DESC for LAST_N, one ASC for FIRST_N
        self.assertIn("DESC", sql)
        self.assertIn("ASC", sql)

    def test_dedup_shared_tile_columns_in_rollup(self) -> None:
        """Test that LAST_N and LAST_DISTINCT_N on same column share one CTE."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
            AggregationSpec(
                function=AggregationType.LAST_DISTINCT_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_DISTINCT_ORDERS",
                params={"n": 3},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # Both share _PARTIAL_LAST_ORDER_VALUE, so only one list_rollup CTE
        self.assertIn("list_rollup_0 AS", sql)
        self.assertNotIn("list_rollup_1", sql)
        # Only one LATERAL FLATTEN
        self.assertEqual(sql.count("LATERAL FLATTEN"), 1)

    def test_rollup_base_cte_includes_ts_columns(self) -> None:
        """Test that the base CTE selects companion TS columns."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # base CTE should select both value and TS columns from tile table
        self.assertIn("t._PARTIAL_LAST_ORDER_VALUE", sql)
        self.assertIn("t._PARTIAL_LAST_TS_ORDER_VALUE", sql)

    def test_rollup_output_includes_ts_columns(self) -> None:
        """Test that rollup output preserves TS columns for downstream use."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        # Final SELECT should include both value and TS columns
        # Find the final SELECT (after all CTEs)
        self.assertIn("l0._PARTIAL_LAST_ORDER_VALUE", sql)
        self.assertIn("l0._PARTIAL_LAST_TS_ORDER_VALUE", sql)


class TemporalRollupSqlGeneratorTest(absltest.TestCase):
    """Unit tests for temporal (range JOIN) rollup SQL generation."""

    def test_temporal_rollup_uses_range_join(self) -> None:
        """Test that temporal columns produce a range JOIN with validity window."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="EFFECTIVE_TS",
            mapping_valid_to_col="EXPIRY_TS",
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertNotIn("MATCH_CONDITION", sql)
        self.assertIn("JOIN", sql)
        self.assertIn("t.TILE_START >= m.EFFECTIVE_TS", sql)
        self.assertIn("m.EXPIRY_TS IS NULL OR t.TILE_START < m.EXPIRY_TS", sql)

    def test_temporal_rollup_without_effective_ts_uses_flat_join(self) -> None:
        """Test that omitting mapping_valid_from_col produces flat JOIN."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertNotIn("MATCH_CONDITION", sql)
        self.assertIn("JOIN", sql)

    def test_temporal_rollup_with_list_features_uses_range_join_in_base_cte(self) -> None:
        """Test that range JOIN is used in the base CTE for list features."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="EFFECTIVE_TS",
            mapping_valid_to_col="EXPIRY_TS",
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertIn("t.TILE_START >= m.EFFECTIVE_TS", sql)
        self.assertIn("m.EXPIRY_TS IS NULL OR t.TILE_START < m.EXPIRY_TS", sql)
        self.assertIn("LATERAL FLATTEN", sql)

    def test_temporal_rollup_mixed_simple_and_list(self) -> None:
        """Test temporal rollup with both simple and list aggregations."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="EFFECTIVE_TS",
            mapping_valid_to_col="EXPIRY_TS",
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertIn("t.TILE_START >= m.EFFECTIVE_TS", sql)
        self.assertIn("simple_rollup AS", sql)
        self.assertIn("list_rollup_0 AS", sql)
        self.assertIn("LATERAL FLATTEN", sql)

    def test_temporal_rollup_range_join_condition_format(self) -> None:
        """Test the exact range JOIN format with all conditions in ON clause."""
        specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="ORDER_TOTAL",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="EFF_TS",
            mapping_valid_to_col="EXP_TS",
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertNotIn("MATCH_CONDITION", sql)
        self.assertIn("ON t.VISITOR_ID = m.VISITOR_ID", sql)
        self.assertIn("t.TILE_START >= m.EFF_TS", sql)
        self.assertIn("m.EXP_TS IS NULL OR t.TILE_START < m.EXP_TS", sql)

    def test_generate_as_cte(self) -> None:
        """Test generate_as_cte returns proper (name, body) tuple."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="EFFECTIVE_TS",
            mapping_valid_to_col="EXPIRY_TS",
        )
        cte_name, cte_body = generator.generate_as_cte("PIT_ROLLUP_FV0")

        self.assertEqual(cte_name, "PIT_ROLLUP_FV0")
        self.assertNotIn("ASOF JOIN", cte_body)
        self.assertIn("t.TILE_START >= m.EFFECTIVE_TS", cte_body)
        self.assertIn("SUM(t._PARTIAL_COUNT_VISITOR_ID)", cte_body)
        self.assertNotIn("WITH", cte_body)

    def test_temporal_rollup_preserves_aggregation_logic(self) -> None:
        """Test that temporal rollup uses the same aggregation expressions as flat rollup."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="ORDER_TOTAL",
            ),
            AggregationSpec(
                function=AggregationType.AVG,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="ORDER_AVG",
            ),
        ]

        flat_gen = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        temporal_gen = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="EFFECTIVE_TS",
            mapping_valid_to_col="EXPIRY_TS",
        )

        flat_sql = flat_gen.generate()
        temporal_sql = temporal_gen.generate()

        for agg_expr in [
            "SUM(t._PARTIAL_COUNT_VISITOR_ID)",
            "SUM(t._PARTIAL_SUM_ORDER_VALUE)",
        ]:
            self.assertIn(agg_expr, flat_sql)
            self.assertIn(agg_expr, temporal_sql)

        # Only temporal should have range JOIN conditions
        self.assertNotIn("EFFECTIVE_TS", flat_sql)
        self.assertIn("t.TILE_START >= m.EFFECTIVE_TS", temporal_sql)

    def test_range_join_conditions_in_on_clause_simple(self) -> None:
        """Test that valid_from and valid_to are in the JOIN ON clause, not WHERE."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="VALID_FROM",
            mapping_valid_to_col="VALID_TO",
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        # Validity conditions should be in the ON clause
        on_clause_idx = sql.index("ON ")
        where_clause_idx = sql.index("WHERE ")
        valid_from_idx = sql.index("t.TILE_START >= m.VALID_FROM")
        valid_to_idx = sql.index("m.VALID_TO IS NULL OR t.TILE_START < m.VALID_TO")
        self.assertGreater(valid_from_idx, on_clause_idx)
        self.assertLess(valid_from_idx, where_clause_idx)
        self.assertGreater(valid_to_idx, on_clause_idx)
        self.assertLess(valid_to_idx, where_clause_idx)

    def test_range_join_conditions_in_on_clause_list(self) -> None:
        """Test that range JOIN conditions appear in list features base CTE ON clause."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="VALID_FROM",
            mapping_valid_to_col="VALID_TO",
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertIn("t.TILE_START >= m.VALID_FROM", sql)
        self.assertIn("m.VALID_TO IS NULL OR t.TILE_START < m.VALID_TO", sql)
        self.assertIn("LATERAL FLATTEN", sql)

    def test_no_temporal_columns_uses_plain_join(self) -> None:
        """Test that omitting both temporal columns produces a plain JOIN."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertNotIn("ASOF JOIN", sql)
        self.assertNotIn("VALID_FROM", sql)
        self.assertNotIn("VALID_TO", sql)
        self.assertIn("JOIN", sql)

    def test_range_join_in_generate_as_cte(self) -> None:
        """Test that generate_as_cte uses range JOIN with both conditions."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="VALID_FROM",
            mapping_valid_to_col="VALID_TO",
        )
        cte_name, cte_body = generator.generate_as_cte("PIT_ROLLUP_FV0")

        self.assertEqual(cte_name, "PIT_ROLLUP_FV0")
        self.assertNotIn("ASOF JOIN", cte_body)
        self.assertIn("t.TILE_START >= m.VALID_FROM", cte_body)
        self.assertIn("m.VALID_TO IS NULL OR t.TILE_START < m.VALID_TO", cte_body)

    def test_validation_rejects_valid_to_only(self) -> None:
        """Test that RollupConfig rejects valid_to without valid_from."""
        from unittest.mock import MagicMock

        from snowflake.ml.feature_store.feature_view import RollupConfig

        mock_fv = MagicMock()
        mock_fv.is_tiled = True
        mock_fv.status = MagicMock()
        mock_fv.status.name = "ACTIVE"
        mock_fv.entities = []

        mock_df = MagicMock()
        mock_df.schema.fields = []

        config = RollupConfig(
            source=mock_fv,
            mapping_df=mock_df,
            mapping_valid_to_col="VALID_TO",
        )
        with self.assertRaisesRegex(ValueError, "must both be provided or both be omitted"):
            config.validate(target_entity_keys=[])

    def test_validation_rejects_valid_from_only(self) -> None:
        """Test that RollupConfig rejects valid_from without valid_to."""
        from unittest.mock import MagicMock

        from snowflake.ml.feature_store.feature_view import RollupConfig

        mock_fv = MagicMock()
        mock_fv.is_tiled = True
        mock_fv.status = MagicMock()
        mock_fv.status.name = "ACTIVE"
        mock_fv.entities = []

        mock_df = MagicMock()
        mock_df.schema.fields = []

        config = RollupConfig(
            source=mock_fv,
            mapping_df=mock_df,
            mapping_valid_from_col="VALID_FROM",
        )
        with self.assertRaisesRegex(ValueError, "must both be provided or both be omitted"):
            config.validate(target_entity_keys=[])


class LifetimeRollupSqlGeneratorTest(absltest.TestCase):
    """Unit tests for lifetime feature support in RollupSqlGenerator."""

    def test_rollup_with_lifetime_count(self) -> None:
        """Rollup with lifetime COUNT produces _CUM_COUNT_* via window functions."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="lifetime",
                output_column="EVENT_COUNT_LIFETIME",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertIn("_PARTIAL_COUNT_VISITOR_ID", sql)
        self.assertIn("_CUM_COUNT_VISITOR_ID", sql)
        self.assertIn("PARTITION BY SUBSCRIBER_ID, COMPANY_ID", sql)
        self.assertIn("ORDER BY TILE_START", sql)
        self.assertIn("ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", sql)
        self.assertIn("base.*", sql)

    def test_rollup_with_lifetime_min_max(self) -> None:
        """Rollup with lifetime MIN and MAX produces _CUM_MIN_* and _CUM_MAX_*."""
        specs = [
            AggregationSpec(
                function=AggregationType.MIN,
                source_column="EVENT_TS",
                window="lifetime",
                output_column="FIRST_EVENT_LIFETIME",
            ),
            AggregationSpec(
                function=AggregationType.MAX,
                source_column="EVENT_TS",
                window="lifetime",
                output_column="LAST_EVENT_LIFETIME",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertIn("_CUM_MIN_EVENT_TS", sql)
        self.assertIn("_CUM_MAX_EVENT_TS", sql)
        self.assertIn("MIN(_PARTIAL_MIN_EVENT_TS) OVER", sql)
        self.assertIn("MAX(_PARTIAL_MAX_EVENT_TS) OVER", sql)

    def test_rollup_with_mixed_lifetime_and_windowed(self) -> None:
        """Mixed lifetime + windowed: _CUM_* only for lifetime specs."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="7d",
                output_column="EVENT_COUNT_7D",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="lifetime",
                output_column="EVENT_COUNT_LIFETIME",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertIn("_PARTIAL_COUNT_VISITOR_ID", sql)
        self.assertIn("_CUM_COUNT_VISITOR_ID", sql)
        self.assertIn("base.*", sql)

    def test_rollup_no_cumulative_without_lifetime(self) -> None:
        """No _CUM_* columns when there are no lifetime features (regression guard)."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="24h",
                output_column="EVENT_COUNT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertNotIn("_CUM_", sql)
        self.assertNotIn("base.*", sql)
        self.assertIn("_PARTIAL_COUNT_VISITOR_ID", sql)

    def test_rollup_with_lists_and_lifetime(self) -> None:
        """CTE-based list rollup path also gets cumulative wrapper for lifetime."""
        specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="ORDER_VALUE",
                window="24h",
                output_column="LAST_ORDERS",
                params={"n": 5},
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="lifetime",
                output_column="EVENT_COUNT_LIFETIME",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID", "COMPANY_ID"],
            new_join_keys=["SUBSCRIBER_ID", "COMPANY_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
        )
        sql = generator.generate()

        self.assertIn("WITH", sql)
        self.assertIn("LATERAL FLATTEN", sql)
        self.assertIn("_CUM_COUNT_VISITOR_ID", sql)
        self.assertIn("base.*", sql)
        self.assertIn("PARTITION BY SUBSCRIBER_ID, COMPANY_ID", sql)

    def test_rollup_cte_with_lifetime(self) -> None:
        """generate_as_cte() (PIT path) with lifetime includes cumulative columns."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="VISITOR_ID",
                window="lifetime",
                output_column="EVENT_COUNT_LIFETIME",
            ),
            AggregationSpec(
                function=AggregationType.MIN,
                source_column="EVENT_TS",
                window="lifetime",
                output_column="FIRST_EVENT",
            ),
        ]
        generator = RollupSqlGenerator(
            parent_tile_table="DB.SCHEMA.VISITOR_FV$V1",
            parent_join_keys=["VISITOR_ID"],
            new_join_keys=["SUBSCRIBER_ID"],
            mapping_query="SELECT * FROM mapping",
            aggregation_specs=specs,
            mapping_valid_from_col="VALID_FROM",
            mapping_valid_to_col="VALID_TO",
        )
        cte_name, cte_body = generator.generate_as_cte("PIT_ROLLUP_FV0")

        self.assertEqual(cte_name, "PIT_ROLLUP_FV0")
        self.assertNotIn("ASOF JOIN", cte_body)
        self.assertIn("t.TILE_START >= m.VALID_FROM", cte_body)
        self.assertIn("_CUM_COUNT_VISITOR_ID", cte_body)
        self.assertIn("_CUM_MIN_EVENT_TS", cte_body)
        self.assertIn("PARTITION BY SUBSCRIBER_ID", cte_body)

    def test_extracted_cumulative_fn_matches_tiling(self) -> None:
        """Extracted _generate_cumulative_expressions matches TilingSqlGenerator output."""
        specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="EVENT_ID",
                window="lifetime",
                output_column="COUNT_LIFETIME",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="lifetime",
                output_column="SUM_LIFETIME",
            ),
            AggregationSpec(
                function=AggregationType.MIN,
                source_column="TS",
                window="lifetime",
                output_column="MIN_LIFETIME",
            ),
            AggregationSpec(
                function=AggregationType.MAX,
                source_column="TS",
                window="lifetime",
                output_column="MAX_LIFETIME",
            ),
        ]
        join_keys = ["USER_ID", "ORG_ID"]

        tiling_gen = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=join_keys,
            timestamp_col="EVENT_TS",
            feature_granularity="1d",
            features=specs,
        )
        tiling_cols = tiling_gen._generate_cumulative_columns()
        extracted_cols = _generate_cumulative_expressions(specs, join_keys)

        self.assertEqual(tiling_cols, extracted_cols)


class TilingSqlGeneratorSecondaryKeyTest(absltest.TestCase):
    """Unit tests for :class:`TilingSqlGenerator` with secondary-key aggregations."""

    def test_tile_rows_keyed_by_entity_tile_and_secondary_key(self) -> None:
        """Tile SQL SELECTs the secondary key and GROUPs BY it alongside the entity and tile."""
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=_prepend_keys_specs(features, ["AD_ID"]),
        )
        sql = generator.generate()

        self.assertIn("SUM(AMOUNT) AS _PARTIAL_SUM_AMOUNT", sql)
        self.assertNotIn("ARRAY_AGG", sql)
        self.assertNotIn("_PARTIAL_KEYS_", sql)
        self.assertNotIn("_PARTIAL_SUM_ARR_", sql)

        # Tile row dimensions: (entity, secondary_key, TILE_START)
        self.assertIn("USER_ID,", sql)
        self.assertIn("AD_ID,", sql)
        self.assertIn("TIME_SLICE(EVENT_TS, 1, 'HOUR', 'START') AS TILE_START", sql)
        self.assertIn("GROUP BY USER_ID, AD_ID, TILE_START", sql)

    def test_secondary_key_avg_emits_scalar_sum_and_count_partials(self) -> None:
        """AVG with a secondary key lands as scalar ``_PARTIAL_SUM`` + ``_PARTIAL_COUNT``.

        The merge CTE recombines these with ``SUM(_PARTIAL_SUM)/SUM(_PARTIAL_COUNT)``
        per secondary key before the outer ``ARRAY_AGG``.
        """
        features = [
            AggregationSpec(
                function=AggregationType.AVG,
                source_column="PRICE",
                window="24h",
                output_column="AVG_PRICE",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=_prepend_keys_specs(features, ["PRODUCT_ID"]),
        )
        sql = generator.generate()

        self.assertIn("SUM(PRICE) AS _PARTIAL_SUM_PRICE", sql)
        self.assertIn("COUNT(PRICE) AS _PARTIAL_COUNT_PRICE", sql)
        self.assertNotIn("_PARTIAL_SUM_ARR_", sql)
        self.assertNotIn("_PARTIAL_COUNT_ARR_", sql)
        self.assertNotIn("ARRAY_AGG", sql)
        self.assertIn("GROUP BY USER_ID, PRODUCT_ID, TILE_START", sql)

    def test_secondary_key_stddev_emits_sum_count_and_sum_sq_partials(self) -> None:
        """STD/VAR with a secondary key lands as scalar ``_PARTIAL_SUM`` + ``_PARTIAL_COUNT`` + ``_PARTIAL_SUM_SQ``.

        These three partials are sufficient to reconstruct variance per
        ``(entity, sk)`` at merge time using the parallel-variance formula.
        """
        features = [
            AggregationSpec(
                function=AggregationType.STD,
                source_column="PRICE",
                window="24h",
                output_column="STD_PRICE",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=_prepend_keys_specs(features, ["PRODUCT_ID"]),
        )
        sql = generator.generate()

        self.assertIn("SUM(PRICE) AS _PARTIAL_SUM_PRICE", sql)
        self.assertIn("COUNT(PRICE) AS _PARTIAL_COUNT_PRICE", sql)
        self.assertIn("SUM(PRICE * PRICE) AS _PARTIAL_SUM_SQ_PRICE", sql)
        self.assertNotIn("ARRAY_AGG", sql)
        self.assertIn("GROUP BY USER_ID, PRODUCT_ID, TILE_START", sql)

    def test_secondary_key_multi_entity_group_by(self) -> None:
        """Multiple entity join keys all appear in the GROUP BY alongside the secondary key."""
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_PER_AD",
            ),
        ]
        generator = TilingSqlGenerator(
            source_query="SELECT * FROM events",
            join_keys=["USER_ID", "SESSION_ID"],
            timestamp_col="EVENT_TS",
            feature_granularity="1h",
            features=_prepend_keys_specs(features, ["AD_ID"]),
        )
        sql = generator.generate()

        self.assertIn("GROUP BY USER_ID, SESSION_ID, AD_ID, TILE_START", sql)


class MergingSqlGeneratorSecondaryKeyTest(absltest.TestCase):
    """Unit tests for :class:`MergingSqlGenerator` with secondary-key aggregations.

    Validates the merge pattern:
      * Inner ``GROUP BY (entity..., TILE_BOUNDARY, sk)`` collapses scalar
        partial tiles within each feature's window into one per-sk scalar.
      * Outer ``ARRAY_AGG(... WITHIN GROUP (ORDER BY sk))`` emits element-aligned
        keys/values arrays per ``(entity, boundary)``.
      * Each ``(secondary_key, window)`` group produces its own keys-array
        column (e.g. ``AD_ID_KEYS_1D`` and ``AD_ID_KEYS_7D`` are independent).
    """

    def _generator(
        self,
        features: list[AggregationSpec],
        *,
        secondary_key: str = "AD_ID",
        **kwargs: object,
    ) -> MergingSqlGenerator:
        return MergingSqlGenerator(
            tile_table=kwargs.get("tile_table", "DB.SCHEMA.USER_TILES"),  # type: ignore[arg-type]
            join_keys=kwargs.get("join_keys", ["USER_ID"]),  # type: ignore[arg-type]
            timestamp_col=kwargs.get("timestamp_col", "EVENT_TS"),  # type: ignore[arg-type]
            feature_granularity=kwargs.get("feature_granularity", "1h"),  # type: ignore[arg-type]
            features=_prepend_keys_specs(features, [secondary_key]),
            spine_timestamp_col=kwargs.get("spine_timestamp_col", "query_ts"),  # type: ignore[arg-type]
            fv_index=kwargs.get("fv_index", 0),  # type: ignore[arg-type]
        )

    def _secondary_cte(self, generator: MergingSqlGenerator, fv_index: int = 0) -> str:
        ctes = generator.generate_all_ctes()
        secondary = next(cte for cte in ctes if cte[0] == f"SECONDARY_KEY_MERGED_FV{fv_index}")
        return secondary[1]

    def test_inner_group_by_then_outer_array_agg(self) -> None:
        """Merge CTE uses inner ``GROUP BY sk`` then outer ``ARRAY_AGG ORDER BY sk``."""
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_PER_AD",
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        # No array-in-tile representation — we read scalar partials and
        # aggregate them on the read side.
        self.assertNotIn("LATERAL FLATTEN", body)

        # Inner: per-(entity, boundary, sk) scalar reduction.
        self.assertIn("SUM(_PARTIAL_COUNT_IMPRESSION) AS IMPRESSIONS_PER_AD", body)
        self.assertIn("GROUP BY USER_ID, TILE_BOUNDARY, AD_ID", body)

        # Outer: collect per-sk scalars into element-aligned arrays. The keys
        # column uses the user's literal window suffix (matches Feature
        # default-output convention).
        self.assertIn(
            "ARRAY_AGG(AD_ID) WITHIN GROUP (ORDER BY AD_ID) AS AD_ID_KEYS_24H",
            body,
        )
        self.assertIn(
            "ARRAY_AGG(NVL(IMPRESSIONS_PER_AD, PARSE_JSON('null'))) "
            "WITHIN GROUP (ORDER BY AD_ID) AS IMPRESSIONS_PER_AD",
            body,
        )
        # Outer is grouped only by (entity, boundary) — the sk is collapsed.
        self.assertIn("GROUP BY USER_ID, TILE_BOUNDARY", body)

    def test_avg_reduces_sum_over_count_per_secondary_key(self) -> None:
        """AVG uses ``SUM(_PARTIAL_SUM)/SUM(_PARTIAL_COUNT)`` per secondary key."""
        features = [
            AggregationSpec(
                function=AggregationType.AVG,
                source_column="PRICE",
                window="24h",
                output_column="AVG_PRICE_PER_PRODUCT",
            ),
        ]
        body = self._secondary_cte(self._generator(features, secondary_key="PRODUCT_ID"))

        # The inner expression divides summed partials, guarded against /0.
        self.assertIn(
            "CASE WHEN SUM(_PARTIAL_COUNT_PRICE) > 0 "
            "THEN SUM(_PARTIAL_SUM_PRICE) / SUM(_PARTIAL_COUNT_PRICE) "
            "ELSE NULL END AS AVG_PRICE_PER_PRODUCT",
            body,
        )

    def test_stddev_uses_parallel_variance_formula_per_secondary_key(self) -> None:
        r"""STD merges across tiles via the parallel-variance formula per ``sk``.

        Per ``(entity, boundary, sk)`` the inner expression computes
        ``SQRT(GREATEST(0, SUM(SUM_SQ)/SUM(COUNT) - (SUM(SUM)/SUM(COUNT))^2))``
        from scalar partials, then the outer query ``ARRAY_AGG``\ s those
        per-sk standard deviations into one element-aligned array.
        """
        features = [
            AggregationSpec(
                function=AggregationType.STD,
                source_column="PRICE",
                window="24h",
                output_column="STD_PRICE_PER_PRODUCT",
            ),
        ]
        body = self._secondary_cte(self._generator(features, secondary_key="PRODUCT_ID"))

        # Inner per-sk expression is the parallel-variance formula wrapped in SQRT,
        # guarded against /0 and clamped to non-negative before the SQRT.
        self.assertIn(
            "CASE WHEN SUM(_PARTIAL_COUNT_PRICE) > 0 "
            "THEN SQRT(GREATEST(0, "
            "SUM(_PARTIAL_SUM_SQ_PRICE) / SUM(_PARTIAL_COUNT_PRICE) "
            "- POWER(SUM(_PARTIAL_SUM_PRICE) / SUM(_PARTIAL_COUNT_PRICE), 2))) "
            "ELSE NULL END AS STD_PRICE_PER_PRODUCT",
            body,
        )
        # Outer wraps the per-sk scalar into the final value array.
        self.assertIn(
            "ARRAY_AGG(NVL(STD_PRICE_PER_PRODUCT, PARSE_JSON('null'))) "
            "WITHIN GROUP (ORDER BY PRODUCT_ID) AS STD_PRICE_PER_PRODUCT",
            body,
        )

    def test_variance_uses_parallel_variance_formula_per_secondary_key(self) -> None:
        """VAR mirrors STD but without the outer SQRT.

        ``GREATEST(0, ...)`` is still applied so floating-point cancellation
        cannot push the result slightly negative.
        """
        features = [
            AggregationSpec(
                function=AggregationType.VAR,
                source_column="PRICE",
                window="24h",
                output_column="VAR_PRICE_PER_PRODUCT",
            ),
        ]
        body = self._secondary_cte(self._generator(features, secondary_key="PRODUCT_ID"))

        self.assertIn(
            "CASE WHEN SUM(_PARTIAL_COUNT_PRICE) > 0 "
            "THEN GREATEST(0, "
            "SUM(_PARTIAL_SUM_SQ_PRICE) / SUM(_PARTIAL_COUNT_PRICE) "
            "- POWER(SUM(_PARTIAL_SUM_PRICE) / SUM(_PARTIAL_COUNT_PRICE), 2)) "
            "ELSE NULL END AS VAR_PRICE_PER_PRODUCT",
            body,
        )
        self.assertNotIn("SQRT(", body)

    def test_multi_window_emits_one_keys_column_per_window(self) -> None:
        """Features on different windows each get their own keys column.

        ``{sk}_KEYS_24H`` is independent from ``{sk}_KEYS_7D`` and the groups
        are joined on ``(entity, boundary)`` side-by-side.
        """
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_24H",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="CLICKS",
                window="7d",
                output_column="CLICKS_7D",
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        self.assertIn("AD_ID_KEYS_24H", body)
        self.assertIn("AD_ID_KEYS_7D", body)
        # Multi-group subqueries are stitched together with LEFT JOIN.
        self.assertIn("LEFT JOIN", body)
        # Each feature has its own value-array column.
        self.assertIn("AS IMPRESSIONS_24H", body)
        self.assertIn("AS CLICKS_7D", body)

    def test_features_sharing_window_and_offset_share_single_keys_column(self) -> None:
        """Two features on the same ``(window, offset)`` share one keys array."""
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_24H",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="CLICKS",
                window="24h",
                output_column="CLICKS_24H",
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        # Exactly one keys column because both features share ``(24h, 0)``.
        self.assertEqual(body.count("AS AD_ID_KEYS_24H"), 1)

    def test_same_window_different_offsets_emit_distinct_keys_columns(self) -> None:
        """Same window with different offsets must not share a keys column.

        Each ``(window, offset)`` group gets its own keys column suffixed with
        ``_OFFSET_{offset}`` for non-zero offsets so the merge CTE can group
        them independently.
        """
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="2h",
                output_column="AMOUNT_2H",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="2h",
                output_column="AMOUNT_2H_OFFSET_1H",
                offset="1h",
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        self.assertIn("AS AD_ID_KEYS_2H", body)
        self.assertIn("AS AD_ID_KEYS_2H_OFFSET_1H", body)
        # The two groups are stitched with a LEFT JOIN so each value column
        # lands alongside its own keys column.
        self.assertIn("LEFT JOIN", body)

    def test_window_filter_applied_as_where_on_inner_query(self) -> None:
        """Window filtering is a ``WHERE`` clause on the inner per-sk query.

        Using ``WHERE`` (rather than ``CASE WHEN ... ELSE 0``) ensures secondary
        keys inactive in this window are absent from the keys array — they
        don't appear with zero values.
        """
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="2h",
                output_column="AMOUNT_PER_AD_2H",
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        self.assertIn("WHERE TILE_START >= DATEADD(HOUR, -2, TILE_BOUNDARY)", body)
        # No CASE WHEN ... ELSE 0 pattern in the inner per-sk expression.
        self.assertNotIn("ELSE 0 END AS AMOUNT_PER_AD_2H", body)
        # Nulls in the secondary key dimension are filtered out.
        self.assertIn("WHERE AD_ID IS NOT NULL", body)

    def test_per_group_window_filter_uses_its_own_window(self) -> None:
        """Each ``(window, offset)`` subquery filters by its *own* window, not the max.

        Regression: a 1h feature must not over-include tiles that a sibling
        7d feature needs — the short-window group bounds itself to 1h.
        """
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="1h",
                output_column="IMPRESSIONS_1H",
            ),
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="CLICKS",
                window="7d",
                output_column="CLICKS_7D",
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        # Both per-window filters are present; neither group uses the other's window.
        self.assertIn("DATEADD(HOUR, -1, TILE_BOUNDARY)", body)
        self.assertIn("DATEADD(HOUR, -168, TILE_BOUNDARY)", body)

    def test_tiles_joined_pulls_through_secondary_key_column(self) -> None:
        """``TILES_JOINED`` exposes the secondary-key column for the merge CTE."""
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_PER_AD",
            ),
        ]
        generator = self._generator(features)
        tiles_joined = next(cte for cte in generator.generate_all_ctes() if cte[0] == "TILES_JOINED_FV0")
        cte_body = tiles_joined[1]

        self.assertIn("TILES.AD_ID", cte_body)

    def test_secondary_key_fv_routes_all_features_through_secondary_path(self) -> None:
        """When a secondary key is set on the FV, every feature flows through ``SECONDARY_KEY_MERGED``.

        With the FV-level refactor, ``aggregation_secondary_keys`` is uniform
        across the FV — there's no mix of "simple" and "secondary-key" features
        within the same FV, so only ``SECONDARY_KEY_MERGED_FV0`` is emitted.
        """
        features = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_PER_AD",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_PER_AD",
            ),
        ]
        ctes = self._generator(features).generate_all_ctes()
        cte_names = [cte[0] for cte in ctes]
        self.assertIn("SECONDARY_KEY_MERGED_FV0", cte_names)
        self.assertNotIn("SIMPLE_MERGED_FV0", cte_names)

        fv_cte = next(cte for cte in ctes if cte[0] == "FV000")
        self.assertIn("SECONDARY_KEY_MERGED_FV0", fv_cte[1])
        self.assertIn("LEFT JOIN", fv_cte[1])

    def test_secondary_key_combined_cte_coalesces_to_empty_array(self) -> None:
        """Combined CTE wraps each secondary-key column in COALESCE(..., ARRAY_CONSTRUCT()).

        Entities with no events in the window produce [] instead of NULL so
        consumers always receive an array and never NULL.
        """
        features = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="IMPRESSION",
                window="24h",
                output_column="IMPRESSIONS_PER_AD",
            ),
        ]
        ctes = self._generator(features).generate_all_ctes()
        fv_cte = next(cte for cte in ctes if cte[0] == "FV000")
        body = fv_cte[1]

        # Both the keys column and the value column are wrapped.
        self.assertIn("COALESCE(secondary_key.AD_ID_KEYS_24H, ARRAY_CONSTRUCT())", body)
        self.assertIn("COALESCE(secondary_key.IMPRESSIONS_PER_AD, ARRAY_CONSTRUCT())", body)

    def test_secondary_key_supports_sketch_aggregations(self) -> None:
        """Sketch aggregations work with a secondary key.

        HLL/T-Digest states are mergeable; per ``(entity, sk)`` we
        ``HLL_COMBINE`` / ``APPROX_PERCENTILE_COMBINE`` then estimate, and the
        outer ``ARRAY_AGG`` collects the per-sk estimates into the value array.
        """
        features = [
            AggregationSpec(
                function=AggregationType.APPROX_COUNT_DISTINCT,
                source_column="USER_AGENT",
                window="24h",
                output_column="UNIQUE_AGENTS_PER_AD",
            ),
            AggregationSpec(
                function=AggregationType.APPROX_PERCENTILE,
                source_column="LATENCY_MS",
                window="24h",
                output_column="P95_LATENCY_PER_AD",
                params={"percentile": 0.95},
            ),
        ]
        body = self._secondary_cte(self._generator(features))

        # HLL: HLL_ESTIMATE(HLL_COMBINE(HLL_IMPORT(_PARTIAL_HLL_USER_AGENT))) per sk
        self.assertIn(
            "HLL_ESTIMATE(HLL_COMBINE(HLL_IMPORT(_PARTIAL_HLL_USER_AGENT))) AS UNIQUE_AGENTS_PER_AD",
            body,
        )
        # T-Digest: APPROX_PERCENTILE_ESTIMATE(APPROX_PERCENTILE_COMBINE(state), pct)
        self.assertIn(
            "APPROX_PERCENTILE_ESTIMATE(APPROX_PERCENTILE_COMBINE(_PARTIAL_TDIGEST_LATENCY_MS), 0.95)"
            " AS P95_LATENCY_PER_AD",
            body,
        )
        # Outer ARRAY_AGG wraps each per-sk estimate.
        self.assertIn(
            "ARRAY_AGG(NVL(UNIQUE_AGENTS_PER_AD, PARSE_JSON('null'))) "
            "WITHIN GROUP (ORDER BY AD_ID) AS UNIQUE_AGENTS_PER_AD",
            body,
        )
        self.assertIn(
            "ARRAY_AGG(NVL(P95_LATENCY_PER_AD, PARSE_JSON('null'))) "
            "WITHIN GROUP (ORDER BY AD_ID) AS P95_LATENCY_PER_AD",
            body,
        )


if __name__ == "__main__":
    absltest.main()
