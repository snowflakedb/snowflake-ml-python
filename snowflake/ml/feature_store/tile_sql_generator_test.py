"""Unit tests for tile_sql_generator module."""

from absl.testing import absltest

from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
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


if __name__ == "__main__":
    absltest.main()
