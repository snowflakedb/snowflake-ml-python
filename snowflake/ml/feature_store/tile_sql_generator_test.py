"""Unit tests for tile_sql_generator module."""

from absl.testing import absltest

from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
from snowflake.ml.feature_store.tile_sql_generator import (
    MergingSqlGenerator,
    TilingSqlGenerator,
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
        # Column names are quoted for case-sensitivity
        self.assertIn('"query_ts"', cte_body)

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
        # Column names are quoted for case-sensitivity
        self.assertIn('"USER_ID"', cte_body)
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


if __name__ == "__main__":
    absltest.main()
