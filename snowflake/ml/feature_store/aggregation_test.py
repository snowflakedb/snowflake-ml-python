"""Unit tests for aggregation module."""

from absl.testing import absltest, parameterized

from snowflake.ml.feature_store.aggregation import (
    AggregationSpec,
    AggregationType,
    format_interval_for_snowflake,
    interval_to_seconds,
    parse_interval,
)
from snowflake.ml.feature_store.feature import Feature


class ParseIntervalTest(parameterized.TestCase):
    """Unit tests for parse_interval function."""

    @parameterized.parameters(  # type: ignore[misc]
        ("1h", 1, "HOUR"),
        ("24h", 24, "HOUR"),
        ("1 hour", 1, "HOUR"),
        ("24 hours", 24, "HOUR"),
        ("30m", 30, "MINUTE"),
        ("30 minutes", 30, "MINUTE"),
        ("1d", 1, "DAY"),
        ("7 days", 7, "DAY"),
        ("60s", 60, "SECOND"),
        ("60 seconds", 60, "SECOND"),
        ("  1h  ", 1, "HOUR"),  # whitespace handling
    )
    def test_parse_interval_valid(self, interval: str, expected_value: int, expected_unit: str) -> None:
        """Test parse_interval with valid inputs."""
        value, unit = parse_interval(interval)
        self.assertEqual(value, expected_value)
        self.assertEqual(unit, expected_unit)

    @parameterized.parameters(  # type: ignore[misc]
        "",
        "invalid",
        "h",
        "1",
        "1x",
        "-1h",
        "0h",
    )
    def test_parse_interval_invalid(self, interval: str) -> None:
        """Test parse_interval with invalid inputs."""
        with self.assertRaises(ValueError):
            parse_interval(interval)


class IntervalToSecondsTest(parameterized.TestCase):
    """Unit tests for interval_to_seconds function."""

    @parameterized.parameters(  # type: ignore[misc]
        ("1s", 1),
        ("60s", 60),
        ("1m", 60),
        ("30m", 1800),
        ("1h", 3600),
        ("24h", 86400),
        ("1d", 86400),
        ("7d", 604800),
    )
    def test_interval_to_seconds(self, interval: str, expected: int) -> None:
        """Test interval_to_seconds conversion."""
        self.assertEqual(interval_to_seconds(interval), expected)


class FormatIntervalForSnowflakeTest(parameterized.TestCase):
    """Unit tests for format_interval_for_snowflake function."""

    @parameterized.parameters(  # type: ignore[misc]
        ("1h", "HOUR"),
        ("30m", "MINUTE"),
        ("1d", "DAY"),
        ("60s", "SECOND"),
    )
    def test_format_interval(self, interval: str, expected: str) -> None:
        """Test format_interval_for_snowflake."""
        self.assertEqual(format_interval_for_snowflake(interval), expected)


class AggregationTypeTest(absltest.TestCase):
    """Unit tests for AggregationType enum."""

    def test_is_simple(self) -> None:
        """Test is_simple method."""
        self.assertTrue(AggregationType.SUM.is_simple())
        self.assertTrue(AggregationType.COUNT.is_simple())
        self.assertTrue(AggregationType.AVG.is_simple())
        self.assertTrue(AggregationType.STD.is_simple())
        self.assertTrue(AggregationType.VAR.is_simple())
        self.assertFalse(AggregationType.LAST_N.is_simple())
        self.assertFalse(AggregationType.LAST_DISTINCT_N.is_simple())
        self.assertFalse(AggregationType.FIRST_N.is_simple())
        self.assertFalse(AggregationType.FIRST_DISTINCT_N.is_simple())

    def test_is_list(self) -> None:
        """Test is_list method."""
        self.assertFalse(AggregationType.SUM.is_list())
        self.assertFalse(AggregationType.COUNT.is_list())
        self.assertFalse(AggregationType.AVG.is_list())
        self.assertFalse(AggregationType.STD.is_list())
        self.assertFalse(AggregationType.VAR.is_list())
        self.assertTrue(AggregationType.LAST_N.is_list())
        self.assertTrue(AggregationType.LAST_DISTINCT_N.is_list())
        self.assertTrue(AggregationType.FIRST_N.is_list())
        self.assertTrue(AggregationType.FIRST_DISTINCT_N.is_list())


class AggregationSpecTest(absltest.TestCase):
    """Unit tests for AggregationSpec dataclass."""

    def test_simple_aggregation(self) -> None:
        """Test creating a simple aggregation spec."""
        spec = AggregationSpec(
            function=AggregationType.SUM,
            source_column="amount",
            window="24h",
            output_column="amount_sum_24h",
        )
        self.assertEqual(spec.function, AggregationType.SUM)
        self.assertEqual(spec.source_column, "amount")
        self.assertEqual(spec.window, "24h")
        self.assertEqual(spec.output_column, "amount_sum_24h")
        self.assertEqual(spec.get_window_seconds(), 86400)

    def test_list_aggregation_with_n(self) -> None:
        """Test creating a list aggregation spec with n parameter."""
        spec = AggregationSpec(
            function=AggregationType.LAST_N,
            source_column="page_id",
            window="1h",
            output_column="recent_pages",
            params={"n": 10},
        )
        self.assertEqual(spec.function, AggregationType.LAST_N)
        self.assertEqual(spec.params["n"], 10)

    def test_list_aggregation_without_n_raises(self) -> None:
        """Test that list aggregation without n parameter raises."""
        with self.assertRaises(ValueError) as cm:
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="page_id",
                window="1h",
                output_column="recent_pages",
            )
        self.assertIn("'n' is required", str(cm.exception))

    def test_invalid_window_raises(self) -> None:
        """Test that invalid window format raises."""
        with self.assertRaises(ValueError):
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="amount",
                window="invalid",
                output_column="amount_sum",
            )

    def test_get_tile_column_name(self) -> None:
        """Test tile column name generation.

        Tile columns are based on (partial_type, source_column) to maximize
        sharing across aggregation types. For example, SUM and AVG on the
        same column share the _PARTIAL_SUM_{col} column.
        """
        spec = AggregationSpec(
            function=AggregationType.SUM,
            source_column="amount",
            window="24h",
            output_column="amount_sum_24h",
        )
        # SUM uses _PARTIAL_SUM_{col}
        self.assertEqual(spec.get_tile_column_name("SUM"), "_PARTIAL_SUM_AMOUNT")

        # AVG uses _PARTIAL_SUM_{col} and _PARTIAL_COUNT_{col}
        avg_spec = AggregationSpec(
            function=AggregationType.AVG,
            source_column="price",
            window="1h",
            output_column="avg_price",
        )
        self.assertEqual(avg_spec.get_tile_column_name("SUM"), "_PARTIAL_SUM_PRICE")
        self.assertEqual(avg_spec.get_tile_column_name("COUNT"), "_PARTIAL_COUNT_PRICE")

        # STD/VAR also use _PARTIAL_SUM_SQ_{col}
        std_spec = AggregationSpec(
            function=AggregationType.STD,
            source_column="value",
            window="2h",
            output_column="std_value",
        )
        self.assertEqual(std_spec.get_tile_column_name("SUM_SQ"), "_PARTIAL_SUM_SQ_VALUE")

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization and deserialization."""
        spec = AggregationSpec(
            function=AggregationType.LAST_N,
            source_column="page_id",
            window="1h",
            output_column="recent_pages",
            params={"n": 10},
        )
        spec_dict = spec.to_dict()
        self.assertEqual(spec_dict["function"], "last_n")
        self.assertEqual(spec_dict["source_column"], "page_id")
        self.assertEqual(spec_dict["window"], "1h")
        self.assertEqual(spec_dict["output_column"], "recent_pages")
        self.assertEqual(spec_dict["params"], {"n": 10})

        # Round-trip
        spec2 = AggregationSpec.from_dict(spec_dict)
        self.assertEqual(spec, spec2)


class FeatureTest(absltest.TestCase):
    """Unit tests for Feature class."""

    def test_sum(self) -> None:
        """Test Feature.sum factory method."""
        feature = Feature.sum("amount", "24h")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.SUM)
        self.assertEqual(spec.source_column, "amount")
        self.assertEqual(spec.window, "24h")
        # Default output name
        self.assertEqual(spec.output_column, "AMOUNT_SUM_24H")

    def test_count(self) -> None:
        """Test Feature.count factory method."""
        feature = Feature.count("transaction_id", "7d")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.COUNT)
        self.assertEqual(spec.source_column, "transaction_id")
        self.assertEqual(spec.window, "7d")

    def test_avg(self) -> None:
        """Test Feature.avg factory method."""
        feature = Feature.avg("price", "1h")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.AVG)
        self.assertEqual(spec.source_column, "price")
        self.assertEqual(spec.window, "1h")

    def test_min(self) -> None:
        """Test Feature.min factory method."""
        feature = Feature.min("price", "24h")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.MIN)
        self.assertEqual(spec.source_column, "price")
        self.assertEqual(spec.window, "24h")
        self.assertEqual(spec.output_column, "PRICE_MIN_24H")

    def test_max(self) -> None:
        """Test Feature.max factory method."""
        feature = Feature.max("price", "24h")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.MAX)
        self.assertEqual(spec.source_column, "price")
        self.assertEqual(spec.window, "24h")
        self.assertEqual(spec.output_column, "PRICE_MAX_24H")

    def test_stddev(self) -> None:
        """Test Feature.stddev factory method."""
        feature = Feature.stddev("price", "24h")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.STD)
        self.assertEqual(spec.source_column, "price")
        self.assertEqual(spec.window, "24h")

    def test_var(self) -> None:
        """Test Feature.var factory method."""
        feature = Feature.var("price", "24h")
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.VAR)
        self.assertEqual(spec.source_column, "price")
        self.assertEqual(spec.window, "24h")

    def test_last_n(self) -> None:
        """Test Feature.last_n factory method."""
        feature = Feature.last_n("page_id", "1h", n=10)
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.LAST_N)
        self.assertEqual(spec.params["n"], 10)

    def test_last_distinct_n(self) -> None:
        """Test Feature.last_distinct_n factory method."""
        feature = Feature.last_distinct_n("category", "24h", n=5)
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.LAST_DISTINCT_N)
        self.assertEqual(spec.params["n"], 5)

    def test_first_n(self) -> None:
        """Test Feature.first_n factory method."""
        feature = Feature.first_n("page_id", "1h", n=10)
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.FIRST_N)
        self.assertEqual(spec.params["n"], 10)

    def test_first_distinct_n(self) -> None:
        """Test Feature.first_distinct_n factory method."""
        feature = Feature.first_distinct_n("category", "6h", n=5)
        spec = feature.to_spec()
        self.assertEqual(spec.function, AggregationType.FIRST_DISTINCT_N)
        self.assertEqual(spec.params["n"], 5)

    def test_alias(self) -> None:
        """Test alias method - default is case-insensitive (uppercase)."""
        feature = Feature.sum("amount", "24h").alias("total_amount_24h")
        spec = feature.to_spec()
        self.assertEqual(spec.output_column, "TOTAL_AMOUNT_24H")

    def test_alias_case_sensitive(self) -> None:
        """Test alias method with case_sensitive=True (quoted)."""
        feature = Feature.sum("amount", "24h").alias("Total_Amount", case_sensitive=True)
        spec = feature.to_spec()
        self.assertEqual(spec.output_column, '"Total_Amount"')

    def test_repr(self) -> None:
        """Test __repr__ method."""
        feature = Feature.sum("amount", "24h").alias("total")
        repr_str = repr(feature)
        self.assertIn("sum", repr_str)
        self.assertIn("amount", repr_str)
        self.assertIn("24h", repr_str)
        self.assertIn("TOTAL", repr_str)  # Uppercase since case_sensitive=False by default

    def test_offset_default(self) -> None:
        """Test that offset defaults to '0' (no offset)."""
        feature = Feature.sum("amount", "24h")
        spec = feature.to_spec()
        self.assertEqual(spec.offset, "0")
        self.assertEqual(spec.get_offset_seconds(), 0)

    def test_offset_explicit(self) -> None:
        """Test explicit offset parameter."""
        feature = Feature.sum("amount", "7d", offset="7d")
        spec = feature.to_spec()
        self.assertEqual(spec.offset, "7d")
        self.assertEqual(spec.get_offset_seconds(), 604800)  # 7 days in seconds

    def test_offset_with_all_aggregation_types(self) -> None:
        """Test offset works with all aggregation factory methods."""
        # Simple aggregations
        self.assertEqual(Feature.sum("a", "1h", offset="1h").to_spec().offset, "1h")
        self.assertEqual(Feature.count("a", "1h", offset="2h").to_spec().offset, "2h")
        self.assertEqual(Feature.avg("a", "1h", offset="1d").to_spec().offset, "1d")
        self.assertEqual(Feature.stddev("a", "1h", offset="1h").to_spec().offset, "1h")
        self.assertEqual(Feature.var("a", "1h", offset="1h").to_spec().offset, "1h")

        # List aggregations
        self.assertEqual(Feature.last_n("a", "1h", n=5, offset="1h").to_spec().offset, "1h")
        self.assertEqual(Feature.last_distinct_n("a", "1h", n=5, offset="1h").to_spec().offset, "1h")
        self.assertEqual(Feature.first_n("a", "1h", n=5, offset="1h").to_spec().offset, "1h")
        self.assertEqual(Feature.first_distinct_n("a", "1h", n=5, offset="1h").to_spec().offset, "1h")

    def test_offset_invalid_format_raises(self) -> None:
        """Test that invalid offset format raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="amount",
                window="24h",
                output_column="amount_sum",
                offset="invalid",
            )
        self.assertIn("Invalid offset", str(cm.exception))

    def test_offset_serialization(self) -> None:
        """Test offset is included in to_dict/from_dict."""
        spec = AggregationSpec(
            function=AggregationType.SUM,
            source_column="amount",
            window="7d",
            output_column="prev_week_sum",
            offset="7d",
        )
        spec_dict = spec.to_dict()
        self.assertEqual(spec_dict["offset"], "7d")

        # Round-trip
        spec2 = AggregationSpec.from_dict(spec_dict)
        self.assertEqual(spec2.offset, "7d")

    def test_offset_from_dict_default(self) -> None:
        """Test from_dict handles missing offset (backwards compatibility)."""
        spec_dict = {
            "function": "sum",
            "source_column": "amount",
            "window": "24h",
            "output_column": "amount_sum",
        }
        spec = AggregationSpec.from_dict(spec_dict)
        self.assertEqual(spec.offset, "0")

    def test_repr_with_offset(self) -> None:
        """Test __repr__ includes offset when non-zero."""
        feature = Feature.sum("amount", "7d", offset="7d").alias("prev_week")
        repr_str = repr(feature)
        self.assertIn("offset='7d'", repr_str)

    def test_repr_without_offset(self) -> None:
        """Test __repr__ omits offset when zero."""
        feature = Feature.sum("amount", "7d")
        repr_str = repr(feature)
        self.assertNotIn("offset", repr_str)


class LifetimeWindowTest(absltest.TestCase):
    """Unit tests for lifetime window support."""

    def test_is_lifetime_window_true(self) -> None:
        """Test is_lifetime_window returns True for 'lifetime'."""
        from snowflake.ml.feature_store.aggregation import is_lifetime_window

        self.assertTrue(is_lifetime_window("lifetime"))
        self.assertTrue(is_lifetime_window("LIFETIME"))
        self.assertTrue(is_lifetime_window("  lifetime  "))
        self.assertTrue(is_lifetime_window("Lifetime"))

    def test_is_lifetime_window_false(self) -> None:
        """Test is_lifetime_window returns False for non-lifetime windows."""
        from snowflake.ml.feature_store.aggregation import is_lifetime_window

        self.assertFalse(is_lifetime_window("1h"))
        self.assertFalse(is_lifetime_window("7d"))
        self.assertFalse(is_lifetime_window("lifetime_"))
        self.assertFalse(is_lifetime_window("my_lifetime"))

    def test_parse_interval_rejects_lifetime(self) -> None:
        """Test parse_interval raises error for lifetime windows."""
        with self.assertRaisesRegex(ValueError, "not a numeric interval"):
            parse_interval("lifetime")

    def test_interval_to_seconds_lifetime_returns_sentinel(self) -> None:
        """Test interval_to_seconds returns -1 for lifetime windows."""
        self.assertEqual(interval_to_seconds("lifetime"), -1)

    def test_lifetime_feature_sum(self) -> None:
        """Test creating a lifetime SUM feature."""
        feature = Feature.sum("amount", "lifetime").alias("total_amount")
        spec = feature.to_spec()

        self.assertEqual(spec.window, "lifetime")
        self.assertTrue(spec.is_lifetime())
        self.assertEqual(spec.output_column, "TOTAL_AMOUNT")

    def test_lifetime_feature_count(self) -> None:
        """Test creating a lifetime COUNT feature."""
        feature = Feature.count("amount", "lifetime").alias("total_count")
        spec = feature.to_spec()

        self.assertEqual(spec.window, "lifetime")
        self.assertTrue(spec.is_lifetime())

    def test_lifetime_feature_avg(self) -> None:
        """Test creating a lifetime AVG feature."""
        feature = Feature.avg("amount", "lifetime").alias("avg_amount")
        spec = feature.to_spec()

        self.assertEqual(spec.window, "lifetime")
        self.assertTrue(spec.is_lifetime())

    def test_lifetime_feature_min_max(self) -> None:
        """Test creating lifetime MIN and MAX features."""
        min_feature = Feature.min("amount", "lifetime").alias("min_amount")
        max_feature = Feature.max("amount", "lifetime").alias("max_amount")

        min_spec = min_feature.to_spec()
        max_spec = max_feature.to_spec()

        self.assertTrue(min_spec.is_lifetime())
        self.assertTrue(max_spec.is_lifetime())

    def test_lifetime_feature_stddev_var(self) -> None:
        """Test creating lifetime STDDEV and VAR features."""
        stddev_feature = Feature.stddev("amount", "lifetime").alias("stddev_amount")
        var_feature = Feature.var("amount", "lifetime").alias("var_amount")

        stddev_spec = stddev_feature.to_spec()
        var_spec = var_feature.to_spec()

        self.assertTrue(stddev_spec.is_lifetime())
        self.assertTrue(var_spec.is_lifetime())

    def test_lifetime_feature_approx_count_distinct_not_supported(self) -> None:
        """Test that lifetime APPROX_COUNT_DISTINCT is not supported."""
        feature = Feature.approx_count_distinct("user_id", "lifetime").alias("unique_users")
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for approx_count_distinct"):
            feature.to_spec()

    def test_lifetime_feature_approx_percentile_not_supported(self) -> None:
        """Test that lifetime APPROX_PERCENTILE is not supported."""
        feature = Feature.approx_percentile("amount", "lifetime", percentile=0.5).alias("median")
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for approx_percentile"):
            feature.to_spec()

    def test_lifetime_feature_last_n_not_supported(self) -> None:
        """Test that lifetime LAST_N is not supported."""
        feature = Feature.last_n("page_id", "lifetime", n=10).alias("last_pages")
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for last_n"):
            feature.to_spec()

    def test_lifetime_feature_first_n_not_supported(self) -> None:
        """Test that lifetime FIRST_N is not supported."""
        feature = Feature.first_n("page_id", "lifetime", n=10).alias("first_pages")
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for first_n"):
            feature.to_spec()

    def test_lifetime_offset_not_allowed(self) -> None:
        """Test that offset is not allowed with lifetime windows."""
        with self.assertRaisesRegex(ValueError, "Offset is not supported with lifetime"):
            Feature.sum("amount", "lifetime", offset="1d").to_spec()

    def test_lifetime_default_output_name(self) -> None:
        """Test default output name for lifetime features."""
        feature = Feature.sum("amount", "lifetime")
        spec = feature.to_spec()

        # Should be AMOUNT_SUM_LIFETIME
        self.assertEqual(spec.output_column, "AMOUNT_SUM_LIFETIME")

    def test_lifetime_get_window_seconds_returns_sentinel(self) -> None:
        """Test get_window_seconds returns -1 for lifetime specs."""
        feature = Feature.sum("amount", "lifetime")
        spec = feature.to_spec()

        self.assertEqual(spec.get_window_seconds(), -1)

    def test_lifetime_get_cumulative_column_name(self) -> None:
        """Test get_cumulative_column_name for lifetime features."""
        feature = Feature.sum("amount", "lifetime")
        spec = feature.to_spec()

        self.assertEqual(spec.get_cumulative_column_name("SUM"), "_CUM_SUM_AMOUNT")
        self.assertEqual(spec.get_cumulative_column_name("COUNT"), "_CUM_COUNT_AMOUNT")


if __name__ == "__main__":
    absltest.main()
