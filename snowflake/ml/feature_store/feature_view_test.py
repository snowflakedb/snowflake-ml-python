"""Unit tests for feature_view module."""

from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
from snowflake.ml.feature_store.feature_view import FeatureView, OnlineConfig


class FeatureViewValidationTest(parameterized.TestCase):
    """Unit tests for FeatureView validation logic."""

    def _create_mock_feature_view_with_specs(self, specs: list[AggregationSpec]) -> FeatureView:
        """Create a FeatureView with mocked DataFrame for testing validation."""
        from snowflake.ml.feature_store.entity import Entity

        mock_df = MagicMock()
        mock_df.columns = ["user_id", "event_ts", "amount"]
        mock_df.queries = {"queries": ["SELECT * FROM source"]}

        # Create a real entity with a join key that matches the DataFrame
        entity = Entity(name="user", join_keys=["user_id"])

        # Create FV - pass specs via _kwargs to bypass Feature.to_spec()
        return FeatureView(
            name="test_fv",
            entities=[entity],
            feature_df=mock_df,
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            _aggregation_specs=specs,  # Pass directly, not via features
        )

    def test_duplicate_feature_alias_raises_error(self) -> None:
        """Test that duplicate feature aliases raise ValueError."""
        specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="amount",
                window="2h",
                output_column="TOTAL",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="amount",
                window="2h",
                output_column="TOTAL",  # Duplicate!
            ),
        ]

        with self.assertRaises(ValueError) as cm:
            self._create_mock_feature_view_with_specs(specs)

        self.assertIn("Duplicate feature alias", str(cm.exception))
        self.assertIn("TOTAL", str(cm.exception))

    def test_duplicate_alias_case_insensitive(self) -> None:
        """Test that duplicate aliases are detected case-insensitively."""
        specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="amount",
                window="2h",
                output_column="Total",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="amount",
                window="2h",
                output_column="TOTAL",  # Same when uppercased
            ),
        ]

        with self.assertRaises(ValueError) as cm:
            self._create_mock_feature_view_with_specs(specs)

        self.assertIn("Duplicate feature alias", str(cm.exception))


class OnlineConfigTest(parameterized.TestCase):
    """Unit tests for OnlineConfig class."""

    @parameterized.parameters(  # type: ignore[misc]
        "10s",
        "5m",
        "2h",
        "1d",
        "10 seconds",
        "5 minutes",
        "2 hours",
        "1 day",
        "1 sec",  # Snowflake supports this
        "30 mins",  # Snowflake supports this
    )
    def test_online_config_valid_target_lag(self, target_lag: str) -> None:
        """Test OnlineConfig accepts valid target_lag values."""
        config = OnlineConfig(enable=True, target_lag=target_lag)
        # Just verify it doesn't throw an error and stores the value as-is (after trim)
        self.assertEqual(config.target_lag, target_lag.strip())

    @parameterized.parameters(  # type: ignore[misc]
        ("  10s  ", "10s"),
        ("\t5m\t", "5m"),
        ("\n2h\n", "2h"),
        ("  10 seconds  ", "10 seconds"),
    )
    def test_online_config_whitespace_handling(self, input_val: str, expected: str) -> None:
        """Test OnlineConfig trims whitespace from target_lag."""
        config = OnlineConfig(enable=True, target_lag=input_val)
        self.assertEqual(config.target_lag, expected)

    @parameterized.parameters(  # type: ignore[misc]
        "",
        "   ",
        "\t",
        "\n",
        123,
        10.5,
    )
    def test_online_config_invalid_target_lag(self, invalid_target_lag: object) -> None:
        """Test OnlineConfig rejects empty/invalid target_lag values."""
        with self.assertRaises(ValueError) as cm:
            OnlineConfig(enable=True, target_lag=invalid_target_lag)  # type: ignore[arg-type]

        error_msg = str(cm.exception)
        self.assertIn("non-empty string", error_msg)


if __name__ == "__main__":
    absltest.main()
