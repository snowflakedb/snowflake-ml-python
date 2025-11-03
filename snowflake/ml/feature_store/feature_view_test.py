"""Unit tests for feature_view module."""

from absl.testing import absltest, parameterized

from snowflake.ml.feature_store.feature_view import OnlineConfig


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
        None,
        123,
        10.5,
        [],
        {},
    )
    def test_online_config_invalid_target_lag(self, invalid_target_lag) -> None:
        """Test OnlineConfig rejects empty/invalid target_lag values."""
        with self.assertRaises(ValueError) as cm:
            OnlineConfig(enable=True, target_lag=invalid_target_lag)  # type: ignore[arg-type]

        error_msg = str(cm.exception)
        self.assertIn("non-empty string", error_msg)


if __name__ == "__main__":
    absltest.main()
