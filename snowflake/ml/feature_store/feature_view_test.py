"""Unit tests for feature_view module."""

import unittest

from snowflake.ml.feature_store.feature_view import OnlineConfig


class OnlineConfigTest(unittest.TestCase):
    """Unit tests for OnlineConfig class."""

    def test_online_config_valid_target_lag(self) -> None:
        """Test OnlineConfig accepts valid target_lag values."""
        valid_cases = [
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
        ]

        for target_lag in valid_cases:
            with self.subTest(target_lag=target_lag):
                config = OnlineConfig(enable=True, target_lag=target_lag)
                # Just verify it doesn't throw an error and stores the value as-is (after trim)
                self.assertEqual(config.target_lag, target_lag.strip())

    def test_online_config_whitespace_handling(self) -> None:
        """Test OnlineConfig trims whitespace from target_lag."""
        test_cases = [
            ("  10s  ", "10s"),
            ("\t5m\t", "5m"),
            ("\n2h\n", "2h"),
            ("  10 seconds  ", "10 seconds"),
        ]

        for input_val, expected in test_cases:
            with self.subTest(input_val=repr(input_val)):
                config = OnlineConfig(enable=True, target_lag=input_val)
                self.assertEqual(config.target_lag, expected)

    def test_online_config_invalid_target_lag(self) -> None:
        """Test OnlineConfig rejects empty/invalid target_lag values."""
        invalid_cases = [
            "",
            "   ",
            "\t",
            "\n",
            None,
            123,
            10.5,
            [],
            {},
        ]

        for invalid_target_lag in invalid_cases:
            with self.subTest(invalid_target_lag=repr(invalid_target_lag)):
                with self.assertRaises(ValueError) as cm:
                    OnlineConfig(enable=True, target_lag=invalid_target_lag)  # type: ignore[arg-type]

                error_msg = str(cm.exception)
                self.assertIn("non-empty string", error_msg)


if __name__ == "__main__":
    unittest.main()
