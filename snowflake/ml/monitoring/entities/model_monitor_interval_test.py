from absl.testing import absltest

from snowflake.ml.monitoring.entities import model_monitor_interval


class ModelMonitorIntervalTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_validate_monitor_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "Failed to parse refresh interval with exception"):
            model_monitor_interval.ModelMonitorRefreshInterval("UNINITIALIZED")

        with self.assertRaisesRegex(ValueError, "Invalid time unit in refresh interval."):
            model_monitor_interval.ModelMonitorRefreshInterval("4 years")

        with self.assertRaisesRegex(ValueError, "Failed to parse refresh interval with exception."):
            model_monitor_interval.ModelMonitorRefreshInterval("2.5 hours")
        ri = model_monitor_interval.ModelMonitorRefreshInterval("1 hours")
        self.assertEqual(ri.minutes, 60)

    def test_predefined_refresh_intervals(self) -> None:
        min_30 = model_monitor_interval.ModelMonitorRefreshInterval.EVERY_30_MINUTES
        hr_1 = model_monitor_interval.ModelMonitorRefreshInterval.HOURLY
        hr_6 = model_monitor_interval.ModelMonitorRefreshInterval.EVERY_6_HOURS
        day_1 = model_monitor_interval.ModelMonitorRefreshInterval.DAILY
        day_7 = model_monitor_interval.ModelMonitorRefreshInterval.WEEKLY
        day_14 = model_monitor_interval.ModelMonitorRefreshInterval.BIWEEKLY
        day_30 = model_monitor_interval.ModelMonitorRefreshInterval.MONTHLY

        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(min_30).minutes, 30)
        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(hr_1).minutes, 60)
        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(hr_6).minutes, 6 * 60)
        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(day_1).minutes, 24 * 60)
        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(day_7).minutes, 7 * 24 * 60)
        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(day_14).minutes, 14 * 24 * 60)
        self.assertEqual(model_monitor_interval.ModelMonitorRefreshInterval(day_30).minutes, 30 * 24 * 60)


if __name__ == "__main__":
    absltest.main()
