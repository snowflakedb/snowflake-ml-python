import json

from absl.testing import absltest

from tests.integ.snowflake.ml.monitoring.model_monitor_integ_test_base import (
    ModelMonitorIntegrationTestBase,
)


class ModelMonitorIntegrationCustomMetricTest(ModelMonitorIntegrationTestBase):
    def test_add_drop_custom_metric_columns(self):
        """Test adding and dropping custom metric columns."""

        self._session.sql("ALTER SESSION SET ENABLE_MODEL_MONITOR_CUSTOM_METRICS = TRUE").collect()
        source_table_name = "source_table_custom_metrics"
        monitor_name = "monitor_custom_metrics"

        # Create table with initial custom metric columns
        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", custom_metric_columns=["initial_metric"]
        )

        # Create monitor with initial custom metric columns
        monitor = self._add_sample_monitor(
            monitor_name=monitor_name,
            source=source_table_name,
            model_version=self._model_version,
            custom_metric_columns=["initial_metric"],
        )

        # Verify monitor was created successfully
        self.assertEqual(monitor.name, monitor_name.upper())

        # Verify monitor was created successfully with initial custom metric
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()
        self.assertIn("INITIAL_METRIC", describe_result[0]["columns"])

        # Add new custom metric columns
        monitor.add_custom_metric_column("input_feature_1")

        # Verify new custom metric columns were added
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()
        self.assertIn("INPUT_FEATURE_1", json.loads(describe_result[0]["columns"])["custom_metric_columns"])

        # Drop a custom metric column
        monitor.drop_custom_metric_column("initial_metric")

        # Verify the custom metric column was dropped
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()
        self.assertNotIn("INITIAL_METRIC", json.loads(describe_result[0]["columns"])["custom_metric_columns"])
        self.assertIn("INITIAL_METRIC", json.loads(describe_result[0]["columns"])["numerical_columns"])


if __name__ == "__main__":
    absltest.main()
