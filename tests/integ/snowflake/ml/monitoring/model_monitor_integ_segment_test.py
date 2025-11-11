from absl.testing import absltest

from snowflake.snowpark.exceptions import SnowparkSQLException
from tests.integ.snowflake.ml.monitoring.model_monitor_integ_test_base import (
    ModelMonitorIntegrationTestBase,
)


class ModelMonitorIntegrationSegmentTest(ModelMonitorIntegrationTestBase):
    def test_add_and_drop_segment_column(self):
        """Test adding and dropping a segment column, including duplicate add validation."""

        # Create table and monitor
        source_table_name = "source_table_segment_drop"
        monitor_name = "monitor_segment_drop"

        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", segment_columns=["customer_segment"]
        )

        monitor = self._add_sample_monitor(
            monitor_name=monitor_name, source=source_table_name, model_version=self._model_version
        )

        # First add a segment column, then drop it
        monitor.add_segment_column("customer_segment")

        # Validate the segment column was added
        segment_columns = self._get_monitor_segment_columns(monitor_name)
        self.assertIn("CUSTOMER_SEGMENT", segment_columns, f"CUSTOMER_SEGMENT not found after add: {segment_columns}")

        # Test that adding the same segment column again should fail
        with self.assertRaisesRegex(SnowparkSQLException, ".*already exists.*"):
            monitor.add_segment_column("customer_segment")  # Should fail since already present

        # Now drop the segment column
        monitor.drop_segment_column("customer_segment")

        # Validate the segment column was removed
        segment_columns = self._get_monitor_segment_columns(monitor_name)
        self.assertNotIn(
            "CUSTOMER_SEGMENT", segment_columns, f"CUSTOMER_SEGMENT still found after drop: {segment_columns}"
        )

    def test_create_monitor_with_segment_columns_happy_path(self):
        """Test creating a monitor with valid segment_columns."""

        source_table_name = "source_table_with_segments"
        monitor_name = "monitor_with_segments"

        # Create table with segment columns
        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", segment_columns=["customer_segment"]
        )

        # Add additional segment column for testing multiple segments
        self._session.sql(
            f"ALTER TABLE {self._db_name}.{self._schema_name}.{source_table_name} ADD COLUMN region STRING"
        ).collect()
        self._session.sql(
            f"UPDATE {self._db_name}.{self._schema_name}.{source_table_name} SET region = 'us-west'"
        ).collect()

        # Create monitor with segment columns - this should succeed
        monitor = self._add_sample_monitor(
            monitor_name=monitor_name,
            source=source_table_name,
            model_version=self._model_version,
            segment_columns=["customer_segment", "region"],
        )

        # Verify monitor was created successfully
        self.assertEqual(monitor.name, monitor_name.upper())

        # Verify it appears in the list of monitors
        monitors = self.registry.show_model_monitors()
        monitor_names = [m["name"] for m in monitors]

        self.assertIn(monitor_name.upper(), monitor_names)

    def test_create_monitor_with_segment_columns_missing_in_source(self):
        """Test creating a monitor with invalid segment_columns should fail."""

        source_table_name = "source_table_invalid_segments"
        monitor_name = "monitor_invalid_segments"

        # Create table with segment columns
        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", segment_columns=["customer_segment"]
        )

        # Try to create monitor with invalid segment columns - this should fail
        with self.assertRaisesRegex(
            ValueError,
            "Segment column\\(s\\): \\['NONEXISTENT_COLUMN', 'ANOTHER_INVALID_COLUMN'\\] do not exist in source\\.",
        ):
            self._add_sample_monitor(
                monitor_name=monitor_name,
                source=source_table_name,
                model_version=self._model_version,
                segment_columns=["nonexistent_column", "another_invalid_column"],
            )

        # Verify monitor was NOT created
        monitors = self.registry.show_model_monitors()
        monitor_names = [m["name"] for m in monitors]
        self.assertNotIn(monitor_name.upper(), monitor_names)


if __name__ == "__main__":
    absltest.main()
