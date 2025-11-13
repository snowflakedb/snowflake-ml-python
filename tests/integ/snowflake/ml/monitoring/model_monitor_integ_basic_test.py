from absl.testing import absltest

from snowflake.snowpark.exceptions import SnowparkSQLException
from tests.integ.snowflake.ml.monitoring.model_monitor_integ_test_base import (
    ModelMonitorIntegrationTestBase,
)


class ModelMonitorIntegrationBasicTest(ModelMonitorIntegrationTestBase):
    def test_show_and_get_monitors(self):
        # Test show returns empty initially
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 0)

        # Create table and monitor
        self._create_sample_table_and_monitor(monitor_name="monitor", table_name="source_table")

        # Test show returns the monitor with correct name
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "MONITOR")

        # Test get by name
        monitor = self.registry.get_monitor(name="monitor")
        self.assertEqual(monitor.name, "MONITOR")

        # Test get by model_version
        model_version = self.registry.get_model(self._model_name).version("V1")
        monitor2 = self.registry.get_monitor(model_version=model_version)
        self.assertEqual(monitor2.name, "MONITOR")

        # Test get by name, doesn't exist
        with self.assertRaisesRegex(ValueError, "Unable to find model monitor 'non_existent_monitor'"):
            self.registry.get_monitor(name="non_existent_monitor")

        # Test get by model_version, doesn't exist
        model_version_not_monitored = self._add_sample_model_version(model_name="fake_model_name", version_name="V2")
        with self.assertRaisesRegex(ValueError, "Unable to find model monitor for the given model version."):
            self.registry.get_monitor(model_version=model_version_not_monitored)

    def test_add_monitor_duplicate_fails(self):
        source_table_name = "source_table"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_name}")
        self._add_sample_monitor(
            monitor_name="test_monitor_name",
            source="source_table",
            model_version=self._model_version,
        )

        with self.assertRaisesRegex(SnowparkSQLException, ".*already exists.*"):
            self._add_sample_monitor(
                monitor_name="test_monitor_name",
                source="source_table",
                model_version=self._model_version,
            )

        with self.assertRaisesRegex(SnowparkSQLException, ".*already exists.*"):
            self._add_sample_monitor(
                monitor_name="test_monitor_name2",
                source="source_table",
                model_version=self._model_version,
            )

    def test_suspend_resume_monitor(self):
        self._create_sample_table_and_monitor(monitor_name="monitor", table_name="source_table")
        monitor = self.registry.get_monitor(name="monitor")
        self.assertTrue(self.registry.show_model_monitors()[0]["monitor_state"] in ["ACTIVE", "RUNNING"])

        monitor.resume()  # resume while already running
        self.assertTrue(self.registry.show_model_monitors()[0]["monitor_state"] in ["ACTIVE", "RUNNING"])
        monitor.suspend()  # suspend after running
        self.assertEqual(self.registry.show_model_monitors()[0]["monitor_state"], "SUSPENDED")
        monitor.suspend()  # suspend while already suspended
        self.assertEqual(self.registry.show_model_monitors()[0]["monitor_state"], "SUSPENDED")
        monitor.resume()  # resume after suspending
        self.assertTrue(self.registry.show_model_monitors()[0]["monitor_state"] in ["ACTIVE", "RUNNING"])

    def test_delete_monitor(self) -> None:
        self._create_sample_table_and_monitor(monitor_name="monitor", table_name="source_table")
        self.assertEqual(len(self.registry.show_model_monitors()), 1)
        self.registry.delete_monitor(name="monitor")
        with self.assertRaisesRegex(ValueError, "Unable to find model monitor 'monitor'"):
            self.registry.get_monitor(name="monitor")
        self.assertEqual(len(self.registry.show_model_monitors()), 0)

    def test_create_model_monitor_from_view(self):
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 0)

        source_table_name = "source_table"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_name}")
        self._session.sql(
            f"""CREATE OR REPLACE VIEW {self._db_name}.{self._schema_name}.{source_table_name}_view
                AS SELECT * FROM {self._db_name}.{self._schema_name}.{source_table_name}"""
        ).collect()

        self._add_sample_monitor(
            monitor_name="monitor",
            source="source_table_view",
            model_version=self._model_version,
        )

        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "MONITOR")


if __name__ == "__main__":
    absltest.main()
