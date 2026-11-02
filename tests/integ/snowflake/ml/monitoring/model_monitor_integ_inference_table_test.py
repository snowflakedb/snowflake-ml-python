import json

import pandas as pd
from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml.model import custom_model, task
from tests.integ.snowflake.ml.monitoring.model_monitor_integ_test_base import (
    INPUT_FEATURE_COLUMNS_NAMES,
    ModelMonitorIntegrationTestBase,
)


class _CustomerSegmentModel(custom_model.CustomModel):
    """Predicts a numeric feature and accepts a segment column on the request."""

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"OUTPUT": input_df[INPUT_FEATURE_COLUMNS_NAMES[0]]})


class ModelMonitorIntegrationInferenceTableTest(ModelMonitorIntegrationTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._log_inference_model()

    def _row_values(self, row: snowpark.Row) -> dict[str, object]:
        return {str(key).lower(): value for key, value in row.as_dict().items()}

    def test_create_show_describe_and_delete(self) -> None:
        monitor_name = "inference_monitor"
        monitor = self._add_inference_table_monitor(monitor_name, self._inference_model_version)
        self.assertEqual(monitor.name, monitor_name.upper())

        shown = self.registry.show_model_monitors()
        self.assertEqual(len(shown), 1)
        shown_row = self._row_values(shown[0])
        self.assertEqual(shown_row["name"], monitor_name.upper())
        self.assertIsNone(shown_row["source"])
        self.assertEqual(shown_row["type"], "MODEL_VERSION_MONITOR")

        self.assertEqual(self.registry.get_monitor(name=monitor_name).name, monitor_name.upper())

        described = self._row_values(monitor._describe())
        self.assertIsNotNone(described["monitor_start_time"])
        columns = json.loads(described["columns"])
        self.assertEqual(columns["custom_metric_columns"], [])
        self.assertIsInstance(columns["segment_columns"], list)
        self.assertGreater(len(columns["numerical_columns"]), 0)

        monitor.suspend()
        self.assertEqual(self._row_values(self.registry.show_model_monitors()[0])["monitor_state"], "SUSPENDED")
        monitor.resume()
        self.assertIn(
            self._row_values(self.registry.show_model_monitors()[0])["monitor_state"],
            ["ACTIVE", "RUNNING"],
        )

        self.registry.delete_monitor(name=monitor_name)
        self.assertEqual(len(self.registry.show_model_monitors()), 0)

    def test_create_with_ground_truth(self) -> None:
        ground_truth_table = "inference_ground_truth"
        monitor_name = "inference_monitor_gt"
        self._session.sql(
            f"""CREATE OR REPLACE TABLE {self._db_name}.{self._schema_name}.{ground_truth_table}
            (id STRING, label FLOAT)"""
        ).collect()

        monitor = self._add_inference_table_monitor(
            monitor_name,
            self._inference_model_version,
            ground_truth=ground_truth_table,
            id_columns=["id"],
        )
        described = self._row_values(monitor._describe())
        ground_truth_value = described.get("ground_truth", described.get("fully_qualified_ground_truth"))
        self.assertIsNotNone(ground_truth_value)
        self.assertIn(ground_truth_table.upper(), str(ground_truth_value).upper())

    def test_add_and_drop_segment_column(self) -> None:
        monitor_name = "inference_monitor_segment"
        # A version-monitor segment is a STRING model input. The name is compared exactly, so it
        # matches the uppercased name sent by the client. A ground-truth table does not supply it.
        segment_column = "CUSTOMER_SEGMENT"
        ground_truth_table = "inference_segment_ground_truth"
        self._session.sql(
            f"""CREATE OR REPLACE TABLE {self._db_name}.{self._schema_name}.{ground_truth_table}
            (id STRING, label FLOAT)"""
        ).collect()
        self._session.sql(
            f"ALTER TABLE {self._db_name}.{self._schema_name}.{ground_truth_table} SET CHANGE_TRACKING = TRUE"
        ).collect()
        model_version = self.registry.log_model(
            model=_CustomerSegmentModel(custom_model.ModelContext()),
            model_name="TEST_INFERENCE_SEGMENT_MODEL",
            version_name="V1",
            sample_input_data=pd.DataFrame({INPUT_FEATURE_COLUMNS_NAMES[0]: [1.0], segment_column: ["A"]}),
            task=task.Task.TABULAR_REGRESSION,
            options={"embed_local_ml_library": True},
        )
        monitor = self._add_inference_table_monitor(
            monitor_name,
            model_version,
            ground_truth=ground_truth_table,
            id_columns=["id"],
            prediction_score_columns=["output"],
            actual_score_columns=["label"],
        )
        self.assertEqual(self._get_monitor_segment_columns(monitor_name), [])

        monitor.add_segment_column(segment_column)
        self.assertEqual(self._get_monitor_segment_columns(monitor_name), [segment_column])
        aggregation_status = self._aggregation_status(monitor_name)
        self.assertEqual(aggregation_status[f"SOURCE_AGGREGATED_{segment_column}"], "ACTIVE")
        self.assertEqual(aggregation_status[f"ACCURACY_AGGREGATED_{segment_column}"], "ACTIVE")

        monitor.drop_segment_column(segment_column)
        self.assertEqual(self._get_monitor_segment_columns(monitor_name), [])
        aggregation_status = self._aggregation_status(monitor_name)
        self.assertNotIn(f"SOURCE_AGGREGATED_{segment_column}", aggregation_status)
        self.assertNotIn(f"ACCURACY_AGGREGATED_{segment_column}", aggregation_status)

    def _aggregation_status(self, monitor_name: str) -> dict[str, object]:
        """Return the monitor's aggregation status.

        Args:
            monitor_name: Monitor to describe.

        Returns:
            Aggregation status keyed by aggregated table.

        Raises:
            AssertionError: If the described status is not an object.
        """
        described = self._row_values(self.registry.get_monitor(name=monitor_name)._describe())
        status = json.loads(str(described["aggregation_status"]))
        if not isinstance(status, dict):
            raise AssertionError(f"Monitor '{monitor_name}' aggregation status is missing.")
        return status


if __name__ == "__main__":
    absltest.main()
