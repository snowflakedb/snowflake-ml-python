import json
import time

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml.model import custom_model, task
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.services.registry_model_deployment_test_base import (
    RegistryModelDeploymentTestBase,
)
from tests.integ.snowflake.ml.test_utils import test_env_utils

_REFRESH_INTERVAL = "1 minute"
_INFERENCE_WAIT_SECONDS = 120
# Refresh is one minute. Allow several intervals before treating a missing COUNT as a failure.
_METRIC_WAIT_SECONDS = 600
_FEATURE_COLUMN = "FEATURE"
_SEGMENT_COLUMN = "SEGMENT"
_ID_COLUMN = "ID"
_LABEL_COLUMN = "label"
_SEGMENT_A = "A"
_SEGMENT_B = "B"
_GROUND_TRUTH_TABLE = "inference_ground_truth"


class _FeatureSegmentModel(custom_model.CustomModel):
    """Predicts the feature value and accepts a segment column on the request."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"OUTPUT": input_df[_FEATURE_COLUMN].tolist()})


@pytest.mark.spcs_deployment_image
class ModelMonitorInferenceTableE2ETest(RegistryModelDeploymentTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.session.sql(f"ALTER DATABASE {self._test_db} SET DATA_RETENTION_TIME_IN_DAYS = 1").collect()
        self.registry = registry.Registry(
            self.session,
            database_name=self._test_db,
            schema_name=self._test_schema,
            options={"enable_monitoring": True},
        )

    @absltest.skip("Inference-table autocapture e2e test is flaky; re-enable once it is stable.")
    def test_inference_run(self) -> None:
        service_name = f"monitor_inference_{self._run_id}"
        scored_rows = pd.DataFrame(
            {
                _FEATURE_COLUMN: [1.0, 2.0, 3.0],
                _SEGMENT_COLUMN: [_SEGMENT_A, _SEGMENT_A, _SEGMENT_B],
                _ID_COLUMN: ["row-0", "row-1", "row-2"],
            }
        )
        labels = [1.0, 2.0, 3.0]
        model = _FeatureSegmentModel(custom_model.ModelContext())
        sample_input = scored_rows[[_FEATURE_COLUMN, _SEGMENT_COLUMN]].head(1)
        # Captured rows must carry the ground-truth id. The deployment service check omits that id, so the
        # scored request is sent separately.
        model_version = self.registry.log_model(
            model=model,
            model_name=f"model_{self._testMethodName}",
            version_name=f"ver_{self._run_id}",
            sample_input_data=sample_input,
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
            ],
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
            task=task.Task.TABULAR_REGRESSION,
        )
        model_version = self._deploy_model_service(
            mv=model_version,
            prediction_assert_fns={
                "predict": (sample_input, lambda result: self.assertEqual(len(result), len(sample_input)))
            },
            service_name=service_name,
            autocapture=True,
            skip_rest_api_test=True,
        )
        self._create_ground_truth(scored_rows[_ID_COLUMN].tolist(), labels, scored_rows[_SEGMENT_COLUMN].tolist())
        self._predict_with_ground_truth_ids(model_version, scored_rows)
        # Rows must be recorded before the monitor is created so the metric wait covers aggregation only.
        self._wait_for_inference_rows(model_version.model_name, service_name, expected_count=len(scored_rows))

        monitor_name = "inference_metric_monitor"
        warehouse_name = self.session.get_current_warehouse() or self._TEST_SPCS_WH
        monitor = self.registry.add_monitor(
            name=monitor_name,
            source_config=model_monitor_config.ModelMonitorSourceConfig._from_inference_table(
                ground_truth=_GROUND_TRUTH_TABLE,
                id_columns=[_ID_COLUMN],
                prediction_score_columns=["output"],
                actual_score_columns=[_LABEL_COLUMN],
            ),
            model_monitor_config=model_monitor_config.ModelMonitorConfig(
                model_version=model_version,
                model_function_name="predict",
                background_compute_warehouse_name=warehouse_name,
                refresh_interval=_REFRESH_INTERVAL,
            ),
        )
        # A version-monitor segment is a model input added after create; the server does not
        # track it correctly when supplied to CREATE MODEL MONITOR.
        monitor.add_segment_column(_SEGMENT_COLUMN)
        self._assert_described_column(monitor_name, key="numerical_columns", column_name=_FEATURE_COLUMN)
        self._assert_described_column(monitor_name, key="segment_columns", column_name=_SEGMENT_COLUMN)
        self._wait_for_monitor_count(monitor_name, _FEATURE_COLUMN, expected_count=len(scored_rows))
        self._wait_for_monitor_count(
            monitor_name,
            _FEATURE_COLUMN,
            expected_count=sum(segment == _SEGMENT_A for segment in scored_rows[_SEGMENT_COLUMN]),
            segment_column=_SEGMENT_COLUMN,
            segment_value=_SEGMENT_A,
        )
        self._wait_for_performance_join(monitor_name, expected_count_used=len(scored_rows))

    def _create_ground_truth(self, row_ids: list[str], labels: list[float], segments: list[str]) -> None:
        table_name = f"{self._test_db}.{self._test_schema}.{_GROUND_TRUTH_TABLE}"
        self.session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                {_ID_COLUMN} STRING,
                {_LABEL_COLUMN} FLOAT,
                {_SEGMENT_COLUMN} STRING
            )
            CHANGE_TRACKING = TRUE
            """
        ).collect()
        values = ", ".join(
            f"('{row_id}', {label}, '{segment}')"
            for row_id, label, segment in zip(row_ids, labels, segments, strict=True)
        )
        self.session.sql(f"INSERT INTO {table_name} VALUES {values}").collect()

    def _predict_with_ground_truth_ids(
        self, model_version: model_version_impl.ModelVersion, scored_rows: pd.DataFrame
    ) -> None:
        endpoint = self._ensure_ingress_url(model_version)
        payload = {
            "dataframe_split": json.loads(scored_rows.to_json(orient="split")),
            "extra_columns": [_ID_COLUMN],
        }
        result = self._inference_using_rest_api(payload, endpoint=endpoint, target_method="predict")
        self.assertEqual(len(result), len(scored_rows))

    def _wait_for_inference_rows(self, model_name: str, service_name: str, *, expected_count: int) -> None:
        deadline = time.monotonic() + _INFERENCE_WAIT_SECONDS
        query = (
            "SELECT COUNT(*) AS ROW_COUNT " f"FROM TABLE(INFERENCE_TABLE('{model_name}', SERVICE => '{service_name}'))"
        )
        last_count = 0
        while True:
            rows = self.session.sql(query).collect()
            last_count = int(self._row_value(rows[0], "row_count")) if rows else 0
            if last_count == expected_count:
                return
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Expected {expected_count} inference rows for '{model_name}', found {last_count}."
                )
            time.sleep(1)

    def _described_monitor(self, monitor_name: str) -> dict[str, object]:
        return {
            str(key).lower(): value
            for key, value in self.registry.get_monitor(name=monitor_name)._describe().as_dict().items()
        }

    def _describe_columns(self, monitor_name: str) -> dict[str, object]:
        columns = json.loads(str(self._described_monitor(monitor_name)["columns"]))
        if not isinstance(columns, dict):
            raise AssertionError(f"Monitor '{monitor_name}' column metadata is missing.")
        return columns

    def _describe_without_columns(self, monitor_name: str) -> str:
        described = {key: value for key, value in self._described_monitor(monitor_name).items() if key != "columns"}
        return f"DESC {described}"

    def _assert_described_column(self, monitor_name: str, *, key: str, column_name: str) -> None:
        described = self._describe_columns(monitor_name).get(key)
        names = [str(name) for name in described] if isinstance(described, list) else []
        self.assertIn(column_name, names)

    def _wait_for_monitor_count(
        self,
        monitor_name: str,
        column_name: str,
        *,
        expected_count: int,
        segment_column: str | None = None,
        segment_value: str | None = None,
    ) -> None:
        segment_arg = ""
        segment_label = ""
        if segment_column is not None and segment_value is not None:
            segment_filter = json.dumps({"SEGMENTS": [{"column": segment_column, "value": segment_value}]})
            segment_arg = f", '{segment_filter}'"
            segment_label = f" segment '{segment_value}'"
        query = f"""
            SELECT METRIC_VALUE
            FROM TABLE(MODEL_MONITOR_STAT_METRIC(
                '{monitor_name}',
                'COUNT',
                '{column_name}',
                '1 DAY',
                DATEADD('day', -14, CURRENT_TIMESTAMP()),
                CURRENT_TIMESTAMP()
                {segment_arg}
            ))
        """
        deadline = time.monotonic() + _METRIC_WAIT_SECONDS
        last_values: list[object] = []
        while True:
            rows = self.session.sql(query).collect()
            last_values = [self._row_value(row, "metric_value") for row in rows]
            observed = sum(float(value) for value in last_values if value is not None)
            if last_values and observed == expected_count:
                return
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Monitor '{monitor_name}' count for '{column_name}'{segment_label} was {last_values}, "
                    f"expected {expected_count}. {self._describe_without_columns(monitor_name)}"
                )
            time.sleep(5)

    def _wait_for_performance_join(self, monitor_name: str, *, expected_count_used: int) -> None:
        query = f"""
            SELECT METRIC_VALUE, COUNT_USED
            FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC(
                '{monitor_name}',
                'MAE',
                '1 DAY',
                DATEADD('day', -14, CURRENT_TIMESTAMP()),
                CURRENT_TIMESTAMP()
            ))
        """
        deadline = time.monotonic() + _METRIC_WAIT_SECONDS
        last_counts: list[object] = []
        while True:
            rows = self.session.sql(query).collect()
            last_counts = [self._row_value(row, "count_used") for row in rows]
            metric_values = [self._row_value(row, "metric_value") for row in rows]
            observed = sum(float(value) for value in last_counts if value is not None)
            has_metric = any(value is not None for value in metric_values)
            if has_metric and observed == expected_count_used:
                return
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Monitor '{monitor_name}' performance join count was {last_counts}, "
                    f"expected {expected_count_used}. {self._describe_without_columns(monitor_name)}"
                )
            time.sleep(5)

    def _row_value(self, row: snowpark.Row, column_name: str) -> object:
        values = {str(key).lower(): value for key, value in row.as_dict().items()}
        return values[column_name]


if __name__ == "__main__":
    absltest.main()
