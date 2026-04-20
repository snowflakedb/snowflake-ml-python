import json
import logging
import time
import uuid
from datetime import timedelta
from typing import Any

import pandas as pd
from absl.testing import absltest
from snowflake.core import Root
from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask

from snowflake.ml.model import custom_model
from snowflake.ml.model.batch import BatchInferenceDefinition, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

logger = logging.getLogger(__name__)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


# TODO(SNOW-3362189): Add integ tests for column_handling, params, vllm inference engine.
class TestBatchInferenceDefinitionInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    _DAG_POLL_INTERVAL_SEC = 15
    _DAG_POLL_MAX_ATTEMPTS = 120  # 30 min total

    def setUp(self) -> None:
        super().setUp()
        self._dag_name = f"test_dag_{uuid.uuid4().hex[:8]}"

    def _log_model(self, model: Any, sample_df: Any) -> Any:
        """Log a model to registry and return the ModelVersion."""
        name = f"model_{uuid.uuid4().hex[:8]}"
        version = f"ver_{self._run_id}"
        from tests.integ.snowflake.ml.test_utils import test_env_utils

        conda_deps = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        return self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_df,
            conda_dependencies=conda_deps,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

    def _poll_dag_completion(self, dag_name: str, task_name: str) -> tuple[str, str]:
        """Poll TASK_HISTORY until the DAG task completes or times out.

        Args:
            dag_name: Name of the DAG to poll.
            task_name: Uppercase name of the child task to wait on.

        Returns:
            A tuple of (state, error_message) for the target task.
        """
        for _ in range(self._DAG_POLL_MAX_ATTEMPTS):
            result = self.session.sql(
                f"""
                SELECT NAME, STATE, ERROR_MESSAGE
                FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
                    SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP())
                ))
                WHERE NAME ILIKE '%{dag_name}%'
                ORDER BY SCHEDULED_TIME DESC
                LIMIT 10
            """
            ).collect()

            for row in result:
                logger.info(f"  Task {row['NAME']}: {row['STATE']} error={row['ERROR_MESSAGE']}")

            # Check if the target child task has completed
            for row in result:
                if task_name in row["NAME"].upper() and row["STATE"] in (
                    "SUCCEEDED",
                    "FAILED",
                    "CANCELLED",
                ):
                    return row["STATE"], row["ERROR_MESSAGE"]

            time.sleep(self._DAG_POLL_INTERVAL_SEC)

        self.fail(
            f"DAG {dag_name} did not complete within {self._DAG_POLL_MAX_ATTEMPTS * self._DAG_POLL_INTERVAL_SEC}s"
        )

    def test_batch_inference_definition_dag(self) -> None:
        # 1. Create and register a simple model
        model = DemoModel(custom_model.ModelContext())
        input_data = [[0, 0], [1, 1]]
        input_cols = ["C1", "C2"]
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)

        mv = self._log_model(model, sp_df)

        # 2. Create input DataFrame and output stage
        input_df = self.session.create_dataframe(input_data, schema=input_cols)
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        # 3. Construct BatchInferenceDefinition
        warehouse = self._TEST_SPCS_WH
        compute_pool = self._TEST_CPU_COMPUTE_POOL
        image_repo = ".".join([self._test_db, self._test_schema, self._test_image_repo])

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=input_df,
            compute_pool=compute_pool,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                warehouse=warehouse,
                function_name="predict",
                image_repo=image_repo,
            ),
        )

        # 4. Create and deploy DAG with data_preparation >> batch_inference
        data_prep_sql = "SELECT 'data_preparation done'"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=warehouse,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition=data_prep_sql)
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            data_prep_task >> batch_inference_task

        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        dag_op.run(dag)

        # 5. Poll for completion
        state, error_msg = self._poll_dag_completion(self._dag_name, task_name="BATCH_INFERENCE")
        self.assertEqual(state, "SUCCEEDED", f"Batch inference DAG task {state}: {error_msg}")

        # 6. Verify output exists
        success_file = output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file}").collect()
        self.assertGreater(len(list_results), 0, f"No _SUCCESS file at: {success_file}")

    @absltest.skip("Requires account-level image release/override.")
    def test_batch_inference_definition_dag_return_value(self) -> None:
        """Verify that a successor task can read the batch inference task return value."""
        # 1. Create and register a simple model
        model = DemoModel(custom_model.ModelContext())
        input_data = [[0, 0], [1, 1]]
        input_cols = ["C1", "C2"]
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)

        mv = self._log_model(model, sp_df)

        # 2. Create input DataFrame and output stage
        input_df = self.session.create_dataframe(input_data, schema=input_cols)
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        # 3. Construct BatchInferenceDefinition
        warehouse = self._TEST_SPCS_WH
        compute_pool = self._TEST_CPU_COMPUTE_POOL
        image_repo = ".".join([self._test_db, self._test_schema, self._test_image_repo])

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=input_df,
            compute_pool=compute_pool,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                warehouse=warehouse,
                function_name="predict",
                image_repo=image_repo,
            ),
        )

        # 4. Create result table to capture the return value from the successor task
        result_table = f"{self._test_db}.{self._test_schema}.dag_result_{uuid.uuid4().hex[:8]}"
        self.session.sql(f"CREATE TABLE {result_table} (return_value VARCHAR)").collect()

        # 5. Create and deploy DAG: data_preparation >> batch_inference >> verify_return_value
        data_prep_sql = "SELECT 'data_preparation done'"
        verify_sql = f"INSERT INTO {result_table} (return_value)" f" SELECT SYSTEM$GET_PREDECESSOR_RETURN_VALUE()"

        dag = DAG(
            self._dag_name,
            schedule=timedelta(days=1),
            warehouse=warehouse,
            stage_location=f"@{self._test_db}.{self._test_schema}.{self._test_stage}",
        )

        with dag:
            data_prep_task = DAGTask("data_preparation", definition=data_prep_sql)
            batch_inference_task = DAGTask("batch_inference", definition=defn)
            verify_task = DAGTask("verify_return_value", definition=verify_sql)
            data_prep_task >> batch_inference_task >> verify_task

        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        dag_op.run(dag)

        # 6. Poll for verify_return_value task completion
        state, error_msg = self._poll_dag_completion(self._dag_name, task_name="VERIFY_RETURN_VALUE")
        self.assertEqual(state, "SUCCEEDED", f"Verify return value task {state}: {error_msg}")

        # 7. Assert the return value contains the expected output_stage_location
        rows = self.session.sql(f"SELECT return_value FROM {result_table}").collect()
        self.assertLen(rows, 1, f"Expected 1 row in result table, got {len(rows)}")
        result = json.loads(rows[0]["RETURN_VALUE"])
        self.assertEqual(result["output_stage_location"], output_stage_location)


if __name__ == "__main__":
    absltest.main()
