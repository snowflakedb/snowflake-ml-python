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

    def _poll_dag_completion(self, dag_name: str) -> tuple[str, str]:
        """Poll TASK_HISTORY until the DAG task completes or times out.

        Returns:
            A tuple of (state, error_message) for the batch inference task.
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

            # Check if the batch inference child task has completed
            for row in result:
                if "BATCH_INFERENCE" in row["NAME"].upper() and row["STATE"] in (
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
            # TODO: Pass defn directly with snowflake.core >= 1.12.0
            batch_inference_task = DAGTask("batch_inference", definition=defn.to_sql())
            data_prep_task >> batch_inference_task

        api_root = Root(self.session)
        schema = api_root.databases[self._test_db].schemas[self._test_schema]
        dag_op = DAGOperation(schema)

        dag_op.deploy(dag, mode="orReplace")
        dag_op.run(dag)

        # 5. Poll for completion
        state, error_msg = self._poll_dag_completion(self._dag_name)
        self.assertEqual(state, "SUCCEEDED", f"Batch inference DAG task {state}: {error_msg}")

        # 6. Verify output exists
        success_file = output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file}").collect()
        self.assertGreater(len(list_results), 0, f"No _SUCCESS file at: {success_file}")


if __name__ == "__main__":
    absltest.main()
