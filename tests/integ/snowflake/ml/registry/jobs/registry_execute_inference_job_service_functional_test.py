import inspect
import uuid

import pandas as pd
from absl.testing import absltest

from snowflake import snowpark
from snowflake.connector import errors as connector_errors
from snowflake.ml.jobs.manager import delete_job, get_job
from snowflake.ml.model import ModelVersion, custom_model
from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)
from tests.integ.snowflake.ml.test_utils import test_env_utils


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestExecuteInferenceJobServiceFunctionalInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
    def _prepare_test(
        self,
    ) -> tuple[DemoModel, str, str, snowpark.DataFrame, pd.DataFrame, snowpark.DataFrame]:
        model = DemoModel(custom_model.ModelContext())
        num_cols = 2

        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]

        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)
        model_output = model.predict(input_pandas_df[input_cols])

        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        return model, job_name, output_stage_location, input_df, expected_predictions, sp_df

    def _log_demo_model(self, model: DemoModel, sample_input_data: snowpark.DataFrame) -> ModelVersion:
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{self._run_id}"
        return self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input_data,
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python"),
            ],
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

    def test_all_defaults(self) -> None:
        """Minimal happy path: only output_spec set, default SaveMode.ERROR."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            expected_predictions=expected_predictions,
        )

    def test_full_spec_override(self) -> None:
        """Every meaningful spec field populated plus function_name and a custom job_name."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()
        job_name = f"BATCH_INFERENCE_{uuid.uuid4().hex.upper()}"

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            input_spec=batch_inference_specs.Input(),
            output_spec=batch_inference_specs.Output(
                stage_location=output_stage_location,
                mode=batch_inference_specs.SaveMode.OVERWRITE,
            ),
            resources_spec=batch_inference_specs.Resources(
                cpu_requests="1",
                memory_requests="2Gi",
            ),
            inference_spec=batch_inference_specs.Inference(
                num_workers=2,
                max_batch_rows=1,
            ),
            image_build_spec=batch_inference_specs.ImageBuild(force_rebuild=False),
            function_name="predict",
            job_name=job_name,
            expected_predictions=expected_predictions,
        )

    def test_replicas(self) -> None:
        """Distributed inference with replicas=2 still produces matching predictions."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            replicas=2,
            expected_predictions=expected_predictions,
        )

    def test_sync_mode(self) -> None:
        """async_=False: the SQL command blocks until the job completes."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            async_=False,
            expected_predictions=expected_predictions,
        )

    def test_output_mode_error_rejects_existing(self) -> None:
        """ERROR mode rejects a run when the job-scoped output subdir already has files."""
        model, _, output_stage_location, input_df, _, sp_df = self._prepare_test()
        mv = self._log_demo_model(model, sp_df)

        # Output is job-scoped as <stage_location>/<job_name>/, so pin the job name and seed a
        # file in that subdir to give mode=error a non-empty destination to reject.
        job_name = f"BATCH_INFERENCE_{uuid.uuid4().hex.upper()}"
        job_scoped_output = output_stage_location.rstrip("/") + "/" + job_name + "/"
        self.session.sql(
            f"COPY INTO {job_scoped_output}stale.txt FROM (SELECT 'pre-existing' AS payload) "
            "FILE_FORMAT = (TYPE = 'CSV') OVERWRITE = TRUE SINGLE = TRUE"
        ).collect()

        with self.assertRaises(connector_errors.DatabaseError):
            mv._run_batch_v2(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
                job_name=job_name,
            )

    def test_output_mode_overwrite_replaces_existing(self) -> None:
        """OVERWRITE clears the job-scoped subdir and writes fresh rows, leaving sibling base files intact."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()
        mv = self._log_demo_model(model, sp_df)

        # Output is job-scoped as <stage_location>/<job_name>/, so pin the job name and seed a
        # stale file in that subdir for overwrite to clear.
        job_name = f"BATCH_INFERENCE_{uuid.uuid4().hex.upper()}"
        job_scoped_output = output_stage_location.rstrip("/") + "/" + job_name + "/"
        self.session.sql(
            f"COPY INTO {job_scoped_output}stale.txt FROM (SELECT 'pre-existing' AS payload) "
            "FILE_FORMAT = (TYPE = 'CSV') OVERWRITE = TRUE SINGLE = TRUE"
        ).collect()

        # Overwrite scopes deletion to the job subdir; seed a sibling under the base to verify it survives.
        sibling_file = output_stage_location.rstrip("/") + "/sibling_keep.txt"
        self.session.sql(
            f"COPY INTO {sibling_file} FROM (SELECT 'keep' AS payload) "
            "FILE_FORMAT = (TYPE = 'CSV') OVERWRITE = TRUE SINGLE = TRUE"
        ).collect()

        self._deploy_execute_inference_job_service(
            mv,
            X=input_df,
            output_spec=batch_inference_specs.Output(
                stage_location=output_stage_location,
                mode=batch_inference_specs.SaveMode.OVERWRITE,
            ),
            job_name=job_name,
            expected_predictions=expected_predictions,
        )

        # Overwrite must have cleared the seeded file before the job wrote its output.
        remaining = self.session.sql(f"LIST {job_scoped_output}").collect()
        self.assertFalse(
            any("stale.txt" in row["name"] for row in remaining),
            f"mode=overwrite should have cleared stale.txt, got: {[row['name'] for row in remaining]}",
        )

        # Files outside the job subdir must be left untouched.
        base_files = self.session.sql(f"LIST {output_stage_location}").collect()
        self.assertTrue(
            any("sibling_keep.txt" in row["name"] for row in base_files),
            f"mode=overwrite must not delete files outside the job subdir, got: {[row['name'] for row in base_files]}",
        )

    def test_mljob_api(self) -> None:
        model, job_name, output_stage_location, input_df, _, sp_df = self._prepare_test()

        replicas = 2

        job = self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            job_name=job_name,
            replicas=replicas,
            blocking=False,
        )

        self.assertEqual(job.id, f"{self._test_db}.{self._test_schema}.{job_name}")
        self.assertEqual(job.min_instances, 1)
        self.assertEqual(job.target_instances, replicas)
        self.assertIn(job.status, ["PENDING", "RUNNING"])
        self.assertEqual(job.name, job_name)

        job.get_logs()
        job.show_logs()

        job.cancel()
        job.wait()
        self.assertEqual(job.status, "CANCELLED")

    def test_mljob_job_manager(self) -> None:
        model, job_name, output_stage_location, input_df, _, sp_df = self._prepare_test()

        job = self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            job_name=job_name,
            blocking=False,
        )

        job2 = get_job(job.id, session=self.session)
        delete_job(job2)

        try:
            job2.wait()
        except Exception as e:
            self.assertIn("does not exist or not authorized", str(e))

    def test_default_system_compute_pool(self) -> None:
        model, job_name, output_stage_location, input_df, _, sp_df = self._prepare_test()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            function_name="predict",
            job_name=job_name,
            replicas=2,
        )

    def test_without_job_name(self) -> None:
        """When no job_name is provided, the server generates one with the BATCH_INFERENCE_ prefix."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

        job = self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            expected_predictions=expected_predictions,
        )

        self.assertTrue(
            job.name.startswith("BATCH_INFERENCE_"),
            f"Expected job name to start with 'BATCH_INFERENCE_', got '{job.name}'",
        )

    def test_custom_image_repo(self) -> None:
        """Succeeds when image_repo is explicitly set on ImageBuild."""
        model, _, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            image_build_spec=batch_inference_specs.ImageBuild(
                image_repo=".".join([self._test_db, self._test_schema, self._test_image_repo]),
            ),
            expected_predictions=expected_predictions,
        )

    def test_user_facing_logs(self) -> None:
        """With snowhouse logging enabled (prod default), only tee'd lines reach user-visible stdout.

        Enable SPCS_MODEL_ENABLE_SNOWHOUSE_LOGGING_FOR_BATCH_INFERENCE to exercise the production routing
        and verify the user-facing log surface.
        """
        param = self._SNOWHOUSE_LOGGING_PARAM
        self.session.sql(f"ALTER SESSION SET {param} = true").collect()
        try:
            model, job_name, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

            job = self._test_registry_execute_inference_job_service(
                model=model,
                sample_input_data=sp_df,
                X=input_df,
                output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
                job_name=job_name,
                expected_predictions=expected_predictions,
            )

            logs = job.get_logs(limit=-1)

            self.assertIn(
                "Starting inference.",
                logs,
                f"Expected user-facing log line 'Starting inference.' missing from job logs:\n{logs}",
            )

            self.assertNotIn(
                "Job configuration set:",
                logs,
                "Internal log 'Job configuration set:' leaked to user-facing stdout; "
                f"expected snowhouse-only routing.\nLogs:\n{logs}",
            )
        finally:
            self.session.sql(f"ALTER SESSION UNSET {param}").collect()


if __name__ == "__main__":
    absltest.main()
