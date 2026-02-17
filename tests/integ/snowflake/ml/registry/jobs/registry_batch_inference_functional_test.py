import pandas as pd
from absl.testing import absltest

from snowflake.ml.jobs.manager import delete_job, get_job
from snowflake.ml.model import custom_model
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestBatchInferenceFunctionalInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def _prepare_test(self):
        model = DemoModel(custom_model.ModelContext())
        num_cols = 2

        # Create input data
        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]

        # Create pandas DataFrame
        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)

        # Generate expected predictions using the original model
        model_output = model.predict(input_pandas_df[input_cols])

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)

        # Create sample input data without INDEX column for model signature
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        return model, job_name, output_stage_location, input_df, expected_predictions, sp_df

    def test_mljob_api(self) -> None:
        model, job_name, output_stage_location, input_df, _, sp_df = self._prepare_test()

        replicas = 2

        job = self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=replicas),
            blocking=False,
        )

        self.assertEqual(job.id, f"{self._test_db}.{self._test_schema}.{job_name}")
        self.assertEqual(job.min_instances, 1)
        self.assertEqual(job.target_instances, replicas)
        self.assertIn(job.status, ["PENDING", "RUNNING"])
        self.assertEqual(job.name, job_name)

        # We just wanted to make sure the log functoin don't throw exceptions
        job.get_logs()
        job.show_logs()

        job.cancel()
        job.wait()  # wait until it is cancelled otherwise the job might be still pending
        self.assertEqual(job.status, "CANCELLED")

    def test_mljob_job_manager(self) -> None:
        model, job_name, output_stage_location, input_df, _, sp_df = self._prepare_test()

        job = self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name),
            blocking=False,
        )

        # the same job in another MLJob wrapper
        job2 = get_job(job.id, session=self.session)
        delete_job(job2)

        # the job will not be queryable any more
        try:
            job2.wait()
        except Exception as e:
            error_message = str(e)
            self.assertIn("does not exist or not authorized", error_message)

    def test_default_system_compute_pool(
        self,
    ) -> None:
        model, job_name, output_stage_location, input_df, _, sp_df = self._prepare_test()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, replicas=2, function_name="predict"),
        )


if __name__ == "__main__":
    absltest.main()
