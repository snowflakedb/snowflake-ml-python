import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestExecuteInferenceJobServiceCustomModelInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
    def _prepare_test(self):
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

    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_custom_model(
        self,
        replicas: int,
        cpu_requests: str,
        num_workers: int,
    ) -> None:
        model, job_name, output_stage_location, input_df, expected_predictions, sp_df = self._prepare_test()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            resources_spec=batch_inference_specs.Resources(cpu_requests=cpu_requests),
            inference_spec=batch_inference_specs.Inference(num_workers=num_workers),
            function_name="predict",
            job_name=job_name,
            replicas=replicas,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
