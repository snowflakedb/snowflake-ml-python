import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestCustomModelBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
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
        input_spec, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)

        # Create sample input data without INDEX column for model signature
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        return model, service_name, output_stage_location, input_spec, expected_predictions, sp_df

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
        model, service_name, output_stage_location, input_spec, expected_predictions, sp_df = self._prepare_test()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=num_workers,
            service_name=service_name,
            replicas=replicas,
            function_name="predict",
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
