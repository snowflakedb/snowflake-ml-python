import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from snowflake.ml.model._client.model import batch_inference_job_specs
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestBatchInferenceSmokeInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Single minimal batch inference test backing the smoke_test CI tier."""

    def test_batch_inference_smoke(self) -> None:
        """Minimal happy path: log a custom model, run a batch job, compare predictions."""
        model = DemoModel(custom_model.ModelContext())
        num_cols = 2

        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]

        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)
        model_output = model.predict(input_pandas_df[input_cols])

        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
