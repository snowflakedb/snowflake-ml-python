import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model._client.model import batch_inference_specs
from tests.integ.snowflake.ml.registry.jobs import (
    registry_execute_inference_job_service_test_base,
)


class DemoModelEmptyStruct(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"], "empty_struct": [{}] * len(input)})


class TestExecuteInferenceJobServiceEmptyStructInteg(
    registry_execute_inference_job_service_test_base.ExecuteInferenceJobServiceTestBase
):
    def test_empty_struct_output(self) -> None:
        model = DemoModelEmptyStruct(custom_model.ModelContext())
        num_cols = 2

        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]

        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)

        model_output = model.predict(input_pandas_df[input_cols])

        input_df, _ = self._prepare_batch_inference_data(input_pandas_df, model_output)

        sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(name="C0", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="C1", dtype=model_signature.DataType.INT64),
            ],
            outputs=[
                model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.INT64),
                model_signature.FeatureGroupSpec(
                    name="empty_struct",
                    specs=[
                        model_signature.FeatureSpec(name="foo", dtype=model_signature.DataType.STRING),
                        model_signature.FeatureSpec(name="bar", dtype=model_signature.DataType.STRING),
                    ],
                ),
            ],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_execute_inference_job_service(
            model=model,
            sample_input_data=None,
            signatures={"predict": sig},
            X=input_df,
            output_spec=batch_inference_specs.Output(stage_location=output_stage_location),
            function_name="predict",
            job_name=job_name,
            replicas=1,
        )


if __name__ == "__main__":
    absltest.main()
