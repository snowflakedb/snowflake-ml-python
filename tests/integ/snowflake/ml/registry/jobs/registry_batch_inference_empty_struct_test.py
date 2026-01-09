import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class DemoModelEmptyStruct(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        # Return an empty struct (DataFrame with no columns)
        return pd.DataFrame({"output": input["C1"], "empty_struct": [{}] * len(input)})


@absltest.skip("Skipping empty struct test for now")
class TestBatchInferenceEmptyStructInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def test_empty_struct_output(self) -> None:
        model = DemoModelEmptyStruct(custom_model.ModelContext())
        num_cols = 2

        # Create input data
        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]

        # Create pandas DataFrame
        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)

        # Generate expected predictions using the model (empty struct)
        model_output = model.predict(input_pandas_df[input_cols])

        # Prepare input data and expected predictions using common function
        input_spec, _ = self._prepare_batch_inference_data(input_pandas_df, model_output)

        # Create fixed signature instead of inferring from sample data
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

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        # Test that batch inference can handle empty struct output without issues
        self._test_registry_batch_inference(
            model=model,
            sample_input_data=None,
            signatures={"predict": sig},
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            service_name=service_name,
            replicas=1,
            function_name="predict",
        )


if __name__ == "__main__":
    absltest.main()
