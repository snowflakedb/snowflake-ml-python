import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class WideInputModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        numeric_sum = input.select_dtypes(include=[np.number]).sum(axis=1)
        string_count = input.select_dtypes(include=[object, "string"]).count(axis=1)
        result = numeric_sum + string_count
        return pd.DataFrame({"output": result})


class TestBatchInferenceWideInputInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def test_custom_model_wide_input(self) -> None:
        n_samples = 10
        n_features = 600
        data = {}

        for i in range(n_features):
            if i % 3 == 0:
                col_name = f'"z_feature_{i:03d}"'
                data[col_name] = 1
            elif i % 3 == 1:
                col_name = f'"b_feature_{i:03d}"'
                data[col_name] = np.random.choice(["X", "Y", "Z"], n_samples)
            else:
                col_name = f"a_feature_{i:03d}"
                data[col_name] = 1

        train_df = pd.DataFrame(data)

        wide_model = WideInputModel(custom_model.ModelContext())

        model_output = wide_model.predict(train_df)

        input_spec, expected_predictions = self._prepare_batch_inference_data(train_df, model_output)

        service_name, output_stage_location, _ = self._prepare_service_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=wide_model,
            sample_input_data=train_df,
            service_name=service_name,
            output_stage_location=output_stage_location,
            X=input_spec,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
