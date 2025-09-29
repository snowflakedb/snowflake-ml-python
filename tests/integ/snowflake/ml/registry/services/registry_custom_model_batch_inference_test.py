import uuid

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


@absltest.skip("Skipping batch inference integration test temporarily")
class TestCustomModelBatchInferenceInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": "1", "cpu_requests": None},
        {"gpu_requests": None, "cpu_requests": None},
    )
    def test_end_to_end_pipeline(
        self,
        gpu_requests: str,
        cpu_requests: str,
    ) -> None:
        model = DemoModel(custom_model.ModelContext())
        num_cols = 2

        sp_df = self.session.create_dataframe(
            [[0] * num_cols, [1] * num_cols], schema=[f'"c{i}"' for i in range(num_cols)]
        )

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"

        input_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/input/"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        # Write the test data to the input stage location
        sp_df.write.copy_into_location(
            location=input_stage_location, file_format_type="PARQUET", header=True, overwrite=True
        )

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            input_stage_location=input_stage_location,
            output_stage_location=output_stage_location,
            gpu_requests=gpu_requests,
            cpu_requests=cpu_requests,
            num_workers=1,
            service_name=f"batch_inference_{name}",
            replicas=2,
        )


if __name__ == "__main__":
    absltest.main()
