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


class TestCustomModelBatchInferenceInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_end_to_end_pipeline(
        self,
        replicas: int,
        cpu_requests: str,
        num_workers: int,
    ) -> None:
        model = DemoModel(custom_model.ModelContext())
        num_cols = 2

        sp_df = self.session.create_dataframe(
            [[0] * num_cols, [1] * num_cols], schema=[f'"c{i}"' for i in range(num_cols)]
        )

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"

        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            input_spec=sp_df,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=num_workers,
            service_name=f"batch_inference_{name}",
            replicas=replicas,
        )


if __name__ == "__main__":
    absltest.main()
