import pandas as pd
from absl.testing import absltest

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


class TestRegistryCustomModelDeploymentSmokeInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Single minimal model deployment test backing the smoke_test CI tier."""

    def test_custom_model_deployment_smoke(self) -> None:
        """Minimal happy path: deploy a custom model to SPCS and check its predictions."""
        model = DemoModel(custom_model.ModelContext())
        num_cols = 2

        sp_df = self.session.create_dataframe(
            [[0] * num_cols, [1] * num_cols], schema=[f'"c{i}"' for i in range(num_cols)]
        )
        expected_predictions = pd.DataFrame([[0], [1]], columns=["output"])

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df.to_pandas(),
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        expected_predictions,
                        check_dtype=False,
                    ),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
