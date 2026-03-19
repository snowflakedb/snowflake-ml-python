"""Integration tests for pip-only model packaging and deployment."""

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class PipOnlyModel(custom_model.CustomModel):
    """A simple custom model with only pip dependencies."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        # Use a pip-only package to verify it's installed correctly
        import requests  # noqa: F401 - imported to verify pip package is available

        return pd.DataFrame({"output": input["value"] * 2})


class TestRegistryPipOnlyModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for pip-only model deployment using dockerfile_template_pip."""

    def test_pip_only_model(self) -> None:
        """Test end-to-end deployment of a pip-only model using inference_image_builder."""
        if not self._has_image_override():
            self.skipTest("Skipping inference_image_builder test: image override not enabled.")

        test_input = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        pip_only_model = PipOnlyModel(custom_model.ModelContext())

        # Enable pip-only packaging via import
        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        # Test deployment using the pip-only path
        self._test_registry_model_deployment(
            model=pip_only_model,
            sample_input_data=test_input,
            prediction_assert_fns={
                "predict": (
                    test_input,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pip_only_model.predict(test_input),
                        check_dtype=False,
                    ),
                ),
            },
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
            use_inference_image_builder=True,
        )


if __name__ == "__main__":
    absltest.main()
