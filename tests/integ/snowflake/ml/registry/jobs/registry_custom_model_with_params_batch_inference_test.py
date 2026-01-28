from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import (
    InputSpec,
    JobSpec,
    OutputSpec,
    custom_model,
    model_signature,
)
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class DemoModelWithParams(custom_model.CustomModel):
    """Custom model that accepts inference parameters as arguments."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        output_format: str = "default",
        deterministic: bool = True,
    ) -> pd.DataFrame:
        """Predict with inference parameters.

        Args:
            input: Input features DataFrame
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k value (default: 50)
            output_format: Output format (default: "default")
            deterministic: Whether to use deterministic inference (default: True)

        Returns:
            DataFrame with output and parameter info
        """
        return pd.DataFrame(
            {
                "output": input["C1"],
                "temperature_used": [temperature] * len(input),
                "top_k_used": [top_k] * len(input),
                "output_format_used": [output_format] * len(input),
                "deterministic_used": [deterministic] * len(input),
            }
        )


class TestCustomModelWithParamsBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test batch inference with custom model that accepts parameters in model signature."""

    def setUp(self) -> None:
        super().setUp()
        # TODO: this is temporary since batch inference server image not released yet
        if not self._with_image_override():
            self.skipTest("Skipping params tests: image override environment variables not set.")

    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_custom_model_with_params(self, num_workers: int, replicas: int, cpu_requests: Optional[str]) -> None:
        """Test custom model with inference parameters passed via InputSpec."""
        model = DemoModelWithParams(custom_model.ModelContext())
        num_cols = 2

        # Create input data
        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]

        # Create pandas DataFrame
        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)

        # Define inference parameters with custom values for test
        test_temperature = 0.8
        test_top_k = 100
        test_output_format = "json"
        test_deterministic = False

        # Generate expected predictions using the model with custom params
        model_output = model.predict(
            input_pandas_df[input_cols],
            temperature=test_temperature,
            top_k=test_top_k,
            output_format=test_output_format,
            deterministic=test_deterministic,
        )

        # Prepare input data and expected predictions
        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)

        # Create sample input/output for signature inference
        sample_input = input_pandas_df.copy()
        sample_output = model_output.copy()

        # Define ParamSpec for inference parameters
        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=1.0,
            ),
            model_signature.ParamSpec(
                name="top_k",
                dtype=model_signature.DataType.INT64,
                default_value=50,
            ),
            model_signature.ParamSpec(
                name="output_format",
                dtype=model_signature.DataType.STRING,
                default_value="default",
            ),
            model_signature.ParamSpec(
                name="deterministic",
                dtype=model_signature.DataType.BOOL,
                default_value=True,
            ),
        ]

        # Infer signature with params
        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        # Create sample input data without INDEX column for model logging
        sp_df = self.session.create_dataframe(input_data, schema=input_cols)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        # Run batch inference with custom params via InputSpec
        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            signatures={"predict": sig},
            X=input_df,
            input_spec=InputSpec(
                params={
                    "temperature": test_temperature,
                    "top_k": test_top_k,
                    "output_format": test_output_format,
                    "deterministic": test_deterministic,
                }
            ),
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                cpu_requests=cpu_requests,
                num_workers=num_workers,
                replicas=replicas,
                function_name="predict",
            ),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
