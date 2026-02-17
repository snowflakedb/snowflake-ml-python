import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

# Signature with params for testing param validation (used with predict_with_params method)
_SIGNATURE_WITH_PARAMS = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C0"),
        model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="C1"),
    ],
    outputs=[
        model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output"),
        model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="output_temperature"),
        model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output_max_tokens"),
    ],
    params=[
        model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.FLOAT, default_value=0.5),
        model_signature.ParamSpec(name="max_tokens", dtype=model_signature.DataType.INT64, default_value=100),
    ],
)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})

    @custom_model.inference_api
    def predict_with_params(
        self, input: pd.DataFrame, *, temperature: float = 0.5, max_tokens: int = 100
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "output": input["C1"],
                "output_temperature": [temperature] * len(input),
                "output_max_tokens": [max_tokens] * len(input),
            }
        )

    @custom_model.partitioned_api
    def predict_partitioned(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestBatchInferenceFailureModeInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def test_run_batch_partitioned_model_raises_error(self) -> None:
        """Test that run_batch raises ValueError for partitioned models."""
        model = DemoModel(custom_model.ModelContext())
        input_df = self.session.create_dataframe([[0, 1]], schema=["C0", "C1"])

        mv = self.registry.log_model(
            model=model,
            model_name="model_test_run_batch_partitioned_model_raises_error",
            version_name=f"ver_{self._run_id}",
            sample_input_data=input_df,
            options={
                "embed_local_ml_library": True,
                "method_options": {"predict_partitioned": {"function_type": "TABLE_FUNCTION"}},
            },
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        with self.assertRaisesRegex(ValueError, r"partitioned model function.*not supported"):
            mv.run_batch(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(job_name=job_name, function_name="predict_partitioned"),
            )

    def test_run_batch_unknown_param_raises_error(self) -> None:
        """Test that run_batch raises ValueError for unknown param names."""
        model = DemoModel(custom_model.ModelContext())
        input_df = self.session.create_dataframe([[0, 1]], schema=["C0", "C1"])

        mv = self.registry.log_model(
            model=model,
            model_name="model_test_run_batch_unknown_param_raises_error",
            version_name=f"ver_{self._run_id}",
            sample_input_data=input_df,
            signatures={"predict_with_params": _SIGNATURE_WITH_PARAMS},
            options={"embed_local_ml_library": True},
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        with self.assertRaisesRegex(ValueError, r"Unknown parameter.*unknown_param"):
            mv.run_batch(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                input_spec=InputSpec(params={"unknown_param": 0.5}),
                job_spec=JobSpec(job_name=job_name, function_name="predict_with_params"),
            )

    def test_run_batch_invalid_param_type_raises_error(self) -> None:
        """Test that run_batch raises ValueError for invalid param types."""
        model = DemoModel(custom_model.ModelContext())
        input_df = self.session.create_dataframe([[0, 1]], schema=["C0", "C1"])

        mv = self.registry.log_model(
            model=model,
            model_name="model_test_run_batch_invalid_param_type_raises_error",
            version_name=f"ver_{self._run_id}",
            sample_input_data=input_df,
            signatures={"predict_with_params": _SIGNATURE_WITH_PARAMS},
            options={"embed_local_ml_library": True},
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        with self.assertRaisesRegex(ValueError, r"not compatible with dtype"):
            mv.run_batch(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                input_spec=InputSpec(params={"max_tokens": "not_an_int"}),
                job_spec=JobSpec(job_name=job_name, function_name="predict_with_params"),
            )


if __name__ == "__main__":
    absltest.main()
