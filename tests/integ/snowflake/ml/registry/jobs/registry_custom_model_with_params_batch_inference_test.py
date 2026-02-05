from typing import Any, Optional

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import (
    InputSpec,
    JobSpec,
    OutputSpec,
    custom_model,
    model_signature,
)
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

# Default values for all parameters
_DEFAULT_INT64 = 1
_DEFAULT_DOUBLE = 1.5
_DEFAULT_BOOL = True
_DEFAULT_STRING = "default"
# TODO: Add bytes and timestamp types after the params fix
# _DEFAULT_BYTES = b""
# _DEFAULT_TIMESTAMP = datetime.datetime(2026, 1, 1)

# Override values for testing
_OVERRIDE_INT64 = 42
_OVERRIDE_DOUBLE = 3.14
_OVERRIDE_BOOL = False
_OVERRIDE_STRING = "overridden"
# TODO: Add bytes and timestamp types after the params fix
# _OVERRIDE_BYTES = b"test"
# _OVERRIDE_TIMESTAMP = datetime.datetime(2026, 6, 15, 12, 30, 0)


class DemoModelWithParams(custom_model.CustomModel):
    """Custom model that accepts inference parameters as arguments."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        int64_type: int = _DEFAULT_INT64,
        double_type: float = _DEFAULT_DOUBLE,
        bool_type: bool = _DEFAULT_BOOL,
        string_type: str = _DEFAULT_STRING,
        # TODO: Add bytes and timestamp types after the params fix
        # bytes_type: bytes = _DEFAULT_BYTES,
        # timestamp_ntz_type: datetime.datetime = _DEFAULT_TIMESTAMP,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "output": input["C1"],
                "int64_type": [int64_type] * len(input),
                "double_type": [double_type] * len(input),
                "bool_type": [bool_type] * len(input),
                "string_type": [string_type] * len(input),
                # "bytes_type": [bytes_type] * len(input),
                # "timestamp_ntz_type": [timestamp_ntz_type] * len(input),
            }
        )


class TestCustomModelWithParamsBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test batch inference with custom model that accepts parameters in model signature."""

    def _get_param_specs(self) -> list:
        """Return the list of ParamSpec definitions for the model signature."""
        return [
            model_signature.ParamSpec(
                name="int64_type", dtype=model_signature.DataType.INT64, default_value=_DEFAULT_INT64
            ),
            model_signature.ParamSpec(
                name="double_type", dtype=model_signature.DataType.DOUBLE, default_value=_DEFAULT_DOUBLE
            ),
            model_signature.ParamSpec(
                name="bool_type", dtype=model_signature.DataType.BOOL, default_value=_DEFAULT_BOOL
            ),
            model_signature.ParamSpec(
                name="string_type", dtype=model_signature.DataType.STRING, default_value=_DEFAULT_STRING
            ),
            # TODO: Add bytes and timestamp types after the params fix
            # model_signature.ParamSpec(
            #     name="bytes_type", dtype=model_signature.DataType.BYTES, default_value=_DEFAULT_BYTES
            # ),
            # model_signature.ParamSpec(
            #     name="timestamp_ntz_type",
            #     dtype=model_signature.DataType.TIMESTAMP_NTZ,
            #     default_value=_DEFAULT_TIMESTAMP,
            # ),
        ]

    def _run_params_test(self, input_spec_params: Optional[dict[str, Any]], model_call_params: dict[str, Any]) -> None:
        """Helper to run a batch inference test with the given params configuration.

        Args:
            input_spec_params: Params to pass to InputSpec (None means no params override).
            model_call_params: Params to pass when calling model.predict() for expected output.
        """
        model = DemoModelWithParams(custom_model.ModelContext())
        num_cols = 2

        input_data = [[0] * num_cols, [1] * num_cols]
        input_cols = [f"C{i}" for i in range(num_cols)]
        input_pandas_df = pd.DataFrame(input_data, columns=input_cols)

        model_output = model.predict(input_pandas_df[input_cols], **model_call_params)

        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)
        sample_input = input_pandas_df.copy()
        sample_output = model_output.copy()

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=self._get_param_specs(),
        )

        sp_df = self.session.create_dataframe(input_data, schema=input_cols)
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        input_spec = InputSpec(params=input_spec_params) if input_spec_params else InputSpec()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            signatures={"predict": sig},
            X=input_df,
            input_spec=input_spec,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                function_name="predict",
            ),
            expected_predictions=expected_predictions,
        )

    def test_all_default_params(self) -> None:
        """Test with all parameters using their default values (no overrides in InputSpec)."""
        self._run_params_test(
            input_spec_params=None,
            model_call_params={},  # Use all defaults
        )

    def test_all_overwritten_params(self) -> None:
        """Test with all parameters overwritten via InputSpec."""
        override_params = {
            "int64_type": _OVERRIDE_INT64,
            "double_type": _OVERRIDE_DOUBLE,
            "bool_type": _OVERRIDE_BOOL,
            "string_type": _OVERRIDE_STRING,
            # TODO: Add bytes and timestamp types after the params fix
            # "bytes_type": _OVERRIDE_BYTES,
            # "timestamp_ntz_type": _OVERRIDE_TIMESTAMP,
        }
        self._run_params_test(
            input_spec_params=override_params,
            model_call_params=override_params,
        )

    def test_mixed_params(self) -> None:
        """Test with some parameters using defaults and some overwritten."""
        # Override only int64_type and string_type
        override_params = {
            "int64_type": _OVERRIDE_INT64,
            "string_type": _OVERRIDE_STRING,
        }
        self._run_params_test(
            input_spec_params=override_params,
            model_call_params=override_params,  # Other params use defaults automatically
        )


if __name__ == "__main__":
    absltest.main()
