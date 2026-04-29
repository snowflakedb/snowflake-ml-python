import datetime
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

# Default values for all parameters
_DEFAULT_INT8 = 1
_DEFAULT_INT16 = 2
_DEFAULT_INT32 = 3
_DEFAULT_INT64 = 4
_DEFAULT_UINT8 = 5
_DEFAULT_UINT16 = 6
_DEFAULT_UINT32 = 7
_DEFAULT_UINT64 = 8
_DEFAULT_FLOAT = 1.5
_DEFAULT_DOUBLE = 2.5
_DEFAULT_BOOL = True
_DEFAULT_STRING = "default"
_DEFAULT_BYTES = b"default"
_DEFAULT_TIMESTAMP = datetime.datetime(2024, 1, 1, 12, 0, 0)
_DEFAULT_WEIGHTS = [1.0, 2.0, 3.0]
_DEFAULT_NESTED_LIST = [[1, 2], [3, 4]]

# Override values for testing
_OVERRIDE_INT8 = 10
_OVERRIDE_INT16 = 200
_OVERRIDE_INT32 = 3000
_OVERRIDE_INT64 = 40000
_OVERRIDE_UINT8 = 15
_OVERRIDE_UINT16 = 300
_OVERRIDE_UINT32 = 4000
_OVERRIDE_UINT64 = 50000
_OVERRIDE_FLOAT = 1.25
_OVERRIDE_DOUBLE = 2.75
_OVERRIDE_BOOL = False
_OVERRIDE_STRING = "overridden"
_OVERRIDE_BYTES = b"hello"
_OVERRIDE_TIMESTAMP = datetime.datetime(2025, 6, 15, 10, 30, 0)
_OVERRIDE_WEIGHTS = [0.5, 1.5, 2.5]
_OVERRIDE_NESTED_LIST = [[5, 6], [7, 8]]


class DemoModelWithParams(custom_model.CustomModel):
    """Custom model that accepts inference parameters as arguments."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        # Signed integers
        int8_param: int = _DEFAULT_INT8,
        int16_param: int = _DEFAULT_INT16,
        int32_param: int = _DEFAULT_INT32,
        int64_param: int = _DEFAULT_INT64,
        # Unsigned integers
        uint8_param: int = _DEFAULT_UINT8,
        uint16_param: int = _DEFAULT_UINT16,
        uint32_param: int = _DEFAULT_UINT32,
        uint64_param: int = _DEFAULT_UINT64,
        # Floating point
        float_param: float = _DEFAULT_FLOAT,
        double_param: float = _DEFAULT_DOUBLE,
        # Other scalars
        bool_param: bool = _DEFAULT_BOOL,
        string_param: str = _DEFAULT_STRING,
        # Extended types
        bytes_param: bytes = _DEFAULT_BYTES,
        timestamp_param: datetime.datetime = _DEFAULT_TIMESTAMP,
        weights_param: list[float] = _DEFAULT_WEIGHTS,
        nested_list: list[list[int]] = _DEFAULT_NESTED_LIST,
    ) -> pd.DataFrame:
        bytes_str = bytes_param.decode("utf-8") if isinstance(bytes_param, bytes) else str(bytes_param)
        timestamp_str = (
            timestamp_param.isoformat() if isinstance(timestamp_param, datetime.datetime) else str(timestamp_param)
        )
        weights_sum = sum(weights_param) if isinstance(weights_param, list) else float(weights_param)

        return pd.DataFrame(
            {
                "output": input["C1"],
                "received_int8": [int8_param] * len(input),
                "received_int16": [int16_param] * len(input),
                "received_int32": [int32_param] * len(input),
                "received_int64": [int64_param] * len(input),
                "received_uint8": [uint8_param] * len(input),
                "received_uint16": [uint16_param] * len(input),
                "received_uint32": [uint32_param] * len(input),
                "received_uint64": [uint64_param] * len(input),
                "received_float": [float_param] * len(input),
                "received_double": [double_param] * len(input),
                "received_bool": [bool_param] * len(input),
                "received_string": [string_param] * len(input),
                "received_bytes_str": [bytes_str] * len(input),
                "received_timestamp_str": [timestamp_str] * len(input),
                "received_weights_sum": [weights_sum] * len(input),
                "received_weights_len": [len(weights_param) if isinstance(weights_param, list) else 1] * len(input),
                "received_nested_list": [nested_list] * len(input),
            }
        )


class TestCustomModelWithParamsBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test batch inference with custom model that accepts parameters in model signature."""

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def _get_param_specs(self) -> list[model_signature.ParamSpec]:
        """Return the list of ParamSpec definitions for the model signature."""
        return [
            model_signature.ParamSpec(
                name="int8_param", dtype=model_signature.DataType.INT8, default_value=_DEFAULT_INT8
            ),
            model_signature.ParamSpec(
                name="int16_param", dtype=model_signature.DataType.INT16, default_value=_DEFAULT_INT16
            ),
            model_signature.ParamSpec(
                name="int32_param", dtype=model_signature.DataType.INT32, default_value=_DEFAULT_INT32
            ),
            model_signature.ParamSpec(
                name="int64_param", dtype=model_signature.DataType.INT64, default_value=_DEFAULT_INT64
            ),
            model_signature.ParamSpec(
                name="uint8_param", dtype=model_signature.DataType.UINT8, default_value=_DEFAULT_UINT8
            ),
            model_signature.ParamSpec(
                name="uint16_param", dtype=model_signature.DataType.UINT16, default_value=_DEFAULT_UINT16
            ),
            model_signature.ParamSpec(
                name="uint32_param", dtype=model_signature.DataType.UINT32, default_value=_DEFAULT_UINT32
            ),
            model_signature.ParamSpec(
                name="uint64_param", dtype=model_signature.DataType.UINT64, default_value=_DEFAULT_UINT64
            ),
            model_signature.ParamSpec(
                name="float_param", dtype=model_signature.DataType.FLOAT, default_value=_DEFAULT_FLOAT
            ),
            model_signature.ParamSpec(
                name="double_param", dtype=model_signature.DataType.DOUBLE, default_value=_DEFAULT_DOUBLE
            ),
            model_signature.ParamSpec(
                name="bool_param", dtype=model_signature.DataType.BOOL, default_value=_DEFAULT_BOOL
            ),
            model_signature.ParamSpec(
                name="string_param", dtype=model_signature.DataType.STRING, default_value=_DEFAULT_STRING
            ),
            model_signature.ParamSpec(
                name="bytes_param", dtype=model_signature.DataType.BYTES, default_value=_DEFAULT_BYTES
            ),
            model_signature.ParamSpec(
                name="timestamp_param",
                dtype=model_signature.DataType.TIMESTAMP_NTZ,
                default_value=_DEFAULT_TIMESTAMP,
            ),
            model_signature.ParamSpec(
                name="weights_param",
                dtype=model_signature.DataType.DOUBLE,
                default_value=list(_DEFAULT_WEIGHTS),
                shape=(3,),
            ),
            model_signature.ParamSpec(
                name="nested_list",
                dtype=model_signature.DataType.INT64,
                default_value=_DEFAULT_NESTED_LIST,
                shape=(2, 2),
            ),
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
            "int8_param": _OVERRIDE_INT8,
            "int16_param": _OVERRIDE_INT16,
            "int32_param": _OVERRIDE_INT32,
            "int64_param": _OVERRIDE_INT64,
            "uint8_param": _OVERRIDE_UINT8,
            "uint16_param": _OVERRIDE_UINT16,
            "uint32_param": _OVERRIDE_UINT32,
            "uint64_param": _OVERRIDE_UINT64,
            "float_param": _OVERRIDE_FLOAT,
            "double_param": _OVERRIDE_DOUBLE,
            "bool_param": _OVERRIDE_BOOL,
            "string_param": _OVERRIDE_STRING,
            "bytes_param": _OVERRIDE_BYTES,
            "timestamp_param": _OVERRIDE_TIMESTAMP,
            "weights_param": _OVERRIDE_WEIGHTS,
            "nested_list": _OVERRIDE_NESTED_LIST,
        }
        self._run_params_test(
            input_spec_params=override_params,
            model_call_params=override_params,
        )

    def test_mixed_params(self) -> None:
        """Test with some parameters using defaults and some overwritten."""
        # Override a subset of parameters across different types
        override_params = {
            "int8_param": _OVERRIDE_INT8,
            "uint16_param": _OVERRIDE_UINT16,
            "float_param": _OVERRIDE_FLOAT,
            "bool_param": _OVERRIDE_BOOL,
            "string_param": _OVERRIDE_STRING,
            "bytes_param": _OVERRIDE_BYTES,
            "weights_param": _OVERRIDE_WEIGHTS,
        }
        self._run_params_test(
            input_spec_params=override_params,
            model_call_params=override_params,  # Other params use defaults automatically
        )


_DEFAULT_RECORDS = [
    {"label": "a", "meta": {"score": 1.0, "tag": "x"}},
    {"label": "b", "meta": {"score": 2.0, "tag": "y"}},
]

_DEFAULT_CONFIG: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 50,
    "use_cache": True,
    "sampling": {"seed": 42, "strategy": "greedy"},
    "penalties": {"enabled": False, "frequency": {"weight": 0.5, "min_count": 1}},
    "records": _DEFAULT_RECORDS,
}

_FULL_CONFIG: dict[str, Any] = {
    "temperature": 0.5,
    "top_k": 100,
    "use_cache": False,
    "sampling": {"seed": 99, "strategy": "beam"},
    "penalties": {"enabled": True, "frequency": {"weight": 0.9, "min_count": 3}},
    "records": [
        {"label": "c", "meta": {"score": 9.0, "tag": "z"}},
        {"label": "d", "meta": {"score": 8.0, "tag": "w"}},
    ],
}

_PARTIAL_CONFIG: dict[str, Any] = {
    "temperature": 3.0,
    "sampling": {"strategy": "beam"},
    "penalties": {"frequency": {"weight": 0.1}},
    "records": [
        {"label": "c", "meta": {"score": 5.0, "tag": "z"}},
        {"label": "d", "meta": {"score": 6.0, "tag": "w"}},
    ],
}

_PARTIAL_MERGED_CONFIG: dict[str, Any] = {
    "temperature": 3.0,
    "top_k": 50,
    "use_cache": True,
    "sampling": {"seed": 42, "strategy": "beam"},
    "penalties": {"enabled": False, "frequency": {"weight": 0.1, "min_count": 1}},
    "records": [
        {"label": "c", "meta": {"score": 5.0, "tag": "z"}},
        {"label": "d", "meta": {"score": 6.0, "tag": "w"}},
    ],
}


class DemoModelWithDictParams(custom_model.CustomModel):
    """Custom model with a dict-typed parameter (ParamGroupSpec)."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input: pd.DataFrame,
        *,
        multiplier: float = 1.0,
        config: dict = _DEFAULT_CONFIG,  # noqa: B006
    ) -> pd.DataFrame:
        n = len(input)
        cfg = config or {}
        sampling = cfg.get("sampling", {}) or {}
        penalties = cfg.get("penalties", {}) or {}
        frequency = penalties.get("frequency", {}) or {}
        records = cfg.get("records", _DEFAULT_RECORDS)
        record_labels = ",".join(r.get("label", "") for r in records)
        record_scores = ",".join(str(r.get("meta", {}).get("score", 0.0)) for r in records)
        return pd.DataFrame(
            {
                "output": input["C1"] * multiplier,
                "received_temperature": [cfg.get("temperature", 1.0)] * n,
                "received_top_k": [cfg.get("top_k", 50)] * n,
                "received_use_cache": [cfg.get("use_cache", True)] * n,
                "received_seed": [sampling.get("seed", 42)] * n,
                "received_strategy": [sampling.get("strategy", "greedy")] * n,
                "received_penalties_enabled": [penalties.get("enabled", False)] * n,
                "received_freq_weight": [frequency.get("weight", 0.5)] * n,
                "received_freq_min_count": [frequency.get("min_count", 1)] * n,
                "received_record_labels": [record_labels] * n,
                "received_record_scores": [record_scores] * n,
            }
        )


class TestDictParamsBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Test batch inference with ParamGroupSpec (dict-typed parameters)."""

    def _get_param_specs(self) -> list[model_signature.BaseParamSpec]:
        return [
            model_signature.ParamSpec(name="multiplier", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            model_signature.ParamGroupSpec(
                name="config",
                specs=[
                    model_signature.ParamSpec(
                        name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=1.0
                    ),
                    model_signature.ParamSpec(name="top_k", dtype=model_signature.DataType.INT64, default_value=50),
                    model_signature.ParamSpec(
                        name="use_cache", dtype=model_signature.DataType.BOOL, default_value=True
                    ),
                    model_signature.ParamGroupSpec(
                        name="sampling",
                        specs=[
                            model_signature.ParamSpec(
                                name="seed", dtype=model_signature.DataType.INT64, default_value=42
                            ),
                            model_signature.ParamSpec(
                                name="strategy", dtype=model_signature.DataType.STRING, default_value="greedy"
                            ),
                        ],
                    ),
                    model_signature.ParamGroupSpec(
                        name="penalties",
                        specs=[
                            model_signature.ParamSpec(
                                name="enabled", dtype=model_signature.DataType.BOOL, default_value=False
                            ),
                            model_signature.ParamGroupSpec(
                                name="frequency",
                                specs=[
                                    model_signature.ParamSpec(
                                        name="weight", dtype=model_signature.DataType.DOUBLE, default_value=0.5
                                    ),
                                    model_signature.ParamSpec(
                                        name="min_count", dtype=model_signature.DataType.INT64, default_value=1
                                    ),
                                ],
                            ),
                        ],
                    ),
                    model_signature.ParamGroupSpec(
                        name="records",
                        specs=[
                            model_signature.ParamSpec(
                                name="label", dtype=model_signature.DataType.STRING, default_value="a"
                            ),
                            model_signature.ParamGroupSpec(
                                name="meta",
                                specs=[
                                    model_signature.ParamSpec(
                                        name="score", dtype=model_signature.DataType.DOUBLE, default_value=1.0
                                    ),
                                    model_signature.ParamSpec(
                                        name="tag", dtype=model_signature.DataType.STRING, default_value="x"
                                    ),
                                ],
                            ),
                        ],
                        shape=(2,),
                        default_value=[
                            {"label": "a", "meta": {"score": 1.0, "tag": "x"}},
                            {"label": "b", "meta": {"score": 2.0, "tag": "y"}},
                        ],
                    ),
                ],
            ),
        ]

    def _run_dict_params_test(
        self, input_spec_params: Optional[dict[str, Any]], model_call_params: dict[str, Any]
    ) -> None:
        model = DemoModelWithDictParams(custom_model.ModelContext())
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

    def test_dict_params_defaults(self) -> None:
        """Test with all dict param defaults (no override)."""
        self._run_dict_params_test(
            input_spec_params=None,
            model_call_params={},
        )

    def test_dict_params_full_override(self) -> None:
        """Test with fully overridden dict param."""
        self._run_dict_params_test(
            input_spec_params={"config": _FULL_CONFIG},
            model_call_params={"config": _FULL_CONFIG},
        )

    def test_dict_params_partial_override_deep_merge(self) -> None:
        """Test partial dict override triggers deep merge (unspecified keys keep defaults)."""
        self._run_dict_params_test(
            input_spec_params={"config": _PARTIAL_CONFIG},
            model_call_params={"config": _PARTIAL_MERGED_CONFIG},
        )

    def test_dict_params_with_scalar_override(self) -> None:
        """Test overriding both scalar and dict params together."""
        self._run_dict_params_test(
            input_spec_params={"multiplier": 2.0, "config": _FULL_CONFIG},
            model_call_params={"multiplier": 2.0, "config": _FULL_CONFIG},
        )


if __name__ == "__main__":
    absltest.main()
