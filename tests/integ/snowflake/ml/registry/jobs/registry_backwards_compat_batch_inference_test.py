"""Backwards compatibility batch inference tests for older snowflake-ml-python versions.

Scenario: a customer on an older snowflake-ml-python version logs a model. The model
artifact pins that older library version, but the batch inference images (Ray worker,
orchestrator) are always the latest release. These tests verify that the current batch
images correctly serve models logged with older client versions.

Tested versions:
    - snowflake-ml-python 1.18.0 (older release: broader backwards compat coverage)
    - snowflake-ml-python 1.25.0 (pre-ParamSpec: no runtime parameters)
    - snowflake-ml-python 1.26.0 (first version to introduce ParamSpec)

Tested scenarios:
    - 1.18.0: basic inference with no params (broad backwards compat)
    - 1.25.0: basic inference with no params (ParamSpec boundary)
    - 1.26.0: inference with default params, overridden params, and mixed params

See model_container_services_deployment/CONTEXT/backwards_compatibility.md
for the full backwards compatibility policy.
"""

import logging
import unittest
from typing import Any, Callable

import pandas as pd
from absl.testing import absltest
from packaging import version

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

logger = logging.getLogger(__name__)

_VERSION_COL = "snowml_version"


# ===========================================================================
# Models
# ===========================================================================


class SimpleModel(custom_model.CustomModel):
    """Minimal custom model with no runtime parameters.

    Used to test backwards compatibility with pre-ParamSpec versions (e.g. 1.25.0).
    Reports the runtime snowflake-ml-python version so tests can verify
    the model is actually running against the expected pinned version.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        from snowflake.ml import version as snowml_version_module

        return pd.DataFrame(
            {
                "input_value": input_df["value"].tolist(),
                "doubled": (input_df["value"] * 2).tolist(),
                _VERSION_COL: [snowml_version_module.VERSION] * len(input_df),
            }
        )


class ModelWithScalarParams(custom_model.CustomModel):
    """Custom model covering scalar ParamSpec types available in 1.26.0.

    Reports the runtime snowflake-ml-python version so tests can verify
    the model is actually running against the expected pinned version.
    """

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        temperature: float = 1.0,
        max_tokens: int = 100,
        label: str = "default",
        verbose: bool = False,
    ) -> pd.DataFrame:
        from snowflake.ml import version as snowml_version_module

        n = len(input_df)
        return pd.DataFrame(
            {
                "input_value": input_df["value"].tolist(),
                "received_temperature": [temperature] * n,
                "received_max_tokens": [max_tokens] * n,
                "received_label": [label] * n,
                "received_verbose": [verbose] * n,
                _VERSION_COL: [snowml_version_module.VERSION] * n,
            }
        )


# ===========================================================================
# Helpers
# ===========================================================================


def _make_batch_validator(
    test_case: absltest.TestCase,
    *,
    expected_version: str,
    expected_predictions: pd.DataFrame,
    index_col: str,
) -> Callable[[pd.DataFrame], None]:
    """Build a prediction_assert_fn that validates both version and output values.

    The base class's expected_predictions comparison requires identical column sets,
    but snowml_version will differ (local HEAD vs pinned remote). This validator:
    1. Checks snowml_version matches the pinned version
    2. Drops snowml_version from both sides
    3. Compares remaining columns via assert_frame_equal
    """

    def validator(actual_output: pd.DataFrame) -> None:
        # Parquet preserves original case while Snowflake SQL uppercases — match case-insensitively.
        version_col_matches = [c for c in actual_output.columns if c.lower() == _VERSION_COL.lower()]
        test_case.assertTrue(
            len(version_col_matches) == 1,
            f"Expected exactly one {_VERSION_COL} column, found {version_col_matches} in {list(actual_output.columns)}",
        )
        version_col = version_col_matches[0]

        for actual in actual_output[version_col]:
            test_case.assertTrue(
                str(actual).startswith(expected_version),
                f"Expected snowml version starting with {expected_version}, got {actual}",
            )

        actual_compare = actual_output.drop(columns=[version_col])
        expected_compare = expected_predictions.copy()
        cols_to_drop = [c for c in expected_compare.columns if c.lower() == _VERSION_COL.lower()]
        expected_compare = expected_compare.drop(columns=cols_to_drop)

        actual_compare = actual_compare.sort_values(index_col).reset_index(drop=True)
        expected_compare = expected_compare.sort_values(index_col).reset_index(drop=True)

        expected_columns = sorted(expected_compare.columns)
        actual_compare = actual_compare[expected_columns]

        pd.testing.assert_frame_equal(
            expected_compare,
            actual_compare,
            check_dtype=False,
            check_exact=False,
            rtol=1e-3,
            atol=1e-6,
        )

    return validator


# ===========================================================================
# Test: pre-ParamSpec (1.25.0) — basic batch inference, no params
# ===========================================================================


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_version() >= version.parse("10.0.0"),
    "Model method signature parameters only available when Snowflake Version >= 10.0.0",
)
class TestBatchBackwardsCompatNoParams(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Backwards compatibility batch inference test for models logged before ParamSpec existed.

    Tests both a significantly older version (1.18.0) for broad backwards compat
    coverage and the immediate pre-ParamSpec version (1.25.0) to ensure the param
    infrastructure doesn't break non-param models in the batch path.
    """

    def _log_and_run_batch(self, snowml_version: str) -> None:
        model = SimpleModel(custom_model.ModelContext())

        input_pandas_df = pd.DataFrame({"value": [10.0, 20.0]})
        model_output = model.predict(input_pandas_df)
        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)

        sig = model_signature.infer_signature(input_data=input_pandas_df, output_data=model_output)
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            signatures={"predict": sig},
            pip_requirements=[f"snowflake-ml-python=={snowml_version}"],
            options={"embed_local_ml_library": False},
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, num_workers=1, replicas=1, function_name="predict"),
            prediction_assert_fn=_make_batch_validator(
                self,
                expected_version=snowml_version,
                expected_predictions=expected_predictions,
                index_col=self._INDEX_COL,
            ),
        )

    def test_no_params_v1_18_0(self) -> None:
        """Deploy with snowflake-ml-python 1.18.0, verify batch inference.

        Tests that the current batch images correctly serve a model from a
        significantly older SDK version, covering model artifact format changes
        and env handling changes beyond just the ParamSpec boundary.
        """
        self._log_and_run_batch("1.18.0")

    def test_no_params_v1_25_0(self) -> None:
        """Deploy with snowflake-ml-python 1.25.0 (pre-ParamSpec), verify batch inference.

        Tests that the current batch inference images correctly handle models
        logged immediately before ParamSpec was introduced.
        """
        self._log_and_run_batch("1.25.0")


# ===========================================================================
# Test: ParamSpec (1.26.0) — batch inference with params
# ===========================================================================


_PARAM_SPECS = [
    model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
    model_signature.ParamSpec(name="max_tokens", dtype=model_signature.DataType.INT64, default_value=100),
    model_signature.ParamSpec(name="label", dtype=model_signature.DataType.STRING, default_value="default"),
    model_signature.ParamSpec(name="verbose", dtype=model_signature.DataType.BOOL, default_value=False),
]

_OVERRIDE_PARAMS: dict[str, Any] = {
    "temperature": 2.5,
    "max_tokens": 200,
    "label": "custom",
    "verbose": True,
}

_PARTIAL_OVERRIDE_PARAMS: dict[str, Any] = {
    "temperature": 0.5,
    "max_tokens": 50,
}


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_version() >= version.parse("10.0.0"),
    "Model method signature parameters only available when Snowflake Version >= 10.0.0",
)
class TestBatchBackwardsCompatWithParams(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Backwards compatibility batch inference test for models logged with ParamSpec.

    Logs a model with snowflake-ml-python 1.26.0, runs batch inference with the
    current images, and verifies param forwarding works correctly.
    """

    _SNOWML_VERSION = "1.26.0"

    def _run_batch_with_params(
        self,
        snowml_version: str,
        *,
        input_spec_params: dict[str, Any],
        model_call_params: dict[str, Any],
    ) -> None:
        model = ModelWithScalarParams(custom_model.ModelContext())

        input_pandas_df = pd.DataFrame({"value": [10.0, 20.0]})
        model_output = model.predict(input_pandas_df, **model_call_params)
        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)

        sig = model_signature.infer_signature(
            input_data=input_pandas_df,
            output_data=model_output,
            params=_PARAM_SPECS,
        )
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        input_spec = InputSpec(params=input_spec_params) if input_spec_params else InputSpec()

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            signatures={"predict": sig},
            pip_requirements=[f"snowflake-ml-python=={snowml_version}"],
            options={"embed_local_ml_library": False},
            input_spec=input_spec,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, num_workers=1, replicas=1, function_name="predict"),
            prediction_assert_fn=_make_batch_validator(
                self,
                expected_version=snowml_version,
                expected_predictions=expected_predictions,
                index_col=self._INDEX_COL,
            ),
        )

    def test_default_params_v1_26_0(self) -> None:
        """Deploy with snowflake-ml-python 1.26.0, verify batch inference with default params."""
        self._run_batch_with_params(
            self._SNOWML_VERSION,
            input_spec_params={},
            model_call_params={},
        )

    def test_override_params_v1_26_0(self) -> None:
        """Deploy with snowflake-ml-python 1.26.0, verify batch inference with all params overridden."""
        self._run_batch_with_params(
            self._SNOWML_VERSION,
            input_spec_params=_OVERRIDE_PARAMS,
            model_call_params=_OVERRIDE_PARAMS,
        )

    def test_partial_params_v1_26_0(self) -> None:
        """Deploy with snowflake-ml-python 1.26.0, verify batch inference with partial param overrides."""
        self._run_batch_with_params(
            self._SNOWML_VERSION,
            input_spec_params=_PARTIAL_OVERRIDE_PARAMS,
            model_call_params=_PARTIAL_OVERRIDE_PARAMS,
        )


if __name__ == "__main__":
    absltest.main()
