"""Integration tests for pip-only model packaging with batch inference."""

import sys
from typing import Any, Callable

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

# Python versions supported by the base image's cached standalone tarballs (dockerfile_template_pip path).
PIP_ONLY_PYTHON_VERSIONS = ("3.10", "3.11", "3.12")


class PipOnlyModel(custom_model.CustomModel):
    """A simple custom model with only pip dependencies."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        import sys

        import requests  # noqa: F401 - imported to verify pip package is available

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        return pd.DataFrame(
            {
                "output": input["value"] * 2,
                "python_version": [py_ver] * len(input),
            }
        )

    @custom_model.inference_api
    def check_env(self, input: pd.DataFrame) -> pd.DataFrame:
        """Check if the environment is using pip-only path (uv/venv) instead of conda.

        Returns environment info to verify:
        - uses_venv: True if running from /opt/venv
        - uses_standalone_python: True if base Python is from /opt/python
        - uses_conda: True if conda is active (should be False for pip-only)
        - python_executable: Full path to Python executable
        - virtual_env: Value of VIRTUAL_ENV env var
        """
        import os
        import sys

        python_exec = sys.executable
        virtual_env = os.environ.get("VIRTUAL_ENV", "")

        uses_venv = "/opt/venv" in python_exec or virtual_env == "/opt/venv"
        uses_standalone_python = "/opt/python" in getattr(sys, "base_prefix", "")
        uses_conda = "/opt/conda" in python_exec

        return pd.DataFrame(
            {
                "uses_venv": [uses_venv],
                "uses_standalone_python": [uses_standalone_python],
                "uses_conda": [uses_conda],
                "python_executable": [python_exec],
                "virtual_env": [virtual_env],
            }
        )


class TestRegistryPipOnlyBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Integration tests for pip-only model batch inference using uv-based image builder.

    Overrides the builder image to use INFERENCE_IMAGE_BUILDER_PATH (sf-inference-image-builder-amd64)
    instead of BUILDER_IMAGE_PATH (sf-kaniko-amd64, which is deprecated and does not support pip-only
    models because its build.py does not handle a missing conda.yml).
    """

    def _has_image_override(self) -> bool:
        image_paths = [
            self.INFERENCE_IMAGE_BUILDER_PATH,
            self.BASE_BATCH_CPU_IMAGE_PATH,
            self.BASE_BATCH_GPU_IMAGE_PATH,
            self.MODEL_LOGGER_PATH,
        ]
        if all(image_paths):
            return True
        elif not any(image_paths):
            return False
        else:
            raise ValueError(
                "Please set or unset INFERENCE_IMAGE_BUILDER_PATH, BASE_BATCH_CPU_IMAGE_PATH, "
                "BASE_BATCH_GPU_IMAGE_PATH, and MODEL_LOGGER_PATH at the same time."
            )

    def _get_batch_image_override_session_params(self) -> dict[str, str]:
        params = super()._get_batch_image_override_session_params()
        # Pip-only models require the BuildKit inference image builder.
        if self.INFERENCE_IMAGE_BUILDER_PATH is not None:
            params["SPCS_MODEL_BUILD_CONTAINER_URL"] = self.INFERENCE_IMAGE_BUILDER_PATH
        params.pop("SPCS_MODEL_INFERENCE_ENGINE_CONTAINER_URLS", None)
        return params

    def _prepare_pip_only_test(
        self,
        model: custom_model.CustomModel,
        input_pandas_df: pd.DataFrame,
    ) -> tuple[Any, pd.DataFrame, pd.DataFrame, str, str]:
        """Prepare test data for pip-only batch inference tests.

        Args:
            model: The custom model to test.
            input_pandas_df: Input data as pandas DataFrame.

        Returns:
            Tuple of (sample_input_df, input_df, expected_predictions, job_name, output_stage_location).
        """
        # Generate expected predictions using the model
        model_output = model.predict(input_pandas_df)

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)

        # Create sample input for model signature
        sp_df = self.session.create_dataframe(input_pandas_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        return sp_df, input_df, expected_predictions, job_name, output_stage_location

    def _assert_pip_only_env(self, res: pd.DataFrame) -> None:
        """Assert that the batch job is running in the pip-only virtual environment, not conda."""
        self.assertTrue(
            res["uses_venv"].iloc[0],
            f"Expected to use /opt/venv but got python_executable={res['python_executable'].iloc[0]}",
        )
        self.assertTrue(
            res["uses_standalone_python"].iloc[0],
            "Expected to use standalone Python from /opt/python",
        )
        self.assertFalse(
            res["uses_conda"].iloc[0],
            f"python_executable={res['python_executable'].iloc[0]}",
        )

    def _assert_pip_only_predict_result(self, py_ver: str, expected: pd.DataFrame) -> Callable[[pd.DataFrame], None]:
        """Assert predict result matches expected and runtime python_version matches requested py_ver."""

        def fn(res: pd.DataFrame) -> None:
            self.assertEqual(res["python_version"].iloc[0], py_ver)
            pd.testing.assert_series_equal(
                res["output"].sort_values().reset_index(drop=True),
                expected["output"].sort_values().reset_index(drop=True),
                check_dtype=False,
            )

        return fn

    def _run_pip_only_env_check_batch_inference(
        self,
        model: custom_model.CustomModel,
        pip_requirements: list[str],
        options: dict,
    ) -> None:
        """Run a batch inference job to verify the pip-only environment (check_env function).

        Uses a separate model name (via stack inspection) so it can be called in the same test
        as a predict job without conflicting model version names.
        """
        env_input = pd.DataFrame({"value": [1.0]})
        sp_df = self.session.create_dataframe(env_input)
        input_df, _ = self._prepare_batch_inference_data(env_input, pd.DataFrame({"uses_venv": [True]}))
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                function_name="check_env",
            ),
            pip_requirements=pip_requirements,
            options=options,
            prediction_assert_fn=self._assert_pip_only_env,
            conda_dependencies=[],
        )

    def test_pip_only_batch_inference(self) -> None:
        """Test end-to-end batch inference of a pip-only model using uv-based image builder.

        Verifies:
        1. Model prediction works correctly
        2. Environment uses pip-only path (uv/venv) instead of conda
        """
        if not self._has_image_override():
            self.skipTest("Skipping pip-only batch inference test: image override not enabled.")

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        model = PipOnlyModel(custom_model.ModelContext())
        input_pandas_df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        sp_df, input_df, expected_predictions, job_name, output_stage_location = self._prepare_pip_only_test(
            model, input_pandas_df
        )

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                function_name="predict",
            ),
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
            expected_predictions=expected_predictions,
            conda_dependencies=[],
        )
        self._run_pip_only_env_check_batch_inference(
            model=model,
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
        )

    @parameterized.parameters(*PIP_ONLY_PYTHON_VERSIONS)  # type: ignore[misc]
    def test_pip_only_batch_inference_python_versions(self, py_ver: str) -> None:
        """E2E test: batch inference with pip-only model for each supported Python version (3.10, 3.11, 3.12).

        Verifies:
        1. Model runs with the correct Python version
        2. Environment uses pip-only path (uv/venv)
        """
        current_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if py_ver != current_ver:
            self.skipTest(
                f"Skipping Python {py_ver} test: model is pickled with {current_ver} and "
                f"cloudpickle cannot deserialize across Python versions."
            )
        if not self._has_image_override():
            self.skipTest("Skipping pip-only batch inference test: image override not enabled.")

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        model = PipOnlyModel(custom_model.ModelContext())
        input_pandas_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        # Generate expected predictions for assertion
        model_output = model.predict(input_pandas_df)

        sp_df, input_df, _, job_name, output_stage_location = self._prepare_pip_only_test(model, input_pandas_df)

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sp_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                function_name="predict",
            ),
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
            python_version=py_ver,
            prediction_assert_fn=self._assert_pip_only_predict_result(py_ver, model_output),
            conda_dependencies=[],
        )
        self._run_pip_only_env_check_batch_inference(
            model=model,
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
        )


if __name__ == "__main__":
    absltest.main()
