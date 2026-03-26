"""Integration tests for pip-only model packaging and deployment."""

from typing import Callable

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

# Python versions supported by the base image's cached standalone tarballs (dockerfile_template_pip path).
PIP_ONLY_PYTHON_VERSIONS = ("3.10", "3.11", "3.12")


class PipOnlyModel(custom_model.CustomModel):
    """A simple custom model with only pip dependencies."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        # Use a pip-only package to verify it's installed correctly
        import sys

        import requests  # noqa: F401 - imported to verify pip package is available

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        return pd.DataFrame(
            {
                "output": input["value"] * 2,
                "python_version": [py_ver] * len(input),
            }
        )


class TestRegistryPipOnlyModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for pip-only model deployment using dockerfile_template_pip."""

    def _assert_pip_only_predict_result(self, py_ver: str, expected: pd.DataFrame) -> Callable[[pd.DataFrame], None]:
        """Assert predict result matches expected and runtime python_version matches requested py_ver."""

        def fn(res: pd.DataFrame) -> None:
            self.assertEqual(res["python_version"].iloc[0], py_ver)
            pd.testing.assert_frame_equal(res, expected, check_dtype=False)

        return fn

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
        )

    def test_pip_only_gpu_model(self) -> None:
        """Test pip-only GPU deployment: verifies cuda_version flows through to image builder
        and that a GPU model runs inference correctly on a GPU compute pool.
        """
        if not self._has_image_override():
            self.skipTest("Skipping inference_image_builder test: image override not enabled.")

        test_input = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        pip_only_model = PipOnlyModel(custom_model.ModelContext())

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

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
            options={
                "cuda_version": model_env.DEFAULT_CUDA_VERSION,
                "enable_explainability": False,
            },
            gpu_requests="1",
        )

    @parameterized.parameters(*PIP_ONLY_PYTHON_VERSIONS)  # type: ignore[misc]
    def test_pip_only_model_python_versions(self, py_ver: str) -> None:
        """E2E test: deploy a pip-only model with each supported Python version (3.10, 3.11, 3.12)."""
        if not self._has_image_override():
            self.skipTest("Skipping inference_image_builder test: image override not enabled.")

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        test_input = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        model = PipOnlyModel(custom_model.ModelContext())
        self._test_registry_model_deployment(
            model=model,
            sample_input_data=test_input,
            prediction_assert_fns={
                "predict": (
                    test_input,
                    self._assert_pip_only_predict_result(py_ver, model.predict(test_input)),
                ),
            },
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
            python_version=py_ver,
            service_name=f"service_pip_only_python_versions_{self._run_id}_{py_ver.replace('.', '')}",
        )


if __name__ == "__main__":
    absltest.main()
