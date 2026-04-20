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

    @custom_model.inference_api
    def check_env(self, input: pd.DataFrame) -> pd.DataFrame:
        """Check if the environment is using a pip-only path (venv) instead of conda.

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
        conda_prefix = os.environ.get("CONDA_PREFIX", "")

        # Check if using venv (pip-only path uses /opt/venv)
        uses_venv = "/opt/venv" in python_exec or virtual_env == "/opt/venv"

        # Check if base Python is from standalone tarball (/opt/python)
        # The venv's base_prefix points to the standalone Python installation
        uses_standalone_python = "/opt/python" in getattr(sys, "base_prefix", "")

        # Check if conda is active (should NOT be for pip-only)
        uses_conda = bool(conda_prefix) or "/opt/conda" in python_exec

        return pd.DataFrame(
            {
                "uses_venv": [uses_venv],
                "uses_standalone_python": [uses_standalone_python],
                "uses_conda": [uses_conda],
                "python_executable": [python_exec],
                "virtual_env": [virtual_env],
            }
        )


class PipOnlyPyTorchModel(custom_model.CustomModel):
    """A custom model with PyTorch dependency for GPU testing."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        import torch

        x = torch.tensor(input["value"].values, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        output = (x * 2).cpu().numpy()
        return pd.DataFrame({"output": output})

    @custom_model.inference_api
    def check_cuda(self, input: pd.DataFrame) -> pd.DataFrame:
        """Check if CUDA is available - used to verify GPU setup."""
        import torch

        return pd.DataFrame(
            {
                "cuda_available": [torch.cuda.is_available()],
                "device_count": [torch.cuda.device_count() if torch.cuda.is_available() else 0],
            }
        )


class TestRegistryPipOnlyModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for pip-only model deployment.

    Uses the same Kaniko builder override as ``RegistryModelDeploymentTestBase`` (``BUILDER_IMAGE_PATH``).
    """

    def _assert_pip_only_predict_result(self, py_ver: str, expected: pd.DataFrame) -> Callable[[pd.DataFrame], None]:
        """Assert predict result matches expected and runtime python_version matches requested py_ver."""

        def fn(res: pd.DataFrame) -> None:
            self.assertEqual(res["python_version"].iloc[0], py_ver)
            pd.testing.assert_frame_equal(res, expected, check_dtype=False)

        return fn

    def _assert_pip_only_env(self, res: pd.DataFrame) -> None:
        """Assert that the deployed service is using a pip-only path (venv)."""
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

    def test_pip_only_model(self) -> None:
        """Test end-to-end deployment of a pip-only model.

        Verifies:
        1. Model prediction works correctly
        2. Environment uses a pip-only path (venv) instead of conda
        """
        if not self._has_image_override():
            self.skipTest("Skipping pip-only model deployment test: image override not enabled.")

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
                "check_env": (
                    pd.DataFrame({"value": [1.0]}),
                    self._assert_pip_only_env,
                ),
            },
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
            conda_dependencies=[],
        )

    def test_pip_only_gpu_model(self) -> None:
        """Test pip-only GPU deployment: verifies cuda_version flows through to image builder
        and that a GPU model runs inference correctly on a GPU compute pool.
        """
        if not self._has_image_override():
            self.skipTest("Skipping pip-only model deployment test: image override not enabled.")

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
            conda_dependencies=[],
        )

    @parameterized.parameters(*PIP_ONLY_PYTHON_VERSIONS)  # type: ignore[misc]
    def test_pip_only_model_python_versions(self, py_ver: str) -> None:
        """E2E test: deploy a pip-only model with each supported Python version (3.10, 3.11, 3.12).

        Verifies:
        1. Model runs with the correct Python version
        2. Environment uses a pip-only path (venv)
        """
        if not self._has_image_override():
            self.skipTest("Skipping pip-only model deployment test: image override not enabled.")

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
                "check_env": (
                    pd.DataFrame({"value": [1.0]}),
                    self._assert_pip_only_env,
                ),
            },
            pip_requirements=["requests>=2.28.0"],
            options={"enable_explainability": False},
            python_version=py_ver,
            service_name=f"service_pip_only_python_versions_{self._run_id}_{py_ver.replace('.', '')}",
            conda_dependencies=[],
        )

    def test_pip_only_pytorch_gpu_model(self) -> None:
        """E2E test: deploy a pip-only PyTorch model on GPU and verify CUDA is available.

        This test verifies that:
        1. PyTorch CUDA index URL is added to requirements.txt (via _get_pytorch_cuda_index_url)
        2. The GPU image builder downloads the correct CUDA-enabled PyTorch wheel
        3. torch.cuda.is_available() returns True on the deployed service
        """
        if not self._has_image_override():
            self.skipTest("Skipping pip-only model deployment test: image override not enabled.")

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        test_input = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        model = PipOnlyPyTorchModel(custom_model.ModelContext())

        def assert_cuda_available(res: pd.DataFrame) -> None:
            """Assert that CUDA is available on the deployed service."""
            self.assertTrue(res["cuda_available"].iloc[0], "CUDA should be available on GPU service")
            self.assertGreater(res["device_count"].iloc[0], 0, "Should have at least 1 GPU device")

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=test_input,
            prediction_assert_fns={
                "check_cuda": (
                    pd.DataFrame({"value": [1.0]}),
                    assert_cuda_available,
                ),
            },
            pip_requirements=["torch>=2.0"],
            options={
                "cuda_version": model_env.DEFAULT_CUDA_VERSION,
                "enable_explainability": False,
            },
            gpu_requests="1",
            service_name=f"service_pip_only_pytorch_gpu_{self._run_id}",
            conda_dependencies=[],
        )


if __name__ == "__main__":
    absltest.main()
