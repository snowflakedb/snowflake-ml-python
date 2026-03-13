import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class CudaDeviceCountModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        import torch

        device_count = torch.cuda.device_count()
        return pd.DataFrame({"device_count": [device_count] * len(input_df)})


class TestRegistryGpuRequestOptionalInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.session.sql("ALTER SESSION SET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL = TRUE").collect()

    def tearDown(self) -> None:
        self.session.sql("ALTER SESSION UNSET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL").collect()
        super().tearDown()

    @parameterized.named_parameters(
        ("with_gpu_requests", "1"),
        ("without_gpu_requests", None),
    )
    def test_gpu_request_optional(
        self,
        gpu_requests: str,
    ) -> None:
        import yaml

        model = CudaDeviceCountModel(custom_model.ModelContext())
        input_df = pd.DataFrame({"dummy": [1]})

        # We explicitly use a GPU compute pool to test that CUDA is available
        # even when gpu_requests is not provided.
        service_compute_pool = self._TEST_GPU_COMPUTE_POOL
        service_name = f"service_gpu_opt_{self._run_id}_{'with' if gpu_requests else 'without'}_gpu"

        def check_gpu_count(res: pd.DataFrame) -> None:
            if gpu_requests == "1":
                self.assertEqual(res["device_count"].iloc[0], 1)
            else:
                self.assertGreater(res["device_count"].iloc[0], 0)

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=input_df,
            prediction_assert_fns={
                "predict": (
                    input_df,
                    check_gpu_count,
                ),
            },
            service_name=service_name,
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests=gpu_requests,
            service_compute_pool=service_compute_pool,
            # Ensure torch is available in the environment
            pip_requirements=["torch"],
        )

        # Verify the service spec for GPU resources
        desc_res = self.session.sql(f"DESC SERVICE {service_name}").collect()
        self.assertGreater(len(desc_res), 0)
        spec_yaml = desc_res[0]["spec"]
        spec = yaml.safe_load(spec_yaml)

        # Find the model-inference container and verify GPU resources
        model_inference_container = next(
            (c for c in spec["spec"]["containers"] if c["name"] == "model-inference"), None
        )
        self.assertIsNotNone(model_inference_container, "model-inference container not found in spec")

        resources = model_inference_container.get("resources", {})
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})

        gpu_resource_key = "nvidia.com/gpu"
        self.assertIn(gpu_resource_key, limits, f"{gpu_resource_key} not found in limits")
        self.assertIn(gpu_resource_key, requests, f"{gpu_resource_key} not found in requests")

        self.assertIsNotNone(limits[gpu_resource_key], "GPU limit should be set")
        self.assertIsNotNone(requests[gpu_resource_key], "GPU request should be set")


if __name__ == "__main__":
    absltest.main()
