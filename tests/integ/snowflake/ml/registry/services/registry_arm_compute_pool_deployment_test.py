import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from snowflake.ml.model._client.model.model_version_impl import ModelVersion
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import test_env_utils

_ARM_SUFFIX = "-cpu-arm64"
_X86_SUFFIX = "-cpu"


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class TestRegistryArmComputePoolDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Model deployment on ARM (aarch64) compute pools.

    Requires the ARM compute pool and ENABLE_MODEL_SERVING_ON_ARM_COMPUTE_POOL on the account; fails otherwise.
    """

    def setUp(self) -> None:
        super().setUp()
        self.assert_compute_pool_exists(self._TEST_ARM_COMPUTE_POOL)

    def _log_demo_model(self, *, cuda_version: str | None = None) -> tuple[ModelVersion, pd.DataFrame]:
        sp_df = self.session.create_dataframe([[0, 0], [1, 1]], schema=['"c1"', '"c2"'])
        options = {"embed_local_ml_library": True}
        if cuda_version is not None:
            # GPU services require a model logged with GPU runtime support.
            options["cuda_version"] = cuda_version
        mv = self.registry.log_model(
            model=DemoModel(custom_model.ModelContext()),
            model_name="arm_demo_model",
            version_name=f"ver_{self._run_id}",
            sample_input_data=sp_df,
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
            ],
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options=options,
        )
        return mv, sp_df.to_pandas()

    def _deploy_and_infer(
        self,
        mv: ModelVersion,
        test_input: pd.DataFrame,
        *,
        service_name: str,
        pool: str,
        force_rebuild: bool,
    ) -> tuple[str, set[str]]:
        """Deploy on ``pool`` (also used to build the image), check predictions, and drop the service.

        Returns:
            The model inference image name of the service and the MODEL_BUILD jobs seen during the deployment.
        """
        expected = pd.DataFrame({"output": test_input["c1"]})
        with self.watch_model_build_jobs() as build_jobs:
            self._deploy_model_service(
                mv=mv,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: pd.testing.assert_frame_equal(res, expected, check_dtype=False),
                    ),
                },
                service_name=service_name,
                service_compute_pool=pool,
                image_build_compute_pool=pool,
                force_rebuild=force_rebuild,
                skip_rest_api_test=True,
            )
        image_name = self.get_model_inference_image_name(service_name)
        mv.delete_service(service_name)
        return image_name, build_jobs

    def test_arm_pool_deployment(self) -> None:
        """A model deployed on an ARM pool runs and uses the arm64 image."""
        mv, test_input = self._log_demo_model()
        image_name, _ = self._deploy_and_infer(
            mv,
            test_input,
            service_name=f"service_arm_{self._run_id}",
            pool=self._TEST_ARM_COMPUTE_POOL,
            force_rebuild=False,
        )
        self.assertTrue(image_name.endswith(_ARM_SUFFIX), image_name)

    def test_arch_cache_isolation(self) -> None:
        """x86 and ARM deployments of the same model keep separate image cache entries.

        The first x86 and ARM deployments force a rebuild so that a build is guaranteed regardless of what
        earlier runs cached. The second ones must reuse their own arch's image without triggering a build.
        If the two arches shared a cache entry, the ARM rebuild would overwrite the x86 entry and the last x86
        deployment would come up with the arm64 image name.
        """
        mv, test_input = self._log_demo_model()
        x86_pool = self._TEST_CPU_COMPUTE_POOL
        arm_pool = self._TEST_ARM_COMPUTE_POOL

        x86_image, builds = self._deploy_and_infer(
            mv, test_input, service_name=f"service_x86_1_{self._run_id}", pool=x86_pool, force_rebuild=True
        )
        self.assertTrue(x86_image.endswith(_X86_SUFFIX), x86_image)
        self.assertTrue(builds, "Expected an image build for the first x86 deployment.")

        arm_image, builds = self._deploy_and_infer(
            mv, test_input, service_name=f"service_arm_1_{self._run_id}", pool=arm_pool, force_rebuild=True
        )
        self.assertTrue(arm_image.endswith(_ARM_SUFFIX), arm_image)
        self.assertNotEqual(arm_image, x86_image)
        self.assertTrue(builds, "Expected an image build for the first ARM deployment.")

        image, builds = self._deploy_and_infer(
            mv, test_input, service_name=f"service_x86_2_{self._run_id}", pool=x86_pool, force_rebuild=False
        )
        self.assertEqual(image, x86_image)
        self.assertFalse(builds, f"Unexpected image build for the second x86 deployment: {builds}")

        image, builds = self._deploy_and_infer(
            mv, test_input, service_name=f"service_arm_2_{self._run_id}", pool=arm_pool, force_rebuild=False
        )
        self.assertEqual(image, arm_image)
        self.assertFalse(builds, f"Unexpected image build for the second ARM deployment: {builds}")

    def _assert_build_pool_rejected(
        self,
        mv: ModelVersion,
        *,
        service_compute_pool: str,
        image_build_compute_pool: str,
        expected_message: str,
        gpu_requests: str | None = None,
    ) -> None:
        """Assert that create_service fails with 402153 before any image build job starts."""
        with self.watch_model_build_jobs() as build_jobs:
            with self.assertRaisesRegex(Exception, rf"(?s)402153.*{expected_message}"):
                mv.create_service(
                    service_name=f"service_rejected_{self._run_id}",
                    service_compute_pool=service_compute_pool,
                    image_build_compute_pool=image_build_compute_pool,
                    gpu_requests=gpu_requests,
                    force_rebuild=True,
                )
        self.assertFalse(build_jobs, f"Unexpected image build for a rejected deployment: {build_jobs}")

    def test_arm_service_rejects_x86_build_pool(self) -> None:
        """An ARM service cannot use an x86 build pool, because the image would be built for x86."""
        mv, _ = self._log_demo_model()
        self._assert_build_pool_rejected(
            mv,
            service_compute_pool=self._TEST_ARM_COMPUTE_POOL,
            image_build_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            expected_message="has CPU architecture x86_64, but the inference compute pool requires aarch64",
        )

    def test_x86_service_rejects_arm_build_pool(self) -> None:
        """An x86 service cannot use an ARM build pool, because the image would be built for ARM."""
        mv, _ = self._log_demo_model()
        self._assert_build_pool_rejected(
            mv,
            service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            image_build_compute_pool=self._TEST_ARM_COMPUTE_POOL,
            expected_message="has CPU architecture aarch64, but the inference compute pool requires x86_64",
        )

    def test_gpu_service_rejects_arm_build_pool(self) -> None:
        """A GPU service cannot use an ARM build pool, because GPU images can only be built on x86."""
        mv, _ = self._log_demo_model(cuda_version=model_env.DEFAULT_CUDA_VERSION)
        self._assert_build_pool_rejected(
            mv,
            service_compute_pool=self._TEST_GPU_COMPUTE_POOL,
            image_build_compute_pool=self._TEST_ARM_COMPUTE_POOL,
            gpu_requests="1",
            expected_message=(
                "has CPU architecture aarch64, but the inference compute pool requires x86_64.*"
                "GPU inference images can only be built on x86_64 compute pools"
            ),
        )


if __name__ == "__main__":
    absltest.main()
