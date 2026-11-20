import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from snowflake.ml.model._client.model import batch_inference_job_specs
from snowflake.ml.model._client.model.model_version_impl import ModelVersion
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

_ARM_SUFFIX = "-cpu-arm64"
_X86_SUFFIX = "-cpu"


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["C1"]})


class TestBatchInferenceArmComputePoolInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Batch inference on ARM (aarch64) compute pools.

    Requires the ARM compute pool and ENABLE_MODEL_SERVING_ON_ARM_COMPUTE_POOL on the account; fails otherwise.
    """

    def setUp(self) -> None:
        super().setUp()
        self.assert_compute_pool_exists(self._TEST_ARM_COMPUTE_POOL)

    def _log_demo_model(self) -> tuple[ModelVersion, pd.DataFrame]:
        input_pandas_df = pd.DataFrame([[0, 0], [1, 1]], columns=["C1", "C2"])
        sample_input_data = self.session.create_dataframe(input_pandas_df)
        mv = self.registry.log_model(
            model=DemoModel(custom_model.ModelContext()),
            model_name="arm_demo_model",
            version_name=f"ver_{self._run_id}",
            sample_input_data=sample_input_data,
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
            ],
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )
        return mv, input_pandas_df

    def _run_and_check(
        self,
        mv: ModelVersion,
        input_pandas_df: pd.DataFrame,
        *,
        pool: str,
        force_rebuild: bool,
    ) -> tuple[str, set[str]]:
        """Run a batch job on ``pool`` and check its predictions.

        Returns:
            The model inference image name of the job and the MODEL_BUILD jobs seen during the run.
        """
        model_output = pd.DataFrame({"output": input_pandas_df["C1"]})
        input_df, expected_predictions = self._prepare_batch_inference_data(input_pandas_df, model_output)
        _, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()
        with self.watch_model_build_jobs() as build_jobs:
            batch_job = self._deploy_batch_inference(
                mv,
                X=input_df,
                output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
                compute_pool=pool,
                image_build_spec=batch_inference_job_specs.ImageBuildSpec(force_rebuild=force_rebuild),
                expected_predictions=expected_predictions,
            )
        return self.get_model_inference_image_name(batch_job.id), build_jobs

    def test_arm_pool_batch_inference(self) -> None:
        """A batch job on an ARM pool runs and uses the arm64 image."""
        mv, input_pandas_df = self._log_demo_model()
        image_name, _ = self._run_and_check(mv, input_pandas_df, pool=self._TEST_ARM_COMPUTE_POOL, force_rebuild=False)
        self.assertTrue(image_name.endswith(_ARM_SUFFIX), image_name)

    def test_arch_cache_isolation(self) -> None:
        """x86 and ARM batch jobs for the same model keep separate image cache entries.

        The first x86 and ARM jobs force a rebuild so that a build is guaranteed regardless of what earlier runs
        cached. The second ones must reuse their own arch's image without triggering a build. If the two arches
        shared a cache entry, the ARM rebuild would overwrite the x86 entry and the last x86 job would come up with
        the arm64 image name.
        """
        mv, input_pandas_df = self._log_demo_model()
        x86_pool = self._TEST_CPU_COMPUTE_POOL
        arm_pool = self._TEST_ARM_COMPUTE_POOL

        x86_image, builds = self._run_and_check(mv, input_pandas_df, pool=x86_pool, force_rebuild=True)
        self.assertTrue(x86_image.endswith(_X86_SUFFIX), x86_image)
        self.assertTrue(builds, "Expected an image build for the first x86 job.")

        arm_image, builds = self._run_and_check(mv, input_pandas_df, pool=arm_pool, force_rebuild=True)
        self.assertTrue(arm_image.endswith(_ARM_SUFFIX), arm_image)
        self.assertNotEqual(arm_image, x86_image)
        self.assertTrue(builds, "Expected an image build for the first ARM job.")

        image, builds = self._run_and_check(mv, input_pandas_df, pool=x86_pool, force_rebuild=False)
        self.assertEqual(image, x86_image)
        self.assertFalse(builds, f"Unexpected image build for the second x86 job: {builds}")

        image, builds = self._run_and_check(mv, input_pandas_df, pool=arm_pool, force_rebuild=False)
        self.assertEqual(image, arm_image)
        self.assertFalse(builds, f"Unexpected image build for the second ARM job: {builds}")


if __name__ == "__main__":
    absltest.main()
