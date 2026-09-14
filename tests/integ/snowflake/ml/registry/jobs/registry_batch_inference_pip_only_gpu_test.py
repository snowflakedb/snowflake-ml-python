"""Integration tests for pip-only GPU model packaging with batch inference. """

import inflection
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model._client.model import batch_inference_job_specs
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry import pip_only_packaging_integ_util
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


@absltest.skip("SNOW-3691688")
class TestBatchInferencePipOnlyGpuInteg(
    pip_only_packaging_integ_util.PipOnlyPackagingIntegMixin,
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase,
):
    """Integration tests for pip-only GPU model batch inference."""

    def _get_batch_image_override_session_params(self) -> dict[str, str]:
        params = super()._get_batch_image_override_session_params()
        params.pop("SPCS_MODEL_INFERENCE_ENGINE_CONTAINER_URLS", None)
        return params

    def test_pip_only_xgboost_gpu_batch_inference(self) -> None:
        """E2E test: pip-only GPU batch inference with XGBoost.

        Verifies that an XGBoost model deployed via the pip-only path with GPU
        produces correct predictions. XGBoost >= 2.0 includes GPU support natively
        via pip (no conda py-xgboost-gpu substitution needed).
        """
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=10, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        regressor.fit(cal_X_train, cal_y_train)

        model_output = regressor.predict(cal_X_test)
        model_output_df = pd.DataFrame({"output_feature_0": model_output})

        input_df, expected_predictions = self._prepare_batch_inference_data(cal_X_test, model_output_df)
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=regressor,
            sample_input_data=cal_X_test,
            X=input_df,
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            resources_spec=batch_inference_job_specs.ResourcesSpec(gpu_requests="1"),
            inference_spec=batch_inference_job_specs.InferenceSpec(num_workers=1),
            job_name=job_name,
            replicas=1,
            pip_requirements=[f"xgboost=={xgboost.__version__}"],
            options={
                "cuda_version": model_env.DEFAULT_CUDA_VERSION,
                "enable_explainability": False,
            },
            expected_predictions=expected_predictions,
            conda_dependencies=[],
        )


if __name__ == "__main__":
    absltest.main()
