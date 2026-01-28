import inflection
import pandas as pd
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model import JobSpec, OutputSpec
from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestXGBoostBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None},
        {"gpu_requests": "1", "cpu_requests": None, "memory_requests": None},
        {"gpu_requests": None, "cpu_requests": "1", "memory_requests": "8Gi"},
    )
    def test_xgb(
        self,
        gpu_requests: str,
        cpu_requests: str,
        memory_requests: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=10, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        regressor.fit(cal_X_train, cal_y_train)

        # Generate expected predictions using the original model
        model_output = regressor.predict(cal_X_test)
        model_output_df = pd.DataFrame({"output_feature_0": model_output})

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(cal_X_test, model_output_df)
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=regressor,
            sample_input_data=cal_X_test,
            options=(
                {"cuda_version": model_env.DEFAULT_CUDA_VERSION, "enable_explainability": False}
                if gpu_requests
                else {"enable_explainability": False}
            ),
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                gpu_requests=gpu_requests,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
                num_workers=1,
                replicas=2,
            ),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
