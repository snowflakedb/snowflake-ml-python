"""Integration tests for pip-only GPU model packaging with batch inference. """

import inflection
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import custom_model
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class PipOnlyPyTorchModel(custom_model.CustomModel):
    """A custom model with PyTorch dependency for GPU batch inference testing."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        import torch
        import torch.nn as nn

        # Use a tiny ``nn.Module`` forward pass (not raw tensor ops) so inference matches
        # typical PyTorch serving. Weights are fixed inside ``predict`` because CustomModel
        # serialization only persists the class and ModelContext, not arbitrary instance state.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = nn.Linear(in_features=1, out_features=1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2.0)
        layer = layer.to(device)
        layer.eval()
        x = torch.tensor(input["value"].values, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            y = layer(x)
        return pd.DataFrame({"output": y.detach().cpu().numpy().ravel()})


class TestRegistryPipOnlyGpuBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Integration tests for pip-only GPU model batch inference."""

    _BATCH_IMAGE_OVERRIDE_MODE = "pip_only_batch"

    def _get_batch_image_override_session_params(self) -> dict[str, str]:
        params = super()._get_batch_image_override_session_params()
        params.pop("SPCS_MODEL_INFERENCE_ENGINE_CONTAINER_URLS", None)
        return params

    def test_pip_only_pytorch_gpu_batch_inference(self) -> None:
        """E2E test: pip-only GPU batch inference with PyTorch."""
        if not self._has_image_override():
            self.skipTest("Skipping pip-only GPU batch inference test: image override not enabled.")

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        model = PipOnlyPyTorchModel(custom_model.ModelContext())
        input_pandas_df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # Placeholder output columns only; we assert batch results via ``prediction_assert_fn``
        # instead of calling ``predict`` here (avoids duplicating inference outside the job).
        placeholder_output = pd.DataFrame({"output": [0.0] * len(input_pandas_df)})
        input_df, _ = self._prepare_batch_inference_data(input_pandas_df, placeholder_output)
        sp_df = self.session.create_dataframe(input_pandas_df)
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def _assert_linear_double(actual_output: pd.DataFrame) -> None:
            idx = self._INDEX_COL
            ordered = actual_output.sort_values(idx).reset_index(drop=True)
            pd.testing.assert_series_equal(
                ordered["output"],
                ordered["value"] * 2.0,
                check_names=False,
                rtol=1e-4,
                atol=1e-4,
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
                gpu_requests="1",
            ),
            pip_requirements=["torch>=2.0"],
            options={
                "cuda_version": model_env.DEFAULT_CUDA_VERSION,
                "enable_explainability": False,
            },
            prediction_assert_fn=_assert_linear_double,
            conda_dependencies=[],
        )

    def test_pip_only_xgboost_gpu_batch_inference(self) -> None:
        """E2E test: pip-only GPU batch inference with XGBoost.

        Verifies that an XGBoost model deployed via the pip-only path with GPU
        produces correct predictions. XGBoost >= 2.0 includes GPU support natively
        via pip (no conda py-xgboost-gpu substitution needed).
        """
        if not self._has_image_override():
            self.skipTest("Skipping pip-only GPU batch inference test: image override not enabled.")

        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

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
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                gpu_requests="1",
            ),
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
