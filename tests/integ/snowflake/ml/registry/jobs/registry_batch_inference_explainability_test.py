import inflection
import pandas as pd
import shap
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import JobSpec, OutputSpec, model_signature
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class RegistryBatchInferenceExplainabilityTest(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @absltest.skip("Skipping test until vLLM PrPr")
    def test_xgb_booster_with_signature_and_sample_data(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        y_pred = pd.DataFrame(
            regressor.predict(xgboost.DMatrix(data=cal_X_test)),
            columns=["output_feature_0"],
        )

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        cal_X_test = pd.DataFrame(cal_X_test)
        expected_explanations = shap.TreeExplainer(regressor)(cal_X_test).values
        # Create DataFrame with string column names to avoid sorting issues with mixed types
        explanation_columns = [f"explanation_{i}" for i in range(expected_explanations.shape[1])]
        expected_explanations_df = pd.DataFrame(expected_explanations, columns=explanation_columns)
        input_df, expected_predictions = self._prepare_batch_inference_data(cal_X_test, expected_explanations_df)

        sig = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}

        self._test_registry_batch_inference(
            model=regressor,
            sample_input_data=cal_X_test,
            options={"enable_explainability": True},
            signatures=sig,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, function_name="explain"),
        )


if __name__ == "__main__":
    absltest.main()
