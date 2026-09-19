from absl.testing import absltest
from sklearn import datasets

from snowflake.ml.model._client.model import batch_inference_job_specs
from snowflake.ml.modeling.xgboost import XGBRegressor
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestBatchInferenceModelingInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def test_snowml_xgboost_batch_inference(self) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"

        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(iris_X)

        test_features = iris_X[INPUT_COLUMNS]

        model_output = regr.predict(iris_X)[[OUTPUT_COLUMNS]]

        input_df, expected_predictions = self._prepare_batch_inference_data(test_features, model_output)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=regr,
            sample_input_data=test_features,
            X=input_df,
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            inference_spec=batch_inference_job_specs.InferenceSpec(num_workers=1),
            function_name="predict",
            job_name=job_name,
            replicas=1,
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
