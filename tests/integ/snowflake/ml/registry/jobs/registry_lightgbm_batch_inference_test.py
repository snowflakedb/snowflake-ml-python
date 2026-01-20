import uuid

import inflection
import lightgbm
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestLightGbmModelBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"cpu_requests": None, "replicas": 1},
        {"cpu_requests": "3", "replicas": 2},
    )
    def test_lightgbm_batch_inference(
        self,
        replicas: int,
        cpu_requests: str,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)

        # Generate expected predictions using the original model
        model_output = classifier.predict(cal_X_test)
        model_output_df = pd.DataFrame({"output_feature_0": model_output})

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(cal_X_test, model_output_df)

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=classifier,
            sample_input_data=cal_X_test,
            X=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=2,
            service_name=f"batch_inference_{name}",
            replicas=replicas,
            function_name="predict",
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
