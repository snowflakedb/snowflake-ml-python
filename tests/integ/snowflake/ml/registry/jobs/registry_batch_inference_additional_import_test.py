import importlib
import os
import shutil
import tempfile
from functools import partial
from typing import Literal

import importlib_resources
import numpy as np
import pandas as pd
import xgboost as xgb
from absl.testing import absltest, parameterized
from sklearn import compose, datasets, impute, pipeline, preprocessing

from snowflake.ml.model import custom_model, model_signature
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.registry.model.my_module.utils import column_labeller


class RegistryBatchInferenceAdditionalImportTest(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @parameterized.parameters([{"import_method": "ext_modules"}, {"import_method": "code_paths"}])
    def test_additional_import(self, import_method: Literal["ext_modules", "code_paths"]) -> None:
        name = f"model_{self._run_id}"
        version = f"ver_{import_method}"

        X, y = datasets.make_classification(random_state=42)
        X = pd.DataFrame(X, columns=["X" + str(i) for i in range(20)])
        log_trans = pipeline.Pipeline(
            [
                ("impute", impute.SimpleImputer()),
                ("scaler", preprocessing.MinMaxScaler()),
                (
                    "logger",
                    preprocessing.FunctionTransformer(
                        np.log1p,
                        feature_names_out=partial(column_labeller, "LOG"),
                    ),
                ),
            ]
        )
        preproc_pipe = compose.ColumnTransformer(
            [("log", log_trans, ["X0", "X1"])],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        preproc_pipe.set_output(transform="pandas")
        preproc_pipe.fit(X, y)

        xgb_data = xgb.DMatrix(preproc_pipe.transform(X), y)
        booster = xgb.train(dict(max_depth=5, seed=42), xgb_data, num_boost_round=10)

        # Generate expected predictions for batch inference validation
        model_output = booster.predict(xgb.DMatrix(preproc_pipe.transform(X)))
        model_output_df = pd.DataFrame({"output": model_output})

        # Prepare batch inference data with INDEX column
        input_spec, expected_predictions = self._prepare_batch_inference_data(X, model_output_df)

        # Prepare service name and output stage for batch inference
        service_name, output_stage_location, _ = self._prepare_service_name_and_stage_for_batch_inference()

        class MyModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                xgb_data = xgb.DMatrix(self.context.model_ref("pipeline").transform(X))
                preds = self.context.model_ref("model").predict(xgb_data)
                res_df = pd.DataFrame({"output": preds})
                return res_df

        my_model = MyModel(
            custom_model.ModelContext(
                models={
                    "pipeline": preproc_pipe,
                    "model": booster,
                },
                artifacts={},
            )
        )

        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name=f"X{i}") for i in range(20)],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="output")],
        )

        if import_method == "ext_modules":
            my_module = importlib.import_module("tests.integ.snowflake.ml.registry.model.my_module")
            mv = self.registry.log_model(
                my_model,
                model_name=name,
                version_name=version,
                signatures={"predict": sig},
                ext_modules=[my_module],
            )
        else:
            src_path = importlib_resources.files("tests.integ.snowflake.ml.registry.model.my_module")
            with tempfile.TemporaryDirectory() as tmpdir:
                dst_path = os.path.join(tmpdir, "tests", "integ", "snowflake", "ml", "registry", "model", "my_module")
                shutil.copytree(src_path, dst_path)

                code_path = os.path.join(tmpdir, "tests")
                mv = self.registry.log_model(
                    my_model,
                    model_name=name,
                    version_name=version,
                    signatures={"predict": sig},
                    code_paths=[code_path],
                )

        self._deploy_batch_inference(
            mv,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            service_name=service_name,
            function_name="predict",
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
