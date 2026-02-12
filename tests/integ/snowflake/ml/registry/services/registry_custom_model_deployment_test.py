import json
import tempfile

import inflection
import numpy as np
import pandas as pd
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.registry.services.registry_model_deployment_test_base import (
    INFERENCE_IMAGE_BUILDER,
    KANIKO_BUILDER,
)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class MyCustomModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        v = open(context.path("config")).read()
        self.bias = json.loads(v)["bias"]

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        model_output = self.context.model_ref("regressor").predict(input)
        return pd.DataFrame({"output": model_output + self.bias})


class WideInputModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        numeric_sum = input.select_dtypes(include=[np.number]).sum(axis=1)
        string_count = input.select_dtypes(include=[object, "string"]).count(axis=1)
        result = numeric_sum + string_count
        return pd.DataFrame({"output": result})


class TestRegistryCustomModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"builder_type": KANIKO_BUILDER},
        {"builder_type": INFERENCE_IMAGE_BUILDER},
    )
    def test_custom_model(
        self,
        builder_type: str,
    ) -> None:
        # inference_image_builder tests only run when image override is enabled
        if builder_type == INFERENCE_IMAGE_BUILDER and not self._has_image_override():
            self.skipTest("Skipping inference_image_builder test: image override not enabled.")

        use_inference_image_builder = builder_type == INFERENCE_IMAGE_BUILDER

        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=10, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        regressor.fit(cal_X_train, cal_y_train)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump({"bias": 0.1}, f)
            temp_config_file = f.name

        mc = custom_model.ModelContext(
            artifacts={"config": temp_config_file},
            models={
                "regressor": regressor,
            },
        )

        my_custom_model = MyCustomModel(mc)

        self._test_registry_model_deployment(
            model=my_custom_model,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        my_custom_model.predict(cal_X_test),
                        rtol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
            use_inference_image_builder=use_inference_image_builder,
        )

    def test_udf_500_column_limit(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        num_cols = 501

        sp_df = self.session.create_dataframe(
            [[0] * num_cols, [1] * num_cols], schema=[f'"c{i}"' for i in range(num_cols)]
        )
        y_df_expected = pd.DataFrame([[0], [1]], columns=["output"])

        self._test_registry_model_deployment(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df.to_pandas(),
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        y_df_expected,
                        check_dtype=False,
                    ),
                ),
            },
        )

    def test_custom_model_wide_input(self) -> None:
        n_samples = 10
        n_features = 600
        data = {}

        for i in range(n_features):
            if i % 3 == 0:
                col_name = f'"z_feature_{i:03d}"'
                data[col_name] = np.random.randint(0, 10, n_samples)
            elif i % 3 == 1:
                col_name = f'"b_feature_{i:03d}"'
                data[col_name] = np.random.choice(["X", "Y", "Z"], n_samples)
            else:
                col_name = f"a_feature_{i:03d}"
                data[col_name] = np.random.normal(5, 2, n_samples)

        train_df = pd.DataFrame(data)
        sp_train_df = self.session.create_dataframe(train_df)
        wide_model = WideInputModel(custom_model.ModelContext())
        expected_predictions = wide_model.predict(train_df)

        self._test_registry_model_deployment(
            model=wide_model,
            sample_input_data=sp_train_df,
            prediction_assert_fns={
                "predict": (
                    train_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        expected_predictions,
                        check_dtype=False,
                    ),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
