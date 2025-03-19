import json
import os
import tempfile

import inflection
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
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


class TestRegistryCustomModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    def test_custom_model(
        self,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
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
        )

    @absltest.skipIf(os.getenv("BASE_CPU_IMAGE_PATH", None) is None, "BASE_CPU_IMAGE_PATH is not set")
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


if __name__ == "__main__":
    absltest.main()
