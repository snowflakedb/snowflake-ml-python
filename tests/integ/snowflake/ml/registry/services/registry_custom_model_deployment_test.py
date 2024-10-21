import json
import tempfile

import inflection
import numpy as np
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


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
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(my_custom_model.predict(cal_X_test), axis=1), rtol=1e-3
                    ),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
