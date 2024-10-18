import inflection
import numpy as np
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    def test_end_to_end_pipeline(
        self,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        mv = self._test_registry_model_deployment(
            model=regressor,
            sample_input_data=cal_X_test,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(regressor.predict(cal_X_test), axis=1), rtol=1e-3
                    ),
                ),
            },
        )

        services_df = mv.list_services()
        services = services_df["service_name"]
        self.assertLen(services, 1)

        for service in services:
            mv.delete_service(service)

        services_df = mv.list_services()
        services = services_df["service_name"]
        self.assertEmpty(services)


if __name__ == "__main__":
    absltest.main()
