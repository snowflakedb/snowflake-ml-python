import inflection
import pandas as pd
import xgboost
from absl.testing import absltest, parameterized
from packaging import version
from sklearn import datasets, model_selection

from snowflake.ml._internal.utils import snowflake_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None},
        {"gpu_requests": "1", "cpu_requests": None, "memory_requests": None},
        {"gpu_requests": None, "cpu_requests": "1", "memory_requests": "8Gi"},
    )
    def test_end_to_end_pipeline(
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
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)

        if snowflake_env.get_current_snowflake_version(self.session, statement_params=None) > version.parse("9.3.0"):
            # CPU and memory argument is only supported in Snowflake > 9.3.0.
            # Remove this if-else condition when Snowflake version is upgraded to 9.3.0.
            mv = self._test_registry_model_deployment(
                model=regressor,
                sample_input_data=cal_X_test,
                prediction_assert_fns={
                    "predict": (
                        cal_X_test,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                            rtol=1e-3,
                            atol=1e-3,
                            check_dtype=False,
                        ),
                    ),
                },
                options=(
                    {"cuda_version": "11.8", "enable_explainability": False}
                    if gpu_requests
                    else {"enable_explainability": False}
                ),
                gpu_requests=gpu_requests,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
            )
        else:
            mv = self._test_registry_model_deployment(
                model=regressor,
                sample_input_data=cal_X_test,
                prediction_assert_fns={
                    "predict": (
                        cal_X_test,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                            rtol=1e-3,
                            atol=1e-3,
                            check_dtype=False,
                        ),
                    ),
                },
                options=(
                    {"cuda_version": "11.8", "enable_explainability": False}
                    if gpu_requests
                    else {"enable_explainability": False}
                ),
                gpu_requests=gpu_requests,
            )

        services_df = mv.list_services()
        services = services_df["name"]
        self.assertLen(services, 1)

        for service in services:
            mv.delete_service(service)

        services_df = mv.list_services()
        self.assertLen(services_df, 0)


if __name__ == "__main__":
    absltest.main()
