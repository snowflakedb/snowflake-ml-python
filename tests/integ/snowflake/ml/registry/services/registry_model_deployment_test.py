from typing import Optional

import inflection
import pandas as pd
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.registry.services.registry_model_deployment_test_base import (
    INFERENCE_IMAGE_BUILDER,
    KANIKO_BUILDER,
)


class TestRegistryModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None, "builder_type": KANIKO_BUILDER},
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None, "builder_type": INFERENCE_IMAGE_BUILDER},
        {"gpu_requests": "1", "cpu_requests": None, "memory_requests": None, "builder_type": KANIKO_BUILDER},
        {"gpu_requests": "1", "cpu_requests": None, "memory_requests": None, "builder_type": INFERENCE_IMAGE_BUILDER},
        {"gpu_requests": None, "cpu_requests": "1", "memory_requests": "8Gi", "builder_type": KANIKO_BUILDER},
    )
    def test_end_to_end_pipeline(
        self,
        gpu_requests: Optional[str],
        cpu_requests: Optional[str],
        memory_requests: Optional[str],
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
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)

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
                {"cuda_version": model_env.DEFAULT_CUDA_VERSION, "enable_explainability": False}
                if gpu_requests
                else {"enable_explainability": False}
            ),
            gpu_requests=gpu_requests,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            use_inference_image_builder=use_inference_image_builder,
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
