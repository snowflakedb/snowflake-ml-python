import re
import uuid

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import deploy_platforms
from tests.integ.snowflake.ml.registry.model_registry_snowservice_integ_test_base import (
    TestModelRegistryIntegSnowServiceBase,
)
from tests.integ.snowflake.ml.test_utils import model_factory


class TestModelRegistryIntegWithSnowServiceDeployment(TestModelRegistryIntegSnowServiceBase):
    def test_snowml_model_deployment_xgboost(self) -> None:
        def _run_deployment() -> None:
            self._test_snowservice_deployment(
                model_name="xgboost_model",
                model_version=uuid.uuid4().hex,
                prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_snowml_model_xgb,
                prediction_assert_fn=lambda local_prediction, remote_prediction: pd.testing.assert_frame_equal(
                    remote_prediction, local_prediction, check_dtype=False
                ),
                deployment_options={
                    "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                    "target_method": "predict",
                    "options": {
                        "compute_pool": self._TEST_CPU_COMPUTE_POOL,
                        "enable_remote_image_build": True,
                    },
                },
            )

        # First deployment
        _run_deployment()

        # Second deployment. Ensure image building is skipped due to similar environment.
        with self.assertLogs(level="WARNING") as cm:
            _run_deployment()
            image_pattern = r"Using existing image .* to skip image build"
            image_pattern_found = any(re.search(image_pattern, s, re.MULTILINE | re.DOTALL) for s in cm.output)
            self.assertTrue(image_pattern_found, "Should skip image build on second deployment")

    def test_snowml_model_deployment_xgboost_with_model_in_image(self) -> None:
        def _run_deployment() -> None:
            self._test_snowservice_deployment(
                model_name="xgboost_model",
                model_version=uuid.uuid4().hex,
                prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_snowml_model_xgb,
                prediction_assert_fn=lambda local_prediction, remote_prediction: pd.testing.assert_frame_equal(
                    remote_prediction, local_prediction, check_dtype=False
                ),
                deployment_options={
                    "platform": deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                    "target_method": "predict",
                    "options": {
                        "compute_pool": self._TEST_CPU_COMPUTE_POOL,
                        "enable_remote_image_build": True,
                        "model_in_image": True,
                    },
                },
            )

        _run_deployment()


if __name__ == "__main__":
    absltest.main()
