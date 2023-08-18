#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import uuid

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import deploy_platforms
from tests.integ.snowflake.ml.registry.model_registry_integ_test_snowservice_base import (
    TestModelRegistryIntegSnowServiceBase,
)
from tests.integ.snowflake.ml.test_utils import model_factory


class TestModelRegistryIntegWithSnowServiceDeployment(TestModelRegistryIntegSnowServiceBase):
    @pytest.mark.pip_incompatible
    def test_snowml_model_deployment_xgboost(self) -> None:
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
                    "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
                    "enable_remote_image_build": True,
                },
            },
        )


if __name__ == "__main__":
    absltest.main()
