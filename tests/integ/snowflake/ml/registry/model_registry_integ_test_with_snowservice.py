#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import functools
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import SkipTest

import numpy as np
import pandas as pd
import pytest
from absl.testing import absltest, parameterized

from snowflake.ml.model import _deployer
from snowflake.ml.registry import model_registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager, model_factory


class TestModelRegistryIntegWithSnowServiceDeployment(parameterized.TestCase):
    _SNOWSERVICE_CONNECTION_NAME = "snowservice"
    _TEST_CPU_COMPUTE_POOL = "MODEL_DEPLOYMENT_INTEG_TEST_POOL"
    _TEST_GPU_COMPUTE_POOL = "MODEL_DEPLOYMENT_INTEG_TEST_POOL_GPU_3"
    _RUN_ID = uuid.uuid4().hex[:2]
    _TEST_DB = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "db").upper()
    _TEST_SCHEMA = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "schema").upper()
    _TEST_IMAGE_REPO = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "repo").upper()
    _TEST_ROLE = "SYSADMIN"
    _TEST_WAREHOUSE = "SNOW_ML_XSMALL"

    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        try:
            login_options = connection_params.SnowflakeLoginOptions(connection_name=cls._SNOWSERVICE_CONNECTION_NAME)
        except KeyError:
            raise SkipTest(
                "SnowService connection parameters not present: skipping "
                "TestModelRegistryIntegWithSnowServiceDeployment."
            )

        cls._session = Session.builder.configs(
            {
                **login_options,
                **{"database": cls._TEST_DB, "schema": cls._TEST_SCHEMA},
            }
        ).create()

        cls._db_manager = db_manager.DBManager(cls._session)
        cls._db_manager.set_role(cls._TEST_ROLE)
        cls._db_manager.set_warehouse(cls._TEST_WAREHOUSE)
        model_registry.create_model_registry(
            session=cls._session, database_name=cls._TEST_DB, schema_name=cls._TEST_SCHEMA
        )
        cls.registry = model_registry.ModelRegistry(
            session=cls._session, database_name=cls._TEST_DB, schema_name=cls._TEST_SCHEMA
        )
        cls._db_manager.create_image_repo(cls._TEST_IMAGE_REPO)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._db_manager.drop_image_repo(cls._TEST_IMAGE_REPO)
        cls._db_manager.drop_database(cls._TEST_DB)
        cls._session.close()

    def _test_snowservice_deployment(
        self,
        model_name: str,
        model_version: str,
        prepare_model_and_feature_fn: Callable[[], Tuple[Any, Any]],
        deployment_options: Dict[str, Any],
        conda_dependencies: Optional[List[str]] = None,
        embed_local_ml_library: Optional[bool] = True,
    ):

        model, test_features, *_ = prepare_model_and_feature_fn()

        self.registry.log_model(
            model_name=model_name,
            model_version=model_version,
            model=model,
            conda_dependencies=conda_dependencies,
            sample_input_data=test_features,
            options={"embed_local_ml_library": embed_local_ml_library},
        )

        model_ref = model_registry.ModelReference(
            registry=self.registry, model_name=model_name, model_version=model_version
        )

        deployment_name = f"{model_name}_{model_version}_deployment"
        deployment_options["deployment_name"] = deployment_name
        model_ref.deploy(**deployment_options)
        target_method = deployment_options["target_method"]
        local_prediction = getattr(model, target_method)(test_features)
        remote_prediction = model_ref.predict(deployment_name, test_features)

        if isinstance(local_prediction, np.ndarray):
            np.testing.assert_allclose(remote_prediction.to_numpy(), np.expand_dims(local_prediction, axis=1))
        else:
            pd.testing.assert_frame_equal(remote_prediction, local_prediction, check_dtype=False)

        model_deployment_list = model_ref.list_deployments().to_pandas()  # type: ignore[attr-defined]
        self.assertEqual(model_deployment_list.shape[0], 1)
        self.assertEqual(model_deployment_list["MODEL_NAME"][0], model_name)
        self.assertEqual(model_deployment_list["MODEL_VERSION"][0], model_version)
        self.assertEqual(model_deployment_list["DEPLOYMENT_NAME"][0], deployment_name)

        model_ref.delete_deployment(deployment_name=deployment_name)  # type: ignore[attr-defined]
        self.assertEqual(model_ref.list_deployments().to_pandas().shape[0], 0)  # type: ignore[attr-defined]

        self.assertEqual(self.registry.list_models().to_pandas().shape[0], 1)
        self.registry.delete_model(model_name=model_name, model_version=model_version, delete_artifact=True)
        self.assertEqual(self.registry.list_models().to_pandas().shape[0], 0)

    # TODO: doesnt work, Mismatched elements: 10 / 100 (10%). could be due to version mismatch?
    @pytest.mark.pip_incompatible
    def test_sklearn_deployment_with_snowml_conda(self) -> None:
        self._test_snowservice_deployment(
            model_name="test_sklearn_model",
            model_version=uuid.uuid4().hex,
            prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_sklearn_model,
            embed_local_ml_library=False,
            conda_dependencies=["snowflake-ml-python==1.0.2"],
            deployment_options={
                "platform": _deployer.TargetPlatform.SNOWPARK_CONTAINER_SERVICE,
                "target_method": "predict",
                "options": {
                    "compute_pool": self._TEST_CPU_COMPUTE_POOL,
                    "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
                },
            },
        )

    @pytest.mark.pip_incompatible
    def test_sklearn_deployment_with_local_source_code(self) -> None:
        self._test_snowservice_deployment(
            model_name="test_sklearn_model",
            model_version=uuid.uuid4().hex,
            prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_sklearn_model,
            deployment_options={
                "platform": _deployer.TargetPlatform.SNOWPARK_CONTAINER_SERVICE,
                "target_method": "predict",
                "options": {
                    "compute_pool": self._TEST_CPU_COMPUTE_POOL,
                    "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
                },
            },
        )

    @pytest.mark.pip_incompatible
    def test_sklearn_deployment(self) -> None:
        self._test_snowservice_deployment(
            model_name="test_sklearn_model",
            model_version=uuid.uuid4().hex,
            prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_sklearn_model,
            deployment_options={
                "platform": _deployer.TargetPlatform.SNOWPARK_CONTAINER_SERVICE,
                "target_method": "predict",
                "options": {
                    "compute_pool": self._TEST_CPU_COMPUTE_POOL,
                    "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
                },
            },
        )

    @pytest.mark.pip_incompatible
    def test_huggingface_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._test_snowservice_deployment(
                model_name="gpt2_model_gpu",
                model_version=uuid.uuid4().hex,
                conda_dependencies=["pytorch", "transformers"],
                prepare_model_and_feature_fn=functools.partial(
                    model_factory.ModelFactory.prepare_gpt2_model, local_cache_dir=tmpdir
                ),
                deployment_options={
                    "platform": _deployer.TargetPlatform.SNOWPARK_CONTAINER_SERVICE,
                    "target_method": "predict",
                    "options": {
                        "compute_pool": self._TEST_GPU_COMPUTE_POOL,
                        "use_gpu": True,
                        "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
                    },
                },
            )

    @pytest.mark.pip_incompatible
    def test_snowml_model_deployment_logistic_with_sourcecode_embedded_in_model(self) -> None:
        self._test_snowservice_deployment(
            model_name="snowml",
            model_version=uuid.uuid4().hex,
            prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_snowml_model_logistic,
            deployment_options={
                "platform": _deployer.TargetPlatform.SNOWPARK_CONTAINER_SERVICE,
                "target_method": "predict",
                "options": {
                    "compute_pool": self._TEST_GPU_COMPUTE_POOL,
                    "image_repo": self._db_manager.get_snowservice_image_repo(repo=self._TEST_IMAGE_REPO),
                },
            },
        )

    #
    # TODO[schen], SNOW-861613, investigate xgboost model prediction hanging issue when run with Gunicorn --preload
    # def test_snowml_model_deployment_xgboost(self) -> None:
    #     self._test_snowservice_deployment(
    #         model_name="snowml",
    #         model_version=uuid.uuid4().hex,
    #         prepare_model_and_feature_fn=model_factory.ModelFactory.prepare_snowml_model,
    #         deployment_options={
    #             "platform": _deployer.TargetPlatform.SNOWPARK_CONTAINER_SERVICE,
    #             "target_method": "predict",
    #             "options": {
    #                 "compute_pool": self._TEST_GPU_COMPUTE_POOL,
    #             }
    #         },
    #     )


if __name__ == "__main__":
    absltest.main()
