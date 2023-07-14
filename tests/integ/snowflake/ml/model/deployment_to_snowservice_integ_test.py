#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import uuid
from unittest import SkipTest

import pandas as pd
import pytest
import sklearn.base
import sklearn.datasets as datasets
from absl.testing import absltest
from sklearn import neighbors

from snowflake.ml.model import _model as model_api, custom_model
from snowflake.ml.model._deploy_client.snowservice import (
    deploy as snowservice_api,
    deploy_options,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager

_IRIS = datasets.load_iris(as_frame=True)
_IRIS_X = _IRIS.data
_IRIS_Y = _IRIS.target


def _get_sklearn_model() -> "sklearn.base.BaseEstimator":
    knn_model = neighbors.KNeighborsClassifier()
    knn_model.fit(_IRIS_X, _IRIS_Y)
    return knn_model


@pytest.mark.pip_incompatible
class DeploymentToSnowServiceIntegTest(absltest.TestCase):
    _RUN_ID = uuid.uuid4().hex[:2]
    # Upper is necessary for `db, schema and repo names for an image repo must be unquoted identifiers.`
    TEST_DB = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "db").upper()
    TEST_SCHEMA = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "schema").upper()
    TEST_STAGE = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "stage").upper()
    TEST_IMAGE_REPO = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "repo").upper()
    TEST_ROLE = "SYSADMIN"
    TEST_COMPUTE_POOL = "MODEL_DEPLOYMENT_INTEG_TEST_POOL"  # PRE-CREATED
    CONNECTION_NAME = "snowservice"  # PRE-CREATED AND STORED IN KEY VAULT

    @classmethod
    def setUpClass(cls) -> None:
        try:
            login_options = connection_params.SnowflakeLoginOptions(connection_name=cls.CONNECTION_NAME)
        except KeyError:
            raise SkipTest("SnowService connection parameters not present: skipping SnowServicesIntegTest.")

        deployment_name = login_options["host"].split(".")[1]
        registry_host = f"{deployment_name}-{login_options['account']}.registry-dev.snowflakecomputing.com"

        cls.FULL_IMAGE_REPO_PATH = f"{registry_host}/{cls.TEST_DB}/{cls.TEST_SCHEMA}/{cls.TEST_IMAGE_REPO}/".lower()

        cls._session = Session.builder.configs({**login_options}).create()
        cls._db_manager = db_manager.DBManager(cls._session)
        cls._db_manager.set_role(cls.TEST_ROLE)
        cls._db_manager.create_stage(cls.TEST_STAGE, cls.TEST_SCHEMA, cls.TEST_DB, sse_encrypted=True)
        cls._db_manager.create_image_repo(cls.TEST_IMAGE_REPO)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._db_manager.drop_image_repo(cls.TEST_IMAGE_REPO)
        # Dropping the db/schema will implicitly terminate the service function and snowservice as well.
        cls._db_manager.drop_database(cls.TEST_DB)
        cls._session.close()

    def setUp(self) -> None:
        # Set up a unique id for each artifact, in addition to the class-level prefix. This is particularly useful when
        # differentiating artifacts generated between different test cases, such as service function names.
        self.uid = uuid.uuid4().hex[:4]

    def _save_model_to_stage(self, model: custom_model.CustomModel, sample_input: pd.DataFrame) -> str:
        stage_path = f"@{self.TEST_STAGE}/{self.uid}/model.zip"
        model_api.save_model(
            name="model",
            session=self._session,
            model_stage_file_path=stage_path,
            model=model,
            sample_input=sample_input,
        )
        return stage_path

    def test_deployment_workflow(self) -> None:
        model_stage_file_path = self._save_model_to_stage(model=_get_sklearn_model(), sample_input=_IRIS_X)
        service_func_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._RUN_ID, f"func_{self.uid}"
        )
        deployment_options: deploy_options.SnowServiceDeployOptionsTypedHint = {
            "stage": self.TEST_STAGE,
            "compute_pool": self.TEST_COMPUTE_POOL,
            "image_repo": self.FULL_IMAGE_REPO_PATH,
        }
        snowservice_api._deploy(
            self._session,
            model_id=uuid.uuid4().hex,
            service_func_name=service_func_name,
            model_zip_stage_path=model_stage_file_path,
            **deployment_options,
        )


if __name__ == "__main__":
    absltest.main()
