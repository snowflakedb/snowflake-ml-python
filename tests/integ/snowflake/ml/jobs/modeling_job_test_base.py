import os
from typing import Any, Optional

import numpy as np
from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as sp_exceptions
from tests.integ.snowflake.ml.jobs import (
    reflection_utils,
    test_constants,
    test_file_helper,
)
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

_TRAIN_MODEL_FUNC = "train_model"

_PREDICT_FUNC = "predict_result"


@absltest.skipIf(
    (region := test_env_utils.get_current_snowflake_region()) is None
    or region["cloud"] not in test_constants._SUPPORTED_CLOUDS,
    "Test only for SPCS supported clouds",
)
class BaseModelTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)
        cls.dbm.cleanup_schemas(prefix=test_constants._TEST_SCHEMA, expire_days=1)
        cls.db = cls.session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            cls.compute_pool = cls.dbm.create_compute_pool(
                test_constants._TEST_COMPUTE_POOL, sql_client.CreationMode(if_not_exists=True), max_nodes=5
            )
        except sp_exceptions.SnowparkSQLException:
            if not cls.dbm.show_compute_pools(test_constants._TEST_COMPUTE_POOL).count() > 0:
                raise cls.failureException(
                    f"Compute pool {test_constants._TEST_COMPUTE_POOL} not available and could not be created"
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbm.drop_schema(cls.schema, if_exists=True)
        super().tearDownClass()

    def get_inference(self, model: Any, module_path: str) -> Any:
        return reflection_utils.run_reflected_func(module_path, _PREDICT_FUNC, model)

    def get_model(self, model_name: str, module_path: str) -> Any:
        return reflection_utils.run_reflected_func(module_path, _TRAIN_MODEL_FUNC, model_name)

    def train_models(
        self,
        model_name: str,
        model_script: str,
        pip_requirements: Optional[list[str]] = None,
        external_access_integrations: Optional[list[str]] = None,
    ) -> None:
        payload = test_file_helper.TestAsset(model_script)
        job = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            pip_requirements=pip_requirements,
            args=[model_name],
            external_access_integrations=external_access_integrations,
        )
        self.assertIsNotNone(job)
        module_path = f"test_files.{os.path.splitext(model_script)[0].replace('/', '.')}"
        model_local = self.get_model(model_name, module_path)
        inference_local = self.get_inference(model_local, module_path)
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        model_remote = job.result()
        inference_remote = self.get_inference(model_remote, module_path)
        np.testing.assert_allclose(inference_local, inference_remote, atol=1e-6)
