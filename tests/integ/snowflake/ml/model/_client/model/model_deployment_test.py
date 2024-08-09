import inspect
import time
import uuid

import numpy as np
from absl.testing import absltest
from sklearn import datasets, linear_model, svm

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class ModelDeploymentTest(absltest.TestCase):
    """Test model container services deployment."""

    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _SPCS_EAI = "SPCS_EGRESS_ACCESS_INTEGRATION"

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        login_options = connection_params.SnowflakeLoginOptions()

        self._run_id = uuid.uuid4().hex[:2]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()
        self._test_image_repo = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "image_repo"
        ).upper()

        self._session = Session.builder.configs(
            {
                **login_options,
                **{"database": self._test_db, "schema": self._test_schema},
            }
        ).create()

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.create_image_repo(self._test_image_repo)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self._session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    @absltest.skip
    def test_create_service(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)

        model_name = f"model_{inspect.stack()[1].function}"
        version_name = f"ver_{self._run_id}"
        mv = self.registry.log_model(
            model=regr,
            model_name=model_name,
            version_name=version_name,
            sample_input_data=iris_X,
        )

        service = f"service_{self._run_id}"
        mv.create_service(
            service_name=service,
            image_build_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            image_repo=self._test_image_repo,
            force_rebuild=True,
            build_external_access_integration=self._SPCS_EAI,
        )
        self.assertTrue(self._wait_for_service(service))

    @absltest.skip
    def test_inference(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        svc = svm.LinearSVC()
        svc.fit(iris_X, iris_y)

        model_name = f"model_{inspect.stack()[1].function}"
        version_name = f"ver_{self._run_id}"
        mv = self.registry.log_model(
            model=svc,
            model_name=model_name,
            version_name=version_name,
            sample_input_data=iris_X,
        )

        service = f"service_{self._run_id}"
        mv.create_service(
            service_name=service,
            image_build_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            image_repo=self._test_image_repo,
            force_rebuild=True,
            build_external_access_integration=self._SPCS_EAI,
        )
        self.assertTrue(self._wait_for_service(service))

        res = mv.run(iris_X, function_name="predict", service_name=service)
        np.testing.assert_allclose(res["output_feature_0"].values, svc.predict(iris_X))

    def _wait_for_service(self, service: str) -> bool:
        service_identifier = sql_identifier.SqlIdentifier(service).identifier()

        # wait for service creation
        while True:
            services = [serv["name"] for serv in self._session.sql("SHOW SERVICES").collect()]
            if service_identifier not in services:
                time.sleep(10)
            else:
                break

        # wait for service to run
        while True:
            status = self._session.sql(f"DESC SERVICE {service_identifier}").collect()[0]["status"]
            if status == "RUNNING":
                return True
            elif status == "PENDING":
                time.sleep(10)
            else:
                return False


if __name__ == "__main__":
    absltest.main()
