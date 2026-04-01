import logging
from typing import Callable, TypeVar

import pandas as pd
from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import db_manager

T = TypeVar("T")


class RegistryModelDeploymentPrivilegeTest(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests verifying privilege requirements for service-based model inference.

    Validates that a user with appropriate privileges on a model and service can invoke
    inference via mv.run(X, service_name=...) and through SQL.

    Required privileges for service-based inference:
        - USAGE on DATABASE and SCHEMA containing the model/service
        - USAGE on WAREHOUSE
        - USAGE on MODEL
        - USAGE on SERVICE
        - SERVICE ROLE <service>!INFERENCE_SERVICE_FUNCTION_USAGE (for calling service functions)

    Test Isolation:
        Each test method gets a fresh session (see CommonTestBase.setUp/tearDown),
        so role changes do not leak between tests or affect parallel execution.
    """

    def setUp(self) -> None:
        super().setUp()

        self._admin_role = self.session.get_current_role().strip('"')
        self._test_warehouse = self.session.get_current_warehouse()

        self._usage_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "SVC_USAGE_ROLE"
        ).upper()

        self._db_manager.create_role(self._usage_role)

        current_user = self.session.get_current_user().strip('"')
        self.session.sql(f"GRANT ROLE {self._usage_role} TO USER {current_user}").collect()

        self.session.sql(f"GRANT USAGE ON DATABASE {self._test_db} TO ROLE {self._usage_role}").collect()
        self.session.sql(
            f"GRANT USAGE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {self._usage_role}"
        ).collect()
        try:
            self.session.sql(f"GRANT USAGE ON WAREHOUSE {self._test_warehouse} TO ROLE {self._usage_role}").collect()
        except Exception:
            logging.error(
                f"Failed to grant warehouse {self._test_warehouse} to role {self._usage_role}. "
                "Tests requiring warehouse access under this role will fail. "
                "Ensure the current role has GRANT OPTION on the warehouse."
            )

    def tearDown(self) -> None:
        self.session.use_role(self._admin_role)
        self._db_manager.drop_role(self._usage_role, if_exists=True)

        super().tearDown()

    def _run_as_role(self, role: str, fn: Callable[[], T]) -> T:
        """Execute a callable under a specific role with secondary roles disabled."""
        prev_role = self.session.get_current_role()
        try:
            self.session.sql("USE SECONDARY ROLES NONE").collect()
            self.session.use_role(role)
            return fn()
        finally:
            self.session.use_role(prev_role)
            self.session.sql("USE SECONDARY ROLES ALL").collect()

    def test_usage_can_run_service_inference(self) -> None:
        """A user with appropriate privileges on model and service can run inference."""
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        test_input = pd.DataFrame(iris_X[:5], columns=[f"input_feature_{i}" for i in range(len(iris_X[0]))])
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X, iris_y)

        service_name = f"service_priv_test_{self._run_id}"
        fully_qualified_service_name = f"{self._test_db}.{self._test_schema}.{service_name}"

        mv = self._test_registry_model_deployment(
            model=classifier,
            sample_input_data=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(classifier.predict(iris_X), columns=res.columns),
                        rtol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            service_name=service_name,
            options={"enable_explainability": False},
        )
        fully_qualified_model_name = mv.fully_qualified_model_name

        self.session.use_role(self._admin_role)

        self.session.sql(f"GRANT USAGE ON MODEL {fully_qualified_model_name} TO ROLE {self._usage_role}").collect()
        self.session.sql(f"GRANT USAGE ON SERVICE {fully_qualified_service_name} TO ROLE {self._usage_role}").collect()
        self.session.sql(
            f"GRANT MONITOR ON SERVICE {fully_qualified_service_name} TO ROLE {self._usage_role}"
        ).collect()

        self.session.sql(
            f"GRANT SERVICE ROLE {fully_qualified_service_name}!INFERENCE_SERVICE_FUNCTION_USAGE "
            f"TO ROLE {self._usage_role}"
        ).collect()

        def _test_sql_inference() -> None:
            """Test direct SQL inference with service function."""
            row_values = test_input.iloc[0].values
            input_args_sql = ", ".join(str(v) for v in row_values)

            sql_query = f"SELECT {fully_qualified_service_name}!predict({input_args_sql}) AS prediction"
            sql_result = self.session.sql(sql_query).collect()
            self.assertEqual(len(sql_result), 1)
            self.assertIsNotNone(sql_result[0]["PREDICTION"], "Prediction should not be null")

        def _test_python_sdk_inference() -> None:
            """Test Python SDK inference via mv.run() with service_name."""
            reg = registry.Registry(self.session)
            model = reg.get_model(mv.model_name)
            mv_as_usage = model.version(mv.version_name)

            result = mv_as_usage.run(test_input, function_name="predict", service_name=fully_qualified_service_name)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(test_input))

        self._run_as_role(self._usage_role, _test_sql_inference)
        self._run_as_role(self._usage_role, _test_python_sdk_inference)


if __name__ == "__main__":
    absltest.main()
