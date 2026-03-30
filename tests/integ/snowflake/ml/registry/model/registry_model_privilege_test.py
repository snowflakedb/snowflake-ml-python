import logging
import tempfile
from typing import Callable, TypeVar

import pandas as pd
from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import db_manager

T = TypeVar("T")


class RegistryModelPrivilegeTest(registry_model_test_base.RegistryModelTestBase):
    """Integration tests verifying privilege requirements for model operations.

    Server-side privilege mapping (ModuleSecurityModel.java):
        EXECUTE          -> USAGE
        SHOW$MODULES     -> ANY
        STAGE$GET/LS     -> READ or OWNERSHIP
        STAGE$PUT/RM     -> OWNERSHIP
        ALTER            -> MODIFY

    Test Isolation:
        Each test method gets a fresh session (see CommonTestBase.setUp/tearDown),
        so role changes do not leak between tests or affect parallel execution.
        The _run_as_role helper uses try/finally to ensure proper cleanup even
        if assertions fail mid-test.
    """

    def setUp(self) -> None:
        super().setUp()

        self._admin_role = self.session.get_current_role().strip('"')

        self._usage_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "USAGE_ROLE"
        ).upper()
        self._read_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "READ_ROLE"
        ).upper()
        self._test_warehouse = self.session.get_current_warehouse()

        self._db_manager.create_role(self._usage_role)
        self._db_manager.create_role(self._read_role)

        current_user = self.session.get_current_user().strip('"')
        for role_name in [self._usage_role, self._read_role]:
            self.session.sql(f"GRANT ROLE {role_name} TO USER {current_user}").collect()
            self.session.sql(f"GRANT USAGE ON DATABASE {self._test_db} TO ROLE {role_name}").collect()
            self.session.sql(f"GRANT USAGE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {role_name}").collect()
            try:
                self.session.sql(f"GRANT USAGE ON WAREHOUSE {self._test_warehouse} TO ROLE {role_name}").collect()
            except Exception:
                logging.error(
                    f"Failed to grant warehouse {self._test_warehouse} to role {role_name}. "
                    "Tests requiring warehouse access under this role will fail. "
                    "Ensure the current role has GRANT OPTION on the warehouse."
                )

        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        self._test_input = pd.DataFrame(iris_X[:5], columns=[f"input_feature_{i}" for i in range(iris_X.shape[1])])

        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X, iris_y)

        self._model_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "PRIV_MODEL"
        ).upper()
        self._version_name = "V1"

        self._mv = self.registry.log_model(
            model=classifier,
            model_name=self._model_name,
            version_name=self._version_name,
            sample_input_data=iris_X,
        )
        self._fq_model_name = self._mv.fully_qualified_model_name

        self.session.sql(f"GRANT USAGE ON MODEL {self._fq_model_name} TO ROLE {self._usage_role}").collect()
        self.session.sql(f"GRANT USAGE ON MODEL {self._fq_model_name} TO ROLE {self._read_role}").collect()
        self.session.sql(f"GRANT READ ON MODEL {self._fq_model_name} TO ROLE {self._read_role}").collect()

    def tearDown(self) -> None:
        self.session.use_role(self._admin_role)
        self._db_manager.drop_role(self._usage_role, if_exists=True)
        self._db_manager.drop_role(self._read_role, if_exists=True)
        super().tearDown()

    def _run_as_role(self, role: str, fn: Callable[[], T]) -> T:
        """Execute a callable under a specific role with secondary roles disabled.

        This ensures strict RBAC testing by:
        1. Saving the current role and secondary roles state
        2. Disabling secondary roles (so only the specified role's privileges apply)
        3. Switching to the test role
        4. Running the test function
        5. Restoring the original role and secondary roles state in finally block

        The finally block guarantees cleanup even if the test function raises an exception.

        Note: Each test gets its own session (see CommonTestBase), so changes here
        do not affect other tests running in parallel or sequentially.
        """
        prev_role = self.session.get_current_role()
        try:
            self.session.sql("USE SECONDARY ROLES NONE").collect()
            self.session.use_role(role)
            return fn()
        finally:
            self.session.use_role(prev_role)
            self.session.sql("USE SECONDARY ROLES ALL").collect()

    def test_usage_can_run_python_inference(self) -> None:
        """USAGE privilege allows Python client inference (EXECUTE -> USAGE)."""

        def _test() -> None:
            reg = registry.Registry(self.session)
            mv = reg.get_model(self._model_name).version(self._version_name)
            result = mv.run(self._test_input, function_name="predict")
            self.assertIsInstance(result, pd.DataFrame)

        self._run_as_role(self._usage_role, _test)

    def test_usage_can_run_sql_inference(self) -> None:
        """USAGE privilege allows SQL inference (EXECUTE -> USAGE)."""

        def _test() -> None:
            input_row = ", ".join(str(v) for v in self._test_input.iloc[0].values)
            result = self.session.sql(f"SELECT {self._fq_model_name}!predict({input_row}) AS prediction").collect()
            self.assertEqual(len(result), 1)

        self._run_as_role(self._usage_role, _test)

    def test_usage_can_show_functions(self) -> None:
        """USAGE privilege allows metadata reads (SHOW$MODULES -> ANY)."""

        def _test() -> None:
            reg = registry.Registry(self.session)
            mv = reg.get_model(self._model_name).version(self._version_name)
            self.assertGreater(len(mv.show_functions()), 0)

        self._run_as_role(self._usage_role, _test)

    def test_usage_can_show_metrics(self) -> None:
        """USAGE privilege allows showing metrics."""

        def _test() -> None:
            reg = registry.Registry(self.session)
            mv = reg.get_model(self._model_name).version(self._version_name)
            metrics = mv.show_metrics()
            self.assertIsInstance(metrics, dict)

        self._run_as_role(self._usage_role, _test)

    def test_usage_cannot_export(self) -> None:
        """USAGE privilege cannot export (requires READ for STAGE$GET)."""

        def _test() -> None:
            reg = registry.Registry(self.session)
            mv = reg.get_model(self._model_name).version(self._version_name)
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(Exception, r"Insufficient privileges"):
                    mv.export(tmpdir)

        self._run_as_role(self._usage_role, _test)

    def test_usage_cannot_load(self) -> None:
        """USAGE privilege cannot load model (requires READ for STAGE$GET)."""

        def _test() -> None:
            reg = registry.Registry(self.session)
            mv = reg.get_model(self._model_name).version(self._version_name)
            with self.assertRaisesRegex(Exception, r"Insufficient privileges"):
                mv.load(force=True)

        self._run_as_role(self._usage_role, _test)

    def test_read_can_export(self) -> None:
        """READ privilege allows export (STAGE$GET/LS -> READ)."""

        def _test() -> None:
            reg = registry.Registry(self.session)
            mv = reg.get_model(self._model_name).version(self._version_name)
            with tempfile.TemporaryDirectory() as tmpdir:
                mv.export(tmpdir)

        self._run_as_role(self._read_role, _test)


if __name__ == "__main__":
    absltest.main()
