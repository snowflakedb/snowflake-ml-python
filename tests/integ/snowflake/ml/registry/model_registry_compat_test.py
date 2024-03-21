import unittest
import uuid
from typing import Callable, Tuple

from absl.testing import absltest, parameterized
from packaging import version
from sklearn import datasets

from snowflake.ml._internal import env
from snowflake.ml.registry import model_registry
from snowflake.snowpark import session
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


@unittest.skipIf(
    version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
    "Skip compat test for Python higher than 3.11 since we previously does not support it.",
)
class ModelRegistryCompatTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self.session)
        self.current_db = self.session.get_current_database()
        self.current_schema = self.session.get_current_schema()
        self.registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "registry_db")

    def tearDown(self) -> None:
        self._db_manager.drop_database(self.registry_name, if_exists=True)
        self.session.use_database(self.current_db)
        self.session.use_schema(self.current_schema)
        super().tearDown()

    def _prepare_registry_fn_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str], None], Tuple[str]]:
        def prepare_registry(session: session.Session, registry_name: str) -> None:
            from snowflake.connector.errors import ProgrammingError
            from snowflake.ml.registry import model_registry

            try:
                model_registry.create_model_registry(session=session, database_name=registry_name)
            except ProgrammingError:
                # Previous versions of library will call use even in the sproc env, which is not allowed.
                # This is to suppress the error
                pass

        return prepare_registry, (self.registry_name,)

    # Starting from 1.0.1 as we had a breaking change at that time.
    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_fn_factory, version_range=">=1.0.1"  # type: ignore[misc]
    )
    def test_open_registry_compat(self) -> None:
        model_registry.ModelRegistry(session=self.session, database_name=self.registry_name, create_if_not_exists=True)

    def _prepare_registry_and_log_model_fn_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        self.registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "registry_db")

        def prepare_registry_and_log_model(session: session.Session, registry_name: str, run_id: str) -> None:
            from sklearn import datasets, linear_model

            from snowflake.connector.errors import ProgrammingError
            from snowflake.ml.registry import model_registry

            try:
                model_registry.create_model_registry(session=session, database_name=registry_name)
            except ProgrammingError:
                # Previous versions of library will call use even in the sproc env, which is not allowed.
                # This is to suppress the error
                pass

            registry = model_registry.ModelRegistry(session=session, database_name=registry_name)

            iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
            # Normalize the column name to avoid set it as case_sensitive where there was a BCR in 1.1.2
            iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]
            # LogisticRegression is for classfication task, such as iris
            regr = linear_model.LogisticRegression()
            regr.fit(iris_X, iris_y)

            registry.log_model(
                model_name="model",
                model_version=run_id,
                model=regr,
                sample_input_data=iris_X,
            )

        return prepare_registry_and_log_model, (self.registry_name, self.run_id)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_and_log_model_fn_factory,  # type: ignore[arg-type]
        version_range=">=1.0.6",
    )
    @parameterized.parameters({"permanent": True})
    def test_log_model_compat(self, permanent: bool) -> None:
        registry = model_registry.ModelRegistry(
            session=self.session, database_name=self.registry_name, create_if_not_exists=True
        )
        model_ref = model_registry.ModelReference(
            registry=registry,
            model_name="model",
            model_version=self.run_id,
        )
        deployment_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict")
        model_ref.deploy(  # type: ignore[attr-defined]
            deployment_name=deployment_name, target_method="predict", permanent=permanent
        )
        iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]
        model_ref.predict(deployment_name, iris_X)


if __name__ == "__main__":
    absltest.main()
