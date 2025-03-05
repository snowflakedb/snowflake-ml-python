import uuid
from typing import Callable, Tuple

from absl.testing import absltest
from packaging import version
from sklearn import datasets

from snowflake.ml._internal import env
from snowflake.ml.registry import Registry
from snowflake.snowpark import session
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


@absltest.skipIf(
    version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
    "Skip compat test for Python higher than 3.11 since we previously does not support it.",
)
class RegistrySklearnCompatTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _prepare_registry_and_log_model_fn_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str, str], None], Tuple[str, str, str]]:
        def prepare_registry_and_log_model(
            session: session.Session, test_db: str, test_schema: str, run_id: str
        ) -> None:
            from sklearn import datasets, linear_model

            from snowflake.ml.registry import Registry

            registry = Registry(session=session, database_name=test_db, schema_name=test_schema)

            iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
            # Normalize the column name to avoid set it as case_sensitive where there was a BCR in 1.1.2
            iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]
            # LogisticRegression is for classfication task, such as iris
            regr = linear_model.LogisticRegression()
            regr.fit(iris_X, iris_y)

            registry.log_model(
                model_name="model",
                version_name="v" + run_id,
                model=regr,
                sample_input_data=iris_X,
            )

        return prepare_registry_and_log_model, (self._test_db, self._test_schema, self._run_id)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_and_log_model_fn_factory,  # type: ignore[arg-type]
        version_range=">=1.2.0",
    )
    def test_log_model_compat(self) -> None:
        registry = Registry(session=self.session, database_name=self._test_db, schema_name=self._test_schema)
        model_ref = registry.get_model("model").version("v" + self._run_id)
        iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]
        try:
            model_ref.load().predict(iris_X)
        except ValueError:
            model_ref.load(force=True).predict(iris_X)


if __name__ == "__main__":
    absltest.main()
