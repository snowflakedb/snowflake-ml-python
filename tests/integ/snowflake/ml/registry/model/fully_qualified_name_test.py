import uuid

from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import db_manager


class FullyQualifiedNameTest(registry_model_test_base.RegistryModelTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.database_name = self._test_db
        self.schema_name = self._test_schema
        self._registry = self.registry
        self._run_id = uuid.uuid4().hex
        self._non_model_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "foo").upper()
        self._non_model_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "bar"
        ).upper()
        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._non_model_db)
        self._db_manager.create_schema(self._non_model_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(
            self.session, database_name=self._non_model_db, schema_name=self._non_model_schema
        )

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._non_model_db)
        super().tearDown()

    def test_random_version_name(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        name = f"{self.database_name}.{self.schema_name}.model_{self._run_id}"
        mv = self.registry.log_model(
            model=regr,
            model_name=name,
            version_name="V1",
            sample_input_data=iris_X,
        )

        mv.run(iris_X, function_name="predict")

        m = self.registry.get_model(name)

        self.assertLen(self._registry.models(), 1)

        m.show_versions()

        self.assertLen(m.versions(), 1)

        m.comment = "This is a comment"
        self.assertEqual(m.comment, "This is a comment")

        m.description = "This is a description"
        self.assertEqual(m.description, "This is a description")

        self.assertEqual(m.default, m.version("V1"))
        m.default.run(iris_X, function_name="predict")

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self._registry.models()])


if __name__ == "__main__":
    absltest.main()
