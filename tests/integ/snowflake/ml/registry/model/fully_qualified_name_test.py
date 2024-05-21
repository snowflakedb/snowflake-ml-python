from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class FullyQualifiedNameTest(registry_model_test_base.RegistryModelTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.database_name = self._test_db
        self.schema_name = self._test_schema
        self._registry = self.registry
        self.registry = registry.Registry(self._session, database_name="foo", schema_name="bar")

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
