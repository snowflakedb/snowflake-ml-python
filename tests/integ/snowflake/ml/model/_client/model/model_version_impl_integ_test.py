import unittest
import uuid

from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    model_factory,
    test_env_utils,
)

MODEL_NAME = "TEST_MODEL"
VERSION_NAME = "V1"


class TestModelVersionImplInteg(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        login_options = connection_params.SnowflakeLoginOptions()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._session = Session.builder.configs(
            {
                **login_options,
                **{"database": self._test_db, "schema": self._test_schema},
            }
        ).create()

        current_sf_version = test_env_utils.get_current_snowflake_version(self._session)

        if current_sf_version < version.parse("8.0.0"):
            raise unittest.SkipTest("This test requires Snowflake Version 8.0.0 or higher.")

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self._session)

        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self._mv = self.registry.log_model(
            model=model,
            model_name=MODEL_NAME,
            version_name=VERSION_NAME,
            sample_input_data=test_features,
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_description(self) -> None:
        description = "test description"
        self._mv.description = description
        self.assertEqual(self._mv.description, description)

    def test_metrics(self) -> None:
        self._mv.set_metric("a", 1)
        expected_metrics = {"a": 2, "b": 1.0, "c": True}
        for k, v in expected_metrics.items():
            self._mv.set_metric(k, v)

        self.assertEqual(self._mv.get_metric("a"), expected_metrics["a"])
        self.assertDictEqual(self._mv.list_metrics(), expected_metrics)

        expected_metrics.pop("b")
        self._mv.delete_metric("b")
        self.assertDictEqual(self._mv.list_metrics(), expected_metrics)
        with self.assertRaises(KeyError):
            self._mv.get_metric("b")


if __name__ == "__main__":
    absltest.main()
