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
VERSION_NAME2 = "V2"


class TestModelImplInteg(parameterized.TestCase):
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
        self._mv2 = self.registry.log_model(
            model=model,
            model_name=MODEL_NAME,
            version_name=VERSION_NAME2,
            sample_input_data=test_features,
        )
        self._model = self.registry.get_model(model_name=MODEL_NAME)

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_description(self) -> None:
        description = "test description"
        self._model.description = description
        self.assertEqual(self._model.description, description)

    def test_default(self) -> None:
        self.assertEqual(self._model.default.version_name, VERSION_NAME)

        self._model.default = VERSION_NAME2
        self.assertEqual(self._model.default.version_name, VERSION_NAME2)


if __name__ == "__main__":
    absltest.main()
