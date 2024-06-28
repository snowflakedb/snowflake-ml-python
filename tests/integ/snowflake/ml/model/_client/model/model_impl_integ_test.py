import unittest
import uuid

from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml._internal.utils import identifier
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

        self._tag_name1 = "MYTAG"
        self._tag_name2 = '"live_version"'

        self._session.sql(f"CREATE TAG {self._tag_name1}").collect()
        self._session.sql(f"CREATE TAG {self._tag_name2}").collect()

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_versions(self) -> None:
        self.assertEqual(self._model.versions(), [self._mv, self._mv2])
        self.assertLen(self._model.show_versions(), 2)

    def test_description(self) -> None:
        description = "test description"
        self._model.description = description
        self.assertEqual(self._model.description, description)

    def test_default(self) -> None:
        self.assertEqual(self._model.default.version_name, VERSION_NAME)

        self._model.default = VERSION_NAME2
        self.assertEqual(self._model.default.version_name, VERSION_NAME2)

    @unittest.skipUnless(
        test_env_utils.get_current_snowflake_version() >= version.parse("8.7.0"),
        "Drop version on model only available when the Snowflake Version is newer than 8.7.0",
    )
    def test_delete_version(self) -> None:
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self.registry.log_model(
            model=model,
            model_name=MODEL_NAME,
            version_name="V3",
            sample_input_data=test_features,
        )
        self._model.delete_version("V3")
        self.assertLen(self._model.show_versions(), 2)

    def test_tag(self) -> None:
        fq_tag_name1 = identifier.get_schema_level_object_identifier(self._test_db, self._test_schema, self._tag_name1)
        fq_tag_name2 = identifier.get_schema_level_object_identifier(self._test_db, self._test_schema, self._tag_name2)
        self.assertDictEqual({}, self._model.show_tags())
        self.assertIsNone(self._model.get_tag(self._tag_name1))
        self._model.set_tag(self._tag_name1, "val1")
        self.assertEqual(
            "val1",
            self._model.get_tag(fq_tag_name1),
        )
        self.assertDictEqual(
            {fq_tag_name1: "val1"},
            self._model.show_tags(),
        )
        self._model.set_tag(fq_tag_name2, "v2")
        self.assertEqual("v2", self._model.get_tag(self._tag_name2))
        self.assertDictEqual(
            {
                fq_tag_name1: "val1",
                fq_tag_name2: "v2",
            },
            self._model.show_tags(),
        )
        self._model.unset_tag(fq_tag_name2)
        self.assertDictEqual(
            {fq_tag_name1: "val1"},
            self._model.show_tags(),
        )
        self._model.unset_tag(self._tag_name1)
        self.assertDictEqual({}, self._model.show_tags())

    def test_rename(self) -> None:
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self.registry.log_model(
            model=model,
            model_name="MODEL",
            version_name="V1",
            sample_input_data=test_features,
        )
        model = self.registry.get_model(model_name="MODEL")
        model.rename("MODEL2")
        self.assertEqual(model.name, "MODEL2")
        self.registry.delete_model("MODEL2")

    def test_rename_fully_qualified_name(self) -> None:
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self.registry.log_model(
            model=model,
            model_name="MODEL",
            version_name="V1",
            sample_input_data=test_features,
        )
        model = self.registry.get_model(model_name="MODEL")
        model.rename(f"{self._test_db}.{self._test_schema}.MODEL2")
        self.assertEqual(model.name, "MODEL2")
        self.registry.delete_model("MODEL2")

    def test_system_aliases(self) -> None:
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self.registry.log_model(
            model=model,
            model_name="MODEL_ALIAS",
            version_name=VERSION_NAME,
            sample_input_data=test_features,
        )
        self.registry.log_model(
            model=model,
            model_name="MODEL_ALIAS",
            version_name=VERSION_NAME2,
            sample_input_data=test_features,
        )
        model = self.registry.get_model(model_name="MODEL_ALIAS")
        self.assertEqual(model.first().version_name, VERSION_NAME)
        self.assertEqual(model.last().version_name, VERSION_NAME2)
        self.assertEqual(model.default.version_name, VERSION_NAME)


if __name__ == "__main__":
    absltest.main()
