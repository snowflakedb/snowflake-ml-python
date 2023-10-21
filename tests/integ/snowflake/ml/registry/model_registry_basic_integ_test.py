import uuid
from typing import Optional

from absl.testing import absltest, parameterized

from snowflake.ml.registry import model_registry
from snowflake.ml.registry.artifact import Artifact, ArtifactType
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager

_RUN_ID = uuid.uuid4().hex
_PRE_CREATED_DB_NAME_UPPER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "REGISTRY_PRE_CREATED_DB_SYSTEM_UPPER"
).upper()
_PRE_CREATED_DB_NAME_LOWER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "registry_pre_created_db_system_lower"
).lower()
_PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "REGISTRY_PRE_CREATED_DB_AND_SCHEMA_UPPER"
).upper()
_PRE_CREATED_DB_AND_SCHEMA_NAME_LOWER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "registry_pre_created_db_and_schema_lower"
).lower()
_CUSTOM_NEW_DB_NAME_UPPER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "REGISTRY_NEW_DB_CUSTOM_UPPER"
).upper()
_CUSTOM_NEW_DB_NAME_LOWER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "registry_new_db_custom_lower"
).lower()
_CUSTOM_NEW_SCHEMA_NAME_UPPER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "REGISTRY_NEW_SCHEMA_CUSTOM_UPPER"
).upper()
_CUSTOM_NEW_SCHEMA_NAME_LOWER = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
    _RUN_ID, "registry_new_schema_custom_lower"
).lower()


class TestModelRegistryBasicInteg(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        cls._database = cls._session.get_current_database()
        cls._schema = cls._session.get_current_schema()
        assert cls._database is not None
        assert cls._schema is not None

        cls._db_manager = db_manager.DBManager(cls._session)

        cls._db_manager.cleanup_databases()

        cls._db_manager.create_database(_PRE_CREATED_DB_NAME_UPPER)
        cls._db_manager.create_database(_PRE_CREATED_DB_NAME_LOWER)
        cls._db_manager.create_schema(
            _PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER,
            _PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER,
        )
        cls._db_manager.create_schema(
            _PRE_CREATED_DB_AND_SCHEMA_NAME_LOWER,
            _PRE_CREATED_DB_AND_SCHEMA_NAME_LOWER,
        )

        # restore the session to use the original database and schema
        cls._session.use_database(cls._database)
        cls._session.use_schema(cls._schema)
        assert cls._database == cls._session.get_current_database()
        assert cls._schema == cls._session.get_current_schema()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._db_manager.drop_database(_PRE_CREATED_DB_NAME_UPPER, if_exists=True)
        cls._db_manager.drop_database(_PRE_CREATED_DB_NAME_LOWER, if_exists=True)
        cls._db_manager.drop_database(_PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER, if_exists=True)
        cls._db_manager.drop_database(_PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER, if_exists=True)
        cls._db_manager.drop_database(_CUSTOM_NEW_DB_NAME_UPPER, if_exists=True)
        cls._db_manager.drop_database(_CUSTOM_NEW_DB_NAME_LOWER, if_exists=True)
        cls._session.close()

    def _validate_restore_db_and_schema(self) -> None:
        """Validate that the database and schema are restored after creating registry."""
        self.assertEqual(self._database, self._session.get_current_database())
        self.assertEqual(self._schema, self._session.get_current_schema())

    @parameterized.parameters(  # type: ignore[misc]
        {"database_name": _PRE_CREATED_DB_NAME_UPPER, "schema_name": None},
        {
            "database_name": db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                _RUN_ID, "REGISTRY_NEW_DB_SYSTEM_UPPER"
            ).upper(),
            "schema_name": None,
        },
        {
            "database_name": _CUSTOM_NEW_DB_NAME_UPPER,
            "schema_name": _CUSTOM_NEW_SCHEMA_NAME_UPPER,
        },
        {"database_name": _PRE_CREATED_DB_NAME_LOWER, "schema_name": None},
        {
            "database_name": db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                _RUN_ID, "registry_new_db_system_lower"
            ).lower(),
            "schema_name": None,
        },
        {
            "database_name": _CUSTOM_NEW_DB_NAME_LOWER,
            "schema_name": _CUSTOM_NEW_SCHEMA_NAME_LOWER,
        },
        {
            "database_name": db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                _RUN_ID, 'registry_new_db_system_with""'
            ).lower(),
            "schema_name": None,
        },
        {
            "database_name": _PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER,
            "schema_name": _CUSTOM_NEW_SCHEMA_NAME_UPPER,
        },
        {
            "database_name": _PRE_CREATED_DB_AND_SCHEMA_NAME_LOWER,
            "schema_name": _CUSTOM_NEW_SCHEMA_NAME_LOWER,
        },
        {
            "database_name": _PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER,
            "schema_name": _PRE_CREATED_DB_AND_SCHEMA_NAME_UPPER,
        },
        {
            "database_name": _PRE_CREATED_DB_AND_SCHEMA_NAME_LOWER,
            "schema_name": _PRE_CREATED_DB_AND_SCHEMA_NAME_LOWER,
        },
    )
    def test_create_and_drop_model_registry(self, database_name: str, schema_name: Optional[str] = None) -> None:
        if schema_name:
            create_result = model_registry.create_model_registry(
                session=self._session, database_name=database_name, schema_name=schema_name
            )
            self.assertTrue(create_result)
            self._validate_restore_db_and_schema()

            # Test create again, should be non-op
            create_result = model_registry.create_model_registry(
                session=self._session, database_name=database_name, schema_name=schema_name
            )

            self.assertTrue(create_result)
            self._validate_restore_db_and_schema()

            _ = model_registry.ModelRegistry(
                session=self._session, database_name=database_name, schema_name=schema_name
            )

            self._db_manager.drop_schema(schema_name, database_name)
            self.assertTrue(self._db_manager.assert_schema_existence(schema_name, database_name, exists=False))
            self._validate_restore_db_and_schema()
        else:
            create_result = model_registry.create_model_registry(session=self._session, database_name=database_name)
            self.assertTrue(create_result)
            self._validate_restore_db_and_schema()

            # Test create again, should be non-op
            create_result = model_registry.create_model_registry(session=self._session, database_name=database_name)
            self.assertTrue(create_result)
            self._validate_restore_db_and_schema()

            _ = model_registry.ModelRegistry(session=self._session, database_name=database_name)

            self._db_manager.drop_database(database_name)
            self.assertTrue(self._db_manager.assert_database_existence(database_name, exists=False))
            self._validate_restore_db_and_schema()

    def test_add_and_delete_ml_artifacts(self) -> None:
        """Test add() and delete() in `_artifact_manager.py` works as expected."""

        artifact_registry = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            _RUN_ID, "artifact_registry"
        ).upper()
        artifact_registry_schema = "PUBLIC"

        try:
            model_registry.create_model_registry(
                session=self._session, database_name=artifact_registry, schema_name=artifact_registry_schema
            )
            registry = model_registry.ModelRegistry(
                session=self._session, database_name=artifact_registry, schema_name=artifact_registry_schema
            )
        except Exception as e:
            self._db_manager.drop_database(artifact_registry)
            raise Exception(f"Test failed with exception:{e}")

        artifact_id = "test_art_123"
        artifact_version = "test_artifact_version"
        artifact_name = "test_artifact"
        artifact = Artifact(type=ArtifactType.DATASET, spec='{"test_property": "test_value"}')

        try:
            art_ref = registry._artifact_manager.add(
                artifact=artifact,
                artifact_id=artifact_id,
                artifact_name=artifact_name,
                artifact_version=artifact_version,
            )

            self.assertTrue(
                registry._artifact_manager.exists(
                    art_ref.name,
                    art_ref.version,
                )
            )

            # Validate the artifact_spec can be parsed as expected
            retrieved_art_df = registry._artifact_manager.get(
                art_ref.name,
                art_ref.version,
            )

            actual_artifact_spec = retrieved_art_df.collect()[0]["ARTIFACT_SPEC"]
            self.assertEqual(artifact._spec, actual_artifact_spec)

            # Validate that `delete_artifact` can remove entries from the artifact table.
            registry._artifact_manager.delete(
                art_ref.name,
                art_ref.version,
            )
            self.assertFalse(
                registry._artifact_manager.exists(
                    art_ref.name,
                    art_ref.version,
                )
            )
        finally:
            self._db_manager.drop_database(artifact_registry, if_exists=True)


if __name__ == "__main__":
    absltest.main()
