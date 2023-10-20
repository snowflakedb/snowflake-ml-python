import uuid
from typing import Any, Dict

from absl.testing import absltest

from snowflake.ml._internal.utils import identifier
from snowflake.ml.registry import (
    _initial_schema,
    _schema,
    _schema_upgrade_plans,
    _schema_version_manager,
    model_registry,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    model_factory,
    test_env_utils,
)


class UpgradePlan_0(_schema_upgrade_plans.BaseSchemaUpgradePlans):
    def __init__(
        self,
        session: Session,
        database_name: str,
        schema_name: str,
        statement_params: Dict[str, Any],
    ) -> None:
        super().__init__(session, database_name, schema_name, statement_params)

    def upgrade(self) -> None:
        self._session.sql(
            f"""ALTER TABLE {self._database}.{self._schema}._SYSTEM_REGISTRY_MODELS
                RENAME COLUMN CREATION_CONTEXT TO CREATION_CONTEXT_ABC
            """
        ).collect()


class UpgradePlan_1(_schema_upgrade_plans.BaseSchemaUpgradePlans):
    def __init__(
        self,
        session: Session,
        database_name: str,
        schema_name: str,
        statement_params: Dict[str, Any],
    ) -> None:
        super().__init__(session, database_name, schema_name, statement_params)

    def upgrade(self) -> None:
        self._session.sql(
            f"""ALTER TABLE {self._database}.{self._schema}._SYSTEM_REGISTRY_MODELS
                RENAME COLUMN CREATION_CONTEXT_ABC TO CREATION_CONTEXT
            """
        ).collect()


class ModelRegistrySchemaEvolutionIntegTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls.session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        cls.run_id = uuid.uuid4().hex
        cls.active_dbs = []
        cls.db_manager = db_manager.DBManager(cls.session)

    @classmethod
    def tearDownClass(cls) -> None:
        for db in cls.active_dbs:
            cls.db_manager.drop_database(db)
        cls.session.close()

    def setUp(self) -> None:
        self.original_registry_schema = _schema._REGISTRY_TABLE_SCHEMA.copy()
        self.original_schema_version = _schema._CURRENT_SCHEMA_VERSION
        self.original_schema_upgrade_plans = _schema._SCHEMA_UPGRADE_PLANS.copy()

    def tearDown(self) -> None:
        _schema._CURRENT_SCHEMA_VERSION = self.original_schema_version
        _schema._SCHEMA_UPGRADE_PLANS = self.original_schema_upgrade_plans
        _schema._REGISTRY_TABLE_SCHEMA = self.original_registry_schema
        _schema._CURRENT_TABLE_SCHEMAS[_initial_schema._MODELS_TABLE_NAME] = _schema._REGISTRY_TABLE_SCHEMA

    def _check_version_table_exist(self, registry_name: str, schema_name: str) -> bool:
        result = self.session.sql(
            f"""SHOW TABLES LIKE '{_schema_version_manager._SCHEMA_VERSION_TABLE_NAME}'
                IN "{registry_name}"."{schema_name}"
            """
        ).collect()
        return len(result) == 1

    def _get_schema_version(self, registry_name: str, schema_name: str) -> int:
        infer_db_name = identifier.get_inferred_name(registry_name)
        infer_schema_name = identifier.get_inferred_name(schema_name)
        full_table_name = f"{infer_db_name}.{infer_schema_name}.{_schema_version_manager._SCHEMA_VERSION_TABLE_NAME}"
        result = self.session.sql(f"SELECT MAX(VERSION) AS MAX_VERSION FROM {full_table_name}").collect()
        return result[0]["MAX_VERSION"]

    def _update_package_schema(
        self, new_version: int, plan: _schema_upgrade_plans.BaseSchemaUpgradePlans, from_col: str, to_col: str
    ):
        _schema._CURRENT_SCHEMA_VERSION = new_version
        _schema._SCHEMA_UPGRADE_PLANS[new_version] = plan  # type: ignore[assignment]
        for i, (col_name, _) in enumerate(_schema._REGISTRY_TABLE_SCHEMA):
            if col_name == from_col:
                _schema._REGISTRY_TABLE_SCHEMA[i] = (to_col, "VARCHAR")
        _schema._CURRENT_TABLE_SCHEMAS[_initial_schema._MODELS_TABLE_NAME] = _schema._REGISTRY_TABLE_SCHEMA

    def test_svm_upgrade_deployed_schema(self) -> None:
        registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "test_svm_upgrade_deployed_schema"
        )
        schema_name = "SVM_TEST_SCHEMA"
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)
        self.active_dbs.append(registry_name)

        # No schema upgrade. svm.try_upgrade() is no-op
        svm = _schema_version_manager.SchemaVersionManager(
            self.session, identifier.get_inferred_name(registry_name), identifier.get_inferred_name(schema_name)
        )

        cur_version = _schema._CURRENT_SCHEMA_VERSION
        self.assertEqual(svm.get_deployed_version(), cur_version)
        svm.validate_schema_version()
        svm.try_upgrade()

        # first upgrade. rename column from "CREATION_CONTEXT" to "CREATION_CONTEXT_ABC"
        self._update_package_schema(cur_version + 1, UpgradePlan_0, "CREATION_CONTEXT", "CREATION_CONTEXT_ABC")

        self.assertEqual(svm.get_deployed_version(), cur_version)
        with self.assertRaisesRegex(RuntimeError, "Registry schema version .* is ahead of deployed"):
            svm.validate_schema_version()
        svm.try_upgrade()
        self.assertEqual(svm.get_deployed_version(), cur_version + 1)
        svm.validate_schema_version()
        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertTrue("CREATION_CONTEXT_ABC" in df.columns)

        # Second upgrade schema: rename column "CREATION_CONTEXT_ABC" back to "CREATION_CONTEXT"
        self._update_package_schema(cur_version + 2, UpgradePlan_1, "CREATION_CONTEXT_ABC", "CREATION_CONTEXT")

        self.assertEqual(svm.get_deployed_version(), cur_version + 1)
        with self.assertRaisesRegex(RuntimeError, "Registry schema version .* is ahead of deployed"):
            svm.validate_schema_version()
        svm.try_upgrade()
        self.assertEqual(svm.get_deployed_version(), cur_version + 2)
        svm.validate_schema_version()
        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertTrue("CREATION_CONTEXT" in df.columns)

    def test_svm_upgrade_package_schema(self) -> None:
        registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "test_svm_upgrade_package_schema"
        )
        schema_name = "SVM_TEST_SCHEMA"
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)
        self.active_dbs.append(registry_name)

        # Upgrade deployed schema to a newer version
        cur_version = _schema._CURRENT_SCHEMA_VERSION
        self._update_package_schema(cur_version + 1, UpgradePlan_0, "CREATION_CONTEXT", "CREATION_CONTEXT_ABC")

        svm = _schema_version_manager.SchemaVersionManager(
            self.session, identifier.get_inferred_name(registry_name), identifier.get_inferred_name(schema_name)
        )
        svm.try_upgrade()
        self.assertEqual(svm.get_deployed_version(), cur_version + 1)
        svm.validate_schema_version()

        # Then downgrade package schema and check version should fail.
        _schema._CURRENT_SCHEMA_VERSION = cur_version
        with self.assertRaisesRegex(RuntimeError, "Deployed registry schema version .* is ahead of current package"):
            svm.validate_schema_version()

    def test_model_registry_upgrade_deployed_schema(self) -> None:
        registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "test_model_registry_upgrade_deployed_schema"
        )
        schema_name = "SVM_TEST_SCHEMA"

        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)
        model_registry.ModelRegistry(session=self.session, database_name=registry_name, schema_name=schema_name)
        self.active_dbs.append(registry_name)

        # Upgrade schema: rename column "CREATION_CONTEXT" to "CREATION_CONTEXT_ABC"
        cur_version = _schema._CURRENT_SCHEMA_VERSION
        self._update_package_schema(cur_version + 1, UpgradePlan_0, "CREATION_CONTEXT", "CREATION_CONTEXT_ABC")

        # model registry will create version table, and update deployed schema to version 1.
        with self.assertRaisesRegex(RuntimeError, "Registry schema version .* is ahead of deployed"):
            model_registry.ModelRegistry(session=self.session, database_name=registry_name, schema_name=schema_name)
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)

        self.assertTrue(self._check_version_table_exist(registry_name, schema_name))
        self.assertEqual(self._get_schema_version(registry_name, schema_name), cur_version + 1)
        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertTrue("CREATION_CONTEXT_ABC" in df.columns)
        self.assertFalse("CREATION_CONTEXT" in df.columns)

        # second upgrade: rename column "CREATION_CONTEXT_ABC" to "CREATION_CONTEXT"
        self._update_package_schema(cur_version + 2, UpgradePlan_1, "CREATION_CONTEXT_ABC", "CREATION_CONTEXT")

        # model registry will update deployed schema to version 2.
        with self.assertRaisesRegex(RuntimeError, "Registry schema version .* is ahead of deployed"):
            model_registry.ModelRegistry(session=self.session, database_name=registry_name, schema_name=schema_name)
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)

        self.assertTrue(self._check_version_table_exist(registry_name, schema_name))
        self.assertEqual(self._get_schema_version(registry_name, schema_name), cur_version + 2)
        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertFalse("CREATION_CONTEXT_ABC" in df.columns)
        self.assertTrue("CREATION_CONTEXT" in df.columns)

    def test_model_registry_upgrade_package_schema(self) -> None:
        registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "test_model_registry_upgrade_package_schema"
        )
        schema_name = "SVM_TEST_SCHEMA"
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)
        self.active_dbs.append(registry_name)

        # Upgrade deployed schema to a newer version
        cur_version = _schema._CURRENT_SCHEMA_VERSION
        self._update_package_schema(cur_version + 1, UpgradePlan_0, "CREATION_CONTEXT", "CREATION_CONTEXT_ABC")

        svm = _schema_version_manager.SchemaVersionManager(
            self.session, identifier.get_inferred_name(registry_name), identifier.get_inferred_name(schema_name)
        )

        svm.try_upgrade()
        self.assertEqual(svm.get_deployed_version(), cur_version + 1)
        svm.validate_schema_version()

        # Then downgrade package schema, and ModelRegistry will panic
        _schema._CURRENT_SCHEMA_VERSION = cur_version
        with self.assertRaisesRegex(RuntimeError, "Deployed registry schema version .* is ahead of current package"):
            model_registry.ModelRegistry(session=self.session, database_name=registry_name, schema_name=schema_name)

        with self.assertRaisesRegex(RuntimeError, "Deployed registry schema version .* is ahead of current package"):
            model_registry.create_model_registry(
                session=self.session, database_name=registry_name, schema_name=schema_name
            )

    def test_model_registry_creation_has_artifact_ids(self) -> None:
        registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "test_model_registry_creation_has_artifact_ids"
        )
        schema_name = "SVM_TEST_SCHEMA"
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)
        self.active_dbs.append(registry_name)

        model_registry.ModelRegistry(session=self.session, database_name=registry_name, schema_name=schema_name)

        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertTrue("ARTIFACT_IDS" in df.columns)
        self.assertTrue(self._get_schema_version(registry_name, schema_name) > 0)

        # downgrade deployed schema and delete training dataset id
        self.session.sql(
            f"""ALTER TABLE "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS
                DROP COLUMN ARTIFACT_IDS
            """
        ).collect()
        self.session.sql(
            f"""DROP TABLE "{registry_name}"."{schema_name}".{_schema_version_manager._SCHEMA_VERSION_TABLE_NAME}
            """
        ).collect()
        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertFalse("ARTIFACT_IDS" in df.columns)
        self.assertFalse(self._check_version_table_exist(registry_name, schema_name))

        # opening model registry will raise, re-create model registry will upgrade deployed schema
        with self.assertRaisesRegex(RuntimeError, "Registry schema version .* is ahead of deployed"):
            model_registry.ModelRegistry(session=self.session, database_name=registry_name, schema_name=schema_name)
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)

        df = self.session.sql(f"""SELECT * FROM "{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_MODELS""")
        self.assertTrue("ARTIFACT_IDS" in df.columns)
        self.assertTrue(self._get_schema_version(registry_name, schema_name) > 0)

    def test_api_schema_validation(self) -> None:
        registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "test_api_schema_validation"
        )
        schema_name = "SVM_TEST_SCHEMA"
        model_registry.create_model_registry(session=self.session, database_name=registry_name, schema_name=schema_name)
        self.active_dbs.append(registry_name)

        registry = model_registry.ModelRegistry(
            session=self.session, database_name=registry_name, schema_name=schema_name
        )
        model, test_features, _ = model_factory.ModelFactory.prepare_snowml_model_xgb()
        registry.log_model(
            model_name="m",
            model_version="v1",
            model=model,
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
            ],
        )

        # upgrade deployed schema
        version_table_path = f'"{registry_name}"."{schema_name}"._SYSTEM_REGISTRY_SCHEMA_VERSION'
        self.session.sql(
            f"""INSERT INTO {version_table_path} (VERSION, CREATION_TIME)
                VALUES ({_schema._CURRENT_SCHEMA_VERSION + 1}, CURRENT_TIMESTAMP())
            """
        ).collect()

        with self.assertRaisesRegex(RuntimeError, "Deployed registry schema version .* is ahead of current package"):
            registry.log_model(
                model_name="m",
                model_version="v2",
                model=model,
                conda_dependencies=[
                    test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
                ],
            )

        model_ref = model_registry.ModelReference(registry=registry, model_name="m", model_version="v1")

        with self.assertRaisesRegex(RuntimeError, "Deployed registry schema version .* is ahead of current package"):
            model_ref.deploy(  # type: ignore[attr-defined]
                deployment_name="test_api_schema_validation",
                target_method="predict",
                permanent=False,
            )

        with self.assertRaisesRegex(RuntimeError, "Deployed registry schema version .* is ahead of current package"):
            model_ref.predict("test_api_schema_validation", test_features)


if __name__ == "__main__":
    absltest.main()
