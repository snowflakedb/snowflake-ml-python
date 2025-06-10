import uuid

from absl.testing import absltest

from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


class RegistryInitializationTest(common_test_base.CommonTestBase):
    def setUp(self):
        super().setUp()
        self._run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.cleanup_databases(expire_hours=6)
        self._created_dbs = []

    def tearDown(self) -> None:
        for db in self._created_dbs:
            self._db_manager.drop_database(db)
        super().tearDown()

    def _create_db_and_schema(self, db_name, schema_name, quoted_db=False, quoted_schema=False):
        db = f'"{db_name}"' if quoted_db else db_name
        schema = f'"{schema_name}"' if quoted_schema else schema_name
        self._db_manager.create_database(db)
        self._db_manager.create_schema(schema)
        self._created_dbs.append(db)

    def test_registry_initialization_variants(self):
        run_id = self._run_id

        db_upper = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "DBUPPER").upper()
        schema_upper = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "SCHEMAUPPER").upper()
        self._create_db_and_schema(db_upper, schema_upper)
        reg = registry.Registry(self.session, database_name=db_upper, schema_name=schema_upper)
        self.assertIsInstance(reg, registry.Registry)

        db_mixed = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "DbMixed")
        schema_mixed = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "SchemaMixed")
        self._create_db_and_schema(db_mixed, schema_mixed)
        reg = registry.Registry(self.session, database_name=f'"{db_mixed}"', schema_name=f'"{schema_mixed}"')
        self.assertIsInstance(reg, registry.Registry)

        db_lower_quoted = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "dblower")
        schema_lower_quoted = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "schemalower")
        self._create_db_and_schema(db_lower_quoted, schema_lower_quoted, quoted_db=True, quoted_schema=True)
        reg = registry.Registry(
            self.session, database_name=f'"""{db_lower_quoted}"""', schema_name=f'"""{schema_lower_quoted}"""'
        )
        self.assertIsInstance(reg, registry.Registry)

        db_upper_quoted = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "DBUPPERQ")
        schema_upper_quoted = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "SCHEMAUPPERQ")
        self._create_db_and_schema(db_upper_quoted, schema_upper_quoted, quoted_db=True, quoted_schema=True)
        reg = registry.Registry(
            self.session, database_name=f'"""{db_upper_quoted}"""', schema_name=f'"""{schema_upper_quoted}"""'
        )
        self.assertIsInstance(reg, registry.Registry)

        db_mixed_quoted = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "DbMixedQ")
        schema_mixed_quoted = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "SchemaMixedQ")
        self._create_db_and_schema(db_mixed_quoted, schema_mixed_quoted, quoted_db=True, quoted_schema=True)
        reg = registry.Registry(
            self.session, database_name=f'"""{db_mixed_quoted}"""', schema_name=f'"""{schema_mixed_quoted}"""'
        )
        self.assertIsInstance(reg, registry.Registry)

    def test_registry_with_invalid_db_or_schema(self):
        with self.assertRaises(ValueError) as cm:
            registry.Registry(self.session, database_name="foo", schema_name="bar")
        self.assertIn("does not exist", str(cm.exception))


if __name__ == "__main__":
    absltest.main()
