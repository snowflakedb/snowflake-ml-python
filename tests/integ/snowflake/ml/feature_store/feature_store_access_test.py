from inspect import isclass
from typing import Any, Callable, Optional, Union
from uuid import uuid4

from absl.testing import absltest, parameterized
from common_utils import create_mock_table, get_test_warehouse_name

from snowflake.ml.feature_store.access_manager import (
    _configure_pre_init_privileges,
    _FeatureStoreRole as Role,
    _SessionInfo,
    setup_feature_store,
)
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewStatus
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session, exceptions as snowpark_exceptions
from snowflake.snowpark.exceptions import SnowparkSQLException
from tests.integ.snowflake.ml.test_utils import db_manager


class FeatureStoreAccessTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Create shared resources for RBAC testing (runs once for all tests)."""
        cls._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        cls._dbm = db_manager.DBManager(cls._session)

        # Clean up stale resources from previous failed test runs (safety net)
        cls._dbm.cleanup_databases(expire_hours=6)
        cls._dbm.cleanup_warehouses(expire_hours=6)
        cls._dbm.cleanup_roles(expire_hours=6)

        # Create test-specific roles with unique names
        run_id = uuid4().hex[:6]
        cls._producer_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            run_id, "FS_PRODUCER"
        ).upper()
        cls._consumer_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            run_id, "FS_CONSUMER"
        ).upper()
        cls._none_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "FS_NONE").upper()

        cls._test_roles = {
            Role.PRODUCER: cls._producer_role,
            Role.CONSUMER: cls._consumer_role,
            Role.NONE: cls._none_role,
        }
        cls._test_warehouse = get_test_warehouse_name(cls._session)
        cls._session.use_warehouse(cls._test_warehouse)

        # Create test-specific database
        run_id = uuid4().hex[:6]
        cls._test_database = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "FS_DB").upper()
        cls._dbm.create_database(cls._test_database, data_retention_time_in_days=1)
        cls._dbm.use_database(cls._test_database)

        cls._test_admin = cls._session.get_current_role()

        # Generate a unique schema name
        cls._test_schema = "FS_TEST_" + uuid4().hex.upper()

        # Pre-create schema WITH MANAGED ACCESS
        # This prevents setup_feature_store from transferring ownership to PRODUCER
        cls._session.sql(
            f"CREATE SCHEMA IF NOT EXISTS {cls._test_database}.{cls._test_schema} WITH MANAGED ACCESS"
        ).collect()

        # Call setup_feature_store to create roles and configure RBAC
        # This creates PRODUCER and CONSUMER roles and sets up the role hierarchy
        cls._feature_store = setup_feature_store(
            cls._session,
            cls._test_database,
            cls._test_schema,
            cls._test_warehouse,
            producer_role=cls._test_roles[Role.PRODUCER],
            consumer_role=cls._test_roles[Role.CONSUMER],
        )

        # Create the NONE role (not created by setup_feature_store)
        cls._dbm.create_role(cls._none_role)

        # Grant all test roles to current user so we can switch to them
        current_user = cls._session.get_current_user().strip('"')
        for role_name in cls._test_roles.values():
            cls._session.sql(f"GRANT ROLE {role_name} TO USER {current_user}").collect()

        # Build hierarchy for NONE role: NONE -> CONSUMER (CONSUMER -> PRODUCER already done by setup_feature_store)
        cls._session.sql(f"GRANT ROLE {cls._none_role} TO ROLE {cls._consumer_role}").collect()

        # Create test data
        cls._mock_table = cls._init_test_data()
        for role_id in cls._test_roles.values():
            # Grant read access to mock source data table
            cls._session.sql(f"GRANT SELECT ON TABLE {cls._mock_table} to role {role_id}").collect()

    def setUp(self) -> None:
        """Reset to admin role before each test."""
        self._session.use_role(self._test_admin)

    @classmethod
    def _cleanup_role_completely(cls, role_name: str) -> None:
        """Deterministically clean up a role by revoking all grants and dependencies."""
        # 1. Revoke all future grants on the database for this role
        try:
            cls._session.sql(
                f"REVOKE ALL PRIVILEGES ON FUTURE SCHEMAS IN DATABASE {cls._test_database} FROM ROLE {role_name}"
            ).collect()
        except Exception:
            pass

        try:
            cls._session.sql(
                f"REVOKE ALL PRIVILEGES ON FUTURE TABLES IN DATABASE {cls._test_database} FROM ROLE {role_name}"
            ).collect()
        except Exception:
            pass

        try:
            cls._session.sql(
                f"REVOKE ALL PRIVILEGES ON FUTURE VIEWS IN DATABASE {cls._test_database} FROM ROLE {role_name}"
            ).collect()
        except Exception:
            pass

        # 2. Show and revoke all grants TO this role
        try:
            grants = cls._session.sql(f"SHOW GRANTS TO ROLE {role_name}").collect()
            for grant in grants:
                try:
                    privilege = grant["privilege"]
                    granted_on = grant["granted_on"]
                    name = grant["name"]

                    # Skip role grants, we'll handle those separately
                    if granted_on == "ROLE":
                        continue

                    # Revoke the privilege
                    cls._session.sql(f"REVOKE {privilege} ON {granted_on} {name} FROM ROLE {role_name}").collect()
                except Exception:
                    pass
        except Exception:
            pass

        # 3. Show and revoke all grants OF this role (role hierarchy)
        try:
            grants = cls._session.sql(f"SHOW GRANTS OF ROLE {role_name}").collect()
            for grant in grants:
                try:
                    grantee_name = grant["grantee_name"]
                    granted_to = grant["granted_to"]
                    if granted_to == "ROLE":
                        cls._session.sql(f"REVOKE ROLE {role_name} FROM ROLE {grantee_name}").collect()
                    elif granted_to == "USER":
                        cls._session.sql(f"REVOKE ROLE {role_name} FROM USER {grantee_name}").collect()
                except Exception:
                    pass
        except Exception:
            pass

        # 4. Now drop the role
        cls._dbm.drop_role(role_name, if_exists=True)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up shared resources after all tests complete."""
        cls._session.use_role(cls._test_admin)

        # Drop the test database - this removes most object-level grants
        try:
            cls._dbm.drop_database(cls._test_database, if_exists=True)
        except Exception:
            # Try to drop schema first if database drop fails
            try:
                cls._session.sql(f"DROP SCHEMA IF EXISTS {cls._test_database}.{cls._test_schema}").collect()
            except Exception:
                pass

        # Deterministically clean up each role in reverse order of hierarchy
        cls._cleanup_role_completely(cls._none_role)
        cls._cleanup_role_completely(cls._consumer_role)
        cls._cleanup_role_completely(cls._producer_role)

        cls._session.close()

    @classmethod
    def _init_test_data(cls) -> str:
        """Initialize test data: entity, feature views, and mock table."""
        prev_role = cls._session.get_current_role()
        try:
            cls._session.use_role(cls._test_roles[Role.PRODUCER])
            test_table: str = create_mock_table(cls._session, cls._test_database, cls._test_schema)

            # Create Entities
            e = Entity("foo", ["id"])
            cls._feature_store.register_entity(e)

            fv1 = FeatureView(
                name="fv1",
                entities=[e],
                feature_df=cls._session.sql(f"SELECT id, name, ts FROM {test_table}"),
                timestamp_col="ts",
                refresh_freq="DOWNSTREAM",
            )
            fv1 = cls._feature_store.register_feature_view(feature_view=fv1, version="v1")

            fv2 = FeatureView(
                name="fv2",
                entities=[e],
                feature_df=cls._session.sql(f"SELECT id, title, ts FROM {test_table}"),
                timestamp_col="ts",
                refresh_freq="DOWNSTREAM",
            )
            fv2 = cls._feature_store.register_feature_view(feature_view=fv2, version="v1")

            return test_table

        finally:
            cls._session.use_role(prev_role)

    def _test_access(
        self,
        method: Callable[[], Any],
        required_access: Role,
        test_access: Role,
        expected_result: Optional[Union[type[Exception], Callable[[Any], Optional[bool]], Any]] = None,
        expected_access_exception: type[Exception] = RuntimeError,
        access_exception_dict: Optional[dict[Role, type[Exception]]] = None,
    ) -> Any:
        """
        Test a Feature Store API given a specified access level.

        Args:
            method: Parameterless callable wrapping method under test
            required_access: Expected minimum access needed to execute method
            test_access: Access level to execute test under
            expected_result: Expected outcome of method call with sufficient access.
                May be an exception if method is expected to throw, a constant value,
                or a callback with custom assertions and/or returns True on acceptance.
            expected_access_exception: Expected exception on insufficient access.
            access_exception_dict: Level-specific expected exceptions. Takes precedence
                over expected_access_exception for matching access levels.
        """
        prev_role = self._session.get_current_role()
        # Get current secondary roles setting to restore later
        try:
            secondary_roles_result = self._session.sql(
                "SHOW PARAMETERS LIKE 'USE_SECONDARY_ROLES' IN SESSION"
            ).collect()
            prev_secondary_roles = secondary_roles_result[0]["value"] if secondary_roles_result else "ALL"
        except Exception:
            prev_secondary_roles = "ALL"  # Default to ALL if we can't query
        try:
            # Disable secondary roles to ensure strict RBAC testing
            # Without this, the session would have combined privileges from all roles granted to the user
            self._session.sql("USE SECONDARY ROLES NONE").collect()
            self._session.use_role(self._test_roles[test_access])

            if test_access.value < required_access.value:
                # Access level specific exception types
                if isinstance(access_exception_dict, dict) and test_access in access_exception_dict:
                    expected_access_exception = access_exception_dict[test_access]

                # TODO: Error pattern
                with self.assertRaises(expected_access_exception):
                    return method()
            elif isclass(expected_result) and issubclass(expected_result, Exception):
                # TODO: Error pattern
                with self.assertRaises(expected_result):
                    return method()
            else:
                result = method()
                if expected_result is not None:
                    if callable(expected_result):
                        # TODO: Use original (admin) role to execute validator?
                        validate_result = expected_result(result)
                        self.assertTrue(validate_result is None or validate_result is True)
                    else:
                        self.assertEqual(expected_result, result)
                return result
        finally:
            # Restore previous role and secondary roles setting
            self._session.use_role(prev_role)
            if prev_secondary_roles.upper() == "NONE":
                self._session.sql("USE SECONDARY ROLES NONE").collect()
            elif prev_secondary_roles.upper() == "ALL":
                self._session.sql("USE SECONDARY ROLES ALL").collect()
            else:
                # Restore to default if it was something else
                self._session.sql("USE SECONDARY ROLES ALL").collect()

    @parameterized.product(
        [
            {
                "init_args": {"creation_mode": CreationMode.CREATE_IF_NOT_EXIST},
                "required_access": Role.PRODUCER,
                "expected_result": None,
            },
            {
                "init_args": {"creation_mode": CreationMode.FAIL_IF_NOT_EXIST},
                "required_access": Role.CONSUMER,
                "expected_result": ValueError,
            },
        ],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_init(
        self,
        init_args: dict[str, Any],
        required_access: Role,
        test_access: Role,
        expected_result: Optional[type[Exception]],
    ) -> None:
        # Generate unique schema name
        schema = "FS_TEST_" + uuid4().hex.upper()

        # Create schema with MANAGED ACCESS
        self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._test_database}.{schema} WITH MANAGED ACCESS").collect()

        # Transfer ownership to PRODUCER role
        self._session.sql(
            f"GRANT OWNERSHIP ON SCHEMA {self._test_database}.{schema} " f"TO ROLE {self._test_roles[Role.PRODUCER]}"
        ).collect()

        # Grant privileges (schema already exists, so ownership won't be transferred again)
        _configure_pre_init_privileges(
            self._session,
            _SessionInfo(self._test_database, schema, self._test_warehouse),
            roles_to_create={
                Role.PRODUCER: self._test_roles[Role.PRODUCER],
                Role.CONSUMER: self._test_roles[Role.CONSUMER],
            },
        )

        def unit_under_test() -> FeatureStore:
            return FeatureStore(
                self._session,
                self._test_database,
                schema,
                default_warehouse=self._test_warehouse,
                **init_args,
            )

        try:
            self._test_access(
                unit_under_test,
                required_access,
                test_access,
                expected_result=expected_result,
                access_exception_dict={Role.NONE: ValueError},  # NONE role can't access warehouse/database
            )
        finally:
            self._session.sql(f"DROP SCHEMA IF EXISTS {self._test_database}.{schema}").collect()

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_clear(self, required_access: Role, test_access: Role) -> None:
        # Create isolated Feature Store to test clearing
        schema_admin = self._session.get_current_role()

        # Generate unique schema name
        schema = "FS_TEST_" + uuid4().hex.upper()

        # Create schema with MANAGED ACCESS
        self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._test_database}.{schema} WITH MANAGED ACCESS").collect()

        # Transfer ownership to PRODUCER role
        self._session.sql(
            f"GRANT OWNERSHIP ON SCHEMA {self._test_database}.{schema} " f"TO ROLE {self._test_roles[Role.PRODUCER]}"
        ).collect()

        try:
            # Setup feature store (schema already exists, so ownership won't be transferred again)
            fs = setup_feature_store(
                self._session,
                self._test_database,
                schema,
                self._test_warehouse,
                producer_role=self._test_roles[Role.PRODUCER],
                consumer_role=self._test_roles[Role.CONSUMER],
            )

            self._session.use_role(self._test_roles[Role.PRODUCER])
            e = Entity(f"test_entity_{uuid4().hex.upper()}"[:32], ["test_key"])
            fs.register_entity(e)

            entity_count = len(fs.list_entities().collect())
            self.assertGreater(entity_count, 0)
            self._test_access(
                lambda: fs._clear(dryrun=False),
                required_access,
                test_access,
            )

            # Do validation on FileSet contents outside _test_access since we need admin access
            expected_entity_count = entity_count if test_access.value < Role.PRODUCER.value else 0
            self.assertEqual(len(fs.list_entities().collect()), expected_entity_count)
        finally:
            self._session.use_role(schema_admin)
            self._session.sql(f"DROP SCHEMA IF EXISTS {self._test_database}.{schema}").collect()

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_register_entity(self, required_access: Role, test_access: Role) -> None:
        e = Entity(f"test_entity_{uuid4().hex.upper()}"[:32], ["id"])

        self._test_access(
            lambda: self._feature_store.register_entity(e),
            required_access,
            test_access,
            lambda _: self.assertIn(e.name, [r["NAME"] for r in self._feature_store.list_entities().collect()]),
        )

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_register_feature_view(self, required_access: Role, test_access: Role) -> None:
        e = self._feature_store.get_entity("foo")
        fv = FeatureView(
            name=f"test_fv_{uuid4().hex.upper()}"[:32],
            entities=[e],
            feature_df=self._session.sql(f"SELECT id, name, ts FROM {self._mock_table}"),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )

        fv = self._test_access(
            lambda: self._feature_store.register_feature_view(fv, "test"),
            required_access,
            test_access,
            expected_result=lambda _fv: self.assertIn(
                _fv.status, (FeatureViewStatus.RUNNING, FeatureViewStatus.ACTIVE)
            ),
        )

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_suspend_feature_view(self, required_access: Role, test_access: Role) -> None:
        self._session.use_role(self._test_roles[Role.PRODUCER])  # Expected case is FeatureView owned by PRODUCER
        e = self._feature_store.get_entity("foo")
        fv = FeatureView(
            name="test_fv",
            entities=[e],
            feature_df=self._session.sql(f"SELECT id, name, ts FROM {self._mock_table}"),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv = self._feature_store.register_feature_view(fv, "test", overwrite=True)

        try:
            self._test_access(
                lambda: self._feature_store.suspend_feature_view(fv),
                required_access,
                test_access,
                lambda _fv: self.assertEqual(FeatureViewStatus.SUSPENDED, _fv.status),
            ),
        finally:
            self._feature_store.delete_feature_view(fv)

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_resume_feature_view(self, required_access: Role, test_access: Role) -> None:
        self._session.use_role(self._test_roles[Role.PRODUCER])  # Expected case is FeatureView owned by PRODUCER
        e = self._feature_store.get_entity("foo")
        fv = FeatureView(
            name="test_fv",
            entities=[e],
            feature_df=self._session.sql(f"SELECT id, name, ts FROM {self._mock_table}"),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv = self._feature_store.register_feature_view(fv, "test", overwrite=True)
        fv = self._feature_store.suspend_feature_view(fv)

        try:
            self._test_access(
                lambda: self._feature_store.resume_feature_view(fv),
                required_access,
                test_access,
                expected_result=lambda _fv: self.assertIn(
                    _fv.status, (FeatureViewStatus.RUNNING, FeatureViewStatus.ACTIVE)
                ),
            )
        finally:
            self._feature_store.delete_feature_view(fv)

    @parameterized.product(
        required_access=[Role.CONSUMER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_generate_training_set_ephemeral(self, required_access: Role, test_access: Role) -> None:
        spine_df = self._session.sql(f"SELECT id FROM {self._mock_table}")
        fv1 = self._feature_store.get_feature_view("fv1", "v1")
        fv2 = self._feature_store.get_feature_view("fv2", "v1")

        self._test_access(
            lambda: self._feature_store.generate_training_set(spine_df, [fv1, fv2]),
            required_access,
            test_access,
            access_exception_dict={Role.NONE: SnowparkSQLException},  # NONE role can't access database
        )

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_generate_training_set_material(self, required_access: Role, test_access: Role) -> None:
        spine_df = self._session.sql(f"SELECT id FROM {self._mock_table}")
        fv1 = self._feature_store.get_feature_view("fv1", "v1")
        fv2 = self._feature_store.get_feature_view("fv2", "v1")
        training_set_name = f"FS_TEST_TRAINING_SET_{uuid4().hex.upper()}"

        self._test_access(
            lambda: self._feature_store.generate_training_set(spine_df, [fv1, fv2], save_as=training_set_name),
            required_access,
            test_access,
            access_exception_dict={Role.NONE: SnowparkSQLException},  # NONE role can't access database
        )

    @parameterized.product(  # type: ignore[misc]
        required_access=[Role.PRODUCER],
        test_access=list(Role),
        output_type=["dataset", "table"],
    )
    def test_generate_dataset(self, required_access: Role, test_access: Role, output_type: str) -> None:
        spine_df = self._session.sql(f"SELECT id FROM {self._mock_table}")
        fv1 = self._feature_store.get_feature_view("fv1", "v1")
        fv2 = self._feature_store.get_feature_view("fv2", "v1")
        dataset_name = f"FS_TEST_DATASET_{uuid4().hex.upper()}"

        self._test_access(
            lambda: self._feature_store.generate_dataset(dataset_name, spine_df, [fv1, fv2], output_type=output_type),
            required_access,
            test_access,
            access_exception_dict={Role.NONE: SnowparkSQLException},  # NONE role can't access database
        )

    @parameterized.product(  # type: ignore[misc]
        required_access=[Role.CONSUMER],
        test_access=list(Role),
        output_type=["dataset", "table"],
    )
    def test_access_dataset(self, required_access: Role, test_access: Role, output_type: str) -> None:
        spine_df = self._session.sql(f"SELECT id FROM {self._mock_table}")
        fv1 = self._feature_store.get_feature_view("fv1", "v1")
        fv2 = self._feature_store.get_feature_view("fv2", "v1")
        dataset_name = f"FS_TEST_DATASET_{uuid4().hex.upper()}"
        dataset = self._feature_store.generate_dataset(dataset_name, spine_df, [fv1, fv2], output_type=output_type)

        dataframe = dataset.read.to_snowpark_dataframe() if output_type == "dataset" else dataset
        self._test_access(
            dataframe.collect,
            required_access,
            test_access,
            expected_result=lambda _pd: self.assertNotEmpty(_pd),
            access_exception_dict={Role.NONE: SnowparkSQLException},  # NONE role can't access database
        )

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_delete_feature_view(self, required_access: Role, test_access: Role) -> None:
        e = self._feature_store.get_entity("foo")
        fv = FeatureView(
            name="test_fv",
            entities=[e],
            feature_df=self._session.sql(f"SELECT id, name, ts FROM {self._mock_table}"),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )

        self._session.use_role(self._test_roles[Role.PRODUCER])
        fv = self._feature_store.register_feature_view(fv, "test", overwrite=True)

        try:
            self._test_access(
                lambda: self._feature_store.delete_feature_view(fv),
                required_access,
                test_access,
                expected_access_exception=snowpark_exceptions.SnowparkSQLException,
                access_exception_dict={Role.NONE: snowpark_exceptions.SnowparkSQLException},
            )
        finally:
            self._feature_store.delete_feature_view(fv)

    @parameterized.product(
        required_access=[Role.PRODUCER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_delete_entity(self, required_access: Role, test_access: Role) -> None:
        e = Entity(f"test_entity_{uuid4().hex.upper()}"[:32], ["test_key"])

        self._session.use_role(self._test_roles[Role.PRODUCER])
        self._feature_store.register_entity(e)

        self._test_access(
            lambda: self._feature_store.delete_entity(e.name),
            required_access,
            test_access,
        )

    @parameterized.product(
        required_access=[Role.CONSUMER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_list_entities(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            self._feature_store.list_entities,
            required_access,
            test_access,
            expected_result=lambda rst: self.assertGreater(len(rst.collect()), 0),
            access_exception_dict={Role.NONE: snowpark_exceptions.SnowparkSQLException},
        )

    @parameterized.product(
        required_access=[Role.CONSUMER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_get_entity(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            lambda: self._feature_store.get_entity("foo"),
            required_access,
            test_access,
            expected_result=lambda rst: self.assertIsInstance(rst, Entity),
        )

    @parameterized.product(
        required_access=[Role.CONSUMER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_list_feature_views(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            lambda: self._feature_store.list_feature_views().collect(),
            required_access,
            test_access,
            expected_result=lambda rst: self.assertGreater(len(rst), 0),
        )

    @parameterized.product(
        required_access=[Role.CONSUMER],
        test_access=[Role.PRODUCER, Role.CONSUMER],
    )  # type: ignore[misc]
    def test_get_feature_view(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            lambda: self._feature_store.get_feature_view("fv1", "v1"),
            required_access,
            test_access,
            expected_result=lambda rst: self.assertIsInstance(rst, FeatureView),
        )

    @parameterized.product(
        required_access=[Role.CONSUMER],
        test_access=list(Role),
    )  # type: ignore[misc]
    def test_retrieve_feature_values(self, required_access: Role, test_access: Role) -> None:
        spine_df = self._session.sql(f"SELECT id FROM {self._mock_table}")
        fv1 = self._feature_store.get_feature_view("fv1", "v1")
        fv2 = self._feature_store.get_feature_view("fv2", "v1")

        self._test_access(
            lambda: self._feature_store.retrieve_feature_values(spine_df, [fv1, fv2]),
            required_access,
            test_access,
            access_exception_dict={Role.NONE: snowpark_exceptions.SnowparkSQLException},
        )

    def test_producer_setup(self) -> None:
        # Generate unique schema name
        schema = "FS_TEST_" + uuid4().hex.upper()

        # Create schema with MANAGED ACCESS
        self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._test_database}.{schema} WITH MANAGED ACCESS").collect()

        # Transfer ownership to PRODUCER role
        self._session.sql(
            f"GRANT OWNERSHIP ON SCHEMA {self._test_database}.{schema} " f"TO ROLE {self._test_roles[Role.PRODUCER]}"
        ).collect()

        # Setup feature store (schema already exists, so ownership won't be transferred again)
        fs = setup_feature_store(
            self._session,
            self._test_database,
            schema,
            self._test_warehouse,
            producer_role=self._test_roles[Role.PRODUCER],
        )
        self.assertTrue(fs is not None)


if __name__ == "__main__":
    absltest.main()
