from inspect import isclass
from typing import Any, Callable, Dict, Optional, Type, Union
from uuid import uuid4

from absl.testing import absltest, parameterized
from access_utils import FeatureStoreRole as Role, configure_roles
from common_utils import (
    FS_INTEG_TEST_DB,
    cleanup_temporary_objects,
    create_mock_table,
    create_random_schema,
    get_test_warehouse_name,
)

from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewStatus
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session, exceptions as snowpark_exceptions

_TEST_ROLE_ADMIN = "FS_ROLE_ADMIN"
_TEST_ROLE_PRODUCER = "FS_ROLE_PRODUCER"
_TEST_ROLE_CONSUMER = "FS_ROLE_CONSUMER"
_TEST_ROLE_NONE = "FS_ROLE_NONE"


class FeatureStoreAccessTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        cleanup_temporary_objects(cls._session)
        cls._test_roles = {
            Role.ADMIN: _TEST_ROLE_ADMIN,
            Role.PRODUCER: _TEST_ROLE_PRODUCER,
            Role.CONSUMER: _TEST_ROLE_CONSUMER,
            Role.NONE: _TEST_ROLE_NONE,
        }
        cls._test_warehouse = get_test_warehouse_name(cls._session)
        cls._session.use_warehouse(cls._test_warehouse)
        cls._test_database = FS_INTEG_TEST_DB
        cls._test_admin = cls._session.get_current_role()

        try:
            cls._test_schema = create_random_schema(
                cls._session, "FS_TEST", database=cls._test_database, additional_options="WITH MANAGED ACCESS"
            )
            cls._feature_store = FeatureStore(
                cls._session,
                cls._test_database,
                cls._test_schema,
                cls._test_warehouse,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )

            configure_roles(
                cls._feature_store,
                admin_role_name=cls._test_roles[Role.ADMIN],
                producer_role_name=cls._test_roles[Role.PRODUCER],
                consumer_role_name=cls._test_roles[Role.CONSUMER],
            )

            cls._mock_table = cls._init_test_data()
            for role_id in cls._test_roles.values():
                # Grant read access to mock source data table
                cls._session.sql(f"GRANT SELECT ON TABLE {cls._mock_table} to role {role_id}").collect()

        except Exception as e:
            cls.tearDownClass()
            raise Exception(f"Test setup failed: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.use_role(cls._test_admin)
        cls._session.sql(f"DROP SCHEMA IF EXISTS {cls._test_database}.{cls._test_schema}").collect()
        cls._session.close()

    def setUp(self) -> None:
        self._session.use_role(self._test_admin)

    @classmethod
    def _init_test_data(cls) -> str:
        prev_role = cls._session.get_current_role()
        try:
            cls._session.use_role(cls._test_roles[Role.ADMIN])
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
        expected_result: Optional[Union[Type[Exception], Callable[[Any], Optional[bool]], Any]] = None,
        expected_access_exception: Type[Exception] = RuntimeError,
        access_exception_dict: Optional[Dict[Role, Type[Exception]]] = None,
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
        try:
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
            self._session.use_role(prev_role)

    @parameterized.product(
        [
            {
                "init_args": {"creation_mode": CreationMode.CREATE_IF_NOT_EXIST},
                "required_access": Role.ADMIN,
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
        init_args: Dict[str, Any],
        required_access: Role,
        test_access: Role,
        expected_result: Optional[Type[Exception]],
    ) -> None:
        schema_name = f"FS_TEST_{uuid4().hex.upper()}"

        def unit_under_test() -> FeatureStore:
            return FeatureStore(
                self._session,
                self._test_database,
                schema_name,
                self._test_warehouse,
                **init_args,
            )

        try:
            self._test_access(
                unit_under_test,
                required_access,
                test_access,
                expected_result=expected_result,
                access_exception_dict={Role.NONE: ValueError},
            )
        finally:
            self._session.sql(f"DROP SCHEMA IF EXISTS {self._test_database}.{schema_name}").collect()

    @parameterized.product(required_access=[Role.ADMIN], test_access=list(Role))  # type: ignore[misc]
    def test_clear(self, required_access: Role, test_access: Role) -> None:
        # Create isolated Feature Store to test clearing
        schema_admin = self._session.get_current_role()
        schema = create_random_schema(
            self._session, "FS_TEST", database=self._test_database, additional_options="WITH MANAGED ACCESS"
        )
        try:
            fs = FeatureStore(
                self._session,
                self._test_database,
                schema,
                self._test_warehouse,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )
            configure_roles(
                fs,
                admin_role_name=self._test_roles[Role.ADMIN],
                producer_role_name=self._test_roles[Role.PRODUCER],
                consumer_role_name=self._test_roles[Role.CONSUMER],
            )

            self._session.use_role(self._test_roles[Role.ADMIN])
            e = Entity(f"test_entity_{uuid4().hex.upper()}"[:32], ["test_key"])
            fs.register_entity(e)

            entity_count = len(fs.list_entities().collect())
            self.assertGreater(entity_count, 0)
            self._test_access(
                fs.clear,
                required_access,
                test_access,
            )

            # Do validation on FileSet contents outside _test_access since we need admin access
            expected_entity_count = entity_count if test_access.value < Role.ADMIN.value else 0
            self.assertEqual(len(fs.list_entities().collect()), expected_entity_count)
        finally:
            self._session.use_role(schema_admin)
            self._session.sql(f"DROP SCHEMA IF EXISTS {self._test_database}.{schema}").collect()

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
    def test_register_entity(self, required_access: Role, test_access: Role) -> None:
        e = Entity(f"test_entity_{uuid4().hex.upper()}"[:32], ["id"])

        self._test_access(
            lambda: self._feature_store.register_entity(e),
            required_access,
            test_access,
            lambda _: self.assertIn(e.name, [r["NAME"] for r in self._feature_store.list_entities().collect()]),
        )

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
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

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
    def test_suspend_feature_view(self, required_access: Role, test_access: Role) -> None:
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

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
    def test_resume_feature_view(self, required_access: Role, test_access: Role) -> None:
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
            ),
        finally:
            self._feature_store.delete_feature_view(fv)

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
    def test_generate_dataset(self, required_access: Role, test_access: Role) -> None:
        spine_df = self._session.sql(f"SELECT id FROM {self._mock_table}")
        fv1 = self._feature_store.get_feature_view("fv1", "v1")
        fv2 = self._feature_store.get_feature_view("fv2", "v1")
        dataset_name = f"FS_TEST_DATASET_{uuid4().hex.upper()}"

        self._test_access(
            lambda: self._feature_store.generate_dataset(spine_df, [fv1, fv2], materialized_table=dataset_name),
            required_access,
            test_access,
            access_exception_dict={Role.NONE: snowpark_exceptions.SnowparkSQLException},
        )

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
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

    @parameterized.product(required_access=[Role.PRODUCER], test_access=list(Role))  # type: ignore[misc]
    def test_delete_entity(self, required_access: Role, test_access: Role) -> None:
        e = Entity(f"test_entity_{uuid4().hex.upper()}"[:32], ["test_key"])

        self._session.use_role(self._test_roles[Role.PRODUCER])
        self._feature_store.register_entity(e)

        self._test_access(
            lambda: self._feature_store.delete_entity(e.name),
            required_access,
            test_access,
        )

    @parameterized.product(required_access=[Role.CONSUMER], test_access=list(Role))  # type: ignore[misc]
    def test_list_entities(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            self._feature_store.list_entities,
            required_access,
            test_access,
            expected_result=lambda rst: self.assertGreater(len(rst.collect()), 0),
            access_exception_dict={Role.NONE: snowpark_exceptions.SnowparkSQLException},
        )

    @parameterized.product(required_access=[Role.CONSUMER], test_access=list(Role))  # type: ignore[misc]
    def test_get_entity(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            lambda: self._feature_store.get_entity("foo"),
            required_access,
            test_access,
            expected_result=lambda rst: self.assertIsInstance(rst, Entity),
        )

    @parameterized.product(required_access=[Role.CONSUMER], test_access=list(Role))  # type: ignore[misc]
    def test_list_feature_views(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            lambda: self._feature_store.list_feature_views(as_dataframe=False),
            required_access,
            test_access,
            expected_result=lambda rst: self.assertGreater(len(rst), 0),
            access_exception_dict={Role.NONE: snowpark_exceptions.SnowparkSQLException},
        )

    @parameterized.product(required_access=[Role.CONSUMER], test_access=list(Role))  # type: ignore[misc]
    def test_get_feature_view(self, required_access: Role, test_access: Role) -> None:
        self._test_access(
            lambda: self._feature_store.get_feature_view("fv1", "v1"),
            required_access,
            test_access,
            expected_result=lambda rst: self.assertIsInstance(rst, FeatureView),
        )

    @parameterized.product(required_access=[Role.CONSUMER], test_access=list(Role))  # type: ignore[misc]
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


if __name__ == "__main__":
    absltest.main()
