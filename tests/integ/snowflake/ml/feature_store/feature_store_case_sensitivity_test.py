from typing import List, Tuple
from uuid import uuid4

from absl.testing import absltest, parameterized
from common_utils import (
    FS_INTEG_TEST_DATASET_SCHEMA,
    FS_INTEG_TEST_DB,
    create_random_schema,
    get_test_warehouse_name,
)

from snowflake.ml._internal.utils.identifier import resolve_identifier
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

# A list of names to be tested.
#   Each tuple is consisted of:
#       1. a list of equivalent names;
#       2. a list of looks similar but different names.
TEST_NAMES = [
    (
        ["QUICK_FOX", "QUICK_fox", "quick_fox", '"QUICK_FOX"'],
        ['"QUICK_fox"', '"quick_fox"'],
    ),
    (['"lazy_PIG"'], ["lazy_PIG", "LAZY_PIG", '"LAZY_PIG"']),
]

WAREHOUSE_NAMES = ["my_warehouse", '"my_""WAREHOUSE"']

FS_LOCATIONS = [
    ("fs_test_db", "fs_test_schema"),
    ('"Fs_Test""_DB"', "fs_test_schema"),
    ("fs_test_db", '"fs_TEST_""scheMA"'),
    ('"Fs_Test""_DB"', '"fs_TEST_""scheMA"'),
]


class FeatureStoreCaseSensitivityTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        cls._active_fs = []
        cls._mock_table = cls._create_mock_table("mock_data")
        cls._test_warehouse_name = get_test_warehouse_name(cls._session)

    @classmethod
    def tearDownClass(cls) -> None:
        for fs in cls._active_fs:
            fs.clear()
            cls._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
        cls._session.sql(f"DROP TABLE IF EXISTS {cls._mock_table}").collect()
        cls._session.close()

    @classmethod
    def _create_mock_table(cls, name: str) -> str:
        table_full_path = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.{name}_{uuid4().hex.upper()}"
        cls._session.sql(f"CREATE OR REPLACE TABLE {table_full_path} (a INT, b INT)").collect()
        cls._session.sql(
            f"""INSERT OVERWRITE INTO {table_full_path} (a, b)
                VALUES
                (1, 20),
                (2, 20)
            """
        ).collect()
        return table_full_path

    @parameterized.parameters(FS_LOCATIONS)  # type: ignore[misc]
    def test_feature_store_location(self, database: str, schema: str) -> None:
        self._session.sql(f"CREATE DATABASE IF NOT EXISTS {database}").collect()

        if schema.startswith('"') and schema.endswith('"'):
            schema = f'{schema[:-1]}_{ uuid4().hex.upper()}"'
        else:
            schema = f"{schema}_{uuid4().hex.upper()}"

        fs = FeatureStore(
            session=self._session,
            database=database,
            name=schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        df = self._session.sql(f"SELECT a, b FROM {self._mock_table}")
        e = Entity(name="e", join_keys=["a"])
        fs.register_entity(e)
        fv = FeatureView(name="fv", entities=[e], feature_df=df, refresh_freq="1 minute")
        fs.register_feature_view(feature_view=fv, version="v1", block=True)
        retrieved_fv = fs.get_feature_view(name="fv", version="v1")
        self.assertEqual(resolve_identifier(database), retrieved_fv.database)
        self.assertEqual(resolve_identifier(schema), retrieved_fv.schema)
        self.assertEqual(2, len(fs.read_feature_view(retrieved_fv).to_pandas()))

        fs = FeatureStore(
            session=self._session,
            database=database,
            name=schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )

    @parameterized.parameters(WAREHOUSE_NAMES)  # type: ignore[misc]
    def test_warehouse_names(self, warehouse: str) -> None:
        current_schema = create_random_schema(self._session, "TEST_WAREHOUSE_NAMES")

        self._session.sql(f"CREATE WAREHOUSE IF NOT EXISTS {warehouse} WITH WAREHOUSE_SIZE='XSMALL'").collect()

        fs = FeatureStore(
            session=self._session,
            database=FS_INTEG_TEST_DB,
            name=current_schema,
            default_warehouse=warehouse,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        df = self._session.sql(f"SELECT a, b FROM {self._mock_table}")
        e = Entity(name="e", join_keys=["a"])
        fs.register_entity(e)
        fv = FeatureView(name="fv", entities=[e], feature_df=df, refresh_freq="1 minute")
        fs.register_feature_view(feature_view=fv, version="v1", block=True)
        retrieved_fv = fs.get_feature_view(name="fv", version="v1")
        self.assertEqual(resolve_identifier(warehouse), retrieved_fv.warehouse)
        self.assertEqual(2, len(fs.read_feature_view(retrieved_fv).to_pandas()))

    # Covered APIs:
    #   1. FeatureStore
    #   2. delete_feature_store
    @parameterized.parameters(TEST_NAMES)  # type: ignore[misc]
    def test_feature_store_names(self, equi_names: List[str], diff_names: List[str]) -> None:
        per_run_id = uuid4().hex.upper()

        def generate_unique_name(names: List[str]) -> List[str]:
            result = []
            for name in names:
                if name.startswith('"') and name.endswith('"'):
                    result.append(f'"{name[1:-1]}_{per_run_id}"')
                else:
                    result.append(f"{name}_{per_run_id}")
            return result

        equi_names = generate_unique_name(equi_names)
        diff_names = generate_unique_name(diff_names)

        original_name = equi_names[0]
        self._session.sql(f"DROP SCHEMA IF EXISTS {FS_INTEG_TEST_DB}.{original_name}").collect()

        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            original_name,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        # get feature store instance with equivalent names without fail
        for equi_name in equi_names:
            FeatureStore(
                self._session,
                FS_INTEG_TEST_DB,
                equi_name,
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
            )

        # fail when trying to get with different names
        with self.assertRaisesRegex(ValueError, "Feature store .* does not exist."):
            for diff_name in diff_names:
                FeatureStore(
                    self._session,
                    FS_INTEG_TEST_DB,
                    diff_name,
                    default_warehouse=self._test_warehouse_name,
                    creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
                )

    # Covered APIs:
    #   1. Entity
    #   2. register_entity
    #   3. get_entity
    #   4. list_entities
    #   5. delete_entity
    @parameterized.parameters(TEST_NAMES)  # type: ignore[misc]
    def test_entity_names(self, equi_names: List[str], diff_names: List[str]) -> None:
        current_schema = create_random_schema(self._session, "TEST_ENTITY_NAMES")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        # register entity with original name
        original_name = equi_names[0]
        e_0 = Entity(name=original_name, join_keys=["a"])
        fs.register_entity(e_0)

        # register another entity with equivalent name will fail
        for equi_name in equi_names:
            e_1 = Entity(name=equi_name, join_keys=["a"])
            with self.assertRaisesRegex(ValueError, "Entity .* already exists."):
                fs.register_entity(e_1)

        # retrieve with equivalent name is fine.
        for equi_name in equi_names:
            fs.get_entity(equi_name)

        # delete with different names will fail
        for diff_name in diff_names:
            with self.assertRaisesRegex(ValueError, "Entity .* does not exist."):
                fs.delete_entity(diff_name)

        # register with different names is fine
        e_2 = Entity(name=diff_names[0], join_keys=["a"])
        fs.register_entity(e_2)

        # registered two entiteis.
        self.assertEqual(len(fs.list_entities().collect()), 2)

    # Covered APIs:
    #   1. register_entity
    #   2. get_entity
    #   3. delete_entity
    #   4. FeatureView.join_keys
    #   5. FeatureView.timestamp_col
    @parameterized.parameters(TEST_NAMES)  # type: ignore[misc]
    def test_join_keys_and_ts_col(self, equi_names: List[str], diff_names: List[str]) -> None:
        current_schema = create_random_schema(self._session, "TEST_JOIN_KEYS_AND_TS_COL")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        for test_name in equi_names:
            df = self._session.create_dataframe([1, 2, 3], schema=[test_name])
            e = Entity(name="MY_COOL_ENTITY", join_keys=[test_name])
            fv_0 = FeatureView(name="MY_FV", entities=[e], feature_df=df, timestamp_col=test_name)
            fs.register_entity(e)
            fv_1 = fs.register_feature_view(fv_0, "V1", block=True)

            retrieved_e = fs.get_entity("MY_COOL_ENTITY")
            self.assertEqual(len(retrieved_e.join_keys), 1)
            self.assertEqual(retrieved_e.join_keys[0], SqlIdentifier(test_name))

            self.assertEqual(len(fv_1.entities), 1)
            self.assertEqual(len(fv_1.entities[0].join_keys), 1)
            self.assertEqual(fv_1.entities[0].join_keys[0], SqlIdentifier(test_name))

            fv_2 = fs.get_feature_view("MY_FV", "V1")
            self.assertEqual(len(fv_2.entities), 1)
            self.assertEqual(len(fv_2.entities[0].join_keys), 1)
            self.assertEqual(fv_2.entities[0].join_keys[0], SqlIdentifier(test_name))
            self.assertEqual(fv_2.timestamp_col, SqlIdentifier(test_name))

            fs.delete_feature_view(fv_2)
            fs.delete_entity("MY_COOL_ENTITY")

    # Covered APIs:
    #   1. FeatureView
    #   2. register_feature_view
    #   3. read_feature_view
    #   4. list_feature_view
    #   5. get_feature_view
    @parameterized.parameters(
        [
            (
                [("foo", "bar"), ("foo", "BAR"), ("FOO", "BAR"), ('"FOO"', "BAR")],
                [('"foo"', "bar")],
            ),
            (
                [('"abc"', "def"), ('"abc"', "DEF")],
                [("abc", "def")],
            ),
        ]
    )  # type: ignore[misc]
    def test_feature_view_names_and_versions_combination(
        self,
        equi_full_names: List[Tuple[str, str]],
        diff_full_names: List[Tuple[str, str]],
    ) -> None:
        current_schema = create_random_schema(self._session, "TEST_FEATURE_VIEW_NAMES")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        df = self._session.create_dataframe([1, 2, 3], schema=["a"])
        e = Entity(name="my_cool_entity", join_keys=["a"])
        original_fv_name, original_version = equi_full_names[0]
        fv_0 = FeatureView(name=original_fv_name, entities=[e], feature_df=df)
        fs.register_entity(e)
        fs.register_feature_view(fv_0, original_version, block=True)

        # register with identical full name will fail
        for equi_full_name in equi_full_names:
            fv_name = equi_full_name[0]
            version = equi_full_name[1]
            with self.assertRaisesRegex(ValueError, "FeatureView .* already exists"):
                fv = FeatureView(name=fv_name, entities=[e], feature_df=df)
                fs.register_feature_view(fv, version, block=True)

        # register with different full name is fine
        for diff_full_name in diff_full_names:
            fv_name = diff_full_name[0]
            version = diff_full_name[1]
            fv = FeatureView(name=fv_name, entities=[e], feature_df=df)
            fv = fs.register_feature_view(fv, version, block=True)
            fs.read_feature_view(fv)

        self.assertEqual(len(fs.list_feature_views(as_dataframe=False)), len(diff_full_names) + 1)
        self.assertEqual(
            len(fs.list_feature_views(entity_name="my_cool_entity", as_dataframe=False)),
            len(diff_full_names) + 1,
        )
        self.assertGreaterEqual(
            len(
                fs.list_feature_views(
                    entity_name="my_cool_entity",
                    feature_view_name=original_fv_name,
                    as_dataframe=False,
                )
            ),
            1,
        )
        for diff_full_name in diff_full_names:
            fv_name = diff_full_name[0]
            self.assertGreaterEqual(
                len(
                    fs.list_feature_views(
                        entity_name="my_cool_entity",
                        feature_view_name=fv_name,
                        as_dataframe=False,
                    )
                ),
                1,
            )

        for equi_name in equi_full_names:
            fv_name = equi_name[0]
            version = equi_name[1]
            fs.get_feature_view(fv_name, version)

        for diff_name in diff_full_names:
            fv_name = diff_name[0]
            version = diff_name[1]
            fs.get_feature_view(fv_name, version)

    @parameterized.parameters(TEST_NAMES)  # type: ignore[misc]
    def test_find_objects(self, equi_names: List[str], diff_names: List[str]) -> None:
        current_schema = create_random_schema(self._session, "TEST_FIND_OBJECTS")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {FS_INTEG_TEST_DB}.{equi_names[0]}").collect()
        for name in equi_names:
            self.assertEqual(len(fs._find_object("SCHEMAS", SqlIdentifier(name))), 1)
        for name in diff_names:
            self.assertEqual(len(fs._find_object("SCHEMAS", SqlIdentifier(name))), 0)
        self._session.sql(f"DROP SCHEMA IF EXISTS {FS_INTEG_TEST_DB}.{equi_names[0]}").collect()

    def test_feature_view_version(self) -> None:
        current_schema = create_random_schema(self._session, "TEST_FEATURE_VIEW_VERSION")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_fs.append(fs)

        df = self._session.create_dataframe([1, 2, 3], schema=["a"])
        e = Entity(name="MY_COOL_ENTITY", join_keys=["a"])
        fs.register_entity(e)
        fv = FeatureView(name="MY_FV", entities=[e], feature_df=df)

        # 1: register with lowercase, get it back with lowercase/uppercase
        fs.register_feature_view(fv, "a1", block=True)
        fs.get_feature_view("MY_FV", "A1")
        fs.get_feature_view("MY_FV", "a1")

        # 2: register with uppercase, get it back with lowercase/uppercase
        fs.register_feature_view(fv, "B2", block=True)
        fs.get_feature_view("MY_FV", "b2")
        fs.get_feature_view("MY_FV", "B2")

        # 3. register with valid characters
        fs.register_feature_view(fv, "V2_1", block=True)
        fs.get_feature_view("MY_FV", "v2_1")
        fs.register_feature_view(fv, "3", block=True)
        fs.get_feature_view("MY_FV", "3")

        # 4: register with invalid characters
        with self.assertRaisesRegex(ValueError, ".* is not a valid feature view version.*"):
            fs.register_feature_view(fv, "abc$", block=True)
        with self.assertRaisesRegex(ValueError, ".* is not a valid feature view version.*"):
            fs.register_feature_view(fv, "abc#", block=True)
        with self.assertRaisesRegex(ValueError, ".* is not a valid feature view version.*"):
            fs.register_feature_view(fv, '"abc"', block=True)


if __name__ == "__main__":
    absltest.main()
