import datetime
from typing import List, Optional, Tuple, Union, cast
from uuid import uuid4

import pandas as pd
from absl.testing import absltest, parameterized
from common_utils import (
    FS_INTEG_TEST_DATASET_SCHEMA,
    FS_INTEG_TEST_DB,
    FS_INTEG_TEST_DUMMY_DB,
    cleanup_temporary_objects,
    compare_dataframe,
    create_mock_session,
    create_random_schema,
    get_test_warehouse_name,
)
from snowflake.ml.version import VERSION

from snowflake.ml import dataset
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import (
    _FEATURE_STORE_OBJECT_TAG,
    CreationMode,
    FeatureStore,
)
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session, exceptions as snowpark_exceptions
from snowflake.snowpark.functions import call_udf, col, udf


class FeatureStoreTest(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self._session_config = SnowflakeLoginOptions()
        self._session = Session.builder.configs(self._session_config).create()
        self._active_feature_store = []

        try:
            self._session.sql(f"CREATE DATABASE IF NOT EXISTS {FS_INTEG_TEST_DUMMY_DB}").collect()
            self._session.sql(f"CREATE DATABASE IF NOT EXISTS {FS_INTEG_TEST_DB}").collect()
            cleanup_temporary_objects(self._session)
            self._session.sql(
                f"CREATE SCHEMA IF NOT EXISTS {FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}"
            ).collect()
            self._test_warehouse_name = get_test_warehouse_name(self._session)
            self._mock_table = self._create_mock_table("customers")
        except Exception as e:
            self.tearDownClass()
            raise Exception(f"Test setup failed: {e}")

    @classmethod
    def tearDownClass(self) -> None:
        for fs in self._active_feature_store:
            try:
                fs._clear(dryrun=False)
            except Exception as e:
                if "Intentional Integ Test Error" not in str(e):
                    raise Exception(f"Unexpected exception happens when clear: {e}")
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()

        self._session.sql(f"DROP TABLE IF EXISTS {self._mock_table}").collect()
        self._session.close()

    @classmethod
    def _create_mock_table(self, name: str) -> str:
        table_full_path = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.{name}_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (name VARCHAR(64), id INT, title VARCHAR(128), age INT, dept VARCHAR(64), ts INT)
            """
        ).collect()
        self._session.sql(
            f"""INSERT OVERWRITE INTO {table_full_path} (name, id, title, age, dept, ts)
                VALUES
                ('jonh', 1, 'boss', 20, 'sales', 100),
                ('porter', 2, 'manager', 30, 'engineer', 200)
            """
        ).collect()
        return table_full_path

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        current_schema = create_random_schema(self._session, "FS_TEST") if name is None else name
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        # Intentionally point session to a different database to make sure feature store code is resilient to
        # session location.
        self._session.use_database(FS_INTEG_TEST_DUMMY_DB)
        return fs

    def _check_tag_value(
        self, fs: FeatureStore, object_name: str, object_domain: str, tag_name: str, expected_value: str
    ) -> None:
        query = f"""
            SELECT
                TAG_VALUE
            FROM TABLE(
                {fs._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES(
                    '{object_name}',
                    '{object_domain}'
                )
            )
            WHERE TAG_NAME = '{tag_name}'
        """
        res = self._session.sql(query).collect()
        self.assertEqual(expected_value, res[0]["TAG_VALUE"])

    def test_fail_if_not_exist(self) -> None:
        name = f"foo_{uuid4().hex.upper()}"
        with self.assertRaisesRegex(ValueError, "Feature store schema .* does not exist."):
            FeatureStore(
                session=self._session,
                database=FS_INTEG_TEST_DB,
                name=name,
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
            )

        self._session.sql(f"create schema {FS_INTEG_TEST_DB}.{name}").collect()

        with self.assertRaisesRegex(ValueError, "Feature store internal tag .* does not exist."):
            fs = FeatureStore(
                session=self._session,
                database=FS_INTEG_TEST_DB,
                name=name,
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
            )

        self._create_feature_store(name)

        fs = FeatureStore(
            session=self._session,
            database=FS_INTEG_TEST_DB,
            name=name,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )
        self.assertIsNotNone(fs)

    def test_invalid_warehouse(self) -> None:
        schema_name = f"TEST_INVALID_WAREHOUSE_{uuid4().hex.upper()}"
        with self.assertRaisesRegex(ValueError, "Cannot find warehouse.*"):
            FeatureStore(
                session=self._session,
                database=FS_INTEG_TEST_DB,
                name=schema_name,
                default_warehouse=schema_name,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )
        # No schema should be created if failure happens in the ctor
        res = self._session.sql(f"SHOW SCHEMAS LIKE '{schema_name}' in DATABASE {FS_INTEG_TEST_DB}").collect()
        self.assertEqual(len(res), 0)

    def test_create_if_not_exist_failure(self) -> None:
        temp_session = create_mock_session(
            "CREATE TAG IF NOT EXISTS",
            snowpark_exceptions.SnowparkSQLException("IntentionalSQLError"),
        )

        schema_name = f"foo_{uuid4().hex.upper()}"
        with self.assertRaisesRegex(RuntimeError, "IntentionalSQLError"):
            FeatureStore(
                session=temp_session,
                database=FS_INTEG_TEST_DB,
                name=schema_name,
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )

        # Schema still exist even feature store creation failed.
        res = self._session.sql(f"SHOW SCHEMAS LIKE '{schema_name}' in DATABASE {FS_INTEG_TEST_DB}").collect()
        self.assertEqual(len(res), 1)
        self._session.sql(f"DROP SCHEMA IF EXISTS {FS_INTEG_TEST_DB}.{schema_name}").collect()

    def test_create_feature_store_when_database_not_exists(self) -> None:
        current_schema = create_random_schema(self._session, "FS_TEST")
        db_name = "RANDOM_NONEXIST_NONSENSE_FOOBAR_XYZ_DB"
        dbs = self._session.sql(f"SHOW DATABASES LIKE '{db_name}'").collect()
        self.assertEqual(len(dbs), 0)

        creation_modes = [CreationMode.FAIL_IF_NOT_EXIST, CreationMode.CREATE_IF_NOT_EXIST]
        for mode in creation_modes:
            with self.assertRaisesRegex(ValueError, "Database .* does not exist."):
                FeatureStore(
                    self._session,
                    db_name,
                    current_schema,
                    default_warehouse=self._test_warehouse_name,
                    creation_mode=mode,
                )
            dbs = self._session.sql(f"SHOW DATABASES LIKE '{db_name}'").collect()
            self.assertEqual(len(dbs), 0)

    def test_create_feature_store_when_tags_missing(self) -> None:
        current_schema = create_random_schema(self._session, "FS_TEST")
        with self.assertRaisesRegex(ValueError, "Feature store internal tag .* does not exist."):
            FeatureStore(
                self._session,
                FS_INTEG_TEST_DB,
                current_schema,
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
            )

    def test_create_and_delete_entities(self) -> None:
        fs = self._create_feature_store()

        entities = {
            "User": Entity("USER", ['"uid"']),
            "Ad": Entity('"aD"', ["aid"]),
            "Product": Entity("Product", ['"pid"', "cid"]),
        }

        # create new entities
        for e in entities.values():
            fs.register_entity(e)

        actual_result = fs.list_entities().to_pandas()
        self.assertEqual(len(actual_result["OWNER"]), 3)

        compare_dataframe(
            actual_df=actual_result.drop(columns="OWNER"),
            target_data={
                "NAME": ["aD", "PRODUCT", "USER"],
                "JOIN_KEYS": ['["AID"]', '["pid,CID"]', '["uid"]'],
                "DESC": ["", "", ""],
            },
            sort_cols=["NAME"],
        )

        # throw when trying to delete non-exist entity
        with self.assertRaisesRegex(ValueError, "Entity .* does not exist."):
            fs.delete_entity("AD")
        # you can still get entity
        fs.get_entity('"aD"')
        # now delete actually entity
        fs.delete_entity('"aD"')
        # throw because it not exist
        with self.assertRaisesRegex(ValueError, "Cannot find Entity with name"):
            fs.get_entity('"aD"')

        actual_result = fs.list_entities().to_pandas()
        self.assertEqual(len(actual_result["OWNER"]), 2)

        compare_dataframe(
            actual_df=actual_result.drop(columns="OWNER"),
            target_data={
                "NAME": ["PRODUCT", "USER"],
                "JOIN_KEYS": ['["pid,CID"]', '["uid"]'],
                "DESC": ["", ""],
            },
            sort_cols=["NAME"],
        )

        # create entity already exists
        with self.assertWarnsRegex(UserWarning, "Entity .* already exists..*"):
            fs.register_entity(Entity("User", ["a", "b", "c"]))
        # captitalized entity name is treated the same
        with self.assertWarnsRegex(UserWarning, "Entity.* already exists.*"):
            fs.register_entity(Entity("USER", ["a", "b", "c"]))

        # test delete entity failure with active feature views
        # create a new feature view
        sql = f'SELECT name, id AS "uid", id AS CID, id AS "pid" FROM {self._mock_table}'
        fv = FeatureView(
            name="fv",
            entities=[entities["User"], entities["Product"]],
            feature_df=self._session.sql(sql),
            refresh_freq="1m",
        )
        fs.register_feature_view(feature_view=fv, version="FIRST")
        with self.assertRaisesRegex(ValueError, "Cannot delete Entity .* due to active FeatureViews.*"):
            fs.delete_entity("User")

    def test_retrieve_entity(self) -> None:
        fs = self._create_feature_store()

        e1 = Entity(name="foo", join_keys=["a", "b"], desc="my foo")
        e2 = Entity(name="bar", join_keys=["c"])
        fs.register_entity(e1)
        fs.register_entity(e2)
        re1 = fs.get_entity("foo")
        re2 = fs.get_entity("bar")

        self.assertEqual(e1.name, re1.name)
        self.assertEqual(e1.join_keys, re1.join_keys)
        self.assertEqual(e1.desc, re1.desc)
        self.assertEqual(e2.name, re2.name)
        self.assertEqual(e2.join_keys, re2.join_keys)
        self.assertEqual(e2.desc, re2.desc)

        actual_result = fs.list_entities().to_pandas()
        self.assertEqual(len(actual_result["OWNER"]), 2)

        compare_dataframe(
            actual_df=actual_result.drop(columns="OWNER"),
            target_data={
                "NAME": ["FOO", "BAR"],
                "JOIN_KEYS": ['["A,B"]', '["C"]'],
                "DESC": ["my foo", ""],
            },
            sort_cols=["NAME"],
        )

    def test_update_entity(self) -> None:
        fs = self._create_feature_store()

        e1 = Entity(name="foo", join_keys=["col_1"], desc="old desc")
        fs.register_entity(e1)
        r1 = fs.list_entities().collect()
        self.assertEqual(r1[0]["DESC"], "old desc")

        # update with none desc
        fs.update_entity(name="foo", desc=None)
        r2 = fs.list_entities().collect()
        self.assertEqual(r2[0]["DESC"], "old desc")

        # update with a new desc
        fs.update_entity(name="foo", desc="NEW DESC")
        r3 = fs.list_entities().collect()
        self.assertEqual(r3[0]["DESC"], "NEW DESC")

        # update with empty desc
        fs.update_entity(name="foo", desc="")
        r4 = fs.list_entities().collect()
        self.assertEqual(r4[0]["DESC"], "")

        # update entity does not exist
        with self.assertWarnsRegex(UserWarning, "Entity .* does not exist."):
            fs.update_entity(name="bar", desc="NEW DESC")

    def test_get_entity_system_error(self) -> None:
        fs = self._create_feature_store()
        fs._session = create_mock_session(
            "SHOW TAGS LIKE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to list entities: .*"):
            fs.get_entity("foo")

    def test_register_entity_system_error(self) -> None:
        fs = self._create_feature_store()
        fs._session = create_mock_session(
            "SHOW TAGS LIKE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        e = Entity("foo", ["id"])

        with self.assertRaisesRegex(RuntimeError, "Failed to find object .*"):
            fs.register_entity(e)

    def test_register_feature_view_with_warehouse(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title, ts FROM {self._mock_table}"
        d1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
            refresh_freq="1d",
        )
        r11 = fs.register_feature_view(feature_view=d1, version="1.0")
        self.assertEqual(r11.warehouse, "REGTEST_ML_4XL_MULTI")
        self.assertEqual(
            fs.list_feature_views().select("warehouse").filter(col("version") == "1.0").collect()[0]["WAREHOUSE"],
            "REGTEST_ML_4XL_MULTI",
        )

        d1.warehouse = "REGTEST_ML_SMALL"  # type: ignore[assignment]
        r12 = fs.register_feature_view(feature_view=d1, version="2.0")
        self.assertEqual(r12.warehouse, "REGTEST_ML_SMALL")
        self.assertEqual(
            fs.list_feature_views().select("warehouse").filter(col("version") == "2.0").collect()[0]["WAREHOUSE"],
            "REGTEST_ML_SMALL",
        )

        d2 = FeatureView(
            name="fv2",
            entities=[e],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
            refresh_freq="1d",
            warehouse="REGTEST_ML_SMALL",
        )
        r2 = fs.register_feature_view(feature_view=d2, version="1.0")
        self.assertEqual(r2.warehouse, "REGTEST_ML_SMALL")
        self.assertEqual(
            fs.list_feature_views().select("warehouse").filter(col("name") == "FV2").collect()[0]["WAREHOUSE"],
            "REGTEST_ML_SMALL",
        )

    def test_register_feature_view_with_unregistered_entity(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])

        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table).select(["NAME", "ID", "TITLE", "AGE", "TS"]),
            timestamp_col="ts",
            desc="foobar",
        ).attach_feature_desc({"AGE": "my age", "TITLE": '"my title"'})

        with self.assertRaisesRegex(ValueError, "Entity .* has not been registered."):
            fs.register_feature_view(feature_view=fv, version="v1")

    def test_register_feature_view_as_view(self) -> None:
        """
        APIs covered by test:
            1. register_feature_view
            2. read_feature_view
            3. list_feature_views
            4. get_feature_view
            5. delete_feature_view
            6. generate_dataset (covers retrieve_feature_values)
        """

        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        # create one feature view
        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table).select(["NAME", "ID", "TITLE", "AGE", "TS"]),
            timestamp_col="ts",
            desc="foobar",
        ).attach_feature_desc({"AGE": "my age", "TITLE": '"my title"'})
        fv = fs.register_feature_view(feature_view=fv, version="2.0")

        self._check_tag_value(
            fs,
            fv.fully_qualified_name(),
            "table",
            _FEATURE_STORE_OBJECT_TAG,
            f"""{{"type": "EXTERNAL_FEATURE_VIEW", "pkg_version": "{VERSION}"}}""",
        )

        self.assertEqual(fv, fs.get_feature_view("fv", "2.0"))

        compare_dataframe(
            actual_df=fs.read_feature_view(fv).to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "ID": [1, 2],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
                "TS": [100, 200],
            },
            sort_cols=["ID", "TS"],
        )

        # create another feature view
        new_fv = FeatureView(
            name="new_fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table).select(["ID", "DEPT", "TS"]),
            timestamp_col="ts",
            desc="foobar",
        )
        new_fv = fs.register_feature_view(feature_view=new_fv, version="V1")

        compare_dataframe(
            actual_df=fs.list_feature_views(entity_name="FOO").to_pandas(),
            target_data={
                "NAME": ["FV", "NEW_FV"],
                "VERSION": ["2.0", "V1"],
                "DATABASE_NAME": [fs._config.database] * 2,
                "SCHEMA_NAME": [fs._config.schema] * 2,
                "DESC": ["foobar", "foobar"],
                "ENTITIES": ['[\n  "FOO"\n]'] * 2,
                "REFRESH_FREQ": [None, None],
                "REFRESH_MODE": [None, None],
                "SCHEDULING_STATE": [None, None],
                "WAREHOUSE": [None, None],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER"],
        )

        # generate data on multiple feature views
        spine_df = self._session.create_dataframe([(1, 101)], schema=["id", "ts"])
        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv, new_fv],
            spine_timestamp_col="ts",
            include_feature_view_timestamp_col=True,
            name="test_ds",
            output_type="table",
        )

        compare_dataframe(
            actual_df=ds.to_pandas(),
            target_data={
                "ID": [1],
                "TS": [101],
                "FV_2.0_TS": [100],
                "NAME": ["jonh"],
                "TITLE": ["boss"],
                "AGE": [20],
                "NEW_FV_V1_TS": [100],
                "DEPT": ["sales"],
            },
            sort_cols=["ID"],
        )

    def test_register_feature_view_system_error(self) -> None:
        fs = self._create_feature_store()
        e = Entity("foo", ["id"])
        fs.register_entity(e)

        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table).select(["NAME", "ID"]),
        )

        fs._session = create_mock_session(
            "CREATE VIEW",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        with self.assertRaisesRegex(RuntimeError, "(?s)Create view .* failed.*"):
            fs.register_feature_view(feature_view=fv, version="v1")

        fs._session = create_mock_session(
            "CREATE DYNAMIC TABLE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        fv2 = FeatureView(
            name="fv2",
            entities=[e],
            feature_df=self._session.table(self._mock_table).select(["NAME", "ID"]),
            refresh_freq="1d",
        )
        with self.assertRaisesRegex(RuntimeError, "(?s)Create dynamic table .* failed.*"):
            fs.register_feature_view(feature_view=fv2, version="v2")

    def test_create_and_delete_feature_views(self) -> None:
        fs = self._create_feature_store()

        e1 = Entity("foo", ["aid"])
        e2 = Entity("bar", ["uid"])
        fs.register_entity(e1)
        fs.register_entity(e2)

        # create a new feature view
        sql0 = f"SELECT name, id AS aid, id AS uid FROM {self._mock_table}"
        fv0 = FeatureView(
            name="fv0",
            entities=[e1, e2],
            feature_df=self._session.sql(sql0),
            refresh_freq="1 minute",
            desc="my_fv0",
        ).attach_feature_desc({"name": "my name"})
        self.assertEqual(fv0.status, FeatureViewStatus.DRAFT)
        self.assertIsNone(fv0.version)
        self.assertEqual(fv0.refresh_freq, "1 minute")
        self.assertEqual(fv0.feature_descs, {"NAME": "my name"})

        # register feature view
        fv0 = fs.register_feature_view(feature_view=fv0, version="FIRST")
        self.assertEqual(fv0.version, "FIRST")
        self.assertTrue(
            fv0.status == FeatureViewStatus.ACTIVE or fv0.status == FeatureViewStatus.RUNNING
        )  # fv0.status == FeatureViewStatus.RUNNING can be removed after BCR 2024_02 gets fully deployed
        self.assertEqual(fv0.refresh_freq, "1 minute")
        self.assertEqual(fv0, fs.get_feature_view("fv0", "FIRST"))
        self._check_tag_value(
            fs,
            fv0.fully_qualified_name(),
            "table",
            _FEATURE_STORE_OBJECT_TAG,
            f"""{{"type": "MANAGED_FEATURE_VIEW", "pkg_version": "{VERSION}"}}""",
        )

        # suspend feature view
        fv0 = fs.suspend_feature_view(fv0)
        self.assertEqual(fv0.status, FeatureViewStatus.SUSPENDED)

        # create a new version
        sql1 = f"SELECT name, id AS aid, id AS uid, ts FROM {self._mock_table}"
        new_fv0 = FeatureView(
            name="fv0",
            entities=[e1, e2],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
            desc="my_new_fv0",
        )
        new_fv0 = fs.register_feature_view(feature_view=new_fv0, version="SECOND")
        self.assertEqual(new_fv0.version, "SECOND")
        self.assertEqual(new_fv0.refresh_freq, "DOWNSTREAM")

        # create another brand new feature view
        fv1 = FeatureView(
            name="fv1",
            entities=[e1, e2],
            feature_df=self._session.sql(sql0),
            refresh_freq="5 minutes",
            desc="my_fv1",
        )
        alternate_warehouse = "REGTEST_ML_SMALL"
        fs.update_default_warehouse(alternate_warehouse)
        fv1 = fs.register_feature_view(feature_view=fv1, version="FIRST")

        compare_dataframe(
            actual_df=fs.list_feature_views().to_pandas(),
            target_data={
                "NAME": ["FV0", "FV0", "FV1"],
                "VERSION": ["FIRST", "SECOND", "FIRST"],
                "DATABASE_NAME": [fs._config.database] * 3,
                "SCHEMA_NAME": [fs._config.schema] * 3,
                "DESC": ["my_fv0", "my_new_fv0", "my_fv1"],
                "ENTITIES": ['[\n  "FOO",\n  "BAR"\n]'] * 3,
                "REFRESH_FREQ": ["1 minute", "DOWNSTREAM", "5 minutes"],
                "REFRESH_MODE": ["INCREMENTAL", "INCREMENTAL", "INCREMENTAL"],
                "SCHEDULING_STATE": ["SUSPENDED", "ACTIVE", "ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        # delete feature view
        with self.assertRaisesRegex(ValueError, "FeatureView .* has not been registered."):
            unmaterialized_fv = FeatureView("unmaterialized", [e1], self._session.sql(sql0))
            fs.delete_feature_view(unmaterialized_fv)

        fs.delete_feature_view(fs.get_feature_view("FV0", "FIRST"))

        compare_dataframe(
            actual_df=fs.list_feature_views().to_pandas(),
            target_data={
                "NAME": ["FV0", "FV1"],
                "VERSION": ["SECOND", "FIRST"],
                "DATABASE_NAME": [fs._config.database] * 2,
                "SCHEMA_NAME": [fs._config.schema] * 2,
                "DESC": ["my_new_fv0", "my_fv1"],
                "ENTITIES": ['[\n  "FOO",\n  "BAR"\n]'] * 2,
                "REFRESH_FREQ": ["DOWNSTREAM", "5 minutes"],
                "REFRESH_MODE": ["INCREMENTAL", "INCREMENTAL"],
                "SCHEDULING_STATE": ["ACTIVE", "ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        # test get feature view obj
        fv = fs.get_feature_view(name="fv1", version="FIRST")
        self.assertEqual(fv.name, "FV1")
        self.assertEqual(fv.version, "FIRST")
        self.assertEqual(fv.query, sql0)
        self.assertTrue(
            fv.status == FeatureViewStatus.ACTIVE or fv.status == FeatureViewStatus.RUNNING
        )  # fv.status == FeatureViewStatus.RUNNING can be removed after BCR 2024_02 gets fully deployed
        self.assertEqual(fv.refresh_freq, "5 minutes")
        self.assertEqual(fv.warehouse, alternate_warehouse)
        self.assertEqual(fv.desc, "my_fv1")
        self.assertEqual(fv.timestamp_col, None)

        fv = fs.get_feature_view(name="fv0", version="SECOND")
        self.assertEqual(str(fv.timestamp_col).upper(), "TS")

        self.assertEqual(fs.list_feature_views().count(), 2)
        fs.delete_feature_view("fv0", "SECOND")
        self.assertEqual(fs.list_feature_views().count(), 1)

    def test_create_duplicated_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"SELECT * FROM {self._mock_table}"
        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="1m",
        )
        fv = fs.register_feature_view(feature_view=fv, version="r1")

        with self.assertWarnsRegex(UserWarning, "FeatureView FV/r1 already exists. Skip registration. .*"):
            fv = fs.register_feature_view(feature_view=fv, version="r1")
            self.assertIsNotNone(fv)

        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="1m",
        )
        with self.assertWarnsRegex(UserWarning, "FeatureView .* already exists..*"):
            fv = fs.register_feature_view(feature_view=fv, version="r1")

    def test_resume_and_suspend_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        my_fv = FeatureView(
            name="my_fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table),
            refresh_freq="DOWNSTREAM",
        )
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1")
        my_fv = fs.suspend_feature_view(my_fv)
        self.assertEqual(my_fv.status, FeatureViewStatus.SUSPENDED)
        my_fv = fs.resume_feature_view(my_fv)
        self.assertTrue(my_fv.status == FeatureViewStatus.ACTIVE)

        # test suspend/resume with name and version
        my_fv = fs.suspend_feature_view("my_fv", "v1")
        self.assertEqual(my_fv.status, FeatureViewStatus.SUSPENDED)
        all_fvs = fs.list_feature_views().filter((col("NAME") == "MY_FV") & (col("VERSION") == "v1")).collect()
        self.assertEqual(len(all_fvs), 1)
        self.assertEqual(all_fvs[0]["SCHEDULING_STATE"], "SUSPENDED")

        my_fv = fs.resume_feature_view("my_fv", "v1")
        self.assertEqual(my_fv.status, FeatureViewStatus.ACTIVE)
        all_fvs = fs.list_feature_views().filter((col("NAME") == "MY_FV") & (col("VERSION") == "v1")).collect()
        self.assertEqual(len(all_fvs), 1)
        self.assertEqual(all_fvs[0]["SCHEDULING_STATE"], "ACTIVE")

    def test_resume_and_suspend_feature_view_system_error(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)
        my_fv = FeatureView(
            name="my_fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table),
            refresh_freq="DOWNSTREAM",
        )
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1")

        original_session = fs._session
        fs._session = create_mock_session(
            "ALTER DYNAMIC TABLE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        with self.assertRaisesRegex(RuntimeError, "Failed to update feature view"):
            my_fv = fs.suspend_feature_view(my_fv)

        fs._session = original_session
        my_fv = fs.suspend_feature_view(my_fv)

        fs._session = create_mock_session(
            "ALTER DYNAMIC TABLE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        with self.assertRaisesRegex(RuntimeError, "Failed to update feature view.*"):
            my_fv = fs.resume_feature_view(my_fv)

        fs._session = original_session

    @parameterized.parameters("1d", "0 0 * * * America/Los_Angeles")  # type: ignore[misc]
    def test_refresh_feature_view(self, refresh_freq: str) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        my_fv = FeatureView(
            name="my_fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table),
            refresh_freq=refresh_freq,
        )
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1", block=True)
        res_1 = fs.get_refresh_history(my_fv).collect()
        self.assertEqual(len(res_1), 1)

        fs.refresh_feature_view(my_fv)
        df_2 = fs.get_refresh_history(my_fv)
        self.assertSameElements(
            df_2.columns, ["NAME", "STATE", "REFRESH_START_TIME", "REFRESH_END_TIME", "REFRESH_ACTION"]
        )

        df_3 = fs.get_refresh_history(my_fv, verbose=True)
        self.assertSameElements(
            df_3.columns,
            [
                "NAME",
                "SCHEMA_NAME",
                "DATABASE_NAME",
                "STATE",
                "STATE_CODE",
                "STATE_MESSAGE",
                "QUERY_ID",
                "DATA_TIMESTAMP",
                "REFRESH_START_TIME",
                "REFRESH_END_TIME",
                "COMPLETION_TARGET",
                "QUALIFIED_NAME",
                "LAST_COMPLETED_DEPENDENCY",
                "STATISTICS",
                "REFRESH_ACTION",
                "REFRESH_TRIGGER",
                "TARGET_LAG_SEC",
                "GRAPH_HISTORY_VALID_FROM",
            ],
        )
        res_3 = df_3.order_by("REFRESH_START_TIME", ascending=False).collect()
        self.assertEqual(len(res_3), 2)
        self.assertEqual(res_3[0]["STATE"], "SUCCEEDED")
        self.assertEqual(res_3[0]["REFRESH_TRIGGER"], "MANUAL")

        # teset refresh_feature_view and get_refresh_history with name and version
        fs.refresh_feature_view("my_fv", "v1")
        df_4 = fs.get_refresh_history("my_fv", "v1", verbose=True)

        self.assertSameElements(
            df_4.columns,
            [
                "NAME",
                "SCHEMA_NAME",
                "DATABASE_NAME",
                "STATE",
                "STATE_CODE",
                "STATE_MESSAGE",
                "QUERY_ID",
                "DATA_TIMESTAMP",
                "REFRESH_START_TIME",
                "REFRESH_END_TIME",
                "COMPLETION_TARGET",
                "QUALIFIED_NAME",
                "LAST_COMPLETED_DEPENDENCY",
                "STATISTICS",
                "REFRESH_ACTION",
                "REFRESH_TRIGGER",
                "TARGET_LAG_SEC",
                "GRAPH_HISTORY_VALID_FROM",
            ],
        )
        res_4 = df_4.order_by("REFRESH_START_TIME", ascending=False).collect()
        self.assertEqual(len(res_4), 3)
        self.assertEqual(res_4[0]["STATE"], "SUCCEEDED")
        self.assertEqual(res_4[0]["REFRESH_TRIGGER"], "MANUAL")

    def test_refresh_static_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        my_fv = FeatureView(
            name="my_static_fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table),
            refresh_freq=None,
        )
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1", block=True)

        with self.assertWarnsRegex(
            UserWarning,
            "Static feature view can't be refreshed. You must set refresh_freq when register_feature_view().",
        ):
            fs.refresh_feature_view(my_fv)

        with self.assertWarnsRegex(UserWarning, "Static feature view never refreshes"):
            fs.get_refresh_history(my_fv)

    def test_read_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"SELECT name, id, title, age, ts FROM {self._mock_table}"
        my_fv = FeatureView(
            name="my_fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="DOWNSTREAM",
        )

        with self.assertRaisesRegex(ValueError, "FeatureView .* has not been registered."):
            fs.read_feature_view(my_fv)

        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1")

        df = fs.read_feature_view(my_fv)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "ID": [1, 2],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
                "TS": [100, 200],
            },
            sort_cols=["NAME"],
        )

        df = fs.read_feature_view("my_fv", "v1")
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "ID": [1, 2],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
                "TS": [100, 200],
            },
            sort_cols=["NAME"],
        )

        with self.assertRaisesRegex(ValueError, "Version must be provided when argument feature_view is a str."):
            fs.read_feature_view("my_fv")  # type: ignore[call-overload]

        with self.assertRaisesRegex(ValueError, "Failed to find FeatureView .*"):
            fs.read_feature_view("my_fv", "v2")

    def test_register_with_cron_expr(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"SELECT name, id, title, age, ts FROM {self._mock_table}"
        my_fv = FeatureView(
            name="my_fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="* * * * * America/Los_Angeles",
        ).attach_feature_desc({"title": "my title"})
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1")
        fv = fs.get_feature_view("my_fv", "v1")
        self.assertEqual(my_fv, fv)

        task_name = FeatureView._get_physical_name(fv.name, fv.version).resolved()  # type: ignore[arg-type]
        res = self._session.sql(f"SHOW TASKS LIKE '{task_name}' IN SCHEMA {fs._config.full_schema_path}").collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["state"], "started")
        self.assertEqual(fv.refresh_freq, "DOWNSTREAM")
        self._check_tag_value(
            fs,
            fv.fully_qualified_name(),
            "task",
            _FEATURE_STORE_OBJECT_TAG,
            f"""{{"type": "FEATURE_VIEW_REFRESH_TASK", "pkg_version": "{VERSION}"}}""",
        )

        fv = fs.suspend_feature_view(fv)
        res = self._session.sql(f"SHOW TASKS LIKE '{task_name}' IN SCHEMA {fs._config.full_schema_path}").collect()
        self.assertEqual(res[0]["state"], "suspended")

        fv = fs.resume_feature_view(fv)
        res = self._session.sql(f"SHOW TASKS LIKE '{task_name}' IN SCHEMA {fs._config.full_schema_path}").collect()
        self.assertEqual(res[0]["state"], "started")

        fs.delete_feature_view(fv)
        res = self._session.sql(f"SHOW TASKS LIKE '{task_name}' IN SCHEMA {fs._config.full_schema_path}").collect()
        self.assertEqual(len(res), 0)

    def test_retrieve_time_series_feature_values(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title, ts FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="1.0")

        sql2 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name="fv2",
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="1.0")

        sql3 = f"SELECT id, dept FROM {self._mock_table}"
        fv3 = FeatureView(
            name="fv3",
            entities=[e],
            feature_df=self._session.sql(sql3),
            refresh_freq="DOWNSTREAM",
        )
        fv3 = fs.register_feature_view(feature_view=fv3, version="1.0")

        spine_df = self._session.create_dataframe([(1, 101), (2, 202), (1, 90)], schema=["id", "ts"])
        df = fs.retrieve_feature_values(
            spine_df=spine_df,
            features=cast(List[Union[FeatureView, FeatureViewSlice]], [fv1, fv2, fv3]),
            spine_timestamp_col="ts",
            include_feature_view_timestamp_col=True,
        )

        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 1, 2],
                "TS": [90, 101, 202],
                "FV1_1.0_TS": [None, 100, 200],
                "NAME": [None, "jonh", "porter"],
                "TITLE": [None, "boss", "manager"],
                "FV2_1.0_TS": [None, 100, 200],
                "AGE": [None, 20, 30],
                "DEPT": ["sales", "sales", "engineer"],
            },
            sort_cols=["ID", "TS"],
        )

    def test_retrieve_feature_values(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1")

        sql2 = f"SELECT id, age FROM {self._mock_table}"
        fv2 = FeatureView(
            name="fv2",
            entities=[e],
            feature_df=self._session.sql(sql2),
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1")

        spine_df = self._session.create_dataframe([(1), (2)], schema=["id"])
        df = fs.retrieve_feature_values(
            spine_df=spine_df,
            features=cast(List[Union[FeatureView, FeatureViewSlice]], [fv1, fv2]),
        )

        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["ID"],
        )

        df = fs.retrieve_feature_values(
            spine_df=spine_df,
            features=cast(List[Union[FeatureView, FeatureViewSlice]], [fv1.slice(["name"]), fv2]),
        )
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "AGE": [20, 30],
            },
            sort_cols=["ID"],
        )

        df = fs.retrieve_feature_values(
            spine_df=spine_df,
            features=cast(List[Union[FeatureView, FeatureViewSlice]], [fv1.slice(["name"]), fv2]),
            exclude_columns=["NAME"],
        )
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "AGE": [20, 30],
            },
            sort_cols=["ID"],
        )

        # test retrieve_feature_values with serialized feature objects
        fv1_slice = fv1.slice(["name"])
        dataset = fs.generate_dataset(
            spine_df=spine_df, features=[fv1_slice, fv2], name="test_ds", output_type="dataset"
        )
        features = fs.load_feature_views_from_dataset(dataset)
        df = fs.retrieve_feature_values(spine_df=spine_df, features=features)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "AGE": [20, 30],
            },
            sort_cols=["ID"],
        )
        self.assertEqual([fv1_slice, fv2], fs.load_feature_views_from_dataset(dataset))

    def test_invalid_load_feature_views_from_dataset(self) -> None:
        fs = self._create_feature_store()
        ds = dataset.create_from_dataframe(
            self._session,
            "test_ds",
            uuid4().hex,
            input_dataframe=self._session.create_dataframe([1, 2, 3], schema=["foo"]),
        )
        with self.assertRaisesRegex(ValueError, r"does not contain valid feature view information\."):
            fs.load_feature_views_from_dataset(ds)

    def test_list_feature_views(self) -> None:
        fs = self._create_feature_store()

        e1 = Entity("foo", ["id"])
        fs.register_entity(e1)
        e2 = Entity("bar", ["name"])
        fs.register_entity(e2)

        compare_dataframe(
            actual_df=fs.list_feature_views(entity_name="FOO").to_pandas(),
            target_data={
                "NAME": [],
                "VERSION": [],
                "DATABASE_NAME": [],
                "SCHEMA_NAME": [],
                "CREATED_ON": [],
                "OWNER": [],
                "DESC": [],
                "ENTITIES": [],
                "REFRESH_FREQ": [],
                "REFRESH_MODE": [],
                "SCHEDULING_STATE": [],
                "WAREHOUSE": [],
            },
            sort_cols=["NAME"],
        )

        # 1. Right side is FeatureViewSlice
        sql1 = f"SELECT id, name, ts FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e1],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv1.attach_feature_desc({"name": "this is my name col"})
        fs.register_feature_view(feature_view=fv1, version="v1")

        sql2 = f"SELECT id, name, title, age FROM {self._mock_table}"
        fv2 = FeatureView(
            name="fv2",
            entities=[e2],
            feature_df=self._session.sql(sql2),
            refresh_freq="DOWNSTREAM",
            desc="foobar",
        )
        fs.register_feature_view(feature_view=fv2, version="v1")

        fv3 = FeatureView(
            name="fv3",
            entities=[e1, e2],
            feature_df=self._session.sql(sql2),
            refresh_freq="DOWNSTREAM",
            desc="foobar",
        )
        fs.register_feature_view(feature_view=fv3, version="v1")

        compare_dataframe(
            actual_df=fs.list_feature_views().to_pandas(),
            target_data={
                "NAME": ["FV1", "FV2", "FV3"],
                "VERSION": ["v1", "v1", "v1"],
                "DATABASE_NAME": [fs._config.database] * 3,
                "SCHEMA_NAME": [fs._config.schema] * 3,
                "DESC": ["", "foobar", "foobar"],
                "ENTITIES": ['[\n  "FOO"\n]', '[\n  "BAR"\n]', '[\n  "FOO",\n  "BAR"\n]'],
                "REFRESH_FREQ": ["DOWNSTREAM", "DOWNSTREAM", "DOWNSTREAM"],
                "REFRESH_MODE": ["INCREMENTAL", "INCREMENTAL", "INCREMENTAL"],
                "SCHEDULING_STATE": ["ACTIVE", "ACTIVE", "ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        compare_dataframe(
            actual_df=fs.list_feature_views(entity_name="FOO").to_pandas(),
            target_data={
                "NAME": ["FV1", "FV3"],
                "VERSION": ["v1", "v1"],
                "DATABASE_NAME": [fs._config.database, fs._config.database],
                "SCHEMA_NAME": [fs._config.schema, fs._config.schema],
                "DESC": ["", "foobar"],
                "ENTITIES": ['[\n  "FOO"\n]', '[\n  "FOO",\n  "BAR"\n]'],
                "REFRESH_FREQ": ["DOWNSTREAM", "DOWNSTREAM"],
                "REFRESH_MODE": ["INCREMENTAL", "INCREMENTAL"],
                "SCHEDULING_STATE": ["ACTIVE", "ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        compare_dataframe(
            actual_df=fs.list_feature_views(feature_view_name="FV2").to_pandas(),
            target_data={
                "NAME": ["FV2"],
                "VERSION": ["v1"],
                "DATABASE_NAME": [fs._config.database],
                "SCHEMA_NAME": [fs._config.schema],
                "DESC": ["foobar"],
                "ENTITIES": ['[\n  "BAR"\n]'],
                "REFRESH_FREQ": ["DOWNSTREAM"],
                "REFRESH_MODE": ["INCREMENTAL"],
                "SCHEDULING_STATE": ["ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        compare_dataframe(
            actual_df=fs.list_feature_views(entity_name="BAR", feature_view_name="FV2").to_pandas(),
            target_data={
                "NAME": ["FV2"],
                "VERSION": ["v1"],
                "DATABASE_NAME": [fs._config.database],
                "SCHEMA_NAME": [fs._config.schema],
                "DESC": ["foobar"],
                "ENTITIES": ['[\n  "BAR"\n]'],
                "REFRESH_FREQ": ["DOWNSTREAM"],
                "REFRESH_MODE": ["INCREMENTAL"],
                "SCHEDULING_STATE": ["ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        compare_dataframe(
            actual_df=fs.list_feature_views(entity_name="FOO", feature_view_name="FV3").to_pandas(),
            target_data={
                "NAME": ["FV3"],
                "VERSION": ["v1"],
                "DATABASE_NAME": [fs._config.database],
                "SCHEMA_NAME": [fs._config.schema],
                "DESC": ["foobar"],
                "ENTITIES": ['[\n  "FOO",\n  "BAR"\n]'],
                "REFRESH_FREQ": ["DOWNSTREAM"],
                "REFRESH_MODE": ["INCREMENTAL"],
                "SCHEDULING_STATE": ["ACTIVE"],
            },
            sort_cols=["NAME"],
            exclude_cols=["CREATED_ON", "OWNER", "WAREHOUSE"],
        )

        compare_dataframe(
            actual_df=fs.list_feature_views(entity_name="BAR", feature_view_name="BAZ").to_pandas(),
            target_data={
                "NAME": [],
                "VERSION": [],
                "DATABASE_NAME": [],
                "SCHEMA_NAME": [],
                "CREATED_ON": [],
                "OWNER": [],
                "DESC": [],
                "ENTITIES": [],
                "REFRESH_FREQ": [],
                "REFRESH_MODE": [],
                "SCHEDULING_STATE": [],
                "WAREHOUSE": [],
            },
            sort_cols=["NAME"],
        )
        fs._check_feature_store_object_versions()

    def test_list_feature_views_system_error(self) -> None:
        fs = self._create_feature_store()
        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fs.register_feature_view(feature_view=fv1, version="v1")

        fs._session = create_mock_session(
            "SHOW DYNAMIC TABLES LIKE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        with self.assertRaisesRegex(RuntimeError, "Failed to find object"):
            fs.list_feature_views()

        fs._session = create_mock_session(
            "SELECT ENTITY_DETAIL",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to lookup tagged objects for"):
            fs.list_feature_views(entity_name="foo")

    def test_generate_training_set(self) -> None:
        fs = self._create_feature_store()

        e = Entity('"fOO"', ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1")

        sql2 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name='"FvfV2"',
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1")
        spine_df = self._session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])

        # Generate unmaterialized dataset
        ds0 = fs.generate_training_set(
            spine_df=spine_df,
            features=[fv1, fv2],
            spine_timestamp_col="ts",
        )

        compare_dataframe(
            actual_df=ds0.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
                "AGE": [20, 20],
            },
            sort_cols=["ID", "TS"],
        )

        # Generate dataset
        ds1 = fs.generate_training_set(
            spine_df=spine_df,
            features=[fv1, fv2],
            save_as="foobar",
            spine_timestamp_col="ts",
        )

        compare_dataframe(
            actual_df=ds1.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
                "AGE": [20, 20],
            },
            sort_cols=["ID", "TS"],
        )
        # FIXME: Attach Feature View metadata to table-backed Datasets
        # self.assertEqual([fv1, fv2], fs.load_feature_views_from_dataset(ds1))

        # Generate dataset with exclude_columns and check both materialization and non-materialization path
        spine_df = self._session.create_dataframe([(1, 101), (2, 202)], schema=["id", "ts"])

        ds4 = fs.generate_training_set(
            spine_df=spine_df,
            features=[fv1, fv2],
            save_as="foobar2",
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
        )
        compare_dataframe(
            actual_df=ds4.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        ds5 = fs.generate_training_set(
            spine_df=spine_df,
            features=[fv1, fv2],
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
            save_as="test_ds",
        )
        compare_dataframe(
            actual_df=ds5.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        # Generate data should fail with errorifexists if table already exist
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            fs.generate_training_set(
                spine_df=spine_df,
                features=[fv1, fv2],
                save_as="foobar",
                spine_timestamp_col="ts",
            )

        # Invalid dataset names should be rejected
        with self.assertRaisesRegex(ValueError, "Invalid identifier"):
            fs.generate_training_set(
                spine_df=spine_df,
                features=[fv1, fv2],
                save_as=".bar",
                spine_timestamp_col="ts",
            )

        # invalid columns in exclude_columns should fail
        with self.assertRaisesRegex(ValueError, "FOO in exclude_columns not exists in.*"):
            fs.generate_training_set(
                spine_df=spine_df,
                features=[fv1, fv2],
                save_as="foobar3",
                spine_timestamp_col="ts",
                exclude_columns=["foo"],
            )

    def test_generate_dataset_as_table(self) -> None:
        fs = self._create_feature_store()

        e = Entity('"fOO"', ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1")

        sql2 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name='"FvfV2"',
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1")
        spine_df = self._session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])

        # Generate dataset
        ds1 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            name="foobar",
            version="test",
            spine_timestamp_col="ts",
            output_type="table",
        )

        compare_dataframe(
            actual_df=ds1.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
                "AGE": [20, 20],
            },
            sort_cols=["ID", "TS"],
        )
        # FIXME: Attach Feature View metadata to table-backed Datasets
        # self.assertEqual([fv1, fv2], fs.load_feature_views_from_dataset(ds1))

        # Generate dataset with exclude_columns and check both materialization and non-materialization path
        spine_df = self._session.create_dataframe([(1, 101), (2, 202)], schema=["id", "ts"])

        ds4 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            name="foobar2",
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
            output_type="table",
        )
        compare_dataframe(
            actual_df=ds4.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        ds5 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
            name="test_ds",
            output_type="table",
        )
        compare_dataframe(
            actual_df=ds5.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        # Generate data should fail with errorifexists if table already exist
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                name="foobar",
                version="test",
                spine_timestamp_col="ts",
                output_type="table",
            )

        # Invalid dataset names should be rejected
        with self.assertRaisesRegex(ValueError, "Invalid identifier"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                name=".bar",
                spine_timestamp_col="ts",
                output_type="table",
            )

        # invalid columns in exclude_columns should fail
        with self.assertRaisesRegex(ValueError, "FOO in exclude_columns not exists in.*"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                name="foobar3",
                spine_timestamp_col="ts",
                exclude_columns=["foo"],
                output_type="table",
            )

    def test_generate_dataset_as_table_external_schema(self) -> None:
        database_name = self._session.get_current_database()
        schema_name = create_random_schema(self._session, "FS_TEST_EXTERNAL_SCHEMA", database=database_name)
        fs = self._create_feature_store()
        self.assertNotEqual(fs._config.schema, schema_name)

        e = Entity('"fOO"', ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        # Generate dataset on external schema
        spine_df = self._session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])
        ds_name = "dataset_external_schema"
        show_tables_query = f"SHOW TABLES LIKE '{ds_name}%' IN SCHEMA {database_name}.{schema_name}"
        self.assertEqual(0, self._session.sql(show_tables_query).count())
        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1],
            name=f"{database_name}.{schema_name}.{ds_name}",
            spine_timestamp_col="ts",
            output_type="table",
        )

        # Generated dataset should be in external schema
        self.assertEqual(1, self._session.sql(show_tables_query).count())

        compare_dataframe(
            actual_df=ds.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
            },
            sort_cols=["ID", "TS"],
        )

        # Fail on non-existent schema
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            fs.generate_dataset(
                spine_df=spine_df, features=[fv1], name="NONEXISTENT_SCHEMA.foobar", output_type="table"
            )

    def test_generate_dataset_as_dataset(self) -> None:
        fs = self._create_feature_store()

        e = Entity('"fOO"', ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1")

        sql2 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name='"FvfV2"',
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1")
        spine_df = self._session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])

        # Generate dataset
        ds1 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            name="foobar",
            version="test",
            spine_timestamp_col="ts",
            output_type="dataset",
        )

        compare_dataframe(
            actual_df=ds1.read.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
                "AGE": [20, 20],
            },
            sort_cols=["ID", "TS"],
        )
        self.assertEqual([fv1, fv2], fs.load_feature_views_from_dataset(ds1))

        # Generate dataset with exclude_columns and check both materialization and non-materialization path
        spine_df = self._session.create_dataframe([(1, 101), (2, 202)], schema=["id", "ts"])

        ds4 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            name="foobar2",
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
            output_type="dataset",
        )
        compare_dataframe(
            actual_df=ds4.read.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        ds5 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
            name="test_ds",
            output_type="dataset",
        )
        compare_dataframe(
            actual_df=ds5.read.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        # Generate data should fail with errorifexists if table already exist
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                name="foobar",
                version="test",
                spine_timestamp_col="ts",
                output_type="dataset",
            )

        # Invalid dataset names should be rejected
        with self.assertRaisesRegex(ValueError, "Invalid identifier"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                name=".bar",
                spine_timestamp_col="ts",
                output_type="dataset",
            )

        # invalid columns in exclude_columns should fail
        with self.assertRaisesRegex(ValueError, "FOO in exclude_columns not exists in.*"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                name="foobar3",
                spine_timestamp_col="ts",
                exclude_columns=["foo"],
                output_type="dataset",
            )

    def test_generate_dataset_as_dataset_external_schema(self) -> None:
        database_name = self._session.get_current_database()
        schema_name = create_random_schema(self._session, "FS_TEST_EXTERNAL_SCHEMA", database=database_name)
        fs = self._create_feature_store()
        self.assertNotEqual(fs._config.schema, schema_name)

        e = Entity('"fOO"', ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        # Generate dataset on external schema
        spine_df = self._session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])
        ds_name = "dataset_external_schema"
        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1],
            name=f"{database_name}.{schema_name}.{ds_name}",
            spine_timestamp_col="ts",
            output_type="dataset",
        )

        # Generated dataset should be in external schema
        self.assertGreater(len(ds.read.files()), 0)
        for file in ds.read.files():
            self.assertContainsExactSubsequence(file, f"{database_name}.{schema_name}.{ds_name}")

        ds_df = ds.read.to_snowpark_dataframe()
        compare_dataframe(
            actual_df=ds_df.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
            },
            sort_cols=["ID", "TS"],
        )

        # Fail on non-existent schema
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1],
                name="NONEXISTENT_SCHEMA.foobar",
                output_type="dataset",
            )

    def test_generate_dataset_disabled(self) -> None:
        with Session.builder.configs(self._session_config).create() as session:
            try:
                session.sql("ALTER SESSION SET FEATURE_DATASET=DISABLED").collect()
            except snowpark_exceptions.SnowparkSQLException as ex:
                self.skipTest("Failed to disable Dataset with error %r" % ex)

            fs = FeatureStore(
                session,
                FS_INTEG_TEST_DB,
                create_random_schema(session, "FS_DATASET_TEST"),
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )

            e = Entity("foo", ["id"])
            fs.register_entity(e)

            sql1 = f"SELECT id, age, ts FROM {self._mock_table}"
            fv1 = FeatureView(
                name='"FvfV1"',
                entities=[e],
                feature_df=session.sql(sql1),
                timestamp_col="ts",
                refresh_freq="DOWNSTREAM",
            )
            fv1 = fs.register_feature_view(feature_view=fv1, version="v1")
            spine_df = session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])

            with self.assertRaisesRegex(RuntimeError, "set FEATURE_DATASET=ENABLED"):
                fs.generate_dataset(
                    spine_df=spine_df,
                    features=[fv1],
                    name="foobar",
                    version="test",
                    spine_timestamp_col="ts",
                    output_type="dataset",
                )

    def test_clear_feature_store_in_existing_schema(self) -> None:
        current_schema = create_random_schema(self._session, "TEST_CLEAR_FEATURE_STORE_IN_EXISTING_SCHEMA")

        # create some objects outside of feature store domain, later will check if they still exists after fs._clear()
        full_schema_path = f"{FS_INTEG_TEST_DB}.{current_schema}"
        sql = f"SELECT name, id AS uid FROM {self._mock_table}"
        self._session.sql(
            f"""
            CREATE DYNAMIC TABLE {current_schema}.my_dynamic_table
            TARGET_LAG='1h'
            WAREHOUSE={self._test_warehouse_name}
            AS {sql}
        """
        ).collect()
        self._session.sql(f"CREATE TABLE {current_schema}.my_table (id int)").collect()
        self._session.sql(f"CREATE VIEW {current_schema}.my_view AS {sql}").collect()
        self._session.sql(f"CREATE TASK {current_schema}.my_task AS SELECT CURRENT_TIMESTAMP").collect()
        self._session.sql(f"CREATE TAG {current_schema}.my_tag").collect()

        fs = self._create_feature_store(current_schema)
        e = Entity(name="foo", join_keys=["id"])
        fs.register_entity(e)
        sql = f"SELECT name, id FROM {self._mock_table}"
        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="* * * * * America/Los_Angeles",
        )
        fv = fs.register_feature_view(feature_view=fv, version="v1")

        def check_fs_objects(expected_count: int) -> None:
            result = self._session.sql(f"SHOW DYNAMIC TABLES LIKE 'FV$V1' IN SCHEMA {full_schema_path}").collect()
            self.assertEqual(len(result), expected_count)
            result = self._session.sql(f"SHOW TASKS LIKE 'FV$V1' IN SCHEMA {full_schema_path}").collect()
            self.assertEqual(len(result), expected_count)

        check_fs_objects(1)
        fs._clear(dryrun=False)
        check_fs_objects(0)

        result = self._session.sql(
            f"SHOW DYNAMIC TABLES LIKE 'my_dynamic_table' IN SCHEMA {full_schema_path}"
        ).collect()
        self.assertEqual(len(result), 1)
        result = self._session.sql(f"SHOW TABLES LIKE 'my_table' IN SCHEMA {full_schema_path}").collect()
        self.assertEqual(len(result), 1)
        result = self._session.sql(f"SHOW VIEWS LIKE 'my_view' IN SCHEMA {full_schema_path}").collect()
        self.assertEqual(len(result), 1)
        result = self._session.sql(f"SHOW TASKS LIKE 'my_task' IN SCHEMA {full_schema_path}").collect()
        self.assertEqual(len(result), 1)
        result = self._session.sql(f"SHOW TAGS LIKE 'my_tag' IN SCHEMA {full_schema_path}").collect()
        self.assertEqual(len(result), 1)

    def test_dynamic_table_full_refresh_warning(self) -> None:
        temp_stage_name = "test_dynamic_table_full_refresh_warning_stage"
        self._session.sql(f"USE DATABASE {FS_INTEG_TEST_DB}").collect()
        self._session.sql(f"CREATE OR REPLACE STAGE {temp_stage_name}").collect()

        udf_name = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.minus_one"

        @udf(  # type: ignore[misc, arg-type]
            name=udf_name,
            session=self._session,
            is_permanent=True,
            stage_location=f"@{temp_stage_name}",
            replace=True,
        )
        def minus_one(x: int) -> int:
            return x - 1

        fs = self._create_feature_store()
        entity = Entity("foo", ["name"])
        fs.register_entity(entity)

        df = self._session.table(self._mock_table).select(call_udf(udf_name, col("id")).alias("uid"), "name")
        fv = FeatureView(name="fv", entities=[entity], feature_df=df, refresh_freq="1h")

        with self.assertWarnsRegex(UserWarning, "Your pipeline won't be incrementally refreshed due to:"):
            fs.register_feature_view(feature_view=fv, version="V1")

    def test_switch_warehouse(self) -> None:
        warehouse = "REGTEST_ML_SMALL"
        current_schema = create_random_schema(self._session, "FS_TEST")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=warehouse,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        original_warehouse = self._session.get_current_warehouse()
        self.assertNotEqual(warehouse, original_warehouse)

        e = Entity("foo", ["name"])
        fs.register_entity(e)
        self.assertEqual(self._session.get_current_warehouse(), original_warehouse)

        fv = FeatureView(
            name="fv0",
            entities=[e],
            feature_df=self._session.sql(f"SELECT name, id, ts FROM {self._mock_table}"),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
            desc="my_new_fv0",
        )
        fv = fs.register_feature_view(feature_view=fv, version="v1")
        self.assertEqual(warehouse, fv.warehouse)
        self.assertEqual(self._session.get_current_warehouse(), original_warehouse)

    def test_update_static_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("FOO", ["id"])
        fs.register_entity(e)

        sql = f"SELECT id, name, title FROM {self._mock_table}"
        fv = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql),
            desc="old desc",
        )
        fv = fs.register_feature_view(feature_view=fv, version="v1")
        with self.assertRaisesRegex(
            RuntimeError, "Static feature view '.*' does not support refresh_freq and warehouse."
        ):
            fs.update_feature_view("fv1", "v1", refresh_freq="1 minute")

        with self.assertRaisesRegex(
            RuntimeError, "Static feature view '.*' does not support refresh_freq and warehouse."
        ):
            fs.update_feature_view("fv1", "v1", warehouse=self._session.get_current_warehouse())

        updated_fv = fs.update_feature_view("fv1", "v1", desc="")
        self.assertEqual(updated_fv.desc, "")

    def test_update_managed_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("FOO", ["id"])
        fs.register_entity(e)

        sql = f"SELECT id, name, title FROM {self._mock_table}"
        alternative_wh = "REGTEST_ML_SMALL"
        old_desc = "this is old desc"
        fv = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="DOWNSTREAM",
            desc=old_desc,
        )

        fv = fs.register_feature_view(feature_view=fv, version="v1")

        def check_fv_properties(
            name: str,
            version: str,
            expected_refresh_freq: str,
            expected_warehouse: str,
            expected_desc: str,
        ) -> None:
            local_fv = fs.get_feature_view(name, version)
            self.assertEqual(expected_refresh_freq, local_fv.refresh_freq)
            self.assertEqual(SqlIdentifier(expected_warehouse), local_fv.warehouse)
            self.assertEqual(expected_desc, local_fv.desc)

        check_fv_properties("fv1", "v1", "DOWNSTREAM", self._session.get_current_warehouse(), old_desc)

        fs.update_feature_view("fv1", "v1", refresh_freq="1 minute", warehouse=alternative_wh)
        check_fv_properties("fv1", "v1", "1 minute", alternative_wh, old_desc)

        fs.update_feature_view("fv1", "v1", refresh_freq="2 minute")
        check_fv_properties("fv1", "v1", "2 minutes", alternative_wh, old_desc)

        fs.update_feature_view("fv1", "v1", warehouse=self._session.get_current_warehouse())
        check_fv_properties("fv1", "v1", "2 minutes", self._session.get_current_warehouse(), old_desc)
        self.assertNotEqual(self._session.get_current_warehouse(), alternative_wh)

        new_desc = "that is NEW desc"
        fs.update_feature_view("fv1", "v1", desc=new_desc)
        check_fv_properties("fv1", "v1", "2 minutes", self._session.get_current_warehouse(), new_desc)

        empty_desc = ""
        fs.update_feature_view("fv1", "v1", desc=empty_desc)
        check_fv_properties("fv1", "v1", "2 minutes", self._session.get_current_warehouse(), empty_desc)

        # TODO(@ewezhou): add cron test

    def test_replace_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        def create_fvs(fs: FeatureStore, sql: str, overwrite: bool) -> Tuple[FeatureView, FeatureView, FeatureView]:
            fv1 = FeatureView(
                name="fv1",
                entities=[e],
                feature_df=self._session.sql(sql),
                refresh_freq="1m",
            )
            fv1 = fs.register_feature_view(feature_view=fv1, version="v1", overwrite=overwrite)

            fv2 = FeatureView(
                name="fv2",
                entities=[e],
                feature_df=self._session.sql(sql),
                refresh_freq="* * * * * America/Los_Angeles",
            )
            fv2 = fs.register_feature_view(feature_view=fv2, version="v2", overwrite=overwrite)

            fv3 = FeatureView(
                name="fv3",
                entities=[e],
                feature_df=self._session.sql(sql),
            )
            fv3 = fs.register_feature_view(feature_view=fv3, version="v3", overwrite=overwrite)

            return fv1, fv2, fv3

        sql = f"SELECT id, name FROM {self._mock_table}"
        fv1, fv2, fv3 = create_fvs(fs, sql, False)
        compare_dataframe(
            actual_df=fs.read_feature_view(fv1).to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
            },
            sort_cols=["ID"],
        )

        # Replace existing feature views
        sql = f"SELECT id, name, title FROM {self._mock_table}"
        fv1, fv2, fv3 = create_fvs(fs, sql, True)

        compare_dataframe(
            actual_df=fs.read_feature_view(fv1).to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
            },
            sort_cols=["ID"],
        )

        # Replace non-existing feature view
        non_existing_fv = FeatureView(
            name="non_existing_fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            refresh_freq="1m",
        )
        fs.register_feature_view(feature_view=non_existing_fv, version="v1", overwrite=True)

    def test_generate_dataset_point_in_time_join(self) -> None:
        fs = self._create_feature_store()

        # Below test requires ASOF join is activated.
        # When _is_asof_join_enabled is false, the alternative union-window-join is used.
        # It's a known issue that union-window-join will fail below test.
        self.assertTrue(fs._is_asof_join_enabled())

        entity = Entity("CUSTOMER", ["CUSTOMER_ID"])
        fs.register_entity(entity)

        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {fs._config.full_schema_path}.CUSTOMER_FEATURES (
                CUSTOMER_ID NUMBER,
                FEATURE_TS TIMESTAMP_NTZ,
                CUST_AVG_AMOUNT_7 NUMBER,
                CUST_AVG_AMOUNT_30 NUMBER)
        """
        ).collect()

        # Each customer_id has 2 rows with different timestamps.
        # This feature value have 2 features: CUST_AVG_AMOUNT_7 and CUST_AVG_AMOUNT_30.
        # Each customer_id has null or non-null values for these two features.
        # generate_dataset() should always return second row as result, regardless of
        # whether the feature value is null or non-null.
        self._session.sql(
            f"""
            INSERT INTO {fs._config.full_schema_path}.CUSTOMER_FEATURES
                (CUSTOMER_ID, FEATURE_TS, CUST_AVG_AMOUNT_7, CUST_AVG_AMOUNT_30)
            VALUES
                (1, '2019-04-01', 1,    1),
                (1, '2019-04-02', 10,   10),
                (2, '2019-04-01', 2,    2),
                (2, '2019-04-02', 20,   null),
                (3, '2019-04-01', 3,    3),
                (3, '2019-04-02', null, 30),
                (4, '2019-04-01', null, 4),
                (4, '2019-04-02', 40,   40),
                (5, '2019-04-01', 5,    5),
                (5, '2019-04-02', null, null)
        """
        ).collect()

        customers_fv = FeatureView(
            name="CUSTOMER_FV",
            entities=[entity],
            feature_df=self._session.sql(f"SELECT * FROM {fs._config.full_schema_path}.CUSTOMER_FEATURES"),
            timestamp_col="FEATURE_TS",
            refresh_freq=None,
        )

        customers_fv = fs.register_feature_view(feature_view=customers_fv, version="V1")

        spine_df = self._session.create_dataframe(
            [
                (1, "2019-04-03"),
                (2, "2019-04-03"),
                (3, "2019-04-03"),
                (4, "2019-04-03"),
                (5, "2019-04-03"),
            ],
            schema=["CUSTOMER_ID", "EVENT_TS"],
        )

        dataset = fs.generate_dataset(
            spine_df=spine_df,
            features=[customers_fv],
            name="customer_frad_training_data",
            spine_timestamp_col="EVENT_TS",
            spine_label_cols=[],
            include_feature_view_timestamp_col=True,
            output_type="table",
        )
        actual_df = dataset.to_pandas()
        actual_df["CUSTOMER_FV_V1_FEATURE_TS"] = actual_df["CUSTOMER_FV_V1_FEATURE_TS"].dt.date

        # CUST_AVG_AMOUNT_7 and CUST_AVG_AMOUNT_30 are expected to be same as the values
        # in second row of each customer_id (with timestamp 2019-04-02).
        compare_dataframe(
            actual_df=actual_df,
            target_data={
                "CUSTOMER_ID": [1, 2, 3, 4, 5],
                "EVENT_TS": ["2019-04-03", "2019-04-03", "2019-04-03", "2019-04-03", "2019-04-03"],
                "CUSTOMER_FV_V1_FEATURE_TS": [
                    datetime.date(2019, 4, 2),
                    datetime.date(2019, 4, 2),
                    datetime.date(2019, 4, 2),
                    datetime.date(2019, 4, 2),
                    datetime.date(2019, 4, 2),
                ],
                "CUST_AVG_AMOUNT_7": [10, 20, None, 40, None],
                "CUST_AVG_AMOUNT_30": [10, None, 30, 40, None],
            },
            sort_cols=["CUSTOMER_ID"],
        )

    def test_cross_feature_store_interop(self) -> None:
        # create first feature store and register feature views
        first_fs = self._create_feature_store()

        first_entity = Entity("foo", ["id"])
        first_fs.register_entity(first_entity)
        first_fv = FeatureView(
            name="fv",
            entities=[first_entity],
            feature_df=self._session.table(self._mock_table).select(["NAME", "ID", "AGE", "TS"]),
            timestamp_col="ts",
            desc="foobar",
        )
        first_fv = first_fs.register_feature_view(feature_view=first_fv, version="v1")

        # create second feature store and register feature views
        second_fs = self._create_feature_store()

        second_entity = Entity("foo", ["id"])
        second_fs.register_entity(second_entity)
        second_fv = FeatureView(
            name="fv",
            entities=[second_entity],
            feature_df=self._session.table(self._mock_table).select(["ID", "DEPT", "TITLE", "TS"]),
            timestamp_col="ts",
            desc="foobar",
        )
        second_fv = second_fs.register_feature_view(feature_view=second_fv, version="v1")

        # make sure these two feature views are in different feature store
        self.assertNotEqual(first_fv.schema, second_fv.schema)

        # generate dataset by joining feature views from different feature store
        spine_df = self._session.create_dataframe([(1, 101)], schema=["id", "ts"])
        for fs in [first_fs, second_fs]:
            ds = fs.generate_dataset(
                spine_df=spine_df,
                features=[first_fv, second_fv],
                spine_timestamp_col="ts",
                name="test_ds",
                output_type="table",
            )
            compare_dataframe(
                actual_df=ds.to_pandas(),
                target_data={
                    "ID": [1],
                    "TS": [101],
                    "NAME": ["jonh"],
                    "AGE": [20],
                    "DEPT": ["sales"],
                    "TITLE": ["boss"],
                },
                sort_cols=["ID"],
            )

    def test_generate_dataset_left_join(self) -> None:
        # testing case for join features without timestamp, which is a left join with the spine
        fs = self._create_feature_store()

        e1 = Entity("foo", ["id", "name"])
        fs.register_entity(e1)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e1],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1")

        e2 = Entity("bar", ["id"])
        fs.register_entity(e2)

        sql2 = f"SELECT id, age FROM {self._mock_table}"
        fv2 = FeatureView(
            name='"FvfV2"',
            entities=[e2],
            feature_df=self._session.sql(sql2),
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1")
        spine_df = self._session.create_dataframe([(1, "jonh"), (2, "porter"), (3, "johnny")], schema=["id", "name"])

        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            name="test_ds",
            output_type="table",
        )

        compare_dataframe(
            actual_df=ds.to_pandas(),
            target_data={
                "ID": [1, 2, 3],
                "NAME": ["jonh", "porter", "johnny"],
                "TITLE": ["boss", "manager", None],
                "AGE": [20, 30, None],
            },
            sort_cols=["ID"],
        )

    def test_attach_feature_desc(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"""SELECT NAME AS "NaMe", ID AS ID, TITLE AS "title", AGE FROM {self._mock_table}"""
        feature_desc = {'"NaMe"': "my name", '"title"': '"my title"', "AGE": "my age"}

        fv = FeatureView(
            name="fv",
            entities=[e],
            feature_df=self._session.sql(sql),
            desc="foobar",
        ).attach_feature_desc(feature_desc)
        fv = fs.register_feature_view(feature_view=fv, version="v1")

        fv = fs.get_feature_view(name="fv", version="v1")
        self.assertEqual(feature_desc, fv.feature_descs)

        with self.assertRaisesRegex(ValueError, ".*not found in FeatureView.*"):
            FeatureView(
                name="fv",
                entities=[e],
                feature_df=self._session.sql(sql),
                desc="foobar",
            ).attach_feature_desc({"ID": "my id"})

        with self.assertRaisesRegex(ValueError, ".*not found in FeatureView.*"):
            FeatureView(
                name="fv",
                entities=[e],
                feature_df=self._session.sql(sql),
                desc="foobar",
            ).attach_feature_desc({"NAME": "my name"})

    def test_multi_fv_asof_join_correctness(self) -> None:
        """
        Cases covered by this test:
            1. spine ts <= fv ts
            2. spine ts > fv ts
            3. fv ts is NULL
            4. id not exist in fv
            5. fv without ts
            6. fv with multiple entities
        """

        def create_point_in_time_test_tables(full_schema_path: str) -> Tuple[str, str, str]:
            table_a = f"{full_schema_path}.A"
            self._session.sql(
                f"""
                    CREATE TABLE IF NOT EXISTS {table_a}
                    (id1 INT, a INT, ts TIMESTAMP)
                """
            ).collect()
            self._session.sql(
                f"""
                INSERT OVERWRITE INTO {table_a} (id1, a, ts) VALUES
                (-1, 0, '2024-01-01'),
                (1, 10, '2024-01-01'),
                (1, 11, '2024-01-03'),
                (2, 20, '2024-01-01')
            """
            ).collect()

            table_b = f"{full_schema_path}.B"
            self._session.sql(
                f"""
                    CREATE TABLE IF NOT EXISTS {table_b}
                    (id1 INT, id2 INT, b INT, ts TIMESTAMP)
                """
            ).collect()
            self._session.sql(
                f"""
                INSERT OVERWRITE INTO {table_b} (id1, id2, b, ts) VALUES
                (1, 10, 10, '2024-01-02'),
                (2, 20, 20, NULL),
                (3, 40, 30, '2024-01-02'),
                (3, 30, 30, '2024-01-04')
            """
            ).collect()

            table_c = f"{full_schema_path}.C"
            self._session.sql(
                f"""
                    CREATE TABLE IF NOT EXISTS {table_c}
                    (id2 INT, c INT)
                """
            ).collect()
            self._session.sql(
                f"""
                INSERT OVERWRITE INTO {table_c} (id2, c) VALUES
                (-1, 0),
                (10, 1),
                (20, 2),
                (40, 4)
            """
            ).collect()
            return table_a, table_b, table_c

        fs = self._create_feature_store()

        self.assertTrue(fs._is_asof_join_enabled())
        a, b, c = create_point_in_time_test_tables(fs._config.full_schema_path)

        e1 = Entity("foo", join_keys=["id1"])
        fs.register_entity(e1)
        e2 = Entity("bar", join_keys=["id2"])
        fs.register_entity(e2)

        fv1 = FeatureView(
            name="fv1",
            entities=[e1],
            feature_df=self._session.sql(f"select * from {a}"),
            timestamp_col="ts",
            refresh_freq="60m",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1")

        fv2 = FeatureView(
            name="fv2",
            entities=[e1, e2],
            feature_df=self._session.sql(f"select * from {b}"),
            timestamp_col="ts",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1")

        fv3 = FeatureView(
            name="fv3",
            entities=[e2],
            feature_df=self._session.sql(f"select * from {c}"),
        )
        fv3 = fs.register_feature_view(feature_view=fv3, version="v1")

        spine_df = self._session.create_dataframe(
            [
                (1, 10, "2024-01-01", 100),
                (1, 10, "2024-01-02", 100),
                (2, 20, "2024-01-02", 200),
                (3, 30, "2024-01-03", 300),
                (4, 40, "2024-01-03", 400),
                (5, 50, "2024-01-01", 500),
            ],
            schema=["id1", "id2", "ts", "label"],
        )
        ds = fs.generate_dataset(
            name="my_ds",
            spine_df=spine_df,
            features=[fv1, fv2, fv3],
            version="v1",
            spine_timestamp_col="ts",
            spine_label_cols=["label"],
            include_feature_view_timestamp_col=True,
        )
        actual_df = ds.read.to_pandas()

        compare_dataframe(
            actual_df=actual_df,
            target_data={
                "ID1": [1, 1, 2, 3, 4, 5],
                "ID2": [10, 10, 20, 30, 40, 50],
                "TS": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-01"],
                "LABEL": [100, 100, 200, 300, 400, 500],
                "FV1_v1_TS": [
                    datetime.datetime(2024, 1, 1, 0, 0, 0),
                    datetime.datetime(2024, 1, 1, 0, 0, 0),
                    datetime.datetime(2024, 1, 1, 0, 0, 0),
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                ],
                "A": [10, 10, 20, None, None, None],
                "FV2_v1_TS": [pd.NaT, datetime.datetime(2024, 1, 2, 0, 0, 0), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "B": [None, 10, None, None, None, None],
                "C": [1, 1, 2, None, 4, None],
            },
            sort_cols=["ID1", "TS"],
        )

    @parameterized.parameters(
        [
            "SELECT * FROM TABLE(RESULT_SCAN())",
            "SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))",
            """SELECT * FROM TABLE(
                    RESULT_SCAN(
                    LAST_QUERY_ID()
                )
        )""",
            """SELECT * FROM TABLE       (
            RESULT_SCAN(


                    LAST_QUERY_ID(   )
                )
        )""",
        ]
    )  # type: ignore[misc]
    def test_invalid_result_scan_query(self, query: str) -> None:
        fs = self._create_feature_store()
        self._session.sql(f"create table {fs._config.full_schema_path}.a(a int)").collect()
        self._session.sql(f"select * from {fs._config.full_schema_path}.a").collect()

        e = Entity(name="e", join_keys=["a"])

        with self.assertRaisesRegex(ValueError, ".*reading from RESULT_SCAN.*"):
            FeatureView(name="foo", entities=[e], feature_df=self._session.sql(query))

    def test_invalid_argument_type(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            refresh_freq="DOWNSTREAM",
        )

        fs.register_feature_view(feature_view=fv1, version="v1")

        with self.assertRaisesRegex(
            ValueError, "Invalid type of argument feature_view. It must be either str or FeatureView type"
        ):
            fs.read_feature_view(123, "v1")  # type: ignore[call-overload]


if __name__ == "__main__":
    absltest.main()
