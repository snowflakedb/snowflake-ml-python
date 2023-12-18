from typing import Optional
from uuid import uuid4

from absl.testing import absltest
from common_utils import (
    FS_INTEG_TEST_DATASET_SCHEMA,
    FS_INTEG_TEST_DB,
    FS_INTEG_TEST_DUMMY_DB,
    compare_dataframe,
    compare_feature_views,
    create_mock_session,
    create_random_schema,
    get_test_warehouse_name,
)

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.dataset.dataset import Dataset
from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
    FeatureViewStatus,
)
from snowflake.ml.feature_store.feature_store import (
    _ENTITY_TAG_PREFIX,
    _FEATURE_STORE_OBJECT_TAG,
    _FEATURE_VIEW_ENTITY_TAG,
    _FEATURE_VIEW_TS_COL_TAG,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session, exceptions as snowpark_exceptions
from snowflake.snowpark.functions import call_udf, col, udf


class FeatureStoreTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._active_feature_store = []

        try:
            self._session.sql(f"CREATE DATABASE IF NOT EXISTS {FS_INTEG_TEST_DUMMY_DB}").collect()
            self._session.sql(f"CREATE DATABASE IF NOT EXISTS {FS_INTEG_TEST_DB}").collect()
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
                fs.clear()
            except RuntimeError as e:
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
            self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        # Intentionally point session to a different database to make sure feature store code is resilient to
        # session location.
        self._session.use_database(FS_INTEG_TEST_DUMMY_DB)
        return fs

    def test_fail_if_not_exist(self) -> None:
        name = f"foo_{uuid4().hex.upper()}"
        with self.assertRaisesRegex(ValueError, "Feature store .* does not exist."):
            FeatureStore(
                session=self._session,
                database=FS_INTEG_TEST_DB,
                name=name,
                default_warehouse=self._test_warehouse_name,
            )
        self._create_feature_store(name)
        fs = FeatureStore(
            session=self._session,
            database=FS_INTEG_TEST_DB,
            name=name,
            default_warehouse=self._test_warehouse_name,
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

    def test_create_if_not_exist_system_error(self) -> None:
        mock_session = create_mock_session(
            "CREATE DATABASE IF NOT EXISTS",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to create feature store .*"):
            FeatureStore(
                session=mock_session,
                database=FS_INTEG_TEST_DB,
                name="foo",
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )

    def test_clear_feature_store_system_error(self) -> None:
        fs = self._create_feature_store()

        original_session = fs._session
        fs._session = create_mock_session(
            "DROP TAG",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        with self.assertRaisesRegex(RuntimeError, "Failed to clear feature store"):
            fs.clear()
        fs._session = original_session

    def test_create_and_delete_entities(self) -> None:
        fs = self._create_feature_store()

        entities = {
            "User": Entity("USER", ["uid"]),
            "Ad": Entity('"aD"', ["aid"]),
            "Product": Entity("Product", ["pid", "cid"]),
        }

        # create new entities
        for e in entities.values():
            fs.register_entity(e)

        compare_dataframe(
            actual_df=fs.list_entities().to_pandas(),
            target_data={
                "NAME": ["aD", "PRODUCT", "USER"],
                "JOIN_KEYS": ["AID", "PID,CID", "UID"],
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

        compare_dataframe(
            actual_df=fs.list_entities().to_pandas(),
            target_data={
                "NAME": ["PRODUCT", "USER"],
                "JOIN_KEYS": ["PID,CID", "UID"],
                "DESC": ["", ""],
            },
            sort_cols=["NAME"],
        )

        # create entity already exists
        with self.assertRaisesRegex(ValueError, "Entity.*already exists.*"):
            fs.register_entity(Entity("User", ["a", "b", "c"]))
        # captitalized entity name is treated the same
        with self.assertRaisesRegex(ValueError, "Entity.*already exists.*"):
            fs.register_entity(Entity("USER", ["a", "b", "c"]))

        # test delete entity failure with active feature views
        # create a new feature view
        sql = f"SELECT name, id AS uid FROM {self._mock_table}"
        fv = FeatureView(name="fv", entities=[entities["User"]], feature_df=self._session.sql(sql), refresh_freq="1m")
        fs.register_feature_view(feature_view=fv, version="FIRST")
        with self.assertRaisesRegex(ValueError, "Cannot delete Entity .* due to active FeatureViews.*"):
            fs.delete_entity("User")

    def test_retrieve_entity(self) -> None:
        fs = self._create_feature_store()

        e1 = Entity(name="foo", join_keys=["a", "b"], desc="my foo")
        e2 = Entity(name="bar", join_keys=["c"])

        fs.register_entity(e1)
        fs.register_entity(e2)

        self.assertEqual(e1, fs.get_entity("foo"))
        self.assertEqual(e2, fs.get_entity("bar"))

        compare_dataframe(
            actual_df=fs.list_entities().to_pandas(),
            target_data={
                "NAME": ["FOO", "BAR"],
                "JOIN_KEYS": ["A,B", "C"],
                "DESC": ["my foo", ""],
            },
            sort_cols=["NAME"],
        )

    def test_get_entity_system_error(self) -> None:
        fs = self._create_feature_store()
        fs._session = create_mock_session(
            "SHOW TAGS LIKE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to find object .*"):
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

    def test_register_feature_view_as_view(self) -> None:
        """
        APIs covered by test:
            1. register_feature_view
            2. read_feature_view
            3. list_feature_views
            4. get_feature_view
            5. delete_feature_view
            6. generate_dataset (covers retrieve_feature_values)
            7. merge_features
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
        fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(fv, fs.get_feature_view("fv", "v1"))

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
        compare_feature_views(fs.list_feature_views(as_dataframe=False), [fv])
        compare_feature_views(fs.list_feature_views(entity_name="FOO", as_dataframe=False), [fv])

        # create another feature view
        new_fv = FeatureView(
            name="new_fv",
            entities=[e],
            feature_df=self._session.table(self._mock_table).select(["ID", "DEPT", "TS"]),
            timestamp_col="ts",
            desc="foobar",
        )
        new_fv = fs.register_feature_view(feature_view=new_fv, version="V1")
        compare_feature_views(fs.list_feature_views(as_dataframe=False), [fv, new_fv])

        # create a merged feature view
        merged_fv = fs.merge_features(features=[fv.slice(["NAME", "AGE"]), new_fv], name="merged_fv")
        merged_fv = fs.register_feature_view(feature_view=merged_fv, version="v1")
        df = fs.read_feature_view(merged_fv)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "AGE": [20, 30],
                "TS": [100, 200],
                "DEPT": ["sales", "engineer"],
            },
            sort_cols=["ID"],
        )
        compare_feature_views(fs.list_feature_views(as_dataframe=False), [fv, merged_fv, new_fv])

        # generate data on multiple feature views
        spine_df = self._session.create_dataframe([(1, 101)], schema=["id", "ts"])
        ds = fs.generate_dataset(spine_df=spine_df, features=[fv, new_fv], spine_timestamp_col="ts")
        compare_dataframe(
            actual_df=ds.df.to_pandas(),
            target_data={
                "ID": [1],
                "TS": [101],
                "NAME": ["jonh"],
                "TITLE": ["boss"],
                "AGE": [20],
                "DEPT": ["sales"],
            },
            sort_cols=["ID"],
        )

        # delete a feature view
        fs.delete_feature_view(merged_fv)
        compare_feature_views(fs.list_feature_views(as_dataframe=False), [fv, new_fv])

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
        self.assertEqual(fv0.status, FeatureViewStatus.RUNNING)
        self.assertEqual(fv0.refresh_freq, "1 minute")
        self.assertEqual(fv0, fs.get_feature_view("fv0", "FIRST"))

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

        compare_feature_views(fs.list_feature_views(as_dataframe=False), [fv0, new_fv0, fv1])

        # delete feature view
        with self.assertRaisesRegex(ValueError, "FeatureView .* has not been registered."):
            unmaterialized_fv = FeatureView("unmaterialized", [e1], self._session.sql(sql0))
            fs.delete_feature_view(unmaterialized_fv)

        fs.delete_feature_view(fs.get_feature_view("FV0", "FIRST"))

        compare_feature_views(fs.list_feature_views(as_dataframe=False), [new_fv0, fv1])

        # test get feature view obj
        fv = fs.get_feature_view(name="fv1", version="FIRST")
        self.assertEqual(fv.name, "FV1")
        self.assertEqual(fv.version, "FIRST")
        self.assertEqual(fv.query, sql0)
        self.assertEqual(fv.status, FeatureViewStatus.RUNNING)
        self.assertEqual(fv.refresh_freq, "5 minutes")
        self.assertEqual(fv.warehouse, alternate_warehouse)
        self.assertEqual(fv.desc, "my_fv1")
        self.assertEqual(fv.timestamp_col, None)

        fv = fs.get_feature_view(name="fv0", version="SECOND")
        self.assertEqual(fv.timestamp_col.upper(), "TS")

    def test_create_duplicated_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"SELECT * FROM {self._mock_table}"
        fv = FeatureView(name="fv", entities=[e], feature_df=self._session.sql(sql), refresh_freq="1m")
        fv = fs.register_feature_view(feature_view=fv, version="v1")

        fv = FeatureView(name="fv", entities=[e], feature_df=self._session.sql(sql), refresh_freq="1m")
        with self.assertRaisesRegex(ValueError, "FeatureView .* already exists."):
            fv = fs.register_feature_view(feature_view=fv, version="v1")

    def test_resume_and_suspend_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        my_fv = FeatureView(
            name="my_fv", entities=[e], feature_df=self._session.table(self._mock_table), refresh_freq="DOWNSTREAM"
        )
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1", block=True)

        with self.assertRaisesRegex(ValueError, "FeatureView.*is not in suspended status.*"):
            fs.resume_feature_view(my_fv)

        my_fv = fs.suspend_feature_view(my_fv)

        with self.assertRaisesRegex(ValueError, "FeatureView.*is not in running status.*"):
            fs.suspend_feature_view(my_fv)

        my_fv = fs.resume_feature_view(my_fv)
        self.assertEqual(my_fv.status, FeatureViewStatus.RUNNING)

    def test_resume_and_suspend_feature_view_system_error(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)
        my_fv = FeatureView(
            name="my_fv", entities=[e], feature_df=self._session.table(self._mock_table), refresh_freq="DOWNSTREAM"
        )
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1", block=True)

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

    def test_read_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"SELECT name, id, title, age, ts FROM {self._mock_table}"
        my_fv = FeatureView(name="my_fv", entities=[e], feature_df=self._session.sql(sql), refresh_freq="DOWNSTREAM")
        my_fv = fs.register_feature_view(feature_view=my_fv, version="v1", block=True)

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

    def test_register_with_cron_expr(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql = f"SELECT name, id, title, age, ts FROM {self._mock_table}"
        my_fv = FeatureView(
            name="my_fv", entities=[e], feature_df=self._session.sql(sql), refresh_freq="* * * * * America/Los_Angeles"
        ).attach_feature_desc({"title": "my title"})
        my_fv = fs.register_feature_view(
            feature_view=my_fv,
            version="v1",
            block=True,
        )
        fv = fs.get_feature_view("my_fv", "v1")
        self.assertEqual(my_fv, fv)

        task_name = fv.physical_name()
        res = self._session.sql(f"SHOW TASKS LIKE '{task_name}' IN SCHEMA {fs._config.full_schema_path}").collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["state"], "started")
        self.assertEqual(fv.refresh_freq, "DOWNSTREAM")

        fs.suspend_feature_view(fv)
        res = self._session.sql(f"SHOW TASKS LIKE '{task_name}' IN SCHEMA {fs._config.full_schema_path}").collect()
        self.assertEqual(res[0]["state"], "suspended")

        fs.resume_feature_view(fv)
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
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        sql2 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name="fv2",
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1", block=True)

        sql3 = f"SELECT id, dept FROM {self._mock_table}"
        fv3 = FeatureView(name="fv3", entities=[e], feature_df=self._session.sql(sql3), refresh_freq="DOWNSTREAM")
        fv3 = fs.register_feature_view(feature_view=fv3, version="v1", block=True)

        spine_df = self._session.create_dataframe([(1, 101), (2, 202), (1, 90)], schema=["id", "ts"])
        df = fs.retrieve_feature_values(
            spine_df=spine_df,
            features=[fv1, fv2, fv3],
            spine_timestamp_col="ts",
        )
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 1, 2],
                "TS": [90, 101, 202],
                "NAME": [None, "jonh", "porter"],
                "TITLE": [None, "boss", "manager"],
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
        fv1 = FeatureView(name="fv1", entities=[e], feature_df=self._session.sql(sql1), refresh_freq="DOWNSTREAM")
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        sql2 = f"SELECT id, age FROM {self._mock_table}"
        fv2 = FeatureView(name="fv2", entities=[e], feature_df=self._session.sql(sql2), refresh_freq="DOWNSTREAM")
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1", block=True)

        spine_df = self._session.create_dataframe([(1), (2)], schema=["id"])
        df = fs.retrieve_feature_values(spine_df=spine_df, features=[fv1, fv2])

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

        df = fs.retrieve_feature_values(spine_df=spine_df, features=[fv1.slice(["name"]), fv2])
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
            spine_df=spine_df, features=[fv1.slice(["name"]), fv2], exclude_columns=["NAME"]
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
        dataset = fs.generate_dataset(spine_df, features=[fv1_slice, fv2])
        df = fs.retrieve_feature_values(spine_df=spine_df, features=dataset.load_features())
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

    def test_invalid_merge_features(self) -> None:
        fs = self._create_feature_store()

        e1 = Entity("foo", ["id"])
        e2 = Entity("bar", ["id"])
        fs.register_entity(e1)
        fs.register_entity(e2)

        # 1. unmaterialized FV
        sql1 = f"SELECT id, name, ts FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e1],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
        )

        sql2 = f"SELECT id, title, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name="fv2",
            entities=[e1],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )

        with self.assertRaisesRegex(ValueError, "FeatureView FV1 has not been registered."):
            fs.merge_features(features=[fv1, fv2], name="merged_fv")

        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)
        with self.assertRaisesRegex(ValueError, "FeatureView FV2 has not been registered."):
            fs.merge_features(features=[fv1, fv2], name="merged_fv")

        # 2. Different Entity
        sql3 = f"SELECT id, title, ts FROM {self._mock_table}"
        fv3 = FeatureView(
            name="fv3",
            entities=[e2],
            feature_df=self._session.sql(sql3),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv3 = fs.register_feature_view(feature_view=fv3, version="v1", block=True)
        with self.assertRaisesRegex(ValueError, "Cannot merge FeatureView .* with different Entities.*"):
            fs.merge_features(features=[fv1, fv3], name="merged_fv")

        # 3. Different timestamp col
        sql4 = f"SELECT id, title, age FROM {self._mock_table}"
        fv4 = FeatureView(
            name="fv4",
            entities=[e1],
            feature_df=self._session.sql(sql4),
            timestamp_col="age",
            refresh_freq="DOWNSTREAM",
        )
        fv4 = fs.register_feature_view(feature_view=fv4, version="v1", block=True)
        with self.assertRaisesRegex(ValueError, "Cannot merge FeatureView .* with different timestamp_col.*"):
            fs.merge_features(features=[fv1, fv4], name="merged_fv")

        # 4. Incorrect size
        with self.assertRaisesRegex(ValueError, "features should have at least two entries"):
            fs.merge_features(features=[], name="foo")

        with self.assertRaisesRegex(ValueError, "features should have at least two entries"):
            fs.merge_features(features=[fv1], name="foo")

    def test_merge_features(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, ts FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1",
            entities=[e],
            feature_df=self._session.sql(sql1),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        sql2 = f"SELECT id, title, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name="fv2",
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1", block=True)

        sql3 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv3 = FeatureView(
            name="fv3",
            entities=[e],
            feature_df=self._session.sql(sql3),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv3 = fs.register_feature_view(feature_view=fv3, version="v1", block=True)

        merged_fv = fs.merge_features(features=[fv1, fv2, fv3], name="merged_fv")
        merged_fv = fs.register_feature_view(feature_view=merged_fv, version="v1", block=True)

        df = fs.read_feature_view(merged_fv)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "TS": [100, 200],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["ID"],
        )

    def test_merge_feature_view_slice(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        # 1. Right side is FeatureViewSlice
        sql1 = f"SELECT id, name FROM {self._mock_table}"
        fv1 = FeatureView(name="fv1", entities=[e], feature_df=self._session.sql(sql1), refresh_freq="DOWNSTREAM")
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        sql2 = f"SELECT id, title, age FROM {self._mock_table}"
        fv2 = FeatureView(name="fv2", entities=[e], feature_df=self._session.sql(sql2), refresh_freq="DOWNSTREAM")
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1", block=True)

        merged_fv = fs.merge_features(features=[fv1, fv2.slice(["title"])], name="merged_fv")
        merged_fv = fs.register_feature_view(feature_view=merged_fv, version="v1", block=True)

        df = fs.read_feature_view(merged_fv)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
            },
            sort_cols=["ID"],
        )

        # 2. Left side is FeatureViewSlice
        fv3 = FeatureView(name="fv3", entities=[e], feature_df=self._session.sql(sql2), refresh_freq="DOWNSTREAM")
        fv3 = fs.register_feature_view(feature_view=fv3, version="v1", block=True)

        fv4 = FeatureView(name="fv4", entities=[e], feature_df=self._session.sql(sql1), refresh_freq="DOWNSTREAM")
        fv4 = fs.register_feature_view(feature_view=fv4, version="v1", block=True)

        merged_fv_2 = fs.merge_features(features=[fv3.slice(["title"]), fv4], name="merged_fv_2")
        merged_fv_2 = fs.register_feature_view(
            feature_view=merged_fv_2,
            version="v1",
            block=True,
        )

        df = fs.read_feature_view(merged_fv_2)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "TITLE": ["boss", "manager"],
                "NAME": ["jonh", "porter"],
            },
            sort_cols=["ID"],
        )

        # 3. FeatureViewSlice with ts
        sql5 = f"SELECT id, name, ts FROM {self._mock_table}"
        fv5 = FeatureView(
            name="fv5",
            entities=[e],
            feature_df=self._session.sql(sql5),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv5 = fs.register_feature_view(feature_view=fv5, version="v1", block=True)

        sql6 = f"SELECT id, title, age, ts FROM {self._mock_table}"
        fv6 = FeatureView(
            name="fv6",
            entities=[e],
            feature_df=self._session.sql(sql6),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv6 = fs.register_feature_view(feature_view=fv6, version="v1", block=True)

        merged_fv_3 = fs.merge_features(features=[fv5, fv6.slice(["title"])], name="merged_fv_3")
        merged_fv_3 = fs.register_feature_view(
            feature_view=merged_fv_3,
            version="v1",
            block=True,
        )

        df = fs.read_feature_view(merged_fv_3)
        compare_dataframe(
            actual_df=df.to_pandas(),
            target_data={
                "ID": [1, 2],
                "NAME": ["jonh", "porter"],
                "TS": [100, 200],
                "TITLE": ["boss", "manager"],
            },
            sort_cols=["ID"],
        )

    def test_invalid_load_feature_views_from_dataset(self) -> None:
        fs = self._create_feature_store()
        dataset = Dataset(self._session, self._session.create_dataframe([1, 2, 3], schema=["foo"]))
        with self.assertRaisesRegex(ValueError, "Dataset.*does not contain valid feature view information."):
            fs.load_feature_views_from_dataset(dataset)

    def test_list_feature_views(self) -> None:
        fs = self._create_feature_store()

        e = Entity("foo", ["id"])
        fs.register_entity(e)

        self.assertEqual(fs.list_feature_views(entity_name="foo", as_dataframe=False), [])

        # 1. Right side is FeatureViewSlice
        sql1 = f"SELECT id, name, ts FROM {self._mock_table}"
        fv1 = FeatureView(
            name="fv1", entities=[e], feature_df=self._session.sql(sql1), timestamp_col="ts", refresh_freq="DOWNSTREAM"
        )
        fv1.attach_feature_desc({"name": "this is my name col"})
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        sql2 = f"SELECT id, title, age FROM {self._mock_table}"
        fv2 = FeatureView(name="fv2", entities=[e], feature_df=self._session.sql(sql2), refresh_freq="DOWNSTREAM")
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1", block=True)
        self.assertEqual(
            sorted(fs.list_feature_views(entity_name="Foo", as_dataframe=False), key=lambda fv: fv.name), [fv1, fv2]
        )
        self.assertEqual(
            fs.list_feature_views(entity_name="foo", feature_view_name="fv1", as_dataframe=False),
            [fv1],
        )

        df = fs.list_feature_views()
        self.assertListEqual(
            df.columns,
            [
                "NAME",
                "ENTITIES",
                "TIMESTAMP_COL",
                "DESC",
                "QUERY",
                "VERSION",
                "STATUS",
                "FEATURE_DESC",
                "REFRESH_FREQ",
                "DATABASE",
                "SCHEMA",
                "WAREHOUSE",
                "REFRESH_MODE",
                "REFRESH_MODE_REASON",
                "PHYSICAL_NAME",
            ],
        )
        result = df.collect()
        self.assertEqual(len(result), 2)

    def test_list_feature_views_system_error(self) -> None:
        fs = self._create_feature_store()
        e = Entity("foo", ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name FROM {self._mock_table}"
        fv1 = FeatureView(name="fv1", entities=[e], feature_df=self._session.sql(sql1), refresh_freq="DOWNSTREAM")
        fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        fs._session = create_mock_session(
            "SHOW DYNAMIC TABLES LIKE",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )
        with self.assertRaisesRegex(RuntimeError, "Failed to find object"):
            fs.list_feature_views()

        fs._session = create_mock_session(
            "INFORMATION_SCHEMA.TAG_REFERENCES",
            snowpark_exceptions.SnowparkClientException("Intentional Integ Test Error"),
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to find object"):
            fs.list_feature_views(entity_name="foo")

    def test_create_and_cleanup_tags(self) -> None:
        current_schema = create_random_schema(self._session, "TEST_CREATE_AND_CLEANUP_TAGS")
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self.assertIsNotNone(fs)

        res = self._session.sql(
            f"SHOW TAGS LIKE '{_FEATURE_VIEW_ENTITY_TAG}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(res), 1)

        self._session.sql(f"DROP SCHEMA IF EXISTS {FS_INTEG_TEST_DB}.{current_schema}").collect()

        row_list = self._session.sql(
            f"SHOW TAGS LIKE '{_FEATURE_VIEW_ENTITY_TAG}' IN DATABASE {fs._config.database}"
        ).collect()
        for row in row_list:
            self.assertNotEqual(row["schema_name"], current_schema)

    def test_generate_dataset(self) -> None:
        fs = self._create_feature_store()

        e = Entity('"fOO"', ["id"])
        fs.register_entity(e)

        sql1 = f"SELECT id, name, title FROM {self._mock_table}"
        fv1 = FeatureView(name="fv1", entities=[e], feature_df=self._session.sql(sql1), refresh_freq="DOWNSTREAM")
        fv1 = fs.register_feature_view(feature_view=fv1, version="v1", block=True)

        sql2 = f"SELECT id, age, ts FROM {self._mock_table}"
        fv2 = FeatureView(
            name='"FvfV2"',
            entities=[e],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="DOWNSTREAM",
        )
        fv2 = fs.register_feature_view(feature_view=fv2, version="v1", block=True)
        spine_df = self._session.create_dataframe([(1, 100), (1, 101)], schema=["id", "ts"])

        # Generate dataset the first time
        ds1 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            materialized_table="foobar",
            spine_timestamp_col="ts",
        )
        compare_dataframe(
            actual_df=ds1.df.to_pandas(),
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

        # Re-generate dataset with same source should not cause any duplication
        ds2 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            materialized_table="foobar",
            spine_timestamp_col="ts",
            save_mode="merge",
        )
        compare_dataframe(
            actual_df=ds2.df.to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
                "AGE": [20, 20],
            },
            sort_cols=["ID", "TS"],
        )

        # New data should properly appear
        spine_df = self._session.create_dataframe([(2, 202)], schema=["id", "ts"])
        ds3 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            materialized_table="foobar",
            spine_timestamp_col="ts",
            save_mode="merge",
        )
        compare_dataframe(
            actual_df=ds3.df.to_pandas(),
            target_data={
                "ID": [1, 1, 2],
                "TS": [100, 101, 202],
                "NAME": ["jonh", "jonh", "porter"],
                "TITLE": ["boss", "boss", "manager"],
                "AGE": [20, 20, 30],
            },
            sort_cols=["ID", "TS"],
        )

        # Snapshot should remain the same
        compare_dataframe(
            actual_df=self._session.sql(f"SELECT * FROM {ds1.snapshot_table}").to_pandas(),
            target_data={
                "ID": [1, 1],
                "TS": [100, 101],
                "NAME": ["jonh", "jonh"],
                "TITLE": ["boss", "boss"],
                "AGE": [20, 20],
            },
            sort_cols=["ID", "TS"],
        )
        compare_dataframe(
            actual_df=self._session.sql(f"SELECT * FROM {ds3.snapshot_table}").to_pandas(),
            target_data={
                "ID": [1, 1, 2],
                "TS": [100, 101, 202],
                "NAME": ["jonh", "jonh", "porter"],
                "TITLE": ["boss", "boss", "manager"],
                "AGE": [20, 20, 30],
            },
            sort_cols=["ID", "TS"],
        )

        # Generate dataset with exclude_columns and check both materialization and non-materialization path
        spine_df = self._session.create_dataframe([(1, 101), (2, 202)], schema=["id", "ts"])

        ds4 = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv1, fv2],
            materialized_table="foobar2",
            spine_timestamp_col="ts",
            exclude_columns=["id", "ts"],
        )
        compare_dataframe(
            actual_df=ds4.df.to_pandas(),
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
        )
        compare_dataframe(
            actual_df=ds5.df.to_pandas(),
            target_data={
                "NAME": ["jonh", "porter"],
                "TITLE": ["boss", "manager"],
                "AGE": [20, 30],
            },
            sort_cols=["AGE"],
        )

        # Generate data should fail with errorifexists if table already exist
        with self.assertRaisesRegex(ValueError, "Dataset table .* already exists."):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                materialized_table="foobar",
                spine_timestamp_col="ts",
                save_mode="errorifexists",
            )

        # registered table should fail with invalid char `.`
        with self.assertRaisesRegex(ValueError, "materialized_table .* contains invalid char `.`"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                materialized_table="foo.bar",
                spine_timestamp_col="ts",
                save_mode="errorifexists",
            )

        # invalid columns in exclude_columns should fail
        with self.assertRaisesRegex(ValueError, "FOO in exclude_columns not exists in.*"):
            fs.generate_dataset(
                spine_df=spine_df,
                features=[fv1, fv2],
                materialized_table="foobar3",
                spine_timestamp_col="ts",
                exclude_columns=["foo"],
            )

    def test_clear_feature_store_in_existing_schema(self) -> None:
        current_schema = create_random_schema(self._session, "TEST_CLEAR_FEATURE_STORE_IN_EXISTING_SCHEMA")

        # create some objects outside of feature store domain, later will check if they still exists after fs.clear()
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
            name="fv", entities=[e], feature_df=self._session.sql(sql), refresh_freq="* * * * * America/Los_Angeles"
        )
        fv = fs.register_feature_view(feature_view=fv, version="v1", block=True)

        spine_df = self._session.create_dataframe([(2, 202)], schema=["id", "ts"])
        fs.generate_dataset(
            spine_df=spine_df,
            features=[fv],
            materialized_table="foo_mt",
            spine_timestamp_col="ts",
            save_mode="errorifexists",
        )

        def check_fs_objects(expected_count: int) -> None:
            result = self._session.sql(f"SHOW DYNAMIC TABLES LIKE 'FV$V1' IN SCHEMA {full_schema_path}").collect()
            self.assertEqual(len(result), expected_count)
            result = self._session.sql(f"SHOW TABLES LIKE 'foo_mt' IN SCHEMA {full_schema_path}").collect()
            self.assertEqual(len(result), expected_count)
            result = self._session.sql(f"SHOW TASKS LIKE 'FV$V1' IN SCHEMA {full_schema_path}").collect()
            self.assertEqual(len(result), expected_count)
            expected_tags = [
                _FEATURE_VIEW_ENTITY_TAG,
                _FEATURE_VIEW_TS_COL_TAG,
                _FEATURE_STORE_OBJECT_TAG,
                f"{_ENTITY_TAG_PREFIX}foo",
            ]
            for tag in expected_tags:
                result = self._session.sql(f"SHOW TAGS LIKE '{tag}' in {full_schema_path}").collect()
                self.assertEqual(len(result), expected_count)

        check_fs_objects(1)
        fs.clear()
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

    def test_update_feature_view(self) -> None:
        fs = self._create_feature_store()

        e = Entity("FOO", ["id"])
        fs.register_entity(e)

        sql = f"SELECT id, name, title FROM {self._mock_table}"
        alternative_wh = "REGTEST_ML_SMALL"
        fv = FeatureView(name="fv1", entities=[e], feature_df=self._session.sql(sql), refresh_freq="DOWNSTREAM")
        with self.assertRaisesRegex(
            RuntimeError, "Feature view .* must be registered and non-static to update refresh_freq.*"
        ):
            fv.refresh_freq = "1 minute"

        with self.assertRaisesRegex(
            RuntimeError, "Feature view .* must be registered and non-static to update warehouse.*"
        ):
            fv.warehouse = alternative_wh

        with self.assertRaisesRegex(
            RuntimeError, "Feature view .* must be registered and non-static so that can be updated.*"
        ):
            fs.update_feature_view(fv)

        fv = fs.register_feature_view(feature_view=fv, version="v1", block=True)
        result = self._session.sql(f"SHOW DYNAMIC TABLES IN SCHEMA {fv.database}.{fv.schema}").collect()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["target_lag"], "DOWNSTREAM")
        self.assertEqual(SqlIdentifier(result[0]["warehouse"]), SqlIdentifier(self._session.get_current_warehouse()))

        fv.refresh_freq = "1 minute"
        fv.warehouse = alternative_wh
        fs.update_feature_view(fv)
        result = self._session.sql(f"SHOW DYNAMIC TABLES IN SCHEMA {fv.database}.{fv.schema}").collect()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["target_lag"], "1 minute")
        self.assertEqual(SqlIdentifier(result[0]["warehouse"]), SqlIdentifier(alternative_wh))


if __name__ == "__main__":
    absltest.main()
