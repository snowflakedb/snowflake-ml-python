from absl.testing import absltest
from common_utils import compare_dataframe
from snowflake.ml.version import VERSION

from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
    _FeatureStoreObjTypes,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

FS_COMPATIBIILTY_TEST_DB = "FEATURE_STORE_COMPATIBILITY_TEST_DB"
FS_COMPATIBIILTY_TEST_SCHEMA = "FEATURE_STORE_COMPATIBILITY_TEST_SCHEMA"
TEST_DATA = "test_data"
# check backward compatibility with two pkg versions in the past
BC_VERSION_LIMITS = 2


def _create_test_data(session: Session) -> str:
    test_table = f"{FS_COMPATIBIILTY_TEST_DB}.{FS_COMPATIBIILTY_TEST_SCHEMA}.{TEST_DATA}"
    session.sql(
        f"""CREATE TABLE IF NOT EXISTS {test_table}
            (name VARCHAR(64), id INT, title VARCHAR(128), age INT, dept VARCHAR(64), ts INT)
        """
    ).collect()
    session.sql(
        f"""INSERT OVERWRITE INTO {test_table} (name, id, title, age, dept, ts)
            VALUES
            ('john', 1, 'boss', 20, 'sales', 100),
            ('porter', 2, 'manager', 30, 'engineer', 200)
        """
    ).collect()
    return test_table


class FeatureStoreCompatibilityTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

        try:
            self._session.sql(f"CREATE DATABASE IF NOT EXISTS {FS_COMPATIBIILTY_TEST_DB}").collect()
            self._session.use_database(FS_COMPATIBIILTY_TEST_DB)
            self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {FS_COMPATIBIILTY_TEST_SCHEMA}").collect()
            self._session.use_schema(FS_COMPATIBIILTY_TEST_SCHEMA)
            self._mock_table = _create_test_data(self._session)
        except Exception as e:
            raise Exception(f"Test setup failed: {e}")

    def test_cross_version_compatibilities(self) -> None:
        fs = FeatureStore(
            self._session,
            FS_COMPATIBIILTY_TEST_DB,
            FS_COMPATIBIILTY_TEST_SCHEMA,
            self._session.get_current_warehouse(),
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._maybe_create_feature_store_objects(fs)

        versions = fs._collapse_object_versions()
        self.assertGreater(len(versions), 0)
        for version in versions[: BC_VERSION_LIMITS + 1]:
            self._check_per_version_access(str(version), fs)

        # TODO: update to check more than 1 with next version release
        # obj from at least two versions should be listed
        entity_df = fs.list_entities()
        self.assertGreater(len(entity_df.collect()), 1)
        fv_df = fs.list_feature_views()
        self.assertGreater(len(fv_df.collect()), 1)

    def test_forward_compatibility_breakage(self) -> None:
        with absltest.mock.patch(
            "snowflake.ml.feature_store.feature_store._FeatureStoreObjTypes.parse", autospec=True
        ) as MockFeatureStoreObjTypesParseFn:
            MockFeatureStoreObjTypesParseFn.return_value = _FeatureStoreObjTypes.UNKNOWN
            with self.assertRaisesRegex(RuntimeError, "The current snowflake-ml-python version .*"):
                FeatureStore(
                    self._session,
                    FS_COMPATIBIILTY_TEST_DB,
                    FS_COMPATIBIILTY_TEST_SCHEMA,
                    self._session.get_current_warehouse(),
                    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
                )

    def test_pkg_version_falling_behind(self) -> None:
        with absltest.mock.patch(
            "snowflake.ml.feature_store.feature_store.snowml_version", autospec=True
        ) as MockSnowMLVersion:
            MockSnowMLVersion.VERSION = "1.0.0"
            with self.assertWarnsRegex(UserWarning, "The current snowflake-ml-python version out of date.*"):
                FeatureStore(
                    self._session,
                    FS_COMPATIBIILTY_TEST_DB,
                    FS_COMPATIBIILTY_TEST_SCHEMA,
                    self._session.get_current_warehouse(),
                    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
                )

    def _check_per_version_access(self, version: str, fs: FeatureStore) -> None:
        entity_names = ["foo", "Bar"]
        for e in entity_names:
            fs.get_entity(self._get_versioned_object_name(e, version))

        feature_view_names = ["unmanaged_fv", "MANAGED_fv"]
        fvs = []
        for fv_name in feature_view_names:
            fv_name = self._get_versioned_object_name(fv_name, version)
            fv = fs.get_feature_view(fv_name, "V1")
            data = fs.read_feature_view(fv)
            self.assertEqual(len(data.collect()), 2)
            fvs.append(fv)

        spine_df = self._session.create_dataframe(
            [(1, "john", 101), (2, "porter", 202), (1, "john", 90)], schema=["id", "name", "ts"]
        )
        data = fs.retrieve_feature_values(
            spine_df=spine_df,
            features=fvs,
            spine_timestamp_col="ts",
        )
        compare_dataframe(
            actual_df=data.to_pandas(),
            target_data={
                "ID": [1, 1, 2],
                "NAME": ["john", "john", "porter"],
                "TS": [90, 101, 202],
                "TITLE": ["boss", "boss", "manager"],
                "AGE": [None, 20, 30],
                "DEPT": [None, "sales", "engineer"],
            },
            sort_cols=["ID", "TS"],
        )

    def _maybe_create_feature_store_objects(self, fs: FeatureStore) -> None:
        e1 = Entity(self._get_versioned_object_name("foo", VERSION), ["id"], f"VERSION={VERSION}")
        fs.register_entity(e1)
        e2 = Entity(self._get_versioned_object_name("Bar", VERSION), ["id", "name"], f"VERSION={VERSION}")
        fs.register_entity(e2)

        sql1 = f"select id, title from {TEST_DATA}"
        fv1 = FeatureView(
            name=self._get_versioned_object_name("unmanaged_fv", VERSION),
            entities=[e1],
            feature_df=self._session.sql(sql1),
        )
        fs.register_feature_view(feature_view=fv1, version="V1")

        sql2 = f"select id, name, age, dept, ts from {TEST_DATA}"
        fv2 = FeatureView(
            name=self._get_versioned_object_name("MANAGED_fv", VERSION),
            entities=[e1, e2],
            feature_df=self._session.sql(sql2),
            timestamp_col="ts",
            refresh_freq="60m",
        )
        fs.register_feature_view(feature_view=fv2, version="V1")

    def _get_versioned_object_name(self, prefix: str, version: str) -> str:
        name = f"{prefix}_{version.replace('.', '_')}"
        if prefix.islower():
            return name
        else:
            return f'"{name}"'


if __name__ == "__main__":
    absltest.main()
