from typing import Optional
from unittest.mock import patch

from absl.testing import absltest
from common_utils import (
    FS_INTEG_TEST_DB,
    cleanup_temporary_objects,
    create_mock_table,
    create_random_schema,
    get_test_warehouse_name,
)

from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
    FeatureViewStatus,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Row, Session


class FeatureSharingTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        cleanup_temporary_objects(self._session)
        self._active_feature_store = []
        self._test_warehouse_name = get_test_warehouse_name(self._session)

    @classmethod
    def tearDownClass(self) -> None:
        for fs in self._active_feature_store:
            fs._clear(dryrun=False)
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
        self._session.close()

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        current_schema = create_random_schema(self._session, "FS_SHARING_TEST") if name is None else name
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        return fs

    def test_read_shared_feature_views(self) -> None:
        # NOTE: shared feature views will have text, scheduling state and warehouse being empty
        # due to privacy constraints. Here we mock the behavior by patching the query results
        # with empty strings for those fields.
        fs = self._create_feature_store()
        table = create_mock_table(self._session)

        e = Entity(name="foo", join_keys=["id"])
        fs.register_entity(e)

        df = self._session.sql(f"select id, name from {table}")
        fv1 = FeatureView(name="fv1", entities=[e], feature_df=df, refresh_freq="1m", desc="managed fv1")
        fs.register_feature_view(fv1, "v1")

        fv2 = FeatureView(name="fv2", entities=[e], feature_df=df, desc="external fv2")
        fs.register_feature_view(fv2, "v2")

        original_vals = fs._get_fv_backend_representations(None)
        mocked_vals = []
        for r, t in original_vals:
            r_dict = r.as_dict()
            r_dict["text"] = ""
            if "scheduling_state" in r_dict:
                r_dict["scheduling_state"] = ""
            if "warehouse" in r_dict:
                r_dict["warehouse"] = ""
            mocked_vals.append((Row(**r_dict), t))

        with patch.object(FeatureStore, "_get_fv_backend_representations", return_value=mocked_vals):
            self.assertEqual(len(fs.list_feature_views().to_pandas().index), 2)

        with patch.object(FeatureStore, "_get_fv_backend_representations", return_value=mocked_vals[:1]):
            fv = fs.get_feature_view("fv1", "v1")
            self.assertEqual(fv.status, FeatureViewStatus.MASKED)
            self.assertEqual(fv.warehouse, None)
            self.assertRegex(fv.query.lower(), r"select \* from *")
            self.assertEqual(fv.desc, "managed fv1")

        with patch.object(FeatureStore, "_get_fv_backend_representations", return_value=mocked_vals[1:]):
            fv = fs.get_feature_view("fv2", "v2")
            self.assertEqual(fv.status, FeatureViewStatus.STATIC)
            self.assertEqual(fv.warehouse, None)
            self.assertRegex(fv.query.lower(), r"select \* from *")
            self.assertEqual(fv.desc, "external fv2")


if __name__ == "__main__":
    absltest.main()
