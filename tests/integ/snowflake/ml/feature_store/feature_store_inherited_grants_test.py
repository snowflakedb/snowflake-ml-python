"""Integration tests for reading feature views through a role with inherited schema grants."""

from uuid import uuid4

from absl.testing import absltest
from common_utils import create_random_schema

from fs_integ_test_base import FeatureStoreIntegTestBase
from snowflake.ml.feature_store import Feature
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView
from tests.integ.snowflake.ml.test_utils import db_manager


class FeatureStoreInheritedGrantsTest(FeatureStoreIntegTestBase):
    """A consumer whose table access comes from ``GRANT INHERITED`` can read tiled feature views.

    Objects reachable only through an inherited grant are listed by ``SHOW TABLES`` but not by
    ``INFORMATION_SCHEMA.TABLES``, so the internal metadata table must be detected with ``SHOW``.

    This problem is supposedly addressed in the 2026_07 BCR bundle but we won't wait for it to rollout.
    """

    def setUp(self) -> None:
        super().setUp()
        self._admin_role = self._session.get_current_role()
        self._consumer_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            uuid4().hex[:6], "FS_INHERITED_CONSUMER"
        ).upper()
        self._source_table = f"{self.test_db}.{self.test_schema}.EVENTS_{uuid4().hex.upper()}"
        self._session.sql(
            f"CREATE TABLE {self._source_table} (USER_ID INT, EVENT_TS TIMESTAMP_NTZ, AMOUNT FLOAT)"
        ).collect()
        self._session.sql(
            f"""INSERT INTO {self._source_table} (USER_ID, EVENT_TS, AMOUNT) VALUES
                (1, '2024-01-01 00:30:00', 10.0),
                (1, '2024-01-01 01:15:00', 30.0),
                (2, '2024-01-01 01:00:00', 100.0),
                (2, '2024-01-01 02:00:00', 200.0)
            """
        ).collect()

    def tearDown(self) -> None:
        self._session.use_role(self._admin_role)
        self._dbm.drop_role(self._consumer_role, if_exists=True)
        super().tearDown()

    def _create_consumer_role(self, fs_schema_path: str) -> None:
        self._dbm.create_role(self._consumer_role)
        current_user = self._session.get_current_user()
        for statement in [
            f"GRANT ROLE {self._consumer_role} TO USER {current_user}",
            f"GRANT USAGE ON WAREHOUSE {self._test_warehouse_name} TO ROLE {self._consumer_role}",
            f"GRANT USAGE ON DATABASE {self.test_db} TO ROLE {self._consumer_role}",
            f"GRANT USAGE ON SCHEMA {self.test_db}.{self.test_schema} TO ROLE {self._consumer_role}",
            f"GRANT SELECT ON TABLE {self._source_table} TO ROLE {self._consumer_role}",
            f"GRANT USAGE ON SCHEMA {fs_schema_path} TO ROLE {self._consumer_role}",
            f"GRANT INHERITED SELECT ON ALL TABLES IN SCHEMA {fs_schema_path} TO ROLE {self._consumer_role}",
            # Dynamic tables are not covered by the inherited table grant.
            f"GRANT SELECT, MONITOR ON FUTURE DYNAMIC TABLES IN SCHEMA {fs_schema_path} TO ROLE {self._consumer_role}",
        ]:
            self._session.sql(statement).collect()

    @absltest.skip("Temporarily disabled. Enable once inherited grants are enabled in all test accounts.")
    def test_consumer_with_inherited_grant_gets_tiled_feature_view(self) -> None:
        fs_schema = create_random_schema(self._session, "FS_INHERITED_GRANTS", database=self.test_db)
        fs = FeatureStore(
            self._session,
            self.test_db,
            fs_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        fs_schema_path = f"{self.test_db}.{fs_schema}"

        # The grant precedes registration, so the metadata table is created after it and is
        # reachable by the consumer only through the inherited grant.
        self._create_consumer_role(fs_schema_path)

        entity = Entity("user", ["USER_ID"])
        fs.register_entity(entity)
        registered_fv = fs.register_feature_view(
            feature_view=FeatureView(
                name="user_amount_stats",
                entities=[entity],
                feature_df=self._session.table(self._source_table),
                timestamp_col="EVENT_TS",
                refresh_freq="1h",
                feature_granularity="1h",
                features=[
                    Feature.sum("AMOUNT", "2h").alias("AMOUNT_SUM_2H"),
                    Feature.count("AMOUNT", "2h").alias("AMOUNT_COUNT_2H"),
                ],
            ),
            version="v1",
        )

        self._session.use_role(self._consumer_role)
        self._session.sql("USE SECONDARY ROLES NONE").collect()
        self._session.use_database(self.test_db)
        self._session.use_schema(fs_schema)
        consumer_fs = FeatureStore(
            self._session,
            self.test_db,
            fs_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )

        fv = consumer_fs.get_feature_view("user_amount_stats", "v1")

        self.assertTrue(fv.is_tiled)
        self.assertEqual(fv.feature_granularity, "1h")
        self.assertEqual(fv.timestamp_col, "EVENT_TS")
        self.assertEqual(fv.feature_names, registered_fv.feature_names)
        self.assertEqual(fv.aggregation_specs, registered_fv.aggregation_specs)


if __name__ == "__main__":
    absltest.main()
