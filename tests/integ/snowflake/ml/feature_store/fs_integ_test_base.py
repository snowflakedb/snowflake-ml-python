from __future__ import annotations

import uuid

from absl.testing import absltest

from snowflake.ml.feature_store import FeatureStore  # type: ignore[attr-defined]
from snowflake.ml.feature_store.feature_store import FeatureStore as FeatureStoreImpl
from snowflake.ml.utils import sql_client
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    external_volume_manager,
    test_env_utils,
)

# Active prefix for online-service-backed test DBs; reclaimed by the bundle's
# 3h targeted cleanup.
SPEC_OFT_E2E_DB_PREFIX = "SNOWML_TEST_SPEC_OFT_E2E_DB_"
SPEC_OFT_E2E_DUMMY_DB_PREFIX = "SNOWML_TEST_SPEC_OFT_E2E_DUMMY_DB_"

# Pre-rename prefixes still lingering in long-running accounts. Cleanup-only.
_LEGACY_SPEC_OFT_E2E_DB_PREFIXES: tuple[str, ...] = (
    "SNOWML_SPEC_OFT_E2E_DB_",
    "SNOWML_SPEC_OFT_E2E_DUMMY_DB_",
)


def cleanup_spec_oft_e2e_databases(dbm: db_manager.DBManager) -> None:
    """3h sweep across every SPEC_OFT_E2E DB prefix (current + legacy).

    Startup-only. Concurrent xdist workers in the same bundle shard share the
    active per-run DB; calling this from teardown would race them. The 3h TTL
    matches the bundle invariant (no single run exceeds 3h).

    Args:
        dbm: DBManager bound to the test session.
    """
    for prefix in (SPEC_OFT_E2E_DB_PREFIX, SPEC_OFT_E2E_DUMMY_DB_PREFIX, *_LEGACY_SPEC_OFT_E2E_DB_PREFIXES):
        dbm.cleanup_databases(prefix=prefix.lower(), expire_hours=3)


class FeatureStoreIntegTestBase(absltest.TestCase):
    """Base for Feature Store integration tests using per-test DBs via db_manager.

    Exposes canonical surfaces `test_db` and `test_schema` for tests to reference.
    Provides an alternate warehouse name for scenarios that require a different
    warehouse than the session default (e.g., to validate warehouse switching).

    Subclasses that provision a Feature Store Online Service must set
    ``_ONLINE_SERVICE_BACKED = True`` so their DBs land under ``SPEC_OFT_E2E_*``
    and are reclaimed by the bundle's 3h targeted cleanup.
    """

    _ONLINE_SERVICE_BACKED: bool = False

    @classmethod
    def _generate_db_names(cls, run_id: str) -> tuple[str, str]:
        """Return ``(test_db, dummy_db)`` per the subclass's online-service flag.

        Args:
            run_id: Short hex slug appended to the prefix.

        Returns:
            ``(test_db, dummy_db)`` upper-cased Snowflake database identifiers.
        """
        if cls._ONLINE_SERVICE_BACKED:
            run_id_up = run_id.upper()
            return (
                f"{SPEC_OFT_E2E_DB_PREFIX}{run_id_up}",
                f"{SPEC_OFT_E2E_DUMMY_DB_PREFIX}{run_id_up}",
            )
        return (
            db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "FS_DB").upper(),
            db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "FS_DUMMY_DB").upper(),
        )

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = test_env_utils.get_available_session()
        self._dbm = db_manager.DBManager(self._session)
        self._evm = external_volume_manager.ExternalVolumeManager(self._session)

        # Stale-resource cleanup (startup-only; see ``cleanup_spec_oft_e2e_databases``).
        self._dbm.cleanup_databases(expire_hours=6)
        cleanup_spec_oft_e2e_databases(self._dbm)
        self._dbm.cleanup_warehouses(expire_hours=6)
        self._dbm.cleanup_roles(expire_hours=6)

        # Create per-test resources with unique names
        run_id = uuid.uuid4().hex[:6]
        self._test_db, self._dummy_db = type(self)._generate_db_names(run_id)

        # Use the session's existing warehouse instead of creating a new one
        # This significantly speeds up test setup (avoids 10-30+ second warehouse creation)
        session_warehouse = self._session.get_current_warehouse()
        if not session_warehouse:
            raise RuntimeError("No warehouse is configured in the current session.")
        self._test_warehouse_name = session_warehouse.strip('"')

        # Lazy creation: alternate warehouse is only created if a test actually needs it
        # This is used by a small subset of tests that validate warehouse switching behavior
        self._alt_warehouse_name_value: str | None = None
        self._alt_warehouse_created = False

        self._dbm.create_database(self._test_db, data_retention_time_in_days=1)
        self._dbm.create_database(self._dummy_db, data_retention_time_in_days=1)

        # Set warehouse and database context
        self._dbm.use_warehouse(self._test_warehouse_name)
        self._dbm.use_database(self._test_db)

        # Create working schema per test
        full_qual_schema = self._dbm.create_random_schema(db_name=self._test_db)
        # db_manager returns a fully-qualified name; extract schema identifier only
        self._test_schema = full_qual_schema.split(".")[-1]

        # Provide a convenient property-compatible name
        self.test_db = self._test_db
        self.test_schema = self._test_schema

    @property
    def _alt_warehouse_name(self) -> str:
        """Lazily create alternate warehouse only when accessed.

        This property ensures the alternate warehouse is only created for tests that
        actually need it (e.g., warehouse switching tests), avoiding the 10-30+ second
        overhead for the majority of tests that don't need a second warehouse.

        Returns:
            The name of the alternate warehouse (created on first access if needed).
        """
        if self._alt_warehouse_name_value is None:
            run_id = uuid.uuid4().hex[:6]
            self._alt_warehouse_name_value = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                run_id, "FS_ALT_WH"
            ).upper()

        if not self._alt_warehouse_created:
            self._dbm.create_warehouse(
                self._alt_warehouse_name_value,
                creation_mode=sql_client.CreationMode(if_not_exists=True),
                size="XSMALL",
            )
            self._alt_warehouse_created = True
            # Restore session to primary test warehouse (CREATE WAREHOUSE switches context)
            self._dbm.use_warehouse(self._test_warehouse_name)

        return self._alt_warehouse_name_value

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Drop per-test resources
        self._dbm.drop_database(self._test_db, if_exists=True)
        self._dbm.drop_database(self._dummy_db, if_exists=True)

        # Only drop alternate warehouse if it was actually created
        if self._alt_warehouse_created:
            self._dbm.drop_warehouse(self._alt_warehouse_name_value, if_exists=True)

        # Don't drop the primary warehouse - it's the session's existing warehouse
        # that may be shared across tests

        self._session.close()

    # Helpers
    def use_feature_store_schema(self, fs: FeatureStore) -> None:  # type: ignore[name-defined]
        """Ensure session schema context matches the given FeatureStore's full schema path.

        Intended for use before registering or reading feature views when object
        resolution depends on the active schema.

        Args:
            fs: FeatureStore instance whose schema context should be used.
        """
        self._session.sql(f"USE SCHEMA {fs._config.full_schema_path}").collect()

    def assert_no_temp_shadow_swap_objects(self, fs: FeatureStore) -> None:  # type: ignore[name-defined]
        """Assert that no temporary shadow swap objects remain in the feature store schema.

        This checks for any VIEWs or DYNAMIC TABLE with the temporary prefixes
        (_TMP_VIEW_ or _TMP_TABLE_) that should have been cleaned up after a shadow swap.

        Args:
            fs: FeatureStore instance to check for leftover temporary objects.
        """
        schema_path = fs._config.full_schema_path

        # Check for leftover temporary views
        tmp_view_prefix = FeatureStoreImpl._TMP_VIEW_PREFIX
        views = self._session.sql(f"SHOW VIEWS LIKE '%{tmp_view_prefix}%' IN SCHEMA {schema_path}").collect()
        self.assertEqual(
            len(views),
            0,
            f"Found leftover temporary views with prefix '{tmp_view_prefix}': {[v['name'] for v in views]}",
        )

        # Check for leftover temporary dynamic tables
        tmp_dt_prefix = FeatureStoreImpl._TMP_DT_PREFIX
        dynamic_tables = self._session.sql(
            f"SHOW DYNAMIC TABLES LIKE '%{tmp_dt_prefix}%' IN SCHEMA {schema_path}"
        ).collect()
        self.assertEqual(
            len(dynamic_tables),
            0,
            f"Found leftover temporary dynamic tables with prefix '{tmp_dt_prefix}': "
            f"{[dt['name'] for dt in dynamic_tables]}",
        )
