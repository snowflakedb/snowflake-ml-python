"""Sharded runner for all spec-based OFT e2e integration tests.

Each Bazel shard gets its own schema (and thus its own Online Service runtime)
within a shared database.  Runtimes are created in parallel across shards so
wall-clock setup time stays ~4 min regardless of shard count.

Usage::

    bazel test //tests/integ/.../feature_store:feature_store_spec_oft_bundle_test \\
        --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)
"""

import logging
import os
import uuid

from absl.testing import absltest
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    wait_online_service_running_with_query_endpoint,
)

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.utils import sql_client
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    external_volume_manager,
    test_env_utils,
)

logger = logging.getLogger(__name__)

_DB_PREFIX = "SNOWML_TEST_SPEC_OFT_E2E_DB_"
_DUMMY_DB_PREFIX = "SNOWML_TEST_SPEC_OFT_E2E_DUMMY_DB_"

_module_state: dict | None = None


def _run_id() -> str:
    # PYTEST_XDIST_TESTRUNUID is set once per pytest invocation and is identical across all xdist
    # workers in the same run, so all workers converge on the same DB name (one shared OFT per job).
    raw = os.environ.get("PYTEST_XDIST_TESTRUNUID") or "LOCAL"
    return "".join(c for c in raw.upper() if c.isalnum())[:12] or "LOCAL"


def _shard_schema_name() -> str:
    shard_idx = os.environ.get("TEST_SHARD_INDEX", "0")
    return f"SNOWML_SFV_E2E_SHARD_{shard_idx}"


def setUpModule() -> None:
    global _module_state

    session = test_env_utils.get_available_session()
    dbm = db_manager.DBManager(session)
    evm = external_volume_manager.ExternalVolumeManager(session)

    dbm.cleanup_databases(expire_hours=6)
    # Targeted shorter TTL for this bundle's own run-scoped DBs (and the dummy
    # DB) so orphaned Postgres-backed OFT runtimes are reclaimed within ~3 hours
    # instead of the global 6h. Safe because run_id is scoped to a single pytest
    # invocation and the bundle's own wall-clock never exceeds 3h.
    dbm.cleanup_databases(prefix=_DB_PREFIX.lower(), expire_hours=3)
    dbm.cleanup_databases(prefix=_DUMMY_DB_PREFIX.lower(), expire_hours=3)
    dbm.cleanup_warehouses(expire_hours=6)
    dbm.cleanup_roles(expire_hours=6)

    run_id = _run_id()
    test_db = f"{_DB_PREFIX}{run_id}"
    dummy_db = f"{_DUMMY_DB_PREFIX}{run_id}"

    session_warehouse = session.get_current_warehouse()
    if not session_warehouse:
        raise RuntimeError("No warehouse is configured in the current session.")
    warehouse = session_warehouse.strip('"')

    dbm.create_database(
        test_db, creation_mode=sql_client.CreationMode(if_not_exists=True), data_retention_time_in_days=1
    )
    dbm.create_database(
        dummy_db, creation_mode=sql_client.CreationMode(if_not_exists=True), data_retention_time_in_days=1
    )
    dbm.use_warehouse(warehouse)
    dbm.use_database(test_db)

    test_schema = _shard_schema_name()
    dbm.create_schema(
        test_schema,
        db_name=test_db,
        creation_mode=sql_client.CreationMode(if_not_exists=True),
    )

    fs = FeatureStore(
        session=session,
        database=test_db,
        name=test_schema,
        default_warehouse=warehouse,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    user_entity = Entity(name="user_entity", join_keys=["USER_ID"], desc="User entity")
    try:
        fs.register_entity(user_entity)
    except Exception:
        # Idempotent: re-running setUpModule on a reused per-run DB will hit "entity already exists".
        # Log at debug so a real registration failure (permissions, schema misconfig) is still
        # observable when the suite runs at -v.
        logger.debug("register_entity(user_entity) raised; assuming already-registered.", exc_info=True)

    producer = session.get_current_role().strip('"')
    consumer = f"SNOWML_TEST_SPEC_OFT_C_{uuid.uuid4().hex[:8]}".upper()
    session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(consumer)}").collect()
    session.sql(f"GRANT ROLE {SqlIdentifier(consumer)} TO ROLE {session.get_current_role()}").collect()

    def _restore_session_context() -> None:
        dbm.use_database(test_db)
        session.sql(f"USE SCHEMA {test_db}.{test_schema}").collect()

    wait_online_service_running_with_query_endpoint(
        session=session,
        fs=fs,
        producer_role=producer,
        consumer_role=consumer,
        on_recreate=_restore_session_context,
    )

    logger.info(
        "Shard %s: Online Service RUNNING for %s.%s — starting e2e tests.",
        os.environ.get("TEST_SHARD_INDEX", "0"),
        test_db,
        test_schema,
    )

    _module_state = {
        "session": session,
        "dbm": dbm,
        "evm": evm,
        "test_db": test_db,
        "dummy_db": dummy_db,
        "warehouse": warehouse,
        "test_schema": test_schema,
        "fs": fs,
        "user_entity": user_entity,
        "consumer_role": consumer,
    }

    StreamingFeatureViewIntegTestBase._module_state = _module_state


def tearDownModule() -> None:
    # Each xdist worker runs this when its own slice of the bundled module is done. Workers in the
    # same run share one DB / schema / Online Service (keyed by PYTEST_XDIST_TESTRUNUID), so dropping
    # them here would race other workers still using them. Reclamation happens via the default
    # cleanup_databases (snowml_test_ prefix) sweep at the start of the next run.
    global _module_state
    if _module_state is None:
        return

    StreamingFeatureViewIntegTestBase._module_state = None

    session = _module_state.get("session")
    if session is not None:
        try:
            session.close()
        except Exception:
            logger.warning("session.close() failed during tearDownModule.", exc_info=True)

    _module_state = None


# ---------------------------------------------------------------------------
# Import test classes — unittest/absltest discovers them automatically.
# Re-assign __module__ so that setUpModule/tearDownModule from THIS file are
# invoked by the test runner (unittest groups tests by __module__).
# ---------------------------------------------------------------------------
from feature_store_batch_online_read_bundled import (  # noqa: E402,F401
    FeatureStoreBatchOnlineReadIntegTest,
)
from feature_store_stream_ingest_bundled import (  # noqa: E402,F401
    FeatureStoreStreamIngestIntegTest,
)
from feature_store_streaming_fv_bundled import (  # noqa: E402,F401
    StreamingFeatureViewIntegTest,
)

FeatureStoreBatchOnlineReadIntegTest.__module__ = __name__
FeatureStoreStreamIngestIntegTest.__module__ = __name__
StreamingFeatureViewIntegTest.__module__ = __name__

if __name__ == "__main__":
    absltest.main()
