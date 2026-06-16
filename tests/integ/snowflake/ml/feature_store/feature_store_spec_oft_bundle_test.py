"""Sharded runner for all spec-based OFT e2e integration tests.

Each Bazel shard gets its own schema (and thus its own Online Service runtime)
within a shared database.  Runtimes are created in parallel across shards so
wall-clock setup time stays ~4 min regardless of shard count.

Usage::

    bazel test //tests/integ/.../feature_store:feature_store_spec_oft_bundle_test \\
        --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)

Reuse mode (skip provisioning, attach to an existing schema with a RUNNING
Online Service)::

    bazel test //tests/integ/.../feature_store:feature_store_spec_oft_bundle_test \\
        --test_env=SPEC_OFT_BUNDLE_REUSE_SCHEMA=DB.SCHEMA

When ``SPEC_OFT_BUNDLE_REUSE_SCHEMA`` is set, setUpModule skips DB create /
schema create / Online Service provisioning and tearDownModule keeps the
schema, DB, and Online Service intact. Only the per-test FVs / RTFVs / FGs
that the bundled tests create are cleaned up.
"""

import logging
import os
import uuid

from absl.testing import absltest
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    wait_online_service_running_with_query_endpoint,
)

from fs_integ_test_base import (
    SPEC_OFT_E2E_DB_PREFIX,
    SPEC_OFT_E2E_DUMMY_DB_PREFIX,
    cleanup_spec_oft_e2e_databases,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.online_service import OnlineServiceAccess
from snowflake.ml.utils import sql_client
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    external_volume_manager,
    test_env_utils,
)

logger = logging.getLogger(__name__)

_module_state: dict | None = None

# Per-process fallback when pytest-xdist is not the runner (e.g. Bazel `py_test`
# with `absltest.main()`). Prevents concurrent CI builds across branches from
# colliding on a shared ``..._LOCAL`` DB.
_RUN_ID_FALLBACK = uuid.uuid4().hex[:12].upper()


def _run_id() -> str:
    # Priority: SPEC_OFT_BUNDLE_RUN_ID (explicit pin for local iteration to reuse a
    # RUNNING Online Service; bring-up is ~15 min) > PYTEST_XDIST_TESTRUNUID (one
    # shared OFT per xdist job) > _RUN_ID_FALLBACK (random per process).
    raw = os.environ.get("SPEC_OFT_BUNDLE_RUN_ID") or os.environ.get("PYTEST_XDIST_TESTRUNUID") or _RUN_ID_FALLBACK
    return "".join(c for c in raw.upper() if c.isalnum())[:12] or _RUN_ID_FALLBACK


def _shard_schema_name() -> str:
    shard_idx = os.environ.get("TEST_SHARD_INDEX", "0")
    return f"SNOWML_SFV_E2E_SHARD_{shard_idx}"


def _parse_reuse_target() -> tuple[str, str] | None:
    """Parse ``SPEC_OFT_BUNDLE_REUSE_SCHEMA=DB.SCHEMA`` for reuse mode."""
    raw = os.environ.get("SPEC_OFT_BUNDLE_REUSE_SCHEMA", "").strip()
    if not raw:
        return None
    parts = raw.split(".")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"SPEC_OFT_BUNDLE_REUSE_SCHEMA must be in the form DB.SCHEMA; got {raw!r}.")
    return parts[0], parts[1]


def _setup_reuse_mode(reuse_db: str, reuse_schema: str) -> None:
    """Attach to an existing DB.SCHEMA with a RUNNING Online Service.

    Skips DB / schema / Online Service provisioning entirely. Used when the
    target account's Postgres provisioner is flaky or when iterating locally
    against a long-lived schema.
    """
    global _module_state

    session = test_env_utils.get_available_session()
    dbm = db_manager.DBManager(session)
    evm = external_volume_manager.ExternalVolumeManager(session)

    session_warehouse = session.get_current_warehouse()
    if not session_warehouse:
        raise RuntimeError("No warehouse is configured in the current session.")
    warehouse = session_warehouse.strip('"')

    dbm.use_warehouse(warehouse)
    session.sql(f"USE DATABASE {SqlIdentifier(reuse_db)}").collect()
    session.sql(f"USE SCHEMA {SqlIdentifier(reuse_db)}.{SqlIdentifier(reuse_schema)}").collect()

    fs = FeatureStore(
        session=session,
        database=reuse_db,
        name=reuse_schema,
        default_warehouse=warehouse,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        online_service_access=OnlineServiceAccess.PUBLIC,
    )

    # Resolve / register the user entity. Reuse mode tolerates a pre-existing
    # entity with the same name + join keys.
    user_entity = Entity(name="user_entity", join_keys=["USER_ID"], desc="User entity")
    # ``register_entity`` is idempotent: an existing entity by the same
    # name returns with a ``UserWarning`` instead of raising.
    fs.register_entity(user_entity)

    # ``dummy_db`` is reused tests' off-target DB. In reuse mode we point it
    # at the same DB so tests that switch databases for a moment land back
    # somewhere sensible without requiring extra cleanup.
    logger.info(
        "Bundle reuse mode: attached to %s.%s (warehouse=%s); skipping OS provisioning + cleanup.",
        reuse_db,
        reuse_schema,
        warehouse,
    )

    _module_state = {
        "session": session,
        "dbm": dbm,
        "evm": evm,
        "test_db": reuse_db,
        "dummy_db": reuse_db,
        "warehouse": warehouse,
        "test_schema": reuse_schema,
        "fs": fs,
        "user_entity": user_entity,
        "consumer_role": None,
        "_reuse_mode": True,
    }

    StreamingFeatureViewIntegTestBase._module_state = _module_state


def setUpModule() -> None:
    global _module_state

    reuse_target = _parse_reuse_target()
    if reuse_target is not None:
        _setup_reuse_mode(*reuse_target)
        return

    session = test_env_utils.get_available_session()
    dbm = db_manager.DBManager(session)
    evm = external_volume_manager.ExternalVolumeManager(session)

    dbm.cleanup_databases(expire_hours=6)
    cleanup_spec_oft_e2e_databases(dbm)
    dbm.cleanup_warehouses(expire_hours=6)
    dbm.cleanup_roles(expire_hours=6)

    run_id = _run_id()
    test_db = f"{SPEC_OFT_E2E_DB_PREFIX}{run_id}"
    dummy_db = f"{SPEC_OFT_E2E_DUMMY_DB_PREFIX}{run_id}"

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
        online_service_access=OnlineServiceAccess.PUBLIC,
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

    if _module_state.get("_reuse_mode"):
        # Reuse mode: caller owns the DB / schema / Online Service. Don't
        # drop anything; just close the session.
        logger.info("Bundle reuse mode: leaving DB / schema / Online Service intact.")

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
from feature_store_distinct_n_online_bundled import (  # noqa: E402,F401
    FeatureStoreDistinctNBatchIntegTest,
    FeatureStoreDistinctNStreamingIntegTest,
)
from feature_store_feature_group_bundled import FeatureGroupIntegTest  # noqa: E402,F401
from feature_store_oft_varchar_length_bundled import (  # noqa: E402,F401
    FeatureStoreOftVarcharLengthIntegTest,
)
from feature_store_realtime_bundled import (  # noqa: E402,F401
    RealtimeFeatureViewIntegTest,
)
from feature_store_stream_ingest_bundled import (  # noqa: E402,F401
    FeatureStoreAppendOnlyOFTIntegTest,
    FeatureStoreStreamIngestIntegTest,
)
from feature_store_streaming_fv_bundled import (  # noqa: E402,F401
    StreamingFeatureViewIntegTest,
)

FeatureStoreBatchOnlineReadIntegTest.__module__ = __name__
FeatureStoreDistinctNStreamingIntegTest.__module__ = __name__
FeatureStoreDistinctNBatchIntegTest.__module__ = __name__
FeatureGroupIntegTest.__module__ = __name__
FeatureStoreOftVarcharLengthIntegTest.__module__ = __name__
RealtimeFeatureViewIntegTest.__module__ = __name__
FeatureStoreStreamIngestIntegTest.__module__ = __name__
FeatureStoreAppendOnlyOFTIntegTest.__module__ = __name__
StreamingFeatureViewIntegTest.__module__ = __name__

if __name__ == "__main__":
    absltest.main()
