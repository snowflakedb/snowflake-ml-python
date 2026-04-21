"""Sharded runner for all spec-based OFT e2e integration tests.

Each Bazel shard gets its own schema (and thus its own Online Service runtime)
within a shared database.  Runtimes are created in parallel across shards so
wall-clock setup time stays ~4 min regardless of shard count.

Usage::

    bazel test //tests/integ/.../feature_store:feature_store_spec_oft_e2e_integ_test \\
        --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)
"""

import logging
import os
import time
import uuid

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.online_service import fetch_online_service_status
from snowflake.ml.utils import sql_client
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    external_volume_manager,
    test_env_utils,
)

logger = logging.getLogger(__name__)

_FIXED_DB = "SNOWML_SPEC_OFT_E2E_DB"
_FIXED_DUMMY_DB = "SNOWML_SPEC_OFT_E2E_DUMMY_DB"

_module_state: dict | None = None


def _shard_schema_name() -> str:
    shard_idx = os.environ.get("TEST_SHARD_INDEX", "0")
    return f"SNOWML_SFV_E2E_SHARD_{shard_idx}"


def setUpModule() -> None:
    global _module_state

    session = test_env_utils.get_available_session()
    dbm = db_manager.DBManager(session)
    evm = external_volume_manager.ExternalVolumeManager(session)

    dbm.cleanup_databases(expire_hours=6)
    dbm.cleanup_warehouses(expire_hours=6)
    dbm.cleanup_roles(expire_hours=6)

    test_db = _FIXED_DB
    dummy_db = _FIXED_DUMMY_DB

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
        pass

    producer = session.get_current_role().strip('"')
    consumer = f"SML_SFVRT_C_{uuid.uuid4().hex[:8]}".upper()
    session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(consumer)}").collect()
    session.sql(f"GRANT ROLE {SqlIdentifier(consumer)} TO ROLE {session.get_current_role()}").collect()

    st = fetch_online_service_status(
        session, fs._config.database, fs._config.schema, statement_params=fs._telemetry_stmp
    )
    if st.status == "ERROR":
        logger.warning("Online Service in ERROR state — dropping before re-creation.")
        try:
            fs.drop_online_service()
        except Exception:
            logger.warning("drop_online_service failed; continuing with create.", exc_info=True)
        dbm.use_database(test_db)
        session.sql(f"USE SCHEMA {test_db}.{test_schema}").collect()
        st = fetch_online_service_status(
            session, fs._config.database, fs._config.schema, statement_params=fs._telemetry_stmp
        )

    if not (st.status == "RUNNING" and any(ep.name == "query" for ep in st.endpoints)):
        dbm.use_database(test_db)
        session.sql(f"USE SCHEMA {test_db}.{test_schema}").collect()
        try:
            fs.create_online_service(producer, consumer)
        except Exception as e:
            logger.warning("create_online_service raised: %s — will poll for RUNNING.", e)

        deadline = time.time() + 900.0
        while time.time() < deadline:
            st = fs.get_online_service_status()
            if st.status == "RUNNING" and any(ep.name == "query" for ep in st.endpoints):
                break
            if st.status == "ERROR":
                raise RuntimeError(f"Online Service entered ERROR state: {st.message}")
            time.sleep(5)
        else:
            raise RuntimeError("Online Service did not reach RUNNING with a query endpoint within timeout.")

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

    from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

    StreamingFeatureViewIntegTestBase._module_state = _module_state


def tearDownModule() -> None:
    global _module_state
    if _module_state is None:
        return

    from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

    StreamingFeatureViewIntegTestBase._module_state = None

    skip_teardown = os.environ.get("SKIP_RUNTIME_TEARDOWN") or os.environ.get("NO_DROP_RUNTIME")
    if skip_teardown:
        logger.info(
            "SKIP_RUNTIME_TEARDOWN/NO_DROP_RUNTIME set — keeping Online Service and schema %s.%s alive.",
            _module_state.get("test_db"),
            _module_state.get("test_schema"),
        )
        session = _module_state.get("session")
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        _module_state = None
        return

    fs = _module_state.get("fs")
    dbm = _module_state.get("dbm")
    session = _module_state.get("session")

    if fs is not None:
        try:
            fs.drop_online_service()
        except Exception:
            logger.warning("Failed to drop Online Service during tearDownModule.", exc_info=True)
        try:
            fs._clear(dryrun=False)
        except Exception:
            pass

    consumer_role = _module_state.get("consumer_role")
    if consumer_role and session is not None:
        try:
            session.sql(f"DROP ROLE IF EXISTS {SqlIdentifier(consumer_role)}").collect()
        except Exception:
            pass

    if dbm is not None:
        test_db = _module_state.get("test_db")
        test_schema = _module_state.get("test_schema")
        if test_db and test_schema:
            try:
                dbm.drop_schema(test_schema, db_name=test_db, if_exists=True)
            except Exception:
                pass

    if session is not None:
        try:
            session.close()
        except Exception:
            pass

    _module_state = None


# ---------------------------------------------------------------------------
# Import test classes — unittest/absltest discovers them automatically.
# Re-assign __module__ so that setUpModule/tearDownModule from THIS file are
# invoked by the test runner (unittest groups tests by __module__).
# ---------------------------------------------------------------------------
from feature_store_batch_online_read_integ_test import (  # noqa: E402,F401
    FeatureStoreBatchOnlineReadIntegTest,
)
from feature_store_stream_ingest_integ_test import (  # noqa: E402,F401
    FeatureStoreStreamIngestIntegTest,
)
from feature_store_streaming_fv_test import (  # noqa: E402,F401
    StreamingFeatureViewIntegTest,
)

FeatureStoreBatchOnlineReadIntegTest.__module__ = __name__
FeatureStoreStreamIngestIntegTest.__module__ = __name__
StreamingFeatureViewIntegTest.__module__ = __name__

if __name__ == "__main__":
    absltest.main()
