"""End-to-end coverage for the WebSocket online-read transport, against a live Online Service.

The gap this closes. Every other test of this transport fakes the socket: the unit, adversarial and
stress suites inject a connection through the ``_ws_factory`` seam, so nothing in CI ever loads the
real ``websocket-client``, performs a real handshake, or proves the service answers a frame. And
nothing anywhere asserts *which transport served a read* -- so the whole suite would pass
identically if the WebSocket path silently fell back to HTTP on every request, which is exactly the
regression most likely to go unnoticed.

So these tests assert two things the rest cannot:

* the values returned over WebSocket are the ingested ones, and match what HTTP returns;
* ``ws_requests`` actually moved, and ``http_fallback_requests`` did not.

That second assertion is the point. A read that quietly degraded to HTTP still returns correct
data, which is why correctness checks alone cannot detect it.

Run it against a runtime built from your own Online Service changes by pointing it at a schema whose Online
Service is already up, which skips the ~15 minute SPCS provision::

    bazel test //tests/integ/snowflake/ml/feature_store:feature_store_ws_transport_integ_test \
        --test_env=SPEC_OFT_BUNDLE_REUSE_SCHEMA=MY_DB.MY_SCHEMA

Requires a PAT: the Online Service query endpoint rejects a session token, so the session must be
built with ``authenticator='PROGRAMMATIC_ACCESS_TOKEN'`` (``test_env_utils`` then promotes it into
``SNOWFLAKE_PAT``) or ``SNOWFLAKE_PAT`` must be set directly.
"""

import logging
import os
import uuid

import pandas as pd
from absl.testing import absltest
from common_utils import skip_unless_ws_route_served
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import online_service_ws_transport
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
)
from snowflake.ml.feature_store.online_service import OnlineServiceAccess
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    external_volume_manager,
    test_env_utils,
)

logger = logging.getLogger(__name__)

# `read_feature_view` splits a request at `_MAX_QUERY_REQUEST_ROWS = 10`, so the transport counters
# count *batches*, not reads. Kept local rather than imported so a change to the library constant
# shows up here as a failure to be looked at rather than silently rescaling the expectation.
MAX_QUERY_REQUEST_ROWS = 10


def _parse_reuse_target() -> tuple[str, str] | None:
    """Parse ``SPEC_OFT_BUNDLE_REUSE_SCHEMA=DB.SCHEMA``.

    Returns:
        ``(database, schema)``, or None when the variable is unset.

    Raises:
        RuntimeError: When set but not in ``DB.SCHEMA`` form.
    """
    raw = os.environ.get("SPEC_OFT_BUNDLE_REUSE_SCHEMA", "").strip()
    if not raw:
        return None
    parts = raw.split(".")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"SPEC_OFT_BUNDLE_REUSE_SCHEMA must be in the form DB.SCHEMA; got {raw!r}.")
    return parts[0], parts[1]


def setUpModule() -> None:
    """Attach to an already-provisioned schema when asked, instead of provisioning one.

    Without this the base class creates databases and a fresh Online Service, which is a ~15
    minute SPCS bring-up per run. Pointing at a schema whose service is already RUNNING is what
    makes this test usable against a runtime built from local Online Service changes.

    Same shape as ``feature_store_spec_oft_bundle_test``: populate the base class's
    ``_module_state`` and it skips provisioning and teardown entirely.

    Raises:
        RuntimeError: When the session has no warehouse configured.
    """
    target = _parse_reuse_target()
    if target is None:
        return
    reuse_db, reuse_schema = target

    session = test_env_utils.get_available_session()
    session_warehouse = session.get_current_warehouse()
    if not session_warehouse:
        raise RuntimeError("No warehouse is configured in the current session.")
    warehouse = session_warehouse.strip('"')

    dbm = db_manager.DBManager(session)
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
    user_entity = Entity(name="user_entity", join_keys=["USER_ID"], desc="User entity")
    fs.register_entity(user_entity)  # idempotent: warns rather than raises if it exists

    logger.info(
        "WS transport test: attached to %s.%s (warehouse=%s); skipping Online Service provisioning.",
        reuse_db,
        reuse_schema,
        warehouse,
    )
    StreamingFeatureViewIntegTestBase._module_state = {
        "session": session,
        "dbm": dbm,
        "evm": external_volume_manager.ExternalVolumeManager(session),
        "test_db": reuse_db,
        "dummy_db": reuse_db,
        "warehouse": warehouse,
        "test_schema": reuse_schema,
        "fs": fs,
        "user_entity": user_entity,
        "consumer_role": None,
        "_reuse_mode": True,
    }


class FeatureStoreWsTransportIntegTest(StreamingFeatureViewIntegTestBase):
    """Live WebSocket reads: real handshake, real frames, real values, attributed transport."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Set before any read: `_get_or_create_online_http_client` caches the transport on first
        # use, so flipping this afterwards measures the instance built under the old setting.
        os.environ[online_service_ws_transport.WS_ENABLED_ENV_VAR] = "true"

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop(online_service_ws_transport.WS_ENABLED_ENV_VAR, None)
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        # Owned here rather than on the base class, which does not define it.
        self._source_tables: list[str] = []
        skip_unless_ws_route_served(self, self.fs)

    def _fresh_transport(self, fs: FeatureStore) -> object:
        """Drop any cached transport so the next read builds one under the current env.

        Args:
            fs: The feature store whose cached transport should be discarded.

        Returns:
            The newly built transport, so a caller can snapshot its counters.
        """
        fs._online_http_client = None  # noqa: SLF001
        return fs._get_or_create_online_http_client()  # noqa: SLF001

    def _counters(self, transport: object) -> tuple[int, int]:
        """Read the transport's attribution counters.

        Args:
            transport: The transport returned by ``_get_or_create_online_http_client``.

        Returns:
            ``(ws_requests, http_fallback_requests)``; zeros when the plain HTTP client is in use,
            which has no counters at all.
        """
        return (
            int(getattr(transport, "ws_requests", 0) or 0),
            int(getattr(transport, "http_fallback_requests", 0) or 0),
        )

    def _register_online_fv(self, fs: FeatureStore, suffix: str, rows: list[tuple[str, float]]) -> str:
        """Create a Postgres-online feature view seeded with ``rows`` and wait for it to materialise.

        Args:
            fs: The feature store to register into.
            suffix: Unique suffix for the table and view names.
            rows: ``(entity_key, amount)`` pairs to ingest.

        Returns:
            The registered feature view's name.
        """
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.WS_SRC_{suffix}"
        self._source_tables.append(table_name)
        self._session.sql(
            f"CREATE OR REPLACE TABLE {table_name} (USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, AMOUNT FLOAT)"
        ).collect()
        values = ", ".join(
            f"({key!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), {amount})" for key, amount in rows
        )
        self._session.sql(f"INSERT INTO {table_name} VALUES {values}").collect()

        fv_name = f"WS_FV_{suffix}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.table(table_name),
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        fs.register_feature_view(fv, "v1")
        self._wait_offline_dt_rows(fs, fv_name, "v1")
        return fv_name

    def test_a_read_is_served_over_websocket_and_returns_the_ingested_values(self) -> None:
        """The headline: real handshake, real frames, right values, and WebSocket actually served it."""
        fs = self._create_feature_store()
        suffix = uuid.uuid4().hex[:8]
        key, amount = f"U_WS_{suffix}", 41.5
        fv_name = self._register_online_fv(fs, suffix, [(key, amount)])

        # Warm up through the polling helper so a still-materialising store does not fail the
        # measured read below.
        self._poll_online_read(fs, fv_name, "v1", keys=[[key]], desc="ws transport warmup")

        transport = self._fresh_transport(fs)
        self.assertIsInstance(
            transport,
            online_service_ws_transport.OnlineServiceWsTransport,
            "WS_ENABLED was set, so the feature store should have built the WebSocket transport",
        )
        before_ws, before_http = self._counters(transport)

        fv_live = fs.get_feature_view(fv_name, "v1")
        pdf = self._read_online_with_retry(fs, fv_live, keys=[[key]], require_non_null_cols=["AMOUNT"])

        self.assertIsInstance(pdf, pd.DataFrame)
        self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), amount, places=3)

        after_ws, after_http = self._counters(transport)
        self.assertGreater(
            after_ws,
            before_ws,
            "no request was served over WebSocket -- the read silently fell back to HTTP",
        )
        self.assertEqual(
            before_http,
            after_http,
            "a request fell back to HTTP; the values are still correct, which is why this needs "
            "asserting rather than eyeballing",
        )

    def test_websocket_and_http_return_the_same_frame(self) -> None:
        """Equivalence, which is the actual contract: the transport must be undetectable."""
        fs = self._create_feature_store()
        suffix = uuid.uuid4().hex[:8]
        key, amount = f"U_CMP_{suffix}", 73.25
        fv_name = self._register_online_fv(fs, suffix, [(key, amount)])
        self._poll_online_read(fs, fv_name, "v1", keys=[[key]], desc="ws/http comparison warmup")
        fv_live = fs.get_feature_view(fv_name, "v1")

        ws_transport = self._fresh_transport(fs)
        ws_pdf = self._read_online_with_retry(fs, fv_live, keys=[[key]], require_non_null_cols=["AMOUNT"])
        served_ws, _ = self._counters(ws_transport)
        self.assertGreater(served_ws, 0, "the WebSocket arm did not use WebSocket")

        os.environ[online_service_ws_transport.WS_ENABLED_ENV_VAR] = ""
        try:
            http_transport = self._fresh_transport(fs)
            self.assertNotIsInstance(
                http_transport,
                online_service_ws_transport.OnlineServiceWsTransport,
                "clearing WS_ENABLED should fall back to the plain HTTP client",
            )
            http_pdf = self._read_online_with_retry(fs, fv_live, keys=[[key]], require_non_null_cols=["AMOUNT"])
        finally:
            os.environ[online_service_ws_transport.WS_ENABLED_ENV_VAR] = "true"
            self._fresh_transport(fs)

        self.assertEqual(list(http_pdf.columns), list(ws_pdf.columns))
        pd.testing.assert_frame_equal(
            http_pdf.reset_index(drop=True),
            ws_pdf.reset_index(drop=True),
            check_dtype=False,
        )

    def test_a_multi_key_read_is_split_into_batches_over_one_connection(self) -> None:
        """More keys than the request cap must become several counted batches, not one.

        This is the behaviour that makes every latency figure for this transport read oddly: the
        public API splits at ``_MAX_QUERY_REQUEST_ROWS``, so a read of N keys is ``ceil(N/10)``
        round trips. Nothing else in the suite exercises that loop against a live service.
        """
        fs = self._create_feature_store()
        suffix = uuid.uuid4().hex[:8]
        rows = [(f"U_B{i:03d}_{suffix}", 100.0 + i) for i in range(MAX_QUERY_REQUEST_ROWS + 3)]
        fv_name = self._register_online_fv(fs, suffix, rows)
        keys = [[key] for key, _ in rows]

        # Every key, not just the first: the counted read below is a single call with no retry,
        # because a retry re-reads the whole key set and so adds another full set of batches to the
        # delta this test asserts on. Warming one key left the rest to materialise on the measured
        # read, which is exactly when that retry fires.
        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=keys,
            validate_fn=lambda pdf: self.assertEqual(
                len(rows), int(pdf["AMOUNT"].notna().sum()), "not every key has a value yet"
            ),
            desc="batching warmup",
        )

        transport = self._fresh_transport(fs)
        before_ws, before_http = self._counters(transport)

        fv_live = fs.get_feature_view(fv_name, "v1")
        pdf = fs.read_feature_view(fv_live, keys=keys, store_type=StoreType.ONLINE, as_pandas=True)

        got = dict(zip(pdf["USER_ID"], pdf["AMOUNT"]))
        for key, amount in rows:
            self.assertIn(key, got, f"{key} missing from a {len(rows)}-key read")
            self.assertAlmostEqual(float(got[key]), amount, places=3, msg=f"wrong value for {key}")

        after_ws, after_http = self._counters(transport)
        expected_batches = -(-len(rows) // MAX_QUERY_REQUEST_ROWS)
        self.assertEqual(
            expected_batches,
            (after_ws - before_ws) + (after_http - before_http),
            f"{len(rows)} keys should be {expected_batches} batches; the counters count batches, not reads",
        )
        self.assertEqual(before_http, after_http, "a batch fell back to HTTP")


if __name__ == "__main__":
    absltest.main()
