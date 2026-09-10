"""Integration tests for Online Service APIs (create/get/alter/drop)."""

import logging
import time
import uuid

from absl.testing import absltest
from feature_store_streaming_fv_integ_base import (
    wait_online_service_running_with_query_endpoint,
)

from fs_integ_test_base import FeatureStoreIntegTestBase
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store, online_service
from snowflake.ml.feature_store.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Ordered smallest to largest, matching the server-side tiers. The server currently accepts only
# increases, so the target is the next tier above whatever the service is at; once decreases are
# enabled server-side this is also where a downgrade case would pick its target.
_SIZES_ASCENDING = ("XS", "S", "M", "L", "XL", "2XL", "3XL")
# Substring of the server's rejection when resizing is not enabled for the account.
_NOT_ENABLED_MARKER = "not enabled"


class OnlineServiceIntegTest(FeatureStoreIntegTestBase):
    _ONLINE_SERVICE_BACKED = True

    def setUp(self) -> None:
        super().setUp()
        self.fs = FeatureStore(
            session=self._session,
            database=self.test_db,
            name=self.test_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=feature_store.CreationMode.CREATE_IF_NOT_EXIST,
        )

    def tearDown(self) -> None:
        # Drop Online Service before base tearDown removes the database. Still best-effort -- the
        # database drop reclaims the service anyway -- but logged, since a test may exit with a size
        # change in flight and a refusal here should not vanish silently.
        try:
            self.fs.drop_online_service()
        except Exception as e:
            logger.warning("Could not drop the Online Service during teardown: %s", e)
        super().tearDown()

    def test_create_online_service_poll_until_running_then_drop(self) -> None:
        # Producer must own the test schema; use the session role (mirrors bundle runner).
        producer = self._current_role()
        consumer = f"SNOWML_TEST_SPEC_OFT_C_{uuid.uuid4().hex[:8]}".upper()
        self._session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(consumer)}").collect()
        self._session.sql(f"GRANT ROLE {SqlIdentifier(consumer)} TO ROLE {self._session.get_current_role()}").collect()

        try:
            # This test verifies the full create/poll/drop cycle, so disable the helper's
            # reuse-if-already-running fast path.
            wait_online_service_running_with_query_endpoint(
                session=self._session,
                fs=self.fs,
                producer_role=producer,
                consumer_role=consumer,
                reuse_if_running=False,
            )
            self._assert_endpoint_urls_well_formed()
            self.fs.drop_online_service()
        finally:
            self._session.sql(f"DROP ROLE IF EXISTS {SqlIdentifier(consumer)}").collect()

    def test_alter_online_service_size_records_request_and_stays_serviceable(self) -> None:
        """A size change is accepted promptly and the service keeps serving while it converges.

        Convergence is deliberately not awaited: a size change is a Postgres compute-family
        failover that can run for hours, which does not fit a test's budget. What is asserted is
        the part a caller depends on immediately -- the request is recorded without blocking, and
        the service stays readable throughout, which is the ``UPDATING_SIZE`` contract.
        """
        # Check enablement before paying for a bring-up: the server checks the account gate before
        # it looks up the runtime, so on a schema with no Online Service yet a rejection naming the
        # gate means resizing is off, while any other rejection means it is on.
        if not self._is_resize_enabled():
            self.skipTest("Online Service resizing is not enabled on this account.")

        producer = self._current_role()
        consumer = f"SNOWML_TEST_SPEC_OFT_A_{uuid.uuid4().hex[:8]}".upper()
        self._session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(consumer)}").collect()
        self._session.sql(f"GRANT ROLE {SqlIdentifier(consumer)} TO ROLE {self._session.get_current_role()}").collect()

        try:
            wait_online_service_running_with_query_endpoint(
                session=self._session,
                fs=self.fs,
                producer_role=producer,
                consumer_role=consumer,
            )

            before = self.fs.get_online_service_status()
            self.assertEqual(before.status, "RUNNING")
            self.assertIsInstance(before.size, str)
            assert before.size is not None  # for type checker
            current = before.size.strip().upper()
            self.assertIn(current, _SIZES_ASCENDING, f"unrecognized size from server: {before.size!r}")

            # XS is a prototyping tier the server refuses to resize away from, and it is what
            # size-capped accounts get, so there is nothing to exercise there.
            if current == "XS":
                self.skipTest(f"Online Service is size {current}, which the server does not allow resizing.")
            index = _SIZES_ASCENDING.index(current)
            if index == len(_SIZES_ASCENDING) - 1:
                self.skipTest(f"Online Service is already at the largest size ({current}); nothing to upgrade to.")
            target = _SIZES_ASCENDING[index + 1]

            started = time.monotonic()
            result = self.fs.alter_online_service(size=target)
            elapsed = time.monotonic() - started
            self.assertEqual(result.status, "SUCCESS")
            # The call records intent and returns; the reconciler converges in the background. A
            # generous bound still catches the call having become synchronous.
            self.assertLess(elapsed, 60.0, f"alter_online_service blocked for {elapsed:.1f}s")

            # The server records the request and moves to UPDATING_SIZE in the same transaction, so
            # the change must be visible immediately: either still converging, or already finished
            # at the requested size. RUNNING at the old size would mean the request was dropped.
            after = self.fs.get_online_service_status()
            self.assertIn(
                after.status,
                ("UPDATING_SIZE", "RUNNING"),
                f"unexpected status after a size change: {after.status}",
            )
            if after.status == "RUNNING":
                self.assertIsNotNone(after.size, "expected a size when status=RUNNING")
                assert after.size is not None  # for type checker
                self.assertEqual(
                    after.size.strip().upper(),
                    target,
                    f"status returned to RUNNING at {after.size!r}, not the requested {target!r}",
                )
            # Whether the change is still in flight or already settled, online reads must not be
            # locked out -- this is the behavior that makes UPDATING_SIZE serviceable.
            online_service.assert_online_service_running_with_query_endpoint(
                self._session,
                self.fs._config.database,
                self.fs._config.schema,
            )
        finally:
            self._session.sql(f"DROP ROLE IF EXISTS {SqlIdentifier(consumer)}").collect()

    def _current_role(self) -> str:
        """Unquoted name of the session's current role, which must own the test schema."""
        role = self._session.get_current_role()
        self.assertIsNotNone(role, "session has no current role")
        assert role is not None  # for type checker
        return role.strip('"')

    def _is_resize_enabled(self) -> bool:
        """Whether this account admits Online Service resize requests.

        Called before any Online Service exists, so the request cannot succeed either way; only the
        reason it is refused is informative.
        """
        try:
            self.fs.alter_online_service(size="M")
        except Exception as e:
            enabled = _NOT_ENABLED_MARKER not in str(e).lower()
            # Logged either way: the skip message alone cannot say which rejection was seen, so an
            # unexpected skip would otherwise be undiagnosable from the test output.
            logger.info("Online Service resize probe refused (reading this as enabled=%s): %s", enabled, e)
            return enabled
        # A success here would mean an Online Service existed after all; treat resizing as enabled.
        return True

    def _assert_endpoint_urls_well_formed(self) -> None:
        """Endpoint URL contract: ``url`` and ``internal_url`` are always present and HTTP(S).
        ``privatelink_url`` is only emitted for PrivateLink-enabled accounts; when present, it must
        also be HTTP(S).
        """
        st = self.fs.get_online_service_status()
        self.assertEqual(st.status, "RUNNING")
        # The service was created without an explicit size, so the server default applies. Assert only
        # that a size round-trips: the exact default is a server-side choice and accounts may cap it.
        self.assertIsInstance(st.size, str)
        self.assertTrue(st.size, "expected a size when status=RUNNING")
        self.assertTrue(st.endpoints, "expected at least one endpoint when status=RUNNING")
        for ep in st.endpoints:
            self.assertTrue(
                ep.url.startswith(("http://", "https://")),
                f"endpoint {ep.name!r} url is not http(s): {ep.url!r}",
            )
            self.assertIsNotNone(ep.internal_url, f"endpoint {ep.name!r} missing internal_url")
            assert ep.internal_url is not None  # for type checker
            self.assertTrue(
                ep.internal_url.startswith(("http://", "https://")),
                f"endpoint {ep.name!r} internal_url is not http(s): {ep.internal_url!r}",
            )
            if ep.privatelink_url is not None:
                self.assertIsInstance(ep.privatelink_url, str)
                self.assertTrue(
                    ep.privatelink_url.startswith(("http://", "https://")),
                    f"endpoint {ep.name!r} privatelink_url is not http(s): {ep.privatelink_url!r}",
                )


if __name__ == "__main__":
    absltest.main()
