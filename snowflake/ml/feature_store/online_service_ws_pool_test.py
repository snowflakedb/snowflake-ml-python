"""Unit tests for online_service_ws_pool (spread, recycling, splay, age-based retirement).

Each of these pins an obligation the service cannot compensate for. The gateway assigns an
instance per connection at random and never moves it, so a client that reuses one connection
pins itself to one instance; and connections that expire together produce a burst of
authentications. Both are client-side behaviours, invisible from the server.
"""

import random
import threading
import time
from typing import Any, Callable, cast
from unittest.mock import patch

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store import (
    online_service_transport,
    online_service_ws_client,
    online_service_ws_pool,
)

_URL = "wss://svc.example.app/api/v1/query/ws"
# A second endpoint, for the cases that need two pool keys. A different feature view no longer
# is one: every read of an endpoint shares its connections.
_OTHER_URL = "wss://other.example.app/api/v1/query/ws"


class _FakeClient:
    """Counts what it was asked to do, so selection can be asserted without a socket."""

    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        # Recorded rather than discarded. The pool forwards the session it was built with and
        # its own clock, and a fake that drops them lets either forwarding be deleted without
        # any test noticing -- there is no integration coverage of this pool to catch it.
        self.session = kwargs.get("session")
        self._clock: Callable[[], float] = kwargs.get("_clock") or time.monotonic
        self.posts = 0
        self.batches = 0
        self.closed = False
        # Counted, not just flagged: a bool cannot distinguish one close from two, so
        # idempotence and close-exactly-once are unassertable without this.
        self.close_calls = 0
        self.urls: list[str] = []
        self.sent: list[dict[str, Any]] = []
        self.server_hostname: str | None = None
        # Set on first use, as the real client sets it at the dial.
        self.connected_at: float | None = None
        _FakeClient.instances.append(self)

    @property
    def is_connected(self) -> bool:
        return self.connected_at is not None and not self.closed

    def _dial(self, url: str) -> None:
        """Record the lazy dial the real client performs on first use.

        Args:
            url: The URL the pool asked for.
        """
        self.urls.append(url)
        if self.connected_at is None:
            self.connected_at = self._clock()

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        del timeout
        self._dial(url)
        self.sent.append(body)
        self.posts += 1
        return 200, b"{}"

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        del timeout
        self._dial(url)
        self.sent.extend(bodies)
        self.batches += 1
        return [(200, b"{}") for _ in bodies]

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def _pool(**kwargs: Any) -> online_service_ws_pool.OnlineServiceWsPool:
    """Build a pool wired to fake clients with a deterministic clock and rng.

    Args:
        **kwargs: Overrides passed to the pool.

    Returns:
        The pool.
    """
    # Release anything a previous test left gated. Several tests hold a caller in flight to
    # observe a reservation, and a gated client that is never released costs its full
    # ``release.wait`` timeout -- on a loaded machine, enough of those turn a 3-second suite
    # into a 15-minute one. Clearing them here bounds the cost to the test that caused it
    # rather than leaving it to thread scheduling.
    for previous in _FakeClient.instances:
        release = getattr(previous, "release", None)
        if release is not None:
            release.set()
    _FakeClient.instances = []
    defaults: dict[str, Any] = {
        "_client_factory": _FakeClient,
        "_rng": random.Random(1234),
        "_clock": lambda: 0.0,
    }
    defaults.update(kwargs)
    return online_service_ws_pool.OnlineServiceWsPool(**defaults)


def _body(name: str = "my_fv", version: str = "v1") -> dict[str, Any]:
    """Build a minimal request body.

    Args:
        name: Feature view name.
        version: Feature view version.

    Returns:
        A body the pool can key on.
    """
    return {"name": name, "version": version, "object_type": "feature_view", "request_rows": []}


def _fill(pool: online_service_ws_pool.OnlineServiceWsPool, n: int, url: str = _URL) -> None:
    """Open up to ``n`` connections, the way ``n`` concurrent callers would.

    The pool grows only while every connection it holds is busy, so sequential reads alone
    keep one. Each selection is held reserved while the next is made, then all are released
    without sending, so no request budget is spent.

    Args:
        pool: The pool to fill.
        n: Concurrent callers to simulate.
        url: The endpoint whose connections to open.
    """
    key = pool.connection_key(url)
    held = [pool._select(key) for _ in range(n)]  # noqa: SLF001 - white-box: reserve without sending
    for pooled in held:
        pool._finish(key, pooled, requests=0, evict=False)  # noqa: SLF001


class ConnectionKeyTest(absltest.TestCase):
    def test_every_read_of_an_endpoint_shares_a_key(self) -> None:
        """Nothing is bound to a connection at the handshake -- each message names what it reads
        -- so keying by object only multiplied sockets by the number of feature views read."""
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key
        self.assertEqual(key(_URL), key(_URL))

    def test_key_separates_endpoints(self) -> None:
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key
        self.assertNotEqual(key(_URL), key(_OTHER_URL))


class PoolSpreadTest(absltest.TestCase):
    def test_sequential_reads_hold_one_connection(self) -> None:
        """Each connection is an authentication and a slot in a shared cap; a caller that never
        has two in flight has no use for a second."""
        pool = _pool(size=3)
        for _ in range(5):
            pool.post_json(_URL, _body(), 30.0)
        self.assertEqual(pool.connection_count, 1)

    def test_concurrent_demand_grows_the_pool_to_its_size_and_no_further(self) -> None:
        pool = _pool(size=3)
        _fill(pool, 5)
        self.assertEqual(pool.connection_count, 3)

    def test_round_robin_actually_rotates(self) -> None:
        """Always reusing connection 0 pins all load to whichever instance it landed on."""
        pool = _pool(size=3)
        _fill(pool, 3)
        for _ in range(9):
            pool.post_json(_URL, _body(), 30.0)

        posts = sorted(client.posts for client in _FakeClient.instances)
        self.assertEqual(posts, [3, 3, 3])

    def test_least_outstanding_prefers_the_idle_connection(self) -> None:
        """The alternative policy must not degenerate into always picking the same one."""
        pool = _pool(size=2, least_outstanding=True)
        _fill(pool, 2)
        for _ in range(4):
            pool.post_json(_URL, _body(), 30.0)

        # With nothing in flight every connection ties, so the tie-break must still spread:
        # ``min`` alone returns connection 0 on every tie and splits these 3/1.
        self.assertEqual(pool.connection_count, 2)
        self.assertEqual(sorted(client.posts for client in _FakeClient.instances), [2, 2])

    def test_feature_views_share_their_endpoint_s_connections(self) -> None:
        pool = _pool(size=1)
        pool.post_json(_URL, _body("fv_a"), 30.0)
        pool.post_json(_URL, _body("fv_b"), 30.0)
        self.assertEqual(pool.connection_count, 1)

    def test_separate_endpoints_get_separate_connections(self) -> None:
        pool = _pool(size=1)
        pool.post_json(_URL, _body(), 30.0)
        pool.post_json(_OTHER_URL, _body(), 30.0)
        self.assertEqual(pool.connection_count, 2)


class RecyclingTest(absltest.TestCase):
    def test_splay_produces_distinct_thresholds(self) -> None:
        """A constant threshold would pass a bounds check and still leave every connection
        expiring in the same instant, which is the burst this exists to prevent."""
        pool = _pool(size=8, max_requests_per_connection=1000, splay_fraction=0.25)
        _fill(pool, 8)

        budgets = pool.request_budgets()
        self.assertLen(budgets, 8)
        self.assertGreater(len(set(budgets)), 1)

    def test_splay_stays_within_its_fraction(self) -> None:
        """Spread must not drift so far that the nominal budget stops meaning anything."""
        pool = _pool(size=16, max_requests_per_connection=1000, splay_fraction=0.25)
        _fill(pool, 16)

        self.assertLen(pool.request_budgets(), 16)
        for budget in pool.request_budgets():
            self.assertGreaterEqual(budget, 750)
            self.assertLessEqual(budget, 1250)

    def test_splay_can_be_switched_off(self) -> None:
        """A deployment that wants deterministic recycling should be able to have it."""
        pool = _pool(size=4, max_requests_per_connection=500, splay_fraction=0.0)
        _fill(pool, 4)
        self.assertEqual(pool.request_budgets(), [500] * 4)

    def test_connection_is_replaced_once_over_its_request_budget(self) -> None:
        """Recycling bounds how long one connection — and so one instance — serves a caller."""
        pool = _pool(size=1, max_requests_per_connection=3, splay_fraction=0.0)
        for _ in range(7):
            pool.post_json(_URL, _body(), 30.0)

        self.assertGreater(len(_FakeClient.instances), 1)
        self.assertTrue(any(client.closed for client in _FakeClient.instances))

    def test_connection_is_replaced_once_over_age(self) -> None:
        """Retiring before the service's own limit keeps the close off the request path."""
        now = [0.0]
        pool = _pool(size=1, max_connection_age_seconds=100.0, _clock=lambda: now[0])
        pool.post_json(_URL, _body(), 30.0)
        first = _FakeClient.instances[0]

        now[0] = 150.0
        pool.post_json(_URL, _body(), 30.0)

        self.assertTrue(first.closed)
        self.assertLen(_FakeClient.instances, 2)

    def test_in_flight_connection_is_not_recycled(self) -> None:
        """Two-phase retirement: a due connection leaves rotation at once but closes only once drained.

        Replacing a connection mid-request would abort work already sent, and waiting for it to
        look idle before retiring it would never retire a busy one.
        """
        now = [0.0]
        pool = _pool(
            size=1,
            max_connection_age_seconds=10.0,
            splay_fraction=0.0,
            _client_factory=_FirstGatedClient,
            _clock=lambda: now[0],
        )
        done = threading.Event()

        def worker() -> None:
            pool.post_json(_URL, _body(), 5.0)
            done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        held = cast(_FirstGatedClient, _FakeClient.instances[0])
        self.assertTrue(held.entered.wait(timeout=5), "the worker never reached the client")

        now[0] = 500.0  # past the age, while the first request is still in flight
        pool.post_json(_URL, _body(), 5.0)

        self.assertLen(_FakeClient.instances, 2, "the due connection was not replaced")
        fresh = _FakeClient.instances[1]
        self.assertEqual(fresh.posts, 1, "the new request did not go to the fresh slot")
        self.assertFalse(held.closed, "a connection was closed while a request ran on it")
        self.assertEqual(pool.draining_connection_count, 1)

        held.release.set()
        self.assertTrue(done.wait(timeout=5))
        thread.join(timeout=5)
        self.assertTrue(held.closed, "the retired connection was not closed once it drained")
        self.assertEqual(pool.draining_connection_count, 0)
        self.assertFalse(fresh.closed)

    def test_age_ceiling_is_splayed_per_connection(self) -> None:
        """Without an advertised age, connections opened together must not all hit the ceiling at once."""
        pool = _pool(size=16, max_connection_age_seconds=720.0, splay_fraction=0.25)
        _fill(pool, 16)

        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        ages = [pooled.max_age_seconds for pooled in pool._connections[key]]  # noqa: SLF001
        self.assertLen(ages, 16)
        self.assertGreater(len(set(ages)), 1, "every connection got the same age ceiling")
        for age in ages:
            self.assertGreaterEqual(age, 540.0)
            self.assertLessEqual(age, 720.0, "the configured age is a ceiling and must not be exceeded")

    def test_age_splay_can_be_switched_off(self) -> None:
        pool = _pool(size=4, max_connection_age_seconds=720.0, splay_fraction=0.0)
        _fill(pool, 4)

        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        self.assertEqual([pooled.max_age_seconds for pooled in pool._connections[key]], [720.0] * 4)  # noqa: SLF001


class BatchingTest(absltest.TestCase):
    def test_a_batch_goes_to_one_connection(self) -> None:
        """Pipelining only saves round trips if the whole burst shares a connection."""
        pool = _pool(size=4)
        results = pool.post_json_many(_URL, [_body(), _body(), _body()], 30.0)

        self.assertLen(results, 3)
        self.assertEqual(sum(client.batches for client in _FakeClient.instances), 1)

    def test_a_batch_counts_every_body_against_the_budget(self) -> None:
        """A batch that carries a connection past its budget must retire it like single requests do."""
        pool = _pool(size=1, max_requests_per_connection=5, splay_fraction=0.0)
        pool.post_json_many(_URL, [_body() for _ in range(5)], 30.0)
        first = _FakeClient.instances[0]

        pool.post_json(_URL, _body(), 30.0)

        self.assertTrue(first.closed, "a connection past its budget was not retired")
        self.assertLen(_FakeClient.instances, 2)

    def test_a_refused_dial_in_a_batch_evicts_the_slot(self) -> None:
        class _RefusingBatch(_FakeClient):
            def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
                raise online_service_ws_client.OnlineServiceWsDialError("at capacity", status=503)

        pool = _pool(size=2, _client_factory=_RefusingBatch)
        with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError):
            pool.post_json_many(_URL, [_body(), _body()], 30.0)

        self.assertEqual(pool.connection_count, 0)
        self.assertTrue(_FakeClient.instances[0].closed)

    def test_empty_batch_opens_nothing(self) -> None:
        """An empty read should cost nothing."""
        pool = _pool()
        self.assertEqual(pool.post_json_many(_URL, [], 30.0), [])
        self.assertEqual(pool.connection_count, 0)


class LifecycleTest(absltest.TestCase):
    def test_close_closes_every_connection_and_is_idempotent(self) -> None:
        """Leaked connections hold a server slot until the service times them out."""
        pool = _pool(size=3)
        for _ in range(3):
            pool.post_json(_URL, _body(), 30.0)
        pool.close()
        pool.close()

        self.assertTrue(all(client.closed for client in _FakeClient.instances))
        self.assertEqual(pool.connection_count, 0)

    def test_post_after_close_raises(self) -> None:
        """Matches the HTTP client, so callers can treat the two identically."""
        pool = _pool()
        pool.close()
        with self.assertRaises(RuntimeError):
            pool.post_json(_URL, _body(), 30.0)

    def test_close_survives_a_failing_connection(self) -> None:
        """Cleanup must never raise, or a failed close would mask the real error."""

        class _ExplodingClient(_FakeClient):
            def close(self) -> None:
                raise OSError("close failed")

        pool = _pool(size=1, _client_factory=_ExplodingClient)
        pool.post_json(_URL, _body(), 30.0)
        pool.close()  # must not raise


class ThreadSafetyTest(absltest.TestCase):
    """A WebSocket permits one reader and one writer.

    Two callers interleaving sends on one socket can each read the other's response, which
    returns another request's feature values with no error raised. These pin the exclusion
    that prevents it — the failure mode is silent, so a regression here would not announce
    itself.
    """

    def test_a_connection_is_never_used_by_two_callers_at_once(self) -> None:
        """Round-robin hands the same connection to concurrent callers by design, so the
        pool must serialize use of each one."""
        overlaps: list[int] = []

        class _OverlapDetectingClient:
            def __init__(self, **kwargs: Any) -> None:
                del kwargs
                self.inside = 0
                self.guard = threading.Lock()
                self.closed = False

            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                del url, body, timeout
                with self.guard:
                    self.inside += 1
                    if self.inside > 1:
                        overlaps.append(id(self))
                time.sleep(0.0005)  # widen the window a real send/recv would occupy
                with self.guard:
                    self.inside -= 1
                return 200, b"{}"

            def close(self) -> None:
                self.closed = True

        pool = _pool(size=4, _client_factory=_OverlapDetectingClient, splay_fraction=0.0)

        def worker() -> None:
            for _ in range(40):
                pool.post_json(_URL, _body(), 1.0)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEmpty(overlaps)

    def test_pool_never_exceeds_its_size_under_concurrent_growth(self) -> None:
        """Growing the pool is a read-then-write, so two callers could both append."""
        pool = _pool(size=3, splay_fraction=0.0)

        def worker() -> None:
            for _ in range(50):
                pool.post_json(_URL, _body(), 1.0)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # How many it opens depends on how often all were busy at once; never more than its size.
        self.assertGreaterEqual(pool.connection_count, 1)
        self.assertLessEqual(pool.connection_count, 3)

    def test_outstanding_counts_return_to_zero(self) -> None:
        """A leaked count would make a connection look permanently busy and never recycle."""
        pool = _pool(size=2, splay_fraction=0.0)

        def worker() -> None:
            for _ in range(40):
                pool.post_json(_URL, _body(), 1.0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for connections in pool._connections.values():  # noqa: SLF001 - white-box, repo style
            for pooled in connections:
                self.assertEqual(pooled.outstanding, 0)


class IntrospectionTest(absltest.TestCase):
    """The pool must be describable, not just countable.

    The server side replaced a bare counter with a registry because a number cannot be
    addressed, only reported. The same applies here: diagnosing uneven instance spread or a
    wedged slot needs per-connection detail.
    """

    def test_connections_snapshot_has_distinct_ids_and_real_counters(self) -> None:
        """Without ids, two connections in a log line are indistinguishable."""
        pool = _pool(size=3, splay_fraction=0.0)
        _fill(pool, 3)
        for _ in range(6):
            pool.post_json(_URL, _body(), 30.0)

        infos = pool.connections()
        self.assertLen(infos, 3)
        self.assertLen({info.connection_id for info in infos}, 3)
        self.assertEqual(sum(info.requests for info in infos), 6)
        self.assertTrue(all(info.outstanding == 0 for info in infos))

    def test_snapshot_reports_the_key_each_connection_is_scoped_to(self) -> None:
        """Connections are pooled per endpoint, so the snapshot says which."""
        pool = _pool(size=1)
        pool.post_json(_URL, _body("fv_a"), 30.0)
        pool.post_json(_OTHER_URL, _body("fv_b"), 30.0)

        keys = {info.key for info in pool.connections()}
        self.assertEqual(keys, {(_URL,), (_OTHER_URL,)})

    def test_server_hostnames_counts_are_empty_without_reported_instances(self) -> None:
        """A fake that reports no instance must not invent one."""
        pool = _pool(size=2)
        pool.post_json(_URL, _body(), 30.0)
        self.assertEmpty(pool.server_hostnames())

    def test_slot_count_and_live_count_differ_while_a_dial_is_pending(self) -> None:
        """Conflating the two is what let a dead slot look like a working connection."""

        class _NeverConnects(_FakeClient):
            @property
            def is_connected(self) -> bool:
                return False

        pool = _pool(size=2, _client_factory=_NeverConnects)
        pool.post_json(_URL, _body(), 30.0)

        self.assertEqual(pool.connection_count, 1)
        self.assertEqual(pool.live_connection_count, 0)


class DialRefusalTest(absltest.TestCase):
    def test_a_refused_dial_does_not_keep_occupying_a_slot(self) -> None:
        """Slots are a shared per-instance resource with no server-side per-caller share, so a
        caller holding dead ones reduces capacity for everybody."""

        class _RefusingClient(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                del url, body, timeout
                raise online_service_ws_client.OnlineServiceWsDialError("at capacity", status=503)

        pool = _pool(size=4, _client_factory=_RefusingClient)
        for _ in range(6):
            with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError):
                pool.post_json(_URL, _body(), 30.0)

        self.assertEqual(pool.connection_count, 0)
        self.assertTrue(all(client.closed for client in _FakeClient.instances))

    def test_pool_recovers_once_dials_succeed_again(self) -> None:
        """A transient capacity refusal must not permanently shrink the pool."""
        refuse = [True]

        class _FlakyClient(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                if refuse[0]:
                    raise online_service_ws_client.OnlineServiceWsDialError("at capacity", status=503)
                return super().post_json(url, body, timeout)

        pool = _pool(size=2, _client_factory=_FlakyClient)
        with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError):
            pool.post_json(_URL, _body(), 30.0)
        self.assertEqual(pool.connection_count, 0)

        refuse[0] = False
        _fill(pool, 2)
        for _ in range(4):
            pool.post_json(_URL, _body(), 30.0)
        self.assertEqual(pool.connection_count, 2)


class DialFailureEvictionTest(absltest.TestCase):
    """A slot held by a connection that never opened is capacity taken from every other caller."""

    def test_a_refused_dial_still_evicts_the_slot(self) -> None:
        """Annotating must not shadow the eviction: a slot held by a dead connection is
        capacity taken from every other caller."""

        class _Refusing(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                raise online_service_ws_client.OnlineServiceWsDialError("refused", status=503)

        pool = _pool(size=1, _client_factory=_Refusing)

        with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError) as caught:
            pool.post_json(_URL, _body(), 30.0)

        self.assertTrue(caught.exception.at_capacity)
        self.assertEqual(pool.connection_count, 0)


class _AdvertisingClient(_FakeClient):
    """A fake whose connection, once used, reports the lifetime the service advertised."""

    advertised: float | None = 60.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_max_conn_age_seconds: float | None = None

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        self.server_max_conn_age_seconds = self.advertised  # learned at the dial, as the real client does
        return super().post_json(url, body, timeout)


class AdvertisedAgeRetirementTest(absltest.TestCase):
    """The service advertises each connection's own age; retiring under it means the client closes first."""

    def _connection(self, advertised: float | None, *, connected_at: float | None) -> Any:
        client = _AdvertisingClient()
        client.server_max_conn_age_seconds = advertised
        # The dial time lives on the client, not the slot, because the client re-dials
        # underneath the pool. See ReconnectRetirementTest for the behaviour that depends on it.
        client.connected_at = connected_at
        pooled = online_service_ws_pool._PooledConnection(
            client=cast(online_service_ws_client.OnlineServiceWsClient, client), request_budget=10**9, created_at=0.0
        )
        return pooled

    def test_retires_a_margin_under_the_advertised_age(self) -> None:
        pooled = self._connection(60.0, connected_at=0.0)
        self.assertFalse(pooled.is_retired(54.9, max_age_seconds=720.0))
        self.assertTrue(pooled.is_retired(55.0, max_age_seconds=720.0))

    def test_age_runs_from_the_dial_not_from_slot_creation(self) -> None:
        """A replacement can sit unused after a retirement; the service's clock starts at the dial."""
        pooled = self._connection(60.0, connected_at=100.0)
        self.assertFalse(pooled.is_retired(150.0, max_age_seconds=720.0))
        self.assertTrue(pooled.is_retired(155.0, max_age_seconds=720.0))

    def test_short_ages_get_a_proportional_margin(self) -> None:
        """A fixed 5s margin would retire a 2s connection at birth."""
        pooled = self._connection(2.0, connected_at=0.0)
        self.assertFalse(pooled.is_retired(1.7, max_age_seconds=720.0))
        self.assertTrue(pooled.is_retired(1.8, max_age_seconds=720.0))

    def test_configured_age_stays_the_ceiling(self) -> None:
        pooled = self._connection(3600.0, connected_at=0.0)
        self.assertTrue(pooled.is_retired(720.0, max_age_seconds=720.0))

    def test_an_undialed_connection_is_not_measured_against_an_advertised_age(self) -> None:
        """A slot whose client has no socket has no dial to date the advertised age from."""
        pooled = self._connection(60.0, connected_at=None)
        self.assertFalse(pooled.is_retired(10_000.0, max_age_seconds=720_000.0))

    def test_without_an_advertised_age_only_the_configured_age_applies(self) -> None:
        pooled = self._connection(None, connected_at=0.0)
        self.assertFalse(pooled.is_retired(719.0, max_age_seconds=720.0))
        self.assertTrue(pooled.is_retired(720.0, max_age_seconds=720.0))

    def test_pool_replaces_the_connection_before_the_advertised_age(self) -> None:
        now = [0.0]
        _AdvertisingClient.advertised = 60.0
        pool = _pool(size=1, _client_factory=_AdvertisingClient, _clock=lambda: now[0])
        pool.post_json(_URL, _body(), 30.0)
        first = _FakeClient.instances[0]

        now[0] = 54.0
        pool.post_json(_URL, _body(), 30.0)
        self.assertLen(_FakeClient.instances, 1)

        now[0] = 56.0
        pool.post_json(_URL, _body(), 30.0)
        self.assertTrue(first.closed)
        self.assertLen(_FakeClient.instances, 2)


class _GatedClient(_FakeClient):
    """Blocks inside ``post_json`` until released, holding the connection in flight."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        self.entered.set()
        self.release.wait(timeout=10)
        return super().post_json(url, body, timeout)


class DiscardAllTest(absltest.TestCase):
    """``discard_all`` must not close a connection out from under the caller using it.

    ``_select`` reserves a connection (``outstanding += 1``) before releasing the lock, so a
    connection with outstanding work has a caller mid-request on it. The trigger for
    ``discard_all`` is an INTERNAL error on *one* connection, so closing eagerly would fail every
    concurrent in-flight request across the pool rather than the one that went wrong.
    """

    def test_an_idle_connection_is_closed_immediately(self) -> None:
        """Nothing is in flight, so there is no reason to defer."""
        pool = _pool(size=1)
        pool.post_json(_URL, _body(), 1.0)
        client = _FakeClient.instances[0]
        self.assertFalse(client.closed)

        self.assertEqual(1, pool.discard_all())
        self.assertTrue(client.closed)
        self.assertEqual(0, pool.connection_count)

    def test_a_busy_connection_is_not_closed_until_its_request_finishes(self) -> None:
        """The caller mid-request must be allowed to finish, then the socket closes."""
        pool = _pool(size=1, _client_factory=_GatedClient)
        done = threading.Event()

        def worker() -> None:
            pool.post_json(_URL, _body(), 5.0)
            done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        client = cast(_GatedClient, _FakeClient.instances[0])
        self.assertTrue(client.entered.wait(timeout=5), "the worker never reached the client")

        # Discard while the request is in flight.
        self.assertEqual(1, pool.discard_all())
        self.assertFalse(client.closed, "a connection was closed while a request ran on it")
        # Out of rotation already, so no new request can pick it.
        self.assertEqual(0, pool.connection_count)

        client.release.set()
        self.assertTrue(done.wait(timeout=10), "the in-flight request never completed")
        thread.join(timeout=5)
        self.assertTrue(client.closed, "a drained connection was never closed")
        self.assertEqual(0, pool.draining_connection_count)

    def test_close_hands_a_draining_connection_to_its_caller_rather_than_leaking_it(self) -> None:
        """A connection still draining when the pool closes must close, but not under its caller.

        ``close`` is permanent, which makes closing the socket eagerly tempting -- nothing will
        be handed out again either way. But a draining connection has a caller holding its lock,
        most likely blocked in ``recv``, and a WebSocket close sends a frame and waits for the
        server's reply: that is a second thread on the socket. The socket closes when its caller
        returns instead, bounded by that caller's own deadline.
        """
        pool = _pool(size=1, _client_factory=_GatedClient)

        def worker() -> None:
            pool.post_json(_URL, _body(), 5.0)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        client = cast(_GatedClient, _FakeClient.instances[0])
        self.assertTrue(client.entered.wait(timeout=5))
        pool.discard_all()

        pool.close()
        self.assertFalse(client.closed, "close() closed a socket its caller was still reading")
        self.assertEqual(1, pool.draining_connection_count)

        client.release.set()
        thread.join(timeout=5)
        self.assertTrue(client.closed, "the draining connection was never closed")
        self.assertEqual(1, client.close_calls, "the connection was closed more than once")
        self.assertEqual(0, pool.draining_connection_count)


class _FirstGatedClient(_GatedClient):
    """Gates only the first instance, so a replacement opened while it is held answers at once."""

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        if self is _FakeClient.instances[0]:
            return super().post_json(url, body, timeout)
        return _FakeClient.post_json(self, url, body, timeout)


def _run_in_thread(target: Callable[[], Any]) -> tuple[threading.Thread, threading.Event]:
    """Start ``target`` on a daemon thread and return it with an event set once it returns."""
    done = threading.Event()

    def run() -> None:
        try:
            target()
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, done


class SelectionUnderLoadTest(absltest.TestCase):
    def test_round_robin_skips_a_busy_connection_when_an_idle_one_exists(self) -> None:
        """Advancing onto a busy connection queues the caller behind a slow batch while others idle."""
        pool = _pool(size=2, _client_factory=_FirstGatedClient)
        thread, done = _run_in_thread(lambda: pool.post_json(_URL, _body(), 5.0))
        held = cast(_FirstGatedClient, _FakeClient.instances[0])
        self.assertTrue(held.entered.wait(timeout=5))

        pool.post_json(_URL, _body(), 5.0)  # grows the pool: connection 1
        # The cursor now points back at connection 0, which is still busy.
        pool.post_json(_URL, _body(), 5.0)

        self.assertEqual(_FakeClient.instances[1].posts, 2, "a caller was sent to the busy connection")
        held.release.set()
        self.assertTrue(done.wait(timeout=5))
        thread.join(timeout=5)


class _BlockingCloseClient(_FakeClient):
    """Its close blocks until released, like a real close waiting on the server's close frame."""

    closing = threading.Event()
    release = threading.Event()

    def close(self) -> None:
        _BlockingCloseClient.closing.set()
        _BlockingCloseClient.release.wait(timeout=10)
        super().close()


class CloseOutsideTheLockTest(absltest.TestCase):
    def setUp(self) -> None:
        _BlockingCloseClient.closing = threading.Event()
        _BlockingCloseClient.release = threading.Event()

    def tearDown(self) -> None:
        _BlockingCloseClient.release.set()

    def test_a_slow_close_on_retirement_does_not_block_other_callers(self) -> None:
        """A close holding the pool lock stalls every selection for every feature view."""
        pool = _pool(size=1, max_requests_per_connection=1, splay_fraction=0.0, _client_factory=_BlockingCloseClient)
        pool.post_json(_URL, _body(), 5.0)  # the connection is now over its budget

        # The next selection retires it, and it is idle, so it is closed straight away.
        thread, done = _run_in_thread(lambda: pool.post_json(_URL, _body(), 5.0))
        self.assertTrue(_BlockingCloseClient.closing.wait(timeout=5), "the retired connection was never closed")

        reader, read = _run_in_thread(lambda: pool.connection_count)
        self.assertTrue(read.wait(timeout=2), "another caller blocked on the pool lock during a close")

        _BlockingCloseClient.release.set()
        self.assertTrue(done.wait(timeout=5))
        thread.join(timeout=5)
        reader.join(timeout=5)


class DialFailureWithOtherCallersTest(absltest.TestCase):
    def test_a_failed_dial_does_not_close_a_client_another_caller_reserved(self) -> None:
        """Eviction removes the slot at once, but the close waits for the last caller reserved on it."""
        attempts: list[int] = []
        entered = threading.Event()
        release = threading.Event()

        class _FailsFirstDial(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                # Like the real client: a closed one refuses, rather than answering as if open. Without
                # this the reserved caller's request "succeeds" even when eviction closed it underneath.
                if self.closed:
                    raise RuntimeError("OnlineServiceWsClient is closed")
                attempts.append(len(attempts))
                if len(attempts) == 1:
                    entered.set()
                    release.wait(timeout=10)
                    raise online_service_ws_client.OnlineServiceWsDialError("refused", status=503)
                return super().post_json(url, body, timeout)

        pool = _pool(size=1, _client_factory=_FailsFirstDial)
        first, first_done = _run_in_thread(
            lambda: self.assertRaises(
                online_service_ws_client.OnlineServiceWsDialError, pool.post_json, _URL, _body(), 5.0
            )
        )
        self.assertTrue(entered.wait(timeout=5))

        # A second caller reserves the same connection and waits on its lock.
        results: list[tuple[int, bytes]] = []
        second, second_done = _run_in_thread(lambda: results.append(pool.post_json(_URL, _body(), 5.0)))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with pool._select_lock:  # noqa: SLF001 - white-box
                key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
                if pool._connections[key][0].outstanding == 2:  # noqa: SLF001
                    break
            time.sleep(0.01)

        release.set()  # the first caller's dial now fails
        self.assertTrue(first_done.wait(timeout=5))
        self.assertTrue(second_done.wait(timeout=5))
        first.join(timeout=5)
        second.join(timeout=5)

        client = _FakeClient.instances[0]
        self.assertEqual(results, [(200, b"{}")], "the reserved caller's request did not get through")
        self.assertEqual(pool.connection_count, 0, "the failed slot stayed in rotation")
        self.assertTrue(client.closed, "the evicted connection was not closed once its last caller finished")
        self.assertEqual(pool.draining_connection_count, 0)


class AccountingLockTest(absltest.TestCase):
    def test_counters_are_only_written_under_the_pool_lock(self) -> None:
        """``outstanding``/``requests``/``retiring`` are read-modify-write and drive the close decision."""
        pool = _pool(size=2, max_requests_per_connection=3, splay_fraction=0.0)
        lock = pool._select_lock  # noqa: SLF001 - white-box
        unguarded: list[str] = []
        original = online_service_ws_pool._PooledConnection

        class _Checked(original):  # type: ignore[misc, valid-type]
            def __setattr__(self, name: str, value: Any) -> None:
                if name in ("outstanding", "requests", "retiring") and not lock.locked():
                    unguarded.append(name)
                super().__setattr__(name, value)

        online_service_ws_pool._PooledConnection = _Checked  # type: ignore[misc]
        try:
            for _ in range(10):
                pool.post_json(_URL, _body(), 5.0)
            pool.post_json_many(_URL, [_body(), _body()], 5.0)
            pool.discard_all()
        finally:
            online_service_ws_pool._PooledConnection = original  # type: ignore[misc]

        self.assertEqual(unguarded, [], f"counters written without the pool lock: {sorted(set(unguarded))}")


class QueueTimeTest(absltest.TestCase):
    """A caller's timeout covers its wait for the connection, not just the request after it."""

    def test_a_caller_queued_past_its_timeout_fails_instead_of_waiting_on(self) -> None:
        pool = _pool(size=1, _client_factory=_GatedClient)
        holder, holder_done = _run_in_thread(lambda: pool.post_json(_URL, _body(), 30.0))
        deadline = time.monotonic() + 5
        while not _FakeClient.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        gated = cast(_GatedClient, _FakeClient.instances[0])
        self.assertTrue(gated.entered.wait(timeout=5))

        started = time.monotonic()
        try:
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as caught:
                pool.post_json(_URL, _body(), 0.2)
            waited = time.monotonic() - started
        finally:
            gated.release.set()
            holder.join(timeout=5)
        self.assertTrue(holder_done.is_set())

        self.assertIsInstance(caught.exception.original_exception, TimeoutError)
        # The transport counts wire timeouts against the WebSocket path but not waits in this
        # queue; this type is the only thing that tells the two apart.
        self.assertIsInstance(caught.exception.original_exception, online_service_ws_pool.PoolWaitTimeout)
        self.assertLess(waited, 2.0, "the queued caller waited for the slow request instead of its own timeout")
        self.assertEqual(gated.posts, 1, "the timed-out caller still sent")
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        pooled = pool._connections[key][0]  # noqa: SLF001 - white-box
        self.assertEqual(pooled.requests, 1, "a request that was never sent was counted against the budget")
        self.assertEqual(pooled.outstanding, 0)

    def test_the_client_gets_only_the_time_left(self) -> None:
        seen: list[float] = []
        now = [0.0]

        class _RecordsTimeout(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                seen.append(timeout)
                return super().post_json(url, body, timeout)

        pool = _pool(size=1, _client_factory=_RecordsTimeout, _clock=lambda: now[0])
        original_select = pool._select  # noqa: SLF001 - white-box

        def slow_select(key: tuple[str, ...]) -> Any:
            now[0] += 3.0  # time spent getting to the connection
            return original_select(key)

        pool._select = slow_select  # type: ignore[method-assign]  # noqa: SLF001
        pool.post_json(_URL, _body(), 10.0)

        self.assertEqual(seen, [7.0])


class QuietKeyRetirementTest(absltest.TestCase):
    """An endpoint that stops being read must not keep its connections forever."""

    def _pool(self, now: list[float]) -> online_service_ws_pool.OnlineServiceWsPool:
        return _pool(size=1, max_connection_age_seconds=100.0, splay_fraction=0.0, _clock=lambda: now[0])

    def test_a_quiet_keys_due_connections_close_when_another_key_is_read(self) -> None:
        now = [0.0]
        pool = self._pool(now)
        pool.post_json(_OTHER_URL, _body(), 1.0)
        quiet_client = _FakeClient.instances[0]

        now[0] = 150.0  # the quiet key's connection is past its age
        pool.post_json(_URL, _body(), 1.0)

        self.assertTrue(quiet_client.closed, "a quiet key's over-age connection stayed open")
        self.assertNotIn(
            online_service_ws_pool.OnlineServiceWsPool.connection_key(_OTHER_URL),
            pool._connections,  # noqa: SLF001 - white-box
            "a key with no connections left was kept",
        )
        self.assertEqual(pool.connection_count, 1)

    def test_a_quiet_keys_young_connection_is_left_alone(self) -> None:
        now = [0.0]
        pool = self._pool(now)
        pool.post_json(_OTHER_URL, _body(), 1.0)
        quiet_client = _FakeClient.instances[0]

        now[0] = 50.0
        pool.post_json(_URL, _body(), 1.0)

        self.assertFalse(quiet_client.closed)
        self.assertEqual(pool.connection_count, 2)

    def test_a_quiet_keys_busy_connection_drains_before_closing(self) -> None:
        """Retiring another key's connection keeps the two-phase rule: never close under a caller."""
        now = [0.0]
        pool = self._pool(now)
        quiet_key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_OTHER_URL)
        pool.post_json(_OTHER_URL, _body(), 1.0)
        quiet_client = _FakeClient.instances[0]
        reserved = pool._select(quiet_key)  # noqa: SLF001 - a caller mid-request on it

        now[0] = 150.0
        pool.post_json(_URL, _body(), 1.0)
        self.assertFalse(quiet_client.closed, "closed under a caller still reserved on it")

        pool._finish(quiet_key, reserved, requests=1, evict=False)  # noqa: SLF001
        self.assertTrue(quiet_client.closed, "not closed once its last caller finished")


class _RedialingClient(_FakeClient):
    """Drops its socket mid-request and re-dials on the next use, as the real client does.

    Every mid-request failure path in the client calls ``_discard_connection``, which clears the
    socket and the advertised age and leaves the client reusable. The pool cannot see that
    happen, so the dial time has to come off the client.
    """

    advertised = 600.0
    fail_on_post = 2

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_max_conn_age_seconds: float | None = None

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        if self.posts + 1 == self.fail_on_post:
            self.posts += 1
            # What _discard_connection does: socket gone, advertised age forgotten, still usable.
            self.connected_at = None
            self.server_max_conn_age_seconds = None
            raise snowml_exceptions.SnowflakeMLException(
                error_code="1510", original_exception=TimeoutError("recv timed out")
            )
        result = super().post_json(url, body, timeout)
        self.server_max_conn_age_seconds = self.advertised  # re-learned on the re-upgrade
        return result


class ReconnectRetirementTest(absltest.TestCase):
    """A connection that re-dials underneath the pool must not be retired at birth.

    The advertised age runs from the dial. Dating it from when the slot was first handed out
    means a socket opened one second ago is measured against a dial 595 seconds old and retired
    immediately -- so every transient failure costs an extra handshake, which is the
    authentication burst the splay machinery exists to prevent.
    """

    def test_the_pool_gives_the_client_its_clock(self) -> None:
        """Otherwise the dial time read back off the client is on a different clock to ``now``."""
        now = [41.0]
        pool = _pool(size=1, _clock=lambda: now[0])
        pool.post_json(_URL, _body(), 30.0)

        self.assertEqual(41.0, _FakeClient.instances[0].connected_at)

    def test_a_client_that_redialed_is_not_retired_at_birth(self) -> None:
        # Advertised 600s, so the retirement point is 595s after whichever dial it is measured
        # from. The sequence below puts the second dial late enough that the two answers differ.
        # The configured ceiling is deliberately out of the way: it is a ceiling on the *slot*,
        # dated from slot creation, so at 720s it would retire this connection for its own
        # reasons and hide what is under test.
        now = [0.0]
        pool = _pool(
            size=1,
            _client_factory=_RedialingClient,
            max_connection_age_seconds=10_000.0,
            splay_fraction=0.0,
            _clock=lambda: now[0],
        )
        pool.post_json(_URL, _body(), 30.0)
        first = cast(_RedialingClient, _FakeClient.instances[0])
        self.assertEqual(0.0, first.connected_at)

        # A recv times out 400s in. The client drops the socket and forgets the advertised age.
        now[0] = 400.0
        with self.assertRaises(snowml_exceptions.SnowflakeMLException):
            pool.post_json(_URL, _body(), 30.0)
        self.assertIsNone(first.connected_at)

        # The next caller re-upgrades the same client onto a fresh stream.
        now[0] = 401.0
        pool.post_json(_URL, _body(), 30.0)
        self.assertEqual(401.0, first.connected_at)
        self.assertLen(_FakeClient.instances, 1, "the pool replaced a client that was still fine")

        # 800s after the slot was first handed out, but only 399s into this socket's own life.
        # Dated from the slot, this is past the 595s retirement point and the connection goes.
        now[0] = 800.0
        pool.post_json(_URL, _body(), 30.0)
        self.assertFalse(first.closed, "a socket 399s old was retired against the previous dial")
        self.assertLen(_FakeClient.instances, 1)

        # Dated from its own dial it retires at 996s, and does.
        now[0] = 996.0
        pool.post_json(_URL, _body(), 30.0)
        self.assertTrue(first.closed, "the re-dialed socket was never retired")
        self.assertLen(_FakeClient.instances, 2)

    def test_a_mid_request_failure_leaves_the_connection_in_rotation(self) -> None:
        """A failure that is not a dial refusal is the connection's business, not the slot's."""
        pool = _pool(size=1, _client_factory=_RedialingClient, splay_fraction=0.0)
        pool.post_json(_URL, _body(), 30.0)

        with self.assertRaises(snowml_exceptions.SnowflakeMLException):
            pool.post_json(_URL, _body(), 30.0)

        self.assertEqual(1, pool.connection_count, "a slot was evicted by a failure that was not a dial")
        self.assertEqual(0, pool.draining_connection_count)


class BudgetChargingTest(absltest.TestCase):
    """The request budget recycles a connection that has done work, so only work counts.

    Charging a request that never left -- a dial refusal, a deadline that expired in the queue,
    a body that would not serialize -- retires failing connections *faster* than working ones,
    so every failure buys an extra handshake. Backwards from what the budget is for.
    """

    class _AlwaysFails(_FakeClient):
        def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
            raise snowml_exceptions.SnowflakeMLException(
                error_code="1510", original_exception=TimeoutError("never sent")
            )

        def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
            raise snowml_exceptions.SnowflakeMLException(
                error_code="1510", original_exception=TimeoutError("never sent")
            )

    def test_a_request_that_failed_is_not_charged(self) -> None:
        pool = _pool(size=1, _client_factory=self._AlwaysFails, splay_fraction=0.0)

        for _ in range(5):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                pool.post_json(_URL, _body(), 30.0)

        budgets = pool.request_budgets()
        self.assertLen(budgets, 1)
        self.assertEqual(0, pool.connections()[0].requests, "a request that never left was charged")

    def test_a_failed_batch_does_not_burn_the_whole_budget(self) -> None:
        """The loud version: a failed 500-body call charging 500 retires the connection outright."""
        pool = _pool(size=1, _client_factory=self._AlwaysFails, max_requests_per_connection=100, splay_fraction=0.0)

        with self.assertRaises(snowml_exceptions.SnowflakeMLException):
            pool.post_json_many(_URL, [_body() for _ in range(500)], 30.0)

        self.assertEqual(0, pool.connections()[0].requests)
        self.assertLen(_FakeClient.instances, 1, "a failed batch retired the connection it never used")

    def test_the_budget_threshold_is_exact(self) -> None:
        """``> 1 instance`` also holds for a pool that opens one connection per request."""
        pool = _pool(size=1, max_requests_per_connection=3, splay_fraction=0.0)

        for _ in range(7):
            pool.post_json(_URL, _body(), 30.0)

        # 3 requests each on two retired connections, 1 on the third.
        self.assertEqual([3, 3, 1], [client.posts for client in _FakeClient.instances])
        self.assertEqual([True, True, False], [client.closed for client in _FakeClient.instances])


class ClosePoolTest(absltest.TestCase):
    """``close`` is permanent, but it still must not be a second thread on a live socket."""

    def test_close_does_not_close_a_connection_mid_request(self) -> None:
        pool = _pool(size=1, _client_factory=_GatedClient)
        thread, done = _run_in_thread(lambda: pool.post_json(_URL, _body(), 5.0))
        client = cast(_GatedClient, _FakeClient.instances[0])
        self.assertTrue(client.entered.wait(timeout=5))

        pool.close()
        self.assertFalse(client.closed, "closed a socket its caller was still reading")

        client.release.set()
        self.assertTrue(done.wait(timeout=5))
        thread.join(timeout=5)
        self.assertTrue(client.closed, "the in-flight connection was never closed")
        self.assertEqual(1, client.close_calls)

    def test_concurrent_close_closes_each_client_exactly_once(self) -> None:
        """A double close is indistinguishable from a single one without counting them."""
        pool = _pool(size=2)
        _fill(pool, 2)
        clients = list(_FakeClient.instances)
        self.assertLen(clients, 2)

        start = threading.Barrier(4)

        def closer() -> None:
            start.wait(timeout=5)
            pool.close()

        threads = [threading.Thread(target=closer, daemon=True) for _ in range(3)]
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual([1, 1], [client.close_calls for client in clients])

    def test_select_refuses_after_close(self) -> None:
        """Pinned separately from the entry check: this is the one that stops a leaked socket.

        A caller that passed ``post_json``'s check microseconds before ``close`` would otherwise
        go on to open a connection and file it in a dict ``close`` has already drained, leaving
        a socket nobody holds a reference to and the service times out.
        """
        pool = _pool(size=1)
        pool.close()

        with self.assertRaises(RuntimeError):
            pool._select(("feature_view", "my_fv", "v1"))  # noqa: SLF001 - the in-lock check under test

    def test_post_json_many_after_close_raises(self) -> None:
        pool = _pool(size=1)
        pool.close()

        with self.assertRaises(RuntimeError):
            pool.post_json_many(_URL, [_body(), _body()], 30.0)


class DiscardAllLeavesThePoolUsableTest(absltest.TestCase):
    """``discard_all`` answers a service error that invalidates the connections, not the pool.

    If it left the pool unusable, the first transient INTERNAL error would turn every later
    online read in the process into a closed-pool error -- a one-connection blip escalated to a
    total client outage.
    """

    def test_the_pool_still_serves_after_discard_all(self) -> None:
        pool = _pool(size=2)
        _fill(pool, 2)
        discarded = list(_FakeClient.instances)

        self.assertEqual(2, pool.discard_all())
        self.assertEqual(0, pool.connection_count)

        _fill(pool, 2)
        pool.post_json(_URL, _body("fv_a"), 30.0)
        pool.post_json(_URL, _body("fv_b"), 30.0)

        self.assertEqual(2, pool.connection_count, "the pool did not re-dial after discard_all")
        fresh = [client for client in _FakeClient.instances if client not in discarded]
        self.assertLen(fresh, 2, "discard_all handed back a connection it had already closed")
        self.assertTrue(all(client.posts == 1 for client in fresh))

    def test_discard_all_counts_a_connection_already_draining(self) -> None:
        """Nothing else in the suite has a drainer present when ``discard_all`` is called."""
        now = [0.0]
        pool = _pool(
            size=1,
            _client_factory=_FirstGatedClient,
            max_connection_age_seconds=100.0,
            splay_fraction=0.0,
            _clock=lambda: now[0],
        )
        thread, done = _run_in_thread(lambda: pool.post_json(_URL, _body(), 5.0))
        gated = cast(_FirstGatedClient, _FakeClient.instances[0])
        self.assertTrue(gated.entered.wait(timeout=5))

        # Past its age, so the next selection retires it -- but it cannot close while in use.
        now[0] = 100.0
        pool.post_json(_URL, _body(), 30.0)
        self.assertEqual(1, pool.draining_connection_count)
        self.assertEqual(1, pool.connection_count)

        # The drainer plus the replacement: two connections left rotation.
        self.assertEqual(2, pool.discard_all(), "a connection already draining was not counted")

        gated.release.set()
        self.assertTrue(done.wait(timeout=5))
        thread.join(timeout=5)
        self.assertTrue(gated.closed)
        self.assertEqual(1, gated.close_calls)

    def test_discard_all_on_a_closed_pool_returns_zero(self) -> None:
        pool = _pool(size=1)
        pool.post_json(_URL, _body(), 30.0)
        pool.close()

        self.assertEqual(0, pool.discard_all())


_KEY = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)


class SaturatedSelectionTest(absltest.TestCase):
    """Selection when no connection is idle, which is the steady state under sustained load.

    Round-robin's idle-first scan finds nothing then. Both policies were only ever driven with
    nothing in flight, where they are indistinguishable -- the old least-outstanding test passed
    with its branch replaced by round-robin outright -- so the regime that decides whether load
    actually spreads had no coverage at all.
    """

    def _pool_with_depths(self, depths: list[int], **kwargs: Any) -> tuple[Any, list[Any]]:
        """Build a pool of ``len(depths)`` connections with those queue depths.

        The depths are assigned directly because :meth:`_pick_locked` is a pure function of the
        cursor and the depths; reaching each shape through concurrent callers would assert the
        same thing less precisely. ``test_least_outstanding_spreads_real_callers`` drives the
        same behaviour through live threads.

        Args:
            depths: Outstanding count per connection, in pool order.
            **kwargs: Overrides for the pool.

        Returns:
            The pool and its connections for ``_KEY``.
        """
        pool = _pool(size=len(depths), **kwargs)
        _fill(pool, len(depths))
        entries = pool._connections[_KEY]  # noqa: SLF001
        self.assertLen(entries, len(depths))
        for entry, depth in zip(entries, depths):
            entry.outstanding = depth
        return pool, entries

    def _pick(self, pool: Any, entries: list[Any], *, cursor: int) -> Any:
        """Run one selection from a known cursor position.

        Args:
            pool: The pool.
            entries: That key's connections.
            cursor: Where the round-robin cursor should start.

        Returns:
            The chosen connection.
        """
        pool._next_index[_KEY] = cursor  # noqa: SLF001
        with pool._select_lock:  # noqa: SLF001 - _pick_locked requires it
            return pool._pick_locked(_KEY, entries)  # noqa: SLF001

    def test_least_outstanding_takes_the_shallowest_queue_not_the_cursor(self) -> None:
        pool, entries = self._pool_with_depths([3, 1, 2], least_outstanding=True)

        self.assertIs(entries[1], self._pick(pool, entries, cursor=0))
        # And it is the depth doing the work, not the position: move the shallow one.
        pool, entries = self._pool_with_depths([1, 3, 2], least_outstanding=True)
        self.assertIs(entries[0], self._pick(pool, entries, cursor=1))

    def test_least_outstanding_still_rotates_when_every_queue_is_equal(self) -> None:
        """``min`` alone returns connection 0 on every tie, which pins an idle pool to one instance."""
        pool, entries = self._pool_with_depths([2, 2, 2], least_outstanding=True)

        self.assertIs(entries[1], self._pick(pool, entries, cursor=1))
        self.assertIs(entries[2], self._pick(pool, entries, cursor=2))

    def test_round_robin_takes_the_cursor_once_nothing_is_idle(self) -> None:
        """The default policy's known weakness, pinned so it stays a decision and not an accident.

        Preferring the shallowest queue here would be better under sustained load, and would
        also make this branch identical to least-outstanding in every reachable state -- an idle
        connection *is* the minimum, and both break ties in cursor order. That is a policy
        change rather than a bug fix, so it is deliberately not in this PR.
        """
        pool, entries = self._pool_with_depths([3, 1, 2], least_outstanding=False)

        self.assertIs(entries[0], self._pick(pool, entries, cursor=0))

    def test_round_robin_prefers_an_idle_connection_over_the_cursor(self) -> None:
        pool, entries = self._pool_with_depths([0, 2, 0], least_outstanding=False)

        self.assertIs(entries[2], self._pick(pool, entries, cursor=1))

    def test_least_outstanding_spreads_real_callers(self) -> None:
        """The same property through live threads, with the depths reached rather than assigned."""
        pool = _pool(size=2, least_outstanding=True, _client_factory=_GatedClient)
        for _ in range(2):
            _run_in_thread(lambda: pool.post_json(_URL, _body(), 30.0))
        self._wait_until(lambda: len(_FakeClient.instances) == 2)
        held = [cast(_GatedClient, client) for client in _FakeClient.instances[:2]]
        for client in held:
            self.assertTrue(client.entered.wait(timeout=5))

        # Both busy at depth 1. Pin the cursor at 0 and add a caller: it lands on 0.
        pool._next_index[_KEY] = 0  # noqa: SLF001
        _run_in_thread(lambda: pool.post_json(_URL, _body(), 30.0))
        self._wait_until(lambda: pool.connections()[0].outstanding == 2)

        # Pin the cursor back at 0. Least-outstanding must still choose the shallower sibling.
        pool._next_index[_KEY] = 0  # noqa: SLF001
        _run_in_thread(lambda: pool.post_json(_URL, _body(), 30.0))
        self._wait_until(lambda: pool.connections()[1].outstanding == 2)

        snapshot = pool.connections()
        self.assertEqual(
            [2, 2], [info.outstanding for info in snapshot], "the cursor was taken over a shallower sibling"
        )

        for client in held:
            client.release.set()

    def _wait_until(self, predicate: Callable[[], bool]) -> None:
        """Spin until ``predicate`` holds, so a reservation can be observed while it exists.

        Args:
            predicate: The condition to wait for.
        """
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("the expected reservation never appeared")


class LockedCountersTest(absltest.TestCase):
    """Every counter the pool publishes is read under the one lock that writes it."""

    def test_draining_count_waits_for_the_pool_lock(self) -> None:
        """``_draining`` is rebound wholesale, so an unguarded read can report a count never true."""
        pool = _pool(size=1)
        observed: list[int] = []

        with pool._select_lock:  # noqa: SLF001 - holding it is the point
            reader = threading.Thread(target=lambda: observed.append(pool.draining_connection_count), daemon=True)
            reader.start()
            reader.join(timeout=0.25)
            still_waiting = reader.is_alive()

        reader.join(timeout=5)
        self.assertTrue(still_waiting, "draining_connection_count read the counter without the lock")
        self.assertEqual([0], observed)


class DefaultsTest(absltest.TestCase):
    """The defaults are what every caller gets, and nothing else in the suite exercises them."""

    def test_the_default_pool_can_hold_more_than_one_connection(self) -> None:
        """Concurrent callers on one connection would all be pinned to the instance it landed on."""
        self.assertGreater(online_service_ws_pool.DEFAULT_POOL_SIZE, 1)

        pool = _pool()  # no size= override
        _fill(pool, online_service_ws_pool.DEFAULT_POOL_SIZE + 2)

        self.assertEqual(online_service_ws_pool.DEFAULT_POOL_SIZE, pool.connection_count)

    def test_the_default_age_does_not_pre_empt_an_advertised_one(self) -> None:
        """The configured ceiling must not bind when the service advertised a lifetime.

        The service caps a connection at WS_MAX_CONN_AGE (5 minutes by default) and jitters it
        *down* before advertising, so an advertised age is never longer than that cap. The
        advertised path already retires a margin under whatever it was given, so a configured
        ceiling below the cap would cut connections short that the service was happy to keep --
        paying an extra handshake, and an extra authentication, for nothing.
        """
        self.assertGreater(online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS, 5 * 60.0)

    def test_the_default_splay_is_on(self) -> None:
        """Splay off by default would make connections opened together retire together."""
        self.assertGreater(online_service_ws_pool.DEFAULT_SPLAY_FRACTION, 0.0)

    def test_a_size_below_one_still_holds_a_connection(self) -> None:
        """``_pick_locked`` divides by the pool length, so an empty pool is not merely useless."""
        pool = _pool(size=0)
        pool.post_json(_URL, _body(), 30.0)

        self.assertEqual(1, pool.connection_count)

    def test_splay_fraction_is_clamped(self) -> None:
        """Unclamped, a fraction at or above 1.0 can splay the age ceiling to zero.

        Every connection would then be retired on the selection after its own creation, forever
        -- an unbounded handshake loop against the service from a single config typo.
        """
        pool = _pool(size=8, splay_fraction=5.0, max_connection_age_seconds=100.0, _rng=random.Random(7))
        _fill(pool, 8)

        ages = [pooled.max_age_seconds for pooled in pool._connections[pool.connection_key(_URL)]]  # noqa: SLF001
        self.assertLen(ages, 8)
        for age in ages:
            self.assertGreaterEqual(age, 10.0, f"the age ceiling was splayed to almost nothing: {ages}")
        for budget in pool.request_budgets():
            self.assertGreaterEqual(budget, 1)


class TransportConformanceTest(absltest.TestCase):
    """The pool's interchangeability with the HTTP client is a prose claim until it is checked."""

    def test_the_pool_satisfies_the_batching_transport_protocol(self) -> None:
        pool = _pool(size=1)

        self.assertIsInstance(pool, online_service_transport.BatchingOnlineServiceTransport)
        self.assertTrue(online_service_transport.supports_batching(pool))


class ForwardingTest(absltest.TestCase):
    """What the pool hands the client. Nothing downstream would catch a dropped argument."""

    def test_the_session_and_url_reach_the_client(self) -> None:
        sentinel = object()
        pool = _pool(size=1, session=cast(Any, sentinel))
        pool.post_json(_URL, _body(), 30.0)

        client = _FakeClient.instances[0]
        self.assertIs(sentinel, client.session, "the pool's session never reached the client")
        self.assertEqual([_URL], client.urls)

    def test_a_batch_arrives_together_and_in_order(self) -> None:
        """One pool call can only make one client call, so counting batches proves nothing."""
        pool = _pool(size=2)
        bodies = [_body(version=f"v{i}") for i in range(3)]

        results = pool.post_json_many(_URL, bodies, 30.0)

        self.assertLen(results, 3)
        sending = [client for client in _FakeClient.instances if client.sent]
        self.assertLen(sending, 1, "a batch was split across connections")
        self.assertEqual(bodies, sending[0].sent, "the batch arrived reordered or incomplete")


class ObservabilityTest(absltest.TestCase):
    """The counters telemetry reads have to count something."""

    class _OnInstance(_FakeClient):
        """Reports a serving instance, as a real response does, round-robin across three."""

        hostnames = ["ws-0.svc", "ws-1.svc", "ws-0.svc", "ws-2.svc"]

        def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
            if self.server_hostname is None:
                index = _FakeClient.instances.index(self) % len(self.hostnames)
                self.server_hostname = self.hostnames[index]
            return super().post_json(url, body, timeout)

    def test_server_hostnames_counts_connections_per_instance(self) -> None:
        """The only client-side view of spread; the aggregation had never run with a hostname."""
        pool = _pool(size=4, _client_factory=self._OnInstance)
        _fill(pool, 4)
        for _ in range(4):
            pool.post_json(_URL, _body(), 30.0)

        self.assertEqual({"ws-0.svc": 2, "ws-1.svc": 1, "ws-2.svc": 1}, pool.server_hostnames())

    def test_live_connection_count_counts_open_sockets(self) -> None:
        """Asserted only as zero before, so ``return 0`` passed."""
        pool = _pool(size=3)
        _fill(pool, 2)
        pool.post_json(_URL, _body(), 30.0)
        pool.post_json(_URL, _body(), 30.0)  # round-robin: one on each, so both have dialed

        self.assertEqual(2, pool.connection_count)
        self.assertEqual(2, pool.live_connection_count)

        _FakeClient.instances[0].close()
        self.assertEqual(1, pool.live_connection_count)

    def test_the_snapshot_is_ordered_by_key_then_position(self) -> None:
        """Repeated calls have to be comparable, which a set of keys cannot show."""
        pool = _pool(size=2)
        for url in (_URL, _OTHER_URL):  # opened out of key order
            _fill(pool, 2, url)

        snapshot = [(info.key, info.position) for info in pool.connections()]
        self.assertEqual(sorted(snapshot), snapshot)
        self.assertEqual(
            [((_OTHER_URL,), 0), ((_OTHER_URL,), 1), ((_URL,), 0), ((_URL,), 1)],
            snapshot,
        )
        self.assertEqual(pool.connections(), pool.connections())


class WholePoolRetirementTest(absltest.TestCase):
    """Every age and budget test used ``size=1``, so the retirement loop only ever ran once."""

    def test_a_whole_pool_can_retire_in_one_selection(self) -> None:
        now = [0.0]
        pool = _pool(size=4, max_connection_age_seconds=100.0, splay_fraction=0.0, _clock=lambda: now[0])
        _fill(pool, 4)
        original = list(_FakeClient.instances)
        self.assertLen(original, 4)

        now[0] = 100.0
        pool.post_json(_URL, _body(), 30.0)

        self.assertTrue(all(client.closed for client in original), "not every due connection was retired")
        self.assertEqual([1, 1, 1, 1], [client.close_calls for client in original])
        # Every due slot is replaced in the same selection, so the pool stays at the size it was
        # driven to: the replacements are built before the old entries are marked.
        self.assertEqual(4, pool.connection_count, "the pool did not stay at its size")
        self.assertEqual(0, pool.draining_connection_count)

    def test_a_replacement_that_cannot_be_built_leaves_the_slot_alone(self) -> None:
        """Marking the old one first would strand an entry that is retiring and still in the pool.

        Both retirement sweeps skip a ``retiring`` entry and the picker does not look at the flag
        at all, so such a slot is never reconsidered and keeps being handed out until its last
        caller closes it under everyone who draws it afterwards.
        """
        now = [0.0]
        pool = _pool(size=1, max_connection_age_seconds=100.0, splay_fraction=0.0, _clock=lambda: now[0])
        pool.post_json(_URL, _body(), 30.0)
        original = _FakeClient.instances[0]

        now[0] = 100.0
        with patch.object(pool, "_new_connection", side_effect=RuntimeError("no slots")):
            with self.assertRaises(RuntimeError):
                pool.post_json(_URL, _body(), 30.0)

        entries = pool._connections[pool.connection_key(_URL)]  # noqa: SLF001
        self.assertLen(entries, 1)
        self.assertFalse(entries[0].retiring, "the slot was left marked retiring and still in the pool")
        self.assertFalse(original.closed)

        # And once a replacement can be built, the retirement completes as normal.
        pool.post_json(_URL, _body(), 30.0)
        self.assertTrue(original.closed)
        self.assertEqual(1, pool.connection_count)


class SaturationTest(absltest.TestCase):
    """A pool with nothing free must not spend the caller's whole deadline queueing.

    That deadline is the HTTP fallback's budget as much as the WebSocket attempt's, so a caller
    that queues until it expires reaches the fallback with no time to be served: a saturated
    pool failed reads outright while the fallback it has sat unused. And because the resulting
    timeout looked like the wire rather than the queue, three of them rested every read on HTTP
    for a minute, so an ordinary burst against a healthy service switched the feature off.
    """

    def test_a_caller_gives_up_queueing_with_budget_left_for_the_fallback(self) -> None:
        """The wait is capped below the deadline, so the fallback inherits usable time."""
        pool = _pool(size=1, _client_factory=_GatedClient)
        holder, holder_done = _run_in_thread(lambda: pool.post_json(_URL, _body(), 30.0))
        deadline = time.monotonic() + 5
        while not _FakeClient.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        gated = cast(_GatedClient, _FakeClient.instances[0])
        self.assertTrue(gated.entered.wait(timeout=5))

        timeout = 1.0
        started = time.monotonic()
        try:
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as caught:
                pool.post_json(_URL, _body(), timeout)
            waited = time.monotonic() - started
        finally:
            gated.release.set()
            holder.join(timeout=5)
        self.assertTrue(holder_done.is_set())

        self.assertIsInstance(caught.exception.original_exception, online_service_ws_pool.PoolWaitTimeout)
        cap = timeout * online_service_ws_pool.POOL_QUEUE_SHARE_OF_TIMEOUT
        self.assertLess(
            waited,
            timeout * 0.9,
            f"queued {waited:.3f}s of a {timeout}s timeout, leaving the HTTP fallback nothing to run on",
        )
        self.assertGreater(waited, cap * 0.5, "gave up so early the pool was barely given a chance")

    def test_a_wire_timeout_after_queueing_reads_as_contention(self) -> None:
        """The common saturated shape: the connection arrives, just too late to be used.

        The transport counts wire timeouts against the WebSocket path, so without this a caller
        starved by its own pool becomes evidence against the server.
        """
        now = [0.0]

        class _TimesOutOnTheWire(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                raise TimeoutError(f"read timed out after {timeout:.3f}s")

        pool = _pool(size=1, _client_factory=_TimesOutOnTheWire, _clock=lambda: now[0])
        original_select = pool._select  # noqa: SLF001 - white-box, repo style

        def slow_select(key: tuple[str, ...]) -> Any:
            now[0] += 0.6  # 60% of the budget gone before the connection is in hand
            return original_select(key)

        pool._select = slow_select  # type: ignore[method-assign]

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as caught:
            pool.post_json(_URL, _body(), 1.0)

        self.assertIsInstance(caught.exception.original_exception, online_service_ws_pool.PoolWaitTimeout)

    def test_a_wire_timeout_without_queueing_still_counts_against_the_path(self) -> None:
        """The guard against over-reaching: a sick socket must still be able to rest the path.

        Excusing every timeout would disable the fallback ladder, which is the mechanism that
        moves reads off a WebSocket path that has genuinely stopped working.
        """

        class _TimesOutOnTheWire(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                raise TimeoutError(f"read timed out after {timeout:.3f}s")

        pool = _pool(size=1, _client_factory=_TimesOutOnTheWire)

        # Passed through as the client raised it, rather than rewrapped: the transport reads a
        # bare TimeoutError as well as a wrapped one, and only the type decides whether it counts.
        with self.assertRaises(TimeoutError) as caught:
            pool.post_json(_URL, _body(), 1.0)

        self.assertNotIsInstance(
            caught.exception,
            online_service_ws_pool.PoolWaitTimeout,
            "an uncontended wire timeout was excused as contention, so the ladder can never trip",
        )

    def test_a_failure_that_is_not_a_timeout_is_never_excused(self) -> None:
        """Queueing explains a timeout. It explains nothing about a connection that errored."""
        now = [0.0]

        class _Breaks(_FakeClient):
            def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
                raise RuntimeError("socket is broken")

        pool = _pool(size=1, _client_factory=_Breaks, _clock=lambda: now[0])
        original_select = pool._select  # noqa: SLF001 - white-box, repo style

        def slow_select(key: tuple[str, ...]) -> Any:
            now[0] += 0.9  # almost the whole budget spent queueing
            return original_select(key)

        pool._select = slow_select  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as broke:
            pool.post_json(_URL, _body(), 1.0)
        self.assertIn("socket is broken", str(broke.exception))


if __name__ == "__main__":
    absltest.main()
