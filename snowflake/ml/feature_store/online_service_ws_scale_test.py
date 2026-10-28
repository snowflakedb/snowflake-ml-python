"""Scale tests for the Online Service WebSocket connection pool.

Connection slots are a shared, per-instance resource: the service caps them at 1024 per
instance, three instances is roughly 3,072 in total, and nothing on the server enforces a
per-caller share. A client that hoards or leaks slots therefore takes capacity away from
every other client. The measured service ceiling of ~18,136 TPS is reached by only ~72 fully
busy connections, because one busy sequential connection sustains ~250 QPS — so a large pool
buys a caller very little and costs everyone else a lot.

The default pool size is 4 and nothing had been exercised above 16. Everything here asks what
happens at 256 and 1024 slots, and while connections churn.

Every test here is deterministic and the file runs in a few seconds, because it is in the
validation pipeline. The measurements that take wall-clock time or read the machine -- the
soak, the churn storm, throughput against pool size, selection cost and per-slot memory --
are in ``tests/perf/feature_store/ws_pool_load.py``, which the pipeline does not run. They
import their fakes from here, so moving one between the two files does not mean copying it.

``KnownDefectTest`` holds deterministic guards for three invariants this suite first saw broken
under load (all since fixed). Forcing the interleaving is what makes them usable as guards --
raced, they fired far too rarely to assert on (the use-after-close
window 5 times in 3.66M requests, the introspection race 62 times in two seconds):

* a connection is chosen, then recycled and closed by another caller before the chooser marks
  it busy, so the request goes out on a closed socket;
* retirement is gated on ``outstanding == 0``, so a connection that is never idle is never
  retired and the age limit stops being enforced under exactly the load it matters for;
* the unguarded introspection properties iterate the key dictionary while another caller's
  first read of a new feature view writes to it.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable, cast

from absl.testing import absltest

from snowflake.ml.feature_store import online_service_ws_client, online_service_ws_pool

logger = logging.getLogger(__name__)

_URL = "wss://svc.example.app/api/v1/query/ws"

# How long a join waits before calling it a deadlock. Generous against the longest run here,
# which is a couple of seconds, so a slow machine cannot turn into a spurious failure.
_JOIN_TIMEOUT_SECONDS = 120.0


class _StressClient:
    """Client-shaped fake that records what the pool did to it, without opening a socket."""

    def __init__(self, recorder: _ClientRecorder, *, refuse: bool) -> None:
        self._recorder = recorder
        self._refuse = refuse
        self.is_connected = False
        self.closed = False
        self.posts = 0
        self.in_call = 0
        # Feature views this connection served, as (name, version). Not the pool key: the pool
        # keys by endpoint, so one connection legitimately serves every view a caller reads.
        self.views: set[tuple[str, str]] = set()

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Answer one request, recording the view served and any use after close.

        Refuses with a dial error, as the service does at its connection cap, when this client
        was built to refuse.

        Args:
            url: Ignored; the pool passes the WebSocket URL through.
            body: Request payload, whose key is recorded to prove per-key isolation.
            timeout: Ignored.

        Returns:
            ``(200, b"{}")``.
        """
        del url, timeout
        return self._serve([body])[0]

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Answer a batch on this connection, refusing the dial when built to refuse.

        Args:
            url: Ignored.
            bodies: Request payloads.
            timeout: Ignored.

        Returns:
            One ``(200, b"{}")`` per body.
        """
        del url, timeout
        return self._serve(bodies)

    def close(self) -> None:
        """Close the connection, recording whether a request was running when it happened."""
        if self.in_call > 0:
            self._recorder.record_closed_while_busy()
        if not self.closed:
            self.closed = True
            self._recorder.record_closed(self.posts)
        self.is_connected = False

    def _serve(self, bodies: list[dict[str, Any]]) -> list[tuple[int, bytes]]:
        """Shared body for both transport methods.

        Args:
            bodies: Request payloads.

        Returns:
            One ``(200, b"{}")`` per body.

        Raises:
            OnlineServiceWsDialError: When this client was built to refuse its handshake.
        """
        if self.closed:
            self._recorder.record_use_after_close()
        if self._refuse:
            raise online_service_ws_client.OnlineServiceWsDialError(
                "Online Service WebSocket connection failed: at capacity", status=503
            )
        self.in_call += 1
        try:
            self.is_connected = True
            for body in bodies:
                self.views.add((str(body.get("name", "")), str(body.get("version", ""))))
            if self._recorder.sleep_seconds > 0:
                time.sleep(self._recorder.sleep_seconds)
            self.posts += len(bodies)
            self._recorder.record_posts(len(bodies))
            return [(200, b"{}") for _ in bodies]
        finally:
            self.in_call -= 1


class _ClientRecorder:
    """Callable client factory that keeps the totals the pool's own counters are checked against.

    Retention is optional because the churn and soak tests build hundreds of thousands of
    clients, and holding them all would measure the test's memory rather than the pool's.
    """

    def __init__(
        self,
        *,
        sleep_seconds: float = 0.0,
        refuse_probability: float = 0.0,
        retain: bool = True,
        seed: int = 0,
    ) -> None:
        self.sleep_seconds = sleep_seconds
        self.refuse_probability = refuse_probability
        self.retain = retain
        self.clients: list[_StressClient] = []
        self.created = 0
        self.closed = 0
        self.posts = 0
        self.use_after_close = 0
        self.closed_while_busy = 0
        self.max_posts_per_client = 0
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def __call__(self, **kwargs: Any) -> _StressClient:
        """Build one client, as the pool's ``_client_factory`` seam.

        Args:
            **kwargs: The pool passes ``session``; it is not needed by the fake.

        Returns:
            A fresh fake client.
        """
        del kwargs
        with self._lock:
            self.created += 1
            refuse = self._rng.random() < self.refuse_probability
        client = _StressClient(self, refuse=refuse)
        if self.retain:
            with self._lock:
                self.clients.append(client)
        return client

    def record_posts(self, count: int) -> None:
        """Add to the independent request total.

        Args:
            count: Requests answered.
        """
        with self._lock:
            self.posts += count

    def record_closed(self, posts: int) -> None:
        """Count one client closed by the pool, keeping the largest request count seen.

        Args:
            posts: Requests this client answered before being closed.
        """
        with self._lock:
            self.closed += 1
            self.max_posts_per_client = max(self.max_posts_per_client, posts)

    def record_use_after_close(self) -> None:
        """Count one request sent on a client the pool had already closed."""
        with self._lock:
            self.use_after_close += 1

    def record_closed_while_busy(self) -> None:
        """Count one close that landed while a request was executing on the connection."""
        with self._lock:
            self.closed_while_busy += 1


def _pool(recorder: Callable[..., Any], **kwargs: Any) -> online_service_ws_pool.OnlineServiceWsPool:
    """Build a pool wired to a recorder, with a frozen clock unless one is passed.

    Args:
        recorder: The client factory to use.
        **kwargs: Overrides passed to the pool.

    Returns:
        The pool.
    """
    defaults: dict[str, Any] = {
        "_client_factory": recorder,
        "_rng": random.Random(20260922),
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


def _pooled_entries(pool: online_service_ws_pool.OnlineServiceWsPool) -> list[Any]:
    """Every pooled entry across every key.

    Args:
        pool: The pool to inspect.

    Returns:
        The pooled connection objects, read white-box because the counters under test are
        deliberately not part of the public surface.
    """
    return [entry for entries in pool._connections.values() for entry in entries]  # noqa: SLF001


def _run_threads(target: Any, count: int) -> list[int]:
    """Run ``target`` on ``count`` threads and collect each one's own request total.

    Per-thread totals are returned rather than accumulated into a shared counter, so the
    test's own bookkeeping cannot be the thing that loses an increment.

    Args:
        target: Callable taking no arguments and returning its request count.
        count: Threads to run.

    Returns:
        One total per thread.

    Raises:
        AssertionError: When a worker raised, did not finish, or outlived the run.
    """
    totals: list[int] = [0] * count
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    baseline = threading.active_count()

    def worker(slot: int) -> None:
        try:
            totals[slot] = target()
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread below
            with errors_lock:
                errors.append(error)

    threads = [threading.Thread(target=worker, args=(index,), name=f"stress-{index}") for index in range(count)]
    for thread in threads:
        thread.start()
    # Bounded, so a deadlock in the pool fails this suite instead of hanging it. An unbounded
    # join turns the hazard several of these tests exist to catch into a stalled CI job with
    # nothing to read, and a daemon thread would be torn down at exit with no report at all.
    deadline = time.monotonic() + _JOIN_TIMEOUT_SECONDS
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise AssertionError(
            f"{len(alive)} of {count} worker threads did not finish within "
            f"{_JOIN_TIMEOUT_SECONDS:.0f}s, so the pool deadlocked or starved them: {alive[:5]}"
        )
    if threading.active_count() != baseline:
        raise AssertionError(f"worker threads outlived the run: {threading.active_count()} vs {baseline}")
    # A thread's exception is otherwise only printed, and its total silently stays 0.
    if errors:
        raise AssertionError(f"{len(errors)} of {count} worker threads raised; first: {errors[0]!r}") from errors[0]
    return totals


def _fill_pool(pool: online_service_ws_pool.OnlineServiceWsPool, size: int, *, url: str = _URL) -> None:
    """Open ``size`` connections by holding that many reservations at once.

    A connection is opened only when every held one is busy, so however many requests a
    sequential caller sends it holds exactly one. Reaching ``size`` therefore takes ``size``
    callers in flight together, which is what reserving without finishing models -- and it
    models it deterministically, where ``size`` real threads would have to be caught
    overlapping. No request is charged, so budgets and ages are untouched.

    Args:
        pool: The pool to fill.
        size: Connections to open.
        url: Endpoint to key on, which is the whole of the pool key.
    """
    key = online_service_ws_pool.OnlineServiceWsPool.connection_key(url)
    reserved = [pool._select(key) for _ in range(size)]  # noqa: SLF001 - reserving is the point
    for pooled in reserved:
        pool._finish(key, pooled, requests=0, evict=False)  # noqa: SLF001 - white-box, repo style


class ScaleTest(absltest.TestCase):
    """Pool sizes far above anything previously exercised, up to one instance's whole cap.

    A pool of 1024 is not a recommendation — it is a third of the entire service's slots in
    one caller's hands. These tests establish that the bookkeeping is correct and cheap at
    that size, so the argument against it stays about shared capacity rather than about the
    client falling over.
    """

    def test_pool_holds_exactly_its_configured_slots_at_256(self) -> None:
        """A pool that overshoots its size takes slots nobody accounted for."""
        self._assert_exact_slots(256)

    def test_pool_holds_exactly_its_configured_slots_at_1024(self) -> None:
        """1024 is one instance's entire connection cap, so overshoot here is a service outage."""
        self._assert_exact_slots(1024)

    def _assert_exact_slots(self, size: int) -> None:
        """Ask for more connections than the pool holds, all at once, and confirm it stops on size.

        Every reservation is held without finishing, so each selection past ``size`` finds the
        whole pool busy -- the moment an uncapped pool would open another. Filling to ``size``
        and then sending requests could never test the cap: requests only ever reuse the held
        connections, so the cap could be deleted and the counts would still match.

        Args:
            size: Pool size to exercise.
        """
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=size, max_requests_per_connection=10**9)
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        reserved = [pool._select(key) for _ in range(size + 64)]  # noqa: SLF001 - all held at once

        self.assertEqual(pool.connection_count, size)
        self.assertEqual(recorder.created, size, "the pool opened a connection past its size")
        self.assertLen({id(pooled) for pooled in reserved}, size)
        for pooled in reserved:
            pool._finish(key, pooled, requests=0, evict=False)  # noqa: SLF001 - white-box, repo style

    def test_a_sequential_caller_holds_one_connection_however_many_requests(self) -> None:
        """The cap is what a pool may hold, not what it opens: a caller with nothing in flight
        beside its own request needs one socket, and each extra one is an authentication and a
        slot out of the service's shared cap."""
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=1024, max_requests_per_connection=10**9)
        for _ in range(5000):
            pool.post_json(_URL, _body(), 1.0)

        self.assertEqual(pool.connection_count, 1)
        self.assertEqual(recorder.created, 1)

    def test_round_robin_still_spreads_evenly_at_1024_slots(self) -> None:
        """Spread is the pool's whole purpose; a rotation that skews at scale pins instances."""
        size = 1024
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=size, max_requests_per_connection=10**9)
        _fill_pool(pool, size)
        for _ in range(size * 3):
            pool.post_json(_URL, _body(), 1.0)

        posts = [client.posts for client in recorder.clients]
        self.assertLen(posts, size)
        self.assertEqual(min(posts), 3)
        self.assertEqual(max(posts), 3)

    def test_close_releases_every_slot_at_1024(self) -> None:
        """A leaked slot is held until the service times it out, denying it to other callers."""
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=1024, max_requests_per_connection=10**9)
        _fill_pool(pool, 1024)
        for _ in range(1024):
            pool.post_json(_URL, _body(), 1.0)

        pool.close()

        self.assertEqual(pool.connection_count, 0)
        self.assertEqual(pool.live_connection_count, 0)
        self.assertEqual(recorder.closed, 1024)
        self.assertTrue(all(client.closed for client in recorder.clients))


class MixedKeysTest(absltest.TestCase):
    """Many feature views at once, over connections that are bound to none of them.

    The handshake carries only the caller's credentials and every message names its own object,
    so any connection to an endpoint can serve any read of it. Slot cost therefore does not
    scale with the number of views a caller touches -- which it did while the pool keyed by
    view, and that is the whole reason these connections are shared.
    """

    def test_many_views_do_not_multiply_the_slots_held(self) -> None:
        """Keying by view multiplied sockets by views read: 60 views at size 4 held 240 of the
        service's ~3,072, for one caller, where this sequential reader needs one."""
        keys, size = 60, 4
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=size, max_requests_per_connection=10**9)
        for round_index in range(size + 2):
            for key_index in range(keys):
                pool.post_json(_URL, _body(f"fv_{key_index}"), 1.0)
            del round_index

        self.assertLessEqual(pool.connection_count, size)
        self.assertEqual(pool.connection_count, 1, "a caller with nothing in flight needs one")
        self.assertEqual(recorder.created, 1)
        logger.info("%d views at size %d held %d slots", keys, size, pool.connection_count)

    def test_views_share_connections_instead_of_multiplying_them(self) -> None:
        """Connections are not scoped to a view, so views share them; what must hold is that
        the pool still stops at its size however many views are read through it."""
        keys, size = 64, 4
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=size, max_requests_per_connection=10**9)
        _fill_pool(pool, size)

        def worker() -> int:
            sent = 0
            for _ in range(40):
                for key_index in range(keys):
                    pool.post_json(_URL, _body(f"fv_{key_index}"), 1.0)
                    sent += 1
            return sent

        _run_threads(worker, 12)

        self.assertEqual(pool.connection_count, size, "views must not add slots")
        # Sharing is the point, so at least one connection must have served several views.
        # Correlation -- that a read gets its own view's rows back -- is covered by
        # CorrelationTest in the adversarial suite, which echoes a marker per request.
        self.assertGreater(max(len(client.views) for client in recorder.clients), 1)

    def test_growth_of_many_keys_at_once_never_oversizes_any_key(self) -> None:
        """Growth is a read-then-write, so concurrent first use could double-append past size."""
        keys, size = 50, 3
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=size, max_requests_per_connection=10**9)

        def worker() -> int:
            for key_index in range(keys):
                pool.post_json(_URL, _body(f"fv_{key_index}"), 1.0)
            return keys

        _run_threads(worker, 16)

        # One endpoint, so one key however many views were read through it.
        self.assertLen(pool._connections, 1)  # noqa: SLF001 - white-box, matching repo style
        for entries in pool._connections.values():  # noqa: SLF001 - white-box, matching repo style
            self.assertLessEqual(len(entries), size, "concurrent growth overshot the pool size")


class DialRefusalStormTest(absltest.TestCase):
    """Refused handshakes, which is what the per-instance cap looks like from the client.

    A slot whose dial failed is worthless to its holder and unavailable to everybody else, so
    the pool must drop it rather than keep it parked.
    """

    def test_slots_do_not_accumulate_when_every_dial_is_refused(self) -> None:
        """Parking dead slots would deny capacity to callers whose dials would have succeeded."""
        size = 8
        recorder = _ClientRecorder(refuse_probability=1.0, retain=False)
        pool = _pool(recorder, size=size, splay_fraction=0.0)
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        # Both tallies stay thread-local and are reduced on the main thread. `refused += 1` and
        # `max(peak, ...)` on a name shared across threads are read-modify-write and lose
        # updates under exactly the concurrency this test runs -- and a lost refusal makes the
        # assertions below easier to pass, not harder, which is the worst way to be wrong.
        peaks: list[int] = []
        peaks_lock = threading.Lock()

        def worker() -> int:
            refused = 0
            peak = 0
            for _ in range(400):
                try:
                    pool.post_json(_URL, _body(), 1.0)
                except online_service_ws_client.OnlineServiceWsDialError:
                    refused += 1
                peak = max(peak, len(pool._connections.get(key, [])))  # noqa: SLF001 - white-box
            with peaks_lock:
                peaks.append(peak)
            return refused

        refused = sum(_run_threads(worker, 12))
        peak = max(peaks)

        logger.info("all-refused storm: %d refusals, peak slots %d, created %d", refused, peak, recorder.created)
        self.assertLessEqual(peak, size)
        self.assertEqual(pool.connection_count, 0)
        self.assertEqual(pool.live_connection_count, 0)

    def test_partial_refusals_leave_no_dead_slot_behind(self) -> None:
        """A half-dead pool would quietly serve at half the spread it reports holding."""
        size = 8
        recorder = _ClientRecorder(refuse_probability=0.5, retain=False, seed=11)
        pool = _pool(recorder, size=size, splay_fraction=0.0)
        # Refusals counted thread-locally and reduced here: a shared `refused += 1` loses
        # updates across these 12 threads, and an undercount makes the bound below easier to
        # satisfy rather than failing the test.
        refusals: list[int] = []
        refusals_lock = threading.Lock()

        def worker() -> int:
            refused = 0
            sent = 0
            for _ in range(2000):
                try:
                    pool.post_json(_URL, _body(), 1.0)
                    sent += 1
                except online_service_ws_client.OnlineServiceWsDialError:
                    refused += 1
            with refusals_lock:
                refusals.append(refused)
            return sent

        served = sum(_run_threads(worker, 12))
        refused = sum(refusals)

        logger.info(
            "half-refused storm: %d served, %d refused, %d dials, %d held",
            served,
            refused,
            recorder.created,
            pool.connection_count,
        )
        # Every slot the pool still holds is one that dialed: a refused dial hands its slot
        # straight back rather than parking it. That is this test's subject, and unlike a count
        # it does not depend on how much the callers happened to overlap -- these fakes answer
        # instantly, so the threads rarely have two requests in flight and one connection
        # serves them all.
        self.assertBetween(pool.connection_count, 1, size)
        self.assertEqual(
            pool.live_connection_count,
            pool.connection_count,
            "a slot whose dial was refused is still parked in the pool",
        )
        # Refusals must be bounded by redials, not proportional to requests: a slot that dials
        # successfully keeps serving, so a steady state of good connections is reached quickly.
        self.assertLess(refused, served // 10)

    def test_pool_refills_once_dials_succeed_again(self) -> None:
        """Capacity comes back; a pool that stayed empty after a 503 storm would never recover."""
        size = 8
        recorder = _ClientRecorder(refuse_probability=1.0, retain=False)
        pool = _pool(recorder, size=size, splay_fraction=0.0)
        for _ in range(50):
            with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError):
                pool.post_json(_URL, _body(), 1.0)
        self.assertEqual(pool.connection_count, 0)

        recorder.refuse_probability = 0.0
        # One request proves a dial is possible again; filling proves the storm did not cost the
        # pool its headroom, which is the recovery that matters -- a pool that could only ever
        # reach one connection again would serve, and silently never spread.
        pool.post_json(_URL, _body(), 1.0)
        self.assertEqual(pool.connection_count, 1)
        _fill_pool(pool, size)
        for _ in range(size * 3):
            pool.post_json(_URL, _body(), 1.0)

        self.assertEqual(pool.connection_count, size)
        self.assertEqual(pool.live_connection_count, size)


class AccountingDriftTest(absltest.TestCase):
    """Counters after enough requests to expose a lost increment or a drifting threshold.

    ``requests`` and ``outstanding`` must only change under the pool lock; a lost update would
    show here as drift. The counter is what decides recycling, so drift means a connection that
    outlives its budget or retires early.
    """

    def test_request_counters_do_not_drift_over_120k_requests(self) -> None:
        """A lost increment makes a connection outlive the budget that bounds instance pinning."""
        threads, per_thread = 8, 15_000
        recorder = _ClientRecorder(retain=False)
        pool = _pool(recorder, size=8, max_requests_per_connection=10**9, splay_fraction=0.0)

        def worker() -> int:
            for _ in range(per_thread):
                pool.post_json(_URL, _body(), 1.0)
            return per_thread

        issued = sum(_run_threads(worker, threads))
        pool_side = sum(entry.requests for entry in _pooled_entries(pool))

        logger.info("accounting: issued %d, pool counted %d, fakes counted %d", issued, pool_side, recorder.posts)
        self.assertEqual(issued, threads * per_thread)
        self.assertEqual(recorder.posts, issued)
        self.assertEqual(pool_side, issued)

    def test_batch_counters_charge_every_body_in_the_batch(self) -> None:
        """Charging a batch as one request would let a pipelining caller never recycle."""
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=1, max_requests_per_connection=10**9, splay_fraction=0.0)
        for _ in range(1000):
            pool.post_json_many(_URL, [_body() for _ in range(10)], 1.0)

        self.assertEqual(recorder.posts, 10_000)
        self.assertEqual(sum(entry.requests for entry in _pooled_entries(pool)), 10_000)

    def test_splay_thresholds_stay_in_bounds_across_thousands_of_recycles(self) -> None:
        """A threshold drifting out of band would resynchronize the redials splay exists to spread."""
        nominal, fraction = 20, 0.25
        recorder = _ClientRecorder(retain=False)
        pool = _pool(
            recorder,
            size=16,
            max_requests_per_connection=nominal,
            splay_fraction=fraction,
            _rng=random.Random(4242),
        )
        _fill_pool(pool, 16)
        for _ in range(40_000):
            pool.post_json(_URL, _body(), 1.0)

        budgets = pool.request_budgets()
        logger.info("after %d recycles: %d budgets in %s", recorder.created, len(budgets), sorted(set(budgets)))
        self.assertLen(budgets, 16)
        for budget in budgets:
            self.assertGreaterEqual(budget, int(nominal * (1 - fraction)))
            self.assertLessEqual(budget, int(nominal * (1 + fraction)))
        self.assertGreater(len(set(budgets)), 1)
        # Every connection retired on its own threshold, so the number recycled is the number
        # of requests divided by the nominal budget, give or take the splay.
        self.assertGreater(recorder.created, 40_000 / (nominal * (1 + fraction)))

    def test_budget_is_honored_exactly_when_requests_do_not_overlap(self) -> None:
        """Overrun is how a connection stays on one instance past the point spread was promised."""
        recorder = _ClientRecorder(retain=False)
        pool = _pool(recorder, size=4, max_requests_per_connection=50, splay_fraction=0.0)
        for _ in range(20_000):
            pool.post_json(_URL, _body(), 1.0)

        # Selection checks the budget before handing a connection out, and with nothing
        # overlapping every connection is idle at that moment, so the threshold is exact.
        for entry in _pooled_entries(pool):
            self.assertLessEqual(entry.requests, entry.request_budget)
        # One connection built per 50 requests, plus the four the pool still holds.
        self.assertBetween(recorder.created, 20_000 // 50 - 4, 20_000 // 50 + 4)


class _KeyAddingClient:
    """Client whose connection check adds a pool key, standing in for a concurrent first read.

    ``live_connection_count`` reads ``client.is_connected`` while iterating the key dictionary,
    so this makes the dictionary grow at exactly the point another thread's first request for a
    new feature view would grow it.
    """

    def __init__(self, pool_holder: list[Any]) -> None:
        self._pool_holder = pool_holder
        self._added = False
        self.closed = False

    @property
    def is_connected(self) -> bool:
        """Report the connection open, adding a key on the first check.

        Returns:
            Always True.
        """
        if not self._added:
            self._added = True
            pool = self._pool_holder[0]
            pool._connections[("feature_view", "arrived_late", "v1")] = []  # noqa: SLF001 - white-box
        return True

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Answer one request.

        Args:
            url: Ignored.
            body: Ignored.
            timeout: Ignored.

        Returns:
            ``(200, b"{}")``.
        """
        del url, body, timeout
        return 200, b"{}"

    def close(self) -> None:
        """Close the connection."""
        self.closed = True


class KnownDefectTest(absltest.TestCase):
    """Deterministic guards for three invariants first seen broken under load, all since fixed.

    The interleavings are forced rather than raced, because each was first found at rates too low
    to assert on: the use-after-close window fired 5 times in 3.66M requests, and the
    introspection race 62 times in two seconds.
    """

    def test_a_selected_connection_is_not_recycled_before_its_request_is_sent(self) -> None:
        """A caller holding a just-chosen connection must not have it closed underneath them."""
        recorder = _ClientRecorder()
        pool = _pool(recorder, size=1, max_requests_per_connection=2, splay_fraction=0.0)
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        pool.post_json(_URL, _body(), 1.0)

        # One request short of its budget, so selection still considers it usable.
        chosen = pool._select(key)  # noqa: SLF001 - holding a reservation is the point
        self.assertEqual(chosen.requests, 1)

        # Another caller completes a request on the same connection, taking it over budget, and
        # a third selection then finds it due for retirement. What keeps it open is that
        # `_select` reserved it before releasing the lock: every close path requires
        # `outstanding == 0`, so a connection a caller is holding drains instead of closing.
        # Reserve after returning instead -- as this did when the defect was live -- and the
        # connection still looks idle here, gets taken as drained, and closes under the caller.
        pool.post_json(_URL, _body(), 1.0)
        pool._select(key)  # noqa: SLF001 - retires the connection the first caller is holding

        self.assertFalse(cast(Any, chosen.client).closed, "the connection a caller is about to use was closed")

    def test_an_over_age_connection_is_not_handed_out_again_while_busy(self) -> None:
        """Retirement must not be starved by load, or the service closes the connection instead."""
        now = [0.0]
        recorder = _ClientRecorder()
        pool = _pool(
            recorder,
            size=2,
            max_connection_age_seconds=100.0,
            max_requests_per_connection=10**9,
            _clock=lambda: now[0],
        )
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(_URL)
        # Both slots opened explicitly: a sequential caller holds one however many requests it
        # sends, so driving this with two requests would leave the second slot unopened.
        _fill_pool(pool, 2)
        for _ in range(2):
            pool.post_json(_URL, _body(), 1.0)
        aged = pool._connections[key][0]  # noqa: SLF001 - white-box, matching repo style

        # A request is in flight on it, and the age limit passes while that request runs. Not
        # closing it now is correct; continuing to hand it out is not, because it can only be
        # closed when idle and under load it is never idle.
        aged.outstanding = 1
        now[0] = 500.0

        handed_out = [pool._select(key) for _ in range(20)]  # noqa: SLF001 - selection under test
        self.assertNotIn(aged, handed_out, "an over-age connection kept being handed out")

    def test_live_connection_count_survives_a_key_being_added(self) -> None:
        """Telemetry sampling the pool must not raise when a new feature view is first read."""
        holder: list[Any] = []
        pool = _pool(lambda **kwargs: _KeyAddingClient(holder), size=1)
        holder.append(pool)
        pool.post_json(_URL, _body(), 1.0)

        self.assertEqual(pool.live_connection_count, 1)


if __name__ == "__main__":
    absltest.main()
