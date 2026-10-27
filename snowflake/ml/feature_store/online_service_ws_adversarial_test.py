"""Adversarial tests for the Online Service WebSocket transport.

These attack the hazards the pooling and pipelining designs created rather than re-covering
happy paths. Every test here aims at a way the code could deadlock, leak a connection,
double-count, hand a caller another caller's feature values, or keep using a socket somebody
already closed.

Modelled on the server-side suite in ``ws_adversarial_test.go``. The client-side hazards are
different in shape — there is no reader goroutine to leak — but identical in kind: a
connection is a shared, per-instance resource, and the one silent failure this transport has
is a response being matched to the wrong request.

Several tests loop internally because the hazard is probabilistic, and all of them use the
injected ``_clock`` / ``_rng`` / ``_sleep`` seams so no test waits on wall-clock time.
"""

import json
import os
import random
import threading
import time
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store import (
    online_service_ws_client,
    online_service_ws_pipeline,
    online_service_ws_pool,
    online_service_ws_transport,
)

_URL = "wss://svc.example.app/api/v1/query/ws"
_HTTP_URL = "https://svc.example.app/api/v1/query"
_PAT = "adversarial-pat-token"

# Frames a peer can send that are not usable responses: malformed JSON, an empty frame, a
# bare JSON null, and a binary frame.
_GARBAGE_FRAMES: tuple[str | bytes, ...] = ("{not json", "", "null", b"\x00\x01")
_GARBAGE_PAYLOADS: frozenset[bytes] = frozenset(
    frame.encode("utf-8") if isinstance(frame, str) else bytes(frame) for frame in _GARBAGE_FRAMES
)


def _body(name: str = "my_fv", version: str = "v1", **extra: Any) -> dict[str, Any]:
    """Build a minimal request body the pool can key on.

    Args:
        name: Feature view name.
        version: Feature view version.
        **extra: Additional fields, e.g. a correlation marker.

    Returns:
        A request body.
    """
    body: dict[str, Any] = {"object_type": "feature_view", "name": name, "version": version, "request_rows": []}
    body.update(extra)
    return body


class _ScriptedPeer:
    """A socket-shaped peer that echoes correlation ids, and misbehaves on request.

    Echoing ``request_id`` and ``marker`` back is what makes cross-talk detectable: a caller
    that receives a marker it did not send has been handed another caller's response.
    """

    def __init__(
        self,
        *,
        out_of_order: bool = False,
        fail_at: int | None = None,
        garbage_every: int | None = None,
        hostname: str | None = None,
    ) -> None:
        """Script one peer.

        Args:
            out_of_order: Answer the newest outstanding request first, as a pipelining peer may.
            fail_at: Raise on this receive and every one after it, as a vanished connection does.
            garbage_every: Send an unusable frame on every n-th receive.
            hostname: Serving instance to report, so spread is observable.
        """
        self.out_of_order = out_of_order
        self.fail_at = fail_at
        self.garbage_every = garbage_every
        self.hostname = hostname
        self.sent: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.receives = 0
        self.close_calls = 0
        self.lock = threading.Lock()

    def send(self, frame: str) -> None:
        """Record one outgoing frame.

        Args:
            frame: The serialized request.
        """
        parsed = json.loads(frame)
        with self.lock:
            self.sent.append(parsed)
            self.pending.append(parsed)

    def recv(self) -> str | bytes:
        """Answer one outstanding request, or misbehave as scripted.

        Returns:
            One response frame.

        Raises:
            OSError: When scripted to vanish, or asked for a response it was never sent.
        """
        with self.lock:
            self.receives += 1
            received = self.receives
            if self.fail_at is not None and received >= self.fail_at:
                raise OSError("connection reset by peer")
            if not self.pending:
                raise OSError("no outstanding request to answer")
            request = self.pending.pop(-1 if self.out_of_order else 0)
        if self.garbage_every and received % self.garbage_every == 0:
            return _GARBAGE_FRAMES[(received // self.garbage_every - 1) % len(_GARBAGE_FRAMES)]
        return json.dumps(
            {
                "status": "SUCCESS",
                "request_id": request.get("request_id"),
                "marker": request.get("marker"),
                "server_hostname": self.hostname,
            }
        )

    def settimeout(self, timeout: float) -> None:
        """Accept a per-request timeout.

        Args:
            timeout: Seconds.
        """
        del timeout

    def close(self) -> None:
        """Count closes, so a double close is visible."""
        with self.lock:
            self.close_calls += 1


def _make_real_client_factory(peers: list[_ScriptedPeer], **peer_kwargs: Any) -> Callable[..., Any]:
    """Build a pool client factory that produces real clients over scripted peers.

    Using the real :class:`OnlineServiceWsClient` is the point: correlation checking, connection
    discarding and closed-client refusal are all code under test, not stubbed out.

    Args:
        peers: Collects every peer opened, in dial order.
        **peer_kwargs: Passed to each :class:`_ScriptedPeer`.

    Returns:
        A factory the pool can call as ``factory(session=...)``.
    """
    guard = threading.Lock()

    def ws_factory(
        url: str,
        header: list[str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> _ScriptedPeer:
        # ``**kwargs`` rather than a fixed signature: anything the client passes and this fake
        # rejects raises TypeError *inside* the dial, which the client classifies as a handshake
        # failure -- so no peer is ever created and every suite in this file decays to a vacuous
        # pass. That happened once, over a handshake argument that was later removed again, and
        # only the decay guards in these tests caught it.
        del url, header, timeout, kwargs
        peer = _ScriptedPeer(**peer_kwargs)
        with guard:
            peers.append(peer)
        return peer

    def client_factory(**kwargs: Any) -> online_service_ws_client.OnlineServiceWsClient:
        del kwargs
        return online_service_ws_client.OnlineServiceWsClient(_ws_factory=ws_factory)

    return client_factory


class _CountingClient:
    """A client-shaped fake that reports how it was used, including misuse after close.

    It mirrors the real client where that matters: a request issued on a closed client raises
    ``RuntimeError``, and ``inside`` is non-zero exactly while a request is in flight, so a
    close landing on a connection somebody is using is observable.
    """

    def __init__(
        self,
        *,
        gate: threading.Event | None = None,
        dial_status: int | None = None,
        on_close: Callable[[], None] | None = None,
        on_is_connected: Callable[[], None] | None = None,
    ) -> None:
        """Build one fake connection.

        Args:
            gate: When set, every request blocks on it, holding the connection in flight.
            dial_status: When set, every request raises a handshake refusal carrying this status.
            on_close: Called once, from inside ``close``, to model a caller racing teardown.
            on_is_connected: Called once, from inside ``is_connected``, to model the same race
                against the pool's introspection.
        """
        self.gate = gate
        self.dial_status = dial_status
        self.on_close = on_close
        self.on_is_connected = on_is_connected
        self.posts = 0
        self.inside = 0
        self.closed = False
        self.close_calls = 0
        self.closed_while_in_use = 0
        self.used_after_close = 0

    @property
    def is_connected(self) -> bool:
        """True until closed; fires the injected hook once on first read."""
        hook, self.on_is_connected = self.on_is_connected, None
        if hook is not None:
            hook()
        return not self.closed

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Serve one request, recording in-flight state.

        Args:
            url: Ignored.
            body: Ignored apart from its marker.
            timeout: Ignored.

        Returns:
            ``(200, body_bytes)`` echoing the request's marker.

        Raises:
            RuntimeError: When the client was already closed, as the real client does.
            OnlineServiceWsDialError: When scripted to refuse the handshake.
        """
        del url, timeout
        if self.closed:
            self.used_after_close += 1
            raise RuntimeError("OnlineServiceWsClient is closed")
        if self.dial_status is not None:
            raise online_service_ws_client.OnlineServiceWsDialError("handshake refused", status=self.dial_status)
        self.inside += 1
        try:
            if self.gate is not None:
                self.gate.wait(10.0)
            self.posts += 1
        finally:
            self.inside -= 1
        if self.closed:
            self.used_after_close += 1
            raise RuntimeError("OnlineServiceWsClient is closed")
        return 200, json.dumps({"status": "SUCCESS", "marker": body.get("marker")}).encode("utf-8")

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Serve a batch as a sequence of single requests.

        Args:
            url: Ignored.
            bodies: Request payloads.
            timeout: Ignored.

        Returns:
            One result per body, in order.
        """
        return [self.post_json(url, body, timeout) for body in bodies]

    def close(self) -> None:
        """Close idempotently, recording a close that landed on a connection in use."""
        self.close_calls += 1
        if self.inside > 0:
            self.closed_while_in_use += 1
        self.closed = True
        hook, self.on_close = self.on_close, None
        if hook is not None:
            hook()


def _make_counting_factory(created: list[_CountingClient], **client_kwargs: Any) -> Callable[..., _CountingClient]:
    """Build a pool client factory producing :class:`_CountingClient` instances.

    Args:
        created: Collects every client the pool opened.
        **client_kwargs: Passed to each client.

    Returns:
        A factory the pool can call as ``factory(session=...)``.
    """
    guard = threading.Lock()

    def factory(**kwargs: Any) -> _CountingClient:
        del kwargs
        client = _CountingClient(**client_kwargs)
        with guard:
            created.append(client)
        return client

    return factory


def _make_pool(**kwargs: Any) -> online_service_ws_pool.OnlineServiceWsPool:
    """Build a pool with deterministic splay and a frozen clock unless overridden.

    Args:
        **kwargs: Overrides passed to the pool.

    Returns:
        The pool.
    """
    defaults: dict[str, Any] = {"splay_fraction": 0.0, "_rng": random.Random(20260922), "_clock": lambda: 0.0}
    defaults.update(kwargs)
    return online_service_ws_pool.OnlineServiceWsPool(**defaults)


def _run_threads(target: Callable[[int], None], count: int) -> None:
    """Run ``target`` on ``count`` threads, join them all, and fail on any worker's exception.

    Args:
        target: Receives the thread's index.
        count: Threads to start.

    Raises:
        AssertionError: When a worker raised or did not finish.
    """
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def run(index: int) -> None:
        try:
            target(index)
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread below
            with errors_lock:
                errors.append(error)

    threads = [threading.Thread(target=run, args=(index,), daemon=True) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
    # A thread still running here means the pool deadlocked or starved it, which is what several
    # of these tests exist to catch. Without this it keeps running while the assertions below
    # read half-finished state, and the hazard reports itself as a confusing count mismatch.
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise AssertionError(f"{len(alive)} of {count} worker threads did not finish in 30s: {alive[:5]}")
    # A thread's exception is otherwise only printed: an assertion inside a worker could never
    # fail the test that started it.
    if errors:
        raise AssertionError(f"{len(errors)} of {count} worker threads raised; first: {errors[0]!r}") from errors[0]


def _fill_pool(pool: Any, size: int, *, url: str = _URL) -> None:
    """Open ``size`` connections by holding that many reservations at once.

    A connection is opened only when every held one is busy, so a caller with one request in
    flight holds one connection however many requests it sends. Reaching ``size`` therefore
    takes ``size`` callers in flight together, which reserving without finishing models without
    needing that many threads to be caught overlapping. No request is charged.

    Args:
        pool: The pool to fill.
        size: Connections to open.
        url: Endpoint to key on, which is the whole of the pool key.
    """
    key = online_service_ws_pool.OnlineServiceWsPool.connection_key(url)
    reserved = [pool._select(key) for _ in range(size)]  # noqa: SLF001 - reserving is the point
    for pooled in reserved:
        pool._finish(key, pooled, requests=0, evict=False)  # noqa: SLF001 - white-box, repo style


def _threads_settled(baseline: int, within_seconds: float = 5.0) -> int:
    """Wait for the live thread count to come back to ``baseline``.

    Distinguishes a leak from ordinary join lag, the way the Go suite's goroutine gate does.

    Args:
        baseline: Thread count before the work started.
        within_seconds: How long to allow.

    Returns:
        The thread count reached.
    """
    deadline = time.monotonic() + within_seconds
    count = threading.active_count()
    while count > baseline and time.monotonic() < deadline:
        time.sleep(0.01)
        count = threading.active_count()
    return count


def _marker_of(payload: bytes) -> Any:
    """Read the correlation marker out of a response payload.

    Args:
        payload: Raw response bytes.

    Returns:
        The marker, or None when the payload carries none.
    """
    try:
        parsed = json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    return parsed.get("marker") if isinstance(parsed, dict) else None


class _FakeHttp:
    """Stands in for the HTTP transport the WebSocket path falls back to."""

    def __init__(self) -> None:
        self.posts = 0
        self.closed = False
        self.lock = threading.Lock()

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Answer over HTTP, echoing the marker so a crossed result is detectable.

        Args:
            url: Ignored.
            body: Request payload.
            timeout: Ignored.

        Returns:
            ``(200, body_bytes)``.
        """
        del url, timeout
        with self.lock:
            self.posts += 1
        return 200, json.dumps({"status": "SUCCESS", "via": "http", "marker": body.get("marker")}).encode("utf-8")

    def close(self) -> None:
        """Mark the fallback transport closed."""
        self.closed = True


class _ThrottlingPool:
    """A pool that asks for a backoff the first time it sees each marker, then succeeds.

    Modelled on the service's own behaviour: a 429 carrying ``retry_after_seconds`` is guidance,
    not a failure, so the transport must wait and resend rather than fall back.
    """

    def __init__(self, *, retry_after_seconds: float = 1.0) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.connection_count = 2
        self.closed = False
        self.throttled: set[Any] = set()
        self.calls = 0
        self.lock = threading.Lock()

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Throttle a marker once, then answer it.

        Args:
            url: Ignored.
            body: Request payload carrying a marker.
            timeout: Ignored.

        Returns:
            ``(429, body_bytes)`` on first sight of the marker, else ``(200, body_bytes)``.
        """
        del url, timeout
        marker = body.get("marker")
        with self.lock:
            self.calls += 1
            first_sight = marker not in self.throttled
            self.throttled.add(marker)
        if first_sight:
            payload = {"error": {"code": "RESOURCE_EXHAUSTED"}, "retry_after_seconds": self.retry_after_seconds}
            payload["marker"] = marker
            return 429, json.dumps(payload).encode("utf-8")
        return 200, json.dumps({"status": "SUCCESS", "marker": marker}).encode("utf-8")

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Serve a batch as a sequence of single requests.

        Args:
            url: Ignored.
            bodies: Request payloads.
            timeout: Ignored.

        Returns:
            One result per body, in order.
        """
        return [self.post_json(url, body, timeout) for body in bodies]

    def close(self) -> None:
        """Mark the pool closed."""
        self.closed = True


def _make_transport(
    pool: Any, **kwargs: Any
) -> tuple[online_service_ws_transport.OnlineServiceWsTransport, list[float]]:
    """Build a transport with a frozen clock, seeded jitter and recorded waits.

    Args:
        pool: The pool the transport should drive.
        **kwargs: Overrides passed to the transport.

    Returns:
        ``(transport, slept)`` where ``slept`` records every wait in order.
    """
    slept: list[float] = []
    record_lock = threading.Lock()

    def sleep(seconds: float) -> None:
        with record_lock:
            slept.append(seconds)

    defaults: dict[str, Any] = {
        "pool": pool,
        "_rng": random.Random(7),
        "_clock": lambda: 0.0,
        "_sleep": sleep,
    }
    defaults.update(kwargs)
    transport = online_service_ws_transport.OnlineServiceWsTransport(_FakeHttp(), **defaults)
    return transport, slept


class PoolLifecycleTest(absltest.TestCase):
    """Teardown hazards. A connection that outlives its pool holds a per-instance slot on the
    service until the service times it out, so a leak here reduces capacity for every caller."""

    def test_many_lifecycles_leak_no_threads_and_no_connections(self) -> None:
        """Repeated create/use/close cycles must leave no live thread and no open connection,
        because a long-lived process does this thousands of times per day."""
        baseline = threading.active_count()
        total_created = 0

        for cycle in range(30):
            created: list[_CountingClient] = []
            pool = _make_pool(_client_factory=_make_counting_factory(created), size=3)
            # Every slot open, so each cycle leaves close() a full pool to tear down. These
            # fakes answer instantly, so four threads rarely have two requests in flight and
            # lazy growth would otherwise give this a single connection to close.
            _fill_pool(pool, 3)

            def worker(index: int, pool: Any = pool) -> None:
                for _ in range(4):
                    pool.post_json(_URL, _body(marker=f"c{index}"), 1.0)

            _run_threads(worker, 4)
            pool.close()
            total_created += len(created)

            self.assertTrue(
                all(client.closed for client in created),
                f"cycle {cycle}: {sum(1 for c in created if not c.closed)} of {len(created)} connections "
                "were left open by close()",
            )
            self.assertEqual(pool.connection_count, 0)

        self.assertGreaterEqual(total_created, 30 * 3)
        settled = _threads_settled(baseline)
        self.assertLessEqual(
            settled,
            baseline,
            f"thread leak: started at {baseline}, settled at {settled} after 30 pool lifecycles",
        )

    def test_close_while_requests_are_in_flight_does_not_deadlock(self) -> None:
        """close() must not block behind in-flight requests, must close every connection, and must
        leave the pool refusing further work — a shutdown that waits on a socket hangs the process."""
        for _ in range(5):
            gate = threading.Event()
            created: list[_CountingClient] = []
            pool = _make_pool(_client_factory=_make_counting_factory(created, gate=gate), size=4)
            outcomes: list[str] = []
            outcome_lock = threading.Lock()

            def worker(
                index: int,
                pool: Any = pool,
                outcomes: list[str] = outcomes,
                outcome_lock: threading.Lock = outcome_lock,
            ) -> None:
                try:
                    # The timeout covers the wait for a connection, and 8 of the 12 workers queue
                    # behind the 4 held ones until close() releases them. A short one would let a
                    # slow main thread time them out, which tests the scheduler, not close().
                    pool.post_json(_URL, _body(marker=f"m{index}"), 30.0)
                    result = "served"
                except RuntimeError as error:
                    result = "closed" if "is closed" in str(error) else f"undocumented: {error}"
                with outcome_lock:
                    outcomes.append(result)

            threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(12)]
            for thread in threads:
                thread.start()
            # Wait until requests are genuinely in flight, not merely started.
            deadline = time.monotonic() + 5.0
            while sum(client.inside for client in created) < 4 and time.monotonic() < deadline:
                time.sleep(0.005)

            started = time.monotonic()
            pool.close()
            elapsed = time.monotonic() - started
            gate.set()
            for thread in threads:
                thread.join(10.0)

            self.assertLess(elapsed, 2.0, f"close() blocked {elapsed:.2f}s behind in-flight requests")
            self.assertTrue(all(client.closed for client in created))
            self.assertLen(outcomes, 12)
            self.assertEmpty([outcome for outcome in outcomes if outcome.startswith("undocumented")])
            with self.assertRaises(RuntimeError):
                pool.post_json(_URL, _body(), 1.0)

    def test_concurrent_close_calls_never_close_a_socket_twice(self) -> None:
        """Many threads closing at once must be idempotent down to the socket, since a second close
        on a real WebSocket raises and cleanup is documented as never raising."""
        with mock.patch.dict(os.environ, {"SNOWFLAKE_PAT": _PAT}, clear=False):
            for _ in range(5):
                peers: list[_ScriptedPeer] = []
                pool = _make_pool(_client_factory=_make_real_client_factory(peers), size=4)
                # All four slots opened, so the concurrent close below has four sockets to race
                # over rather than the one a sequential caller would hold. The requests then
                # dial them: a slot is claimed on selection, the socket opened on first use.
                _fill_pool(pool, 4)
                for index in range(8):
                    pool.post_json(_URL, _body(marker=f"m{index}"), 1.0)
                self.assertLen(peers, 4)

                errors: list[str] = []

                def closer(index: int, pool: Any = pool, errors: list[str] = errors) -> None:
                    del index
                    try:
                        pool.close()
                    except Exception as error:  # noqa: BLE001 - close is documented as never raising
                        errors.append(repr(error))

                _run_threads(closer, 24)

                self.assertEmpty(errors, f"close() raised under concurrent calls: {errors[:3]}")
                for peer in peers:
                    self.assertEqual(peer.close_calls, 1, "a socket was closed more than once")
                self.assertEqual(pool.connection_count, 0)


class CloseRacesSelectionTest(absltest.TestCase):
    """``close()`` must stay safe while other callers are selecting: no raise, no socket left open.

    ``close()`` takes its snapshot and sets ``_closed`` under ``_select_lock``, and closes clients
    after releasing it. Each test forces one interleaving that would break a close() that did
    less; each failure mode is a live socket or a lost connection rather than a wrong answer,
    and none of them announces itself.
    """

    def test_close_racing_post_json_raises_nothing_undocumented(self) -> None:
        """Hammering close() against live callers may only ever surface the documented closed-pool
        RuntimeError, because the query path maps unknown exceptions to a failed feature read."""
        for _ in range(5):
            created: list[_CountingClient] = []
            pool = _make_pool(_client_factory=_make_counting_factory(created), size=4)
            # Warm every key first: growing the dict during close() is a separate defect, pinned below.
            pool.post_json(_URL, _body(), 1.0)
            surprises: list[str] = []
            stop = threading.Event()

            def worker(
                index: int,
                pool: Any = pool,
                surprises: list[str] = surprises,
                stop: threading.Event = stop,
            ) -> None:
                while not stop.is_set():
                    try:
                        pool.post_json(_URL, _body(marker=f"m{index}"), 1.0)
                    except RuntimeError as error:
                        if "is closed" not in str(error):
                            surprises.append(repr(error))
                    except Exception as error:  # noqa: BLE001 - anything else is the finding
                        surprises.append(repr(error))

            threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(10)]
            for thread in threads:
                thread.start()
            for _ in range(50):
                pool.close()
            stop.set()
            for thread in threads:
                thread.join(10.0)

            self.assertEmpty(surprises, f"undocumented exception escaped a closed pool: {surprises[:3]}")
            with self.assertRaises(RuntimeError):
                pool.post_json(_URL, _body(), 1.0)

    def test_close_closes_every_connection_even_when_a_caller_races_it(self) -> None:
        """A caller that passed the closed check before close() ran must not repopulate the pool.

        If it could, that connection would never be closed: a live socket holding a service slot
        forever. close() sets ``_closed`` under the selection lock, so the late caller is refused.
        """
        reached = threading.Event()
        release = threading.Event()
        armed = threading.Event()

        def gated_clock() -> float:
            """Pause one caller between the closed check and selection.

            ``post_json`` reads the clock for its deadline after refusing a closed pool and
            before ``_select``, which is exactly the window. It used to be held open from
            inside ``connection_key``, but that keys on the endpoint now and never touches the
            body. Armed after warm-up so only the racer pauses, and once, so the pool's other
            clock reads run normally.

            Returns:
                A frozen clock reading.
            """
            if armed.is_set() and not reached.is_set():
                reached.set()
                release.wait(5.0)
            return 0.0

        created: list[_CountingClient] = []
        pool = _make_pool(_client_factory=_make_counting_factory(created), size=1, _clock=gated_clock)
        pool.post_json(_URL, _body(), 1.0)
        armed.set()

        racer_error: list[BaseException] = []

        def race() -> None:
            try:
                pool.post_json(_URL, _body(), 1.0)
            except BaseException as e:  # noqa: BLE001 - recorded and asserted on below
                racer_error.append(e)

        racer = threading.Thread(target=race, daemon=True)
        racer.start()
        self.assertTrue(reached.wait(5.0))
        pool.close()
        release.set()
        racer.join(10.0)

        # Losing the race to close() is the correct outcome for the racer; anything else is a bug.
        self.assertLen(racer_error, 1)
        self.assertIsInstance(racer_error[0], RuntimeError)
        self.assertIn("closed", str(racer_error[0]))
        left_open = [client for client in created if not client.closed]
        self.assertEmpty(left_open, f"{len(left_open)} of {len(created)} connections survived close()")
        self.assertEqual(pool.connection_count, 0)

    def test_close_does_not_raise_while_another_caller_selects_a_new_key(self) -> None:
        """close() closes from a snapshot, so the key dictionary growing mid-close cannot break it.

        A caller's first request to another endpoint adds a key. The hook adds one directly, while
        close() is closing clients: going through ``_select`` would be refused once ``_closed`` is
        set, which is why an earlier version of this test could not fail. A close() that iterated
        ``_connections`` itself would raise 'dictionary changed size during iteration' here and
        abandon every connection after the first.
        """
        holder: list[Any] = []

        def add_a_key() -> None:
            holder[0]._connections[("feature_view", "other_fv", "v1")] = []  # noqa: SLF001 - white-box

        created: list[_CountingClient] = []
        pool = _make_pool(_client_factory=_make_counting_factory(created, on_close=add_a_key), size=2)
        holder.append(pool)
        _fill_pool(pool, 2)

        pool.close()  # documented as never raising
        self.assertLen(created, 2)
        self.assertTrue(all(client.closed for client in created), "close() abandoned connections after the first")

    def test_live_connection_count_does_not_raise_while_a_key_is_added(self) -> None:
        """The introspection counts must not raise while selection adds a key.

        ``live_connection_count`` (and ``connection_count``, read on every request's failure path)
        sum over ``_connections``; they snapshot it under the selection lock, so a first read of a
        new feature view landing mid-count cannot break them.
        """
        holder: list[Any] = []

        def select_another_key() -> None:
            holder[0]._select(("feature_view", "other_fv", "v1"))  # noqa: SLF001 - white-box, repo style

        created: list[_CountingClient] = []
        pool = _make_pool(
            _client_factory=_make_counting_factory(created, on_is_connected=select_another_key),
            size=1,
        )
        holder.append(pool)
        pool.post_json(_URL, _body(), 1.0)

        self.assertGreaterEqual(pool.live_connection_count, 1)


class RecyclingRaceTest(absltest.TestCase):
    """Retirement runs inside selection, so it decides to replace a connection while other threads
    are using the same pool. Replacing one mid-request would abort work already sent."""

    def test_recycling_never_closes_a_connection_that_is_in_use(self) -> None:
        """Retirement must skip any connection with a request outstanding, and accounting must come
        back to zero, or a connection looks permanently busy and is never recycled at all."""
        closed_while_in_use = 0
        used_after_close = 0
        requests = 0

        for trial in range(6):
            created: list[_CountingClient] = []
            now = [0.0]
            pool = _make_pool(
                _client_factory=_make_counting_factory(created),
                size=4,
                max_requests_per_connection=2,
                max_connection_age_seconds=0.5,
                _clock=lambda bound=now: bound[0],
                _rng=random.Random(trial),
            )
            surprises: list[str] = []

            def worker(
                index: int,
                pool: Any = pool,
                surprises: list[str] = surprises,
                now: list[float] = now,
            ) -> None:
                for step in range(30):
                    now[0] += 0.05  # age the pool from every thread, so retirement fires under load
                    try:
                        pool.post_json(_URL, _body(marker=f"t{index}-{step}"), 1.0)
                    except RuntimeError as error:
                        if "is closed" not in str(error):
                            surprises.append(repr(error))
                    except Exception as error:  # noqa: BLE001 - anything else is the finding
                        surprises.append(repr(error))

            _run_threads(worker, 12)

            self.assertEmpty(surprises, f"undocumented exception during recycling: {surprises[:3]}")
            for entries in pool._connections.values():  # noqa: SLF001 - white-box, repo style
                for pooled in entries:
                    self.assertEqual(pooled.outstanding, 0, "outstanding count drifted; the slot never recycles")
            closed_while_in_use += sum(client.closed_while_in_use for client in created)
            used_after_close += sum(client.used_after_close for client in created)
            requests += 12 * 30
            pool.close()

        self.assertEqual(
            closed_while_in_use,
            0,
            "recycling closed a connection while a request was in flight on it",
        )
        # Exactly zero, not a small rate. This tolerated a fraction of a percent while the
        # select-then-use window was open -- a caller could be handed a connection retirement
        # closed before it sent. Selection now reserves the connection before releasing the
        # pool lock and every close path requires ``outstanding == 0``, so a held connection
        # drains instead of closing and the window is shut. A tolerance here would let a
        # regression that reopens it pass: the window fired 5 times in 3.66M requests when it
        # was live, which is far under any threshold this test could set.
        self.assertEqual(
            used_after_close,
            0,
            f"{used_after_close} of {requests} requests were issued on a connection recycling had "
            "already closed; the select-then-use window is open again",
        )

    def test_dial_refusal_evicts_the_slot_without_corrupting_accounting(self) -> None:
        """A refused handshake must free the slot it never opened and leave the pool usable, because
        a slot is a shared per-instance resource and dead ones shrink everyone's capacity."""
        created: list[_CountingClient] = []
        pool = _make_pool(_client_factory=_make_counting_factory(created, dial_status=503), size=3)
        # One refusal appended per caller rather than a shared counter incremented: `refused += 1`
        # across these eight threads is read-modify-write and loses updates, and an undercount
        # makes the assertion below easier to satisfy rather than failing it.
        refusals: list[int] = []
        misclassified: list[str] = []
        record_lock = threading.Lock()

        def worker(index: int) -> None:
            for _ in range(10):
                try:
                    pool.post_json(_URL, _body(marker=f"m{index}"), 1.0)
                except online_service_ws_client.OnlineServiceWsDialError as error:
                    with record_lock:
                        if not error.at_capacity:
                            misclassified.append(f"dial error not at_capacity: {error}")
                        refusals.append(1)
                except Exception as error:  # noqa: BLE001 - collected; asserted on the main thread
                    # A refused dial must not close the client under callers already reserved on
                    # it: each of them has to see the refusal itself, not "client is closed".
                    with record_lock:
                        misclassified.append(f"{type(error).__name__}: {error}")

        _run_threads(worker, 8)

        self.assertEqual([], misclassified, "a caller saw something other than its own dial refusal")
        # Every request was refused, and every refusal reached its own caller.
        self.assertLen(refusals, 8 * 10)
        self.assertEqual(pool.connection_count, 0, "a slot that never opened is still occupied")
        self.assertTrue(all(client.closed for client in created))
        for entries in pool._connections.values():  # noqa: SLF001 - white-box, repo style
            for pooled in entries:
                self.assertEqual(pooled.outstanding, 0)


class CapacityTest(absltest.TestCase):
    """The service caps connections per instance. Refusals must free slots and the pool must
    recover once dials succeed again, or a transient cap becomes a permanent outage."""

    def test_capacity_exhaustion_then_release_lets_the_pool_recover(self) -> None:
        """Dials refused with 503 must free their slots and the pool must refill and serve once
        capacity returns, because a leaked dead slot wedges the caller at reduced capacity forever."""
        refusing = [True]
        created: list[_CountingClient] = []
        guard = threading.Lock()

        def factory(**kwargs: Any) -> _CountingClient:
            del kwargs
            client = _CountingClient(dial_status=503 if refusing[0] else None)
            with guard:
                created.append(client)
            return client

        pool = _make_pool(_client_factory=factory, size=3)

        # Every dial is refused: each attempt must give its slot straight back.
        refused = 0
        for _ in range(9):
            try:
                pool.post_json(_URL, _body(marker="refused"), 1.0)
            except online_service_ws_client.OnlineServiceWsDialError as error:
                self.assertTrue(error.at_capacity)
                refused += 1
        self.assertEqual(refused, 9)
        self.assertEqual(pool.connection_count, 0, "refused dials are still holding slots")
        self.assertTrue(all(client.closed for client in created))

        # Capacity returns: the pool must dial again, refill to its size, and answer its callers.
        # Filling is explicit because these calls are sequential and a caller with one request
        # in flight needs one connection -- the recovery to check is that the storm did not cost
        # the pool its headroom, not that a lone caller re-opens three sockets it cannot use.
        refusing[0] = False
        results = [pool.post_json(_URL, _body(marker=f"after{index}"), 1.0) for index in range(6)]
        self.assertEqual(pool.connection_count, 1, "the pool did not dial again after capacity freed up")
        _fill_pool(pool, 3)
        self.assertEqual(pool.connection_count, 3, "the pool could not refill after capacity freed up")
        for index, (status, payload) in enumerate(results):
            self.assertEqual(status, 200)
            self.assertEqual(_marker_of(payload), f"after{index}")

        # A later cap on new dials must not disturb connections that are already open.
        refusing[0] = True
        status, payload = pool.post_json(_URL, _body(marker="established"), 1.0)
        self.assertEqual(status, 200)
        self.assertEqual(_marker_of(payload), "established")
        self.assertEqual(pool.connection_count, 3)
        pool.close()


class ConnectionLossTest(absltest.TestCase):
    """A connection can die at any point in a pipelined batch. The danger is a partial batch being
    returned as if complete, or the dead connection being reused for the next batch."""

    def test_connection_vanishing_mid_batch_surfaces_and_discards_the_connection(self) -> None:
        """A peer that stops answering mid-batch must produce an error and a discarded connection,
        never a short result list, because unread responses would land in the next caller's batch."""
        with mock.patch.dict(os.environ, {"SNOWFLAKE_PAT": _PAT}, clear=False):
            for fail_at in range(1, 7):
                peers: list[_ScriptedPeer] = []
                factory = _make_real_client_factory(peers, fail_at=fail_at)
                client = factory()
                bodies = [_body(marker=f"m{index}") for index in range(6)]

                with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                    client.post_json_many(_URL, bodies, 1.0)

                self.assertLen(peers, 1)
                self.assertFalse(client.is_connected, "a connection that failed mid-batch was kept")
                self.assertEqual(peers[0].close_calls, 1)

                # A fresh connection must still serve the same batch, so the failure was the
                # socket's and not the batch's.
                healthy_peers: list[_ScriptedPeer] = []
                healthy = _make_real_client_factory(healthy_peers)()
                results = healthy.post_json_many(_URL, bodies, 1.0)
                self.assertLen(results, 6)
                self.assertEqual([_marker_of(payload) for _, payload in results], [f"m{index}" for index in range(6)])
                healthy.close()

    def test_a_request_answered_twice_is_refused_rather_than_filling_another_slot(self) -> None:
        """A peer that answers one request twice must raise, because the duplicate would otherwise
        fill the slot of a request that never got an answer and the caller could not tell."""
        for repeats in range(5):
            sent: list[str] = []

            def send(frame: str, sent: list[str] = sent) -> None:
                sent.append(json.loads(frame)["request_id"])

            def recv(sent: list[str] = sent) -> str | bytes:
                # Always answer the first request, however many are outstanding.
                return json.dumps({"status": "SUCCESS", "request_id": sent[0]})

            bodies = [_body(marker=f"m{index}") for index in range(2 + repeats)]
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                online_service_ws_pipeline.pipelined_exchange(send, recv, bodies)


class FrameHazardTest(absltest.TestCase):
    """Frames a peer sends that are not usable responses, under concurrency. Each caller must get
    exactly one outcome, and never another caller's payload."""

    def test_mixed_valid_and_garbage_frames_never_cross_callers(self) -> None:
        """Malformed JSON, empty, null and binary frames interleaved with valid ones must each land
        on the caller that provoked them, because a crossed payload is this transport's silent failure."""
        with mock.patch.dict(os.environ, {"SNOWFLAKE_PAT": _PAT}, clear=False):
            peers: list[_ScriptedPeer] = []
            pool = _make_pool(_client_factory=_make_real_client_factory(peers, garbage_every=3), size=3)
            outcomes: list[str] = []
            crossed: list[str] = []
            outcome_lock = threading.Lock()

            def worker(index: int) -> None:
                for step in range(20):
                    marker = f"t{index}-{step}"
                    try:
                        _, payload = pool.post_json(_URL, _body(marker=marker), 1.0)
                    except Exception as error:  # noqa: BLE001 - an error is one valid outcome
                        with outcome_lock:
                            outcomes.append(f"raised: {type(error).__name__}")
                        continue
                    seen = _marker_of(payload)
                    if payload in _GARBAGE_PAYLOADS or seen is None:
                        with outcome_lock:
                            outcomes.append("garbage")
                        continue
                    if seen != marker:
                        with outcome_lock:
                            crossed.append(f"{marker} received {seen}")
                    with outcome_lock:
                        outcomes.append("served")

            _run_threads(worker, 8)

            self.assertEmpty(crossed, f"a caller received another caller's response: {crossed[:3]}")
            self.assertLen(outcomes, 8 * 20, "some caller got no outcome at all")
            self.assertGreater(outcomes.count("garbage"), 0, "no garbage frame was delivered; the test decayed")
            self.assertGreater(outcomes.count("served"), 0, "no valid response survived; the test decayed")
            pool.close()

    def test_duplicate_request_ids_in_a_concurrent_burst_are_rejected(self) -> None:
        """A batch with two identical ids must be refused before anything is sent, because matching
        responses to requests would be guesswork and the wrong guess is silent."""
        with mock.patch.dict(os.environ, {"SNOWFLAKE_PAT": _PAT}, clear=False):
            peers: list[_ScriptedPeer] = []
            pool = _make_pool(_client_factory=_make_real_client_factory(peers), size=2)
            rejected = [0]
            reject_lock = threading.Lock()

            def worker(index: int) -> None:
                for step in range(5):
                    bodies = [
                        _body(marker=f"t{index}-{step}-a", request_id="duplicate"),
                        _body(marker=f"t{index}-{step}-b", request_id="duplicate"),
                    ]
                    with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                        pool.post_json_many(_URL, bodies, 1.0)
                    with reject_lock:
                        rejected[0] += 1

            _run_threads(worker, 6)

            self.assertEqual(rejected[0], 30, "a duplicate-id batch was accepted by some thread")
            self.assertEqual(
                sum(len(peer.sent) for peer in peers),
                0,
                "a request went out on a batch whose correlation was already ambiguous",
            )
            pool.close()


class CorrelationTest(absltest.TestCase):
    """The one silent failure this transport has: a well-formed response carrying another request's
    feature values, with no error raised. Pipelining plus a shared pool is exactly how that would
    happen, so these are the most important tests in the file."""

    def test_concurrent_pipelined_batches_never_cross_responses(self) -> None:
        """Every response must match its own request even with many threads pipelining through one
        pool against peers answering out of order, because a crossed response is never detected."""
        with mock.patch.dict(os.environ, {"SNOWFLAKE_PAT": _PAT}, clear=False):
            peers: list[_ScriptedPeer] = []
            pool = _make_pool(_client_factory=_make_real_client_factory(peers, out_of_order=True), size=2)
            crossed: list[str] = []
            batches = [0]
            record_lock = threading.Lock()

            def worker(index: int) -> None:
                for step in range(6):
                    expected = [f"t{index}-{step}-{position}" for position in range(4)]
                    # Guard: identical markers would make any pairing look correct.
                    self.assertLen(set(expected), len(expected))
                    bodies = [_body(marker=marker) for marker in expected]
                    results = pool.post_json_many(_URL, bodies, 1.0)
                    got = [_marker_of(payload) for _, payload in results]
                    with record_lock:
                        batches[0] += 1
                        if got != expected:
                            crossed.append(f"sent {expected} got {got}")

            _run_threads(worker, 8)

            self.assertEmpty(crossed, f"a pipelined batch returned another request's response: {crossed[:2]}")
            self.assertEqual(batches[0], 48)
            self.assertGreater(sum(peer.receives for peer in peers), 0)
            pool.close()

    def test_pipelined_exchange_holds_order_under_shuffled_answers(self) -> None:
        """Results must come back in request order however the peer shuffles them, since the caller
        zips them against its own row list positionally."""
        crossed: list[str] = []
        record_lock = threading.Lock()

        def worker(index: int) -> None:
            rng = random.Random(index)
            for step in range(20):
                expected = [f"p{index}-{step}-{position}" for position in range(5)]
                with record_lock:
                    self.assertLen(set(expected), len(expected))
                bodies = [_body(marker=marker) for marker in expected]
                queued: list[dict[str, Any]] = []

                def send(frame: str, queued: list[dict[str, Any]] = queued) -> None:
                    queued.append(json.loads(frame))

                def recv(queued: list[dict[str, Any]] = queued, rng: random.Random = rng) -> str | bytes:
                    request = queued.pop(rng.randrange(len(queued)))
                    return json.dumps({"request_id": request["request_id"], "marker": request["marker"]})

                results = online_service_ws_pipeline.pipelined_exchange(send, recv, bodies)
                got = [_marker_of(payload) for _, payload in results]
                if got != expected:
                    with record_lock:
                        crossed.append(f"sent {expected} got {got}")

        _run_threads(worker, 8)
        self.assertEmpty(crossed, f"shuffled answers were returned out of request order: {crossed[:2]}")

    def test_a_response_for_a_foreign_request_is_refused_under_concurrency(self) -> None:
        """A response whose id belongs to nobody must raise rather than be handed to whoever is
        waiting, because that caller would silently receive another object's feature values."""
        refused = [0]
        refuse_lock = threading.Lock()

        def worker(index: int) -> None:
            for step in range(10):
                bodies = [_body(marker=f"f{index}-{step}-{position}") for position in range(3)]

                def send(frame: str) -> None:
                    del frame

                def recv() -> str | bytes:
                    return json.dumps({"status": "SUCCESS", "request_id": f"foreign-{index}"})

                with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                    online_service_ws_pipeline.pipelined_exchange(send, recv, bodies)
                with refuse_lock:
                    refused[0] += 1

        _run_threads(worker, 6)
        self.assertEqual(refused[0], 60)


class BackoffTest(absltest.TestCase):
    """Service-requested backoff is honored in the transport, which is shared by every thread. The
    hazard is one thread's wait or retry being credited to another's request."""

    def test_backoff_waits_and_no_thread_receives_another_thread_s_result(self) -> None:
        """Throttled requests must be waited out and resent for their own caller, because a retry
        that returns somebody else's response is indistinguishable from a correct one."""
        pool = _ThrottlingPool(retry_after_seconds=1.0)
        transport, slept = _make_transport(pool, failure_threshold=100)
        crossed: list[str] = []
        served = [0]
        record_lock = threading.Lock()

        def worker(index: int) -> None:
            for step in range(5):
                marker = f"b{index}-{step}"
                status, payload = transport.post_json(_HTTP_URL, _body(marker=marker), 30.0)
                with record_lock:
                    served[0] += 1
                    if status != 200 or _marker_of(payload) != marker:
                        crossed.append(f"{marker} received status {status} marker {_marker_of(payload)}")

        _run_threads(worker, 12)

        self.assertEmpty(crossed, f"a throttled retry returned the wrong result: {crossed[:3]}")
        self.assertEqual(served[0], 60)
        self.assertGreater(len(slept), 0, "a backoff request was never waited out")
        self.assertEqual(set(slept), {1.0}, "the honored wait was not the requested one")
        self.assertGreater(transport.honored_backoffs, 0)
        self.assertFalse(transport.using_fallback, "honored backoff must not be counted as a failure")
        self.assertEqual(transport.http_fallback_requests, 0)


class ChurnTest(absltest.TestCase):
    """The broadest stress: a full pool shared by many feature views, opening, recycling and
    closing at once. It exists to surface anything the narrower tests miss."""

    def test_many_views_over_a_churning_pool(self) -> None:
        """A pool churning under eight feature views must end closed with zero outstanding,
        because a leaked connection holds a service slot until the service times it out."""
        with mock.patch.dict(os.environ, {"SNOWFLAKE_PAT": _PAT}, clear=False):
            keys = [f"fv_{index}" for index in range(8)]
            peers: list[_ScriptedPeer] = []
            now = [0.0]
            pool = _make_pool(
                _client_factory=_make_real_client_factory(peers, hostname="instance-a"),
                size=4,
                max_requests_per_connection=5,
                max_connection_age_seconds=1.0,
                _clock=lambda: now[0],
            )
            # Every slot open before the churn starts. These eight views share one pool -- they
            # used to hold four connections each, 32 for this one caller -- so the slots held
            # are the pool's size no matter how many views are read through it.
            _fill_pool(pool, 4)
            for name in keys:
                pool.post_json(_URL, _body(name=name, marker="warm"), 1.0)
            self.assertEqual(pool.connection_count, 4)

            surprises: list[str] = []

            def worker(index: int) -> None:
                for step in range(20):
                    now[0] += 0.02
                    name = keys[(index + step) % len(keys)]
                    try:
                        # The fake clock is shared and every worker advances it, so ~4s of it
                        # passes in all; a 1s deadline can lapse for a worker the scheduler starves.
                        # Recycling is driven by the 1s max age, not by this deadline.
                        pool.post_json(_URL, _body(name=name, marker=f"c{index}-{step}"), 60.0)
                    except RuntimeError as error:
                        if "is closed" not in str(error):
                            surprises.append(repr(error))
                    except Exception as error:  # noqa: BLE001 - anything else is the finding
                        surprises.append(repr(error))

            _run_threads(worker, 10)

            self.assertEmpty(surprises, f"undocumented exception under churn: {surprises[:3]}")
            self.assertEqual(pool.connection_count, 4, "the pool drifted off its configured size")
            recycled = sum(1 for peer in peers if peer.close_calls > 0)
            self.assertGreater(recycled, 0, "nothing recycled; the churn never exercised retirement")
            for entries in pool._connections.values():  # noqa: SLF001 - white-box, repo style
                for pooled in entries:
                    self.assertEqual(pooled.outstanding, 0)

            pool.close()
            self.assertEqual(pool.connection_count, 0)
            for peer in peers:
                self.assertEqual(peer.close_calls, 1, "a socket was closed twice, or never")


class _InternalErrorPool:
    """A pool that answers every request with 500, optionally in lockstep.

    500 is the one status that rests the transport on the first occurrence rather than waiting for
    a failure streak, so it is the only fault that *walks* the ladder. ``barrier_parties`` holds
    each caller inside ``post_json`` until that many have arrived, which is what makes the
    concurrent case assertable: the hazard is several callers passing the ``using_fallback`` gate
    before any of them has set ``_fallback_until``, and left to chance that interleaving is rare.
    """

    def __init__(self, *, barrier_parties: int = 0) -> None:
        self.connection_count = 2
        self.closed = False
        self.calls = 0
        self.discarded = 0
        self.lock = threading.Lock()
        self._barrier = threading.Barrier(barrier_parties) if barrier_parties else None

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Answer with an INTERNAL error, after waiting for the other callers if asked.

        Args:
            url: Ignored.
            body: Request payload carrying a marker.
            timeout: Ignored.

        Returns:
            ``(500, body_bytes)``.
        """
        del url, timeout
        with self.lock:
            self.calls += 1
        if self._barrier is not None:
            self._barrier.wait(timeout=10)
        payload = {"error": {"code": "INTERNAL"}, "marker": body.get("marker")}
        return 500, json.dumps(payload).encode("utf-8")

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Serve a batch as a sequence of single requests.

        Args:
            url: Ignored.
            bodies: Request payloads.
            timeout: Ignored.

        Returns:
            One result per body, in order.
        """
        return [self.post_json(url, body, timeout) for body in bodies]

    def discard_all(self) -> int:
        """Record the teardown a 500 is supposed to trigger.

        Returns:
            The number of connections discarded.
        """
        with self.lock:
            self.discarded += 1
        return self.connection_count

    def close(self) -> None:
        """Mark the pool closed."""
        self.closed = True


class LadderEscalationTest(absltest.TestCase):
    """How far the fallback ladder advances per incident, and how concurrency changes that.

    ``_rest_transport`` advances one rung per 500 and the rung indexes
    ``DEFAULT_FALLBACK_LADDER_SECONDS``. Read one request at a time, that is a gentle escalation.
    But the rung is advanced *per 500* with no debounce, and ``using_fallback`` is only consulted
    on entry -- so every caller already past the gate when the first 500 lands advances it again.
    N callers in flight cost N rungs, not one.

    The serial case is covered single-threaded in ``online_service_ws_transport_test``. These
    exist because the concurrent case behaves differently, and a live run at four concurrent
    callers reached the ladder's ceiling on its first bad moment -- where the serial reading of
    the same code suggests only the first rung.

    Expectations are derived from ``DEFAULT_FALLBACK_LADDER_SECONDS`` rather than hardcoded, so
    changing the ladder does not silently invalidate these tests. That matters here: the ladder
    has already been lengthened once, from three rungs to four.
    """

    @property
    def ladder(self) -> tuple[float, ...]:
        """The ladder under test.

        Returns:
            The configured rest durations, shortest first.
        """
        return online_service_ws_transport.DEFAULT_FALLBACK_LADDER_SECONDS

    def test_one_caller_walks_the_ladder_one_rung_at_a_time(self) -> None:
        """Serial 500s advance exactly one rung each, then hold at the ceiling."""
        pool = _InternalErrorPool()
        now = [0.0]
        transport, _ = _make_transport(pool, _clock=lambda: now[0])
        top = len(self.ladder) - 1

        for index, rest in enumerate(self.ladder):
            transport.post_json(_HTTP_URL, _body(marker=f"m{index}"), 1.0)
            self.assertEqual(transport._fallback_rung, index, f"rung after incident {index + 1}")  # noqa: SLF001
            self.assertAlmostEqual(transport._fallback_until, now[0] + rest)  # noqa: SLF001
            # Step past this rest so the next request probes rather than short-circuiting.
            now[0] += rest

        # One more incident cannot climb past the ceiling, and the rest stays the last rung's.
        transport.post_json(_HTTP_URL, _body(marker="over"), 1.0)
        self.assertEqual(transport._fallback_rung, top, "the ladder must not climb past its last rung")  # noqa: SLF001
        self.assertAlmostEqual(transport._fallback_until, now[0] + self.ladder[top])  # noqa: SLF001

        incidents = len(self.ladder) + 1
        self.assertEqual(transport.http_fallback_requests, incidents)
        self.assertEqual(pool.discarded, incidents, "every 500 must discard the pool")

    def test_concurrent_callers_cost_one_rung_not_the_ceiling(self) -> None:
        """Callers that pass the gate together are one incident, so they advance the ladder once.

        Each in-flight caller reports the same 500. Counting each as its own incident would turn
        one bad moment into the ceiling's rest instead of the first rung's.
        """
        callers = len(self.ladder)
        pool = _InternalErrorPool(barrier_parties=callers)
        transport, _ = _make_transport(pool, _clock=lambda: 0.0)

        def worker(index: int) -> None:
            transport.post_json(_HTTP_URL, _body(marker=f"c{index}"), 1.0)

        _run_threads(worker, callers)

        self.assertEqual(pool.calls, callers, "every caller must have reached the pool")
        self.assertEqual(
            transport._fallback_rung,  # noqa: SLF001
            0,
            f"{callers} simultaneous 500s are one incident and should advance the ladder once",
        )
        self.assertAlmostEqual(transport._fallback_until, self.ladder[0])  # noqa: SLF001
        self.assertTrue(transport.using_fallback)
        # Correctness is unaffected: every caller still got an answer, over HTTP.
        self.assertEqual(transport.http_fallback_requests, callers)

    def test_callers_arriving_after_the_latch_do_not_advance_the_ladder(self) -> None:
        """Once rested, requests skip the pool entirely, so they cannot escalate further.

        The counterpart to the test above: escalation is bounded by how many callers are *already
        in flight*, not by how much traffic follows.
        """
        pool = _InternalErrorPool()
        transport, _ = _make_transport(pool, _clock=lambda: 0.0)

        transport.post_json(_HTTP_URL, _body(marker="first"), 1.0)
        self.assertEqual(transport._fallback_rung, 0)  # noqa: SLF001
        calls_after_first = pool.calls

        def worker(index: int) -> None:
            transport.post_json(_HTTP_URL, _body(marker=f"later{index}"), 1.0)

        _run_threads(worker, 8)

        self.assertEqual(pool.calls, calls_after_first, "a rested transport must not touch the pool")
        self.assertEqual(transport._fallback_rung, 0, "traffic during a rest must not lengthen it")  # noqa: SLF001
        self.assertEqual(transport.http_fallback_requests, 9)


if __name__ == "__main__":
    absltest.main()
