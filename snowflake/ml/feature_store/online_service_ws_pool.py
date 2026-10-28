"""Connection pool for the Online Service WebSocket transport.

A WebSocket connection is expensive to open — the handshake is the one authenticated request
of its life — and cheap to keep, so the pool exists to hold connections open across calls and
hand them out.

Three pooling decisions here are obligations rather than optimizations, because the service
cannot compensate for any of them:

* **Spread work across connections.** The gateway assigns an instance per connection, at
  random, once, and that connection stays on that instance for life. A client that always
  reuses the handiest connection pins all its load to one instance while the others idle.
  Round-robin and least-outstanding both avoid that; the client cannot target an instance
  and does not know which one it got.
* **Open connections only as concurrency needs them.** Each one is an authentication and a
  slot in the service's shared connection cap, so a new one is opened only when every held
  connection is busy, up to the pool size. A caller that never has two requests in flight
  needs one connection, and the service's short connection lifetime is what moves it between
  instances; concurrent callers spread over as many as they keep busy.
* **Recycle with splay.** Connections opened together expire together, so N connections
  redialing in the same instant is a burst of N authentications. Every connection therefore
  gets its own recycle thresholds -- request budget and age ceiling both -- spread around the
  nominal ones.

The pool also retires connections *before* the service's own age limit, so a caller is not
interrupted mid-request by a close it could have anticipated. The service advertises each
connection's lifetime on the upgrade, and the pool retires a small margin under it; the
configured age is the ceiling, and the only limit when nothing is advertised.

All per-connection accounting (``outstanding``, ``requests``, ``retiring``) is read and written
under the pool's one lock, and nothing is closed while that lock is held: a WebSocket close
waits for the server's reply, and holding the lock through it would stall every caller.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from snowflake.ml.feature_store import online_service_ws_client
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# Ceiling on connection age, and the only limit when the service does not advertise one. 12
# minutes sits under the 15-minute cap the service had when it did not advertise its age.
DEFAULT_MAX_CONNECTION_AGE_SECONDS = 12 * 60.0

# Retire this far under the age the service advertised for a connection: the smaller of a
# fixed margin and a fraction of the age, so short-lived connections are not retired at birth.
# It covers the latency between the service starting its clock and a request reaching it.
SERVER_AGE_MARGIN_SECONDS = 5.0
SERVER_AGE_MARGIN_FRACTION = 0.1

# Nominal request budget per connection, with a fraction of splay either side so connections
# opened together do not retire together.
DEFAULT_MAX_REQUESTS_PER_CONNECTION = 100_000
DEFAULT_SPLAY_FRACTION = 0.25

# Ceiling on connections held per endpoint, not a number to open: one is opened only when every
# one held is busy. That makes growth demand-driven per selection but not self-limiting over time
# -- the transient all-busy moments accumulate, and a long-lived caller reaches this ceiling and
# stays there. Four threads against a ceiling of 32 finished a 20-minute run holding all 32.
#
# 16 is where measured throughput stops paying at both ten and a hundred keys per read: past it,
# added connections inflate each other's wire time, and below it the tail and the hundred-key
# regime pay. A high ceiling costs fleet connection slots, not latency. One interpreter tops out
# well before 16 busy connections, so add processes, not threads. Measurements are in
# docs/websocket_query_transport/WEBSOCKET_CLIENT_DESIGN.md ("Pool size is load-bearing").
DEFAULT_POOL_SIZE = 16

# Most of a caller's timeout that may be spent waiting for a connection. Past this the pool gives
# up rather than queueing on, because the caller's deadline is the HTTP fallback's budget too: a
# caller that queues until the deadline reaches the fallback with nothing left, and the timeout it
# is owed is the only thing the fallback can raise. Giving up at half leaves the other half to
# serve the read over HTTP, which is the whole point of having a fallback -- a saturated pool is
# exactly the case it exists for.
POOL_QUEUE_SHARE_OF_TIMEOUT = 0.5

# Share of a caller's timeout that, once spent queueing, makes a wire timeout contention rather
# than evidence against the connection. Below the share above, so it covers the caller that did
# get a connection but late: handed the lock with a third of its budget left, it is not a fair
# test of a healthy socket, and counting it rests every read on HTTP for a minute -- so an
# ordinary burst would switch the feature off.
CONTENTION_SHARE_OF_TIMEOUT = 0.25

# Monotonic ids so a connection can be named in a log line or a snapshot. The server
# side learned this the hard way: it replaced a bare counter with a registry because a
# number cannot be addressed, only reported.
_connection_ids = itertools.count(1)


class PoolWaitTimeout(TimeoutError):
    """A caller's deadline was spent queueing for a pooled connection rather than on the wire.

    Still a ``TimeoutError``, so a caller sees the same failure as a timeout on the wire. It is
    a distinct type because the transport above must not read it as one: a wait in the queue
    is contention on a healthy connection, not evidence the WebSocket path is broken.

    Raised for a wait that outlived the deadline before anything was sent, and also for a wire
    timeout on a request that had already spent most of its budget queueing -- the common shape
    once a pool is saturated, since a caller usually gets the lock just too late rather than not
    at all. Both are the pool being too small for the offered concurrency, which is a reason to
    raise ``SNOWFLAKE_ML_ONLINE_WS_POOL_SIZE`` and not a reason to stop using WebSockets.
    """


class _PooledConnection:
    """One connection plus the counters that decide when to retire it."""

    def __init__(
        self,
        client: online_service_ws_client.OnlineServiceWsClient,
        request_budget: int,
        created_at: float,
        max_age_seconds: float = DEFAULT_MAX_CONNECTION_AGE_SECONDS,
    ) -> None:
        self.connection_id = next(_connection_ids)
        self.client = client
        self.request_budget = request_budget
        self.created_at = created_at
        # This connection's own age ceiling, splayed like the request budget so connections
        # opened together do not all reach it in the same instant.
        self.max_age_seconds = max_age_seconds
        self.requests = 0
        # Requests reserved or in flight on this connection. Guarded, with ``requests`` and
        # ``retiring``, by the pool's ``_select_lock``: they are read-modify-write and are read to
        # decide whether the connection can be closed, so updating them outside that lock races.
        self.outstanding = 0
        # Set once the connection is due to go. It is removed from the pool at that moment so
        # no new request picks it, and closed later, once its in-flight work drains.
        self.retiring = False
        # A WebSocket permits one reader and one writer. Two callers interleaving sends on
        # one socket can each read the other's response, which would return another
        # request's feature values with no error raised, so use of a connection is
        # serialized rather than merely accounted for.
        self.lock = threading.Lock()

    def age_seconds(self, now: float) -> float:
        """Seconds since this connection was created.

        Args:
            now: A ``time.monotonic()`` reading.

        Returns:
            Age in seconds.
        """
        return now - self.created_at

    def is_retired(self, now: float, max_age_seconds: float) -> bool:
        """Whether this connection should be replaced before being handed out again.

        Args:
            now: A ``time.monotonic()`` reading.
            max_age_seconds: Age at which a connection is retired.

        Returns:
            True when the connection is over age, over its request budget, or within the margin
            of the lifetime the service advertised for it.
        """
        if self.requests >= self.request_budget or self.age_seconds(now) >= max_age_seconds:
            return True
        advertised: float | None = getattr(self.client, "server_max_conn_age_seconds", None)
        if not advertised:
            return False
        # Dated from the client's dial rather than from when this slot was handed out. The two
        # differ twice over: a replacement slot can sit unused for a while after a retirement,
        # and the client re-dials transparently on the next use after a mid-request failure. A
        # slot-local timestamp would measure the new socket against the old socket's dial and
        # retire a connection seconds old -- one handshake per transient failure, which is the
        # authentication burst the splay exists to prevent.
        #
        # Read only once the advertised age is known to exist, rather than above this check:
        # a service that advertises nothing takes the early return, and this runs for every
        # connection on every selection.
        dialed_at: float | None = getattr(self.client, "connected_at", None)
        if dialed_at is None:
            return False
        margin = min(SERVER_AGE_MARGIN_SECONDS, advertised * SERVER_AGE_MARGIN_FRACTION)
        return now - dialed_at >= advertised - margin


@dataclass(frozen=True)
class ConnectionInfo:
    """One pooled connection's observable state.

    Attributes:
        connection_id: Monotonic id, stable for this connection's life.
        key: The feature view or feature group the connection is scoped to.
        position: Index within that key's pool.
        age_seconds: Seconds since the slot was created.
        requests: Requests sent on it so far.
        outstanding: Requests in flight right now.
        request_budget: Recycle threshold for this connection.
        is_connected: Whether the socket is actually open.
        server_hostname: The serving instance, once a response has reported one.
    """

    connection_id: int
    key: tuple[str, ...]
    position: int
    age_seconds: float
    requests: int
    outstanding: int
    request_budget: int
    is_connected: bool
    server_hostname: str | None


class OnlineServiceWsPool:
    """Holds WebSocket connections open across calls and spreads requests across them.

    Satisfies :class:`online_service_transport.BatchingOnlineServiceTransport`. Connections
    are keyed by endpoint, not by what is read over them: the handshake carries only the
    caller's credentials, and every message names its own feature view or feature group, so
    any connection to an endpoint can serve any read of it.
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        size: int = DEFAULT_POOL_SIZE,
        max_connection_age_seconds: float = DEFAULT_MAX_CONNECTION_AGE_SECONDS,
        max_requests_per_connection: int = DEFAULT_MAX_REQUESTS_PER_CONNECTION,
        splay_fraction: float = DEFAULT_SPLAY_FRACTION,
        least_outstanding: bool = False,
        _client_factory: Callable[..., Any] | None = None,
        _rng: random.Random | None = None,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        """Build an empty pool. Connections are opened on demand.

        Args:
            session: Snowpark session used only for auth resolution.
            size: Most connections held per endpoint. Opened only as concurrency needs them.
            max_connection_age_seconds: Retire a connection once it reaches this age, or earlier
                when the service advertises a shorter lifetime for it.
            max_requests_per_connection: Nominal request budget before recycling.
            splay_fraction: Fraction of the budget to spread thresholds by, either side. The age
                ceiling is spread by the same fraction, downward only, since it is a ceiling.
            least_outstanding: Select the least-busy connection instead of round-robin. Ties go to
                the round-robin cursor, so an idle pool still rotates.
            _client_factory: Test seam returning a client-shaped object.
            _rng: Test seam for deterministic splay.
            _clock: Test seam returning monotonic seconds.
        """
        self._session = session
        self._size = max(1, size)
        self._max_connection_age_seconds = max_connection_age_seconds
        self._max_requests_per_connection = max(1, max_requests_per_connection)
        self._splay_fraction = min(max(splay_fraction, 0.0), 0.9)
        self._least_outstanding = least_outstanding
        self._client_factory = _client_factory or online_service_ws_client.OnlineServiceWsClient
        self._rng = _rng or random.Random()
        self._clock = _clock or time.monotonic

        self._connections: dict[tuple[str, ...], list[_PooledConnection]] = {}
        # Retired connections still finishing their in-flight work. Out of rotation, not
        # yet closed; bounded by the per-request deadline, since a request cannot outlive it.
        self._draining: list[_PooledConnection] = []
        self._next_index: dict[tuple[str, ...], int] = {}
        self._closed = False
        # Guards the selection bookkeeping below: growing a pool, rotating the
        # round-robin index, and swapping a retired connection are all read-then-write.
        self._select_lock = threading.Lock()

    # ----- keying ----------------------------------------------------------

    @staticmethod
    def connection_key(url: str) -> tuple[str, ...]:
        """Derive the pool key for a request.

        The endpoint alone. Keying by the object read instead multiplied sockets by the number
        of feature views a caller touched -- 25 views at the default size held 100 authenticated
        connections -- for no gain, since nothing is bound to a connection at the handshake.

        Args:
            url: The WebSocket query URL.

        Returns:
            A tuple key identifying the endpoint.
        """
        return (url,)

    # ----- selection -------------------------------------------------------

    def _request_budget(self) -> int:
        """Pick this connection's recycle threshold, splayed around the nominal budget.

        Returns:
            A request count. Distinct per connection when splay is enabled, so connections
            opened together do not retire together.
        """
        if self._splay_fraction <= 0:
            return self._max_requests_per_connection
        spread = self._max_requests_per_connection * self._splay_fraction
        offset = self._rng.uniform(-spread, spread)
        return max(1, int(self._max_requests_per_connection + offset))

    def _max_age(self) -> float:
        """Pick this connection's age ceiling, splayed downward from the configured one.

        Without this every connection opened together reaches the configured age in the same
        instant whenever the service does not advertise its own -- the redial burst splay is
        there to prevent. Downward only: the configured value is a ceiling the caller set.

        Returns:
            Seconds, between ``(1 - splay_fraction)`` of the configured age and the age itself.
        """
        if self._splay_fraction <= 0:
            return self._max_connection_age_seconds
        return self._max_connection_age_seconds * (1.0 - self._rng.uniform(0.0, self._splay_fraction))

    def _new_connection(self) -> _PooledConnection:
        """Open a pooled connection with its own splayed budget and age ceiling.

        Returns:
            The new pooled connection.
        """
        # The pool's clock goes to the client too, so the dial time :meth:`is_retired` reads
        # back off it is on the same clock as the ``now`` it compares against.
        client = self._client_factory(session=self._session, _clock=self._clock)
        return _PooledConnection(client, self._request_budget(), self._clock(), self._max_age())

    def _select(self, key: tuple[str, ...]) -> _PooledConnection:
        """Choose a connection for ``key``, growing the pool and retiring stale entries.

        Args:
            key: Pool key from :meth:`connection_key`.

        Returns:
            A usable pooled connection, already reserved — the caller must hand it back through
            :meth:`_finish` when done.
        """
        with self._select_lock:
            chosen, to_close = self._select_locked(key)
        # Outside the lock: a close waits on the server's close reply.
        for client in to_close:
            _close_quietly(client)
        return chosen

    def _select_locked(self, key: tuple[str, ...]) -> tuple[_PooledConnection, list[Any]]:
        """Selection proper. Callers must hold ``_select_lock``.

        Args:
            key: Pool key from :meth:`connection_key`.

        Returns:
            A usable pooled connection, already reserved, and the clients of drained retired
            connections, which the caller must close once it has released the lock.

        Raises:
            RuntimeError: When the pool was closed while this caller was on its way in.
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsPool is closed")
        now = self._clock()
        pool = self._connections.setdefault(key, [])

        # Retire in two phases, because waiting for a connection to look idle before retiring
        # it means never retiring the busy ones — and under sustained load every connection is
        # busy every time it is examined, so the age limit and the request budget would both go
        # unenforced exactly when they matter. Instead the due connection leaves the pool
        # immediately, so no new request picks it, and it is closed once its in-flight work
        # drains. Same shape as the server's own drain: stop handing it out, then wait.
        for position, pooled in enumerate(pool):
            if not pooled.retiring and pooled.is_retired(now, pooled.max_age_seconds):
                logger.debug(
                    "Retiring online feature retrieval connection after %d requests, age %.1fs",
                    pooled.requests,
                    pooled.age_seconds(now),
                )
                # Replacement first, then retire. If building it raises, the slot is left
                # untouched and is reconsidered on the next selection; marking first would
                # leave an entry that is ``retiring`` *and* still in the pool, which both
                # retirement sweeps skip and the picker would keep handing out until its
                # last caller closed it under everybody.
                replacement = self._new_connection()
                pooled.retiring = True
                self._draining.append(pooled)
                pool[position] = replacement

        # Other keys too. Only the requested key was ever examined, so a feature view that was
        # read and then went quiet kept its connections indefinitely, past both the age limit and
        # the server's own close. Nobody is reading those keys, so a due connection is removed
        # rather than replaced -- the next read of that key opens one on demand -- and a key left
        # with nothing is dropped. keys x size is small, so the scan is cheap.
        for other_key in [k for k in self._connections if k != key]:
            kept = []
            for pooled in self._connections[other_key]:
                if not pooled.retiring and pooled.is_retired(now, pooled.max_age_seconds):
                    pooled.retiring = True
                    self._draining.append(pooled)
                else:
                    kept.append(pooled)
            if kept:
                self._connections[other_key] = kept
            else:
                del self._connections[other_key]
                self._next_index.pop(other_key, None)

        to_close = self._take_drained_locked()

        # Grow only when nothing held is idle. Growing on every selection opened the full size on
        # the first few reads even for a caller that never has two in flight, and each of those
        # sockets is an authentication and a slot in the service's shared connection cap.
        if len(pool) < self._size and all(pooled.outstanding > 0 for pooled in pool):
            chosen = self._new_connection()
            pool.append(chosen)
        else:
            chosen = self._pick_locked(key, pool)

        # Reserve before releasing the lock. Incrementing after the caller has the connection
        # leaves a window where it still looks idle, and a concurrent selection can retire it
        # out from under a caller that is about to send on it.
        chosen.outstanding += 1
        return chosen, to_close

    def _pick_locked(self, key: tuple[str, ...], pool: list[_PooledConnection]) -> _PooledConnection:
        """Pick from a full pool, scanning from the round-robin cursor. Callers must hold ``_select_lock``.

        Round-robin takes the first idle connection at or after the cursor, and the cursor itself
        only when every connection is busy: advancing onto a busy one queues the caller on its
        lock behind a slow batch while other connections sit idle. Least-outstanding takes the
        least-busy connection, breaking ties in cursor order -- ``min`` alone returns the first
        entry on every tie, which on an idle pool is always connection 0.

        Args:
            key: Pool key, for the cursor.
            pool: That key's connections; never empty.

        Returns:
            The chosen connection. The cursor moves past it.
        """
        start = self._next_index.get(key, 0) % len(pool)
        order = [(start + offset) % len(pool) for offset in range(len(pool))]
        if self._least_outstanding:
            index = min(order, key=lambda i: pool[i].outstanding)  # min keeps the first of equals
        else:
            index = next((i for i in order if pool[i].outstanding == 0), start)
        self._next_index[key] = (index + 1) % len(pool)
        return pool[index]

    # ----- transport surface -----------------------------------------------

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Send one body on a pooled connection.

        Args:
            url: The WebSocket query URL.
            body: JSON-serializable payload.
            timeout: Per-request timeout in seconds.

        Returns:
            ``(status_code, body_bytes)``.

        Raises:
            RuntimeError: When the pool has been closed.
            online_service_ws_client.OnlineServiceWsDialError: When the connection could not
                be opened; the slot is evicted before this propagates.
            SnowflakeMLException: Wrapping a :class:`PoolWaitTimeout` when the deadline went on
                queueing rather than on the wire. Any other failure from the exchange propagates
                as the connection raised it.

        # noqa: DAR401
        # noqa: DAR402
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsPool is closed")
        key = self.connection_key(url)
        arrived = self._clock()
        deadline = arrived + timeout
        pooled = self._select(key)  # returns reserved
        dial_failed = False
        sent = 0
        try:
            with _locked_until(pooled.lock, deadline, self._clock, max_wait=timeout * POOL_QUEUE_SHARE_OF_TIMEOUT):
                # Measured from the caller's arrival, not from the lock, so it covers the wait
                # for the pool's own lock as well as for this connection: both are contention.
                queued = self._clock() - arrived
                # Counted after the exchange, not before: the budget exists to recycle a
                # connection that has *done* work, and charging a request that never left --
                # a dial refusal, a deadline that expired while queued, an unserializable
                # body -- retires failing connections faster than working ones, so every
                # failure buys an extra handshake.
                try:
                    result = pooled.client.post_json(url, body, _remaining(deadline, self._clock))
                except BaseException as error:
                    contention = _as_contention(error, queued, timeout)
                    if contention is None:
                        raise
                    raise contention from error
                sent = 1
                return result
        except online_service_ws_client.OnlineServiceWsDialError:
            # A connection that never opened must not keep occupying a slot: the slot is a
            # shared, per-instance resource, and a caller holding dead ones reduces the
            # service's effective capacity for everybody.
            dial_failed = True
            raise
        finally:
            self._finish(key, pooled, requests=sent, evict=dial_failed)

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Send a batch on a single pooled connection, so the burst shares one round trip.

        Args:
            url: The WebSocket query URL.
            bodies: JSON-serializable payloads, in the order results are wanted.
            timeout: Timeout in seconds for the whole batch.

        Returns:
            One ``(status_code, body_bytes)`` per entry in ``bodies``, in the same order.

        Raises:
            RuntimeError: When the pool has been closed.
            online_service_ws_client.OnlineServiceWsDialError: When the connection could not
                be opened; the slot is evicted before this propagates.
            SnowflakeMLException: Wrapping a :class:`PoolWaitTimeout` when the deadline went on
                queueing rather than on the wire. Any other failure from the exchange propagates
                as the connection raised it.

        # noqa: DAR401
        # noqa: DAR402
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsPool is closed")
        if not bodies:
            return []
        key = self.connection_key(url)
        arrived = self._clock()
        deadline = arrived + timeout
        pooled = self._select(key)  # returns reserved
        dial_failed = False
        sent = 0
        try:
            with _locked_until(pooled.lock, deadline, self._clock, max_wait=timeout * POOL_QUEUE_SHARE_OF_TIMEOUT):
                queued = self._clock() - arrived
                # Counted after the exchange, for the reason given in :meth:`post_json`. A
                # batch makes it louder: a failed 500-body call would otherwise burn 500 of
                # the connection's budget having sent nothing.
                try:
                    results = pooled.client.post_json_many(url, bodies, _remaining(deadline, self._clock))
                except BaseException as error:
                    contention = _as_contention(error, queued, timeout)
                    if contention is None:
                        raise
                    raise contention from error
                sent = len(bodies)
                return results
        except online_service_ws_client.OnlineServiceWsDialError:
            dial_failed = True
            raise
        finally:
            self._finish(key, pooled, requests=sent, evict=dial_failed)

    def discard_all(self) -> int:
        """Close and forget every pooled connection, leaving the pool usable.

        Distinct from :meth:`close`, which is permanent. This is for a service error that
        invalidates the connections rather than the pool: the design doc's response to an
        INTERNAL error is to close the WebSocket rather than keep querying over it, and the pool
        must still be able to re-dial once the transport is retried.

        Two-phase, like the age and budget retirement in :meth:`_select_locked`, and for the same
        reason: ``_select`` reserves a connection before releasing the lock, so a connection with
        ``outstanding > 0`` has a caller mid-request on it. Closing it under that caller turns a
        request that was about to succeed into a failure -- and the trigger for this is an
        INTERNAL error on *one* connection, so closing eagerly would fail every concurrent
        in-flight request across the whole pool rather than the one that went wrong. Idle
        connections close now; busy ones leave the pool immediately, so no new request picks
        them, and close when their last in-flight request finishes.

        Returns:
            How many connections were removed from rotation, whether closed yet or not.
        """
        with self._select_lock:
            if self._closed:
                return 0
            held = [pooled for pool in self._connections.values() for pooled in pool]
            held.extend(self._draining)
            self._connections.clear()
            self._next_index.clear()

            idle = [pooled.client for pooled in held if pooled.outstanding == 0]
            busy = [pooled for pooled in held if pooled.outstanding != 0]
            for pooled in busy:
                pooled.retiring = True
            self._draining = busy

        # Outside the lock, for the same reason close() does it: a close handler may call back in.
        for client in idle:
            _close_quietly(client)
        return len(held)

    def close(self) -> None:
        """Close every pooled connection. Idempotent, and never raises.

        Two-phase like :meth:`discard_all`, and for the same reason: a connection with
        ``outstanding > 0`` has a caller mid-request on it, holding ``pooled.lock`` and very
        likely blocked in ``recv``. A WebSocket permits one reader and one writer, and closing
        sends a close frame and waits for the server's reply, so closing from here would be a
        second thread on that socket -- the hazard ``pooled.lock`` exists to prevent, and one
        that can misattribute a frame rather than merely cancel a request. Closing the pool is
        permanent, so nothing new is handed out either way; the busy ones simply close as their
        callers finish, through the same :meth:`_finish` path a retirement uses.
        """
        with self._select_lock:
            if self._closed:
                return
            # Set inside the lock and refuse later selections, so a caller that passed the
            # closed check cannot repopulate the pool behind us and leave a socket open.
            self._closed = True
            held = [pooled for pool in self._connections.values() for pooled in pool]
            held.extend(self._draining)
            self._connections.clear()
            self._next_index.clear()

            clients = [pooled.client for pooled in held if pooled.outstanding == 0]
            busy = [pooled for pooled in held if pooled.outstanding != 0]
            for pooled in busy:
                pooled.retiring = True
            self._draining = busy

        # Outside the lock: a close handler may call back into the pool, and holding the lock
        # across an arbitrary callback would deadlock.
        for client in clients:
            _close_quietly(client)

    def _finish(self, key: tuple[str, ...], pooled: _PooledConnection, *, requests: int, evict: bool) -> None:
        """Hand a reserved connection back, evicting it after a failed dial, and close it if it is done.

        Everything is decided under ``_select_lock``, the same lock ``_select`` reserves under:
        ``outstanding`` and ``requests`` are read-modify-write, ``retiring`` can be set by a
        concurrent selection, and the close decision reads all three. Split across locks, a lost
        decrement keeps a drained connection open forever, and a lost increment lets it be closed
        under a caller still reserved on it.

        A failed dial removes the slot from rotation at once, but closes the client only if no
        other caller is still reserved on it; otherwise it drains like any retired connection and
        its last caller closes it.

        Args:
            key: The pool key the connection is held under.
            pooled: The connection this caller reserved.
            requests: Requests this caller sent on it.
            evict: Whether this caller's dial failed.
        """
        with self._select_lock:
            pooled.outstanding -= 1
            pooled.requests += requests
            if evict:
                entries = self._connections.get(key)
                if entries is not None and pooled in entries:
                    entries.remove(pooled)
                if not pooled.retiring:
                    pooled.retiring = True
                    self._draining.append(pooled)
            close_now = pooled.retiring and pooled.outstanding == 0 and pooled in self._draining
            if close_now:
                self._draining.remove(pooled)
        if close_now:
            _close_quietly(pooled.client)

    def _take_drained_locked(self) -> list[Any]:
        """Remove retired connections whose in-flight work has finished, for closing outside the lock.

        Callers must hold ``_select_lock``, and must close the returned clients only after
        releasing it.

        Returns:
            The clients to close.
        """
        if not self._draining:
            return []
        drained = [pooled.client for pooled in self._draining if pooled.outstanding == 0]
        self._draining = [pooled for pooled in self._draining if pooled.outstanding != 0]
        return drained

    # ----- introspection, for tests and telemetry --------------------------

    @property
    def connection_count(self) -> int:
        """Slots currently held across all keys, opened or not."""
        with self._select_lock:
            return sum(len(pool) for pool in self._connections.values())

    @property
    def live_connection_count(self) -> int:
        """Slots whose connection is actually open.

        Distinct from :attr:`connection_count` because a slot is claimed before its
        connection is dialed, so the two differ whenever a dial is pending or has failed.

        Returns:
            The number of slots whose socket is open.
        """
        # Snapshot under the lock: a first read of a new feature view adds a key, and telemetry
        # sampling the pool must not fail because it happened to look during that.
        with self._select_lock:
            entries = [pooled for pool in self._connections.values() for pooled in pool]
        return sum(1 for pooled in entries if getattr(pooled.client, "is_connected", False))

    @property
    def draining_connection_count(self) -> int:
        """Retired connections still finishing in-flight work, out of rotation.

        Returns:
            How many connections are draining.
        """
        # Under the lock like every other counter here: ``_draining`` is rebound wholesale by
        # discard_all, close and the drain sweep, so an unguarded read can land on either
        # generation and report a count that was never true.
        with self._select_lock:
            return len(self._draining)

    def connections(self) -> list[ConnectionInfo]:
        """Snapshot every held connection, for diagnostics and telemetry.

        Returns:
            One entry per slot, with its id, age, counters and serving instance. Ordered by
            key then position so repeated calls are comparable.
        """
        now = self._clock()
        snapshot: list[ConnectionInfo] = []
        with self._select_lock:
            for key, entries in sorted(self._connections.items()):
                for position, pooled in enumerate(entries):
                    snapshot.append(
                        ConnectionInfo(
                            connection_id=pooled.connection_id,
                            key=key,
                            position=position,
                            age_seconds=pooled.age_seconds(now),
                            requests=pooled.requests,
                            outstanding=pooled.outstanding,
                            request_budget=pooled.request_budget,
                            is_connected=bool(getattr(pooled.client, "is_connected", False)),
                            server_hostname=getattr(pooled.client, "server_hostname", None),
                        )
                    )
        return snapshot

    def server_hostnames(self) -> dict[str, int]:
        """Count connections per serving instance, so uneven spread is observable.

        Returns:
            Instance hostname to connection count, for connections that have reported one.
        """
        counts: dict[str, int] = {}
        for info in self.connections():
            if info.server_hostname:
                counts[info.server_hostname] = counts.get(info.server_hostname, 0) + 1
        return counts

    def request_budgets(self) -> list[int]:
        """Per-connection recycle thresholds, for asserting that splay actually spread them.

        Returns:
            One threshold per held connection.
        """
        with self._select_lock:
            return [pooled.request_budget for pool in self._connections.values() for pooled in pool]


def _remaining(deadline: float, clock: Callable[[], float]) -> float:
    """Seconds left before ``deadline``, raising the client's timeout error once none are.

    Args:
        deadline: Absolute deadline on ``clock``.
        clock: Monotonic seconds.

    Returns:
        A positive number of seconds.

    Raises:
        SnowflakeMLException: When the deadline has passed, wrapping a :class:`PoolWaitTimeout`.

    # noqa: DAR401
    # noqa: DAR402
    """
    left = deadline - clock()
    if left <= 0:
        raise online_service_ws_client._deadline_exceeded(  # noqa: SLF001
            "waiting for a pooled connection", PoolWaitTimeout
        )
    return left


@contextlib.contextmanager
def _locked_until(
    lock: threading.Lock,
    deadline: float,
    clock: Callable[[], float],
    max_wait: float | None = None,
) -> Iterator[None]:
    """Hold ``lock``, waiting no later than ``deadline`` and no longer than ``max_wait``.

    A caller queued behind a slow request on the same connection used to wait on the lock
    unbounded and then hand the client its whole timeout afresh, so its real worst case was
    the queue time plus the timeout. The wait now spends the caller's own deadline.

    ``max_wait`` stops it spending *all* of it. The caller's deadline is also the HTTP
    fallback's budget, so a caller that queues to the deadline arrives at the fallback with no
    time to be served and gets the timeout it was owed instead of its rows -- which is how a
    saturated pool failed reads outright while the fallback it has sat unused.

    Args:
        lock: The connection's lock.
        deadline: Absolute deadline on ``clock``.
        clock: Monotonic seconds.
        max_wait: Longest wait to allow, if sooner than the deadline.

    Yields:
        Nothing; the lock is held for the body of the ``with``.

    Raises:
        SnowflakeMLException: When the lock is not free before the deadline, wrapping a
            :class:`PoolWaitTimeout`.

    # noqa: DAR401
    # noqa: DAR402
    """
    wait = _remaining(deadline, clock)
    capped = False
    if max_wait is not None and max_wait < wait:
        wait = max(0.0, max_wait)
        capped = True
    if not lock.acquire(timeout=wait):
        where = (
            f"after waiting {wait:.3f}s for a pooled connection; the pool is saturated, so the "
            f"rest of the timeout is left for the HTTP fallback and raising "
            f"SNOWFLAKE_ML_ONLINE_WS_POOL_SIZE would serve this read over WebSocket"
            if capped
            else "waiting for a pooled connection"
        )
        raise online_service_ws_client._deadline_exceeded(where, PoolWaitTimeout)  # noqa: SLF001
    try:
        yield
    finally:
        lock.release()


def _as_contention(error: BaseException, queued: float, timeout: float) -> BaseException | None:
    """Re-read a failed exchange as pool contention, when that is what the clock says it was.

    A saturated pool rarely denies a caller the connection outright. It hands it over just too
    late, and the exchange then times out on whatever budget is left -- a plain timeout, which
    the transport counts against the WebSocket path. Three of those rest every read on HTTP for
    a minute, so an ordinary burst against a healthy service switches the feature off, and the
    pool is the only part that knows how the deadline was divided.

    Args:
        error: What the exchange raised.
        queued: Seconds spent between the caller arriving and the connection being granted.
        timeout: The caller's whole timeout.

    Returns:
        A :class:`PoolWaitTimeout` to raise instead, or None to let ``error`` stand -- which it
        does unless the failure was a timeout and queueing took most of the budget. Anything
        that is not a timeout is evidence about the connection whatever the queue did.
    """
    if timeout <= 0 or queued < timeout * CONTENTION_SHARE_OF_TIMEOUT:
        return None
    timed_out = error if isinstance(error, TimeoutError) else getattr(error, "original_exception", None)
    if not isinstance(timed_out, TimeoutError) or isinstance(timed_out, PoolWaitTimeout):
        return None
    return online_service_ws_client._deadline_exceeded(  # noqa: SLF001
        f"after spending {queued:.3f}s of its {timeout:.3f}s timeout waiting for a pooled "
        f"connection; the pool is saturated, so raise SNOWFLAKE_ML_ONLINE_WS_POOL_SIZE",
        PoolWaitTimeout,
    )


def _close_quietly(client: Any) -> None:
    """Close a client, suppressing failures.

    Args:
        client: Anything with a ``close`` method.
    """
    try:
        client.close()
    except Exception:  # noqa: BLE001 - best-effort close, never raise from cleanup
        logger.debug("Online feature retrieval connection close suppressed", exc_info=True)
