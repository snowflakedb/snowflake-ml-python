"""Policy layer for the Online Service WebSocket transport.

This is what the Feature Store hands to the query path. It owns every decision about *when*
to use the WebSocket transport, when to retry, and when to stop trying:

* **Disconnects are normal.** The service closes every connection on a timer, so a caller
  sees several closes an hour by design. Treating each one as an incident would drive a
  permanent fall back to HTTP and forfeit the whole benefit of the transport, so a close on
  its own never counts against the fallback budget.
* **Closes are not diagnosed.** Intent is not inferred from connection age, which misreads a
  redeploy because a drain closes connections of every age; nor from the close code, which is
  absent whenever a close is frameless and so cannot carry control flow. A dead socket earns one
  redial, and the *redial* is the signal: it either succeeds, or it fails as a dial error carrying a
  real HTTP status. A close must therefore reach this module as an exception — an empty frame is a
  close, not a response, and is rejected in the client rather than decoded.
* **Redials are jittered.** A close affects every connection at once, and every handshake is
  an authentication, so synchronized redials arrive as a burst. Each redial waits a random
  interval instead.
* **Backoff guidance is honored.** When the service asks for a delay before a retry, the
  wait happens here — this is the only layer that can see it, because the status-mapping the
  caller uses collapses several distinct failures into one error. A delay the caller has no
  time for is not waited at all: the throttled answer goes back instead of a timeout.
* **One budget per call.** The caller's timeout covers every WebSocket attempt, the waits
  between them and the HTTP fallback, so retrying can never stretch a read past it. A timeout
  is not retried, since the budget it spent is the one a retry would need, and a wait for a
  pooled connection is contention rather than a broken transport, so it is not counted.
* **Fallback is a ladder, not a switch.** Repeated hard failures move to HTTP for a growing
  interval and then probe again, rather than giving up permanently or hammering.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from importlib import util as importlib_util
from typing import Any, Callable, TypeVar

from snowflake.ml.feature_store import (
    online_service_transport,
    online_service_ws_client,
    online_service_ws_pool,
)
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# Redial delay bounds. Small enough to be invisible next to a handshake, large enough to
# desynchronize a pool that all closed at once.
DEFAULT_REDIAL_JITTER_SECONDS = 0.25

# Fallback ladder: HTTP for this long, then probe the WebSocket transport again. The last rung is
# the ceiling: a persistent outage is probed every ten minutes for as long as it lasts, so a
# recovered service is picked up again without a restart, while a dead one costs one probe in ten
# minutes. A refusal that will not clear soon (404 transport off, 401/403) goes straight to it.
DEFAULT_FALLBACK_LADDER_SECONDS: tuple[float, ...] = (60.0, 120.0, 300.0, 600.0)

# Consecutive hard failures tolerated before falling back. A close is not a hard failure.
DEFAULT_FAILURE_THRESHOLD = 3


# Consecutive successes required to step the fallback ladder down one rung. More than one so a
# service alternating failure and success cannot hold the ladder at its shortest rest forever.
DEFAULT_SUCCESSES_TO_STEP_DOWN = 3

# Longest a single backoff request is honored, so a bad value cannot stall a read.
MAX_HONORED_RETRY_AFTER_SECONDS = 30.0

_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})

# An INTERNAL error is the one response status that invalidates the transport rather than the
# query: the design doc gives it "No" in the keep-the-connection column and says to close the
# WebSocket, serve over HTTP, and re-dial fresh after 30-60s. Everything else is either the
# caller's problem (400, 404) or a wait-and-resend (429/502/503/504).
_ABANDON_TRANSPORT_STATUS = 500

# Retried once on the same connection even when the server sent no retry_after_seconds. With the
# field present the backoff path above handles them; without it the design doc still mandates one
# retry, and skipping it silently returned a failure the caller could have been spared.
_RETRY_ONCE_STATUSES = frozenset({502, 504})

# The query API's relative path. This transport is shared with the ingest path, so a request has to
# be identified as a read before a WebSocket URL is derived for it. Kept in step with the HTTP
# transport's own path by test_query_api_path_matches_http_transport.
_QUERY_API_REL_PATH = "api/v1/query"

# Lets _attempt preserve its caller's return type rather than widening to Any.
_Result = TypeVar("_Result")

# Opt-in, mirroring how the service itself gates the transport, so enabling or disabling it
# needs no client release. The ``SNOWFLAKE_ML_`` prefix is the package's namespace for user-set
# behaviour flags (as in SNOWFLAKE_ML_DISABLE_LOADED_MODEL_TELEMETRY); SNOWFLAKE_ is reserved for
# session and connection settings, and SNOWML_ for internals and test knobs.
WS_ENABLED_ENV_VAR = "SNOWFLAKE_ML_ONLINE_WS_ENABLED"
WS_PIPELINE_DEPTH_ENV_VAR = "SNOWFLAKE_ML_ONLINE_WS_PIPELINE_DEPTH"
WS_POOL_SIZE_ENV_VAR = "SNOWFLAKE_ML_ONLINE_WS_POOL_SIZE"
WS_MAX_CONN_AGE_ENV_VAR = "SNOWFLAKE_ML_ONLINE_WS_MAX_CONN_AGE_SECONDS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def ws_enabled() -> bool:
    """Whether the WebSocket transport is switched on for this process.

    Returns:
        True only when the environment variable is set to a recognized truthy value. Absent
        or unrecognized means the HTTP transport, unchanged.
    """
    return os.environ.get(WS_ENABLED_ENV_VAR, "").strip().lower() in _TRUTHY


def websocket_client_installed() -> bool:
    """Whether the ``websocket-client`` package the transport is built on can be imported.

    Checked when the transport would be built, not left to the first dial: the client imports
    it lazily, so a missing package surfaces as an ``ImportError`` on every request, which the
    policy here reads as a dead socket -- the flag then never works and nothing says why.

    Returns:
        True when the package is importable.
    """
    return importlib_util.find_spec("websocket") is not None


def pool_size() -> int:
    """Most connections to hold per Online Service endpoint.

    Connection slots are a shared, per-instance resource on the service and nothing there
    enforces a per-caller share, so the default is deliberately small, and the pool opens
    connections only as concurrency needs them. The sizing rule is
    ``required_QPS / per_connection_QPS`` — past a caller's own throughput need, extra
    connections buy nothing and consume a slot another caller needs. Prefer raising pipeline
    depth over connection count: depth multiplies in-flight requests on a slot already held.

    Returns:
        A pool size of at least 1; the module default when unset or unparsable.
    """
    raw = os.environ.get(WS_POOL_SIZE_ENV_VAR, "").strip()
    if not raw:
        return online_service_ws_pool.DEFAULT_POOL_SIZE
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: expected a positive integer, using the default of %d",
            WS_POOL_SIZE_ENV_VAR,
            raw,
            online_service_ws_pool.DEFAULT_POOL_SIZE,
        )
        return online_service_ws_pool.DEFAULT_POOL_SIZE


def max_conn_age_seconds() -> float:
    """Ceiling on the age at which the pool retires a connection, in seconds.

    The client must close first: the side that closes gets TIME_WAIT, and a client-initiated
    close never interrupts a request in flight. The service advertises each connection's own
    lifetime on the upgrade and the pool retires a small margin under it, so this value only
    binds when it is lower, or when the service advertises nothing. The default of 12 minutes
    sits under the 15-minute cap the service had before it advertised its age.

    It stays tunable because a deployment that does not advertise its age can still run a short
    cap, and then this is the only way to keep the client closing first.

    Returns:
        A positive age in seconds; the pool default when unset or unparsable.
    """
    raw = os.environ.get(WS_MAX_CONN_AGE_ENV_VAR, "").strip()
    if not raw:
        return online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: expected a number of seconds, using the default of %.0f",
            WS_MAX_CONN_AGE_ENV_VAR,
            raw,
            online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS,
        )
        return online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS
    if parsed <= 0:
        logger.warning(
            "Ignoring %s=%r: an age of zero or less would retire every connection immediately, "
            "using the default of %.0f",
            WS_MAX_CONN_AGE_ENV_VAR,
            raw,
            online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS,
        )
        return online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS
    return parsed


def pipeline_depth() -> int:
    """Requests allowed in flight per connection.

    Defaults to 1, which keeps the transport strictly sequential: one request outstanding at
    a time means responses cannot arrive out of order, so the correlation risk that pipelining
    introduces stays inert until someone opts into it.

    Returns:
        A depth of at least 1.
    """
    raw = os.environ.get(WS_PIPELINE_DEPTH_ENV_VAR, "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: expected a positive integer, using sequential requests",
            WS_PIPELINE_DEPTH_ENV_VAR,
            raw,
        )
        return 1


class OnlineServiceWsTransport:
    """Runs feature retrieval over WebSockets, falling back to HTTP when it cannot.

    Satisfies :class:`online_service_transport.BatchingOnlineServiceTransport`, so the query
    path cannot tell which transport answered.
    """

    def __init__(
        self,
        http_transport: online_service_transport.OnlineServiceTransport,
        *,
        session: Session | None = None,
        pool: online_service_ws_pool.OnlineServiceWsPool | None = None,
        pool_size: int = online_service_ws_pool.DEFAULT_POOL_SIZE,
        pipeline_depth: int = 1,
        redial_jitter_seconds: float = DEFAULT_REDIAL_JITTER_SECONDS,
        fallback_ladder_seconds: tuple[float, ...] = DEFAULT_FALLBACK_LADDER_SECONDS,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        successes_to_step_down: int = DEFAULT_SUCCESSES_TO_STEP_DOWN,
        _rng: random.Random | None = None,
        _clock: Callable[[], float] | None = None,
        _sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Wrap an HTTP transport with a WebSocket path and the policy governing it.

        Args:
            http_transport: The transport to fall back to. Always required — the WebSocket
                path is an optimization, never the only way to read.
            session: Snowpark session used only for auth resolution.
            pool: Connection pool. One is created when omitted.
            pool_size: Most connections per endpoint, used only when ``pool`` is omitted.
            pipeline_depth: Requests allowed in flight per connection. 1 keeps the transport
                strictly sequential, which means responses cannot arrive out of order.
            redial_jitter_seconds: Upper bound on the random pre-redial wait.
            fallback_ladder_seconds: How long to stay on HTTP after each successive failure
                before probing the WebSocket transport again.
            failure_threshold: Consecutive hard failures before falling back.
            successes_to_step_down: Consecutive successes required to shorten the fallback rest
                by one rung.
            _rng: Test seam for deterministic jitter.
            _clock: Test seam returning monotonic seconds.
            _sleep: Test seam for waits.
        """
        self._http = http_transport
        self._session = session
        self._pool = (
            pool
            if pool is not None
            else online_service_ws_pool.OnlineServiceWsPool(
                session=session, size=pool_size, max_connection_age_seconds=max_conn_age_seconds()
            )
        )
        self._pipeline_depth = max(1, pipeline_depth)
        self._redial_jitter_seconds = max(0.0, redial_jitter_seconds)
        self._fallback_ladder_seconds = fallback_ladder_seconds or (60.0,)
        self._failure_threshold = max(1, failure_threshold)
        self._successes_to_step_down = max(1, successes_to_step_down)
        self._rng = _rng or random.Random()
        self._clock = _clock or time.monotonic
        self._sleep = _sleep or time.sleep

        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._fallback_rung = -1
        self._fallback_until = 0.0
        self._closed = False
        # Guards the ladder state above. The transport is shared by every thread reading through
        # one FeatureStore, and each update is a read-then-write: two unguarded step-downs can
        # carry the rung to -2, which the next failure indexes as [-1] -- the longest rest.
        self._policy_lock = threading.Lock()

        # Observable counters. Telemetry hooks read these rather than parsing logs.
        self.ws_requests = 0
        self.http_fallback_requests = 0
        # Requests that were never WebSocket candidates (ingest has no WebSocket endpoint), kept
        # apart so ``http_fallback_requests`` counts only reads that fell back.
        self.ingest_requests = 0
        self.redials = 0
        self.honored_backoffs = 0

    # ----- capability ------------------------------------------------------

    @property
    def pipeline_depth(self) -> int:
        """Requests allowed in flight per connection."""
        return self._pipeline_depth

    @property
    def is_pipelining(self) -> bool:
        """True when batches may be sent before responses are read."""
        return self._pipeline_depth > 1

    @property
    def using_fallback(self) -> bool:
        """True while the WebSocket transport is being rested and HTTP is serving."""
        return self._clock() < self._fallback_until

    @property
    def send_ahead_window(self) -> int:
        """Bodies worth sending before the first response is checked; see ``send_ahead_window``.

        Returns:
            The pipeline depth while this transport is pipelining over WebSocket, and 1 otherwise:
            at depth 1, or while resting on HTTP, every body is its own round trip, so sending
            ahead only delays the caller's check of the first response.
        """
        if not self.is_pipelining or self.using_fallback:
            return 1
        return self._pipeline_depth

    # ----- transport surface -----------------------------------------------

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Send one body over WebSockets when available, else over HTTP.

        Args:
            url: The HTTP query endpoint URL. The WebSocket URL is derived from it.
            body: JSON-serializable payload.
            timeout: Seconds for the whole request: every WebSocket attempt, any wait between
                them, and the HTTP fallback all spend this one budget.

        Returns:
            ``(status_code, body_bytes)``.

        Raises:
            RuntimeError: When the transport has been closed.
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsTransport is closed")
        return self._post_one(url, body, self._clock() + timeout)

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Send several bodies, pipelining them when the configured depth allows.

        At depth 1 this is a plain loop, so it behaves exactly as repeated
        :meth:`post_json` calls and responses cannot arrive out of order.

        Args:
            url: The HTTP query endpoint URL.
            bodies: JSON-serializable payloads, in the order results are wanted.
            timeout: Timeout in seconds for the whole batch, fallback included.

        Returns:
            One ``(status_code, body_bytes)`` per entry in ``bodies``, in the same order.

        Raises:
            RuntimeError: When the transport has been closed.
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsTransport is closed")
        if not bodies:
            return []
        deadline = self._clock() + timeout
        if not self.is_pipelining or _query_base_of(url) is None:
            return [self._post_one(url, body, deadline) for body in bodies]
        if self.using_fallback:
            return [self._via_http(url, body, deadline) for body in bodies]

        results: list[tuple[int, bytes]] = []
        for chunk in _chunks(bodies, self._pipeline_depth):
            # Checked per chunk, not once: a dial refusal on the first chunk rests the transport,
            # and the remaining chunks must not each dial (and each authenticate) into it again.
            if self.using_fallback:
                results.extend(self._via_http(url, body, deadline) for body in chunk)
                continue
            try:
                answered = self._attempt(self._batch_sender(chunk), url, deadline)
            except _WsUnavailable:
                results.extend(self._via_http(url, body, deadline) for body in chunk)
                continue
            self.ws_requests += len(chunk)
            results.extend(answered)
        return results

    def close(self) -> None:
        """Close the pool and the fallback transport. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._pool.close()
        self._http.close()

    def _post_one(self, url: str, body: dict[str, Any], deadline: float) -> tuple[int, bytes]:
        """Send one body against a deadline already started by the caller.

        Args:
            url: The HTTP query endpoint URL.
            body: JSON-serializable payload.
            deadline: Absolute deadline on this transport's clock.

        Returns:
            ``(status_code, body_bytes)``.
        """
        if _query_base_of(url) is None:
            # Not a read, so there is no WebSocket endpoint to fall back from: this is HTTP by
            # design, and counting it as a fallback would overstate how often WebSocket fails.
            self.ingest_requests += 1
            return self._via_http(url, body, deadline, fallback=False)
        if self.using_fallback:
            return self._via_http(url, body, deadline)
        try:
            result = self._attempt(lambda ws_url, budget: self._pool.post_json(ws_url, body, budget), url, deadline)
        except _WsUnavailable:
            return self._via_http(url, body, deadline)
        self.ws_requests += 1
        return result

    def _batch_sender(self, batch: list[dict[str, Any]]) -> Callable[[str, float], list[tuple[int, bytes]]]:
        """Bind one batch to a sender that resends only what needs it.

        The first call sends the whole batch. A later call -- the policy's one retry after a
        backoff or a 502/504 -- sends only the bodies whose answer was a retryable status and
        splices the new answers in, so a chunk with one 429 does not resend its 200s to a service
        that is shedding load. If the first call raised, nothing was answered and all is resent.

        Args:
            batch: The payloads this sender will submit.

        Returns:
            A callable taking the WebSocket URL and the seconds left, returning that batch's results.
        """
        answered: list[tuple[int, bytes]] = []

        def send(ws_url: str, budget: float) -> list[tuple[int, bytes]]:
            if not answered:
                answered.extend(self._pool.post_json_many(ws_url, batch, budget))
                return list(answered)
            redo = [index for index, (status, _) in enumerate(answered) if status in _RETRYABLE_STATUSES]
            if redo:
                again = self._pool.post_json_many(ws_url, [batch[index] for index in redo], budget)
                for index, result in zip(redo, again, strict=True):
                    answered[index] = result
            return list(answered)

        return send

    # ----- policy ----------------------------------------------------------

    def _attempt(self, call: Callable[[str, float], _Result], url: str, deadline: float) -> _Result:
        """Run one WebSocket call, applying retry, backoff and fallback policy.

        Every send gets only what is left of ``deadline``, so a retry cannot reset the caller's
        clock and the HTTP fallback inherits the remainder rather than a fresh timeout.

        Args:
            call: Receives the WebSocket URL and the seconds left, and performs the request.
            url: The HTTP query endpoint URL.
            deadline: Absolute deadline on this transport's clock.

        Returns:
            Whatever ``call`` returned.

        Raises:
            _WsUnavailable: When the WebSocket path should not serve this request.
        """
        query_base = _query_base_of(url)
        if query_base is None:
            # Not the query API, so there is no WebSocket endpoint for it. Declining here routes the
            # request to REST without counting against the fallback state, which matters because
            # this transport also serves ingest.
            logger.debug("Online Service request is not the query API; using REST for it")
            raise _WsUnavailable
        try:
            ws_url = online_service_ws_client.ws_query_url(query_base)
        except Exception:  # noqa: BLE001 - a malformed endpoint is not worth failing a read over
            logger.debug("Online feature retrieval could not derive a WebSocket URL", exc_info=True)
            raise _WsUnavailable from None

        for attempt in range(2):  # one retry: reads are idempotent, so a single retry is safe
            if attempt > 0 and self.using_fallback:
                # Another caller rested the transport while this one was redialing; the retry
                # would only dial into the rest.
                raise _WsUnavailable
            budget = deadline - self._clock()
            if budget <= 0:
                # A redial wait or an honored backoff spent what was left. Not a failure of the
                # transport; the fallback raises the timeout the caller is owed.
                raise _WsUnavailable
            try:
                result = call(ws_url, budget)
            except online_service_ws_client.OnlineServiceWsDialError as e:
                self._handle_dial_error(e)
                raise _WsUnavailable from None
            except Exception as e:  # noqa: BLE001 - transport failures arrive in several shapes
                timed_out = _timeout_of(e)
                if timed_out is not None:
                    # The deadline is spent, so a retry could only run on time it does not have.
                    # Contention is not evidence against the path and is not counted: the pool
                    # raises PoolWaitTimeout both for a wait that outlived the deadline and for a
                    # wire timeout on a request that had already spent most of its budget queueing,
                    # which is the common shape once the pool is saturated. It is the only layer
                    # that knows how the deadline was divided. A timeout that is not contention is
                    # counted, so a path that keeps stalling still ends up resting on HTTP.
                    if not isinstance(timed_out, online_service_ws_pool.PoolWaitTimeout):
                        self._record_failure()
                    logger.debug("Online feature retrieval over WebSocket timed out: %s", e)
                    raise _WsUnavailable from None
                if attempt == 0:
                    # A dead socket is not diagnosed: redial once and let the redial be the signal.
                    # If the service is really gone the redial raises a dial error, classified
                    # above. A second failure on this request falls through and is counted.
                    self._redial(deadline)
                    continue
                self._record_failure()
                logger.debug("Online feature retrieval over WebSocket failed: %s", e)
                raise _WsUnavailable from None

            backoff = _retry_after_of(result)
            if backoff > 0 and attempt == 0:
                # The service asked for a delay. Honor it, then resend the same request on
                # the same connection rather than treating this as a failure.
                if self._honor_backoff(backoff, deadline):
                    continue
                # The delay would outlast the caller's deadline, so waiting could only end in a
                # timeout. The throttled answer is the truthful one; hand it back now. Not a
                # success: the ladder must not step down on a service that is shedding load.
                return result

            statuses = _statuses_of(result)
            if _ABANDON_TRANSPORT_STATUS in statuses:
                # Close the connections and rest the transport, rather than handing the caller an
                # error over a WebSocket that just reported it cannot serve.
                self._pool.discard_all()
                self._rest_transport()
                logger.info("Online feature retrieval: server reported an internal error, using HTTP")
                raise _WsUnavailable from None
            if attempt == 0 and statuses & _RETRY_ONCE_STATUSES:
                continue

            # A retryable status that came back again is the service still shedding load: not a
            # failure of the transport, but not a success the ladder may step down on either.
            if not statuses & _RETRYABLE_STATUSES:
                self._record_success()
            return result

        self._record_failure()
        raise _WsUnavailable from None

    def _redial(self, deadline: float) -> None:
        """Wait a random interval, then let the pool reopen on next use.

        The wait is what keeps a pool that closed together from redialing together. It never
        runs past ``deadline``: the jitter is there to spread handshakes, not to spend the
        caller's time.

        Args:
            deadline: Absolute deadline on this transport's clock.
        """
        self.redials += 1
        if self._redial_jitter_seconds > 0:
            self._sleep(min(self.next_redial_delay(), max(0.0, deadline - self._clock())))

    def next_redial_delay(self) -> float:
        """Draw one jittered redial delay.

        Returns:
            Seconds to wait, uniformly drawn from the jitter window.
        """
        return self._rng.uniform(0.0, self._redial_jitter_seconds)

    def _honor_backoff(self, seconds: float, deadline: float) -> bool:
        """Wait out a service-requested delay, if the caller has the time for it.

        Args:
            seconds: Requested delay.
            deadline: Absolute deadline on this transport's clock.

        Returns:
            True when the delay was waited out and a resend can follow; False when it would
            have spent the rest of the budget, in which case nothing was waited.
        """
        bounded = min(seconds, MAX_HONORED_RETRY_AFTER_SECONDS)
        if bounded >= deadline - self._clock():
            return False
        self.honored_backoffs += 1
        logger.info("Online feature retrieval backing off for %.1fs as requested", bounded)
        self._sleep(bounded)
        return True

    def _handle_dial_error(self, error: online_service_ws_client.OnlineServiceWsDialError) -> None:
        """Classify a refused handshake and set fallback accordingly.

        Args:
            error: The dial failure, carrying the handshake status.
        """
        if error.transport_unavailable:
            # Not served here at all. Rest for the longest interval rather than probing often.
            logger.info("Online feature retrieval: WebSocket transport not available, using HTTP")
            self._rest_on_top_rung()
            return
        if error.auth_rejected:
            logger.info("Online feature retrieval: WebSocket credentials rejected, using HTTP")
            self._rest_on_top_rung()
            return
        # At capacity, or an unclassified refusal. When the service said how long to wait, rest
        # the WebSocket path for that long rather than sleeping it out: the request that hit this
        # is served over HTTP straight after either way, so a sleep only delays it -- once per
        # chunk of a pipelined read -- without making the retry any likelier to land.
        if error.retry_after_seconds > 0:
            rest = min(float(error.retry_after_seconds), MAX_HONORED_RETRY_AFTER_SECONDS)
            self.honored_backoffs += 1
            logger.info("Online feature retrieval: WebSocket at capacity, using HTTP for %.1fs as asked", rest)
            with self._policy_lock:
                self._fallback_until = max(self._fallback_until, self._clock() + rest)
        self._record_failure()

    def _rest_on_top_rung(self) -> None:
        """Rest for the longest interval: the refusal will not clear on its own soon."""
        with self._policy_lock:
            self._fallback_rung = len(self._fallback_ladder_seconds) - 1
            self._fallback_until = self._clock() + self._fallback_ladder_seconds[self._fallback_rung]

    def _rest_transport(self) -> None:
        """Enter the fallback rest now, without waiting for a failure streak.

        The streak in :meth:`_record_failure` exists so that ordinary disconnects do not drive a
        fallback. An INTERNAL error is not an ordinary disconnect — the design doc calls for
        resting the transport on the first one — so this advances the ladder directly. Repeats
        still lengthen the rest, because the rung advances each time -- but only once the current
        rest has ended. Callers already in flight when the first INTERNAL error lands each report
        one too; without this they each advanced a rung, so one bad moment cost the ceiling's 600 s
        rather than the first rung's 60 s.
        """
        with self._policy_lock:
            if self._clock() < self._fallback_until:
                return
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._fallback_rung = min(self._fallback_rung + 1, len(self._fallback_ladder_seconds) - 1)
            rest = self._fallback_ladder_seconds[self._fallback_rung]
            self._fallback_until = self._clock() + rest
        logger.info("Online feature retrieval resting the WebSocket transport for %.0fs", rest)

    def _record_failure(self) -> None:
        """Count a hard failure and advance the fallback ladder once the threshold is met."""
        with self._policy_lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if self._consecutive_failures < self._failure_threshold:
                return
            self._consecutive_failures = 0
            self._fallback_rung = min(self._fallback_rung + 1, len(self._fallback_ladder_seconds) - 1)
            rest = self._fallback_ladder_seconds[self._fallback_rung]
            self._fallback_until = self._clock() + rest
        logger.info("Online feature retrieval falling back to HTTP for %.0fs", rest)

    def _record_success(self) -> None:
        """Clear the failure streak, and step the ladder down after a run of successes.

        A run rather than a single success: stepping down on every success lets a service that
        alternates one failure with one success pin the ladder at its lowest rung forever, so the
        longer rests are never reached no matter how long the flapping continues.
        """
        with self._policy_lock:
            self._consecutive_failures = 0
            if self._fallback_rung < 0:
                return
            self._consecutive_successes += 1
            if self._consecutive_successes >= self._successes_to_step_down:
                self._consecutive_successes = 0
                # Clamped as well as locked: -1 is "no rung", and below it [-1] indexes the top.
                self._fallback_rung = max(-1, self._fallback_rung - 1)

    def _via_http(self, url: str, body: dict[str, Any], deadline: float, *, fallback: bool = True) -> tuple[int, bytes]:
        """Serve one request over HTTP with whatever is left of its deadline.

        Args:
            url: The HTTP query endpoint URL.
            body: JSON-serializable payload.
            deadline: Absolute deadline on this transport's clock.
            fallback: Whether this is a read falling back, as opposed to a request (an ingest)
                that never had a WebSocket path; only the former is counted as a fallback.

        Returns:
            ``(status_code, body_bytes)``.

        Raises:
            SnowflakeMLException: When the WebSocket path already spent the whole budget; the
                same timeout the client raises, rather than an HTTP request with no time to run.

        # noqa: DAR401
        # noqa: DAR402
        """
        budget = deadline - self._clock()
        if budget <= 0:
            raise online_service_ws_client._deadline_exceeded("before it could fall back to HTTP")  # noqa: SLF001
        if fallback:
            self.http_fallback_requests += 1
        return self._http.post_json(url, body, budget)


class _WsUnavailable(Exception):
    """Internal signal that this request should be served over HTTP instead."""


def _chunks(bodies: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split ``bodies`` into runs of at most ``size``.

    Args:
        bodies: Request payloads.
        size: Maximum run length.

    Returns:
        The runs, in order.
    """
    return [bodies[start : start + size] for start in range(0, len(bodies), size)]


def _timeout_of(error: BaseException) -> TimeoutError | None:
    """The timeout behind a failed WebSocket call, if a timeout is what it was.

    The client and the pool both report a spent deadline as a ``SnowflakeMLException`` wrapping
    a ``TimeoutError`` (the pool's own :class:`~online_service_ws_pool.PoolWaitTimeout` for a
    wait in its queue); a bare ``TimeoutError`` is accepted too.

    Args:
        error: What the call raised.

    Returns:
        The ``TimeoutError``, or None when ``error`` is some other failure.
    """
    if isinstance(error, TimeoutError):
        return error
    wrapped = getattr(error, "original_exception", None)
    return wrapped if isinstance(wrapped, TimeoutError) else None


def _query_base_of(url: str) -> str | None:
    """Strip the query API path off a URL so the WebSocket URL can be derived from it.

    Returns ``None`` for any URL that is not a query endpoint. This transport is shared with the
    ingest path, which has no WebSocket endpoint to reach, so a permissive derivation would build a
    URL the service does not serve. That dial fails as a 404, which is classified as "transport
    unavailable" and rests the transport — disabling WebSocket for *reads* because of an ingest.

    Args:
        url: A resolved Online Service API URL.

    Returns:
        The endpoint base URL, or ``None`` when ``url`` is not the query API.
    """
    trimmed = url.strip().rstrip("/")
    if trimmed.endswith(_QUERY_API_REL_PATH):
        return trimmed[: -len(_QUERY_API_REL_PATH)]
    return None


def _statuses_of(result: Any) -> frozenset[int]:
    """Collect the status codes in a result, whether it holds one pair or a batch of them.

    Args:
        result: Either one ``(status, bytes)`` pair or a list of them.

    Returns:
        The distinct status codes present.
    """
    pairs = result if isinstance(result, list) else [result]
    found = set()
    for entry in pairs:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], int):
            found.add(entry[0])
    return frozenset(found)


def _retry_after_of(result: Any) -> float:
    """Read a backoff request out of a result, if there is one.

    Args:
        result: Either one ``(status, bytes)`` pair or a list of them.

    Returns:
        The largest requested delay in seconds across retryable entries, else 0.
    """
    pairs: list[tuple[int, bytes]]
    if isinstance(result, list):
        pairs = result
    elif isinstance(result, tuple):
        pairs = [result]
    else:
        return 0.0

    longest = 0.0
    for entry in pairs:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            continue
        status, payload = entry
        if status not in _RETRYABLE_STATUSES:
            continue
        try:
            parsed = json.loads(bytes(payload).decode("utf-8", "replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        requested = parsed.get("retry_after_seconds")
        if isinstance(requested, (int, float)) and requested > longest:
            longest = float(requested)
    return longest
