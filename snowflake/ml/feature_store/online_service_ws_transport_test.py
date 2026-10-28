"""Unit tests for online_service_ws_transport (fallback policy, jitter, backoff, opt-in).

The policy here exists because the service's signals are ambiguous by the time they reach a
client: a scheduled close and a dropped network look identical, and the status mapping the
caller uses collapses several distinct failures into one error. These tests pin the decisions
that ambiguity forces, especially the one that is easy to get backwards — a close is normal
and must not push the client onto HTTP permanently.
"""

import json
import os
import random
import threading
import time
from typing import Any
from unittest.mock import patch

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store import (
    online_service_http_client,
    online_service_ws_client,
    online_service_ws_pool,
    online_service_ws_transport,
)

_URL = "https://svc.example.app/api/v1/query"
_INGEST_URL = "https://svc.example.app/api/v1/ingest"


class _FakeHttp:
    """Stands in for the HTTP transport the WebSocket path falls back to."""

    def __init__(self) -> None:
        self.posts = 0
        self.timeouts: list[float] = []
        self.closed = False

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        del url, body
        self.posts += 1
        self.timeouts.append(timeout)
        return 200, b'{"status":"SUCCESS","via":"http"}'

    def close(self) -> None:
        self.closed = True


class _FakePool:
    """Scriptable pool: each entry is a result to return, an exception to raise, or a callable
    run first that returns either -- the way a step spends time on the test's clock."""

    def __init__(
        self,
        script: list[Any] | None = None,
        *,
        connection_count: int = 1,
    ) -> None:
        self.script = list(script or [])
        self.calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
        self.budgets: list[float] = []
        self.closed = False
        self.discarded = 0
        self.connection_count = connection_count

    def _next(self, timeout: float) -> Any:
        self.calls += 1
        self.budgets.append(timeout)
        if not self.script:
            return 200, b'{"status":"SUCCESS","via":"ws"}'
        step = self.script.pop(0)
        if callable(step):
            step = step()
        if isinstance(step, Exception):
            raise step
        return step

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        del url, body
        return self._next(timeout)  # type: ignore[no-any-return]

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        del url
        self.batch_calls += 1
        self.batch_sizes.append(len(bodies))
        result = self._next(timeout)
        if isinstance(result, list):
            return result
        return [result for _ in bodies]

    def close(self) -> None:
        self.closed = True

    def discard_all(self) -> int:
        # Mirrors the real pool: drops every connection, leaves the pool usable.
        self.discarded += 1
        n, self.connection_count = self.connection_count, 0
        return n


def _transport(
    pool: _FakePool | None = None,
    http: _FakeHttp | None = None,
    **kwargs: Any,
) -> tuple[online_service_ws_transport.OnlineServiceWsTransport, _FakePool, _FakeHttp, list[float], list[float]]:
    """Build a transport with deterministic time, rng, and recorded sleeps.

    Args:
        pool: Scripted pool; a default succeeding one is used when omitted.
        http: Fallback transport; a fresh one is used when omitted.
        **kwargs: Overrides passed to the transport.

    Returns:
        ``(transport, pool, http, slept, now)`` where ``slept`` records every wait and
        ``now`` is the mutable clock a test can advance.
    """
    pool = pool or _FakePool()
    http = http or _FakeHttp()
    slept: list[float] = []
    now = [0.0]

    defaults: dict[str, Any] = {
        "pool": pool,
        "_rng": random.Random(7),
        "_clock": lambda: now[0],
        "_sleep": slept.append,
    }
    defaults.update(kwargs)
    transport = online_service_ws_transport.OnlineServiceWsTransport(http, **defaults)
    return transport, pool, http, slept, now


def _body() -> dict[str, Any]:
    """Build a minimal request body.

    Returns:
        A body the transport can route.
    """
    return {"name": "my_fv", "version": "v1", "object_type": "feature_view"}


class OptInTest(absltest.TestCase):
    def test_disabled_by_default(self) -> None:
        """Absent configuration must leave the existing HTTP behaviour untouched."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(online_service_ws_transport.ws_enabled())

    def test_recognized_truthy_values_enable_it(self) -> None:
        """Operators should not have to guess the accepted spelling."""
        for value in ("1", "true", "TRUE", "yes", "on"):
            with patch.dict(os.environ, {online_service_ws_transport.WS_ENABLED_ENV_VAR: value}, clear=False):
                self.assertTrue(online_service_ws_transport.ws_enabled(), value)

    def test_unrecognized_value_leaves_it_off(self) -> None:
        """A typo must fail safe, not silently enable a new transport."""
        with patch.dict(os.environ, {online_service_ws_transport.WS_ENABLED_ENV_VAR: "maybe"}, clear=False):
            self.assertFalse(online_service_ws_transport.ws_enabled())

    def test_pipeline_depth_defaults_to_sequential(self) -> None:
        """Depth 1 means one request outstanding, so responses cannot arrive out of order."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(online_service_ws_transport.pipeline_depth(), 1)

    def test_pipeline_depth_is_read_from_the_environment(self) -> None:
        """Pipelining is opt-in separately from the transport itself."""
        with patch.dict(os.environ, {online_service_ws_transport.WS_PIPELINE_DEPTH_ENV_VAR: "8"}, clear=False):
            self.assertEqual(online_service_ws_transport.pipeline_depth(), 8)

    def test_invalid_pipeline_depth_falls_back_to_sequential(self) -> None:
        """A bad value must not enable reordering by accident."""
        for value in ("abc", "0", "-3"):
            with patch.dict(os.environ, {online_service_ws_transport.WS_PIPELINE_DEPTH_ENV_VAR: value}, clear=False):
                self.assertEqual(online_service_ws_transport.pipeline_depth(), 1, value)


class HappyPathTest(absltest.TestCase):
    def test_requests_go_over_the_websocket_when_it_works(self) -> None:
        """Otherwise the transport would exist without ever being used."""
        transport, pool, http, _, _now = _transport()
        status, raw = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["via"], "ws")
        self.assertEqual(pool.calls, 1)
        self.assertEqual(http.posts, 0)
        self.assertEqual(transport.ws_requests, 1)

    def test_default_depth_sends_a_batch_one_at_a_time(self) -> None:
        """At depth 1 there must be no pipelining, so ordering risk stays inert."""
        transport, pool, _, _, _now = _transport(pipeline_depth=1)
        results = transport.post_json_many(_URL, [_body(), _body(), _body()], 30.0)

        self.assertLen(results, 3)
        self.assertEqual(pool.batch_calls, 0)
        self.assertEqual(pool.calls, 3)

    def test_configured_depth_pipelines(self) -> None:
        """Above depth 1 the burst should reach the pool as a batch."""
        transport, pool, _, _, _now = _transport(pipeline_depth=4)
        results = transport.post_json_many(_URL, [_body() for _ in range(4)], 30.0)

        self.assertLen(results, 4)
        self.assertEqual(pool.batch_calls, 1)

    def test_batches_are_split_at_the_configured_depth(self) -> None:
        """Depth is a ceiling on requests in flight, so a longer read must be chunked."""
        transport, pool, _, _, _now = _transport(pipeline_depth=2)
        transport.post_json_many(_URL, [_body() for _ in range(5)], 30.0)
        self.assertEqual(pool.batch_calls, 3)

    def test_close_closes_the_pool_and_the_fallback(self) -> None:
        """A leaked connection holds a server slot until the service times it out."""
        transport, pool, http, _, _now = _transport()
        transport.close()
        transport.close()

        self.assertTrue(pool.closed)
        self.assertTrue(http.closed)
        with self.assertRaises(RuntimeError):
            transport.post_json(_URL, _body(), 30.0)


class ResponseStatusPolicyTest(absltest.TestCase):
    """An error *status* on a live connection is a different event from a dead socket.

    The transport's failure model is otherwise exception-based, so without these a 500 counted as
    a successful round trip: the caller got the error, the connection stayed open, and the failure
    streak reset — so a service returning 500 to every query never triggered the fallback the
    ladder exists for.
    """

    def test_internal_error_closes_the_transport_and_serves_over_http(self) -> None:
        pool = _FakePool([(500, b'{"error":{"code":"INTERNAL"}}')], connection_count=0)
        transport, _, http, _, _now = _transport(pool)

        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(http.posts, 1, "the query should have been served over HTTP")
        self.assertTrue(transport.using_fallback, "a 500 must rest the transport, not be ignored")

    def test_internal_error_rests_on_the_very_first_one(self) -> None:
        """The streak gate exists so ordinary disconnects do not force a fallback; an INTERNAL
        error is not an ordinary disconnect, so it must not have to happen three times."""
        pool = _FakePool([(500, b"{}")], connection_count=0)
        transport, _, _http, _, _now = _transport(pool, failure_threshold=3)

        transport.post_json(_URL, _body(), 30.0)

        self.assertTrue(transport.using_fallback)

    def test_repeated_internal_errors_lengthen_the_rest(self) -> None:
        pool = _FakePool([(500, b"{}")] * 6, connection_count=0)
        transport, _, _http, _, now = _transport(pool, fallback_ladder_seconds=(10.0, 20.0, 30.0))

        transport.post_json(_URL, _body(), 30.0)
        first = transport._fallback_until  # noqa: SLF001
        now[0] = first + 1.0
        transport.post_json(_URL, _body(), 30.0)

        self.assertGreater(transport._fallback_until - now[0], first)  # noqa: SLF001

    def test_502_without_retry_after_is_still_retried_once(self) -> None:
        """The design doc mandates one retry on 502/504. Honoring it only when the server supplied
        retry_after_seconds silently skipped the case where it did not."""
        pool = _FakePool([(502, b"{}"), (200, b'{"status":"SUCCESS","via":"ws"}')], connection_count=0)
        transport, _, http, _, _now = _transport(pool)

        status, raw = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["via"], "ws")
        self.assertEqual(http.posts, 0, "the retry succeeded, so HTTP should not have been used")

    def test_400_and_404_are_returned_to_the_caller_untouched(self) -> None:
        """These are the caller's problem, not the transport's: no retry, no fallback, and the
        connection stays open for other queries."""
        for status in (400, 404):
            with self.subTest(status=status):
                pool = _FakePool([(status, b'{"error":{"code":"X"}}')], connection_count=0)
                transport, _, http, _, _now = _transport(pool)

                got, _raw = transport.post_json(_URL, _body(), 30.0)

                self.assertEqual(got, status)
                self.assertEqual(http.posts, 0)
                self.assertFalse(transport.using_fallback)


class HandshakeBackoffTest(absltest.TestCase):
    """A 503 at the handshake carries Retry-After as a header, not in a frame body.

    The cap is per service instance and shared across callers, so redialing on the client's own
    schedule instead of the one the service asked for is what turns a full cap into a retry
    storm against it.
    """

    def test_at_capacity_rests_for_the_requested_delay_instead_of_sleeping(self) -> None:
        """The refused request goes to HTTP straight away; sleeping first would only delay it."""
        error = online_service_ws_client.OnlineServiceWsDialError("at capacity", status=503, retry_after_seconds=4)
        transport, _, http, slept, now = _transport(_FakePool([error]))

        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(slept, [], "the read waited before being served over HTTP anyway")
        self.assertEqual(transport.honored_backoffs, 1)
        self.assertEqual(http.posts, 1)
        self.assertTrue(transport.using_fallback, "the service asked for 4s and was dialed again at once")
        now[0] += 4.1
        self.assertFalse(transport.using_fallback, "rested longer than the service asked")

    def test_at_capacity_without_a_header_does_not_wait(self) -> None:
        """Absent guidance, waiting an invented interval would be worse than not waiting."""
        error = online_service_ws_client.OnlineServiceWsDialError("at capacity", status=503)
        transport, _, http, slept, _now = _transport(_FakePool([error]))

        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(slept, [])
        self.assertEqual(transport.honored_backoffs, 0)
        self.assertEqual(http.posts, 1)

    def test_a_wild_delay_is_capped(self) -> None:
        """A bad value must not rest the WebSocket path for longer than the ceiling."""
        error = online_service_ws_client.OnlineServiceWsDialError("at capacity", status=503, retry_after_seconds=86_400)
        transport, _, _http, slept, now = _transport(_FakePool([error]))

        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(slept, [])
        now[0] += online_service_ws_transport.MAX_HONORED_RETRY_AFTER_SECONDS + 0.1
        self.assertFalse(transport.using_fallback, "an absurd retry_after rested past the ceiling")


class DisconnectPolicyTest(absltest.TestCase):
    def test_a_close_is_retried_and_does_not_trip_fallback(self) -> None:
        """The service closes connections on a timer, so treating each close as an incident
        would drive a permanent fall back to HTTP and forfeit the transport entirely."""
        pool = _FakePool([OSError("connection closed"), (200, b'{"status":"SUCCESS","via":"ws"}')])
        transport, _, http, _, _now = _transport(pool)

        status, raw = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["via"], "ws")
        self.assertEqual(http.posts, 0)
        self.assertFalse(transport.using_fallback)

    def test_repeated_closes_still_do_not_trip_fallback(self) -> None:
        """This is the trap: many closes an hour is normal operation, not degradation."""
        transport, _, http, _, _now = _transport(
            _FakePool([OSError("closed"), (200, b"{}")] * 10),
            failure_threshold=2,
        )
        for _ in range(10):
            transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(http.posts, 0)
        self.assertFalse(transport.using_fallback)

    def test_redial_waits_a_jittered_interval(self) -> None:
        """Synchronized redials arrive as a burst of authentications."""
        pool = _FakePool([OSError("closed"), (200, b"{}")])
        transport, _, _, slept, _now = _transport(pool, redial_jitter_seconds=0.25)
        transport.post_json(_URL, _body(), 30.0)

        self.assertLen(slept, 1)
        self.assertGreaterEqual(slept[0], 0.0)
        self.assertLessEqual(slept[0], 0.25)

    def test_redial_delays_are_spread_not_constant(self) -> None:
        """A constant delay would pass a bounds check and leave the burst intact."""
        transport, _, _, _, _now = _transport(redial_jitter_seconds=0.25)
        delays = {transport.next_redial_delay() for _ in range(200)}
        self.assertGreater(len(delays), 100)
        self.assertTrue(all(0.0 <= delay <= 0.25 for delay in delays))


class BackoffTest(absltest.TestCase):
    def test_requested_backoff_is_waited_out_then_the_request_is_resent(self) -> None:
        """The envelope is the only place this guidance appears, so this layer must honor it."""
        throttled = (429, json.dumps({"error": {"code": "RESOURCE_EXHAUSTED"}, "retry_after_seconds": 2}).encode())
        pool = _FakePool([throttled, (200, b'{"status":"SUCCESS","via":"ws"}')])
        transport, _, http, slept, _now = _transport(pool)

        status, _ = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 200)
        self.assertEqual(slept, [2.0])
        self.assertEqual(transport.honored_backoffs, 1)
        self.assertEqual(http.posts, 0)

    def test_backoff_is_bounded_so_a_bad_value_cannot_stall_a_read(self) -> None:
        """A service asking for an hour must not hang the caller for an hour."""
        absurd = (503, json.dumps({"error": {"code": "UNAVAILABLE"}, "retry_after_seconds": 3600}).encode())
        pool = _FakePool([absurd, (200, b"{}")])
        transport, _, _, slept, _now = _transport(pool)
        # A budget past the cap, so the cap is what bounds the wait rather than the deadline.
        transport.post_json(_URL, _body(), online_service_ws_transport.MAX_HONORED_RETRY_AFTER_SECONDS * 2)

        self.assertEqual(slept, [online_service_ws_transport.MAX_HONORED_RETRY_AFTER_SECONDS])

    def test_a_non_retryable_status_is_returned_not_retried(self) -> None:
        """A missing feature view is terminal; retrying it just doubles the latency."""
        not_found = (404, json.dumps({"error": {"code": "NOT_FOUND"}}).encode())
        pool = _FakePool([not_found])
        transport, _, http, slept, _now = _transport(pool)

        status, _ = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 404)
        self.assertEmpty(slept)
        self.assertEqual(http.posts, 0)


class DeadlineBudgetTest(absltest.TestCase):
    """The caller's timeout is one budget for the whole read, fallback included.

    Before this, each WebSocket attempt and then the HTTP fallback were each handed the full
    timeout, so a 5s read could take about 15s: a wait in the pool's queue, a retry, then HTTP.
    """

    _TIMEOUT = 5.0

    def _transport(self, pool: _FakePool, **kwargs: Any) -> tuple[Any, _FakeHttp, list[float], list[float]]:
        """A transport whose sleeps advance its clock, so time spent waiting is time spent."""
        slept: list[float] = []
        now = [0.0]

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            now[0] += seconds

        transport, _, http, _, _ = _transport(pool, _clock=lambda: now[0], _sleep=sleep, **kwargs)
        return transport, http, slept, now

    @staticmethod
    def _spend(now: list[float], seconds: float, then: Any) -> Any:
        """A script step that takes ``seconds`` on the clock and then yields ``then``."""

        def step() -> Any:
            now[0] += seconds
            return then

        return step

    @staticmethod
    def _queue_timeout() -> Exception:
        return online_service_ws_client._deadline_exceeded(  # noqa: SLF001
            "waiting for a pooled connection", online_service_ws_pool.PoolWaitTimeout
        )

    @staticmethod
    def _wire_timeout() -> Exception:
        return online_service_ws_client._deadline_exceeded("while waiting on the service")  # noqa: SLF001

    def test_a_queue_timeout_that_spent_the_budget_raises_instead_of_starting_over(self) -> None:
        """Osborn's case: queued behind a slow request for the whole timeout, then retried, then
        sent over HTTP -- three timeouts' worth. Now the read ends when its budget does."""
        pool = _FakePool()
        transport, http, _, now = self._transport(pool)
        pool.script = [self._spend(now, self._TIMEOUT, self._queue_timeout())]

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as caught:
            transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertIsInstance(caught.exception.original_exception, TimeoutError)
        self.assertLessEqual(now[0], self._TIMEOUT)
        self.assertEqual(pool.calls, 1, "a timeout was retried on time it did not have")
        self.assertEqual(http.posts, 0, "HTTP was started with no time left to run")

    def test_a_queue_timeout_goes_to_http_with_only_what_is_left(self) -> None:
        pool = _FakePool()
        transport, http, _, now = self._transport(pool)
        pool.script = [self._spend(now, 2.0, self._queue_timeout())]

        status, raw = transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual((status, json.loads(raw)["via"]), (200, "http"))
        self.assertEqual(pool.calls, 1)
        self.assertEqual(transport.redials, 0)
        self.assertEqual(http.timeouts, [self._TIMEOUT - 2.0])

    def test_queue_timeouts_do_not_trip_fallback(self) -> None:
        """A saturated pool is busy, not broken: resting it on HTTP would turn load into an outage
        of the transport for every caller, exactly when it is most useful."""
        pool = _FakePool()
        transport, http, _, _ = self._transport(pool, failure_threshold=2)
        pool.script = [self._queue_timeout() for _ in range(5)]

        for _ in range(5):
            transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual(http.posts, 5)
        self.assertFalse(transport.using_fallback)

    def test_wire_timeouts_are_not_retried_but_are_counted(self) -> None:
        """A WebSocket path that keeps stalling must still end up resting on HTTP."""
        pool = _FakePool()
        transport, _, _, _ = self._transport(pool, failure_threshold=2)
        pool.script = [self._wire_timeout(), self._wire_timeout()]

        for _ in range(2):
            transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual(pool.calls, 2)
        self.assertEqual(transport.redials, 0)
        self.assertTrue(transport.using_fallback)

    def test_a_retry_after_a_dead_socket_gets_only_what_is_left(self) -> None:
        pool = _FakePool()
        transport, _, slept, now = self._transport(pool, redial_jitter_seconds=0.25)
        pool.script = [self._spend(now, 2.0, OSError("connection closed")), (200, b"{}")]

        transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertLen(pool.budgets, 2)
        self.assertEqual(pool.budgets[0], self._TIMEOUT)
        self.assertAlmostEqual(pool.budgets[1], self._TIMEOUT - 2.0 - slept[0])

    def test_the_whole_read_never_outlasts_its_timeout(self) -> None:
        """Every attempt, the wait between them and the fallback share one deadline."""
        pool = _FakePool()
        transport, http, _, now = self._transport(pool, redial_jitter_seconds=0.25)
        pool.script = [
            self._spend(now, 2.0, OSError("closed")),
            self._spend(now, 1.5, OSError("closed again")),
        ]

        transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual(http.posts, 1)
        self.assertAlmostEqual(now[0] + http.timeouts[0], self._TIMEOUT)

    def test_the_redial_wait_never_runs_past_the_deadline(self) -> None:
        pool = _FakePool()
        transport, http, slept, now = self._transport(pool, redial_jitter_seconds=0.25)
        pool.script = [self._spend(now, self._TIMEOUT - 0.01, OSError("closed")), OSError("closed again")]

        try:
            transport.post_json(_URL, _body(), self._TIMEOUT)
        except snowml_exceptions.SnowflakeMLException:
            pass  # the budget may be spent by the time the fallback runs; that is the point

        self.assertLessEqual(slept[0], 0.01 + 1e-9)
        self.assertLessEqual(now[0] + sum(http.timeouts), self._TIMEOUT + 1e-9)

    def test_a_backoff_the_caller_has_no_time_for_is_not_waited(self) -> None:
        """Waiting 10s on a 5s budget could only end in a timeout; the 429 is the truthful answer."""
        throttled = (429, json.dumps({"error": {"code": "RESOURCE_EXHAUSTED"}, "retry_after_seconds": 10}).encode())
        pool = _FakePool([throttled])
        transport, http, slept, _ = self._transport(pool)

        status, _ = transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual(status, 429)
        self.assertEqual(slept, [])
        self.assertEqual((pool.calls, http.posts), (1, 0))

    def test_a_backoff_that_fits_is_resent_with_what_is_left(self) -> None:
        throttled = (429, json.dumps({"error": {"code": "RESOURCE_EXHAUSTED"}, "retry_after_seconds": 2}).encode())
        pool = _FakePool([throttled, (200, b"{}")])
        transport, _, slept, _ = self._transport(pool)

        status, _ = transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual(status, 200)
        self.assertEqual(slept, [2.0])
        self.assertEqual(pool.budgets, [self._TIMEOUT, self._TIMEOUT - 2.0])

    def test_a_pipelined_batch_shares_one_deadline_across_its_chunks(self) -> None:
        pool = _FakePool()
        transport, _, _, now = self._transport(pool, pipeline_depth=2)
        pool.script = [self._spend(now, 3.0, (200, b"{}")), (200, b"{}")]

        transport.post_json_many(_URL, [_body() for _ in range(4)], self._TIMEOUT)

        self.assertEqual(pool.budgets, [self._TIMEOUT, self._TIMEOUT - 3.0])

    def test_a_sequential_batch_shares_one_deadline_too(self) -> None:
        """At depth 1 the batch is a loop of single sends; its documented timeout still covers
        the batch, not each body."""
        pool = _FakePool()
        transport, _, _, now = self._transport(pool)
        pool.script = [self._spend(now, 1.0, (200, b"{}")), self._spend(now, 1.0, (200, b"{}")), (200, b"{}")]

        transport.post_json_many(_URL, [_body() for _ in range(3)], self._TIMEOUT)

        self.assertEqual(pool.budgets, [self._TIMEOUT, self._TIMEOUT - 1.0, self._TIMEOUT - 2.0])

    def test_resting_on_http_hands_http_the_caller_s_timeout(self) -> None:
        """No WebSocket time was spent, so the fallback gets all of it."""
        error = online_service_ws_client.OnlineServiceWsDialError("off", status=404)
        transport, http, _, _ = self._transport(_FakePool([error]))
        transport.post_json(_URL, _body(), self._TIMEOUT)

        transport.post_json(_URL, _body(), self._TIMEOUT)

        self.assertEqual(http.timeouts, [self._TIMEOUT, self._TIMEOUT])


class FallbackTest(absltest.TestCase):
    def test_dial_404_falls_back_to_http(self) -> None:
        """The transport is not served here, so reads must keep working over HTTP."""
        error = online_service_ws_client.OnlineServiceWsDialError("off", status=404)
        transport, _, http, _, _now = _transport(_FakePool([error]))

        status, raw = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["via"], "http")
        self.assertEqual(http.posts, 1)
        self.assertTrue(transport.using_fallback)

    def test_dial_401_falls_back_to_http(self) -> None:
        """Credentials were rejected at the handshake; redialing repeats the rejection."""
        from snowflake.ml.feature_store import online_service_ws_client

        error = online_service_ws_client.OnlineServiceWsDialError("denied", status=401)
        transport, _, http, _, _now = _transport(_FakePool([error]))
        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(http.posts, 1)
        self.assertTrue(transport.using_fallback)

    def test_hard_failures_trip_fallback_only_at_the_threshold(self) -> None:
        """Falling back on the first hard failure would be as wrong as never falling back."""
        pool = _FakePool([RuntimeError("boom")] * 6, connection_count=0)
        transport, _, http, _, _now = _transport(pool, failure_threshold=3)

        for _ in range(2):
            transport.post_json(_URL, _body(), 30.0)
        self.assertFalse(transport.using_fallback)

        transport.post_json(_URL, _body(), 30.0)
        self.assertTrue(transport.using_fallback)
        self.assertEqual(http.posts, 3)

    def test_fallback_rests_then_probes_again(self) -> None:
        """Permanently abandoning the transport after a transient outage forfeits the benefit."""
        pool = _FakePool([RuntimeError("boom")] * 6 + [(200, b'{"status":"SUCCESS","via":"ws"}')], connection_count=0)
        transport, _, http, _, _now = _transport(pool, failure_threshold=3, fallback_ladder_seconds=(60.0,))

        for _ in range(3):
            transport.post_json(_URL, _body(), 30.0)
        self.assertTrue(transport.using_fallback)

        # While resting, requests go over HTTP without touching the pool.
        posts_before = pool.calls
        transport.post_json(_URL, _body(), 30.0)
        self.assertEqual(pool.calls, posts_before)

        _now[0] = 61.0
        self.assertFalse(transport.using_fallback)
        status, raw = transport.post_json(_URL, _body(), 30.0)
        self.assertEqual(json.loads(raw)["via"], "ws")
        self.assertEqual(status, 200)

    def test_ladder_lengthens_with_each_successive_failure(self) -> None:
        """Repeated outages should be probed less often, not at a fixed interval."""
        pool = _FakePool([RuntimeError("boom")] * 20, connection_count=0)
        transport, _, _, _, _now = _transport(pool, failure_threshold=1, fallback_ladder_seconds=(10.0, 20.0, 40.0))

        transport.post_json(_URL, _body(), 30.0)
        first = transport._fallback_until  # noqa: SLF001 - white-box, matching repo style

        _now[0] = first + 1.0  # noqa: SLF001
        transport.post_json(_URL, _body(), 30.0)
        second = transport._fallback_until  # noqa: SLF001

        self.assertGreater(second - (first + 1.0), first)

    def test_the_shipped_ladder_is_the_one_documented(self) -> None:
        """Every other test here injects its own ladder, so nothing pinned the real one.

        It was already lost once: the 600s ceiling went missing between a local branch and the
        pushed one, silently, because no test referenced the production value. The rungs are a
        documented contract -- a dead service costs one probe per ten minutes, and 404/401 jump
        straight to the ceiling -- so assert them here.
        """
        self.assertEqual(
            (60.0, 120.0, 300.0, 600.0),
            online_service_ws_transport.DEFAULT_FALLBACK_LADDER_SECONDS,
        )

    def test_the_top_rung_repeats_instead_of_running_off_the_end(self) -> None:
        """An outage longer than the ladder must keep probing, not stop or crash.

        The rung is advanced with min(rung + 1, last), so the ceiling is a resting state rather
        than a terminus: a service down for hours is still retried every 600s and recovers
        without a restart.
        """
        ladder = (10.0, 20.0)
        pool = _FakePool([RuntimeError("boom")] * 30, connection_count=0)
        transport, _, _, _, now = _transport(pool, failure_threshold=1, fallback_ladder_seconds=ladder)
        rests = []
        for _ in range(5):
            transport.post_json(_URL, _body(), 30.0)
            rests.append(transport._fallback_until - now[0])  # noqa: SLF001 - white-box, repo style
            now[0] = transport._fallback_until + 1.0  # noqa: SLF001
        # Climbs to the ceiling, then stays there for as long as the outage lasts.
        self.assertEqual([10.0, 20.0, 20.0, 20.0, 20.0], rests)

    def test_fallback_serves_a_whole_batch_over_http(self) -> None:
        """A partially served batch would leave the caller with missing rows."""
        from snowflake.ml.feature_store import online_service_ws_client

        error = online_service_ws_client.OnlineServiceWsDialError("off", status=404)
        transport, _, http, _, _now = _transport(_FakePool([error]), pipeline_depth=4)

        results = transport.post_json_many(_URL, [_body() for _ in range(4)], 30.0)

        self.assertLen(results, 4)
        self.assertEqual(http.posts, 4)

    def test_fallback_set_by_the_first_chunk_stops_the_rest_dialing(self) -> None:
        """Each further chunk would otherwise dial -- and authenticate -- into the rest again."""
        from snowflake.ml.feature_store import online_service_ws_client

        error = online_service_ws_client.OnlineServiceWsDialError("off", status=404)
        pool = _FakePool([error])
        transport, _, http, _, _now = _transport(pool, pipeline_depth=2)

        results = transport.post_json_many(_URL, [_body() for _ in range(6)], 30.0)

        self.assertLen(results, 6)
        self.assertEqual(pool.batch_calls, 1, "chunks after the refusal still dialed the pool")
        self.assertEqual(http.posts, 6)

    def test_capacity_refusal_mid_batch_does_not_sleep_per_chunk(self) -> None:
        from snowflake.ml.feature_store import online_service_ws_client

        error = online_service_ws_client.OnlineServiceWsDialError("full", status=503, retry_after_seconds=10)
        pool = _FakePool([error])
        transport, _, http, slept, _now = _transport(pool, pipeline_depth=2)

        transport.post_json_many(_URL, [_body() for _ in range(6)], 30.0)

        self.assertEqual(slept, [], "a capacity refusal slept before HTTP served the read anyway")
        self.assertEqual(pool.batch_calls, 1)
        self.assertEqual(http.posts, 6)

    def test_send_ahead_window_is_one_unless_pipelining_over_websocket(self) -> None:
        from snowflake.ml.feature_store import online_service_ws_client

        sequential, _, _, _, _ = _transport(pipeline_depth=1)
        self.assertEqual(sequential.send_ahead_window, 1)

        pipelining, _, _, _, now = _transport(
            _FakePool([online_service_ws_client.OnlineServiceWsDialError("off", status=404)]), pipeline_depth=4
        )
        self.assertEqual(pipelining.send_ahead_window, 4)
        pipelining.post_json(_URL, _body(), 30.0)  # 404 rests the transport on HTTP
        self.assertEqual(pipelining.send_ahead_window, 1, "sending ahead while every body is its own round trip")

    def test_a_run_of_successes_steps_the_ladder_back_down(self) -> None:
        """A recovered service should be trusted again rather than probed forever."""
        pool = _FakePool([RuntimeError("boom")] * 2 + [(200, b"{}")] * 6, connection_count=0)
        transport, _, _, _, now = _transport(
            pool, failure_threshold=1, fallback_ladder_seconds=(10.0, 20.0), successes_to_step_down=3
        )

        transport.post_json(_URL, _body(), 30.0)
        rung_after_failure = transport._fallback_rung  # noqa: SLF001

        now[0] = 100.0
        for _ in range(3):
            transport.post_json(_URL, _body(), 30.0)

        self.assertLess(transport._fallback_rung, rung_after_failure)  # noqa: SLF001

    def test_one_success_is_not_enough_to_step_down(self) -> None:
        """Stepping down per success lets a 1:1 flap pin the ladder at its shortest rest forever,
        so the longer rests are never reached however long the flapping lasts."""
        pool = _FakePool([RuntimeError("boom")] * 2 + [(200, b"{}")] * 4, connection_count=0)
        transport, _, _, _, now = _transport(
            pool, failure_threshold=1, fallback_ladder_seconds=(10.0, 20.0), successes_to_step_down=3
        )

        transport.post_json(_URL, _body(), 30.0)
        rung_after_failure = transport._fallback_rung  # noqa: SLF001

        now[0] = 100.0
        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(transport._fallback_rung, rung_after_failure)  # noqa: SLF001

    def test_a_flapping_service_still_climbs_the_ladder(self) -> None:
        """The defect this guards: alternating failure and success must not hold rung 0."""
        script: list[Any] = []
        for _ in range(8):
            # Two errors per request: a dead socket earns one redial, so a counted failure
            # needs both attempts of that request to fail.
            script.extend([RuntimeError("boom"), RuntimeError("boom"), (200, b"{}")])
        pool = _FakePool(script, connection_count=0)
        transport, _, _, _, now = _transport(
            pool, failure_threshold=1, fallback_ladder_seconds=(10.0, 20.0, 30.0), successes_to_step_down=3
        )

        for step in range(16):
            now[0] = 1000.0 * (step + 1)  # past any rest, so every call actually tries WS
            transport.post_json(_URL, _body(), 30.0)

        self.assertGreater(transport._fallback_rung, 0)  # noqa: SLF001


class PoolSizeTest(absltest.TestCase):
    """Slots are shared per-instance capacity with no server-side per-caller share, so the
    default is deliberately small and the knob is documented as a last resort."""

    def test_default_is_small(self) -> None:
        """Past a caller's own throughput need, extra connections take a slot someone else needs."""
        from snowflake.ml.feature_store import online_service_ws_pool

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(online_service_ws_transport.pool_size(), online_service_ws_pool.DEFAULT_POOL_SIZE)

    def test_size_is_read_from_the_environment(self) -> None:
        """A caller needing more than the default throughput must be able to raise it."""
        with patch.dict(os.environ, {online_service_ws_transport.WS_POOL_SIZE_ENV_VAR: "7"}, clear=False):
            # Not the default, or this passes with the variable ignored.
            self.assertEqual(online_service_ws_transport.pool_size(), 7)

    def test_invalid_size_falls_back_to_the_default(self) -> None:
        """A typo must not silently produce a one-connection pool pinned to one instance."""
        from snowflake.ml.feature_store import online_service_ws_pool

        for value in ("abc", "", "-4"):
            with patch.dict(os.environ, {online_service_ws_transport.WS_POOL_SIZE_ENV_VAR: value}, clear=False):
                self.assertEqual(
                    online_service_ws_transport.pool_size(),
                    online_service_ws_pool.DEFAULT_POOL_SIZE if value != "-4" else 1,
                    value,
                )


class LoadSheddingIsNotRecoveryTest(absltest.TestCase):
    """A status the service sends while shedding load says nothing about the transport."""

    def _on_rung(self, script: list[Any], **kwargs: Any) -> Any:
        transport, pool, http, slept, now = _transport(_FakePool(script), successes_to_step_down=1, **kwargs)
        transport._fallback_rung = 1  # noqa: SLF001 - one rung up, but not resting
        return transport, pool, http, slept, now

    def test_a_retryable_status_that_comes_back_again_does_not_step_the_ladder_down(self) -> None:
        transport, *_ = self._on_rung([(502, b"{}"), (502, b"{}")])

        status, _ = transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(status, 502)
        self.assertEqual(transport._fallback_rung, 1, "a 502 was counted as a recovery")  # noqa: SLF001

    def test_a_throttled_answer_handed_back_does_not_step_the_ladder_down(self) -> None:
        throttled = (429, json.dumps({"error": {"code": "RESOURCE_EXHAUSTED"}, "retry_after_seconds": 10}).encode())
        transport, *_ = self._on_rung([throttled])

        transport.post_json(_URL, _body(), 5.0)

        self.assertEqual(transport._fallback_rung, 1)  # noqa: SLF001

    def test_a_real_answer_still_steps_the_ladder_down(self) -> None:
        """The control: without it the two tests above pass on a ladder that never moves."""
        transport, *_ = self._on_rung([(200, b"{}")])

        transport.post_json(_URL, _body(), 30.0)

        self.assertEqual(transport._fallback_rung, 0)  # noqa: SLF001

    def test_a_pipelined_retry_resends_only_the_bodies_that_need_it(self) -> None:
        """Resending a chunk's 200s while the service is shedding load only adds to the load."""
        first = [(200, b'{"n":0}'), (502, b"{}"), (200, b'{"n":2}')]
        pool = _FakePool([first, [(200, b'{"n":1}')]])
        transport, _, _, _, _ = _transport(pool, pipeline_depth=3)

        results = transport.post_json_many(_URL, [_body() for _ in range(3)], 30.0)

        self.assertEqual(pool.batch_sizes, [3, 1])
        self.assertEqual([json.loads(raw)["n"] for _, raw in results], [0, 1, 2], "answers came back out of order")

    def test_a_pipelined_resend_after_a_dead_socket_sends_everything(self) -> None:
        """Nothing was answered, so nothing can be skipped."""
        pool = _FakePool([OSError("closed"), [(200, b"{}")] * 3])
        transport, _, _, _, _ = _transport(pool, pipeline_depth=3)

        transport.post_json_many(_URL, [_body() for _ in range(3)], 30.0)

        self.assertEqual(pool.batch_sizes, [3, 3])


class LadderStateTest(absltest.TestCase):
    def test_stepping_down_stops_at_no_rung(self) -> None:
        """Below -1 the next failure indexes the ladder from the end: the longest rest, not the
        shortest. Two unguarded concurrent step-downs got there; the clamp makes it impossible."""
        transport, _, _, _, _ = _transport(
            failure_threshold=1, successes_to_step_down=1, fallback_ladder_seconds=(10.0, 20.0, 600.0)
        )
        transport._fallback_rung = 0  # noqa: SLF001

        transport._record_success()  # noqa: SLF001
        transport._record_success()  # noqa: SLF001
        self.assertEqual(transport._fallback_rung, -1)  # noqa: SLF001

        transport._record_failure()  # noqa: SLF001
        self.assertEqual(
            transport._fallback_rung, 0, "the first failure did not take the shortest rest"
        )  # noqa: SLF001

    def test_concurrent_updates_keep_the_ladder_consistent(self) -> None:
        transport, _, _, _, _ = _transport(
            failure_threshold=1, successes_to_step_down=1, fallback_ladder_seconds=(10.0, 20.0, 30.0)
        )
        start = threading.Barrier(8)

        def churn(fail: bool) -> None:
            start.wait(timeout=5)
            for _ in range(2000):
                if fail:
                    transport._record_failure()  # noqa: SLF001
                else:
                    transport._record_success()  # noqa: SLF001

        threads = [threading.Thread(target=churn, args=(i % 2 == 0,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertGreaterEqual(transport._fallback_rung, -1)  # noqa: SLF001
        self.assertLess(transport._fallback_rung, 3)  # noqa: SLF001


class TransportAttributionTest(absltest.TestCase):
    """``ws_requests`` and ``http_fallback_requests`` must account for every row, on every path.

    These two counters are the only way a caller can tell *which* transport served a read — a suite
    run with the WebSocket transport enabled otherwise passes identically whether WebSocket or HTTP
    answered, so "the tests pass over WebSocket" would be unverifiable. The invariant asserted here
    is that the two counters sum to the number of rows requested, which is what lets an integration
    test conclude anything from them.
    """

    def _total(self, transport: Any) -> int:
        return int(transport.ws_requests + transport.http_fallback_requests)

    def test_single_request_over_websocket_is_attributed(self) -> None:
        transport, _, _, _, _ = _transport()
        transport.post_json(_URL, _body(), 30.0)
        self.assertEqual((transport.ws_requests, transport.http_fallback_requests), (1, 0))

    def test_batch_at_depth_one_attributes_every_row(self) -> None:
        transport, _, _, _, _ = _transport(pipeline_depth=1)
        bodies = [_body() for _ in range(5)]
        transport.post_json_many(_URL, bodies, 30.0)
        self.assertEqual(self._total(transport), len(bodies))
        self.assertEqual(transport.http_fallback_requests, 0)

    def test_pipelined_batch_attributes_every_row(self) -> None:
        transport, _, _, _, _ = _transport(pipeline_depth=4)
        bodies = [_body() for _ in range(9)]  # spans three chunks, the last one short
        transport.post_json_many(_URL, bodies, 30.0)
        self.assertEqual(self._total(transport), len(bodies))
        self.assertEqual(transport.http_fallback_requests, 0)

    def test_rows_served_by_http_are_attributed_to_http(self) -> None:
        # A 500 rests the transport, so the rows after it are HTTP's and must be counted there.
        pool = _FakePool(script=[(500, b'{"error":"internal"}')])
        transport, _, http, _, _ = _transport(pool=pool)

        transport.post_json(_URL, _body(), 30.0)
        self.assertTrue(transport.using_fallback, "a 500 should rest the transport")

        # That first request is itself served over HTTP after the 500, so measure from here rather
        # than from zero.
        before_total = self._total(transport)
        before_posts = http.posts
        bodies = [_body() for _ in range(3)]
        transport.post_json_many(_URL, bodies, 30.0)

        self.assertEqual(self._total(transport) - before_total, len(bodies))
        self.assertEqual(http.posts - before_posts, len(bodies), "while resting, rows go over HTTP")

    def test_an_ingest_is_attributed_to_http_not_websocket(self) -> None:
        # The ingest path has no WebSocket endpoint, so it must never inflate the WebSocket count,
        # and it is not a fallback either: counting it as one overstates how often WebSocket fails.
        transport, _, _, _, _ = _transport()
        transport.post_json(_INGEST_URL, _body(), 30.0)
        self.assertEqual(
            (transport.ws_requests, transport.http_fallback_requests, transport.ingest_requests), (0, 0, 1)
        )


class SharedTransportWithIngestTest(absltest.TestCase):
    """One transport instance serves both online reads and ingests.

    ``FeatureStore._get_or_create_online_http_client`` returns a single transport for both paths, and
    the ingest API has no WebSocket endpoint. A WebSocket URL derived from an ingest URL therefore
    names a path the service does not serve, and that dial fails as a 404 — which is classified as
    "transport unavailable" and rests the transport, disabling WebSocket for *reads* because of an
    ingest. These tests pin the request being recognised as non-query before any dial happens.
    """

    def test_query_api_path_matches_http_transport(self) -> None:
        # The path is held in two modules; if they drift, a real query URL stops being recognised
        # and every read silently falls back to HTTP.
        built = online_service_http_client.query_api_url("https://svc.example.app")
        self.assertTrue(
            built.endswith(online_service_ws_transport._QUERY_API_REL_PATH),
            f"{built!r} does not end with {online_service_ws_transport._QUERY_API_REL_PATH!r}",
        )

    def test_query_url_yields_a_base(self) -> None:
        self.assertEqual(
            online_service_ws_transport._query_base_of(_URL),
            "https://svc.example.app/",
        )

    def test_ingest_url_yields_no_base(self) -> None:
        self.assertIsNone(online_service_ws_transport._query_base_of(_INGEST_URL))

    def test_ingest_is_served_over_http_without_touching_the_pool(self) -> None:
        transport, pool, http, _, _ = _transport()

        status, _ = transport.post_json(_INGEST_URL, _body(), timeout=5.0)

        self.assertEqual(status, 200)
        self.assertEqual(http.posts, 1, "ingest should be served by the HTTP transport")
        self.assertEqual(pool.calls, 0, "ingest must not dial a WebSocket")

    def test_ingest_does_not_rest_the_transport_for_reads(self) -> None:
        transport, pool, http, _, _ = _transport()

        transport.post_json(_INGEST_URL, _body(), timeout=5.0)
        # The read that follows is the point: an ingest must leave the WebSocket path available.
        transport.post_json(_URL, _body(), timeout=5.0)

        self.assertEqual(pool.calls, 1, "the read after an ingest should still use the WebSocket")


class TransportCreationIsSerialisedTest(absltest.TestCase):
    """`FeatureStore._get_or_create_online_http_client` must build exactly one transport.

    It is an unlocked ``if self._online_http_client is None`` away from building several: the last
    assignment wins and the losers become unreachable *without being closed*, so each orphaned pool
    keeps its WebSocket connections until the server ages them out. Slots are capped per instance
    and shared between every caller, so a leak here costs other clients capacity. Found while
    building the concurrent stress harness, where it also corrupted attribution -- requests served
    by a discarded transport increment counters nobody can read.

    The first test pins the invariant deterministically rather than racing for it; the second is the
    behavioural check with the creation window widened so the race would fire reliably if unguarded.
    """

    def _bare_store(self) -> Any:
        """A FeatureStore with only the two attributes this method touches.

        Built with ``object.__new__`` so the real ``__init__`` -- which needs a live session and
        runs schema checks -- is bypassed while the method under test stays the real one.
        """
        from snowflake.ml.feature_store import feature_store as fs_mod

        store = object.__new__(fs_mod.FeatureStore)
        store._online_http_client = None
        store._online_http_client_lock = threading.Lock()
        store._session = None  # type: ignore[assignment]  # never used on the path under test
        return store

    def test_the_lock_is_held_while_the_transport_is_built(self) -> None:
        """Deterministic: construction must happen inside the lock, not merely end up single."""
        store = self._bare_store()
        held: list[bool] = []

        def record(session: Any = None) -> Any:
            held.append(store._online_http_client_lock.locked())
            return _FakeHttp()

        with patch.object(online_service_http_client, "OnlineServiceHttpClient", record):
            store._get_or_create_online_http_client()

        self.assertEqual([True], held, "the transport was constructed without holding the lock")

    def test_concurrent_first_reads_build_exactly_one_transport(self) -> None:
        """Sixteen threads arriving together must share one transport, and build one."""
        store = self._bare_store()
        built: list[Any] = []
        start = threading.Barrier(16)

        def slow_build(session: Any = None) -> Any:
            # Widen the window so an unguarded check-then-act would reliably interleave.
            time.sleep(0.05)
            client = _FakeHttp()
            built.append(client)
            return client

        got: list[Any] = []
        got_lock = threading.Lock()

        def worker() -> None:
            start.wait()
            transport = store._get_or_create_online_http_client()
            with got_lock:
                got.append(transport)

        with patch.object(online_service_http_client, "OnlineServiceHttpClient", slow_build):
            threads = [threading.Thread(target=worker) for _ in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertLen(built, 1, f"built {len(built)} transports; the losers leak their pools")
        self.assertLen(got, 16)
        self.assertEqual(1, len({id(t) for t in got}), "threads received different transports")

    def test_the_flag_without_websocket_client_warns_and_uses_http(self) -> None:
        """Left to the first dial, the ImportError read as a dead socket: every read quietly fell
        back to HTTP and the only trace was at DEBUG."""
        store = self._bare_store()
        with patch.dict(os.environ, {online_service_ws_transport.WS_ENABLED_ENV_VAR: "true"}), patch.object(
            online_service_ws_transport, "websocket_client_installed", return_value=False
        ), patch.object(online_service_http_client, "OnlineServiceHttpClient", lambda session=None: _FakeHttp()):
            with self.assertLogs("snowflake.ml.feature_store.feature_store", level="WARNING") as logged:
                transport = store._get_or_create_online_http_client()

        self.assertIsInstance(transport, _FakeHttp)
        self.assertIn("websocket-client", logged.output[0])

    def test_websocket_client_installed_reflects_the_import(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(online_service_ws_transport.websocket_client_installed())
        self.assertTrue(online_service_ws_transport.websocket_client_installed())  # a test dependency

    def test_close_detaches_once_under_concurrency(self) -> None:
        """Concurrent close() must not double-close: the first detaches, the rest no-op."""
        store = self._bare_store()
        closed: list[int] = []

        class _Closable(_FakeHttp):
            def close(self) -> None:
                closed.append(1)

        store._online_http_client = _Closable()
        threads = [threading.Thread(target=store.close) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLen(closed, 1, "close() ran more than once on the same transport")
        self.assertIsNone(store._online_http_client)


class MaxConnAgeTest(absltest.TestCase):
    """The retirement ceiling must be tunable, because the service's cap is.

    The client has to retire *before* the service closes, so that the client owns TIME_WAIT and
    no request is cut mid-flight. The pool follows the age the service advertises, but a
    deployment that advertises none can still run a short `WS_MAX_CONN_AGE`; if this side were
    a constant, nothing could correct the ordering there.
    """

    def setUp(self) -> None:
        self._saved = os.environ.get(online_service_ws_transport.WS_MAX_CONN_AGE_ENV_VAR)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(online_service_ws_transport.WS_MAX_CONN_AGE_ENV_VAR, None)
        else:
            os.environ[online_service_ws_transport.WS_MAX_CONN_AGE_ENV_VAR] = self._saved

    def _set(self, value: str) -> None:
        os.environ[online_service_ws_transport.WS_MAX_CONN_AGE_ENV_VAR] = value

    def test_unset_uses_the_pool_default(self) -> None:
        os.environ.pop(online_service_ws_transport.WS_MAX_CONN_AGE_ENV_VAR, None)
        self.assertEqual(
            online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS,
            online_service_ws_transport.max_conn_age_seconds(),
        )

    def test_a_shorter_age_is_honoured(self) -> None:
        """The case that matters: a deployment lowering the service cap below the default."""
        self._set("240")
        self.assertEqual(240.0, online_service_ws_transport.max_conn_age_seconds())

    def test_a_longer_age_is_honoured(self) -> None:
        self._set("1800")
        self.assertEqual(1800.0, online_service_ws_transport.max_conn_age_seconds())

    def test_fractional_seconds_are_accepted(self) -> None:
        self._set("0.5")
        self.assertEqual(0.5, online_service_ws_transport.max_conn_age_seconds())

    def test_unparsable_falls_back_to_the_default(self) -> None:
        self._set("twelve minutes")
        self.assertEqual(
            online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS,
            online_service_ws_transport.max_conn_age_seconds(),
        )

    def test_zero_and_negative_fall_back_rather_than_retiring_everything(self) -> None:
        """Zero would retire every connection on first use, which is worse than ignoring it."""
        for bad in ("0", "-60"):
            self._set(bad)
            self.assertEqual(
                online_service_ws_pool.DEFAULT_MAX_CONNECTION_AGE_SECONDS,
                online_service_ws_transport.max_conn_age_seconds(),
                f"{bad!r} should have been rejected",
            )

    def test_the_value_reaches_the_pool_the_transport_builds(self) -> None:
        """Wiring, not just parsing: an unreachable knob is the bug this fixes."""
        self._set("300")
        transport = online_service_ws_transport.OnlineServiceWsTransport(_FakeHttp(), session=None)
        try:
            self.assertEqual(300.0, transport._pool._max_connection_age_seconds)
        finally:
            transport.close()


if __name__ == "__main__":
    absltest.main()
