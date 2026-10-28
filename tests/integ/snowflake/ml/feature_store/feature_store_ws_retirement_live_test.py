"""Live proof that the client, not the server, closes every WebSocket connection — and only just in time.

The service draws a jittered lifetime for each connection, enforces it as that connection's read
deadline, and advertises it on the upgrade as ``X-WS-Max-Conn-Age-Seconds``. The pool is meant to
read that value and retire the connection a small margin before it (``SERVER_AGE_MARGIN_SECONDS`` /
``SERVER_AGE_MARGIN_FRACTION``), so the client owns the close and no request is ever cut mid-flight.
Every other test of that path uses a fake socket that invents the header; none proves that the real
service sends it, that it survives the ingress, or that retirement lands before the real deadline under
load. This one does, against a live Online Service, with many connections at once.

For every connection lifetime it records the advertised age and the moment the client closed the
socket, and fails if any of these hold:

* the **server** closed a connection first (an empty receive or a lost connection mid-request);
* the client closed it **after** the advertised deadline;
* the client closed it more than ``GRACE_SECONDS`` **before** the deadline (retiring far too early wastes
  a handshake, which is an authentication, per connection — and would mean the header was ignored);
* a connection was retired for any reason other than age (its request budget, say), which would make the
  timing above meaningless;
* any read returned a non-200 response, or a value other than the seeded one (``WS_LIVE_EXPECT``).

It is opt-in and live-only, so it skips unless pointed at a service::

    QUERY_URL=https://<query-endpoint> SNOWFLAKE_PAT=<pat> \\
    WS_LIVE_FV='"user_profile_info"' WS_LIVE_FV_VERSION=v1 \\
    WS_LIVE_JOIN_KEY=USER_ID WS_LIVE_KEY=user_42 WS_LIVE_EXPECT=Tokyo \\
      python -m pytest tests/integ/snowflake/ml/feature_store/feature_store_ws_retirement_live_test.py

The service must advertise the header (the online service with per-connection advertised age). Against one that
does not, the test fails loudly rather than skipping: a missing header is the defect it exists to catch.
Runtime is ``WS_LIVE_RETIRE_SECONDS`` (default 330 s, which outlasts the service's 5 minute cap so every
first-generation connection is retired at least once).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from absl.testing import absltest

from snowflake.ml.feature_store import online_service_ws_client, online_service_ws_pool

GRACE_SECONDS = 30.0

_ENV_REQUIRED = (
    "QUERY_URL",
    "SNOWFLAKE_PAT",
    "WS_LIVE_FV",
    "WS_LIVE_FV_VERSION",
    "WS_LIVE_JOIN_KEY",
    "WS_LIVE_KEY",
    "WS_LIVE_EXPECT",
)


@dataclass
class _Life:
    """One connection lifetime, from handshake to close, on the client's monotonic clock."""

    connected_at: float
    advertised_s: float | None
    closed_at: float | None = None
    closed_by: str | None = None  # "client" or "server"
    server_error: str | None = None
    requests: int = 0


@dataclass
class _Recorder:
    """Shared, thread-safe log of every connection lifetime the pool creates."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    lives: list[_Life] = field(default_factory=list)
    stopping: bool = False  # set before pool.close(): those closes are teardown, not retirement


def _recording_client_factory(recorder: _Recorder) -> Any:
    """Build a pool ``_client_factory`` producing real clients that log connect and close times.

    Only timestamps are added; every socket operation is the library's own.

    Args:
        recorder: Where lifetimes are logged.

    Returns:
        A factory with the signature the pool calls.
    """

    class _RecordingClient(online_service_ws_client.OnlineServiceWsClient):
        _life: _Life | None = None

        def connect(self, url: str, timeout: float | None = None) -> None:
            fresh = self._ws is None  # noqa: SLF001 - white-box: connect() is a no-op when already open
            super().connect(url, timeout)
            if fresh and self._ws is not None:  # noqa: SLF001
                life = _Life(connected_at=time.monotonic(), advertised_s=self.server_max_conn_age_seconds)
                self._life = life
                with recorder.lock:
                    recorder.lives.append(life)

        def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
            try:
                result = super().post_json(url, body, timeout)
            except online_service_ws_client.OnlineServiceWsDialError:
                raise  # never connected: no lifetime to attribute
            except Exception as e:  # noqa: BLE001 - every mid-request failure is the server closing first
                life = self._life
                if life is not None and life.closed_at is None:
                    life.closed_at, life.closed_by = time.monotonic(), "server"
                    life.server_error = f"{type(e).__name__}: {str(e)[:200]}"
                self._life = None
                raise
            if self._life is not None:
                self._life.requests += 1
            return result

        def close(self) -> None:
            life = self._life
            if life is not None and life.closed_at is None:
                life.closed_at = time.monotonic()
                life.closed_by = "teardown" if recorder.stopping else "client"
            self._life = None
            super().close()

    return _RecordingClient


class WsClientRetiresBeforeServerLiveTest(absltest.TestCase):
    """Many live connections; every one must be retired by the client inside the grace window."""

    def setUp(self) -> None:
        super().setUp()
        missing = [name for name in _ENV_REQUIRED if not os.environ.get(name, "").strip()]
        if missing:
            self.skipTest(f"live test: set {', '.join(missing)} to run against an Online Service")

    def test_every_connection_is_retired_by_the_client_within_the_grace_window(self) -> None:
        connections = int(os.environ.get("WS_LIVE_RETIRE_CONNECTIONS", "100"))
        duration = float(os.environ.get("WS_LIVE_RETIRE_SECONDS", "330"))
        pace = float(os.environ.get("WS_LIVE_RETIRE_PACE_SECONDS", "0.02"))  # per reader, keeps the budget idle
        url = online_service_ws_client.ws_query_url(os.environ["QUERY_URL"])
        body = {
            "name": os.environ["WS_LIVE_FV"],
            "version": os.environ["WS_LIVE_FV_VERSION"],
            "object_type": "feature_view",
            "metadata_options": {"include_names": True},
            "request_rows": [{"entity": {os.environ["WS_LIVE_JOIN_KEY"]: os.environ["WS_LIVE_KEY"]}}],
        }

        expect = os.environ["WS_LIVE_EXPECT"]
        recorder = _Recorder()
        pool = online_service_ws_pool.OnlineServiceWsPool(
            size=connections,
            _client_factory=_recording_client_factory(recorder),
        )
        # Claim every slot before the readers start. A connection is opened only when every one
        # the pool holds is busy, and these readers are deliberately idle for most of their cycle
        # -- a few ms in flight per `pace` seconds, so the request budget stays well clear and
        # retirement is by age. Left to grow on its own the pool would open a fraction of
        # `connections`, and the run would observe a fraction of the lifetimes it exists to check.
        # Reserving claims the slots without dialing; the readers dial them on first use.
        key = online_service_ws_pool.OnlineServiceWsPool.connection_key(url)
        reserved = [pool._select(key) for _ in range(connections)]  # noqa: SLF001 - reserving is the point
        for pooled in reserved:
            pool._finish(key, pooled, requests=0, evict=False)  # noqa: SLF001 - white-box, repo style

        bad: list[tuple[float, str, str]] = []  # (seconds into the run, kind, detail)
        sent = [0]
        started = time.monotonic()
        stop_at = started + duration

        def reader() -> None:
            while time.monotonic() < stop_at:
                try:
                    status, raw = pool.post_json(url, body, 10.0)
                    payload = json.loads(raw)
                    results = payload.get("results")
                    first = (results[0].get("features") or [None])[0] if isinstance(results, list) and results else None
                    if status != 200 or len(results or []) != 1 or str(first) != expect:
                        with recorder.lock:
                            bad.append((time.monotonic() - started, f"status={status}", repr(raw[:300])))
                except Exception as e:  # noqa: BLE001 - recorded; the per-life check attributes it
                    with recorder.lock:
                        status_of = getattr(e, "status", None)
                        kind = type(e).__name__ + (f" {status_of}" if status_of else "")
                        bad.append((time.monotonic() - started, kind, str(e)[:600]))
                with recorder.lock:
                    sent[0] += 1
                time.sleep(pace)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(connections)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(duration + 60)
        recorder.stopping = True
        pool.close()

        with recorder.lock:
            lives = list(recorder.lives)
        retired = [life for life in lives if life.closed_by == "client"]
        server_first = [life for life in lives if life.closed_by == "server"]
        no_header = [life for life in lives if not life.advertised_s]
        late, early = [], []
        for life in retired:
            deadline = life.connected_at + float(life.advertised_s or 0)
            lead = deadline - float(life.closed_at or 0)  # seconds the client closed before the server would
            if lead <= 0:
                late.append(round(lead, 2))
            elif lead > GRACE_SECONDS:
                early.append(round(lead, 2))
        leads = sorted(
            round(life.connected_at + float(life.advertised_s or 0) - float(life.closed_at or 0), 2) for life in retired
        )
        budget_floor = online_service_ws_pool.DEFAULT_MAX_REQUESTS_PER_CONNECTION * (
            1 - online_service_ws_pool.DEFAULT_SPLAY_FRACTION
        )
        over_budget = [life.requests for life in retired if life.requests >= budget_floor]

        summary = (
            f"{len(lives)} connection lifetimes, {sent[0]} reads; client-retired {len(retired)}, "
            f"server-first {len(server_first)}, no header {len(no_header)}, late {len(late)}, "
            f"early {len(early)}; lead p0/p50/p100 = "
            f"{leads[0] if leads else '-'} / {leads[len(leads) // 2] if leads else '-'} / "
            f"{leads[-1] if leads else '-'} s; advertised range "
            f"{min((life.advertised_s or 0) for life in lives) if lives else '-'}-"
            f"{max((life.advertised_s or 0) for life in lives) if lives else '-'} s"
        )
        print(summary, flush=True)
        failures: dict[str, list[tuple[float, str]]] = {}
        for at, kind, detail in bad:
            failures.setdefault(kind, []).append((at, detail))
        breakdown = "; ".join(
            f"{kind}: {len(hits)} between t+{hits[0][0]:.0f}s and t+{hits[-1][0]:.0f}s, first: {hits[0][1]}"
            for kind, hits in failures.items()
        )
        if breakdown:
            print(f"failed reads: {breakdown}", flush=True)

        self.assertEmpty(no_header, f"service did not advertise X-WS-Max-Conn-Age-Seconds — {summary}")
        self.assertEmpty(
            [life.server_error for life in server_first],
            f"the server closed {len(server_first)} connection(s) before the client did — {summary}",
        )
        self.assertEmpty(late, f"client closed after the advertised deadline (lead s): {late[:10]} — {summary}")
        self.assertEmpty(early, f"client closed > {GRACE_SECONDS}s early (lead s): {early[:10]} — {summary}")
        self.assertEmpty(over_budget, f"retired by request budget, not age: {over_budget[:10]} — {summary}")
        self.assertGreaterEqual(
            len(retired), connections, f"expected every first-generation connection to retire — {summary}"
        )
        self.assertEqual(0, len(bad), f"{len(bad)} reads failed or were malformed — {breakdown} — {summary}")


if __name__ == "__main__":
    absltest.main()
