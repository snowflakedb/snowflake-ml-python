# WebSocket Online-Read Client — Design

A second transport for `read_feature_view(..., store_type=ONLINE)`, alongside the HTTP client.
One authenticated upgrade carries many feature reads, removing per-request connection setup and
per-request ingress authentication from the hot path. The server side is documented in
the online service's own design document; this covers the client.

**Measured in-region against a live deployment: 2.3-3.0x lower latency than HTTP at every batch
size, zero failures across 1,993,760 reads where HTTP loses 83-87% above one concurrent reader,
and 7.2M queries in a 73-minute soak with a value asserted on every read.** The HTTP path is
unchanged and remains the default; WebSocket is off unless explicitly enabled.

---

## Architecture

Four layers, each wholly containing the next. The nesting is why the harness can attribute
latency: subtract adjacent layers and the difference is that layer's own cost.

| File | Role |
|---|---|
| `online_service_ws_client.py` | One connection: handshake, URL, framing, `request_id` correlation, dial errors |
| `online_service_ws_pipeline.py` | Several requests in flight on one connection, correlated by `request_id` |
| `online_service_ws_pool.py` | Many connections: keying, selection policy, age/budget recycling, two-phase retirement |
| `online_service_ws_transport.py` | Policy: opt-in, the HTTP fallback decision, the retry ladder, telemetry counters |
| `online_service_transport.py` | The interface both transports satisfy; no transport conditionals in callers |

**Selection reserves before releasing the lock.** `outstanding` is incremented inside
`_select_locked`, not after the caller has the connection — otherwise a connection still looks
idle in the window between selection and use, and a concurrent selection can retire it out from
under a caller about to send on it.

**Retirement is two-phase.** A connection due for replacement is marked `retiring` and removed
from the pool immediately so no new request picks it, then closed once its in-flight work drains.
Waiting for a connection to look idle before retiring it means never retiring the busy ones — and
under sustained load every connection is busy every time it is examined, so both the age limit and
the request budget would go unenforced exactly when they matter.

**A failed dial drains like a retirement.** It leaves rotation at once, but its client is closed
only when no other caller is still reserved on it. Closing it immediately turned every queued
caller's request into a misleading "client is closed" error, which the transport then spent its one
retry on. `outstanding`, `requests` and `retiring` all change under the selection lock, and sockets
are closed after releasing it, because a WebSocket close can wait seconds for the server's frame.

**A connection is used under a lock, not merely accounted for.** A WebSocket permits one reader
and one writer; two callers interleaving sends on one socket can each read the other's response,
returning another request's feature values with no error raised.

**Connections are keyed by endpoint, not by what is read over them.** The handshake carries only
the caller's credentials and the query URL is store-wide; every message names its own feature view
or feature group, so the service binds no scope at dial time and any connection to an endpoint can
serve any read of it. Keying by the object read instead multiplies sockets by the number of views a
caller touches — 25 views at the default size held 100 authenticated connections, out of roughly
3,072 across the fleet — for no gain in return.

### Slots are claimed eagerly, sockets dialed lazily

Three counts, and they differ:

| property | meaning |
|---|---|
| `connection_count` | slots held, opened or not |
| `live_connection_count` | slots whose socket is actually open |
| `draining_connection_count` | retired, out of rotation, still finishing in-flight work |

`connect()` is idempotent — `if self._ws is not None: return` — so the "do I need to upgrade?"
question is one null check, not a state machine. A slot is also added only when every one the
pool holds is busy, so the configured size is a ceiling on concurrency rather than a count of
sockets to fill: a caller with one request in flight holds one connection however large the pool
is and however many requests it sends, and concurrent callers open as many as they keep busy.

---

## The batching boundary, and why it dominates every number

`read_feature_view` splits a **read** into **requests** at `_MAX_QUERY_REQUEST_ROWS = 10`
(`online_service.py`), so a 100-key read is **ten requests, ten sequential round trips**, and a
1,000-key read is a hundred. Those two words are kept apart everywhere below: a *read* is one
`read_feature_view` call, a *request* is one message on the wire carrying up to 10 rows. Every
latency in this document is **per read**; the server's own figures are per request, so the two
are directly comparable only at 10 keys or fewer, where a read is one request.

**Depth is the third unit, and it acts on requests.** `SNOWFLAKE_ML_ONLINE_WS_PIPELINE_DEPTH`
is how many requests one connection keeps in flight at once, so a pool of connections and a depth
multiply:

```text
concurrent requests in flight  =  pool size x depth
      4 connections, depth  1  =   4
      4 connections, depth 16  =  64
```

Depth therefore only does something when a read is more than one request -- above ten keys. At ten
keys or fewer there is nothing to send ahead, and depth is inert.

**Per-read latency is `ceil(keys / 10) x per-round-trip cost` and almost nothing else.** This is
the single most common way these numbers get misread: the server's own per-query latency is one
round trip, so comparing it against a multi-key read compares one round trip to ten or a hundred.
A transport-level benchmark cannot see this loop at all.

---

## Error contract and the fallback ladder

`using_fallback` is one comparison — `_clock() < _fallback_until`. There is no thread, timer or
background redial. While resting, every request goes to HTTP **without attempting WebSocket at
all**; when the clock passes, the next request *itself* attempts WebSocket inline and that request
is the probe. Three consecutive successes step the rung back down.

Rungs are `(60, 120, 300, 600)` seconds, the last a ceiling: a persistent outage is probed every
ten minutes, so a recovered service is picked up without a restart while a dead one costs one
probe per ten minutes.

| trigger | effect |
|---|---|
| dial refused `404` (transport off), `401`/`403` (auth) | straight to the ceiling — it will not clear soon |
| response status `500` | discards the pool, advances one rung per occurrence |
| response `503`/`504` with `retry_after_seconds` | honored and resent; **not** a failure |
| dial refused at capacity, `retry_after_seconds` | WS rested that long (max 30 s), served over HTTP; a failure |
| generic transport failure | one redial, then counted; three in a row trip the ladder |
| server-initiated close | **not** a failure, by design — the service closes connections on a schedule |

**`http_fallback_requests` increments on the first failing request**, before any rest is set. The
ladder only decides whether *subsequent* requests skip the WebSocket attempt.

---

## Configuration

Off by default. All read when a `FeatureStore` builds its online transport, on its first online read
or ingest, so set them before that; import order does not matter.

| Variable | Default | Effect |
|---|---|---|
| `SNOWFLAKE_ML_ONLINE_WS_ENABLED` | unset | Opt in; anything falsy keeps the HTTP path |
| `SNOWFLAKE_ML_ONLINE_WS_POOL_SIZE` | `16` | Ceiling on connections per endpoint; lower it to spend fewer |
| `SNOWFLAKE_ML_ONLINE_WS_PIPELINE_DEPTH` | `1` | Requests in flight per connection; collapses round trips |
| `SNOWFLAKE_ML_ONLINE_WS_MAX_CONN_AGE_SECONDS` | pool default | Client retirement age; keep **below** the server cap |

Those four are the whole surface: nothing else here reads the environment, and `FeatureStore`
exposes no keyword for any of it. A non-integer value logs a warning and falls back to the
default rather than failing the read.

A pool smaller than the caller count reports its own queueing as latency — see
[Pool size is load-bearing](#pool-size-is-load-bearing).

### What cannot be tuned, and when you will notice

These are compile-time constants. They are listed because each one is visible in behaviour, so
reaching for a setting that does not exist is otherwise a support question.

| Constant | Value | What it governs |
|---|---|---|
| `DEFAULT_MAX_REQUESTS_PER_CONNECTION` | `100,000` ±`25%` | Requests before a connection recycles |
| `SERVER_AGE_MARGIN_SECONDS` | `5.0`, or 10% of the advertised age | How early the client closes |
| `DEFAULT_FALLBACK_LADDER_SECONDS` | `60, 120, 300, 600` | How long reads rest on HTTP |
| `DEFAULT_FAILURE_THRESHOLD` | `3` | Failures before resting |
| `DEFAULT_SUCCESSES_TO_STEP_DOWN` | `3` | Clean reads to come back a rung |
| `DEFAULT_REDIAL_JITTER_SECONDS` | `0.25` | Spread on a redial |
| `POOL_QUEUE_SHARE_OF_TIMEOUT` | `0.5` | Share of the deadline spent queueing |
| selection policy | round-robin | Which held connection serves a read |

Three of those have consequences worth stating outright.

**The request budget silently outranks the age setting.** A connection recycles after ~100,000
requests whatever its age, and above roughly 139 requests/s on one connection that arrives first.
So raising `WS_MAX_CONN_AGE_SECONDS` does not keep a busy connection alive longer. Retiring
*earlier* than the age ceiling is always safe — the guarantee is that the client closes before the
server, and closing sooner cannot break it.

**A credential blip costs ten minutes.** A refused handshake (`401`/`403`) goes straight to the
ladder's `600s` rung rather than climbing, so a brief auth failure parks every read on HTTP for ten
minutes.

**The close margin is flat in practice.** It is the smaller of 5s and 10% of the advertised age, so
for any advertised lifetime above 50s the 10% term is larger and the margin is simply 5 seconds.

### Threads per process: about four, then add processes

The pool is per `FeatureStore` instance, so in normal use it is per process -- and a Python
process, not the transport or the service, is what limits a caller. A read costs far more
interpreter time than the ~3.4 ms round trip it waits on, so threads in one process queue for the
GIL rather than overlapping on the wire.

In deployment terms, a **process** is one replica of your service. A **connection** is borrowed by
a read for as long as that read is in flight and handed straight back, so the connections a replica
uses is its **peak simultaneous reads** -- at most its thread count, and usually far fewer. A thread
doing anything other than a feature read holds nothing, so a replica with a thousand threads holds
four connections if only four are ever mid-read at once. Threads and connections are not the same
thing; they coincide only in a handler pool where every request reads features.

So a replica sized at four simultaneous reads serves ~870 reads per second at ~4.6 ms each, and
scaling is a matter of replicas. **This library never creates processes** -- forking inside a
notebook, a container entrypoint or a web-server worker is not its business. Your deployment
already decides this: gunicorn or uvicorn workers, Kubernetes replicas, Snowpark Container Services
instances. A caller needing 100,000 reads per second needs roughly 80 replicas of 4 connections,
which is 320 of the roughly 3,072 connections a three-instance deployment holds -- and a caller
operating at that scale is already running a fleet, not a single process.

Measured in-region, one process, pool sized to match threads, one request per read:

| threads | reads/s | p50 | p99 | per connection |
|---:|---:|---:|---:|---:|
| 1 | 262 | **3.4 ms** | 5.2 ms | 262 |
| 2 | 477 | 4.1 ms | 5.7 ms | 238 |
| **4** | **819** | **4.6 ms** | 7.4 ms | 205 |
| 8 | 1,222 | 6.2 ms | 11.2 ms | 153 |
| 16 | 1,235 | 12.1 ms | 28.0 ms | 77 |
| 32 | 1,216 | 23.7 ms | 65.7 ms | 38 |

Throughput is flat from 8 threads on -- 1,222 to 1,235 to 1,216 -- while p50 doubles and doubles
again. **Four threads per process is the recommended operating point**: it holds p50 under 5 ms,
and the 49% more throughput that 8 threads buys costs 35% more latency. Past 8 a caller pays
latency for nothing.

This arm varies the pool with the thread count, so it describes caller concurrency and says
nothing about the pool on its own. Holding threads at sixteen and varying only the pool separates
them, same one-request reads:

| pool | reads/s | mean | wire per request | dials |
|---:|---:|---:|---:|---:|
| 4 | 869 | 18.4 ms | 3.8 ms | 11 |
| **8** | **1,195** | **13.4 ms** | 5.1 ms | 19 |
| 16 | 1,273 | 12.5 ms | 9.2 ms | 52 |
| 32 | 1,240 | 12.9 ms | 12.0 ms | 122 |

The gain is 37% from 4 to 8 and 6.6% from 8 to 16, against a measured run-to-run spread of about
±3%; 32 lands below 16. Eight connections reach 94% of the peak while dialing a third as often as
sixteen -- and since every dial is an authentication, dial rate is the cost that matters once
throughput has flattened. A hundred-key read shifts the balance: there the same sixteen threads
give 111.7 reads/s at pool 4 against 288.1 at pool 16, because a read that splits into ten
requests holds its connection ten times as long.

**Scale with processes, not threads.** They compose almost exactly, because each gets its own
interpreter. Same thirty-two connections, three ways:

| shape | connections | reads/s | p50 | failures |
|---|---:|---:|---:|---:|
| 32 threads x 1 process | 32 | 1,182 | 26.9 ms | 0 |
| 4 threads x 8 processes | 32 | 6,878 | 4.45 ms | 0 |
| **2 threads x 16 processes** | 32 | **7,403** | **4.18 ms** | 0 |

Identical socket count and identical service; 6x the throughput at one sixth the latency purely
from splitting the interpreter. Eight processes of four threads was predicted at 6,880 from the
per-process figure above and measured 6,878; sixteen of two was predicted at ~7,600 and measured
7,403. A caller needing more than ~1,200 reads/s needs another process.

### Sizing the pool from planned load

A connection carries one request at a time, so what it sustains is one round trip's worth —
about **262 requests per second** in-region, measured at 3.4 ms on a single unloaded connection
— and the pool's ceiling is `size x that`.

**Count requests, not keys and not reads.** This is the batching boundary again, and it is the
easiest place to be wrong by 10x. One request is one batch of up to `_MAX_QUERY_REQUEST_ROWS =
10` keys, so a connection sustains ~262 requests/s, which is up to ~2,620 keys/s, which is ~26
`read_feature_view` calls/s at 100 keys each. Convert your load to requests first:

```text
requests_per_second  =  reads_per_second x ceil(keys_per_read / 10)

size  >=  requests_per_second / 262                 # measure the round trip outside us-west-2
size  >=  requests_per_second / (262 x depth)       # with SNOWFLAKE_ML_ONLINE_WS_PIPELINE_DEPTH set

# and the limit this arithmetic does not express:
one process  <=  ~1,230 requests/s                  # whatever the pool size -- see above
```

So 1,000 requests/s needs about 4 connections and 5,000 needs about 20 — but 1,000
*`read_feature_view` calls* per second at 100 keys each is 10,000 requests/s and needs about 40.
Size to your **peak concurrent reads**, not your thread count: thread count is an upper bound on
read concurrency and usually a loose one. Where the two coincide -- a handler pool where every
request reads features -- a caller with 8 such handlers wants 8 connections, or the extras queue
and that wait lands in their latency.

**Where the useful range ends, and it depends on the batch.** Writing `S` for the time a
connection is held per read and `D` for the Python cost per read, which serializes under the
interpreter lock, adding connections stops helping once `c` passes roughly `S/D`. The two regimes
land far apart, and the measured curves at sixteen threads are what to size against:

| pool | 10 keys/read | | 100 keys/read | |
|---:|---:|---:|---:|---:|
| | keys/s | p99 | keys/s | p99 |
| 4 | 8,693 | 70.7 ms | 11,170 | 367.3 ms |
| 8 | 11,949 | 67.8 ms | 17,380 | 214.6 ms |
| **16** | **12,732** | **31.2 ms** | **28,810** | **96.5 ms** |
| 32 | 12,401 | 28.0 ms | 31,040 | 67.4 ms |

**Sixteen connections is the plateau in both regimes.** At ten keys the curve peaks there and
turns back down -- per-request wire time inflates from 3.8 ms to 12.0 ms as connections are added,
so they degrade each other's service time -- and thirty-two is 2.6% worse. At a hundred keys
thirty-two is 7.7% better, which is barely past the ±3% run-to-run spread and costs twice the
dials. Below sixteen both curves pay: eight connections reach 94% of peak at ten keys but only 60%
at a hundred, and four reach 68% and 39%.

Note what the pool buys in the **tail** rather than the median. Going from 8 to 16 connections
moves p99 from 67.8 ms to 31.2 ms at ten keys while the median barely shifts: queueing lands
disproportionately in the tail, which is usually where an SLO is written.

**Batch your reads.** The same pool and the same threads move **2.3x more keys per second** at a
hundred keys per read than at ten -- 28,810 against 12,732 -- because the fixed per-call cost
(view lookup, frame assembly, validation) amortizes across the batch: 810 us of Python for a
one-request read against 347 us per request for a ten-request one. The trade is per-call latency,
and it is steep: p50 is 6.3 ms for a ten-key read and 53.2 ms for a hundred-key one. **Small reads
for latency, large reads for key throughput** -- optimizing the wrong one of those costs more than
2x either way.

Treat `S/D` as a floor rather than a prediction. `S` is not constant across a sweep -- it inflates
with the pool -- so extrapolating the connection bound from one measured point overstates every
larger pool. The figures above are measured at each size.

**A high ceiling is close to free in latency and is not free in connections.** Four threads
against a cap of 32 matched a cap of 4 exactly on median and throughput (4.6 ms, ~810 reads/s) --
but finished the run holding all 32 connections, having dialed 236 against the other arm's 26.
Growth is lazy per selection, not self-limiting over time: a connection opens only when every one
held is busy, and over twenty minutes those transient moments accumulate until the pool reaches
its ceiling, where it stays. So the ceiling is reached by any long-lived caller regardless of its
concurrency, and each connection is an authentication and a slot out of the roughly 3,072 the
fleet has.

**The number of feature views does not enter this calculation** — every view shares the same
connections, so a caller reading 100 of them at 500 requests/s needs the same pool as one reading
a single view at 500 requests/s. Size for request rate, not for breadth.

Two directions to get this wrong, and they fail differently. Undersized, the pool does not
refuse work: callers queue on a connection and that wait is charged to the request, so outgrowing
your pool looks like a latency problem rather than a capacity error — the 3.4x above is exactly
this. Oversized, every connection is an authentication and a slot out of the roughly 3,072 the
fleet has (1,024 per instance, three instances), with nothing on the server reserving a share
per caller, so a pool far past your planned QPS costs every other client and buys you nothing.

---

## Testing

Five layers, each faking something different: **unit** (connection mechanics against a fake
socket), **adversarial** (the concurrency hazards, run repeatedly because races do not show on one
pass), **scale** (1,024 slots, counter drift and the known-defect guards, all deterministic and on
an injected clock), **integ** (a live service, including an offline-vs-online oracle), and
**perf** (the harnesses in `tests/perf/feature_store/` — the only layer that measures rather than
asserts).

The first three are in the validation pipeline and together run in about 11 seconds. Nothing that
waits on wall time or reads the machine is: the soak, the churn storm, throughput against pool
size, selection cost and per-slot memory live under `tests/perf/feature_store/`, which has no
build target.

```bash
# the gate — the whole feature store suite, which is where a regression shows up
pytest -q snowflake/ml/feature_store/

# adversarial repeated, because a single green pass is not evidence about a race
for i in 1 2 3; do pytest -q snowflake/ml/feature_store/online_service_ws_adversarial_test.py; done

# scale and the known-defect guards — deterministic, a few seconds
pytest -q snowflake/ml/feature_store/online_service_ws_scale_test.py

# the timed ones, out of the pipeline: ~100s at the default 60s soak and 30s churn storm
python -u tests/perf/feature_store/ws_pool_load.py
SNOWML_WS_SOAK_SECONDS=3600 python -u tests/perf/feature_store/ws_pool_load.py SoakTest

# integ over WebSocket, against a live deployment
SNOWFLAKE_CONNECTION_NAME=<conn> SNOWFLAKE_ML_ONLINE_WS_ENABLED=true \
  pytest -q tests/integ/snowflake/ml/feature_store/feature_store_online_read_refresh_test.py

# the transport-asserting integ test: real websocket-client, and ws_requests must move
bazel test //tests/integ/snowflake/ml/feature_store:feature_store_ws_transport_integ_test \
  --test_env=SPEC_OFT_BUNDLE_REUSE_SCHEMA=MY_DB.MY_SCHEMA
```

**Set `SNOWFLAKE_ML_ONLINE_WS_ENABLED` before a `FeatureStore`'s first online read.** Each
`FeatureStore` builds its transport once, on that first read, and keeps it, so flipping the variable
afterwards changes nothing for that instance until `close()` drops the transport -- which is what
makes a threaded run that flips it unreproducible.

**Run the integ targets over HTTP as well, as a control.** They had never been run on the bench
host at all, so without the control a WebSocket failure cannot be distinguished from a pre-existing
one.

### What each layer is actually for

`online_service_ws_client_test.py` and `online_service_ws_pipeline_test.py` pin the one silent
failure this transport has — a response matched to the wrong request. Everything else fails loudly;
cross-talk returns another caller's feature values with no error raised at all.

`online_service_ws_pool_test.py` covers the hazards pooling introduced rather than re-covering
happy paths: a connection closed while in use, a slot leaked past `close()`, a retirement that
starves under sustained load.

`online_service_ws_adversarial_test.py` is modelled on the server's own adversarial suite. Several
of its tests began as reproductions of defects found while writing it, which is why they force
interleavings with barriers rather than hoping for a rare one. `LadderEscalationTest` came from the
live campaign instead of from review: fault injection at four concurrent callers reached the
ladder's ceiling on its first bad moment, where the same code read serially suggests the first rung.

`online_service_ws_scale_test.py` exists for properties that only appear at scale — 1,024 slots
held at once, a connection never closed while a request runs on it, counters that do not drift over
120,000 requests. Its `KnownDefectTest` cases are deterministic reproductions of three races found
under load and kept as guards; the rarest fired **5 times in 3.66M requests**, which is why forcing
the interleaving is the only way to assert on it.

### Known gaps

- **The real `websocket-client` library — live-only.** The unit, adversarial and stress suites inject
  a fake socket through the `_ws_factory` seam. Only `feature_store_ws_transport_integ_test` and
  `feature_store_ws_retirement_live_test` load the real library and perform a real handshake, and
  both need a live service; there is still no
  hermetic test against a hand-rolled RFC 6455 peer.
- **Which transport answered — asserted only by that integ test.** It requires `ws_requests` to move
  and `http_fallback_requests` to stay 0. The other integ suites pass identically over either
  transport, so on their own a silent fall back to HTTP would go unnoticed.
- **Bazel.** The `py_test` targets' status under the repo's own aspect is unconfirmed; local pytest
  is clean.
- **`least_outstanding` and round-robin under real contention.** Tie-breaking (`[2, 2]`) and
  round-robin skipping a busy connection are unit-tested; neither has been measured against a live
  service as a choice between the two policies. Pool *size* under concurrency is measured — see
  [Pool size is load-bearing](#pool-size-is-load-bearing) — but which picker is better at a given
  size is not.
- **Partial batch mid-stream.** A peer answering 3 of 6 then closing has no test, unit or live.
- **`±Inf`, high-precision `NUMBER`.** Serialization edges through the JSON envelope, untested.
- **Ingest over WebSocket.** The transport is shared but only reads use it; never routed.
- **Mid-upgrade, mid-crash.** Unobserved from this client. The one rolling-upgrade observation was
  the server's Go bench; an instance kill has no reachable mechanism.
- **Pipelining cleared.** Depth-4 evidence reaches ~1.2M requests at zero mismatches, against a race
  that fired 5 times in 3.66M. Evidence, not proof.

**A load test asserts almost nothing** — it reports percentiles and exits 0 regardless of
correctness. These harnesses are the exception by construction: `ws_stress_e2e.py` seeds
key-derived values and checks every returned row, and `ws_fault_inject.py` asserts that a degraded
read is still a *correct* read. A 200 is not a result, and an all-null frame arrives as success.

---

## Verification against real infrastructure

Load alone cannot exercise the ladder: millions of healthy queries produce zero fallbacks. These
are injected faults, run against the live deployment through the real ingress.

1. **Handshake refused `404` (wrong path).** Fell back and latched the ceiling; **19.2% / 19.3%** of
   reads lost at 4 workers over two runs, **zero wrong values**.
2. **Handshake refused `401`/`403` (bad credential).** Same shape, **20.3%** lost, zero wrong values;
   the HTTP path stayed healthy while WebSocket auth was broken.
3. **Response status `500`, held 720 s, 1 worker.** Walked all four rungs — `t+1.0` rung 0, `t+61.0`
   rung 1, `t+181.0` rung 2, `t+481.1` rung 3 — each probe firing exactly when the previous rest
   expired.
4. **Fault lifted mid-rest, 1 worker.** Probe at **`t+61.0`** on a 60 s rest; rung returned to −1;
   21,721 reads then served over WebSocket with **0 failures**.
5. **Fault lifted mid-rest, 4 workers.** Probe at **`t+600.4`** on the 600 s ceiling; recovered
   unaided; 12.2% lost while degraded, zero wrong values.
6. **Server-initiated close, client age above the server cap.** Repeated server-first closes
   absorbed with **0 failures, 0 mismatches**.
7. **Integ suite over WebSocket.** The offline-vs-online oracle and the online-management suite both
   pass; the oracle also passes over HTTP as a control.

**The fallback is a correctness mechanism, not an availability one.** Every arm returned right
answers or a loud `RuntimeError` — never wrong data. But at four workers it loses 12-20% of reads,
because HTTP cannot carry that concurrency (see the ladder above). At one worker it loses nothing.

**Escalation is concurrency-dependent.** `_rest_transport` advances a rung on every `500` with no
debounce, and `using_fallback` is consulted only on entry, so every caller already past that gate
advances the ladder again. N callers in flight cost N rungs — a concurrent client reaches the
600 s ceiling on its first bad moment, where the same code read serially suggests 60 s. Covered
deterministically by `LadderEscalationTest`.

---

## Performance

All arms in-region (~2.8 ms RTT) against a live deployment of 3 instances, measured through the
public API rather than the transport in isolation. **Every arm below is reproducible command for
command: see [`REPRODUCING.md`](REPRODUCING.md).**

### One round trip — the quotable number

10 keys, 1 batch, 1 worker — **one hour per transport**. One request per read, and both arms
complete their work, so every percentile below is a like-for-like comparison:

| | WS | HTTP |
|---|---:|---:|
| p50 | **3.9 ms** | 9.6 ms |
| p95 | **4.7 ms** | 12.1 ms |
| p99 | **5.2 ms** | 13.4 ms |
| throughput | **257.5 reads/s** | 101.7 reads/s |
| reads | **926,994** | 366,298 |
| failed | **0** | 19 |

**2.46x on latency, 2.53x on throughput.** An earlier 5,000-read version of this same arm gave
4.9 ms against 11.2 ms (2.29x); the hour-long run is 185x the sample and slightly kinder to both.
The older figures, for completeness:

| | WS | HTTP |
|---|---:|---:|
| p50 | 4.9 ms | 11.2 ms |
| p99 | 5.9 ms | 14.4 ms |
| throughput | 201.3 reads/s | 87.4 reads/s |
| failed / mismatches | 0 / 0 | 0 / 0 |

**2.29x.** At one worker the decomposition shows no client overhead left to find:

| layer | p50 |
|---|---:|
| `read_helper` | 4.3 ms |
| `transport` | 4.3 ms |
| `pool_total` | 4.2 ms |
| `client_wire` | **4.2 ms** |
| derived pool wait | 0.009 ms (0%) |

`client_wire` 4.2 ms against the server's own measured p50 of 4.0 ms: **the transport costs
~0.2 ms and the wrapper ~0.7 ms.**

**This holds at one worker, and inverts under concurrency in one process.** The decomposition
above has no client overhead left to find because a single thread never waits for the
interpreter. Thirty-two threads in one process measured the same code at `client_wire` 14.3 ms
with 46% of each read spent waiting for a connection -- not because the wire changed, but because
a read costs ~846 us of Python and threads queue for it. Read "latency is round trips, not client
code" as true of a read in isolation; at scale the client's own CPU is what a caller runs into
first. See [Threads per process](#threads-per-process-about-four-then-add-processes).

### Latency is linear in batch count

1 worker, 1,000 reads per arm:

| keys | batches | WS p50 | per batch | HTTP p50 | per batch | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 4.5 ms | 4.50 | 11.7 ms | 11.70 | 2.60x |
| 20 | 2 | 9.3 ms | 4.65 | 21.8 ms | 10.90 | 2.34x |
| 50 | 5 | 21.1 ms | 4.22 | 53.5 ms | 10.70 | 2.54x |
| 100 | 10 | 33.3 ms | 3.33 | 100.2 ms | 10.02 | 3.01x |
| 1000 | 100 | 392.3 ms | 3.92 | 1,066.0 ms | 10.66 | 2.72x |

Per-round-trip cost is flat from 1 to 100 batches, with no intercept worth naming.

### Concurrency — WS scales, HTTP breaks at two readers

100 keys, one connection per worker, no pipelining:

| workers | WS p50 | WS throughput | WS failed | HTTP p50 | HTTP throughput | HTTP failed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.2 ms | 29.0 r/s | 0 | 105.2 ms | 9.5 r/s | 0 / 200 |
| 2 | 44.8 ms | 44.3 r/s | 0 | 109.1 ms | 4.9 r/s | **348 / 400 (87%)** |
| 4 | 42.4 ms | 94.0 r/s | 0 | 115.1 ms | 10.3 r/s | **672 / 800 (84%)** |
| 8 | 45.2 ms | 174.5 r/s | 0 | 128.4 ms | 19.4 r/s | **1,323 / 1,600 (83%)** |
| 16 | 55.2 ms | **280.9 r/s** | 0 | — | — | — |

**WebSocket scales across the whole ladder with zero failures at every rung** — 9.7x in throughput
while p50 moves only 33 to 55 ms. HTTP's knee is two workers: from there it completes 13-17% of its
reads, so the figures to read on its side of the table are throughput and completion.

Across all WebSocket arms: **36 arms, 1,993,760 reads, 0 failures, 0 value mismatches.**

### Sustained comparisons — an hour or more per arm

The arms above are minutes. These are hours, and they are where the tail shows up.

**Unbounded, ten connections, one hour each** (closed-loop, 2026-10-17, driving the pool
directly rather than `read_feature_view`):

| | WS | HTTP/1.1 |
|---|---:|---:|
| throughput | **2,223 reads/s** | 1,162 reads/s |
| p50 | **3.6 ms** | 8.2 ms |
| success | 100% | 100% |

**1.9x the throughput at 2.3x lower latency**, both arms completing every read. Spread over ten
processes rather than ten threads, the same transport holds **p99 5.2 ms** — see
[Threads per process](#threads-per-process-about-four-then-add-processes).

**Constant 2,000 req/s offered for an hour, 100 readers** — matched offered rate, so delivered
work is comparable (2026-10-17, pool layer):

| | WS | HTTP/1.1 |
|---|---:|---:|
| reads | 7,192,742 | 6,998,696 |
| achieved | 99.9% | 97.2% |
| p50 / p90 | **3.7 / 4.5 ms** | 9.1 / 11.9 ms |
| p99 / p99.9 | **5.2 / 8.3 ms** | **186.0 / 340.7 ms** |
| server p50 / p99 | 1 / 2 ms | 1 / 2 ms |

**2.5x at the median and 36x at p99.** Server time is identical on both transports, so the whole
gap is transport. This is the widest margin in any measurement here, and it is a tail margin:
HTTP's median is fine and its p99 is 186 ms.

**Thirty-two connections, sixteen processes, one hour each** — the only sustained *end-to-end*
concurrent comparison, through `read_feature_view` (2026-10-21):

| | WS | HTTP |
|---|---:|---:|
| throughput | **7,403 reads/s** | 2,298 reads/s |
| p50 | **4.18 ms** | 10.57 ms |
| reads | **26,653,603** | 8,287,521 |
| failed | **0** | **309,690 (3.6%)** |
| value mismatches | 0 | 0 |

**3.2x the throughput, and zero failures against 309,690.** Throughput and completion are the
figures to quote at this concurrency: WebSocket delivered every one of its 26.7 M reads.

### Pool size is load-bearing

Same 16 workers, same server, only the pool changed: **pool 16 gives 55.7 ms p50; pool 4 gives
187.4 ms.** That 3.4x is queueing for a connection, charged to the request and reported as service
time. Raise workers and connections together or not at all.

This is why the default is 16 rather than a smaller number a caller would have to discover and
raise. A pool opens a connection only when every one it holds is busy, so the ceiling costs no
latency until concurrency reaches it — a four-thread caller measures the same 4.6 ms p50 against a
ceiling of 4 or of 32 — while a ceiling below the caller's concurrency queues, and that wait is
charged to the request. Setting it where both measured curves flatten means the common case needs
no tuning; a caller who would rather spend fewer connections lowers it.

### Client pipelining — the largest available win, with no server change

WS, 1 worker, 100 keys. Depth collapses network round trips, so it pays even against a sequential
server:

| depth | chunks per read | p50 | throughput | vs sequential |
|---:|---:|---:|---:|---:|
| 1 | 10 | 40.9 ms | 26.2 r/s | — |
| 2 | 5 | 28.3 ms | 34.8 r/s | 1.45x |
| **4** | 3 | **19.8 ms** | **49.8 r/s** | **2.07x** |
| 8 | 2 | 20.2 ms | 47.7 r/s | 2.02x |

**Depth and concurrency overlap rather than compound.** At 8 workers the same sweep gives only
1.53x (45.7 to 27.8 ms), because both work by keeping more requests in flight and so compete for
the same headroom. Best point measured: depth 8 at 16 workers, **363.2 reads/s at 41.7 ms p50**.

**Replicated on a later deployment, and it went further.** The same three experiments were re-run
through `read_feature_view` on 2026-10-21, twice -- once at reduced and once at full sample sizes
-- both `ALL_OK, 0 mismatches, http_fallback_requests = 0`. One worker, 100 keys:

| depth | p50 above | p50 replicated | throughput replicated |
|---:|---:|---:|---:|
| 1 | 40.9 ms | 46.2 ms | 21.5 reads/s |
| 2 | 28.3 ms | 20.7 ms | 47.7 reads/s |
| 4 | 19.8 ms | 12.1 ms | 81.5 reads/s |
| 8 | 20.2 ms | **8.3 ms** | 116.4 reads/s |
| 16 | — | **5.7 ms** | **168.6 reads/s** |

Depth 1 agrees within run-to-run noise, which is what makes the rest comparable. From there the
replication diverges: depth 8 measured **8.3 ms against the 20.2 ms above, and depth 16 reached
5.7 ms with 7.8x the throughput of depth 1.** The two runs are different days and different
service images, and the second is consistent with a server executing pipelined requests
concurrently rather than serialising them -- but `WS_MAX_INFLIGHT_PER_CONN` was never read off
that image, so that reading is inference. Both ladders are shown because neither supersedes the
other: the sequential one is the conservative floor, the replication is what a concurrent server
appears to give.

**Depth composes across processes.** Eight processes of four threads, 100 keys per read, same-day
control at depth 1 (2026-10-21):

| depth | reads/s | **keys/s** | p50 | p99 | reads | mismatches |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 855 | 85,530 | 36.9 ms | 52.4 ms | 1,026,348 | 0 |
| 8 | 2,167 | 216,720 | 13.7 ms | 28.8 ms | 5,201,799 | 0 |
| **16** | **2,629** | **262,910** | **11.4 ms** | **23.5 ms** | 6,309,844 | 0 |

Depth 16 delivers **3.1x the reads and a third of the latency** of depth 1 at the same shape, and
it keeps improving where a single worker's ladder had flattened -- better p50 *and* better p99 than
depth 8. 12.5 M reads across the three arms, zero mismatches and zero fallbacks.

**Batch size pays even without depth.** 100 keys at depth 1 delivers 85,530 keys/s against 68,780
for 10 keys at depth 1 in the same eight-process shape -- 1.24x, from nothing but fewer, fatter
requests amortising the per-read client cost.

**None of this applies at 10 keys or fewer.** One request per read leaves nothing to pipeline, so
depth is irrelevant below the batching boundary, and the sustained arms above ran at depth 1.

This also establishes the server is sequential on this deployment: at depth 4 the measured 19.8 ms
matches the ~21 ms predicted for a sequential server, against ~11 ms for a concurrent one.

### Longevity

| soak | config | duration | reads | queries | p50 | failed | mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| depth 1 | 8 workers, 100 keys | 73 min | 720,000 | **7,200,000** | 47.5 ms | **0** | **0** |
| server age-out | 8 workers, client age 1800 s | ~27 min | — | — | 48.4 ms | **0** | **0** |

`client_wire` held **3.8 ms for 73 minutes** — no drift under sustained load. A value is asserted
on every read, so this is a correctness soak and not only a throughput one.

---

## Known gaps

**Unverified, worth doing:**

- `cProfile` at one thread puts ~60% of a read's client-side cost in
  `rows_to_pandas_for_postgres_online`. Building the frame from columns rather than per-row Series
  would raise the per-process ceiling with no change required of a caller.
- `websocket-client` revalidates UTF-8 on every received frame (~143 us per read) before the payload
  is parsed as JSON, which validates it again. `skip_utf8_validation=True` removes the duplicate.
- Pipelining above depth 16, and the interaction between depth and pool size.
- Many independent client processes against one deployment: measured to sixteen, and the fleet's
  connection cap is ~1,024 per instance.
- `MAX_ENTITIES_PER_QUERY` is a dynamic server parameter at 10. Raising it would turn a 100-key read
  from ten requests into one, which the batch-size result suggests is the largest remaining lever.

**Known behaviour, not bugs:**

- Depth pays only above ten keys per read: at ten or fewer, a read is already one request and there
  is nothing to send ahead.
- Every latency here is per **read**. The server's own figures are per **request**, so the two
  coincide at ten keys or fewer and differ by `ceil(keys / 10)` above that.
- Connections are opened as read concurrency needs them, so a process holding a thousand threads
  holds four connections if only four are ever mid-read at once.
- `WS_MAX_INFLIGHT_PER_CONN` is read once per connection, so a redeploy that changes it leaves
  existing and new connections at different depths until they recycle.
