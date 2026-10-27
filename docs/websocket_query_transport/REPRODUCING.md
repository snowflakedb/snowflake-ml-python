# Reproducing the WebSocket client's performance figures

Every table in [`WEBSOCKET_CLIENT_DESIGN.md`](WEBSOCKET_CLIENT_DESIGN.md#performance) comes from one
of the arms below. Each is the same harness —
[`tests/perf/feature_store/ws_stress_e2e.py`](../../tests/perf/feature_store/ws_stress_e2e.py) —
driving `fs.read_feature_view(store_type=ONLINE)`, so a figure covers the public API including its
batching loop, not a transport call in isolation.

## Setup, once

```bash
# in-region host, in the same region as the deployment
export PYTHONPATH=$PWD
export SNOWFLAKE_PAT=<programmatic access token>

# Where the harnesses create their throwaway source table and feature view. Every arm below
# needs these three; pass them as --connection/--database/--warehouse instead if you prefer.
export SNOWFLAKE_CONNECTION_NAME=<your-connection>   # from ~/.snowflake/connections.toml
export SNOWFLAKE_DATABASE=<your-database>
export SNOWFLAKE_WAREHOUSE=<your-warehouse>
export SNOWFLAKE_SCHEMA=PUBLIC                        # optional, this is the default

H=tests/perf/feature_store/ws_stress_e2e.py
```

The figures below were produced against one particular account, database and warehouse; the
harnesses take yours. Nothing about the shape of a run depends on which — only that the host is
in the same region as the deployment, since from elsewhere most of every latency figure is WAN.

Resume the warehouse before the first arm — the harness seeds through it:

```sql
ALTER WAREHOUSE <wh> RESUME IF SUSPENDED;
SHOW WAREHOUSES LIKE '<wh>';     -- wait for STARTED
```

**Why `--keys-per-read 10` in most arms.** `_MAX_QUERY_REQUEST_ROWS = 10`, so a read of ten keys is
one request and one round trip. That is the only point where a read and a request are the same
thing, and therefore the only point directly comparable to the server's own per-request figures.
Arms that exercise pipelining use 100 keys, because depth needs more than one request to work with.

## One round trip — the quotable pair

The cleanest comparison in the set: one connection, one request per read, an hour per transport,
both arms completing their work.

```bash
python -u $H --transport ws   --workers 1 --pool-size 1 --keys-per-read 10 \
    --seconds 3600 --rows 2000 --warmup 20 --tag WS1T
python -u $H --transport http --workers 1 --pool-size 1 --keys-per-read 10 \
    --seconds 3600 --rows 2000 --warmup 20 --tag HTTP1T
```

| | WS | HTTP |
|---|---:|---:|
| p50 | **3.9 ms** | 9.6 ms |
| p99 | **5.2 ms** | 13.4 ms |
| throughput | **257.5 reads/s** | 101.7 reads/s |
| reads | 926,994 | 366,298 |

**2.46x on latency, 2.53x on throughput.** Build any per-request cost model from this pair.

## Latency against batch count

One worker, 1,000 reads per arm. Establishes that per-round-trip cost is flat, so a read's latency
is `ceil(keys / 10)` round trips and nothing else.

```bash
for k in 10 20 50 100 1000; do
  python -u $H --transport both --workers 1 --pool-size 1 --keys-per-read $k \
      --reads-per-worker 1000 --rows 2000 --warmup 10 --tag BATCH$k
done
```

## The concurrency ladder

100 keys, one connection per worker, no pipelining. Shows WebSocket scaling 9.7x in throughput
across the ladder with zero failures at every rung.

```bash
for w in 1 2 4 8 16; do
  python -u $H --transport both --workers $w --pool-size $w --keys-per-read 100 \
      --reads-per-worker 100 --rows 2000 --warmup 10 --tag CONC$w
done
```

## Thirty-two connections, sixteen processes — the sustained concurrent pair

Sixteen processes of two threads each, an hour per transport. The only sustained end-to-end
concurrent comparison, and the best WebSocket latency at this connection count.

```bash
for t in ws http; do
  for i in $(seq 0 15); do
    setsid nohup python -u $H --transport $t --workers 2 --pool-size 2 --keys-per-read 10 \
        --seconds 3600 --rows 2000 --warmup 20 --tag M7${t}16p${i} \
        > ${t}_16p2t_p$i.log 2>&1 < /dev/null &
    sleep 8
  done
  wait
done
```

| | WS | HTTP |
|---|---:|---:|
| throughput | **7,403 reads/s** | 2,298 reads/s |
| p50 | **4.18 ms** | 10.57 ms |
| reads | **26,653,603** | 8,287,521 |
| value mismatches | 0 | 0 |

Aggregate across the sixteen logs; each process reports its own percentiles.

## Threads per process — where one process stops scaling

One process, pool matched to threads, 120 s per point.

```bash
for n in 1 2 4 8 16 32; do
  python -u $H --transport ws --workers $n --pool-size $n --keys-per-read 10 \
      --seconds 120 --rows 2000 --warmup 20 --tag SWP$n
done
```

| threads | reads/s | p50 | mean |
|---:|---:|---:|---:|
| 1 | 262 | 3.4 ms | 3.8 ms |
| 2 | 477 | 4.1 ms | 4.2 ms |
| 4 | 819 | 4.6 ms | 4.8 ms |
| 8 | 1,222 | 6.2 ms | 6.5 ms |
| 16 | 1,235 | 12.1 ms | 12.9 ms |
| 32 | 1,216 | 23.7 ms | 26.3 ms |

Throughput plateaus near eight threads, and past that latency rises in proportion to the thread
count while throughput does not move — `read_feature_view` costs ~810 µs of Python per read, which
serializes, so one process tops out around 1,235 reads/s however the concurrency is arranged.
Concurrency beyond that belongs in another process — see the next arm.

Note what this arm **cannot** establish: pool size moves with the thread count at every point, so
it measures caller concurrency, not the pool. The pool is isolated in the two arms below.

## Pool size at fixed concurrency — the knee

Sixteen threads held constant, pool varied. Ten keys per read, so one read is one request.

```bash
for c in 4 8 16 32; do
  python -u $H --transport ws --workers 16 --pool-size $c --keys-per-read 10 \
      --seconds 600 --rows 2000 --warmup 10 --tag OPTP$c
done
```

| pool | reads/s | mean | wire/req | dials | reads/conn |
|---:|---:|---:|---:|---:|---:|
| 4 | 869.3 | 18.4 ms | 3.8 ms | 11 | 47,419 |
| **8** | **1,194.9** | **13.4 ms** | 5.1 ms | 19 | 37,737 |
| 16 | 1,273.2 | 12.5 ms | 9.2 ms | 52 | 14,691 |
| 32 | 1,240.1 | 12.9 ms | 12.0 ms | 122 | 6,099 |

The curve is peaked rather than flat: 4 → 8 gains 37%, 8 → 16 gains 6.6%, and 32 lands below 16.
Per-request wire time inflates 3.8 → 12.0 ms across the sweep, so added connections degrade each
other's service time and cancel their own gain. Dial count — each dial an authentication — scales
with the pool, which is what a high cap actually costs.

`pool 16` at 16 threads was also measured in the 120 s sweep above at 1,235 reads/s and 12.9 ms
mean. Reproduced here at 600 s it gives 1,273 and 12.5 ms, so **run-to-run spread is about ±3%** —
the resolution of every figure in this document.

## The same effect from the other side — pool fixed at 4, threads varied

The sweep above holds threads and varies the pool. This holds the pool at the shipped default and
varies the caller instead, which is the shape a caller actually controls.

```bash
for w in 4 8 16; do
  python -u $H --transport ws --workers $w --pool-size 4 --keys-per-read 10 \
      --seconds 1200 --rows 2000 --warmup 10 --tag OS${w}P4K10
  python -u $H --transport ws --workers $w --pool-size 4 --keys-per-read 100 \
      --seconds 1200 --rows 2000 --warmup 10 --tag OS${w}P4K100
done
```

| threads | 10 keys | | 100 keys | |
|---:|---:|---:|---:|---:|
| | reads/s | p50 | reads/s | p50 |
| 4 | 819.0 | 4.6 ms | ~107 | 36.9 ms |
| 8 | 864.1 | 5.9 ms | 105.9 | 69.5 ms |
| 16 | 869.3 | 6.4 ms | 111.7 | 140.1 ms |

Throughput is flat — 819 to 869 across a 4x increase in callers — while latency rises in step with
the oversubscription factor. Four connections are already saturated at four concurrent reads, so
every extra caller queues rather than adding work: at 100 keys the p50 goes 36.9 → 69.5 → 140.1 ms
for no throughput at all. Measured queue wait confirms where the time goes — 1.47 ms at 8 threads
and 2.27 ms at 16, against a 3.8 ms wire.

This is the failure mode an undersized pool actually produces. It does not refuse work or report
an error; it charges the wait to the caller's latency, so outgrowing the pool looks like a
transport slowdown rather than a capacity limit.

## Pool size at 100 keys — the knee moves with batch size

A read above ten keys splits into one request per ten rows, which multiplies the time a connection
is held but not the per-call Python cost, much of which is fixed and amortizes across the batch.

```bash
for c in 4 8 16 32; do
  python -u $H --transport ws --workers 16 --pool-size $c --keys-per-read 100 \
      --seconds 1200 --rows 2000 --warmup 10 --tag WT16P$c
done
```

| pool | reads/s | keys/s | req/s | p50 | p99 | hold per read | failed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 111.7 | 11,170 | 1,117 | 140.1 ms | 367.3 ms | 35.8 ms | 0 |
| 8 | 173.8 | 17,380 | 1,738 | 83.4 ms | 214.6 ms | 46.0 ms | 0 |
| **16** | **288.1** | **28,810** | **2,881** | **53.2 ms** | **96.5 ms** | 55.5 ms | 0 |
| 32 | 310.4 | 31,040 | 3,104 | 50.4 ms | 67.4 ms | — | 0 |

2.58x the throughput and 2.63x lower latency from pool size alone between 4 and 16, at an
identical caller shape. The gains are 56% from 4 to 8, 66% from 8 to 16, then **7.7% from 16 to
32** — so the curve flattens at sixteen here, as it peaks at sixteen in the ten-key sweep. Pool 8
reaches 94% of peak at ten keys and only 60% at a hundred.

Writing `S` for the time a connection is held per read and `D` for the serialized Python demand
per read, `c* = S/D` gives about 5.7 at ten keys and 10.3 at a hundred. Treat that as a **floor**
on the useful width rather than the knee: `S` is not constant across a sweep, inflating
35.8 → 46.0 → 55.5 ms as connections are added, so extrapolating `c/S` from one measured point
overstates every larger pool. The table reports measured points at each size.

### Batching amortizes the per-call cost

Comparing the two sweeps in `keys/s` rather than `reads/s` — the only fair unit across different
key counts — shows the batch itself is a throughput win:

| pool 16 | keys/s | req/s | p50 per read |
|---|---:|---:|---:|
| 10 keys/read | 12,732 | 1,273 | 6.3 ms |
| 100 keys/read | **28,810** | **2,881** | 53.2 ms |

**2.3x the keys per second** at the same pool, threads and transport, because the fixed per-call
cost amortizes: 810 us of Python for a one-request read against 347 us per request for a
ten-request one. The trade is per-call latency, 6.3 ms against 53.2 ms. Small reads for latency,
large reads for key throughput.

## Single-request reads under concurrency — both transports, every pool size

Sixteen threads, ten keys per read, so a read is one round trip on either transport. The HTTP arm
has no pool to set: sixteen threads means sixteen connections.

```bash
for c in 4 8 16; do
  python -u $H --transport ws --workers 16 --pool-size $c --keys-per-read 10 \
      --seconds 600 --rows 2000 --warmup 10 --tag OPTP$c
done
python -u $H --transport http --workers 16 --pool-size 16 --keys-per-read 10 \
    --seconds 600 --rows 2000 --warmup 10 --tag OPTHTTP
```

| transport | reads/s | p50 | mean | failed reads |
|---|---:|---:|---:|---:|
| WebSocket, pool 4 | 869.3 | **6.4 ms** | 18.4 ms | 0 of 521,613 |
| **WebSocket, pool 8** | **1,194.9** | **6.3 ms** | **13.4 ms** | **0 of 717,005** |
| WebSocket, pool 16 | 1,273.2 | 11.2 ms | 12.5 ms | 0 of 763,926 |
| HTTP, 16 connections | 613.2 | 23.1 ms | 25.0 ms | 1,903 of 369,852 |

WebSocket leads at every pool size, the throttled one included: 1.4x to 2.1x the throughput and
about 3.6x the median at pool 4 or 8. HTTP loses 0.51% of its reads here against 46% at a hundred
keys, which locates its failure in request *rate* rather than in concurrency itself — ten round
trips per read is what exhausts its sockets.

## Batched reads under concurrency — where HTTP stops completing

The same caller shape over both transports. A hundred-key read is ten round trips, so sixteen
threads ask for about a hundred and sixty connection slots at once.

```bash
for t in ws http; do
  python -u $H --transport $t --workers 16 --pool-size 16 --keys-per-read 100 \
      --seconds 1200 --rows 2000 --warmup 10 --tag M16$t
done
```

| transport | completed | p50 | failed reads |
|---|---:|---:|---:|
| WebSocket, pool 4 | 111.7 /s | 140.1 ms | 0 of 134,033 |
| **WebSocket, pool 16** | **288.1 /s** | **53.2 ms** | **0 of 345,717** |
| HTTP, 16 connections | 55.5 /s | 150.5 ms | 56,224 of 122,828 |

WebSocket leads on throughput and latency, and completes every read; the HTTP arm loses 46% of its
reads to `Online Service is unreachable: [Errno 11] Resource temporarily unavailable`, the local
socket pool running dry. Read the medians with that in mind — HTTP's covers only the reads that
finished, so completed work is the comparison that holds. WebSocket leads even in the row where its
pool is deliberately throttled to four connections.

## A high cap is cheap in latency and not in connections

Four threads — never more than four reads in flight — against a cap of 32.

```bash
python -u $H --transport ws --workers 4 --pool-size 32 --keys-per-read 10 \
    --seconds 1200 --rows 2000 --warmup 10 --tag CAP32T4
```

| cap | reads/s | p50 | held at end | dials | reads/conn | max read |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 819.0 | 4.6 ms | 4 | ~26 | 39,882 | 1,441 ms |
| 32 | 808.5 | 4.6 ms | 32 | 236 | 4,111 | 55,326 ms |

Identical median and throughput, 8x the connections and ~10x the handshakes. The pool opens a
connection only when every one it holds is busy, but it **ratchets**: across twenty minutes the
transient all-busy moments accumulate and the pool reaches its cap without ever shrinking, so a
four-thread caller finished the run holding all 32. Growth is lazy per selection, not
self-limiting over time — which is the cost of defaulting high, and a connection-budget cost
rather than a latency one.

## Processes compose — the same connections, split three ways

Thirty-two connections arranged three different ways, an hour each. Establishes that processes scale
almost linearly and that read concurrency, not connection count, is what a process is limited by.

```bash
# 4 threads x 8 processes
for i in $(seq 0 7); do
  setsid nohup python -u $H --transport ws --workers 4 --pool-size 4 --keys-per-read 10 \
      --seconds 3600 --rows 2000 --warmup 20 --tag MC1H$i > proc$i.log 2>&1 < /dev/null &
  sleep 12      # stagger: concurrent feature-view creations contend on metadata
done; wait
```

| shape | connections | reads/s | p50 |
|---|---:|---:|---:|
| 2 threads x 16 processes | 32 | **7,403** | **4.18 ms** |
| 4 threads x 8 processes | 32 | 6,878 | 4.45 ms |
| 32 threads x 1 process | 32 | 1,182 | 26.9 ms |

Predicted from the per-process figures above at 6,880 and 7,600; measured 6,878 and 7,403.

## Pipelining — depth across processes

100 keys per read, so a read is ten requests and depth has something to send ahead. Eight processes
of four threads, with a same-day depth-1 control.

```bash
for d in 1 8 16; do
  for i in $(seq 0 7); do
    setsid nohup python -u $H --transport ws --workers 4 --pool-size 4 \
        --keys-per-read 100 --pipeline-depth $d --seconds 2400 \
        --rows 2000 --warmup 10 --tag PD${d}P${i} > d${d}_p$i.log 2>&1 < /dev/null &
    sleep 10
  done
  wait
done
```

| depth | reads/s | **keys/s** | p50 | p99 |
|---:|---:|---:|---:|---:|
| 1 | 855 | 85,530 | 36.9 ms | 52.4 ms |
| 8 | 2,167 | 216,720 | 13.7 ms | 28.8 ms |
| **16** | **2,629** | **262,910** | **11.4 ms** | **23.5 ms** |

**3.82x the keys per second of the ten-key arm**, over 12.5 M reads with zero value mismatches.
Depth multiplies requests in flight per connection, so four connections at depth 16 carry sixty-four
concurrent requests. Compare arms by **keys/s** whenever the key count differs; `reads/s` is only
comparable at a fixed `--keys-per-read`.

## Longevity

Two hours of continuous traffic, every value asserted against what was ingested.

```bash
python -u $H --transport ws --workers 32 --pool-size 32 --keys-per-read 10 \
    --seconds 7200 --rows 2000 --warmup 20 --tag SOAK
```

8,511,130 reads, **zero failures, zero value mismatches, zero HTTP fallbacks**, and
`ws_requests == reads` exactly — every read served over WebSocket.

## Reading the output

Each arm prints, per process:

```text
reads=…  keys/read=…  workers=…  pool=…  depth=…
wall=…s  throughput=… reads/s
latency per read: n=… p50=… p95=… p99=… max=… mean=… (ms)
transport: ws_requests=… http_fallback_requests=…
value mismatches: …   failed reads: …
connections: dialed=… held_at_end=… reads_per_connection=…
latency decomposition (each layer contains the next):
  read_helper / transport / pool_total / pool_select / client_wire
```

Three of those are the ones to check first:

- **`ws_requests` against `reads`** — equal means one request per read, so the arm stayed inside the
  ten-row boundary. At 100 keys expect ten times the reads.
- **`http_fallback_requests = 0`** — the arm was served over WebSocket throughout.
- **`value mismatches = 0`** — every returned value matched what was ingested, so an all-null frame
  would fail rather than pass.

The decomposition nests: each layer wholly contains the next, so subtracting adjacent rows gives
that layer's own cost. `pool_total − client_wire` is time spent waiting for a connection.

## Two things the output depends on

**One arm at a time.** Concurrent arms contend on one deployment, and neither result is then
attributable. The multi-process arms are the exception: that contention is the measurement.

**`setsid nohup`, and a marker file to detect completion.** A bare `nohup` races the ssh session
closing. And guard on a file the runner touches when done rather than on `pgrep -f ws_stress_e2e` —
that pattern matches the guard's own command line, so it never terminates.
