"""Latency breakdown for ``fs.read_feature_view(..., store_type=ONLINE)`` on Postgres.

Requires ``SNOWFLAKE_PAT``. Prints p50/p95/max/mean per hot-path phase across materialization modes.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
import unittest
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from absl.testing import absltest
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store import (
    online_service as os_mod,
    online_service_http_client as os_http_mod,
)
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
)

logger = logging.getLogger(__name__)

_ITERATIONS_DEFAULT = 50
_WARMUP_DEFAULT = 10


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@contextmanager
def _patch_attr(target: Any, attr: str, bucket: list[float]) -> Iterator[None]:
    """Time every call to ``target.attr`` and record elapsed ms into ``bucket``."""
    original = getattr(target, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            bucket.append((time.perf_counter() - t0) * 1000.0)

    setattr(target, attr, wrapped)
    try:
        yield
    finally:
        setattr(target, attr, original)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ordered = sorted(xs)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def _format_report(label: str, buckets: dict[str, list[float]]) -> str:
    lines = [
        f"\n=== {label} ===",
        f"{'phase':<26} {'n':>4}  {'p50':>8}  {'p95':>8}  {'max':>8}  {'mean':>8}   (ms)",
        "-" * 80,
    ]
    for name, xs in buckets.items():
        if not xs:
            lines.append(f"{name:<26} {0:>4}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}")
            continue
        lines.append(
            f"{name:<26} {len(xs):>4}  "
            f"{_percentile(xs, 50):>8.2f}  {_percentile(xs, 95):>8.2f}  "
            f"{max(xs):>8.2f}  {statistics.fmean(xs):>8.2f}"
        )
    return "\n".join(lines)


def _run_measurement(
    fs: FeatureStore,
    fv: Any,
    keys: list[list[str]],
    *,
    iterations: int,
    warmup: int,
    materialize: str,
) -> dict[str, list[float]]:
    """Run the harness and return per-phase timing buckets (milliseconds)."""
    buckets: dict[str, list[float]] = {
        "total": [],
        "read_feature_view": [],
        "wh_get_current": [],
        "wh_use": [],
        "online_http_post_json": [],
        "read_pg_online_helper": [],
        "create_dataframe": [],
        "materialize": [],
    }

    session = fs._session
    use_as_pandas = materialize == "as_pandas"

    with _patch_attr(session, "get_current_warehouse", buckets["wh_get_current"]), _patch_attr(
        session, "use_warehouse", buckets["wh_use"]
    ), _patch_attr(session, "create_dataframe", buckets["create_dataframe"]), _patch_attr(
        os_http_mod.OnlineServiceHttpClient, "post_json", buckets["online_http_post_json"]
    ), _patch_attr(
        os_mod, "read_postgres_online_features", buckets["read_pg_online_helper"]
    ):
        for _ in range(warmup):
            out = fs.read_feature_view(fv, keys=keys, store_type=StoreType.ONLINE, as_pandas=use_as_pandas)
            if not use_as_pandas:
                if materialize == "to_pandas":
                    out.to_pandas()
                elif materialize == "collect":
                    out.collect()

        for b in buckets.values():
            b.clear()

        for _ in range(iterations):
            t_start = time.perf_counter()

            t0 = time.perf_counter()
            out = fs.read_feature_view(fv, keys=keys, store_type=StoreType.ONLINE, as_pandas=use_as_pandas)
            buckets["read_feature_view"].append((time.perf_counter() - t0) * 1000.0)

            if not use_as_pandas and materialize != "none":
                t1 = time.perf_counter()
                if materialize == "to_pandas":
                    out.to_pandas()
                else:
                    out.collect()
                buckets["materialize"].append((time.perf_counter() - t1) * 1000.0)

            buckets["total"].append((time.perf_counter() - t_start) * 1000.0)

    return buckets


class PostgresOnlineReadPerfIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Benchmark ``read_feature_view(..., store_type=ONLINE)`` on a Postgres FV."""

    def _register_minimal_fv(self) -> tuple[str, str, str]:
        """Create a single-row Postgres batch FV; returns (fv_name, version, key)."""
        s = uuid.uuid4().hex[:8]
        fv_name = f"PERF_BATCH_ONLINE_FV_{s}"
        key = f"U_PERF_{s}"

        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.PERF_BATCH_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {src_table} VALUES
            ({key!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), 42.0)
            """
        ).collect()

        feature_df = self._session.table(src_table)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv, "v1")
        return fv_name, "v1", key

    def _wait_until_online_read_returns_rows(self, fv_name: str, version: str, key: str, timeout: float = 300.0) -> Any:
        """Poll online read until a row is returned; returns the live FV handle."""
        deadline = time.time() + timeout
        last_err: str | None = None
        while time.time() < deadline:
            try:
                fv_live = self.fs.get_feature_view(fv_name, version)
                # Postgres online reads default to pandas; use len() for a backend-agnostic non-empty check.
                out = self.fs.read_feature_view(fv_live, keys=[[key]], store_type=StoreType.ONLINE)
                if len(out) > 0:
                    return fv_live
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(10)
        self.fail(
            f"Online read for {fv_name}/{version} did not return rows within " f"{timeout}s; last_err={last_err!r}"
        )

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for Postgres online read (Online Service Query API).",
    )
    def test_postgres_online_read_latency_breakdown(self) -> None:
        iterations = _env_int("PERF_ITERATIONS", _ITERATIONS_DEFAULT)
        warmup = _env_int("PERF_WARMUP", _WARMUP_DEFAULT)

        fv_name, version, key = self._register_minimal_fv()
        fv_live = self._wait_until_online_read_returns_rows(fv_name, version, key)

        # prime compute before measurement
        self._session.sql("SELECT 1").collect()

        configs = [
            ("materialize=none", "none"),
            ("materialize=collect", "collect"),
            ("materialize=to_pandas", "to_pandas"),
            ("materialize=as_pandas", "as_pandas"),
        ]

        reports: list[str] = []
        for label, mode in configs:
            buckets = _run_measurement(
                self.fs,
                fv_live,
                keys=[[key]],
                iterations=iterations,
                warmup=warmup,
                materialize=mode,
            )
            reports.append(_format_report(label, buckets))
            if mode == "as_pandas":
                self.assertEqual(
                    len(buckets["create_dataframe"]),
                    0,
                    "as_pandas=True must skip Session.create_dataframe; the local-build fast path regressed.",
                )

        # Warehouse-mismatch mode: verify Postgres online reads still skip USE WAREHOUSE.
        default_wh = self.fs._default_warehouse
        try:
            current_wh = self._session.get_current_warehouse()
        except Exception:  # pragma: no cover - best-effort introspection
            current_wh = None
        alt_wh = None
        if current_wh and current_wh != str(default_wh):
            alt_wh = current_wh
        else:
            # Try to find any other warehouse the caller can USE. If we cannot, skip the extra mode.
            try:
                rows = self._session.sql("SHOW WAREHOUSES").collect()
                for row in rows:
                    candidate = row["name"] if "name" in row.as_dict() else row[0]
                    if candidate and candidate != str(default_wh):
                        alt_wh = candidate
                        break
            except Exception:  # pragma: no cover - permission / env dependent
                alt_wh = None

        if alt_wh is not None:
            original_wh = current_wh
            try:
                self._session.use_warehouse(alt_wh)
                buckets = _run_measurement(
                    self.fs,
                    fv_live,
                    keys=[[key]],
                    iterations=iterations,
                    warmup=warmup,
                    materialize="as_pandas",
                )
                reports.append(_format_report(f"wh_mismatch (alt={alt_wh}) materialize=as_pandas", buckets))
                self.assertEqual(
                    len(buckets["wh_use"]),
                    0,
                    "Postgres online read must not call session.use_warehouse even when the session "
                    "warehouse differs from the FS default. Fix C regressed.",
                )
            finally:
                if original_wh is not None:
                    try:
                        self._session.use_warehouse(original_wh)
                    except Exception:  # pragma: no cover - best-effort restore
                        logger.warning("Failed to restore original session warehouse %r", original_wh)

        # Print as a single block so bazel's streamed output shows it atomically.
        banner = "=" * 80
        out = "\n".join(
            [
                "",
                banner,
                f"Postgres online read perf breakdown  (iterations={iterations}, warmup={warmup})",
                banner,
                *reports,
                banner,
            ]
        )
        print(out)
        logger.info(out)


if __name__ == "__main__":
    absltest.main()
