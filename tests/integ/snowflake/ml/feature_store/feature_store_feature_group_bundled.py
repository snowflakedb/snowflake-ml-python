"""E2E integration tests for ``FeatureGroup`` over Postgres-backed OFTs (bundled).

This file contributes its ``FeatureGroupIntegTest`` class to the
``feature_store_spec_oft_bundle_test`` runner, which provisions one shared
per-shard Online Feature Store in ``setUpModule`` and discovers test classes
declared in sibling ``*_bundled.py`` files via ``__module__`` reassignment.

Coverage:

- Register a ``FeatureGroup`` over two Postgres-backed FeatureViews and assert
  the OFT (``<name>$<version>$ONLINE``) shows up tagged as ``FEATURE_GROUP``.
- ``get_feature_group`` reconstructs the original (name, desc, auto_prefix,
  source refs incl. slice + alias).
- ``list_feature_groups`` returns at least one row containing our FG.
- Heterogeneous-key sources produce an OFT keyed on the ordered superset.
- Registering a FG whose source FV is on a non-Postgres / offline store fails
  with the expected ``ValueError`` *before* any side effect.
- ``read_feature_group`` round-trips against the live Postgres OFT.
- ``generate_training_set(feature_group=...)`` joins a spine against the FG
  (with and without PIT timestamps).
- ``delete_feature_group`` is idempotent and removes the FG from listings.

Reuses ``StreamingFeatureViewIntegTestBase`` for the class-scoped Feature Store,
``USER_ID`` entity, and Online Service. The bundle's ``setUpModule`` requires
``SNOWFLAKE_PAT`` to provision the Postgres-backed OFT runtime, e.g.
``bazel test ... --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)``.
FG-with-RTFV ``read_feature_group`` round-trips use the same gate: if
``SNOWFLAKE_PAT`` is set, CI must target an account where that path is
supported (no separate opt-out env).
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    identity_transform,
)

from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_group import FeatureGroup
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewSlice,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType

logger = logging.getLogger(__name__)


def _rtfv_weighted_balance(request_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
    """RTFV ``compute_fn`` for the FG mixed-source round-trip.

    Operates on features only with positional row alignment (no merge on
    entity keys). Mirrors the reference fn in
    ``feature_store_realtime_bundled.py``; reproduced here because py_test
    targets cannot share srcs across files.

    Args:
        request_df: Request payload with ``WEIGHT`` (RequestSource fields only).
        balance_df: Upstream BFV rows with ``BALANCE``, row-aligned with ``request_df``.

    Returns:
        DataFrame with ``WEIGHTED_BALANCE = BALANCE * WEIGHT``, row-aligned.
    """
    weight = request_df["WEIGHT"].astype(float).reset_index(drop=True)
    balance = balance_df["BALANCE"].fillna(0.0).reset_index(drop=True)
    return pd.DataFrame({"WEIGHTED_BALANCE": balance * weight})


def _rtfv_doubled_balance(balance_df: pd.DataFrame) -> pd.DataFrame:
    """RTFV ``compute_fn`` with no RequestSource: doubles the upstream ``BALANCE``.

    Args:
        balance_df: Upstream BFV rows with ``BALANCE``.

    Returns:
        DataFrame with ``DOUBLED_BALANCE = BALANCE * 2``, row-aligned.
    """
    balance = balance_df["BALANCE"].fillna(0.0).reset_index(drop=True)
    return pd.DataFrame({"DOUBLED_BALANCE": balance * 2.0})


def _normalize_column_name(name: str) -> str:
    """Strip SQL quotes and upper-case so quoted ``output_columns`` and unquoted ``df.columns`` compare equal."""
    return name.strip('"').upper()


class FeatureGroupIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """End-to-end FG CRUD against a live Postgres OFT."""

    # OFS catalog is multi-replica / eventually consistent: a freshly registered
    # FG can 404 on one replica while 200 on another. ~30s (6 * 5s) is enough
    # budget for one replica to converge.
    _POST_READINESS_RETRIES = 6
    _POST_READINESS_BACKOFF_SEC = 5.0

    def _register_postgres_fv(
        self, *, suffix: str, store_type: OnlineStoreType = OnlineStoreType.POSTGRES
    ) -> tuple[str, str, str]:
        """Create + register a 1-row Postgres batch FV.

        Args:
            suffix: Short label embedded in the source-table / FV name.
            store_type: Online store backing the FV (negative tests pass
                non-POSTGRES values to exercise rejection paths).

        Returns:
            Tuple of ``(fv_name, src_table_fqn, seeded_user_id)``.
        """
        s = uuid.uuid4().hex[:8]
        fv_name = f"FG_INTEG_FV_{suffix}_{s}"
        seeded_user_id = f"U_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.FG_INTEG_SRC_{suffix}_{s}"

        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT,
                BALANCE FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {src_table} VALUES
            ({seeded_user_id!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), 42.0, 1000.0)
            """
        ).collect()

        feature_df = self._session.table(src_table)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=store_type),
        )
        self.fs.register_feature_view(fv, "v1")
        return fv_name, src_table, seeded_user_id

    def _register_postgres_fv_multikey(self, *, suffix: str, join_keys: list[str]) -> str:
        """Create + register a Postgres FV keyed on multiple join keys. Returns the FV name."""
        s = uuid.uuid4().hex[:8]
        fv_name = f"FG_INTEG_MK_FV_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.FG_INTEG_MK_SRC_{suffix}_{s}"
        entity_name = f"FG_INTEG_MK_ENT_{suffix}_{s}"

        entity = Entity(name=entity_name, join_keys=join_keys, desc="multi-key entity for FG integ")
        self.fs.register_entity(entity)

        key_cols_ddl = ", ".join(f"{k} VARCHAR" for k in join_keys)
        key_value_csv = ", ".join(f"'V_{i}'" for i, _ in enumerate(join_keys))
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                {key_cols_ddl},
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {src_table} VALUES (
                {key_value_csv},
                DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ),
                7.0
            )
            """
        ).collect()

        feature_df = self._session.table(src_table)
        fv = FeatureView(
            name=fv_name,
            entities=[entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv, "v1")
        return fv_name

    def _register_postgres_tiled_sfv(self, *, suffix: str) -> tuple[str, str]:
        """Create + register a tiled streaming FV on Postgres with SUM/MAX/COUNT aggs.

        The FV joins on ``USER_ID`` (matching ``_register_postgres_fv``) so it can
        be combined with batch FVs in a single FeatureGroup. Uses the base class's
        stream-source / backfill-table helpers so the backfill DT is populated by
        the time the FG is registered.

        Args:
            suffix: Short label embedded in the FV / stream / table names.

        Returns:
            Tuple of ``(fv_name, seeded_user_id)``. ``seeded_user_id`` matches a row
            in the backfill table so downstream reads return non-empty data.
        """
        s = uuid.uuid4().hex[:8]
        # StreamSource has a 32-char name limit.
        stream_name = f"FG_TS_{suffix}_{s}"
        fv_name = f"FG_INTEG_TILED_SFV_{suffix}_{s}"
        seeded_user_id = "u1"  # `_create_backfill_table` seeds u1/u2/u3.

        self._make_stream_source(self.fs, stream_name)
        backfill_table = self._create_backfill_table(self.fs, suffix=f"{suffix}_{s}")
        backfill_df = self._session.table(backfill_table)

        stream_config = StreamConfig(
            stream_source=stream_name,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )
        features = [
            Feature.sum("AMOUNT", "1d").alias(f"AMOUNT_SUM_1D_{suffix}"),
            Feature.max("AMOUNT", "1d").alias(f"AMOUNT_MAX_1D_{suffix}"),
            Feature.count("AMOUNT", "1d").alias(f"AMOUNT_COUNT_1D_{suffix}"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.is_streaming)
        self.assertTrue(registered_fv.is_tiled)

        deadline = time.time() + 180.0
        while time.time() < deadline:
            count = self._session.table(registered_fv.fully_qualified_name()).count()
            if count > 0:
                break
            time.sleep(5)

        return fv_name, seeded_user_id

    def _register_postgres_tiled_bfv(self, *, suffix: str) -> tuple[str, str]:
        """Create + register a tiled batch FV on Postgres with SUM/MAX/COUNT aggs.

        Args:
            suffix: Short label embedded in the FV / source-table names.

        Returns:
            Tuple of ``(fv_name, seeded_user_id)``. ``seeded_user_id`` matches a row
            in the source table so downstream reads return non-empty data.
        """
        s = uuid.uuid4().hex[:8]
        fv_name = f"FG_INTEG_TILED_BFV_{suffix}_{s}"
        seeded_user_id = f"U_BTILED_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.FG_BTILED_SRC_{suffix}_{s}"

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
            INSERT INTO {src_table}
            SELECT column1, column2, column3
            FROM VALUES
                (
                    {seeded_user_id!r},
                    DATEADD('hour', 1, DATEADD('day', -1, DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))),
                    10.0
                ),
                (
                    {seeded_user_id!r},
                    DATEADD('hour', 2, DATEADD('day', -1, DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))),
                    20.0
                ),
                (
                    {seeded_user_id!r},
                    DATEADD('hour', 3, DATEADD('day', -1, DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))),
                    30.0
                )
            """
        ).collect()
        feature_df = self._session.table(src_table)

        features = [
            Feature.sum("AMOUNT", "1d").alias(f"AMOUNT_SUM_1D_{suffix}"),
            Feature.max("AMOUNT", "1d").alias(f"AMOUNT_MAX_1D_{suffix}"),
            Feature.count("AMOUNT", "1d").alias(f"AMOUNT_COUNT_1D_{suffix}"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertFalse(registered_fv.is_streaming)
        self.assertTrue(registered_fv.is_tiled)

        deadline = time.time() + 180.0
        while time.time() < deadline:
            count = self._session.table(registered_fv.fully_qualified_name()).count()
            if count > 0:
                break
            time.sleep(5)

        return fv_name, seeded_user_id

    def _wait_until_fg_read_returns_rows(self, fg_live: FeatureGroup, key: str, timeout: float = 600.0) -> None:
        """Poll ``read_feature_group`` until at least one row is returned."""
        deadline = time.time() + timeout
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                pdf = self.fs.read_feature_group(fg_live, keys=[[key]])
                if len(pdf) > 0:
                    return
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "read_feature_group(%s/%s) wait-loop transient error: %s",
                    fg_live.name,
                    fg_live.version,
                    last_err,
                )
            time.sleep(5)
        self.fail(
            f"read_feature_group({fg_live.name}/{fg_live.version}) returned no rows within "
            f"{timeout}s; last_err={last_err!r}"
        )

    def _read_feature_group_with_retry(self, fg_live: FeatureGroup, keys: list[list[Any]]) -> pd.DataFrame:
        """Call :meth:`FeatureStore.read_feature_group` with bounded retry against transient OFS replica skew.

        Args:
            fg_live: Hydrated FeatureGroup to read against.
            keys: Join-key rows (same shape as ``read_feature_group``).

        Returns:
            The pandas DataFrame from the first successful attempt.

        Raises:
            last_err: The last exception observed if every attempt fails.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._POST_READINESS_RETRIES):
            try:
                pdf = self.fs.read_feature_group(fg_live, keys=keys)
                if attempt > 0:
                    logger.info(
                        "read_feature_group succeeded on attempt %d/%d",
                        attempt + 1,
                        self._POST_READINESS_RETRIES,
                    )
                return pdf
            except Exception as e:
                last_err = e
                logger.warning(
                    "read_feature_group retry %d/%d on transient error: %s: %s",
                    attempt + 1,
                    self._POST_READINESS_RETRIES,
                    type(e).__name__,
                    e,
                )
                time.sleep(self._POST_READINESS_BACKOFF_SEC)
        assert last_err is not None
        raise last_err

    def test_feature_group_full_round_trip(self) -> None:
        """register -> get -> list -> delete with a slice + alias on one source."""
        fv1_name, _src1, _key1 = self._register_postgres_fv(suffix="A")
        fv2_name, _src2, _key2 = self._register_postgres_fv(suffix="B")

        fv1 = self.fs.get_feature_view(fv1_name, "v1")
        fv2 = self.fs.get_feature_view(fv2_name, "v1")

        # Upper-case suffix avoids Snowflake's unquoted-identifier case-folding
        # when comparing against ``registered.name``; slice + alias exercises the round-trip.
        fg_name = f"FG_INTEG_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(
            name=fg_name,
            features=[fv1, fv2.slice(["AMOUNT"]).with_name("sender")],
            desc="integ-test FG",
            auto_prefix=True,
        )

        fg_version = "v1"
        registered: Optional[FeatureGroup] = None
        try:
            registered = self.fs.register_feature_group(fg, fg_version)
            self.assertEqual(registered.name, fg_name)
            self.assertIsNotNone(registered.version)
            self.assertEqual(str(registered.version), fg_version)
            self.assertEqual(registered.desc, "integ-test FG")
            self.assertTrue(registered.auto_prefix)
            self.assertEqual(len(registered.features), 2)

            self.assertIsInstance(registered.features[0], FeatureView)
            second = registered.features[1]
            self.assertIsInstance(second, FeatureViewSlice)
            self.assertEqual([n.resolved() for n in second.names], ["AMOUNT"])
            self.assertEqual(second.column_alias, "sender")
            # Mixed-case alias prefix forces SQL-quoting on output column names.
            self.assertIn('"sender_AMOUNT"', registered.output_columns)

            fetched = self.fs.get_feature_group(fg_name, fg_version)
            self.assertEqual(fetched.name, fg_name)
            self.assertEqual(str(fetched.version), fg_version)
            self.assertEqual(fetched.desc, "integ-test FG")
            self.assertEqual(fetched.auto_prefix, True)
            self.assertEqual(fetched.output_columns, registered.output_columns)

            rows = self.fs.list_feature_groups().collect()
            self.assertIn("VERSION", {f.name for f in self.fs.list_feature_groups().schema.fields})
            matching = [r for r in rows if r["NAME"] == fg_name and r["VERSION"] == fg_version]
            self.assertEqual(len(matching), 1, f"expected exactly one row for {fg_name}/{fg_version}")
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)
            self.fs.delete_feature_group(fg_name, fg_version)  # idempotence

        rows_after = self.fs.list_feature_groups().collect()
        self.assertFalse(
            any(r["NAME"] == fg_name and r["VERSION"] == fg_version for r in rows_after),
            "deleted FG should no longer appear in list_feature_groups",
        )

    def test_feature_group_pk_is_union_when_sources_have_different_join_keys(self) -> None:
        """FG over a (USER_ID) FV and a (USER_ID, ITEM_ID) FV builds an OFT keyed on the superset."""
        fv_single_name, _src_s, _key_s = self._register_postgres_fv(suffix="SK")
        fv_multi_name = self._register_postgres_fv_multikey(suffix="MK", join_keys=["USER_ID", "ITEM_ID"])

        fv_single = self.fs.get_feature_view(fv_single_name, "v1")
        fv_multi = self.fs.get_feature_view(fv_multi_name, "v1")

        fg_name = f"FG_INTEG_UNION_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[fv_single, fv_multi], auto_prefix=True)

        fg_version = "v1"
        try:
            registered = self.fs.register_feature_group(fg, fg_version)
            self.assertEqual(str(registered.version), fg_version)

            # Spec-derived superset PK: USER_ID (from FV_A) then ITEM_ID (new in FV_B).
            spec = registered._to_spec(
                database=self.fs._config.database.resolved(),
                schema=self.fs._config.schema.resolved(),
                version=fg_version,
            )
            self.assertEqual(list(spec.spec.ordered_entity_column_names), ["USER_ID", "ITEM_ID"])
            # Acceptance of the union-PK DDL is itself the assertion: register
            # succeeded, which means CREATE ... PRIMARY KEY ("USER_ID", "ITEM_ID")
            # FROM SPECIFICATION <multi-source> was accepted by the server.
            # Postgres-backed OFTs are unreadable via SQL — reads go through the
            # Online Service Query API (covered by read tests below).
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)

    def test_register_rejects_non_postgres_source(self) -> None:
        """Source FVs must be online + Postgres; mismatch fails before any side effect."""
        fv_pg_name, _src_pg, _key_pg = self._register_postgres_fv(suffix="PGOK")
        fv_hybrid_name, _src_hyb, _key_hyb = self._register_postgres_fv(
            suffix="HYB", store_type=OnlineStoreType.HYBRID_TABLE
        )

        fv_pg = self.fs.get_feature_view(fv_pg_name, "v1")
        fv_hybrid = self.fs.get_feature_view(fv_hybrid_name, "v1")

        fg_name = f"FG_INTEG_BAD_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[fv_pg, fv_hybrid])
        with self.assertRaisesRegex(ValueError, "OnlineStoreType.POSTGRES"):
            self.fs.register_feature_group(fg, "v1")

        # Failed registration must not have created the OFT.
        rows = self.fs.list_feature_groups().collect()
        self.assertFalse(any(r["NAME"] == fg_name for r in rows))

    def test_read_feature_group_round_trip(self) -> None:
        """``read_feature_group`` returns pandas with the FG's predetermined output columns."""
        fv1_name, _src1, entity_key = self._register_postgres_fv(suffix="RA")
        fv2_name, _src2, _key2 = self._register_postgres_fv(suffix="RB")

        fv1 = self.fs.get_feature_view(fv1_name, "v1")
        fv2 = self.fs.get_feature_view(fv2_name, "v1")

        fg_name = f"FG_INTEG_R_{uuid.uuid4().hex[:8].upper()}"
        fg_version = "v1"
        # ``auto_prefix=False`` keeps the output column set predictable.
        fg = FeatureGroup(
            name=fg_name,
            features=[fv1, fv2.slice(["AMOUNT"]).with_name("sender")],
            auto_prefix=False,
        )

        try:
            registered = self.fs.register_feature_group(fg, fg_version)

            fg_live = self.fs.get_feature_group(fg_name, fg_version)
            self._wait_until_fg_read_returns_rows(fg_live, entity_key)
            pdf = self._read_feature_group_with_retry(fg_live, keys=[[entity_key]])

            self.assertIsInstance(pdf, pd.DataFrame)
            self.assertEqual(len(pdf), 1)
            actual_cols = {_normalize_column_name(c) for c in pdf.columns}
            expected_cols = {_normalize_column_name(c) for c in registered.output_columns}
            self.assertTrue(expected_cols.issubset(actual_cols), f"missing FG cols: {expected_cols - actual_cols}")
            # ``output_columns`` is feature-only; ``USER_ID`` is the join key the response also carries.
            self.assertIn("USER_ID", actual_cols)
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)

    def test_read_feature_group_rejects_empty_keys(self) -> None:
        """Empty ``keys`` is rejected client-side without an HTTP round-trip."""
        fv_name, _src, _key = self._register_postgres_fv(suffix="REK")
        fv = self.fs.get_feature_view(fv_name, "v1")
        fg_name = f"FG_INTEG_EK_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[fv])
        try:
            registered = self.fs.register_feature_group(fg, "v1")
            with self.assertRaisesRegex(ValueError, "at least one row in `keys`"):
                self.fs.read_feature_group(registered, keys=[])
        finally:
            self.fs.delete_feature_group(fg_name, "v1")

    def test_generate_training_set_from_feature_group(self) -> None:
        """``generate_training_set(feature_group=...)`` joins a spine against the FG."""
        fv1_name, src1, _key1 = self._register_postgres_fv(suffix="TA")
        fv2_name, _src2, _key2 = self._register_postgres_fv(suffix="TB")
        fv1 = self.fs.get_feature_view(fv1_name, "v1")
        fv2 = self.fs.get_feature_view(fv2_name, "v1")

        fg_name = f"FG_INTEG_T_{uuid.uuid4().hex[:8].upper()}"
        fg_version = "v1"
        # auto_prefix=True disambiguates the two sources (both have AMOUNT, BALANCE).
        fg = FeatureGroup(name=fg_name, features=[fv1, fv2], auto_prefix=True)

        try:
            self.fs.register_feature_group(fg, fg_version)
            fg_live = self.fs.get_feature_group(fg_name, fg_version)

            keys_df = self._session.sql(f"SELECT USER_ID FROM {src1}")
            spine = keys_df.select("USER_ID")

            ts = self.fs.generate_training_set(spine, feature_group=fg_live)
            cols = {_normalize_column_name(c) for c in ts.columns}
            self.assertIn("USER_ID", cols)
            self.assertTrue(
                any("AMOUNT" in c for c in cols),
                f"FG features missing from training set: ts.columns={sorted(cols)}",
            )

            # Schema-only smoke: data may be empty before the FV refresh backfills.
            ts.limit(1).collect()
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)

    def _register_rtfv_with_upstream(self, *, suffix: str, upstream: FeatureView) -> tuple[str, FeatureView]:
        """Register an RTFV that consumes ``upstream`` and computes ``WEIGHTED_BALANCE``.

        Args:
            suffix: Short label embedded in the RTFV name.
            upstream: Already-registered Postgres BFV providing ``BALANCE``.

        Returns:
            ``(rtfv_name, registered_rtfv)`` for the freshly registered RTFV.
        """
        rtfv_name = f"FG_INTEG_RTFV_{suffix}_{uuid.uuid4().hex[:8].upper()}"
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=RealtimeConfig(
                compute_fn=_rtfv_weighted_balance,
                # WEIGHT is the per-request scalar; USER_ID is prepended server-side
                # from the entity row, so declaring it here would be rejected by
                # ``validate_rtfv_entity_contract``.
                sources=[RequestSource(schema=StructType([StructField("WEIGHT", DoubleType())])), upstream],
                output_schema=StructType([StructField("WEIGHTED_BALANCE", DoubleType())]),
            ),
        )
        registered = self.fs.register_feature_view(rtfv, "v1")
        return rtfv_name, registered

    def _register_no_rs_rtfv_with_upstream(self, *, suffix: str, upstream: FeatureView) -> tuple[str, FeatureView]:
        """Register an RTFV with no ``RequestSource`` that doubles the upstream ``BALANCE``.

        Args:
            suffix: Short label embedded in the RTFV name.
            upstream: Already-registered Postgres BFV providing ``BALANCE``.

        Returns:
            ``(rtfv_name, registered_rtfv)`` for the freshly registered RTFV.
        """
        rtfv_name = f"FG_INTEG_NRSRT_{suffix}_{uuid.uuid4().hex[:8].upper()}"
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=RealtimeConfig(
                compute_fn=_rtfv_doubled_balance,
                sources=[upstream],
                output_schema=StructType([StructField("DOUBLED_BALANCE", DoubleType())]),
            ),
        )
        registered = self.fs.register_feature_view(rtfv, "v1")
        return rtfv_name, registered

    def test_register_and_read_mixed_fg_round_trip(self) -> None:
        """FG over one BFV + one RTFV: read returns BFV column and RTFV-computed column on the same row.

        Verifies the FG-with-RTFV-upstream end-to-end: register both
        sources, register the FG, read with a per-row ``request_context``,
        and assert both the BFV-materialized ``BALANCE`` (1000.0) and the
        RTFV-computed ``WEIGHTED_BALANCE`` (= ``BALANCE * WEIGHT`` =
        2500.0) ride along on the same row keyed by ``USER_ID``.
        """
        bfv_name, _src, user_id = self._register_postgres_fv(suffix="MFR")
        bfv = self.fs.get_feature_view(bfv_name, "v1")
        rtfv_name, _ = self._register_rtfv_with_upstream(suffix="MFR", upstream=bfv)
        rtfv = self.fs.get_feature_view(rtfv_name, "v1")

        fg_name = f"FG_INTEG_MIXED_{uuid.uuid4().hex[:8].upper()}"
        fg_version = "v1"
        # ``auto_prefix=False`` keeps the output set predictable for assertions;
        # BFV emits BALANCE/AMOUNT and the RTFV emits WEIGHTED_BALANCE, so there
        # is no collision to disambiguate.
        fg = FeatureGroup(name=fg_name, features=[bfv, rtfv], auto_prefix=False)

        try:
            registered = self.fs.register_feature_group(fg, fg_version)
            self.assertIn("BALANCE", {_normalize_column_name(c) for c in registered.output_columns})
            self.assertIn("WEIGHTED_BALANCE", {_normalize_column_name(c) for c in registered.output_columns})

            fg_live = self.fs.get_feature_group(fg_name, fg_version)

            # Log the FG spec the server is materializing so a NaN/missing-column
            # response can be cross-referenced against the client-side intent.
            fg_spec = registered._to_spec(
                database=self.fs._config.database.resolved(),
                schema=self.fs._config.schema.resolved(),
                version=fg_version,
            )
            logger.info(
                "FG spec for %s/%s: ordered_entity_columns=%s output_columns=%s sources=%s",
                fg_name,
                fg_version,
                list(fg_spec.spec.ordered_entity_column_names),
                list(registered.output_columns),
                [(s.name, getattr(s, "version", None)) for s in fg_spec.spec.sources],
            )

            # Poll until WEIGHTED_BALANCE materializes as a finite value. A freshly
            # registered FG-with-RTFV can transiently return rows whose RTFV-computed
            # columns are NaN: the FG read can emit one row per entity key as soon
            # as the BFV's OFT side is populated, but the RTFV's compute_fn invocation
            # against its upstream BFV can still see a stale/empty replica during
            # the same call. Wait for the RTFV-computed column itself to converge.
            deadline = time.time() + 300.0
            last_err: Optional[str] = None
            pdf: Optional[pd.DataFrame] = None
            request_context = pd.DataFrame({"WEIGHT": [2.5]})
            attempt = 0
            while time.time() < deadline:
                attempt += 1
                try:
                    pdf = self.fs.read_feature_group(fg_live, keys=[[user_id]], request_context=request_context)
                    # Log every attempt's full response shape + values so the
                    # difference between "missing column" / "NaN value" /
                    # "non-NaN" is obvious in the test log.
                    if len(pdf) > 0:
                        logger.info(
                            "attempt=%d cols=%s dtypes=%s row0=%s",
                            attempt,
                            list(pdf.columns),
                            {c: str(pdf[c].dtype) for c in pdf.columns},
                            pdf.iloc[0].to_dict(),
                        )
                        weighted_col = next(
                            (c for c in pdf.columns if _normalize_column_name(c) == "WEIGHTED_BALANCE"),
                            None,
                        )
                        if weighted_col is not None and pd.notna(pdf.iloc[0][weighted_col]):
                            break
                    else:
                        logger.info("attempt=%d empty response", attempt)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    logger.info("attempt=%d read raised %s", attempt, last_err)
                time.sleep(10)
            else:
                snapshot = (
                    None
                    if pdf is None
                    else {
                        "columns": list(pdf.columns),
                        "dtypes": {c: str(pdf[c].dtype) for c in pdf.columns},
                        "row0": None if len(pdf) == 0 else pdf.iloc[0].to_dict(),
                    }
                )
                self.fail(
                    f"read_feature_group({fg_name}/{fg_version}) did not materialize "
                    f"WEIGHTED_BALANCE within 300s; attempts={attempt} last_err={last_err!r} "
                    f"last_response={snapshot!r}"
                )

            assert pdf is not None  # narrowed by the loop
            self.assertEqual(len(pdf), 1)
            cols = {_normalize_column_name(c) for c in pdf.columns}
            self.assertIn("USER_ID", cols)
            self.assertIn("BALANCE", cols)
            self.assertIn("WEIGHTED_BALANCE", cols)

            balance_col = next(c for c in pdf.columns if _normalize_column_name(c) == "BALANCE")
            weighted_col = next(c for c in pdf.columns if _normalize_column_name(c) == "WEIGHTED_BALANCE")
            self.assertAlmostEqual(float(pdf.iloc[0][balance_col]), 1000.0, places=4)
            self.assertAlmostEqual(float(pdf.iloc[0][weighted_col]), 2500.0, places=4)
        finally:
            # Ordered teardown: FG references both FVs, so drop FG first.
            self.fs.delete_feature_group(fg_name, fg_version)
            self.fs.delete_feature_view(rtfv_name, "v1")
            self.fs.delete_feature_view(bfv_name, "v1")

    def test_register_and_read_fg_with_no_request_source_rtfv_round_trip(self) -> None:
        """FG over a no-RequestSource RTFV: read with ``request_context=None`` returns the doubled BFV column.

        Mirrors the parent's no-RequestSource RTFV pattern at the FG layer.
        Registers a Postgres BFV (``BALANCE``=1000.0), an RTFV without a
        ``RequestSource`` that doubles ``BALANCE``, then an FG over both
        sources. Reads with ``request_context=None`` and asserts the
        RTFV-computed ``DOUBLED_BALANCE`` (= 2000.0) and the upstream
        ``BALANCE`` ride along on the same row keyed by ``USER_ID``.
        """
        bfv_name, _src, user_id = self._register_postgres_fv(suffix="NRS")
        bfv = self.fs.get_feature_view(bfv_name, "v1")
        rtfv_name, _ = self._register_no_rs_rtfv_with_upstream(suffix="NRS", upstream=bfv)
        rtfv = self.fs.get_feature_view(rtfv_name, "v1")

        fg_name = f"FG_INTEG_NRS_{uuid.uuid4().hex[:8].upper()}"
        fg_version = "v1"
        fg = FeatureGroup(name=fg_name, features=[bfv, rtfv], auto_prefix=False)

        try:
            registered = self.fs.register_feature_group(fg, fg_version)
            persisted_cols = {_normalize_column_name(c) for c in registered.output_columns}
            self.assertIn("BALANCE", persisted_cols)
            self.assertIn("DOUBLED_BALANCE", persisted_cols)

            fg_live = self.fs.get_feature_group(fg_name, fg_version)

            # Same OFT-replica race as the with-RequestSource sibling test:
            # poll until the RTFV-computed column converges to a non-NaN value.
            # Per-attempt logging mirrors test_register_and_read_mixed_fg_round_trip
            # so a "no DOUBLED_BALANCE / always-NaN / always-empty" failure is
            # diagnosable from the test log alone.
            deadline = time.time() + 300.0
            last_err: Optional[str] = None
            pdf: Optional[pd.DataFrame] = None
            attempt = 0
            while time.time() < deadline:
                attempt += 1
                try:
                    pdf = self.fs.read_feature_group(fg_live, keys=[[user_id]])
                    if len(pdf) > 0:
                        logger.info(
                            "attempt=%d cols=%s dtypes=%s row0=%s",
                            attempt,
                            list(pdf.columns),
                            {c: str(pdf[c].dtype) for c in pdf.columns},
                            pdf.iloc[0].to_dict(),
                        )
                        doubled_col = next(
                            (c for c in pdf.columns if _normalize_column_name(c) == "DOUBLED_BALANCE"),
                            None,
                        )
                        if doubled_col is not None and pd.notna(pdf.iloc[0][doubled_col]):
                            break
                    else:
                        logger.info("attempt=%d empty response", attempt)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    logger.info("attempt=%d read raised %s", attempt, last_err)
                time.sleep(10)
            else:
                snapshot = (
                    None
                    if pdf is None
                    else {
                        "columns": list(pdf.columns),
                        "dtypes": {c: str(pdf[c].dtype) for c in pdf.columns},
                        "row0": None if len(pdf) == 0 else pdf.iloc[0].to_dict(),
                    }
                )
                self.fail(
                    f"read_feature_group({fg_name}/{fg_version}) did not materialize "
                    f"DOUBLED_BALANCE within 300s; attempts={attempt} last_err={last_err!r} "
                    f"last_response={snapshot!r}"
                )

            assert pdf is not None  # narrowed by the loop
            self.assertEqual(len(pdf), 1)
            balance_col = next(c for c in pdf.columns if _normalize_column_name(c) == "BALANCE")
            doubled_col = next(c for c in pdf.columns if _normalize_column_name(c) == "DOUBLED_BALANCE")
            self.assertAlmostEqual(float(pdf.iloc[0][balance_col]), 1000.0, places=4)
            self.assertAlmostEqual(float(pdf.iloc[0][doubled_col]), 2000.0, places=4)
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)
            self.fs.delete_feature_view(rtfv_name, "v1")
            self.fs.delete_feature_view(bfv_name, "v1")

    def test_read_fg_missing_request_context_rejected(self) -> None:
        """``read_feature_group`` on an FG with an RTFV source rejects ``request_context=None`` client-side.

        No HTTP call is needed for the rejection; the failure asserts the
        client-side guard fires before the Online Service is contacted, so
        this test runs even when the server-side RTFV read phase is gated
        off.
        """
        bfv_name, _src, user_id = self._register_postgres_fv(suffix="MR")
        bfv = self.fs.get_feature_view(bfv_name, "v1")
        rtfv_name, _ = self._register_rtfv_with_upstream(suffix="MR", upstream=bfv)
        rtfv = self.fs.get_feature_view(rtfv_name, "v1")

        fg_name = f"FG_INTEG_MR_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[bfv, rtfv], auto_prefix=False)
        try:
            registered = self.fs.register_feature_group(fg, "v1")
            with self.assertRaisesRegex(ValueError, "request_context.*is required"):
                self.fs.read_feature_group(registered, keys=[[user_id]])
        finally:
            self.fs.delete_feature_group(fg_name, "v1")
            self.fs.delete_feature_view(rtfv_name, "v1")
            self.fs.delete_feature_view(bfv_name, "v1")

    def test_read_fg_request_context_extras_dropped(self) -> None:
        """Extra ``request_context`` columns emit a ``UserWarning`` and are dropped before the HTTP call."""
        bfv_name, _src, user_id = self._register_postgres_fv(suffix="MX")
        bfv = self.fs.get_feature_view(bfv_name, "v1")
        rtfv_name, _ = self._register_rtfv_with_upstream(suffix="MX", upstream=bfv)
        rtfv = self.fs.get_feature_view(rtfv_name, "v1")

        fg_name = f"FG_INTEG_MX_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[bfv, rtfv], auto_prefix=False)
        try:
            self.fs.register_feature_group(fg, "v1")
            fg_live = self.fs.get_feature_group(fg_name, "v1")

            request_context = pd.DataFrame({"WEIGHT": [2.5], "STRAY": ["x"]})

            # Wait until the OFS catches up; assertWarns covers the call that
            # eventually succeeds since the warning fires on every call.
            deadline = time.time() + 300.0
            last_err: Optional[str] = None
            pdf: Optional[pd.DataFrame] = None
            while time.time() < deadline:
                try:
                    with self.assertWarns(UserWarning) as warn_ctx:
                        pdf = self.fs.read_feature_group(fg_live, keys=[[user_id]], request_context=request_context)
                    if len(pdf) > 0:
                        break
                except AssertionError:
                    raise
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                time.sleep(10)
            else:
                self.fail(f"read_feature_group({fg_name}/v1) returned no rows within 300s; " f"last_err={last_err!r}")

            self.assertIn("STRAY", str(warn_ctx.warning))
            assert pdf is not None
            self.assertEqual(len(pdf), 1)
            # Extras drop client-side, so the response only carries the FG outputs:
            # USER_ID + BFV columns + WEIGHTED_BALANCE. STRAY must not leak into the response.
            self.assertNotIn("STRAY", {_normalize_column_name(c) for c in pdf.columns})
        finally:
            self.fs.delete_feature_group(fg_name, "v1")
            self.fs.delete_feature_view(rtfv_name, "v1")
            self.fs.delete_feature_view(bfv_name, "v1")

    def test_generate_training_set_from_feature_group_with_pit(self) -> None:
        """``spine_timestamp_col`` is forwarded so PIT correctness still works on the FG path."""
        fv_name, src, _key = self._register_postgres_fv(suffix="P")
        fv = self.fs.get_feature_view(fv_name, "v1")

        fg_name = f"FG_INTEG_P_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[fv], auto_prefix=False)

        try:
            self.fs.register_feature_group(fg, "v1")
            fg_live = self.fs.get_feature_group(fg_name, "v1")

            spine = self._session.sql(f"SELECT USER_ID, EVENT_TIME FROM {src}")

            ts = self.fs.generate_training_set(spine, feature_group=fg_live, spine_timestamp_col="EVENT_TIME")
            cols = {_normalize_column_name(c) for c in ts.columns}
            self.assertIn("EVENT_TIME", cols)
            ts.limit(1).collect()
        finally:
            self.fs.delete_feature_group(fg_name, "v1")

    def test_generate_training_set_from_fg_with_rtfv(self) -> None:
        """FG = [BFV, RTFV]: training set has BFV and RTFV-computed columns on the same row.

        Exercises the FG -> RTFV-aware ``generate_training_set`` path
        end-to-end against offline upstream tables (no Online Service).
        """
        bfv_name, _src, user_id = self._register_postgres_fv(suffix="DG")
        bfv = self.fs.get_feature_view(bfv_name, "v1")

        rtfv_name = f"RTFV_DG_FG_{uuid.uuid4().hex[:8].upper()}"
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=RealtimeConfig(
                compute_fn=_rtfv_weighted_balance,
                sources=[RequestSource(schema=StructType([StructField("WEIGHT", DoubleType())])), bfv],
                output_schema=StructType([StructField("WEIGHTED_BALANCE", DoubleType())]),
            ),
        )
        self.fs.register_feature_view(rtfv, "v1")

        fg_name = f"FG_DG_RTFV_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[bfv, self.fs.get_feature_view(rtfv_name, "v1")], auto_prefix=False)

        try:
            self.fs.register_feature_group(fg, "v1")
            fg_live = self.fs.get_feature_group(fg_name, "v1")

            spine = self._session.create_dataframe(
                [(user_id, 2.5)],
                schema=StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("WEIGHT", DoubleType()),
                    ]
                ),
            )
            result = self.fs.generate_training_set(spine, feature_group=fg_live).to_pandas()

            cols = {_normalize_column_name(c) for c in result.columns}
            self.assertIn("BALANCE", cols)
            self.assertIn("WEIGHTED_BALANCE", cols)
            self.assertEqual(len(result), 1)
            balance_col = next(c for c in result.columns if _normalize_column_name(c) == "BALANCE")
            weighted_col = next(c for c in result.columns if _normalize_column_name(c) == "WEIGHTED_BALANCE")
            self.assertAlmostEqual(float(result.iloc[0][balance_col]), 1000.0)
            self.assertAlmostEqual(float(result.iloc[0][weighted_col]), 1000.0 * 2.5)
        finally:
            try:
                self.fs.delete_feature_group(fg_name, "v1")
            finally:
                self.fs.delete_feature_view(rtfv_name, "v1")

    def test_feature_group_with_tiled_sfv_online_read(self) -> None:
        """A FG over a tiled SFV upstream registers and ``read_feature_group`` returns its schema."""
        sfx = "TS"
        fv_tiled_name, seeded_user_id = self._register_postgres_tiled_sfv(suffix=sfx)
        fv_batch_name, _src_b, _key_b = self._register_postgres_fv(suffix="TB2")

        fv_tiled = self.fs.get_feature_view(fv_tiled_name, "v1")
        fv_batch = self.fs.get_feature_view(fv_batch_name, "v1")

        fg_name = f"FG_INTEG_TILED_{uuid.uuid4().hex[:8].upper()}"
        fg_version = "v1"
        # auto_prefix=True keeps the SFV's agg outputs distinct from the batch FV's columns.
        fg = FeatureGroup(name=fg_name, features=[fv_tiled, fv_batch], auto_prefix=True)

        try:
            registered = self.fs.register_feature_group(fg, fg_version)
            normalized_outputs = {_normalize_column_name(c) for c in registered.output_columns}
            for agg in (f"AMOUNT_SUM_1D_{sfx}", f"AMOUNT_MAX_1D_{sfx}", f"AMOUNT_COUNT_1D_{sfx}"):
                self.assertTrue(
                    any(agg in c for c in normalized_outputs),
                    f"expected agg '{agg}' in FG output_columns; got {sorted(normalized_outputs)}",
                )

            fg_live = self.fs.get_feature_group(fg_name, fg_version)
            # Schema-only smoke: data may be empty before the SFV backfill lands in the OFT.
            pdf = self._read_feature_group_with_retry(fg_live, keys=[[seeded_user_id]])
            pdf_cols = {_normalize_column_name(c) for c in pdf.columns}
            for agg in (f"AMOUNT_SUM_1D_{sfx}", f"AMOUNT_MAX_1D_{sfx}", f"AMOUNT_COUNT_1D_{sfx}"):
                self.assertTrue(
                    any(agg in c for c in pdf_cols),
                    f"expected agg '{agg}' in read columns; got {sorted(pdf_cols)}",
                )
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)

    def test_feature_group_with_tiled_bfv_online_read(self) -> None:
        """A FG over a tiled BFV upstream registers and ``read_feature_group`` returns its schema."""
        sfx = "TB"
        fv_tiled_name, seeded_user_id = self._register_postgres_tiled_bfv(suffix=sfx)
        fv_batch_name, _src_b, _key_b = self._register_postgres_fv(suffix="TBB")

        fv_tiled = self.fs.get_feature_view(fv_tiled_name, "v1")
        fv_batch = self.fs.get_feature_view(fv_batch_name, "v1")

        fg_name = f"FG_INTEG_TILED_BFV_{uuid.uuid4().hex[:8].upper()}"
        fg_version = "v1"
        # auto_prefix=True keeps the tiled BFV's agg outputs distinct from the batch FV's columns.
        fg = FeatureGroup(name=fg_name, features=[fv_tiled, fv_batch], auto_prefix=True)

        try:
            registered = self.fs.register_feature_group(fg, fg_version)
            normalized_outputs = {_normalize_column_name(c) for c in registered.output_columns}
            for agg in (f"AMOUNT_SUM_1D_{sfx}", f"AMOUNT_MAX_1D_{sfx}", f"AMOUNT_COUNT_1D_{sfx}"):
                self.assertTrue(
                    any(agg in c for c in normalized_outputs),
                    f"expected agg '{agg}' in FG output_columns; got {sorted(normalized_outputs)}",
                )

            fg_live = self.fs.get_feature_group(fg_name, fg_version)
            # Schema-only smoke: data may be empty before the BFV tile DT backfills.
            pdf = self._read_feature_group_with_retry(fg_live, keys=[[seeded_user_id]])
            pdf_cols = {_normalize_column_name(c) for c in pdf.columns}
            for agg in (f"AMOUNT_SUM_1D_{sfx}", f"AMOUNT_MAX_1D_{sfx}", f"AMOUNT_COUNT_1D_{sfx}"):
                self.assertTrue(
                    any(agg in c for c in pdf_cols),
                    f"expected agg '{agg}' in read columns; got {sorted(pdf_cols)}",
                )
        finally:
            self.fs.delete_feature_group(fg_name, fg_version)

    def test_generate_training_set_from_feature_group_with_tiled_sfv(self) -> None:
        """``generate_training_set(feature_group=...)`` works when one upstream FV is tiled."""
        sfx = "TS2"
        fv_tiled_name, _seeded_user_id = self._register_postgres_tiled_sfv(suffix=sfx)
        fv_tiled = self.fs.get_feature_view(fv_tiled_name, "v1")

        fg_name = f"FG_INTEG_TILED_TS_{uuid.uuid4().hex[:8].upper()}"
        fg = FeatureGroup(name=fg_name, features=[fv_tiled], auto_prefix=False)

        try:
            self.fs.register_feature_group(fg, "v1")
            fg_live = self.fs.get_feature_group(fg_name, "v1")

            spine = self._session.create_dataframe(
                [
                    ("u1", datetime(2024, 1, 5, 0, 0, 0)),
                    ("u2", datetime(2024, 1, 5, 0, 0, 0)),
                ],
                schema=["USER_ID", "QUERY_TS"],
            )
            ts = self.fs.generate_training_set(spine, feature_group=fg_live, spine_timestamp_col="QUERY_TS")
            cols = {_normalize_column_name(c) for c in ts.columns}
            for agg in (f"AMOUNT_SUM_1D_{sfx}", f"AMOUNT_MAX_1D_{sfx}", f"AMOUNT_COUNT_1D_{sfx}"):
                self.assertIn(agg, cols, f"expected agg '{agg}' in training-set columns; got {sorted(cols)}")
            # Schema-only smoke: data may be empty before the tile DT backfills.
            ts.limit(1).collect()
        finally:
            self.fs.delete_feature_group(fg_name, "v1")


if __name__ == "__main__":
    absltest.main()
