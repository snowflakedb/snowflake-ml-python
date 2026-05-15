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
"""

from __future__ import annotations

import logging
import os
import time
import unittest
import uuid
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_group import FeatureGroup
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewSlice,
    OnlineConfig,
    OnlineStoreType,
)

logger = logging.getLogger(__name__)


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

    def _wait_until_fg_read_returns_rows(self, fg_live: FeatureGroup, key: str, timeout: float = 300.0) -> None:
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
            time.sleep(10)
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

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for FG online read (Online Service Query API).",
    )
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


if __name__ == "__main__":
    absltest.main()
