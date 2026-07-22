"""E2E integration tests for RealtimeFeatureView authoring and reads (bundled).

Reuses ``StreamingFeatureViewIntegTestBase`` for the class-scoped Feature
Store, ``USER_ID`` entity, and Online Service.

The registration / listing / deletion tests do not require
``SNOWFLAKE_PAT``. The ``read_feature_view`` round-trip tests run when
``SNOWFLAKE_PAT`` is set and call the Online Service query API; the target
account must support RTFV reads on the bundled Online Service.

Run via the bundled runner::

    bazel test //tests/integ/snowflake/ml/feature_store:feature_store_spec_oft_bundle_test \\
        --test_filter=RealtimeFeatureViewIntegTest
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import pandas as pd
from absl.testing import absltest
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    identity_transform,
)

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType

logger = logging.getLogger(__name__)


def _rtfv_compute_fn(request_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
    """Reference RTFV ``compute_fn`` used by the round-trip tests.

    Operates on features only with positional row alignment (no merge on
    entity keys). Returns just the derived feature; entity keys are
    attached to the read response by the framework.

    Args:
        request_df: Request payload with ``WEIGHT`` (RequestSource fields only).
        balance_df: Upstream FV rows with ``BALANCE``, row-aligned with ``request_df``.

    Returns:
        DataFrame with ``WEIGHTED_BALANCE = BALANCE * WEIGHT``, row-aligned.
    """
    weight = request_df["WEIGHT"].astype(float).reset_index(drop=True)
    balance = balance_df["BALANCE"].fillna(0.0).reset_index(drop=True)
    return pd.DataFrame({"WEIGHTED_BALANCE": balance * weight})


def _rtfv_mismatch_compute_fn(request_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
    """Variant ``compute_fn`` used by entity-contract / non-Postgres / DRAFT tests.

    Returns an empty-feature row (the test cares only that registration
    fails before this is ever invoked).

    Args:
        request_df: Request payload (RequestSource fields only).
        balance_df: Upstream FV rows.

    Returns:
        DataFrame with a single ``MARKER`` column, row-aligned.
    """
    n = len(request_df) if request_df is not None else len(balance_df)
    return pd.DataFrame({"MARKER": [0.0] * n})


def _rtfv_two_upstream_compute_fn(
    request_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine two upstream FVs into ``COMBINED = BALANCE * WEIGHT + SCORE``.

    Args:
        request_df: Request payload with ``WEIGHT``.
        balance_df: First upstream FV with ``BALANCE``.
        score_df: Second upstream FV with ``SCORE``.

    Returns:
        DataFrame with ``COMBINED``, row-aligned with the inputs.
    """
    weight = request_df["WEIGHT"].astype(float).reset_index(drop=True)
    balance = balance_df["BALANCE"].fillna(0.0).reset_index(drop=True)
    score = score_df["SCORE"].fillna(0.0).reset_index(drop=True)
    return pd.DataFrame({"COMBINED": balance * weight + score})


def _rtfv_no_request_compute_fn(balance_df: pd.DataFrame) -> pd.DataFrame:
    """RTFV ``compute_fn`` that has no RequestSource.

    Args:
        balance_df: Upstream FV rows with ``BALANCE``.

    Returns:
        DataFrame with ``DOUBLED_BALANCE = BALANCE * 2``, row-aligned.
    """
    balance = balance_df["BALANCE"].fillna(0.0).reset_index(drop=True)
    return pd.DataFrame({"DOUBLED_BALANCE": balance * 2.0})


# Fixed alias names for tiled-upstream tests so the compute_fn can reference them by name.
_TILED_SUM_COL = "AMOUNT_SUM_1D"
_TILED_MAX_COL = "AMOUNT_MAX_1D"
_TILED_COUNT_COL = "AMOUNT_COUNT_1D"


def _rtfv_tiled_upstream_compute_fn(request_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """RTFV ``compute_fn`` over a tiled-FV upstream.

    Args:
        request_df: Request payload with ``WEIGHT``.
        agg_df: Upstream tiled FV rows exposing ``AMOUNT_SUM_1D``, row-aligned.

    Returns:
        DataFrame with ``WEIGHTED_SUM = AMOUNT_SUM_1D * WEIGHT``, row-aligned.
    """
    weight = request_df["WEIGHT"].astype(float).reset_index(drop=True)
    upstream_sum = agg_df["AMOUNT_SUM_1D"].fillna(0.0).astype(float).reset_index(drop=True)
    return pd.DataFrame({"WEIGHTED_SUM": upstream_sum * weight})


class RealtimeFeatureViewIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """RTFV registration/listing/deletion against a live Online Service."""

    # ----- helpers -------------------------------------------------------

    def _register_postgres_fv(
        self,
        *,
        suffix: str,
        store_type: OnlineStoreType = OnlineStoreType.POSTGRES,
    ) -> tuple[str, str, str]:
        """Create + register a 1-row Postgres batch FV used as the RTFV upstream."""
        s = uuid.uuid4().hex[:8]
        fv_name = f"RTFV_INTEG_FV_{suffix}_{s}"
        seeded_user_id = f"U_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.RTFV_INTEG_SRC_{suffix}_{s}"

        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                BALANCE FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {src_table} VALUES
            ({seeded_user_id!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), 1000.0)
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

    def _register_named_feature_postgres_fv(
        self,
        *,
        suffix: str,
        feature_column: str,
        feature_value: float,
        user_id: str,
    ) -> str:
        """Variant of :meth:`_register_postgres_fv` with a caller-chosen feature column.

        Args:
            suffix: Short suffix to disambiguate the FV / source table.
            feature_column: Feature column name to expose alongside ``USER_ID``.
            feature_value: Value to seed for the single row.
            user_id: ``USER_ID`` value to seed.

        Returns:
            The registered FV name.
        """
        s = uuid.uuid4().hex[:8]
        fv_name = f"RTFV_INTEG_FV_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.RTFV_INTEG_NF_SRC_{suffix}_{s}"

        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                {feature_column} FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {src_table} VALUES
            ({user_id!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), {feature_value})
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
        return fv_name

    def _register_multi_row_postgres_fv(
        self,
        *,
        suffix: str,
        rows: list[tuple[str, float]],
    ) -> tuple[str, str]:
        """Create + register a Postgres batch FV seeded with multiple rows.

        Args:
            suffix: Short suffix used to disambiguate the FV / source table
                within the bundled run.
            rows: ``(user_id, balance)`` pairs to seed into the source table.

        Returns:
            ``(fv_name, src_table)`` -- the registered FV name and the fully
            qualified source-table identifier (for cleanup if the caller
            wants it).
        """
        s = uuid.uuid4().hex[:8]
        fv_name = f"RTFV_INTEG_FV_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.RTFV_INTEG_MR_SRC_{suffix}_{s}"

        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                BALANCE FLOAT
            )
            """
        ).collect()
        values_clause = ", ".join(
            f"({user_id!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), {balance})"
            for user_id, balance in rows
        )
        self._session.sql(f"INSERT INTO {src_table} VALUES {values_clause}").collect()

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
        return fv_name, src_table

    def _register_postgres_tiled_sfv(self, *, suffix: str) -> tuple[str, str]:
        """Register a tiled streaming FV (PG online) with SUM/MAX/COUNT aggs over AMOUNT.

        Args:
            suffix: Short label embedded in the FV / stream / table names.

        Returns:
            Tuple of ``(fv_name, seeded_user_id)``. ``seeded_user_id`` matches a row
            in the backfill table.
        """
        s = uuid.uuid4().hex[:8]
        # StreamSource has a 32-char name limit.
        stream_name = f"RT_TS_{suffix}_{s}"
        fv_name = f"RTFV_INTEG_TILED_SFV_{suffix}_{s}"
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
            Feature.sum("AMOUNT", "1d").alias(_TILED_SUM_COL),
            Feature.max("AMOUNT", "1d").alias(_TILED_MAX_COL),
            Feature.count("AMOUNT", "1d").alias(_TILED_COUNT_COL),
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
        """Register a tiled batch FV (PG online) with SUM/MAX/COUNT aggs over AMOUNT.

        Args:
            suffix: Short label embedded in the FV / source-table names.

        Returns:
            Tuple of ``(fv_name, seeded_user_id)``. ``seeded_user_id`` matches a row
            in the source table.
        """
        s = uuid.uuid4().hex[:8]
        fv_name = f"RTFV_INTEG_TILED_BFV_{suffix}_{s}"
        seeded_user_id = f"U_BTILED_{suffix}_{s}"
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.RTFV_BTILED_SRC_{suffix}_{s}"

        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {src_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
            """
        ).collect()
        # Anchor to UTC day boundaries so tiles align with the online store's UTC window.
        utc_now_ntz = "CONVERT_TIMEZONE('UTC', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ"
        utc_yesterday = f"DATEADD('day', -1, DATE_TRUNC('day', {utc_now_ntz}))"
        self._session.sql(
            f"""
            INSERT INTO {src_table}
            SELECT column1, column2, column3
            FROM VALUES
                ({seeded_user_id!r}, DATEADD('hour', 1, {utc_yesterday}), 10.0),
                ({seeded_user_id!r}, DATEADD('hour', 2, {utc_yesterday}), 20.0),
                ({seeded_user_id!r}, DATEADD('hour', 3, {utc_yesterday}), 30.0)
            """
        ).collect()
        feature_df = self._session.table(src_table)

        features = [
            Feature.sum("AMOUNT", "1d").alias(_TILED_SUM_COL),
            Feature.max("AMOUNT", "1d").alias(_TILED_MAX_COL),
            Feature.count("AMOUNT", "1d").alias(_TILED_COUNT_COL),
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

    def _wait_until_rtfv_read_returns_rows(
        self,
        rtfv_live: FeatureView,
        *,
        keys: list[list[str]],
        request_context: Optional[pd.DataFrame] = None,
        timeout: float = 600.0,
    ) -> pd.DataFrame:
        """Poll ``read_feature_view`` until the RTFV's OFT-backed upstream is ingested.

        Args:
            rtfv_live: Hydrated RealtimeFeatureView (i.e. the one returned
                from ``get_feature_view`` after ``register_feature_view``).
            keys: Entity tuples to read.
            request_context: Per-row request context (same length as ``keys``).
                ``None`` for RTFVs registered without a RequestSource.
            timeout: Maximum total wait, in seconds. Matches the FG bundle's
                default of 300s and accounts for the Online Service refresh.

        Returns:
            The pandas DataFrame from the first successful read.
        """
        deadline = time.time() + timeout
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                pdf = self.fs.read_feature_view(
                    rtfv_live,
                    keys=keys,
                    request_context=request_context,
                )
                if len(pdf) >= len(keys):
                    return pdf
            except Exception as e:
                # Broad catch: the read API unwraps to the original exception, and a
                # transient online-serving 404 surfaces as RuntimeError, not ValueError.
                last_err = f"{type(e).__name__}: {e}"
                logger.info("RTFV read not yet ready: %s", last_err)
            time.sleep(5)
        self.fail(
            f"read_feature_view({rtfv_live.name}/{rtfv_live.version}) returned no rows within "
            f"{timeout}s; last_err={last_err!r}"
        )

    def _make_request_source(self) -> RequestSource:
        # Entity join keys are supplied at read time and prepended server-side;
        # declaring them here is rejected by ``validate_rtfv_entity_contract``.
        schema = StructType([StructField("WEIGHT", DoubleType())])
        return RequestSource(schema=schema)

    def _make_output_schema(self) -> StructType:
        # Features only; entity keys come from the ``keys`` parameter at read
        # time and from the spine at training-set time -- the framework, not
        # ``compute_fn``, attaches them to the response.
        return StructType([StructField("WEIGHTED_BALANCE", DoubleType())])

    # ----- happy path ----------------------------------------------------

    def test_realtime_feature_view_full_round_trip(self) -> None:
        """register -> get -> list -> delete for an RTFV with one Postgres upstream."""
        upstream_name, _src, _key = self._register_postgres_fv(suffix="RT")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=self._make_output_schema(),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
            desc="integ-test RTFV",
        )

        version = "v1"
        registered: Optional[FeatureView] = None
        try:
            registered = self.fs.register_feature_view(rtfv, version)
            self.assertEqual(registered.name.resolved(), rtfv_name)
            self.assertEqual(str(registered.version), version)
            self.assertEqual(registered.status, FeatureViewStatus.ACTIVE)
            self.assertTrue(registered.is_realtime_feature_view)
            self.assertEqual(registered.desc, "integ-test RTFV")
            self.assertIsNotNone(registered.realtime_config)

            fetched = self.fs.get_feature_view(rtfv_name, version)
            self.assertTrue(fetched.is_realtime_feature_view)
            self.assertEqual(str(fetched.version), version)
            self.assertEqual(fetched.desc, "integ-test RTFV")
            self.assertIsNotNone(fetched.realtime_config)
            # ``output_schema`` round-trips through the metadata-table JSON.
            output_names = [f.name for f in fetched.realtime_config.output_schema.fields]
            self.assertEqual(output_names, ["WEIGHTED_BALANCE"])

            rows = self.fs.list_feature_views().collect()
            kinds = {r["NAME"]: r["KIND"] for r in rows}
            self.assertEqual(kinds.get(rtfv_name), "REALTIME")
            descs = {r["NAME"]: r["DESC"] for r in rows}
            self.assertEqual(descs.get(rtfv_name), "integ-test RTFV")

            # Verify desc round-trips through update_feature_view.
            updated = self.fs.update_feature_view(rtfv_name, version, desc="updated RTFV desc")
            self.assertEqual(updated.desc, "updated RTFV desc")

            re_fetched = self.fs.get_feature_view(rtfv_name, version)
            self.assertEqual(re_fetched.desc, "updated RTFV desc")
        finally:
            self.fs.delete_feature_view(rtfv_name, version)
            # Best-effort second delete: telemetry unwraps SnowflakeMLException
            # and re-raises the original ValueError on a now-missing FV, so the
            # only contract we assert here is "second delete does not crash the
            # process". Either a clean no-op or a not-found error is acceptable.
            try:
                self.fs.delete_feature_view(rtfv_name, version)
            except (snowml_exceptions.SnowflakeMLException, ValueError):
                pass

        rows_after = self.fs.list_feature_views().collect()
        self.assertFalse(
            any(r["NAME"] == rtfv_name for r in rows_after),
            "deleted RTFV should no longer appear in list_feature_views",
        )

    def test_realtime_feature_view_no_request_source_round_trip(self) -> None:
        """register -> get -> delete for an RTFV without a RequestSource."""
        upstream_name, _src, _key = self._register_postgres_fv(suffix="NRSRT")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_NRS_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_no_request_compute_fn,
            sources=[upstream],
            output_schema=StructType([StructField("DOUBLED_BALANCE", DoubleType())]),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
            desc="integ-test RTFV without RequestSource",
        )

        version = "v1"
        try:
            registered = self.fs.register_feature_view(rtfv, version)
            self.assertTrue(registered.is_realtime_feature_view)
            self.assertIsNone(registered.realtime_config.request_source)

            fetched = self.fs.get_feature_view(rtfv_name, version)
            self.assertTrue(fetched.is_realtime_feature_view)
            self.assertIsNone(fetched.realtime_config.request_source)
            self.assertEqual(len(fetched.realtime_config.feature_view_sources), 1)
            output_names = [f.name for f in fetched.realtime_config.output_schema.fields]
            self.assertEqual(output_names, ["DOUBLED_BALANCE"])
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_read_feature_view_realtime_round_trip(self) -> None:
        """``read_feature_view(rtfv, keys, request_context=...)`` returns the computed value.

        Registers a Postgres BFV seeded with one row (BALANCE=1000.0), an
        RTFV with a deterministic compute_fn (WEIGHTED_BALANCE = BALANCE *
        WEIGHT), and verifies a single-row read returns WEIGHTED_BALANCE =
        2500.0 = 1000.0 * 2.5.
        """
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="RD1")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_RD1_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=self._make_output_schema(),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
            desc="integ-test RTFV read round-trip",
        )

        version = "v1"
        try:
            self.fs.register_feature_view(rtfv, version)
            rtfv_live = self.fs.get_feature_view(rtfv_name, version)

            request_context = pd.DataFrame({"WEIGHT": [2.5]})
            pdf = self._wait_until_rtfv_read_returns_rows(
                rtfv_live,
                keys=[[user_id]],
                request_context=request_context,
            )

            self.assertIsInstance(pdf, pd.DataFrame)
            self.assertEqual(len(pdf), 1)
            actual_cols = {c.upper() for c in pdf.columns}
            self.assertIn("USER_ID", actual_cols)
            self.assertIn("WEIGHTED_BALANCE", actual_cols)
            weighted_col = next(c for c in pdf.columns if c.upper() == "WEIGHTED_BALANCE")
            self.assertAlmostEqual(float(pdf.iloc[0][weighted_col]), 2500.0, places=4)
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_read_feature_view_realtime_no_request_source_round_trip(self) -> None:
        """``read_feature_view(rtfv, keys)`` works for an RTFV without a RequestSource.

        Registers a Postgres BFV seeded with one row (BALANCE=1000.0), an
        RTFV with a deterministic compute_fn (DOUBLED_BALANCE = BALANCE * 2)
        and no RequestSource, and verifies a single-row read returns 2000.0
        without any ``request_context`` argument.
        """
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="RDN")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_RDN_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_no_request_compute_fn,
            sources=[upstream],
            output_schema=StructType([StructField("DOUBLED_BALANCE", DoubleType())]),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
            desc="integ-test RTFV read without RequestSource",
        )

        version = "v1"
        try:
            self.fs.register_feature_view(rtfv, version)
            rtfv_live = self.fs.get_feature_view(rtfv_name, version)

            pdf = self._wait_until_rtfv_read_returns_rows(rtfv_live, keys=[[user_id]])

            self.assertIsInstance(pdf, pd.DataFrame)
            self.assertEqual(len(pdf), 1)
            actual_cols = {c.upper() for c in pdf.columns}
            self.assertIn("USER_ID", actual_cols)
            self.assertIn("DOUBLED_BALANCE", actual_cols)
            doubled_col = next(c for c in pdf.columns if c.upper() == "DOUBLED_BALANCE")
            self.assertAlmostEqual(float(pdf.iloc[0][doubled_col]), 2000.0, places=4)
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_read_feature_view_realtime_multi_row(self) -> None:
        """Per-row alignment between ``keys`` and ``request_context`` is preserved end-to-end."""
        s = uuid.uuid4().hex[:8].upper()
        user_ids = [f"U_RD2_{s}_{i}" for i in range(4)]
        balances = [100.0, 200.0, 300.0, 400.0]
        upstream_name, _src = self._register_multi_row_postgres_fv(
            suffix="RD2",
            rows=list(zip(user_ids, balances)),
        )
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_RD2_{s}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=self._make_output_schema(),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )

        version = "v1"
        weights = [1.0, 1.5, 2.0, 3.0]
        expected = {uid: balance * weight for uid, balance, weight in zip(user_ids, balances, weights)}

        try:
            self.fs.register_feature_view(rtfv, version)
            rtfv_live = self.fs.get_feature_view(rtfv_name, version)

            request_context = pd.DataFrame({"WEIGHT": weights})
            keys = [[uid] for uid in user_ids]
            pdf = self._wait_until_rtfv_read_returns_rows(
                rtfv_live,
                keys=keys,
                request_context=request_context,
            )

            self.assertEqual(len(pdf), 4)
            user_col = next(c for c in pdf.columns if c.upper() == "USER_ID")
            weighted_col = next(c for c in pdf.columns if c.upper() == "WEIGHTED_BALANCE")
            actual = {row[user_col]: float(row[weighted_col]) for _, row in pdf.iterrows()}
            for uid, expected_value in expected.items():
                self.assertAlmostEqual(
                    actual[uid],
                    expected_value,
                    places=4,
                    msg=f"WEIGHTED_BALANCE for {uid} was {actual.get(uid)!r}, expected {expected_value}",
                )
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_read_feature_view_realtime_two_upstreams_round_trip(self) -> None:
        """End-to-end read against an RTFV with two Postgres upstreams that share ``USER_ID``.

        Exercises :func:`resolve_realtime_join_key_fields` across multiple
        upstreams. ``BALANCE = 100.0`` + ``SCORE = 7.5`` + ``WEIGHT = 3.0``
        -> ``COMBINED = 307.5``.
        """
        s = uuid.uuid4().hex[:8].upper()
        user_id = f"U_RD3_{s}"
        upstream_balance = self._register_named_feature_postgres_fv(
            suffix=f"RD3A_{s}",
            feature_column="BALANCE",
            feature_value=100.0,
            user_id=user_id,
        )
        upstream_score = self._register_named_feature_postgres_fv(
            suffix=f"RD3B_{s}",
            feature_column="SCORE",
            feature_value=7.5,
            user_id=user_id,
        )
        balance_fv = self.fs.get_feature_view(upstream_balance, "v1")
        score_fv = self.fs.get_feature_view(upstream_score, "v1")

        rtfv_name = f"RTFV_INTEG_RD3_{s}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_two_upstream_compute_fn,
            sources=[self._make_request_source(), balance_fv, score_fv],
            output_schema=StructType([StructField("COMBINED", DoubleType())]),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )

        version = "v1"
        try:
            self.fs.register_feature_view(rtfv, version)
            rtfv_live = self.fs.get_feature_view(rtfv_name, version)

            request_context = pd.DataFrame({"WEIGHT": [3.0]})
            pdf = self._wait_until_rtfv_read_returns_rows(
                rtfv_live,
                keys=[[user_id]],
                request_context=request_context,
            )

            self.assertEqual(len(pdf), 1)
            user_col = next(c for c in pdf.columns if c.upper() == "USER_ID")
            combined_col = next(c for c in pdf.columns if c.upper() == "COMBINED")
            self.assertEqual(pdf.iloc[0][user_col], user_id)
            self.assertAlmostEqual(float(pdf.iloc[0][combined_col]), 307.5, places=4)
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_read_feature_view_realtime_key_miss(self) -> None:
        """Read with a key absent from upstream; left-join + fillna -> ``WEIGHTED_BALANCE = 0.0``."""
        s = uuid.uuid4().hex[:8].upper()
        existing_user = f"U_RD4_{s}_EXISTS"
        missing_user = f"U_RD4_{s}_NONE"
        upstream_name = self._register_named_feature_postgres_fv(
            suffix=f"RD4_{s}",
            feature_column="BALANCE",
            feature_value=999.0,
            user_id=existing_user,
        )
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_RD4_{s}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=self._make_output_schema(),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )

        version = "v1"
        try:
            self.fs.register_feature_view(rtfv, version)
            rtfv_live = self.fs.get_feature_view(rtfv_name, version)

            # Key miss always produces a single deterministic row; 120s covers
            # first-call startup latency without the standard 300s ceiling.
            request_context = pd.DataFrame({"WEIGHT": [2.0]})
            pdf = self._wait_until_rtfv_read_returns_rows(
                rtfv_live,
                keys=[[missing_user]],
                request_context=request_context,
                timeout=120.0,
            )

            self.assertEqual(len(pdf), 1)
            user_col = next(c for c in pdf.columns if c.upper() == "USER_ID")
            weighted_col = next(c for c in pdf.columns if c.upper() == "WEIGHTED_BALANCE")
            self.assertEqual(pdf.iloc[0][user_col], missing_user)
            self.assertAlmostEqual(float(pdf.iloc[0][weighted_col]), 0.0, places=4)
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    # ----- tiled-FV upstream cases ---------------------------------------

    def test_realtime_feature_view_with_tiled_sfv_upstream_round_trip(self) -> None:
        """register -> get -> delete for an RTFV whose upstream is a tiled streaming FV."""
        upstream_name, _user_id = self._register_postgres_tiled_sfv(suffix="RTS")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_TILED_SFV_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_tiled_upstream_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=StructType([StructField("WEIGHTED_SUM", DoubleType())]),
        )
        rtfv = FeatureView(name=rtfv_name, entities=[self.user_entity], realtime_config=realtime_config)

        version = "v1"
        try:
            registered = self.fs.register_feature_view(rtfv, version)
            self.assertTrue(registered.is_realtime_feature_view)
            self.assertEqual(str(registered.version), version)

            fetched = self.fs.get_feature_view(rtfv_name, version)
            output_names = [f.name for f in fetched.realtime_config.output_schema.fields]
            self.assertEqual(output_names, ["WEIGHTED_SUM"])
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_realtime_feature_view_with_tiled_bfv_upstream_round_trip(self) -> None:
        """register -> get -> delete for an RTFV whose upstream is a tiled batch FV."""
        upstream_name, _user_id = self._register_postgres_tiled_bfv(suffix="RTB")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_TILED_BFV_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_tiled_upstream_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=StructType([StructField("WEIGHTED_SUM", DoubleType())]),
        )
        rtfv = FeatureView(name=rtfv_name, entities=[self.user_entity], realtime_config=realtime_config)

        version = "v1"
        try:
            registered = self.fs.register_feature_view(rtfv, version)
            self.assertTrue(registered.is_realtime_feature_view)
            self.assertEqual(str(registered.version), version)

            fetched = self.fs.get_feature_view(rtfv_name, version)
            output_names = [f.name for f in fetched.realtime_config.output_schema.fields]
            self.assertEqual(output_names, ["WEIGHTED_SUM"])
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    def test_read_feature_view_realtime_with_tiled_sfv_upstream(self) -> None:
        """``read_feature_view(rtfv, ...)`` works when the RTFV's upstream is a tiled SFV."""
        upstream_name, user_id = self._register_postgres_tiled_sfv(suffix="RTRD")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_TILED_RD_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_tiled_upstream_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=StructType([StructField("WEIGHTED_SUM", DoubleType())]),
        )
        rtfv = FeatureView(name=rtfv_name, entities=[self.user_entity], realtime_config=realtime_config)

        version = "v1"
        try:
            self.fs.register_feature_view(rtfv, version)
            rtfv_live = self.fs.get_feature_view(rtfv_name, version)

            request_context = pd.DataFrame({"WEIGHT": [2.5]})
            pdf = self._wait_until_rtfv_read_returns_rows(
                rtfv_live,
                keys=[[user_id]],
                request_context=request_context,
            )
            self.assertEqual(len(pdf), 1)
            actual_cols = {c.upper() for c in pdf.columns}
            self.assertIn("USER_ID", actual_cols)
            self.assertIn("WEIGHTED_SUM", actual_cols)
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    # ----- negative cases (no side effects expected) ---------------------

    def test_register_rejects_non_postgres_upstream(self) -> None:
        """RTFV upstreams must be online + Postgres; mismatch fails before any OFT is created."""
        upstream_name, _src, _key = self._register_postgres_fv(suffix="NP", store_type=OnlineStoreType.HYBRID_TABLE)
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_NP_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_mismatch_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=StructType([StructField("MARKER", DoubleType())]),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )

        with self.assertRaisesRegex(Exception, "POSTGRES"):
            self.fs.register_feature_view(rtfv, "v1")

        rows = self.fs.list_feature_views().collect()
        self.assertFalse(any(r["NAME"] == rtfv_name for r in rows))

    def test_register_rejects_draft_upstream(self) -> None:
        """Unregistered upstream FVs must be rejected with a NOT_FOUND error."""
        s = uuid.uuid4().hex[:8]
        src_table = f"{self.test_db}.{self.fs._config.schema.identifier()}.RTFV_INTEG_DRAFT_SRC_{s}"
        self._session.sql(
            f"CREATE OR REPLACE TABLE {src_table} (USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, BALANCE FLOAT)"
        ).collect()

        draft_upstream = FeatureView(
            name=f"RTFV_INTEG_DRAFT_FV_{s}",
            entities=[self.user_entity],
            feature_df=self._session.table(src_table),
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )

        rtfv_name = f"RTFV_INTEG_DRAFT_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_mismatch_compute_fn,
            sources=[self._make_request_source(), draft_upstream],
            output_schema=StructType([StructField("MARKER", DoubleType())]),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )

        with self.assertRaisesRegex(Exception, "not registered"):
            self.fs.register_feature_view(rtfv, "v1")

    def test_register_rejects_upstream_key_outside_declared_superset(self) -> None:
        """Each upstream FV's join keys must be a subset of the RTFV's declared entities."""
        upstream_name, _src, _key = self._register_postgres_fv(suffix="EC")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        bogus_entity = Entity(name=f"bogus_{uuid.uuid4().hex[:6]}", join_keys=["SESSION_ID"])
        self.fs.register_entity(bogus_entity)

        rtfv_name = f"RTFV_INTEG_EC_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_mismatch_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=StructType([StructField("MARKER", DoubleType())]),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[bogus_entity],
            realtime_config=realtime_config,
        )

        with self.assertRaisesRegex(Exception, "must be a subset"):
            self.fs.register_feature_view(rtfv, "v1")

    def test_register_rejects_name_collision(self) -> None:
        """A second registration with the same (name, version) must fail."""
        upstream_name, _src, _key = self._register_postgres_fv(suffix="NC")
        upstream = self.fs.get_feature_view(upstream_name, "v1")

        rtfv_name = f"RTFV_INTEG_NC_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=self._make_output_schema(),
        )
        rtfv = FeatureView(
            name=rtfv_name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )

        version = "v1"
        try:
            self.fs.register_feature_view(rtfv, version)

            duplicate = FeatureView(
                name=rtfv_name,
                entities=[self.user_entity],
                realtime_config=realtime_config,
            )
            with self.assertRaisesRegex(Exception, "already exists"):
                self.fs.register_feature_view(duplicate, version)
        finally:
            self.fs.delete_feature_view(rtfv_name, version)

    # ----- generate_training_set (offline) -------------------------------
    #
    # The dataset-generation path is not gated on SNOWFLAKE_PAT because it
    # runs entirely against offline upstream tables -- map_in_pandas
    # evaluates compute_fn in the warehouse, no Online Service / HTTP.

    _SPINE_SCHEMA = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("WEIGHT", DoubleType()),
        ]
    )

    def _register_dataset_rtfv(
        self,
        *,
        suffix: str,
        upstream: FeatureView,
        rtfv_name: Optional[str] = None,
        compute_fn=_rtfv_compute_fn,
    ) -> FeatureView:
        """Register an RTFV configured for the dataset-gen path.

        Same compute_fn / output_schema shape as the online tests --
        features only, no entity keys.

        Args:
            suffix: Test-scoped name suffix.
            upstream: Registered batch FV used as the RTFV's upstream source.
            rtfv_name: Optional explicit RTFV name (default: ``RTFV_DG_<suffix>_<rand>``).
            compute_fn: Optional override for the RTFV's compute_fn (default
                ``_rtfv_compute_fn``).

        Returns:
            The registered RTFV ``FeatureView``.
        """
        name = rtfv_name or f"RTFV_DG_{suffix}_{uuid.uuid4().hex[:8].upper()}"
        realtime_config = RealtimeConfig(
            compute_fn=compute_fn,
            sources=[self._make_request_source(), upstream],
            output_schema=self._make_output_schema(),
        )
        rtfv = FeatureView(
            name=name,
            entities=[self.user_entity],
            realtime_config=realtime_config,
        )
        self.fs.register_feature_view(rtfv, "v1")
        return self.fs.get_feature_view(name, "v1")

    def test_generate_training_set_with_single_rtfv(self) -> None:
        """One RTFV, one upstream BFV: per-row outputs match compute_fn; hidden upstream cols absent."""
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="DG1")
        upstream = self.fs.get_feature_view(upstream_name, "v1")
        rtfv = self._register_dataset_rtfv(suffix="DG1", upstream=upstream)

        try:
            spine = self._session.create_dataframe([(user_id, 2.5)], schema=self._SPINE_SCHEMA)
            result = self.fs.generate_training_set(spine, [rtfv]).to_pandas()

            self.assertIn("WEIGHTED_BALANCE", result.columns)
            # Hidden upstream feature column must NOT leak into the result.
            self.assertNotIn("BALANCE", result.columns)
            # Synthetic row id must be dropped.
            self.assertNotIn("_RTFV_SPINE_ROW_ID", result.columns)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(float(result.iloc[0]["WEIGHTED_BALANCE"]), 1000.0 * 2.5)
        finally:
            self.fs.delete_feature_view(rtfv.name, rtfv.version)

    def test_generate_training_set_rtfv_duplicate_entity_keys_in_spine(self) -> None:
        """Two spine rows with the same USER_ID but different WEIGHTs both compute correctly.

        This is the synthetic-row-id correctness gate: joining the
        per-RTFV result on entity keys (instead of the synthetic id)
        would either misassociate weights or cartesian-blow-up.
        """
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="DGD")
        upstream = self.fs.get_feature_view(upstream_name, "v1")
        rtfv = self._register_dataset_rtfv(suffix="DGD", upstream=upstream)

        try:
            spine = self._session.create_dataframe(
                [(user_id, 1.5), (user_id, 3.0)],
                schema=self._SPINE_SCHEMA,
            )
            result = self.fs.generate_training_set(spine, [rtfv]).to_pandas()
            self.assertEqual(len(result), 2)
            weighted = sorted(float(v) for v in result["WEIGHTED_BALANCE"].tolist())
            self.assertEqual(weighted, [1000.0 * 1.5, 1000.0 * 3.0])
        finally:
            self.fs.delete_feature_view(rtfv.name, rtfv.version)

    def test_generate_training_set_rtfv_slice(self) -> None:
        """Slice of an RTFV projects compute_fn output to the slice's names."""
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="DGS")
        upstream = self.fs.get_feature_view(upstream_name, "v1")
        rtfv = self._register_dataset_rtfv(suffix="DGS", upstream=upstream)

        try:
            spine = self._session.create_dataframe([(user_id, 2.0)], schema=self._SPINE_SCHEMA)
            sliced = rtfv.slice(["WEIGHTED_BALANCE"])
            result = self.fs.generate_training_set(spine, [sliced]).to_pandas()
            self.assertIn("WEIGHTED_BALANCE", result.columns)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(float(result.iloc[0]["WEIGHTED_BALANCE"]), 2000.0)
        finally:
            self.fs.delete_feature_view(rtfv.name, rtfv.version)

    def test_generate_training_set_rtfv_with_auto_prefix(self) -> None:
        """``auto_prefix=True`` prefixes RTFV outputs but compute_fn still sees authored upstream names."""
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="DGP")
        upstream = self.fs.get_feature_view(upstream_name, "v1")
        rtfv = self._register_dataset_rtfv(suffix="DGP", upstream=upstream)

        try:
            spine = self._session.create_dataframe([(user_id, 4.0)], schema=self._SPINE_SCHEMA)
            result = self.fs.generate_training_set(spine, [rtfv], auto_prefix=True).to_pandas()
            # The RTFV output column should be prefixed with the RTFV's name + version.
            prefixed_cols = [c for c in result.columns if "WEIGHTED_BALANCE" in c]
            self.assertEqual(len(prefixed_cols), 1)
            self.assertNotEqual(prefixed_cols[0], "WEIGHTED_BALANCE")  # it's prefixed
            self.assertAlmostEqual(float(result.iloc[0][prefixed_cols[0]]), 4000.0)
        finally:
            self.fs.delete_feature_view(rtfv.name, rtfv.version)

    def test_generate_training_set_rtfv_save_as(self) -> None:
        """``save_as`` materializes the post-RTFV-apply DataFrame to a Snowflake Table."""
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="DGSA")
        upstream = self.fs.get_feature_view(upstream_name, "v1")
        rtfv = self._register_dataset_rtfv(suffix="DGSA", upstream=upstream)
        save_as = f"RTFV_DS_OUT_{uuid.uuid4().hex[:8].upper()}"

        try:
            spine = self._session.create_dataframe([(user_id, 1.0)], schema=self._SPINE_SCHEMA)
            saved = self.fs.generate_training_set(spine, [rtfv], save_as=save_as)
            rows = saved.to_pandas()
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows.iloc[0]["WEIGHTED_BALANCE"]), 1000.0)
        finally:
            try:
                self.fs.delete_feature_view(rtfv.name, rtfv.version)
            finally:
                self._session.sql(
                    f"DROP TABLE IF EXISTS {self.test_db}.{self.fs._config.schema.identifier()}.{save_as}"
                ).collect()

    def test_generate_training_set_interleaved_features_order(self) -> None:
        """``[rtfv_a, bfv_b, rtfv_c]`` outputs columns in original list order."""
        upstream_name, _src, user_id = self._register_postgres_fv(suffix="DGIO")
        upstream = self.fs.get_feature_view(upstream_name, "v1")
        rtfv_a = self._register_dataset_rtfv(suffix="DGIOA", upstream=upstream)
        rtfv_c = self._register_dataset_rtfv(suffix="DGIOC", upstream=upstream)

        try:
            spine = self._session.create_dataframe([(user_id, 2.0)], schema=self._SPINE_SCHEMA)
            # auto_prefix=True so each RTFV's output column has a unique name.
            result = self.fs.generate_training_set(
                spine,
                [rtfv_a, upstream, rtfv_c],
                auto_prefix=True,
            ).to_pandas()
            cols = list(result.columns)
            # Find the index of each ref's output columns.
            rtfv_a_idx = next(
                i for i, c in enumerate(cols) if rtfv_a.name.resolved() in c.upper() and "WEIGHTED" in c.upper()
            )
            upstream_idx = next(
                i for i, c in enumerate(cols) if upstream.name.resolved() in c.upper() and "BALANCE" in c.upper()
            )
            rtfv_c_idx = next(
                i for i, c in enumerate(cols) if rtfv_c.name.resolved() in c.upper() and "WEIGHTED" in c.upper()
            )
            self.assertLess(rtfv_a_idx, upstream_idx)
            self.assertLess(upstream_idx, rtfv_c_idx)
        finally:
            for fv in (rtfv_a, rtfv_c):
                try:
                    self.fs.delete_feature_view(fv.name, fv.version)
                except (snowml_exceptions.SnowflakeMLException, ValueError):
                    pass


if __name__ == "__main__":
    absltest.main()
