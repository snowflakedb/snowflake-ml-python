"""Integration test for ``FeatureStore.stream_ingest`` (Online Service Ingest HTTP API).

Includes **passthrough** (raw ``AMOUNT``) and **tiled** (e.g. ``AMOUNT_SUM_2D``, ``TXN_COUNT_2D``) streaming FVs.

Registers a streaming FV with a **minimal** backfill row (probe) to satisfy ``StreamConfig`` registration,
then ingests a **different** entity key via ``stream_ingest`` and asserts Postgres online read returns that
key only (not relying on backfill data for the asserted key).

Uses ``_stream_ingest_with_retry`` to replace the former blind 2-minute post-registration sleep,
retrying the ingest call until the Online Service accepts it (stream propagation is ready).

Uses the **existing** Online Service from class setup (no per-test drop/recreate). Create/drop the service
yourself in Snowflake when needed; optional setUp create is controlled by
``SNOWML_STREAMING_FV_TEST_CREATE_ONLINE_SERVICE`` (see ``feature_store_streaming_fv_integ_base``).

Requires ``SNOWFLAKE_PAT`` (same token as the Online Service Query API), e.g.
``bazel test ... --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)``.
"""

import datetime
import os
import unittest
import uuid

import pandas as pd
from absl.testing import absltest
from common_utils import execute_snapshot_refresh
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    identity_transform,
)

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)


def _category_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform for HLL streaming FV: selects USER_ID, EVENT_TIME, CATEGORY."""
    return df[["USER_ID", "EVENT_TIME", "CATEGORY"]]


_category_transform.__module__ = "__main__"


def _multi_entity_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform for multi-entity SFV: selects USER_ID, DEVICE_ID, EVENT_TIME, AMOUNT."""
    return df[["USER_ID", "DEVICE_ID", "EVENT_TIME", "AMOUNT"]]


_multi_entity_transform.__module__ = "__main__"


class FeatureStoreStreamIngestIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Online Service ingest + online read for a key supplied only via ``stream_ingest``."""

    def _create_minimal_probe_backfill_table(self, fs: FeatureStore, suffix: str) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_PROBE_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ('probe_row', '2024-01-01 00:00:00', 0.0)
        """
        ).collect()
        return table_name

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for stream_ingest and Postgres online read.",
    )
    def test_stream_ingest_then_spec_oft_online_read_new_key(self) -> None:
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_INGEST_FV_{s}"
        self._make_stream_source(fs, stream)
        backfill_table = self._create_minimal_probe_backfill_table(fs, s)
        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        ingested_key = f"U_STREAM_{s}"
        self._stream_ingest_with_retry(
            fs,
            stream,
            {
                "USER_ID": ingested_key,
                "AMOUNT": 999.0,
                "EVENT_TIME": datetime.datetime(2024, 6, 1, 12, 0, 0),
            },
        )

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), 999.0, places=3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[ingested_key]], validate_fn=_validate, desc="stream ingest passthrough"
        )

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for stream_ingest and Postgres online read.",
    )
    def test_stream_ingest_tiled_fv_spec_oft_online_read(self) -> None:
        """Tiled streaming FV: ingest multiple raw rows; online read returns tile aggregates for that key."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_INGEST_TILED_{s}"
        self._make_stream_source(fs, stream)
        backfill_table = self._create_minimal_probe_backfill_table(fs, s)
        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )
        features = [
            Feature.sum("AMOUNT", "2d").alias("AMOUNT_SUM_2D"),
            Feature.count("AMOUNT", "2d").alias("TXN_COUNT_2D"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_streaming)
        self.assertTrue(registered.is_tiled)

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        ingested_key = f"U_TILED_{s}"
        base = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        amounts = (100.0, 200.0, 300.0)
        ingest_rows = [
            {
                "USER_ID": ingested_key,
                "AMOUNT": amt,
                "EVENT_TIME": base + datetime.timedelta(hours=i),
            }
            for i, amt in enumerate(amounts)
        ]
        self._stream_ingest_with_retry(fs, stream, ingest_rows)
        expected_sum = sum(amounts)
        expected_count = float(len(amounts))

        def _validate_tiled(pdf):
            self.assertIn("AMOUNT_SUM_2D", pdf.columns)
            self.assertIn("TXN_COUNT_2D", pdf.columns)
            row = pdf.iloc[0]
            self.assertAlmostEqual(float(row["AMOUNT_SUM_2D"]), expected_sum, places=2)
            self.assertAlmostEqual(float(row["TXN_COUNT_2D"]), expected_count, places=2)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[ingested_key]], validate_fn=_validate_tiled, desc="stream ingest tiled"
        )

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for stream_ingest and Postgres online read.",
    )
    def test_stream_ingest_tiled_continuous_fv_spec_oft_online_read(self) -> None:
        """Tiled streaming FV with CONTINUOUS aggregation: ingest rows; online read returns aggregates."""
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_INGEST_CONT_{s}"
        self._make_stream_source(fs, stream)
        backfill_table = self._create_minimal_probe_backfill_table(fs, s)
        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )
        features = [
            Feature.sum("AMOUNT", "2d").alias("AMOUNT_SUM_2D"),
            Feature.count("AMOUNT", "2d").alias("TXN_COUNT_2D"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
            feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_streaming)
        self.assertTrue(registered.is_tiled)
        self.assertEqual(registered.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        ingested_key = f"U_CONT_{s}"
        base = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        amounts = (100.0, 200.0, 300.0)
        ingest_rows = [
            {
                "USER_ID": ingested_key,
                "AMOUNT": amt,
                "EVENT_TIME": base + datetime.timedelta(minutes=i),
            }
            for i, amt in enumerate(amounts)
        ]
        self._stream_ingest_with_retry(fs, stream, ingest_rows)
        expected_sum = sum(amounts)
        expected_count = float(len(amounts))

        def _validate_continuous(pdf):
            self.assertIn("AMOUNT_SUM_2D", pdf.columns)
            self.assertIn("TXN_COUNT_2D", pdf.columns)
            row = pdf.iloc[0]
            self.assertAlmostEqual(float(row["AMOUNT_SUM_2D"]), expected_sum, places=2)
            self.assertAlmostEqual(float(row["TXN_COUNT_2D"]), expected_count, places=2)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[ingested_key]],
            validate_fn=_validate_continuous,
            desc="stream ingest continuous tiled",
        )

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for stream_ingest and Postgres online read.",
    )
    def test_stream_ingest_tiled_approx_count_distinct_online_read(self) -> None:
        """Tiled streaming FV with approx_count_distinct: ingest rows; online read returns HLL estimate."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"CAT_{s}"
        fv_name = f"STREAM_INGEST_HLL_{s}"

        fs.register_stream_source(
            StreamSource(
                name=stream,
                schema=StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
                        StructField("CATEGORY", StringType()),
                    ]
                ),
                desc="Category events stream for HLL test",
            )
        )

        backfill_table = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_HLL_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {backfill_table} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, CATEGORY VARCHAR
            )
        """
        ).collect()
        self._session.sql(
            f"INSERT INTO {backfill_table} VALUES ('probe', '2024-01-01 00:00:00', 'probe_cat')"
        ).collect()

        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=_category_transform,
            backfill_df=self._session.table(backfill_table),
        )
        features = [Feature.approx_count_distinct("CATEGORY", "2d").alias("UNIQUE_CATS_2D")]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_tiled)

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        key_a = f"U_HLL_A_{s}"
        key_b = f"U_HLL_B_{s}"
        base = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        ingest_rows = [
            {"USER_ID": key_a, "CATEGORY": "electronics", "EVENT_TIME": base + datetime.timedelta(hours=1)},
            {"USER_ID": key_a, "CATEGORY": "books", "EVENT_TIME": base + datetime.timedelta(hours=2)},
            {"USER_ID": key_a, "CATEGORY": "electronics", "EVENT_TIME": base + datetime.timedelta(hours=3)},
            {"USER_ID": key_b, "CATEGORY": "toys", "EVENT_TIME": base + datetime.timedelta(hours=1)},
        ]
        self._stream_ingest_with_retry(fs, stream, ingest_rows)

        def _validate(pdf):
            self.assertIn("UNIQUE_CATS_2D", pdf.columns)
            val = float(pdf.iloc[0]["UNIQUE_CATS_2D"])
            self.assertAlmostEqual(val, 2.0, delta=0.1)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[key_a]], validate_fn=_validate, desc="stream ingest approx_count_distinct"
        )

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for stream_ingest and Postgres online read.",
    )
    def test_stream_ingest_multi_entity_sfv(self) -> None:
        """Multi-entity streaming FV: composite key (USER_ID + DEVICE_ID), ingest + online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_ME_{s}"
        fv_name = f"STREAM_INGEST_ME_{s}"

        # Register a second entity for DEVICE_ID.
        device_entity = Entity(name=f"device_entity_{s}", join_keys=["DEVICE_ID"], desc="Device entity")
        fs.register_entity(device_entity)

        # Stream source with composite keys.
        fs.register_stream_source(
            StreamSource(
                name=stream,
                schema=StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("DEVICE_ID", StringType()),
                        StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
                        StructField("AMOUNT", DoubleType()),
                    ]
                ),
                desc="Multi-entity transaction events",
            )
        )

        # Backfill table with both keys.
        backfill_table = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_ME_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {backfill_table} (
                USER_ID VARCHAR, DEVICE_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(
            f"INSERT INTO {backfill_table} VALUES ('probe', 'dev_probe', '2024-01-01 00:00:00', 0.0)"
        ).collect()
        backfill_df = self._session.table(backfill_table)

        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=_multi_entity_transform,
            backfill_df=backfill_df,
        )
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity, device_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        ingested_user = f"U_ME_{s}"
        ingested_device = f"D_ME_{s}"
        self._stream_ingest_with_retry(
            fs,
            stream,
            {
                "USER_ID": ingested_user,
                "DEVICE_ID": ingested_device,
                "AMOUNT": 777.0,
                "EVENT_TIME": datetime.datetime(2024, 6, 1, 12, 0, 0),
            },
        )

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), 777.0, places=3)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[ingested_user, ingested_device]],
            validate_fn=_validate,
            desc="stream ingest multi-entity",
        )


class FeatureStoreAppendOnlyOFTIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Append-only feature view + Postgres OFT compatibility tests.

    Append-only FVs use a CRON-based Dynamic Table plus a per-FV snapshot table
    that accumulates each cron-tick's rows. Enabling a Postgres OFT on the same
    FV is meant to be additive: latest-per-key online reads come from the OFT,
    while PIT-correct training data continues to come from the snapshot via
    ``generate_training_set(spine_timestamp_col=...)``. This class verifies
    that compatibility end-to-end against the shared spec OFT Online Service.

    Tests use ``execute_snapshot_refresh`` (the production Task body) to drive
    snapshot refreshes synchronously, instead of waiting for the cron Task to
    fire.
    """

    def _create_append_only_source_table(self, fs: FeatureStore, suffix: str) -> str:
        """Create a 3-row source table with ``USER_ID``, ``EVENT_TIME``, ``AMOUNT``."""
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.APPEND_ONLY_SRC_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
                ('u1', '2024-01-01 00:00:00', 10.0),
                ('u2', '2024-01-01 00:00:00', 20.0),
                ('u3', '2024-01-01 00:00:00', 30.0)
        """
        ).collect()
        self.addCleanup(lambda: self._session.sql(f"DROP TABLE IF EXISTS {table_name}").collect())
        return table_name

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_append_only_fv_spec_oft_online_read_by_key(self) -> None:
        """Append-only FV with a Postgres OFT: snapshot table populated by manual refresh,
        OFT serves latest-per-key online reads alongside the snapshot table.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"APPEND_ONLY_OFT_FV_{s}"

        src_table = self._create_append_only_source_table(fs, s)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.sql(f"SELECT USER_ID, EVENT_TIME, AMOUNT FROM {src_table}"),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            append_only=True,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.append_only)
        self.assertTrue(registered.online)
        self.assertEqual(registered.online_config.store_type, OnlineStoreType.POSTGRES)

        # Snapshot table exists and is initially empty (no cron tick yet, no backup_source).
        snapshot_fqn = registered.fully_qualified_snapshot_table_name()
        self.assertEqual(self._session.table(snapshot_fqn).count(), 0)

        # OFT exists as a separate Snowflake object alongside the snapshot table.
        online_fqn = registered.fully_qualified_online_table_name()
        self.assertIsNotNone(online_fqn)
        self.assertIn("$ONLINE", online_fqn)

        # Drive a snapshot refresh synchronously via the production Task body.
        fv_physical_name = FeatureView._get_physical_name(SqlIdentifier(fv_name), registered.version)
        execute_snapshot_refresh(fs, fv_physical_name, registered.fully_qualified_name())

        # Snapshot table now carries the 3 source rows; the DT carries the same latest-per-key view.
        self.assertEqual(self._session.table(snapshot_fqn).count(), 3)
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        # OFT serves latest-per-key online reads — the snapshot enable did not block OFT population.
        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), 10.0, places=3)

        self._poll_online_read(fs, fv_name, "v1", keys=[["u1"]], validate_fn=_validate, desc="append-only OFT u1")

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_append_only_fv_pit_training_set_unchanged_by_oft(self) -> None:
        """Enabling a Postgres OFT on an append-only FV must not change the offline PIT path.

        Drives two snapshot refreshes with a source mutation between them so the
        snapshot table accumulates two distinct timestamped batches. A spine
        timestamp between the two batches must ASOF-join to the first batch
        (older values); a spine timestamp after the second batch must ASOF-join
        to the second batch (newer values). Both queries are unaffected by the
        OFT, which serves a separate latest-per-key store.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"APPEND_ONLY_PIT_OFT_FV_{s}"

        src_table = self._create_append_only_source_table(fs, s)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.sql(f"SELECT USER_ID, EVENT_TIME, AMOUNT FROM {src_table}"),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            append_only=True,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        fv_physical_name = FeatureView._get_physical_name(SqlIdentifier(fv_name), registered.version)
        fqn = registered.fully_qualified_name()

        # First snapshot batch: AMOUNT = 10 / 20 / 30 at 2024-01-01.
        execute_snapshot_refresh(fs, fv_physical_name, fqn)

        # Mutate AMOUNT and EVENT_TIME so the next snapshot is visibly distinct.
        self._session.sql(f"UPDATE {src_table} SET AMOUNT = AMOUNT * 10").collect()
        self._session.sql(f"UPDATE {src_table} SET EVENT_TIME = '2024-02-01 00:00:00'").collect()
        execute_snapshot_refresh(fs, fv_physical_name, fqn)

        # Snapshot table should now carry both batches (3 + 3 = 6 rows).
        snapshot_fqn = registered.fully_qualified_snapshot_table_name()
        self.assertEqual(self._session.table(snapshot_fqn).count(), 6)

        retrieved_fv = fs.get_feature_view(fv_name, "v1")

        # Spine between the two batches: ASOF should resolve to the first batch (AMOUNT 10/20).
        spine_between = self._session.create_dataframe(
            [("u1", "2024-01-15 00:00:00", 0), ("u2", "2024-01-15 00:00:00", 1)],
            schema=["USER_ID", "QUERY_TS", "LABEL"],
        )
        ts_between = fs.generate_training_set(
            spine_df=spine_between,
            features=[retrieved_fv],
            spine_timestamp_col="QUERY_TS",
            spine_label_cols=["LABEL"],
        )
        result_between = ts_between.to_pandas().sort_values("USER_ID").reset_index(drop=True)
        self.assertAlmostEqual(float(result_between.loc[0, "AMOUNT"]), 10.0, places=3)
        self.assertAlmostEqual(float(result_between.loc[1, "AMOUNT"]), 20.0, places=3)

        # Spine after the second batch: ASOF should resolve to the second batch (AMOUNT 100/200).
        spine_after = self._session.create_dataframe(
            [("u1", "2024-03-01 00:00:00", 0), ("u2", "2024-03-01 00:00:00", 1)],
            schema=["USER_ID", "QUERY_TS", "LABEL"],
        )
        ts_after = fs.generate_training_set(
            spine_df=spine_after,
            features=[retrieved_fv],
            spine_timestamp_col="QUERY_TS",
            spine_label_cols=["LABEL"],
        )
        result_after = ts_after.to_pandas().sort_values("USER_ID").reset_index(drop=True)
        self.assertAlmostEqual(float(result_after.loc[0, "AMOUNT"]), 100.0, places=3)
        self.assertAlmostEqual(float(result_after.loc[1, "AMOUNT"]), 200.0, places=3)

        # OFT continues to serve latest-per-key (the second batch values).
        def _validate_online(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), 100.0, places=3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[["u1"]], validate_fn=_validate_online, desc="append-only PIT + OFT latest"
        )

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_append_only_fv_with_backup_source_spec_oft_online_read(self) -> None:
        """Append-only FV with ``backup_source`` + Postgres OFT.

        At registration the snapshot table is zero-copy cloned from
        ``backup_source`` (so historical rows survive without a cron tick),
        and the OFT is provisioned in parallel. After a manual snapshot
        refresh, the OFT serves latest-per-key online reads from the
        DT-current values and the snapshot table accumulates both backup
        history and the live cron-tick batch.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"APPEND_ONLY_BF_OFT_FV_{s}"

        src_table = self._create_append_only_source_table(fs, s)

        backup_table = f"{self.test_db}.{fs._config.schema.identifier()}.APPEND_ONLY_BACKUP_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {backup_table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {backup_table} VALUES
                ('u1', '2023-06-01 00:00:00', 1.5),
                ('u2', '2023-07-01 00:00:00', 2.5),
                ('u3', '2023-08-01 00:00:00', 3.5)
        """
        ).collect()
        self.addCleanup(lambda: self._session.sql(f"DROP TABLE IF EXISTS {backup_table}").collect())

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.sql(f"SELECT USER_ID, EVENT_TIME, AMOUNT FROM {src_table}"),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            append_only=True,
            backup_source=backup_table,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.append_only)
        self.assertEqual(registered.backup_source, backup_table)
        self.assertTrue(registered.online)

        # Backup rows surface in the snapshot table immediately (no refresh required).
        snapshot_fqn = registered.fully_qualified_snapshot_table_name()
        self.assertEqual(self._session.table(snapshot_fqn).count(), 3)

        # First cron-tick refresh: snapshot now carries backup + live rows (3 + 3 = 6).
        fv_physical_name = FeatureView._get_physical_name(SqlIdentifier(fv_name), registered.version)
        execute_snapshot_refresh(fs, fv_physical_name, registered.fully_qualified_name())
        self.assertEqual(self._session.table(snapshot_fqn).count(), 6)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        # OFT serves the live (post-refresh) latest-per-key — backup rows have older
        # EVENT_TIMEs and are dominated by the live cron tick for the OFT's latest view.
        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), 10.0, places=3)

        self._poll_online_read(fs, fv_name, "v1", keys=[["u1"]], validate_fn=_validate, desc="append-only backup + OFT")

    def test_append_only_fv_manual_refresh_rejected_with_oft(self) -> None:
        """``refresh_feature_view`` is rejected for append-only FVs even when a Postgres OFT is enabled.

        The guardrail is on the append-only flag, not on the OFT — refreshes
        must always go through the scheduled Task to keep the DT and snapshot
        table consistent.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"APPEND_ONLY_REFRESH_REJECT_OFT_FV_{s}"

        src_table = self._create_append_only_source_table(fs, s)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.sql(f"SELECT USER_ID, EVENT_TIME, AMOUNT FROM {src_table}"),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            append_only=True,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.append_only)
        self.assertTrue(registered.online)

        with self.assertRaises(Exception) as ctx:
            fs.refresh_feature_view(registered)
        message = str(ctx.exception)
        self.assertIn("append_only", message)
        self.assertIn("Manual refresh is not supported", message)

    def test_append_only_fv_overwrite_rejected_with_oft(self) -> None:
        """Re-registering an append-only FV (with a Postgres OFT) using ``overwrite=True`` is rejected.

        The guardrail protects accumulated snapshot history from silent loss;
        enabling an OFT alongside the snapshot does not change that contract.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"APPEND_ONLY_OVERWRITE_REJECT_OFT_FV_{s}"

        src_table = self._create_append_only_source_table(fs, s)
        feature_sql = f"SELECT USER_ID, EVENT_TIME, AMOUNT FROM {src_table}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.sql(feature_sql),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            append_only=True,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        fs.register_feature_view(fv, "v1")

        replacement = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.sql(feature_sql),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            append_only=True,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        with self.assertRaises(Exception) as ctx:
            fs.register_feature_view(replacement, "v1", overwrite=True)
        message = str(ctx.exception)
        self.assertIn("append_only", message)
        self.assertIn("overwrite", message)


if __name__ == "__main__":
    absltest.main()
