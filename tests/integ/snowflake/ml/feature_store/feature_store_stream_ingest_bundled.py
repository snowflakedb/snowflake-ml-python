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
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    identity_transform,
)

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


if __name__ == "__main__":
    absltest.main()
