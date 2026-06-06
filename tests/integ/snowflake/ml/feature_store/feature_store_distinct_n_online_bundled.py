"""E2E integration tests for ``first_distinct_n`` / ``last_distinct_n`` over the Postgres (Quake) online flow.

Covers both ingestion paths that feed an Online Feature Table:

- **Streaming**: ``stream_ingest`` raw events -> Online Service builds online tiles -> Query API returns the
  per-key distinct-N arrays.
- **Batch (tiled)**: source table -> tiled Dynamic Table -> reverse-ETL into Postgres -> Query API. The batch
  test also asserts the offline merge result so the snowml-side SQL is validated end-to-end.

``Feature.last_distinct_n`` / ``Feature.first_distinct_n`` dedupe and
truncate to ``n`` at tile-build time and write the Quake-contract partial columns
(``_PARTIAL_(LAST|FIRST)_DISTINCT_<N>[_TS]_<COL>``).

Reuses ``StreamingFeatureViewIntegTestBase`` for the class-scoped Feature Store, ``USER_ID`` entity, and Online
Service. Requires ``SNOWFLAKE_PAT`` for the Online Service ingest / Query API, e.g.
``bazel test ... --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)``.
"""

import datetime
import json
import os
import unittest
import uuid
from typing import Any, Optional

import pandas as pd
from absl.testing import absltest
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
)
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark.types import (
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)

# Distinct page values; chosen so dedupe + ordering + truncation + windowing are all observable.
_PAGE_A = "P_A"  # only in the recent tile (day -1)
_PAGE_B = "P_B"  # only in the recent tile (day -1)
_PAGE_C = "P_C"  # only in the older in-window tile (day -2)
_PAGE_D = "P_D"  # only in the older in-window tile (day -2)
_PAGE_Z = "P_Z"  # only in the out-of-window tile (day -6); must never surface

# Events span three daily tiles within a 4d window / 1d granularity. Each in-window tile has only 2
# distinct values, so n=3 forces every result to combine distinct values from BOTH in-window tiles:
#   day -2 (in-window):  C, D, C   (distinct C, D)
#   day -1 (recent):     A, B, A   (distinct A, B)
#   day -6 (too old):    Z         -> dropped by the window
# Both first and last distinct-N return values in ASCENDING timestamp order:
#   first = 3 oldest distinct      (C@day-2, D@day-2, A@day-1)     -> [C, D, A]
#   last  = 3 most-recent distinct (C@day-2, B@day-1, A@day-1)     -> [C, B, A]
# Each result mixes the day -2 and day -1 tiles; Z would surface if the window leaked.
_EXPECTED_LAST_DISTINCT_3 = [_PAGE_C, _PAGE_B, _PAGE_A]
_EXPECTED_FIRST_DISTINCT_3 = [_PAGE_C, _PAGE_D, _PAGE_A]


def _page_url_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Stream transform selecting USER_ID, EVENT_TIME, PAGE_URL."""
    return df[["USER_ID", "EVENT_TIME", "PAGE_URL"]]


# See base module note: __module__ = "__main__" forces cloudpickle by-value serialization.
_page_url_transform.__module__ = "__main__"


def _as_list(value: Any) -> Optional[list]:
    """Normalize an array column value (Python list or JSON string) to a list."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


class FeatureStoreDistinctNStreamingIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Streaming distinct-N: ingest events, read per-key distinct-N arrays from the Postgres OFT."""

    def _register_page_url_stream(self, fs: FeatureStore, stream_name: str) -> None:
        fs.register_stream_source(
            StreamSource(
                name=stream_name,
                schema=StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
                        StructField("PAGE_URL", StringType()),
                    ]
                ),
                desc="Page view events stream for distinct-N test",
            )
        )

    def _create_page_url_backfill_probe(self, fs: FeatureStore, suffix: str) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_DISTINCT_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, PAGE_URL VARCHAR
            )
        """
        ).collect()
        self._session.sql(
            f"INSERT INTO {table_name} VALUES ('probe_row', '2024-01-01 00:00:00', 'probe_page')"
        ).collect()
        return table_name

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for stream_ingest and Postgres online read.",
    )
    def test_stream_ingest_distinct_n_online_read(self) -> None:
        """Tiled streaming FV with first/last distinct-N: ingest events; online read returns distinct arrays."""
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"PAGE_{s}"
        fv_name = f"STREAM_DISTINCT_N_{s}"

        self._register_page_url_stream(fs, stream)
        backfill_df = self._session.table(self._create_page_url_backfill_probe(fs, s))
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=_page_url_transform,
            backfill_df=backfill_df,
        )
        features = [
            Feature.last_distinct_n("PAGE_URL", "4d", n=3).alias("RECENT_DISTINCT_PAGES"),
            Feature.first_distinct_n("PAGE_URL", "4d", n=3).alias("FIRST_DISTINCT_PAGES"),
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

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        ingested_key = f"U_DISTINCT_{s}"
        # Anchor on start-of-day so per-day events never cross a midnight tile boundary.
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        day1 = today_start - datetime.timedelta(days=1)  # recent tile
        day2 = today_start - datetime.timedelta(days=2)  # older tile, still inside the 4d window
        day6 = today_start - datetime.timedelta(days=6)  # outside the 4d window
        ingest_rows = [
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_Z, "EVENT_TIME": day6 + datetime.timedelta(hours=1)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_C, "EVENT_TIME": day2 + datetime.timedelta(hours=1)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_D, "EVENT_TIME": day2 + datetime.timedelta(hours=2)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_C, "EVENT_TIME": day2 + datetime.timedelta(hours=3)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_A, "EVENT_TIME": day1 + datetime.timedelta(hours=1)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_B, "EVENT_TIME": day1 + datetime.timedelta(hours=2)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_A, "EVENT_TIME": day1 + datetime.timedelta(hours=3)},
        ]
        self._stream_ingest_with_retry(fs, stream, ingest_rows)

        def _validate(pdf: pd.DataFrame) -> None:
            self.assertIn("RECENT_DISTINCT_PAGES", pdf.columns)
            self.assertIn("FIRST_DISTINCT_PAGES", pdf.columns)
            last = _as_list(pdf.iloc[0]["RECENT_DISTINCT_PAGES"])
            first = _as_list(pdf.iloc[0]["FIRST_DISTINCT_PAGES"])
            # Cross-tile merge + window + dedupe + truncate-to-n, ascending ts: last=[C, B, A], first=[C, D, A]
            # (Z is out of window).
            self.assertEqual(last, _EXPECTED_LAST_DISTINCT_3)
            self.assertEqual(first, _EXPECTED_FIRST_DISTINCT_3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[ingested_key]], validate_fn=_validate, desc="stream ingest distinct-N"
        )


class FeatureStoreDistinctNBatchIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Batch tiled distinct-N: source table -> tiled DT -> Postgres OFT; asserts offline merge + online read."""

    def _create_page_url_source_table(self, fs: FeatureStore, suffix: str, entity_key: str) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_DISTINCT_SRC_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, PAGE_URL VARCHAR
            )
        """
        ).collect()
        # Daily tiles anchored on start-of-day. The 4d window covers day -1..day -4, so day -2 is
        # included while day -6 is excluded.
        today = "DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ)"
        day1 = f"DATEADD('day', -1, {today})"  # recent tile
        day2 = f"DATEADD('day', -2, {today})"  # older tile, inside the window
        day6 = f"DATEADD('day', -6, {today})"  # outside the window
        self._session.sql(
            f"""
            INSERT INTO {table_name}
            SELECT column1, column2, column3 FROM VALUES
                ({entity_key!r}, DATEADD('hour', 1, {day6}), {_PAGE_Z!r}),
                ({entity_key!r}, DATEADD('hour', 1, {day2}), {_PAGE_C!r}),
                ({entity_key!r}, DATEADD('hour', 2, {day2}), {_PAGE_D!r}),
                ({entity_key!r}, DATEADD('hour', 3, {day2}), {_PAGE_C!r}),
                ({entity_key!r}, DATEADD('hour', 1, {day1}), {_PAGE_A!r}),
                ({entity_key!r}, DATEADD('hour', 2, {day1}), {_PAGE_B!r}),
                ({entity_key!r}, DATEADD('hour', 3, {day1}), {_PAGE_A!r})
        """
        ).collect()
        return table_name

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for Postgres online read (Online Service Query API).",
    )
    def test_batch_tiled_distinct_n_offline_and_online_read(self) -> None:
        """Tiled batch FV with first/last distinct-N: validate offline merge + Postgres online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_DISTINCT_N_{s}"
        batch_key = f"U_BATCH_DISTINCT_{s}"

        src_table = self._create_page_url_source_table(fs, s, batch_key)
        features = [
            Feature.last_distinct_n("PAGE_URL", "4d", n=3).alias("RECENT_DISTINCT_PAGES"),
            Feature.first_distinct_n("PAGE_URL", "4d", n=3).alias("FIRST_DISTINCT_PAGES"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.table(src_table),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertFalse(registered.is_streaming)
        self.assertTrue(registered.is_tiled)
        self.assertTrue(registered.online)

        # Force a synchronous offline refresh so the tile Dynamic Table is materialized immediately
        # (instead of waiting for the scheduled lag), then validate the snowml-side merge SQL against
        # the warehouse (offline read).
        fv_live = fs.get_feature_view(fv_name, "v1")
        fs.refresh_feature_view(fv_live, store_type=StoreType.OFFLINE)

        opdf = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE, keys=[[batch_key]]).to_pandas()
        self.assertIn("RECENT_DISTINCT_PAGES", opdf.columns)
        self.assertIn("FIRST_DISTINCT_PAGES", opdf.columns)
        self.assertEqual(_as_list(opdf.iloc[0]["RECENT_DISTINCT_PAGES"]), _EXPECTED_LAST_DISTINCT_3)
        self.assertEqual(_as_list(opdf.iloc[0]["FIRST_DISTINCT_PAGES"]), _EXPECTED_FIRST_DISTINCT_3)

        def _validate_online(pdf: pd.DataFrame) -> None:
            self.assertIn("RECENT_DISTINCT_PAGES", pdf.columns)
            self.assertIn("FIRST_DISTINCT_PAGES", pdf.columns)
            last = _as_list(pdf.iloc[0]["RECENT_DISTINCT_PAGES"])
            first = _as_list(pdf.iloc[0]["FIRST_DISTINCT_PAGES"])
            self.assertEqual(last, _EXPECTED_LAST_DISTINCT_3)
            self.assertEqual(first, _EXPECTED_FIRST_DISTINCT_3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate_online, desc="batch tiled distinct-N"
        )


if __name__ == "__main__":
    absltest.main()
