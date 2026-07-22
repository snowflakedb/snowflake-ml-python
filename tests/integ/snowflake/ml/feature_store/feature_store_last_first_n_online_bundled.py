"""E2E integration tests for ``first_n`` / ``last_n`` (non-distinct) over the Postgres (Quake) online flow.

Covers both ingestion paths that feed an Online Feature Table:

- **Streaming**: ``stream_ingest`` raw events -> Online Service builds online tiles -> Query API returns the
  per-key last-N / first-N arrays.
- **Batch (tiled)**: source table -> tiled Dynamic Table -> reverse-ETL into Postgres -> Query API. The batch
  test also asserts the offline merge result so the snowml-side SQL is validated end-to-end.

Unlike the distinct-N variants, ``Feature.last_n`` / ``Feature.first_n`` KEEP DUPLICATES — they write the
shared, n-agnostic partial columns (``_PARTIAL_(LAST|FIRST)[_TS]_<COL>``) and slice to ``n`` at read. These
tests pin that duplicates survive.

Reuses ``StreamingFeatureViewIntegTestBase`` for the class-scoped Feature Store, ``USER_ID`` entity, and Online
Service. Requires ``SNOWFLAKE_PAT`` for the Online Service ingest / Query API.
"""

import datetime
import json
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

# Page values chosen so ordering + truncation + windowing + DUPLICATES are all observable.
_PAGE_A = "P_A"
_PAGE_B = "P_B"
_PAGE_C = "P_C"
_PAGE_Z = "P_Z"  # out-of-window (day -6); must never surface

# In-window chronological events (oldest -> newest), spanning two tiles; last_n/first_n do NOT dedupe:
#   day -2:  A@1h, B@2h
#   day -1:  B@1h, C@2h
#   day -6:  Z          -> dropped by the 4d window
# Full in-window sequence oldest->newest: A, B, B, C.  Output is ALWAYS ascending by timestamp
# (oldest-first); "first"/"last" only selects WHICH n survive. n=3 spans both tiles and keeps duplicates:
#   last_n=3  (3 most-recent raw: B@d2+2h, B@d1+1h, C@d1+2h) -> [P_B, P_B, P_C]  (B twice)
#   first_n=3 (3 oldest raw:      A@d2+1h, B@d2+2h, B@d1+1h) -> [P_A, P_B, P_B]  (B twice)
# Both expected arrays are NON-palindromic, so a newest-first regression (which would yield
# [P_C, P_B, P_B]) fails the assertion; both contain a duplicate, pinning last_n vs last_distinct_n.
_EXPECTED_LAST_N_3 = [_PAGE_B, _PAGE_B, _PAGE_C]
_EXPECTED_FIRST_N_3 = [_PAGE_A, _PAGE_B, _PAGE_B]

# Guardrails: direction must be observable (non-palindrome) and duplicates must be present.
assert _EXPECTED_LAST_N_3 != list(reversed(_EXPECTED_LAST_N_3)), "last_n expected must be non-palindromic"
assert _EXPECTED_FIRST_N_3 != list(reversed(_EXPECTED_FIRST_N_3)), "first_n expected must be non-palindromic"
assert len(set(_EXPECTED_LAST_N_3)) < len(_EXPECTED_LAST_N_3), "last_n expected must contain a duplicate"
assert len(set(_EXPECTED_FIRST_N_3)) < len(_EXPECTED_FIRST_N_3), "first_n expected must contain a duplicate"


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


class FeatureStoreLastFirstNStreamingIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Streaming last_n/first_n: ingest events, read per-key arrays (duplicates kept) from the Postgres OFT."""

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
                desc="Page view events stream for last_n/first_n test",
            )
        )

    def _create_page_url_backfill_probe(self, fs: FeatureStore, suffix: str) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_ORDERED_{suffix}"
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

    def test_stream_ingest_last_first_n_online_read(self) -> None:
        """Tiled streaming FV with last_n/first_n: ingest events; online read returns arrays with duplicates."""
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        stream = f"PAGE_{s}"
        fv_name = f"STREAM_ORDERED_N_{s}"

        self._register_page_url_stream(fs, stream)
        backfill_df = self._session.table(self._create_page_url_backfill_probe(fs, s))
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=_page_url_transform,
            backfill_df=backfill_df,
        )
        features = [
            Feature.last_n("PAGE_URL", "4d", n=3).alias("RECENT_PAGES"),
            Feature.first_n("PAGE_URL", "4d", n=3).alias("FIRST_PAGES"),
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

        ingested_key = f"U_ORDERED_{s}"
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        day1 = today_start - datetime.timedelta(days=1)  # recent tile
        day2 = today_start - datetime.timedelta(days=2)  # older tile, inside 4d window
        day6 = today_start - datetime.timedelta(days=6)  # outside 4d window
        ingest_rows = [
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_Z, "EVENT_TIME": day6 + datetime.timedelta(hours=1)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_A, "EVENT_TIME": day2 + datetime.timedelta(hours=1)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_B, "EVENT_TIME": day2 + datetime.timedelta(hours=2)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_B, "EVENT_TIME": day1 + datetime.timedelta(hours=1)},
            {"USER_ID": ingested_key, "PAGE_URL": _PAGE_C, "EVENT_TIME": day1 + datetime.timedelta(hours=2)},
        ]
        self._stream_ingest_with_retry(fs, stream, ingest_rows)

        def _validate(pdf: pd.DataFrame) -> None:
            self.assertIn("RECENT_PAGES", pdf.columns)
            self.assertIn("FIRST_PAGES", pdf.columns)
            last = _as_list(pdf.iloc[0]["RECENT_PAGES"])
            first = _as_list(pdf.iloc[0]["FIRST_PAGES"])
            # Cross-tile merge + window + truncate-to-n, ascending ts, duplicates kept:
            # last=[B, B, C], first=[A, B, B]  (Z is out of window).
            self.assertEqual(last, _EXPECTED_LAST_N_3)
            self.assertEqual(first, _EXPECTED_FIRST_N_3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[ingested_key]], validate_fn=_validate, desc="stream ingest last_n/first_n"
        )


class FeatureStoreLastFirstNBatchIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Batch tiled last_n/first_n: source table -> tiled DT -> Postgres OFT; asserts offline merge + online read."""

    def _create_page_url_source_table(self, fs: FeatureStore, suffix: str, entity_key: str) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_ORDERED_SRC_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, PAGE_URL VARCHAR
            )
        """
        ).collect()
        # Anchor to UTC day boundaries so tiles align with the online store's UTC window.
        today = "DATE_TRUNC('day', CONVERT_TIMEZONE('UTC', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ)"
        day1 = f"DATEADD('day', -1, {today})"  # recent tile
        day2 = f"DATEADD('day', -2, {today})"  # older tile, inside the window
        day6 = f"DATEADD('day', -6, {today})"  # outside the window
        self._session.sql(
            f"""
            INSERT INTO {table_name}
            SELECT column1, column2, column3 FROM VALUES
                ({entity_key!r}, DATEADD('hour', 1, {day6}), {_PAGE_Z!r}),
                ({entity_key!r}, DATEADD('hour', 1, {day2}), {_PAGE_A!r}),
                ({entity_key!r}, DATEADD('hour', 2, {day2}), {_PAGE_B!r}),
                ({entity_key!r}, DATEADD('hour', 1, {day1}), {_PAGE_B!r}),
                ({entity_key!r}, DATEADD('hour', 2, {day1}), {_PAGE_C!r})
        """
        ).collect()
        return table_name

    def test_batch_tiled_last_first_n_offline_and_online_read(self) -> None:
        """Tiled batch FV with last_n/first_n: validate offline merge + Postgres online read (duplicates kept)."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_ORDERED_N_{s}"
        batch_key = f"U_BATCH_ORDERED_{s}"

        src_table = self._create_page_url_source_table(fs, s, batch_key)
        features = [
            Feature.last_n("PAGE_URL", "4d", n=3).alias("RECENT_PAGES"),
            Feature.first_n("PAGE_URL", "4d", n=3).alias("FIRST_PAGES"),
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

        # Force a synchronous offline refresh so the tile Dynamic Table is materialized immediately,
        # then validate the snowml-side merge SQL against the warehouse (offline read).
        fv_live = fs.get_feature_view(fv_name, "v1")
        fs.refresh_feature_view(fv_live, store_type=StoreType.OFFLINE)

        opdf = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE, keys=[[batch_key]]).to_pandas()
        self.assertIn("RECENT_PAGES", opdf.columns)
        self.assertIn("FIRST_PAGES", opdf.columns)
        self.assertEqual(_as_list(opdf.iloc[0]["RECENT_PAGES"]), _EXPECTED_LAST_N_3)
        self.assertEqual(_as_list(opdf.iloc[0]["FIRST_PAGES"]), _EXPECTED_FIRST_N_3)

        def _validate_online(pdf: pd.DataFrame) -> None:
            self.assertIn("RECENT_PAGES", pdf.columns)
            self.assertIn("FIRST_PAGES", pdf.columns)
            last = _as_list(pdf.iloc[0]["RECENT_PAGES"])
            first = _as_list(pdf.iloc[0]["FIRST_PAGES"])
            self.assertEqual(last, _EXPECTED_LAST_N_3)
            self.assertEqual(first, _EXPECTED_FIRST_N_3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate_online, desc="batch tiled last_n/first_n"
        )


if __name__ == "__main__":
    absltest.main()
