"""Bundled integ test for the non-tiled batch OFT spec VARCHAR-length fix.

Registers a batch FV whose source query reports an unconstrained VARCHAR
length, then registers an RTFV on top. Pre-fix this combination failed with
a server-side schema-mismatch error.

Runs inside the spec-OFT bundle, sharing the per-shard DB / schema / Online
Service that ``feature_store_spec_oft_bundle_test`` provisions in
``setUpModule``.
"""

from __future__ import annotations

import logging
import uuid

import pandas as pd
from absl.testing import absltest
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store.feature_group import FeatureGroup
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import DoubleType, StructField, StructType

logger = logging.getLogger(__name__)


def _rtfv_compute_fn(request_df: pd.DataFrame, upstream_df: pd.DataFrame) -> pd.DataFrame:
    weight = request_df["WEIGHT"].astype(float).reset_index(drop=True)
    total = upstream_df["TOTAL_AMOUNT"].fillna(0.0).reset_index(drop=True)
    return pd.DataFrame({"WEIGHTED_TOTAL": total * weight})


class FeatureStoreOftVarcharLengthIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    def _make_source_table(self, suffix: str) -> str:
        table = f"{self.test_db}.{self.fs._config.schema.identifier()}.OFT_VARCHAR_SRC_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table} (
                USER_ID VARCHAR,
                DEVICE_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table} VALUES
                ('u1', 'd1', DATEADD('minute', -10, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 10.0),
                ('u1', 'd1', DATEADD('minute', -8,  CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 5.0),
                ('u2', 'd2', DATEADD('minute', -7,  CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 7.5)
            """
        ).collect()
        return table

    def test_rtfv_over_non_tiled_batch_fv_with_unconstrained_varchar_registers(self) -> None:
        suffix = uuid.uuid4().hex[:8].upper()
        src_table = self._make_source_table(suffix)
        batch_fv_name = f"OFT_VARCHAR_BATCH_{suffix}"
        rtfv_name = f"OFT_VARCHAR_RTFV_{suffix}"

        feature_df = self._session.sql(
            f"""
            SELECT
                USER_ID,
                TO_JSON(ARRAY_AGG(DISTINCT DEVICE_ID)) AS DEVICE_LIST,
                MAX(EVENT_TIME) AS EVENT_TIME,
                SUM(AMOUNT) AS TOTAL_AMOUNT
            FROM {src_table}
            GROUP BY USER_ID
            """
        )
        device_list_field = next(f for f in feature_df.schema.fields if f.name.upper().strip('"') == "DEVICE_LIST")
        self.assertEqual(
            device_list_field.datatype.length,
            134217728,
            "Precondition broken: TO_JSON(...) no longer reports an unconstrained VARCHAR length.",
        )

        batch_fv = FeatureView(
            name=batch_fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )

        # FV cleanup is handled by the base class's per-test teardown (registration hook).
        try:
            registered_batch = self.fs.register_feature_view(batch_fv, "v1")
            self.assertIsNotNone(registered_batch)

            upstream = self.fs.get_feature_view(batch_fv_name, "v1")

            request_source = RequestSource(schema=StructType([StructField("WEIGHT", DoubleType())]))
            output_schema = StructType([StructField("WEIGHTED_TOTAL", DoubleType())])
            realtime_config = RealtimeConfig(
                compute_fn=_rtfv_compute_fn,
                sources=[request_source, upstream],
                output_schema=output_schema,
            )
            rtfv = FeatureView(
                name=rtfv_name,
                entities=[self.user_entity],
                realtime_config=realtime_config,
                desc="OFT VARCHAR length regression",
            )

            registered_rtfv = self.fs.register_feature_view(rtfv, "v1")
            self.assertIsNotNone(registered_rtfv)
            self.assertTrue(registered_rtfv.is_realtime_feature_view)
        finally:
            try:
                self._session.sql(f"DROP TABLE IF EXISTS {src_table}").collect()
            except Exception as e:
                logger.warning("Source table cleanup failed: %s", e)

    def test_feature_group_over_non_tiled_batch_fv_with_unconstrained_varchar_registers(self) -> None:
        """A FeatureGroup over a batch FV that exposes an unconstrained-VARCHAR
        feature must register: the FG source column shape comes from the
        upstream's materialized DT schema, matching the upstream's stored
        OutputColumn for the Online Service exact-shape check.
        """
        suffix = uuid.uuid4().hex[:8].upper()
        src_table = self._make_source_table(suffix)
        batch_fv_name = f"OFT_VARCHAR_FG_BATCH_{suffix}"
        fg_name = f"OFT_VARCHAR_FG_{suffix}"

        feature_df = self._session.sql(
            f"""
            SELECT
                USER_ID,
                TO_JSON(ARRAY_AGG(DISTINCT DEVICE_ID)) AS DEVICE_LIST,
                MAX(EVENT_TIME) AS EVENT_TIME,
                SUM(AMOUNT) AS TOTAL_AMOUNT
            FROM {src_table}
            GROUP BY USER_ID
            """
        )
        device_list_field = next(f for f in feature_df.schema.fields if f.name.upper().strip('"') == "DEVICE_LIST")
        self.assertEqual(
            device_list_field.datatype.length,
            134217728,
            "Precondition broken: TO_JSON(...) no longer reports an unconstrained VARCHAR length.",
        )

        batch_fv = FeatureView(
            name=batch_fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )

        # FV/FG cleanup is handled by the base class's per-test teardown (registration hook).
        try:
            registered_batch = self.fs.register_feature_view(batch_fv, "v1")
            self.assertIsNotNone(registered_batch)

            upstream = self.fs.get_feature_view(batch_fv_name, "v1")

            fg = FeatureGroup(
                name=fg_name,
                features=[upstream],
                desc="OFT VARCHAR length regression (FeatureGroup)",
                auto_prefix=True,
            )
            registered_fg = self.fs.register_feature_group(fg, "v1")
            self.assertIsNotNone(registered_fg)
            self.assertEqual(registered_fg.name, fg_name)
        finally:
            try:
                self._session.sql(f"DROP TABLE IF EXISTS {src_table}").collect()
            except Exception as e:
                logger.warning("Source table cleanup failed: %s", e)


if __name__ == "__main__":
    absltest.main()
