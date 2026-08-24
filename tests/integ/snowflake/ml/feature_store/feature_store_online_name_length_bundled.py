"""Bundled integ test for the relaxed non-tiled feature-name length limit (Postgres online store).

A non-tiled (passthrough) feature view maps each feature column 1:1 to a Postgres online
column with no tile prefix, so its name may go well beyond the 30-char cap that tiled feature
views need. This test proves a long non-tiled feature name registers and round-trips through a
**running** Online Service without silent truncation or collision at PostgreSQL's 63-byte limit.

Runs inside the spec-OFT bundle, sharing the per-shard DB / schema / Online Service that
``feature_store_spec_oft_bundle_test`` provisions in ``setUpModule``.
"""

from __future__ import annotations

import logging
import uuid

from absl.testing import absltest
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store.feature_view import (
    _POSTGRES_ONLINE_MAX_PASSTHROUGH_COLUMN_LEN,
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
)

logger = logging.getLogger(__name__)


def _make_column_of_length(length: int) -> str:
    """Return a valid Snowflake identifier of exactly ``length`` characters (starts with a letter)."""
    return "F" + "X" * (length - 1)


class FeatureStoreOnlineNameLengthIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Non-tiled Postgres FV with a long feature name: register -> materialize -> online read."""

    def _make_source_table(self, suffix: str, long_column: str, value: float) -> str:
        table = f"{self.test_db}.{self.fs._config.schema.identifier()}.LONG_NAME_SRC_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                {long_column} FLOAT
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table} VALUES
                ('u_long', DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), {value})
            """
        ).collect()
        return table

    def _run_long_name_round_trip(self, *, column_length: int, value: float) -> None:
        suffix = uuid.uuid4().hex[:8].upper()
        long_column = _make_column_of_length(column_length)
        fv_name = f"LONG_NAME_FV_{suffix}"
        entity_key = "u_long"

        src_table = self._make_source_table(suffix, long_column, value)
        feature_df = self._session.table(src_table)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )

        # FV cleanup is handled by the base class's per-test teardown (registration hook).
        try:
            registered = self.fs.register_feature_view(fv, "v1")
            self.assertTrue(registered.online)
            self.assertFalse(registered.is_tiled)

            self._wait_offline_dt_rows(self.fs, fv_name, "v1")

            resolved_column = long_column.upper()

            def _validate(pdf) -> None:
                self.assertIn(
                    resolved_column,
                    pdf.columns,
                    f"long feature column {resolved_column!r} missing from online read (columns={list(pdf.columns)})",
                )
                self.assertAlmostEqual(float(pdf.iloc[0][resolved_column]), value, places=3)

            self._poll_online_read(
                self.fs,
                fv_name,
                "v1",
                keys=[[entity_key]],
                validate_fn=_validate,
                desc=f"non-tiled long feature name (len={column_length})",
            )
        finally:
            try:
                self._session.sql(f"DROP TABLE IF EXISTS {src_table}").collect()
            except Exception as e:
                logger.warning("Source table cleanup failed: %s", e)

    def test_non_tiled_long_feature_name_online_round_trip(self) -> None:
        """A 50-char feature name (above the old 30 cap) registers and reads back online."""
        self._run_long_name_round_trip(column_length=50, value=123.5)

    def test_non_tiled_feature_name_at_max_length_online_round_trip(self) -> None:
        """A feature name at the passthrough budget exercises the 1-byte margin end-to-end."""
        self._run_long_name_round_trip(
            column_length=_POSTGRES_ONLINE_MAX_PASSTHROUGH_COLUMN_LEN,
            value=456.75,
        )


if __name__ == "__main__":
    absltest.main()
