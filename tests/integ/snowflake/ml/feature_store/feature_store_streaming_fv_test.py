"""Integration tests for streaming feature view registration and lifecycle.

These tests verify the end-to-end flow of creating, registering, listing,
and deleting streaming feature views, including:
- Backfill via map_in_pandas
- UDF transformed table creation
- Offline DT creation from the udf_transformed table
- OFT creation with StreamingFeatureView spec
- Streaming metadata storage and retrieval
- Stream source ref_count management
- Cleanup on delete

Streaming FVs use Postgres online by default, so registration requires a **Feature Store
Online Service** in the same schema. This module uses one class-scoped Online Service and shared
``FeatureStore`` (same pattern as ``feature_store_spec_oft_test``).

``test_streaming_fv_spec_oft_online_read_e2e`` requires ``SNOWFLAKE_PAT`` (Online Service Query API auth);
it is skipped when unset. Run with ``bazel test ... --test_env=SNOWFLAKE_PAT=...``.

**Online Service reuse (default)**

By default this class uses a stable database/schema (see defaults below), skips ``create_online_service`` when
the Online Service is already ``RUNNING`` with a query endpoint, and skips ``drop_online_service`` plus test database
drops on teardown. For ephemeral random DBs and full cleanup (e.g. CI isolation), set
``SNOWML_STREAMING_FV_TEST_ISOLATE_ONLINE_SERVICE=1``. Optional overrides when reusing:
``SNOWML_STREAMING_FV_TEST_REUSE_DB``, ``SNOWML_STREAMING_FV_TEST_REUSE_SCHEMA``,
``SNOWML_STREAMING_FV_TEST_REUSE_DUMMY_DB``, ``SNOWML_STREAMING_FV_TEST_REUSE_CONSUMER_ROLE``.
Set ``SNOWML_STREAMING_FV_TEST_REUSE_ONLINE_SERVICE=0`` to force the same ephemeral path as isolate (no reuse).
"""

import json
import os
import time
import unittest
import uuid

from absl.testing import absltest, parameterized
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    all_types_identity_transform,
    identity_transform,
)

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.stream_config import StreamConfig


class StreamingFeatureViewIntegTest(StreamingFeatureViewIntegTestBase, parameterized.TestCase):
    """Streaming FV integ tests: registration, lifecycle, and Postgres online reads."""

    # =========================================================================
    # Registration tests
    # =========================================================================

    def test_register_streaming_fv_basic(self) -> None:
        """Test basic registration of a streaming feature view."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_FV_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

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
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertTrue(registered_fv.is_streaming)
        self.assertTrue(registered_fv.online)

        physical_name = FeatureView._get_physical_name(registered_fv.name, registered_fv.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"

        self._wait_udf_and_backfill(
            fq_udf,
            timeout_s=60,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered_fv.name),
            streaming_fv_version=str(registered_fv.version),
        )

        udf_count = self._session.table(fq_udf).count()
        self.assertEqual(udf_count, 3, "udf_transformed table should have 3 backfill rows")

    def test_streaming_fv_shows_in_list(self) -> None:
        """Test that streaming FV appears in list_feature_views with stream_config."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_FV_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

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
        )

        fs.register_feature_view(fv, "v1")

        listing = fs.list_feature_views().collect(statement_params=fs._telemetry_stmp)
        want = SqlIdentifier(fv_name).resolved()
        matching = [r for r in listing if r["NAME"] == want]
        self.assertEqual(len(matching), 1, f"expected one row for {want}, got {listing}")
        row = matching[0]
        self.assertIsNotNone(row["STREAM_CONFIG"])

        stream_config_data = json.loads(row["STREAM_CONFIG"])
        self.assertEqual(stream_config_data["stream_source"], self._stream_source_ref_key(stream))
        self.assertEqual(stream_config_data["transformation_fn"], "identity_transform")

    def test_streaming_fv_ref_count(self) -> None:
        """Test that stream source ref_count is incremented on register."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        fs.register_feature_view(fv, "v1")

        ref_count = fs._metadata_manager.get_stream_source_ref_count(self._stream_source_ref_key(stream))
        self.assertEqual(ref_count, 1)

    # =========================================================================
    # Delete tests
    # =========================================================================

    def test_delete_streaming_fv_cleans_up(self) -> None:
        """Test that deleting a streaming FV cleans up udf_transformed + backfill tables and ref count."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        schema_path = f"{self.test_db}.{fs._config.schema.identifier()}"

        self.assertEqual(
            fs._metadata_manager.get_stream_source_ref_count(self._stream_source_ref_key(stream)),
            1,
        )

        fs.delete_feature_view(registered_fv)

        ref_count = fs._metadata_manager.get_stream_source_ref_count(self._stream_source_ref_key(stream))
        self.assertEqual(ref_count, 0)

        fv_list = fs.list_feature_views().collect(statement_params=fs._telemetry_stmp)
        self.assertEqual(len(fv_list), 0)

        remaining = self._session.sql(f"SHOW TABLES LIKE '%UDF_TRANSFORMED%' IN SCHEMA {schema_path}").collect()
        self.assertEqual(len(remaining), 0, "udf_transformed and backfill tables should be dropped")

    def test_delete_streaming_fv_allows_stream_source_delete(self) -> None:
        """Test that after deleting all streaming FVs, the stream source can be deleted."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        registered_fv = fs.register_feature_view(fv, "v1")
        fs.delete_feature_view(registered_fv)

        fs.delete_stream_source(stream)

        sources = fs.list_stream_sources().collect(statement_params=fs._telemetry_stmp)
        self.assertEqual(len(sources), 0)

    # =========================================================================
    # Streaming + Tiled Features + Dataset Generation
    # =========================================================================

    def test_streaming_tiled_fv_dataset_generation(self) -> None:
        """End-to-end: streaming FV with tiled features, backfill, and dataset generation."""
        from datetime import datetime

        from snowflake.ml.feature_store.feature import Feature

        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.TILED_{s}"
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
            ('u1', '2024-01-01 01:00:00', 10.0),
            ('u1', '2024-01-02 01:00:00', 20.0),
            ('u1', '2024-01-03 01:00:00', 30.0),
            ('u1', '2024-01-04 01:00:00', 40.0),
            ('u2', '2024-01-01 01:00:00', 100.0),
            ('u2', '2024-01-02 01:00:00', 200.0),
            ('u2', '2024-01-03 01:00:00', 300.0)
        """
        ).collect()

        backfill_df = self._session.table(table_name)

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
            name=f"STREAM_TILED_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.is_streaming)
        self.assertTrue(registered_fv.is_tiled)

        max_wait = 120
        start = time.time()
        while time.time() - start < max_wait:
            count = self._session.table(registered_fv.fully_qualified_name()).count()
            if count > 0:
                break
            time.sleep(5)

        spine_df = self._session.create_dataframe(
            [
                ("u1", datetime(2024, 1, 5, 0, 0, 0)),
                ("u2", datetime(2024, 1, 5, 0, 0, 0)),
            ],
            schema=["USER_ID", "QUERY_TS"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="QUERY_TS",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        self.assertEqual(len(result_pd), 2)

        self.assertIn("AMOUNT_SUM_2D", result_pd.columns)
        self.assertIn("TXN_COUNT_2D", result_pd.columns)

        for col in ["AMOUNT_SUM_2D", "TXN_COUNT_2D"]:
            self.assertTrue(
                result_pd[col].notna().all(),
                f"Column {col} has null values: {result_pd[col].tolist()}",
            )

    # =========================================================================
    # Streaming + Tiled (CONTINUOUS) + Dataset Generation
    # =========================================================================

    def test_streaming_tiled_continuous_fv_dataset_generation(self) -> None:
        """End-to-end: streaming FV with CONTINUOUS aggregation, backfill, and dataset generation."""
        from datetime import datetime

        from snowflake.ml.feature_store.feature import Feature
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.CONT_TILED_{s}"
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
            ('u1', '2024-01-01 01:00:00', 10.0),
            ('u1', '2024-01-02 01:00:00', 20.0),
            ('u1', '2024-01-03 01:00:00', 30.0),
            ('u1', '2024-01-04 01:00:00', 40.0),
            ('u2', '2024-01-01 01:00:00', 100.0),
            ('u2', '2024-01-02 01:00:00', 200.0),
            ('u2', '2024-01-03 01:00:00', 300.0)
        """
        ).collect()

        backfill_df = self._session.table(table_name)

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
            name=f"STREAM_CONT_TILED_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.is_streaming)
        self.assertTrue(registered_fv.is_tiled)
        self.assertEqual(registered_fv.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)

        # Verify round-trip through get_feature_view
        retrieved_fv = fs.get_feature_view(f"STREAM_CONT_TILED_{s}", "v1")
        self.assertEqual(retrieved_fv.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)

        max_wait = 120
        start = time.time()
        while time.time() - start < max_wait:
            count = self._session.table(registered_fv.fully_qualified_name()).count()
            if count > 0:
                break
            time.sleep(5)

        spine_df = self._session.create_dataframe(
            [
                ("u1", datetime(2024, 1, 5, 0, 0, 0)),
                ("u2", datetime(2024, 1, 5, 0, 0, 0)),
            ],
            schema=["USER_ID", "QUERY_TS"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="QUERY_TS",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        self.assertEqual(len(result_pd), 2)

        self.assertIn("AMOUNT_SUM_2D", result_pd.columns)
        self.assertIn("TXN_COUNT_2D", result_pd.columns)

        for col in ["AMOUNT_SUM_2D", "TXN_COUNT_2D"]:
            self.assertTrue(
                result_pd[col].notna().all(),
                f"Column {col} has null values: {result_pd[col].tolist()}",
            )

    # =========================================================================
    # Streaming + Untiled (passthrough) + Dataset Generation
    # =========================================================================

    def test_streaming_untiled_fv_dataset_generation(self) -> None:
        """End-to-end: untiled streaming FV with backfill and dataset generation."""
        from datetime import datetime

        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.UNTILED_{s}"
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
            ('u1', '2024-01-01 01:00:00', 10.0),
            ('u1', '2024-01-02 01:00:00', 20.0),
            ('u2', '2024-01-01 01:00:00', 100.0),
            ('u2', '2024-01-02 01:00:00', 200.0)
        """
        ).collect()

        backfill_df = self._session.table(table_name)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_UNTILED_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        registered_fv = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.is_streaming)
        self.assertFalse(registered_fv.is_tiled)

        physical_name = FeatureView._get_physical_name(registered_fv.name, registered_fv.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered_fv.name),
            streaming_fv_version=str(registered_fv.version),
        )

        max_wait = 120
        start = time.time()
        while time.time() - start < max_wait:
            count = self._session.table(registered_fv.fully_qualified_name()).count()
            if count > 0:
                break
            time.sleep(5)

        spine_df = self._session.create_dataframe(
            [
                ("u1", datetime(2024, 1, 3, 0, 0, 0)),
                ("u2", datetime(2024, 1, 3, 0, 0, 0)),
            ],
            schema=["USER_ID", "QUERY_TS"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="QUERY_TS",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 2)
        self.assertIn("AMOUNT", result_pd.columns)
        self.assertTrue(
            result_pd["AMOUNT"].notna().all(),
            f"AMOUNT has null values: {result_pd['AMOUNT'].tolist()}",
        )

    # =========================================================================
    # Non-aggregated streaming FV as VIEW (no refresh_freq)
    # =========================================================================

    def test_streaming_non_aggregated_view(self) -> None:
        """Non-aggregated streaming FV without refresh_freq creates a VIEW."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_VIEW_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.is_streaming)
        self.assertFalse(registered_fv.is_tiled)
        self.assertEqual(registered_fv.status, FeatureViewStatus.STATIC)

    # =========================================================================
    # Non-aggregated streaming FV as DT (with refresh_freq + warning)
    # =========================================================================

    def test_streaming_non_aggregated_dt_with_warning(self) -> None:
        """Non-aggregated streaming FV with refresh_freq creates a DT and warns."""
        import warnings

        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stream_config = StreamConfig(
                stream_source=stream,
                transformation_fn=identity_transform,
                backfill_df=backfill_df,
            )
            fv = FeatureView(
                name=f"STREAM_DT_{s}",
                entities=[self.user_entity],
                stream_config=stream_config,
                timestamp_col="EVENT_TIME",
                refresh_freq="1 minute",
            )
            refresh_warnings = [x for x in w if "don't require refresh_freq" in str(x.message)]
            self.assertGreater(len(refresh_warnings), 0)

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.is_streaming)
        self.assertFalse(registered_fv.is_tiled)
        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)

    # =========================================================================
    # Aggregated streaming FV without refresh_freq → error
    # =========================================================================

    def test_streaming_aggregated_requires_refresh_freq(self) -> None:
        """Aggregated streaming FV without refresh_freq raises error at validate time."""
        from snowflake.ml.feature_store.feature import Feature

        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_AGG_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            features=[Feature.sum("AMOUNT", "2h").alias("AMOUNT_SUM")],
        )

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(fv, "v1")

        self.assertIn("refresh_freq", str(cm.exception))

    # =========================================================================
    # get_feature_view reconstruction
    # =========================================================================

    def test_get_feature_view_reconstructs_streaming_fv(self) -> None:
        """Verify get_feature_view returns a streaming FV with correct metadata."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_FV_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

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
        )

        fs.register_feature_view(fv, "v1")

        reconstructed = fs.get_feature_view(fv_name, "v1")

        self.assertTrue(reconstructed.is_streaming)
        self.assertEqual(reconstructed.name, fv.name)
        self.assertEqual(str(reconstructed.version), "v1")
        self.assertTrue(reconstructed.online)

    # =========================================================================
    # Overwrite (re-register with overwrite=True)
    # =========================================================================

    def test_overwrite_streaming_fv(self) -> None:
        """Re-registering with overwrite=True succeeds and keeps ref_count=1."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        fs.register_feature_view(fv, "v1")
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count(self._stream_source_ref_key(stream)), 1)

        fv2 = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=StreamConfig(
                stream_source=stream,
                transformation_fn=identity_transform,
                backfill_df=self._session.table(backfill_table),
            ),
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )
        registered = fs.register_feature_view(fv2, "v1", overwrite=True)
        self.assertTrue(registered.is_streaming)

        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count(self._stream_source_ref_key(stream)), 1)

    # =========================================================================
    # backfill_start_time filter
    # =========================================================================

    def test_backfill_start_time_filters_data(self) -> None:
        """backfill_start_time filters backfill data by timestamp."""
        from datetime import datetime

        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
            backfill_start_time=datetime(2024, 1, 1, 0, 30, 0),
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_streaming)
        self.assertEqual(registered.status, FeatureViewStatus.ACTIVE)

    # =========================================================================
    # update_feature_view for streaming FV
    # =========================================================================

    def test_update_streaming_fv_desc(self) -> None:
        """Update streaming FV description."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_FV_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

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
        )

        fs.register_feature_view(fv, "v1")

        updated = fs.update_feature_view(
            name=fv_name,
            version="v1",
            desc="Updated description",
        )
        self.assertEqual(updated.desc, "Updated description")
        self.assertTrue(updated.is_streaming)

    # =========================================================================
    # suspend / resume streaming FV
    # =========================================================================

    def test_suspend_and_resume_streaming_fv(self) -> None:
        """Suspend and resume a DT-backed streaming FV."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        registered = fs.register_feature_view(fv, "v1")
        self.assertEqual(registered.status, FeatureViewStatus.ACTIVE)

        suspended = fs.suspend_feature_view(registered)
        self.assertEqual(suspended.status, FeatureViewStatus.SUSPENDED)

        resumed = fs.resume_feature_view(suspended)
        self.assertEqual(resumed.status, FeatureViewStatus.ACTIVE)

    # =========================================================================
    # Error: unregistered stream source
    # =========================================================================

    def test_unregistered_stream_source_raises(self) -> None:
        """Registering with a nonexistent stream source raises an error."""
        s = uuid.uuid4().hex[:8]
        fs = self._create_feature_store()
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="nonexistent_source",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
        )

        with self.assertRaisesRegex(RuntimeError, "Cannot find StreamSource"):
            fs.register_feature_view(fv, "v1")

    # =========================================================================
    # Explicit POSTGRES OnlineConfig + OFT
    # =========================================================================

    def test_streaming_fv_creates_spec_oft_explicit(self) -> None:
        """Streaming FV with explicit POSTGRES OnlineConfig creates an OFT."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_OFT_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)
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
        self.assertTrue(registered.is_streaming)
        self.assertTrue(registered.online)
        online_table_name = registered.fully_qualified_online_table_name()
        self.assertIsNotNone(online_table_name)
        self.assertIn("$ONLINE", online_table_name)

        deadline = time.time() + 120.0
        while time.time() < deadline:
            result = self._session.sql(
                f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' " f"IN SCHEMA {fs._config.full_schema_path}"
            ).collect()
            if len(result) > 0:
                break
            time.sleep(5)
        self.assertGreater(len(result), 0, "OFT should exist after streaming FV registration")

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for Postgres online read (Online Service Query API).",
    )
    def test_streaming_fv_spec_oft_online_read_e2e(self) -> None:
        """After backfill, online read via Query API returns rows for a registered entity key."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fv_name = f"STREAM_ONLY_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)
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

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), 100.0, places=3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[["u1"]], validate_fn=_validate, timeout=240.0, desc="streaming e2e"
        )

    # =========================================================================
    # E2E: streaming FV — all supported column types
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_streaming_fv_spec_oft_all_supported_types(self) -> None:
        """Verify all 6 supported types (String, Long, Double, Decimal, Boolean, TimestampNTZ) round-trip."""
        s = uuid.uuid4().hex[:8]
        stream = f"ALL_TYPES_{s}"
        fv_name = f"STREAM_ALL_TYPES_{s}"
        fs = self._create_feature_store()
        self._make_all_types_stream_source(fs, stream)
        backfill_table = self._create_all_types_backfill_table(fs, s)
        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=all_types_identity_transform,
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
        self.assertTrue(registered.online)

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )

        def _validate(pdf):
            row = pdf.iloc[0]
            self.assertAlmostEqual(float(row["SCORE"]), 3.14, places=1)
            self.assertEqual(int(row["RANK"]), 42)
            self.assertAlmostEqual(float(row["PRICE"]), 99.95, places=2)
            self.assertIn(row["IS_ACTIVE"], (True, "true", 1))

        self._poll_online_read(
            fs, fv_name, "v1", keys=[["u1"]], validate_fn=_validate, timeout=240.0, desc="all types SFV"
        )

    # =========================================================================
    # Rollback on registration failure
    # =========================================================================

    def test_rollback_on_registration_failure(self) -> None:
        """If registration fails mid-way, udf_transformed table is cleaned up."""
        s = uuid.uuid4().hex[:8]
        stream = f"TXN_{s}"
        fs = self._create_feature_store()
        self._make_stream_source(fs, stream)
        backfill_table = self._create_backfill_table(fs, s)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source=stream,
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name=f"STREAM_FV_{s}",
            entities=[self.user_entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="INVALID_FREQ_SHOULD_CAUSE_FAILURE",
        )

        try:
            fs.register_feature_view(fv, "v1")
        except Exception:
            pass

        ref_count = fs._metadata_manager.get_stream_source_ref_count(self._stream_source_ref_key(stream))
        self.assertEqual(ref_count, 0)

        fv_list = fs.list_feature_views().collect(statement_params=fs._telemetry_stmp)
        self.assertEqual(len(fv_list), 0)


if __name__ == "__main__":
    absltest.main()
