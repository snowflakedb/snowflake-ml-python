"""E2E integration tests for **batch** feature views with spec-based OFT (Postgres online).

Covers the full pipeline for all batch FV variants:
- **Non-tiled (timeseries)**: passthrough columns with ``timestamp_col`` + ``refresh_freq``
- **Tiled (timeseries)**: aggregated features via ``Feature.*`` + ``feature_granularity``
- **Non-timeseries**: no ``timestamp_col`` (with ``refresh_freq`` or static)
- **Multi-entity**: composite join keys

Each e2e test covers: registration -> OFT creation -> DT materialisation -> online read via Query API.
Offline dataset generation via ``generate_training_set`` is also tested per variant.

Lifecycle tests (delete, overwrite, suspend/resume, enable/disable, serialization, schema validation)
are migrated from the former ``feature_store_spec_oft_test``.

Reuses ``StreamingFeatureViewIntegTestBase`` for the class-scoped Feature Store, ``USER_ID`` entity,
and Online Service (see that module — reuse DB/schema is the default).

Requires ``SNOWFLAKE_PAT`` for spec OFT online read, e.g.
``bazel test ... --test_env=SNOWFLAKE_PAT=$(tr -d '\\n' < ~/mypat)``.
"""

import json
import math
import os
import time
import unittest
import uuid

from absl.testing import absltest
from common_utils import FS_INTEG_TEST_DATASET_SCHEMA
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
)


class FeatureStoreBatchOnlineReadIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """Batch FV + spec-based OFT: e2e tests covering registration through online read and offline dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.product_entity = Entity(name="product_entity", join_keys=["PRODUCT_ID"], desc="Product entity")
        try:
            cls.fs.register_entity(cls.product_entity)
        except Exception:
            pass

        cls._session.sql(f"CREATE SCHEMA IF NOT EXISTS {cls.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}").collect()
        cls._events_table = cls._create_events_table_class()

        cls._sample_table_name = f"TEST_SPEC_OFT_DATA_{uuid.uuid4().hex.upper()[:8]}"
        cls._session.sql(
            f"""
            CREATE OR REPLACE TABLE {cls.fs._config.full_schema_path}.{cls._sample_table_name} (
                USER_ID INT,
                PRODUCT_ID INT,
                PURCHASE_TIME TIMESTAMP_NTZ,
                PURCHASE_AMOUNT FLOAT
            )
        """
        ).collect()
        cls._session.sql(
            f"""
            INSERT INTO {cls.fs._config.full_schema_path}.{cls._sample_table_name} VALUES
            (1, 100, DATEADD('day', -1, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 10.5),
            (2, 200, DATEADD('day', -2, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 20.0),
            (3, 300, DATEADD('day', -3, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 30.5)
        """
        ).collect()
        cls.sample_data = cls._session.table(f"{cls.fs._config.full_schema_path}.{cls._sample_table_name}")

    def setUp(self) -> None:
        super().setUp()
        self.product_entity = type(self).product_entity
        self.sample_data = type(self).sample_data
        self._events_table = type(self)._events_table

    @classmethod
    def _create_events_table_class(cls) -> str:
        table_full_path = f"{cls.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.events_{uuid.uuid4().hex.upper()}"
        cls._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (USER_ID INT, EVENT_TS TIMESTAMP_NTZ, AMOUNT FLOAT)
            """
        ).collect()
        cls._session.sql(
            f"""INSERT INTO {table_full_path} (USER_ID, EVENT_TS, AMOUNT) VALUES
                (1, DATEADD('hour', -48, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 10.0),
                (1, DATEADD('hour', -47, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 20.0),
                (2, DATEADD('hour', -47, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 30.0),
                (2, DATEADD('hour', -46, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, 40.0)
            """
        ).collect()
        return table_full_path

    def _get_events_df(self):
        return self._session.table(self._events_table)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _create_batch_source_table(self, fs: FeatureStore, suffix: str, entity_key: str, amount: float) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_ONLINE_SRC_{suffix}"
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
            ({entity_key!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), {amount})
        """
        ).collect()
        return table_name

    def _create_batch_tiled_source_table(
        self, fs: FeatureStore, suffix: str, entity_key: str
    ) -> tuple[str, float, int]:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_TILED_SRC_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
        """
        ).collect()
        amounts = (10.0, 20.0, 30.0)
        self._session.sql(
            f"""
            INSERT INTO {table_name}
            SELECT column1, column2, column3
            FROM VALUES
                (
                    {entity_key!r},
                    DATEADD('hour', 1, DATEADD('day', -1, DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))),
                    {amounts[0]}
                ),
                (
                    {entity_key!r},
                    DATEADD('hour', 2, DATEADD('day', -1, DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))),
                    {amounts[1]}
                ),
                (
                    {entity_key!r},
                    DATEADD('hour', 3, DATEADD('day', -1, DATE_TRUNC('day', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))),
                    {amounts[2]}
                )
        """
        ).collect()
        return table_name, sum(amounts), len(amounts)

    # =========================================================================
    # E2E: Batch non-tiled (timeseries) — registration -> online read
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_fv_spec_oft_online_read_by_key(self) -> None:
        """Register batch FV over a small table; assert Query API returns AMOUNT for the keyed row."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_ONLINE_FV_{s}"
        batch_key = f"U_BATCH_{s}"
        expected_amount = 888.0

        src_table = self._create_batch_source_table(fs, s, batch_key, expected_amount)
        feature_df = self._session.table(src_table)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertFalse(registered.is_streaming)
        self.assertTrue(registered.online)

        online_name = registered.fully_qualified_online_table_name()
        self.assertIsNotNone(online_name)
        self.assertIn("$ONLINE", online_name)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount, places=3)

        self._poll_online_read(fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate, desc="batch non-tiled")

    # =========================================================================
    # E2E: Batch tiled (timeseries) — registration -> online read
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_tiled_fv_spec_oft_full_online_read_by_key(self) -> None:
        """Tiled batch FV: multiple source rows per key; online read returns tile aggregates."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_TILED_ONLINE_FV_{s}"
        batch_key = f"U_BATCH_TILED_{s}"

        src_table, expected_sum, expected_count = self._create_batch_tiled_source_table(fs, s, batch_key)
        feature_df = self._session.table(src_table)
        features = [
            Feature.sum("AMOUNT", "2d").alias("AMOUNT_SUM_2D"),
            Feature.count("AMOUNT", "2d").alias("TXN_COUNT_2D"),
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
        registered = fs.register_feature_view(fv, "v1")
        self.assertFalse(registered.is_streaming)
        self.assertTrue(registered.is_tiled)
        self.assertTrue(registered.online)

        offline_deadline = time.time() + 300.0
        while time.time() < offline_deadline:
            try:
                fv_live = fs.get_feature_view(fv_name, "v1")
                off = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE, keys=[[batch_key]])
                if off.count() > 0:
                    opdf = off.to_pandas()
                    if "AMOUNT_SUM_2D" not in opdf.columns or "TXN_COUNT_2D" not in opdf.columns:
                        time.sleep(5)
                        continue
                    o0 = opdf.iloc[0]
                    try:
                        s_val = float(o0["AMOUNT_SUM_2D"])
                        c_val = float(o0["TXN_COUNT_2D"])
                    except (TypeError, ValueError):
                        time.sleep(5)
                        continue
                    if math.isnan(s_val) or math.isnan(c_val):
                        time.sleep(5)
                        continue
                    if abs(s_val - expected_sum) < 0.01 and abs(c_val - float(expected_count)) < 0.01:
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            self.fail(
                f"Timed out waiting for tiled batch FV offline aggregates to match "
                f"sum={expected_sum} count={expected_count} for key {batch_key!r}."
            )

        def _validate_tiled(pdf):
            self.assertIn("AMOUNT_SUM_2D", pdf.columns)
            self.assertIn("TXN_COUNT_2D", pdf.columns)
            row = pdf.iloc[0]
            s_val = float(row["AMOUNT_SUM_2D"])
            c_val = float(row["TXN_COUNT_2D"])
            if math.isnan(s_val) or math.isnan(c_val):
                raise AssertionError(f"nan aggregates (sum={s_val}, count={c_val})")
            self.assertAlmostEqual(s_val, expected_sum, places=2)
            self.assertAlmostEqual(c_val, float(expected_count), places=2)

        self._poll_online_read(fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate_tiled, desc="batch tiled")

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_tiled_fv_spec_oft_incremental_online_read_by_key(self) -> None:
        """Tiled batch FV: multiple source rows per key; online read returns tile aggregates."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_TILED_ONLINE_FV_{s}"
        batch_key = f"U_BATCH_TILED_{s}"

        src_table, expected_sum, expected_count = self._create_batch_tiled_source_table(fs, s, batch_key)
        feature_df = self._session.table(src_table)
        features = [
            Feature.sum("AMOUNT", "2d").alias("AMOUNT_SUM_2D"),
            Feature.count("AMOUNT", "2d").alias("TXN_COUNT_2D"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_mode="INCREMENTAL",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertFalse(registered.is_streaming)
        self.assertTrue(registered.is_tiled)
        self.assertTrue(registered.online)

        offline_deadline = time.time() + 300.0
        while time.time() < offline_deadline:
            try:
                fv_live = fs.get_feature_view(fv_name, "v1")
                off = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE, keys=[[batch_key]])
                if off.count() > 0:
                    opdf = off.to_pandas()
                    if "AMOUNT_SUM_2D" not in opdf.columns or "TXN_COUNT_2D" not in opdf.columns:
                        time.sleep(5)
                        continue
                    o0 = opdf.iloc[0]
                    try:
                        s_val = float(o0["AMOUNT_SUM_2D"])
                        c_val = float(o0["TXN_COUNT_2D"])
                    except (TypeError, ValueError):
                        time.sleep(5)
                        continue
                    if math.isnan(s_val) or math.isnan(c_val):
                        time.sleep(5)
                        continue
                    if abs(s_val - expected_sum) < 0.01 and abs(c_val - float(expected_count)) < 0.01:
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            self.fail(
                f"Timed out waiting for tiled batch FV offline aggregates to match "
                f"sum={expected_sum} count={expected_count} for key {batch_key!r}."
            )

        def _validate_tiled(pdf):
            self.assertIn("AMOUNT_SUM_2D", pdf.columns)
            self.assertIn("TXN_COUNT_2D", pdf.columns)
            row = pdf.iloc[0]
            s_val = float(row["AMOUNT_SUM_2D"])
            c_val = float(row["TXN_COUNT_2D"])
            if math.isnan(s_val) or math.isnan(c_val):
                raise AssertionError(f"nan aggregates (sum={s_val}, count={c_val})")
            self.assertAlmostEqual(s_val, expected_sum, places=2)
            self.assertAlmostEqual(c_val, float(expected_count), places=2)

        self._poll_online_read(fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate_tiled, desc="batch tiled")

    # =========================================================================
    # E2E: Batch non-timeseries (with refresh_freq) — registration -> online read
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_non_timeseries_fv_spec_oft_online_read_by_key(self) -> None:
        """Batch FV without timestamp_col: register -> DT materialization -> online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_NO_TS_FV_{s}"
        entity_key = f"U_NO_TS_{s}"
        expected_amount = 42.0

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_NO_TS_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {table_name} VALUES ({entity_key!r}, {expected_amount})").collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)
        self.assertIsNone(registered.timestamp_col)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount, places=3)

        self._poll_online_read(fs, fv_name, "v1", keys=[[entity_key]], validate_fn=_validate, desc="batch non-ts")

    # =========================================================================
    # E2E: Batch static (no refresh_freq, no timestamp) — registration -> online read
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_static_fv_spec_oft_online_read_by_key(self) -> None:
        """Static batch FV (no refresh_freq, no timestamp): register -> VIEW -> online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_STATIC_FV_{s}"
        entity_key = f"U_STATIC_{s}"
        expected_amount = 77.0

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_STATIC_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {table_name} VALUES ({entity_key!r}, {expected_amount})").collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)
        self.assertIsNone(registered.refresh_freq)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount, places=3)

        self._poll_online_read(fs, fv_name, "v1", keys=[[entity_key]], validate_fn=_validate, desc="batch static")

    # =========================================================================
    # E2E: Multi-entity batch FV — registration -> online read
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_multi_entity_fv_spec_oft_online_read(self) -> None:
        """Multi-entity batch FV: register with composite keys -> online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_MULTI_ENT_FV_{s}"
        user_key = 1
        product_key = 100
        expected_amount = 10.5

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity, self.product_entity],
            feature_df=self.sample_data,
            timestamp_col="PURCHASE_TIME",
            refresh_freq="5m",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)
        self.assertEqual(len(registered.entities), 2)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            self.assertIn("PURCHASE_AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["PURCHASE_AMOUNT"]), expected_amount, places=1)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[user_key, product_key]], validate_fn=_validate, desc="multi-entity"
        )

    # =========================================================================
    # Offline dataset: batch non-tiled
    # =========================================================================

    def test_batch_non_tiled_fv_spec_oft_offline_dataset(self) -> None:
        """generate_training_set on a non-tiled batch FV with spec-based OFT."""
        from datetime import datetime, timedelta

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_NT_DS_FV_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_NT_DS_SRC_{s}"
        now = datetime.utcnow()
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ('u1', '{(now - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')}'::TIMESTAMP_NTZ, 10.0),
            ('u1', '{(now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')}'::TIMESTAMP_NTZ, 20.0),
            ('u2', '{(now - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')}'::TIMESTAMP_NTZ, 100.0),
            ('u2', '{(now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')}'::TIMESTAMP_NTZ, 200.0)
        """
        ).collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        spine_df = self._session.create_dataframe(
            [("u1", now), ("u2", now)],
            schema=["USER_ID", "QUERY_TS"],
        )
        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered],
            spine_timestamp_col="QUERY_TS",
            join_method="cte",
        )
        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 2)
        self.assertIn("AMOUNT", result_pd.columns)
        self.assertTrue(result_pd["AMOUNT"].notna().all(), f"Null AMOUNT values: {result_pd['AMOUNT'].tolist()}")

    # =========================================================================
    # Offline dataset: batch tiled
    # =========================================================================

    def test_batch_tiled_fv_spec_oft_offline_dataset(self) -> None:
        """generate_training_set on a tiled batch FV with spec-based OFT."""
        from datetime import datetime

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_TILED_DS_FV_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_TILED_DS_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, AMOUNT FLOAT
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
        feature_df = self._session.table(table_name)
        features = [
            Feature.sum("AMOUNT", "2d").alias("AMOUNT_SUM_2D"),
            Feature.count("AMOUNT", "2d").alias("TXN_COUNT_2D"),
        ]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_tiled)
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        spine_df = self._session.create_dataframe(
            [("u1", datetime(2024, 1, 5, 0, 0, 0)), ("u2", datetime(2024, 1, 5, 0, 0, 0))],
            schema=["USER_ID", "QUERY_TS"],
        )
        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered],
            spine_timestamp_col="QUERY_TS",
            join_method="cte",
        )
        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 2)
        self.assertIn("AMOUNT_SUM_2D", result_pd.columns)
        self.assertIn("TXN_COUNT_2D", result_pd.columns)
        for col in ["AMOUNT_SUM_2D", "TXN_COUNT_2D"]:
            self.assertTrue(result_pd[col].notna().all(), f"Column {col} has null values: {result_pd[col].tolist()}")

    # =========================================================================
    # Offline dataset: batch non-timeseries
    # =========================================================================

    def test_batch_non_timeseries_fv_spec_oft_offline_dataset(self) -> None:
        """generate_training_set on a non-timeseries batch FV (no timestamp_col)."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_NO_TS_DS_FV_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_NO_TS_DS_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES ('u1', 10.0), ('u2', 200.0)
        """
        ).collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        spine_df = self._session.create_dataframe([("u1",), ("u2",)], schema=["USER_ID"])
        result_df = fs.generate_training_set(spine_df=spine_df, features=[registered], join_method="cte")
        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 2)
        self.assertIn("AMOUNT", result_pd.columns)
        self.assertTrue(result_pd["AMOUNT"].notna().all(), f"Null AMOUNT values: {result_pd['AMOUNT'].tolist()}")

    # =========================================================================
    # Lifecycle: spec OFT get preserves config
    # =========================================================================

    def test_spec_oft_get_feature_view_preserves_config(self) -> None:
        """get_feature_view preserves the spec OFT online config."""
        fv_name = f"spec_oft_get_fv_{uuid.uuid4().hex[:8]}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT", "PURCHASE_TIME"),
            timestamp_col="PURCHASE_TIME",
            refresh_freq="5m",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv, "v1")
        retrieved_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertTrue(retrieved_fv.online)
        self.assertIsNotNone(retrieved_fv.online_config)
        self.assertEqual(retrieved_fv.online_config.target_lag, "10 seconds")

    # =========================================================================
    # Lifecycle: delete
    # =========================================================================

    def test_spec_oft_delete_feature_view(self) -> None:
        """Deleting a FV with spec-based OFT also deletes the OFT."""
        fv_name = f"spec_oft_delete_{uuid.uuid4().hex[:8]}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT"),
            refresh_freq="1 minute",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online)

        self.fs.delete_feature_view(registered_fv)

        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.name == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 0)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 0)

    # =========================================================================
    # Lifecycle: overwrite
    # =========================================================================

    def test_spec_oft_overwrite(self) -> None:
        """Overwriting a FV with spec-based OFT recreates the OFT."""
        fv_name = f"spec_oft_overwrite_{uuid.uuid4().hex[:8]}"
        fv1 = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT"),
            refresh_freq="1 minute",
            desc="original",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv1, "v1")

        fv2 = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT"),
            refresh_freq="2 minutes",
            desc="overwritten",
            online_config=OnlineConfig(enable=True, target_lag="30s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv2, "v1", overwrite=True)

        retrieved = self.fs.get_feature_view(fv_name, "v1")
        self.assertTrue(retrieved.online)
        self.assertEqual(retrieved.online_config.target_lag, "30 seconds")
        self.assertEqual(retrieved.desc, "overwritten")

    # =========================================================================
    # Lifecycle: update enable/disable
    # =========================================================================

    def test_spec_oft_update_enable_disable(self) -> None:
        """Enable and disable spec-based OFT via update_feature_view."""
        fv_name = f"spec_oft_toggle_{uuid.uuid4().hex[:8]}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT", "PURCHASE_TIME"),
            timestamp_col="PURCHASE_TIME",
            refresh_freq="15m",
            online_config=OnlineConfig(enable=False),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertFalse(registered_fv.online)

        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.assertTrue(updated_fv.online)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

        disabled_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=OnlineConfig(enable=False),
        )
        self.assertFalse(disabled_fv.online)

        online_tables_after = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables_after), 0)

    # =========================================================================
    # Lifecycle: suspend/resume
    # =========================================================================

    def test_spec_oft_suspend_resume(self) -> None:
        """Suspend/resume on a FV with spec-based OFT."""
        fv_name = f"spec_oft_susp_{uuid.uuid4().hex[:8]}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT", "PURCHASE_TIME"),
            timestamp_col="PURCHASE_TIME",
            refresh_freq="10m",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv, "v1")

        suspended_fv = self.fs.suspend_feature_view(fv_name, "v1")
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)
        self.assertEqual(online_tables[0]["scheduling_state"], "SUSPENDED")

        resumed_fv = self.fs.resume_feature_view(fv_name, "v1")
        self.assertEqual(resumed_fv.status.value, "ACTIVE")

        online_tables_after = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables_after), 1)
        self.assertIn(online_tables_after[0]["scheduling_state"], ["RUNNING"])

    # =========================================================================
    # Lifecycle: config serialization
    # =========================================================================

    def test_online_config_spec_oft_serialization(self) -> None:
        """OnlineConfig serialization round-trips with Postgres store type."""
        config = OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES)
        json_str = config.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["store_type"], "postgres")

        reconstructed = OnlineConfig.from_json(json_str)
        self.assertEqual(reconstructed.store_type, OnlineStoreType.POSTGRES)
        self.assertEqual(reconstructed.enable, True)
        self.assertEqual(reconstructed.target_lag, "10s")

    def test_online_config_backward_compat_no_store_type(self) -> None:
        """Old configs without store_type deserialize to HYBRID_TABLE."""
        old_json = '{"enable": true, "target_lag": "10s"}'
        config = OnlineConfig.from_json(old_json)
        self.assertEqual(config.store_type, OnlineStoreType.HYBRID_TABLE)

    # =========================================================================
    # Lifecycle: list shows online config
    # =========================================================================

    def test_list_feature_views_spec_oft_online_config(self) -> None:
        """list_feature_views shows spec OFT store_type in online config."""
        fv_name = f"spec_oft_list_{uuid.uuid4().hex[:8]}"
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("USER_ID", "PURCHASE_AMOUNT"),
            refresh_freq="1 minute",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self.fs.register_feature_view(fv, "v1")

        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)

        online_config = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertTrue(online_config["enable"])

    # =========================================================================
    # Schema validation: all supported column types
    # =========================================================================

    @unittest.skipUnless(
        os.environ.get("SNOWFLAKE_PAT", "").strip(),
        "SNOWFLAKE_PAT must be set for spec OFT online read (Online Service Query API).",
    )
    def test_batch_fv_spec_oft_all_supported_types(self) -> None:
        """Verify all 6 supported types (String, Long, Double, Decimal, Boolean, TimestampNTZ) round-trip."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_ALL_TYPES_{s}"
        entity_key = f"U_ALL_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.ALL_TYPES_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                SCORE FLOAT,
                RANK INT,
                PRICE NUMBER(10,2),
                IS_ACTIVE BOOLEAN
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ({entity_key!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), 3.14, 42, 99.95, TRUE)
        """
        ).collect()

        feature_df = self._session.table(table_name)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="10 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            row = pdf.iloc[0]
            self.assertAlmostEqual(float(row["SCORE"]), 3.14, places=1)
            self.assertEqual(int(row["RANK"]), 42)
            self.assertAlmostEqual(float(row["PRICE"]), 99.95, places=2)
            self.assertIn(row["IS_ACTIVE"], (True, "true", 1))

        self._poll_online_read(fs, fv_name, "v1", keys=[[entity_key]], validate_fn=_validate, desc="all types BFV")

    # =========================================================================
    # Schema validation: unsupported column type
    # =========================================================================

    def test_spec_oft_rejects_unsupported_column_type(self) -> None:
        """Unsupported column types (e.g. DATE) are rejected with a clear error."""
        bad_table_name = f"TEST_BAD_TYPES_{uuid.uuid4().hex.upper()[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{bad_table_name} (
                USER_ID INT,
                BIRTHDAY DATE,
                PURCHASE_TIME TIMESTAMP_NTZ
            )
        """
        ).collect()

        schema_path = self.fs._config.full_schema_path
        self.addCleanup(lambda: self._session.sql(f"DROP TABLE IF EXISTS {schema_path}.{bad_table_name}").collect())

        bad_data = self._session.table(f"{self.fs._config.full_schema_path}.{bad_table_name}")

        fv = FeatureView(
            name=f"spec_oft_bad_types_{uuid.uuid4().hex[:8]}",
            entities=[self.user_entity],
            feature_df=bad_data,
            timestamp_col="PURCHASE_TIME",
            refresh_freq="1 minute",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )

        with self.assertRaises(Exception) as ctx:
            self.fs.register_feature_view(fv, "v1")

        self.assertIn("DateType", str(ctx.exception))
        self.assertIn("BIRTHDAY", str(ctx.exception))


if __name__ == "__main__":
    absltest.main()
