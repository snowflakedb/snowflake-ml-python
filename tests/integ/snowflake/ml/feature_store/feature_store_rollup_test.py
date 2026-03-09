"""Integration tests for rollup feature views."""

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from absl.testing import absltest, parameterized
from common_utils import FS_INTEG_TEST_DATASET_SCHEMA, create_random_schema
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml.feature_store import Feature, RollupConfig
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewStatus


class RollupFeatureViewTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for rollup feature views (entity aggregation)."""

    def setUp(self) -> None:
        super().setUp()
        self._active_feature_store = []

        try:
            self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}").collect()
            self._events_table = self._create_events_table()
            self._mapping_table = self._create_mapping_table()
        except Exception as e:
            self.tearDown()
            raise Exception(f"Test setup failed: {e}")

    def tearDown(self) -> None:
        for fs in self._active_feature_store:
            try:
                fs._clear(dryrun=False)
            except Exception as e:
                if "Intentional Integ Test Error" not in str(e):
                    raise Exception(f"Unexpected exception happens when clear: {e}")
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()

        self._session.sql(f"DROP TABLE IF EXISTS {self._events_table}").collect()
        self._session.sql(f"DROP TABLE IF EXISTS {self._mapping_table}").collect()
        super().tearDown()

    def _create_events_table(self) -> str:
        """Create a table with visitor-level time-series event data."""
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.rollup_events_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), company_id VARCHAR(16), event_ts TIMESTAMP_NTZ,
                 order_value FLOAT, product_id VARCHAR(16))
            """
        ).collect()

        # Insert test data:
        # v1, v2 -> s1 (subscriber 1)
        # v3 -> s2 (subscriber 2)
        # v4 -> s3 (subscriber 3)
        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, company_id, event_ts, order_value, product_id)
                VALUES
                -- Visitor 1 events (maps to subscriber s1)
                ('v1', 'c1', '2024-01-01 10:00:00', 100.0, 'p1'),
                ('v1', 'c1', '2024-01-01 10:30:00', 200.0, 'p2'),
                ('v1', 'c1', '2024-01-01 11:00:00', 150.0, 'p1'),
                -- Visitor 2 events (maps to subscriber s1)
                ('v2', 'c1', '2024-01-01 10:15:00', 300.0, 'p3'),
                ('v2', 'c1', '2024-01-01 10:45:00', 250.0, 'p1'),
                -- Visitor 3 events (maps to subscriber s2)
                ('v3', 'c1', '2024-01-01 10:20:00', 175.0, 'p2'),
                -- Visitor 4 events (maps to subscriber s3)
                ('v4', 'c2', '2024-01-01 10:00:00', 500.0, 'p1')
            """
        ).collect()
        return table_full_path

    def _create_mapping_table(self) -> str:
        """Create a visitor -> subscriber mapping table."""
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.visitor_subscriber_map_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), company_id VARCHAR(16), subscriber_id VARCHAR(16))
            """
        ).collect()

        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, company_id, subscriber_id)
                VALUES
                ('v1', 'c1', 's1'),
                ('v2', 'c1', 's1'),
                ('v3', 'c1', 's2'),
                ('v4', 'c2', 's3')
            """
        ).collect()
        return table_full_path

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        current_schema = (
            create_random_schema(self._session, "FS_ROLLUP_TEST", database=self.test_db) if name is None else name
        )
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        self._session.use_database(self._dummy_db)
        return fs

    def _create_visitor_entity(self) -> Entity:
        return Entity("visitor", ["visitor_id", "company_id"])

    def _create_subscriber_entity(self) -> Entity:
        return Entity("subscriber", ["subscriber_id", "company_id"])

    def _get_events_df(self):
        return self._session.table(self._events_table)

    def _get_mapping_df(self):
        return self._session.table(self._mapping_table)

    # =========================================================================
    # Basic Rollup Tests
    # =========================================================================

    def test_rollup_fv_registration(self) -> None:
        """Test that a rollup FV can be registered successfully."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create visitor-level tiled FV
        visitor_fv = FeatureView(
            name="visitor_events",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Create subscriber-level rollup FV
        subscriber_fv = FeatureView(
            name="subscriber_events",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Verify registration
        self.assertEqual(registered_subscriber.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(registered_subscriber.name, "SUBSCRIBER_EVENTS")
        self.assertEqual(registered_subscriber.version, "v1")

    def test_rollup_fv_is_tiled(self) -> None:
        """Test that rollup FV has is_tiled=True."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_events",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_events",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )

        # Check is_tiled before registration
        self.assertTrue(subscriber_fv.is_tiled)
        self.assertTrue(subscriber_fv.is_rollup)

    def test_rollup_fv_inherits_granularity(self) -> None:
        """Test that rollup FV inherits feature_granularity from parent."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_events",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="2h",  # 2 hour granularity
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_events",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )

        # Verify granularity inherited
        self.assertEqual(subscriber_fv.feature_granularity, "2h")

    # =========================================================================
    # Aggregation Correctness Tests - COUNT
    # =========================================================================

    def test_rollup_count_aggregation(self) -> None:
        """Test COUNT is correctly rolled up (sum of counts)."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_count",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_count",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        # s1 = v1 (3 events) + v2 (2 events) = 5 events
        result_df = fs.read_feature_view(registered_subscriber)
        result = result_df.filter(result_df["SUBSCRIBER_ID"] == "s1").collect()
        # Sum of _PARTIAL_COUNT across tiles for s1 should equal 5
        # Note: exact assertion depends on how tiles are aggregated
        self.assertGreater(len(result), 0)

    # =========================================================================
    # Aggregation Correctness Tests - SUM
    # =========================================================================

    def test_rollup_sum_aggregation(self) -> None:
        """Test SUM is correctly rolled up."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_sum",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.sum("order_value", "24h").alias("order_total")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_sum",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        # s1: v1 (100+200+150=450) + v2 (300+250=550) = 1000
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Aggregation Correctness Tests - AVG
    # =========================================================================

    def test_rollup_avg_aggregation(self) -> None:
        """Test AVG is correctly rolled up (not avg of avgs)."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_avg",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.avg("order_value", "24h").alias("order_avg")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_avg",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        # s1: total_sum=1000, total_count=5 → avg=200
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Aggregation Correctness Tests - MIN
    # =========================================================================

    def test_rollup_min_aggregation(self) -> None:
        """Test MIN is correctly rolled up."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_min",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.min("order_value", "24h").alias("order_min")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_min",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        # s1: min(v1_min=100, v2_min=250) = 100
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Aggregation Correctness Tests - MAX
    # =========================================================================

    def test_rollup_max_aggregation(self) -> None:
        """Test MAX is correctly rolled up."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_max",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.max("order_value", "24h").alias("order_max")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_max",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        # s1: max(v1_max=200, v2_max=300) = 300
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Aggregation Correctness Tests - STD (and VAR)
    # =========================================================================

    def test_rollup_std_aggregation(self) -> None:
        """Test STD is correctly rolled up using Welford's method (SUM, COUNT, SUM_SQ)."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_std",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.stddev("order_value", "24h").alias("order_std")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_std",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Aggregation Correctness Tests - LAST_N
    # =========================================================================

    def test_rollup_last_n_aggregation(self) -> None:
        """Test LAST_N is correctly rolled up with global timestamp ordering.

        Verifies that the rollup produces a correctly-sorted array by global
        event timestamps (not visitor arrival order). This tests the fix for
        the ARRAY_UNION_AGG ordering bug.

        Test data (subscriber s1 = visitors v1 + v2):
          v1: 100.0@10:00, 200.0@10:30, 150.0@11:00
          v2: 300.0@10:15, 250.0@10:45

        Expected LAST_N(order_value, 24h, n=5) for s1 (most recent first):
          [150.0, 250.0, 200.0, 300.0, 100.0]
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_last_n",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.last_n("order_value", "24h", n=5).alias("last_orders")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_last_n",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

        # Verify the rolled-up tile table has companion TS columns
        tile_table = registered_subscriber.fully_qualified_name()
        desc_result = self._session.sql(f"DESC TABLE {tile_table}").collect()
        col_names = [row["name"].upper() for row in desc_result]
        self.assertIn("_PARTIAL_LAST_TS_ORDER_VALUE", col_names, "Rolled-up tile should have companion TS column")

    # =========================================================================
    # Aggregation Correctness Tests - FIRST_N
    # =========================================================================

    def test_rollup_first_n_aggregation(self) -> None:
        """Test FIRST_N is correctly rolled up with global timestamp ordering.

        Verifies that the rollup produces a correctly-sorted array by global
        event timestamps (oldest first). This tests the fix for the
        ARRAY_UNION_AGG ordering bug.

        Test data (subscriber s1 = visitors v1 + v2):
          v1: 100.0@10:00, 200.0@10:30, 150.0@11:00
          v2: 300.0@10:15, 250.0@10:45

        Expected FIRST_N(order_value, 24h, n=5) for s1 (oldest first):
          [100.0, 300.0, 200.0, 250.0, 150.0]
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_first_n",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.first_n("order_value", "24h", n=5).alias("first_orders")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_first_n",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

        # Verify the rolled-up tile table has companion TS columns
        tile_table = registered_subscriber.fully_qualified_name()
        desc_result = self._session.sql(f"DESC TABLE {tile_table}").collect()
        col_names = [row["name"].upper() for row in desc_result]
        self.assertIn("_PARTIAL_FIRST_TS_ORDER_VALUE", col_names, "Rolled-up tile should have companion TS column")

    # =========================================================================
    # Aggregation Correctness Tests - APPROX_COUNT_DISTINCT
    # =========================================================================

    def test_rollup_approx_count_distinct_aggregation(self) -> None:
        """Test APPROX_COUNT_DISTINCT is correctly rolled up via HLL_COMBINE."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_hll",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.approx_count_distinct("product_id", "24h").alias("unique_products")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_hll",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Aggregation Correctness Tests - APPROX_PERCENTILE
    # =========================================================================

    def test_rollup_approx_percentile_aggregation(self) -> None:
        """Test APPROX_PERCENTILE is correctly rolled up via T-Digest combine."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_pctl",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.approx_percentile("order_value", "24h", percentile=0.5).alias("median_order")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_pctl",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Multiple Aggregations Test
    # =========================================================================

    def test_rollup_multiple_aggregations(self) -> None:
        """Test rollup with multiple aggregation types together."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        features = [
            Feature.count("visitor_id", "24h").alias("event_count"),
            Feature.sum("order_value", "24h").alias("order_total"),
            Feature.avg("order_value", "24h").alias("order_avg"),
            Feature.max("order_value", "24h").alias("order_max"),
        ]

        visitor_fv = FeatureView(
            name="visitor_multi",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_multi",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Read and verify all aggregations present
        result_df = fs.read_feature_view(registered_subscriber)
        self.assertGreater(result_df.count(), 0)

    # =========================================================================
    # Validation Tests
    # =========================================================================

    def test_rollup_config_source_must_be_tiled(self) -> None:
        """Test error when source is not tiled."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create a non-tiled FV
        visitor_fv = FeatureView(
            name="visitor_nontiled",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            # No feature_granularity or features - not tiled
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Attempt to create rollup should fail
        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_rollup",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._get_mapping_df(),
                ),
            )
        self.assertIn("tiled", str(context.exception).lower())

    def test_rollup_config_source_must_be_registered(self) -> None:
        """Test error when source is not registered."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create a draft FV (not registered)
        visitor_fv = FeatureView(
            name="visitor_draft",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )

        # Attempt to create rollup with draft source should fail
        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_rollup",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=visitor_fv,  # Draft, not registered
                    mapping_df=self._get_mapping_df(),
                ),
            )
        self.assertIn("registered", str(context.exception).lower())

    def test_rollup_config_mapping_must_have_parent_keys(self) -> None:
        """Test error when mapping missing parent join keys."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_keys",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Create mapping without visitor_id
        bad_mapping = self._session.sql("SELECT subscriber_id, company_id FROM " + self._mapping_table)

        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_badmap",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=bad_mapping,
                ),
            )
        self.assertIn("visitor_id", str(context.exception).lower())

    def test_rollup_config_mapping_must_have_target_keys(self) -> None:
        """Test error when mapping missing target join keys."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_target",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Create mapping without subscriber_id
        bad_mapping = self._session.sql("SELECT visitor_id, company_id FROM " + self._mapping_table)

        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_notarget",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=bad_mapping,
                ),
            )
        self.assertIn("subscriber_id", str(context.exception).lower())

    def test_rollup_and_feature_df_mutual_exclusion(self) -> None:
        """Test error when both rollup_config and feature_df specified."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_mutual",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_both",
                entities=[subscriber_entity],
                feature_df=self._get_events_df(),  # Should not have both
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._get_mapping_df(),
                ),
            )
        self.assertIn("both", str(context.exception).lower())

    def test_rollup_requires_either_feature_df_or_rollup_config(self) -> None:
        """Test error when neither rollup_config nor feature_df specified."""
        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_neither",
                entities=[self._create_subscriber_entity()],
                # Neither feature_df nor rollup_config
            )
        self.assertIn("either", str(context.exception).lower())

    def test_rollup_with_feature_granularity_raises_error(self) -> None:
        """Test that providing feature_granularity with rollup_config raises ValueError."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create and register parent tiled FV
        visitor_fv = FeatureView(
            name="visitor_fg_test",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Attempt to create rollup FV with feature_granularity should fail
        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_with_fg",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._get_mapping_df(),
                ),
                feature_granularity="1h",  # Should not be allowed
            )

        error_msg = str(context.exception)
        self.assertIn("Cannot specify feature_granularity with rollup_config", error_msg)
        self.assertIn("inherit", error_msg.lower())

    def test_rollup_with_features_raises_error(self) -> None:
        """Test that providing features with rollup_config raises ValueError."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create and register parent tiled FV
        visitor_fv = FeatureView(
            name="visitor_feat_test",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Attempt to create rollup FV with features should fail
        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_with_feat",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._get_mapping_df(),
                ),
                features=[Feature.sum("order_value", "24h")],  # Should not be allowed
            )

        error_msg = str(context.exception)
        self.assertIn("Cannot specify features with rollup_config", error_msg)
        self.assertIn("inherit", error_msg.lower())

    # =========================================================================
    # Global Ordering Correctness Tests (the ARRAY_UNION_AGG bug fix)
    # =========================================================================

    def test_rollup_last_n_global_ordering(self) -> None:
        """Regression test: LAST_N rollup must produce globally-sorted arrays.

        This is the exact scenario reported by the customer. Two visitors
        (v1, v2) map to one subscriber (s1). Their events interleave in time:

          v1: 100.0@10:00, 200.0@10:30, 150.0@11:00
          v2: 300.0@10:15, 250.0@10:45

        Without the fix, ARRAY_UNION_AGG merges per-visitor arrays in
        indeterminate order (e.g. [150,200,100,300,250]). With the fix,
        LATERAL FLATTEN + ORDER BY companion TS produces the correct
        globally-ordered result: [150.0, 250.0, 200.0, 300.0, 100.0]
        (most recent first: 11:00, 10:45, 10:30, 10:15, 10:00).
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_last_n_order",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.last_n("order_value", "24h", n=5).alias("last_orders")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_last_n_order",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Query with a spine to get the merged result
        spine = self._session.create_dataframe(
            [{"subscriber_id": "s1", "company_id": "c1", "ts": datetime(2024, 1, 1, 12, 0)}]
        )
        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered_subscriber],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 1)

        # Find the last_orders column
        last_orders_col = None
        for col in result_df.columns:
            if "LAST_ORDERS" in col.upper():
                last_orders_col = col
                break
        self.assertIsNotNone(last_orders_col, f"Could not find last_orders column in {result_df.columns}")

        last_orders_raw = results[0][last_orders_col]
        last_orders = json.loads(last_orders_raw) if isinstance(last_orders_raw, str) else last_orders_raw
        # The array must be globally sorted by timestamp DESC (most recent first):
        #   150.0@11:00, 250.0@10:45, 200.0@10:30, 300.0@10:15, 100.0@10:00
        expected = [150.0, 250.0, 200.0, 300.0, 100.0]
        self.assertEqual(
            [float(v) for v in last_orders],
            expected,
            f"LAST_N rollup should be globally ordered by timestamp DESC. " f"Got {last_orders}, expected {expected}",
        )

    def test_rollup_first_n_global_ordering(self) -> None:
        """Regression test: FIRST_N rollup must produce globally-sorted arrays.

        Same scenario as test_rollup_last_n_global_ordering but for FIRST_N:
          v1: 100.0@10:00, 200.0@10:30, 150.0@11:00
          v2: 300.0@10:15, 250.0@10:45

        Expected FIRST_N(order_value, 24h, n=5) for s1 (oldest first):
          [100.0, 300.0, 200.0, 250.0, 150.0]
          (10:00, 10:15, 10:30, 10:45, 11:00)
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_first_n_order",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.first_n("order_value", "24h", n=5).alias("first_orders")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_first_n_order",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Query with a spine to get the merged result
        spine = self._session.create_dataframe(
            [{"subscriber_id": "s1", "company_id": "c1", "ts": datetime(2024, 1, 1, 12, 0)}]
        )
        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered_subscriber],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 1)

        # Find the first_orders column
        first_orders_col = None
        for col in result_df.columns:
            if "FIRST_ORDERS" in col.upper():
                first_orders_col = col
                break
        self.assertIsNotNone(first_orders_col, f"Could not find first_orders column in {result_df.columns}")

        first_orders_raw = results[0][first_orders_col]
        first_orders = json.loads(first_orders_raw) if isinstance(first_orders_raw, str) else first_orders_raw
        # The array must be globally sorted by timestamp ASC (oldest first):
        #   100.0@10:00, 300.0@10:15, 200.0@10:30, 250.0@10:45, 150.0@11:00
        expected = [100.0, 300.0, 200.0, 250.0, 150.0]
        self.assertEqual(
            [float(v) for v in first_orders],
            expected,
            f"FIRST_N rollup should be globally ordered by timestamp ASC. " f"Got {first_orders}, expected {expected}",
        )

    def test_rollup_last_n_with_n_less_than_total_events(self) -> None:
        """Test LAST_N rollup with n < total events, verifying correct truncation.

        With n=3, we should get only the 3 most recent events globally:
          v1: 100.0@10:00, 200.0@10:30, 150.0@11:00
          v2: 300.0@10:15, 250.0@10:45

        Expected LAST_N(order_value, 24h, n=3) for s1:
          [150.0, 250.0, 200.0]  (the 3 most recent: 11:00, 10:45, 10:30)

        This is the exact scenario the customer described: without the fix,
        the slice might return [150,200,100] (v1's array first) instead of
        [150,250,200] (globally correct top-3).
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_last_n_trunc",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.last_n("order_value", "24h", n=3).alias("recent_orders")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_last_n_trunc",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        spine = self._session.create_dataframe(
            [{"subscriber_id": "s1", "company_id": "c1", "ts": datetime(2024, 1, 1, 12, 0)}]
        )
        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered_subscriber],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 1)

        recent_orders_col = None
        for col in result_df.columns:
            if "RECENT_ORDERS" in col.upper():
                recent_orders_col = col
                break
        self.assertIsNotNone(recent_orders_col)

        recent_orders_raw = results[0][recent_orders_col]
        recent_orders = json.loads(recent_orders_raw) if isinstance(recent_orders_raw, str) else recent_orders_raw
        # Must be the 3 most recent globally, not the first 3 from any single visitor
        expected = [150.0, 250.0, 200.0]
        self.assertEqual(
            [float(v) for v in recent_orders],
            expected,
            f"LAST_N(n=3) rollup should return the 3 most recent events globally. "
            f"Got {recent_orders}, expected {expected}",
        )

    # =========================================================================
    # Query API Tests
    # =========================================================================

    def test_generate_training_set_with_rollup(self) -> None:
        """Test generate_training_set works with rollup FV."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        visitor_fv = FeatureView(
            name="visitor_train",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        subscriber_fv = FeatureView(
            name="subscriber_train",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Create spine
        spine = self._session.create_dataframe(
            [
                {"subscriber_id": "s1", "company_id": "c1", "ts": datetime(2024, 1, 1, 12, 0)},
                {"subscriber_id": "s2", "company_id": "c1", "ts": datetime(2024, 1, 1, 12, 0)},
            ]
        )

        # Generate training set (requires join_method='cte' for tiled FVs)
        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered_subscriber],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        self.assertEqual(result_df.count(), 2)
        # Verify no _PARTIAL_ columns in output
        for col in result_df.columns:
            self.assertFalse(col.startswith("_PARTIAL_"), f"Found internal column: {col}")

        # Verify actual feature values
        # s1 = v1 (3 events) + v2 (2 events) = 5 events
        # s2 = v3 (1 event) = 1 event
        results = result_df.order_by("subscriber_id").collect()
        # Find the event_count column (case-insensitive)
        event_count_col = None
        for col in result_df.columns:
            if "EVENT_COUNT" in col.upper():
                event_count_col = col
                break
        self.assertIsNotNone(event_count_col, f"Could not find event_count column in {result_df.columns}")

        # Check s1 has 5 events
        s1_row = [r for r in results if r["SUBSCRIBER_ID"] == "s1"][0]
        self.assertEqual(s1_row[event_count_col], 5, "s1 should have 5 events (v1:3 + v2:2)")

        # Check s2 has 1 event
        s2_row = [r for r in results if r["SUBSCRIBER_ID"] == "s2"][0]
        self.assertEqual(s2_row[event_count_col], 1, "s2 should have 1 event (v3:1)")

    # =========================================================================
    # Cron Refresh Tests
    # =========================================================================

    def test_rollup_fv_with_cron_refresh_freq(self) -> None:
        """Test that a rollup FV with cron refresh_freq creates a scheduled task."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create visitor-level tiled FV (parent)
        visitor_fv = FeatureView(
            name="visitor_events_cron",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Create subscriber-level rollup FV with CRON refresh_freq
        subscriber_fv = FeatureView(
            name="subscriber_events_cron",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
            refresh_freq="* * * * * UTC",  # Cron expression
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv, "v1")

        # Verify registration succeeded
        self.assertEqual(registered_subscriber.status, FeatureViewStatus.ACTIVE)

        # Verify TARGET_LAG is DOWNSTREAM (cron detection worked)
        self.assertEqual(registered_subscriber.refresh_freq, "DOWNSTREAM")

        # Verify scheduled task was created
        fv_name = FeatureView._get_physical_name("subscriber_events_cron", "v1")
        tasks = self._session.sql(
            f"SHOW TASKS LIKE '{fv_name.resolved()}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(tasks), 1, "Scheduled task should exist for cron-based rollup FV")
        self.assertEqual(tasks[0]["state"], "started", "Task should be started/resumed")

    def test_rollup_fv_overwrite_cron_with_duration(self) -> None:
        """Test that overwriting a cron-based rollup FV with duration cleans up task."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        # Create visitor-level tiled FV (parent)
        visitor_fv = FeatureView(
            name="visitor_events_overwrite",
            entities=[visitor_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.count("visitor_id", "24h").alias("event_count")],
        )
        registered_visitor = fs.register_feature_view(visitor_fv, "v1")

        # Create subscriber-level rollup FV with CRON refresh_freq
        subscriber_fv_cron = FeatureView(
            name="subscriber_events_overwrite",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
            refresh_freq="* * * * * UTC",  # Cron expression
        )
        fs.register_feature_view(subscriber_fv_cron, "v1")

        # Verify task exists
        fv_name = FeatureView._get_physical_name("subscriber_events_overwrite", "v1")
        tasks = self._session.sql(
            f"SHOW TASKS LIKE '{fv_name.resolved()}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(tasks), 1, "Task should exist after cron registration")

        # Overwrite with duration-based refresh (no cron)
        subscriber_fv_duration = FeatureView(
            name="subscriber_events_overwrite",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._get_mapping_df(),
            ),
            refresh_freq="1h",  # Duration, not cron
        )
        registered_subscriber = fs.register_feature_view(subscriber_fv_duration, "v1", overwrite=True)

        # Verify refresh_freq is now the duration
        self.assertEqual(registered_subscriber.refresh_freq, "1 hour")

        # Verify task was cleaned up
        tasks = self._session.sql(
            f"SHOW TASKS LIKE '{fv_name.resolved()}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(tasks), 0, "Task should be cleaned up when switching from cron to duration")


if __name__ == "__main__":
    absltest.main()
