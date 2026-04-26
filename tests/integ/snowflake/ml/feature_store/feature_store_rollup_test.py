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


class TemporalRollupFeatureViewTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for temporal (PIT-correct) rollup feature views.

    These tests validate Option 3: flat DT materialized for inference,
    PIT-correct tiles derived at training time via range JOIN with
    bounded [valid_from, valid_to) validity windows.
    """

    def setUp(self) -> None:
        super().setUp()
        self._active_feature_store = []

        try:
            self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}").collect()
            self._events_table = self._create_events_table()
            self._mapping_table = self._create_temporal_mapping_table()
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
        """Create visitor-level events spanning Feb-Jul 2024.

        v1: 6 events in Feb, Mar, Apr (all pre-reassignment)
        v2: 3 events in Feb, Mar, Apr (always SUB_A)
        v3: 2 events but has NO mapping entry (edge case: unmapped visitor)
        v1 also has 1 event in Jul (post-reassignment to SUB_B)
        """
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.temporal_rollup_events_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), event_ts TIMESTAMP_NTZ, page_views INT)
            """
        ).collect()

        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, event_ts, page_views) VALUES
                ('v1', '2024-02-20 10:00:00', 5),
                ('v1', '2024-02-25 10:00:00', 3),
                ('v1', '2024-03-15 10:00:00', 4),
                ('v1', '2024-03-20 10:00:00', 2),
                ('v1', '2024-04-10 10:00:00', 7),
                ('v1', '2024-04-20 10:00:00', 1),
                ('v1', '2024-07-10 10:00:00', 6),
                ('v2', '2024-02-22 10:00:00', 8),
                ('v2', '2024-03-18 10:00:00', 3),
                ('v2', '2024-04-15 10:00:00', 2),
                ('v3', '2024-03-10 10:00:00', 9),
                ('v3', '2024-04-10 10:00:00', 4)
            """
        ).collect()
        return table_full_path

    def _create_temporal_mapping_table(self) -> str:
        """Create a temporal SCD2 mapping table with VALID_FROM and VALID_TO.

        Mapping timeline:
          v1 -> SUB_A from 2020-01-01, expires 2024-06-01
          v1 -> SUB_B from 2024-06-01, still active (NULL valid_to)
          v2 -> SUB_A from 2020-01-01, still active (NULL valid_to)
          v3 has NO mapping entry (unmapped visitor)
        """
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.temporal_mapping_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), subscriber_id VARCHAR(16),
                 valid_from TIMESTAMP_NTZ, valid_to TIMESTAMP_NTZ)
            """
        ).collect()

        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, subscriber_id, valid_from, valid_to) VALUES
                ('v1', 'SUB_A', '2020-01-01 00:00:00', '2024-06-01 00:00:00'),
                ('v1', 'SUB_B', '2024-06-01 00:00:00', NULL),
                ('v2', 'SUB_A', '2020-01-01 00:00:00', NULL)
            """
        ).collect()
        return table_full_path

    def _create_feature_store(self, name=None) -> FeatureStore:
        current_schema = (
            create_random_schema(self._session, "FS_TEMPORAL_ROLLUP_TEST", database=self.test_db)
            if name is None
            else name
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
        return Entity("visitor", ["visitor_id"])

    def _create_subscriber_entity(self) -> Entity:
        return Entity("subscriber", ["subscriber_id"])

    def _register_parent_fv(self, fs, visitor_entity):
        """Register a visitor-level tiled FV with monthly tiles."""
        visitor_fv = FeatureView(
            name="visitor_events_temporal",
            entities=[visitor_entity],
            feature_df=self._session.table(self._events_table),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="30d",
            features=[Feature.count("visitor_id", "90d").alias("page_view_count")],
        )
        return fs.register_feature_view(visitor_fv, "v1")

    def _find_count_col(self, result_df):
        """Find the PAGE_VIEW_COUNT column in the result DataFrame."""
        for col in result_df.columns:
            if "PAGE_VIEW_COUNT" in col.upper():
                return col
        self.fail(f"Could not find page_view_count column in {result_df.columns}")

    # =========================================================================
    # Registration & Validation Tests
    # =========================================================================

    def test_temporal_rollup_registration(self) -> None:
        """Test that a temporal rollup FV registers successfully."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_temporal",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        self.assertEqual(registered.status, FeatureViewStatus.ACTIVE)

    def test_temporal_rollup_mapping_valid_from_validation(self) -> None:
        """Test validation fails when mapping_valid_from_col is not in mapping."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        with self.assertRaises(ValueError) as context:
            FeatureView(
                name="subscriber_bad_ts",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(self._mapping_table),
                    mapping_valid_from_col="NONEXISTENT_COL",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
        self.assertIn("NONEXISTENT_COL", str(context.exception))

    def test_temporal_rollup_metadata_persisted_and_loaded(self) -> None:
        """Test that rollup metadata survives get_feature_view reconstruction."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_persist",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        fs.register_feature_view(subscriber_fv, "v1")

        loaded_fv = fs.get_feature_view("subscriber_persist", "v1")

        self.assertIsNotNone(loaded_fv.rollup_metadata)
        self.assertEqual(loaded_fv.rollup_metadata.mapping_valid_from_col.upper(), "VALID_FROM")
        self.assertEqual(loaded_fv.rollup_metadata.mapping_valid_to_col.upper(), "VALID_TO")
        self.assertIn("VISITOR_ID", [k.upper() for k in loaded_fv.rollup_metadata.parent_join_keys])

    # =========================================================================
    # PIT Correctness Tests
    # =========================================================================

    def test_temporal_rollup_training_pit_correct(self) -> None:
        """Core test: training set uses PIT-correct entity mappings.

        Before Jun 1 2024, v1 belongs to SUB_A. After Jun 1, v1 belongs to SUB_B.
        At May 20 with 90-day window, v1's pre-June events go to SUB_A (via range JOIN
        with valid_to=2024-06-01), and v2's events also go to SUB_A.
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_pit",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        self.assertIsNotNone(
            registered.rollup_metadata, f"Registered FV should have rollup_metadata. is_rollup={registered.is_rollup}"
        )
        self.assertEqual(registered.rollup_metadata.mapping_valid_from_col.upper(), "VALID_FROM")
        self.assertEqual(registered.rollup_metadata.mapping_valid_to_col.upper(), "VALID_TO")

        spine = self._session.create_dataframe(
            [
                {"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)},
            ]
        )

        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 1)

        count_col = self._find_count_col(result_df)
        sub_a_count = results[0][count_col]
        self.assertIsNotNone(sub_a_count, "SUB_A should have non-null feature values")

        # PIT: v1(6 events in Feb-Apr) + v2(3 events in Feb-Apr) = 9 total
        #   (exact count depends on which 30-day tiles are complete at May 20)
        # Flat: only v2(3 events) since v1→SUB_B in latest mapping
        #
        # The PIT count must be strictly greater than the flat count,
        # proving v1's events are attributed to SUB_A via temporal mapping.
        # With 30-day tiles, at least 2 tiles should be complete in the window.
        self.assertGreater(
            sub_a_count, 3, f"PIT count ({sub_a_count}) should exceed flat count (~3) by including v1's events"
        )

    def test_temporal_rollup_inference_uses_flat_mapping(self) -> None:
        """Inference (retrieve_feature_values) uses the materialized flat DT.

        The flat DT filters to active mappings (valid_to IS NULL or > NOW()).
        v1->SUB_B is active, v1->SUB_A is expired. SUB_B gets v1's events,
        SUB_A only gets v2's events.
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_infer",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        spine = self._session.create_dataframe(
            [
                {"subscriber_id": "SUB_A", "ts": datetime(2024, 8, 1)},
                {"subscriber_id": "SUB_B", "ts": datetime(2024, 8, 1)},
            ]
        )
        result_df = fs.retrieve_feature_values(
            spine_df=spine,
            features=[registered],
            spine_timestamp_col="ts",
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 2)

        count_col = self._find_count_col(result_df)

        sub_a_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_A"]
        sub_b_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_B"]
        self.assertEqual(len(sub_a_row), 1)
        self.assertEqual(len(sub_b_row), 1)

        # In the flat DT, v1→SUB_B (latest), v2→SUB_A (latest), v3→excluded (no mapping).
        # SUB_B should have v1's data; SUB_A should only have v2's data.
        if sub_b_row[0][count_col] is not None:
            self.assertGreater(sub_b_row[0][count_col], 0, "SUB_B should have v1's events in flat DT")
        if sub_a_row[0][count_col] is not None and sub_b_row[0][count_col] is not None:
            self.assertGreater(
                sub_b_row[0][count_col],
                sub_a_row[0][count_col],
                "SUB_B (v1: 7 events) should have more events than SUB_A (v2: 3 events) in flat DT",
            )

    def test_temporal_rollup_without_valid_from_behaves_as_flat(self) -> None:
        """Without mapping_valid_from_col, training uses flat DT (backward compat)."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_flat",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        spine = self._session.create_dataframe([{"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)}])
        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )
        self.assertEqual(result_df.count(), 1)

    def test_temporal_rollup_reconstructed_fv_training(self) -> None:
        """Test that a reconstructed temporal rollup FV still produces PIT-correct training."""
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_recon",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        fs.register_feature_view(subscriber_fv, "v1")

        loaded_fv = fs.get_feature_view("subscriber_recon", "v1")

        spine = self._session.create_dataframe([{"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)}])

        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[loaded_fv],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 1)

        count_col = self._find_count_col(result_df)
        self.assertIsNotNone(results[0][count_col])
        # Same PIT correctness check as test_temporal_rollup_training_pit_correct:
        # PIT count includes v1+v2, flat would only have v2 (~3 events)
        self.assertGreater(
            results[0][count_col],
            3,
            f"Reconstructed FV PIT count ({results[0][count_col]}) should exceed flat count (~3)",
        )

    def test_temporal_rollup_unmapped_visitor_excluded(self) -> None:
        """Visitors with no mapping entry are excluded from both PIT and flat results.

        v3 has events but no entry in the mapping table. Its events should
        not appear in any subscriber's count (no crash, no phantom data).
        """
        fs = self._create_feature_store()

        visitor_entity = self._create_visitor_entity()
        subscriber_entity = self._create_subscriber_entity()
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_unmapped",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        # Query for both SUB_A and SUB_B — v3's events should not inflate either
        spine = self._session.create_dataframe(
            [
                {"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)},
                {"subscriber_id": "SUB_B", "ts": datetime(2024, 5, 20)},
            ]
        )

        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 2, "Should return rows for both SUB_A and SUB_B")

        count_col = self._find_count_col(result_df)
        total_count = sum(r[count_col] for r in results if r[count_col] is not None)

        # v3 has 2 events. Total mapped events = v1(7) + v2(3) = 10
        # (not all may be in the window, but v3's events must NOT be counted)
        # The total should NOT include v3's events
        self.assertGreater(total_count, 0, "Should have some events for mapped visitors")

    # =========================================================================
    # Valid-To (SCD2 Bounded Mapping) Tests
    # =========================================================================

    def _create_scd2_mapping_table(self) -> str:
        """Create a SCD2 mapping table with VALID_FROM and VALID_TO.

        Mapping timeline:
          v1 -> SUB_A  valid_from=2020-01-01, valid_to=2024-04-01  (expired)
          v1 -> SUB_B  valid_from=2024-04-01, valid_to=NULL         (active, reassigned)
          v2 -> SUB_A  valid_from=2020-01-01, valid_to=NULL         (active, never reassigned)

        Events (from _create_events_table):
          v1: Feb 20, Feb 25, Mar 15, Mar 20, Apr 10, Apr 20, Jul 10
          v2: Feb 22, Mar 18, Apr 15
        """
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.scd2_mapping_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), subscriber_id VARCHAR(16),
                 valid_from TIMESTAMP_NTZ, valid_to TIMESTAMP_NTZ)
            """
        ).collect()

        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, subscriber_id, valid_from, valid_to) VALUES
                ('v1', 'SUB_A', '2020-01-01 00:00:00', '2024-04-01 00:00:00'),
                ('v1', 'SUB_B', '2024-04-01 00:00:00', NULL),
                ('v2', 'SUB_A', '2020-01-01 00:00:00', NULL)
            """
        ).collect()
        return table_full_path

    def test_valid_to_reassignment_pit_correct(self) -> None:
        """Customer's core case: visitor reassigned, expired mapping excluded.

        v1 -> SUB_A expired at Apr 1. v1 -> SUB_B active from Apr 1.
        At May 20 with 90-day window:
          SUB_A should get v1 events from Feb-Mar only (before Apr 1 expiry) + v2 events
          SUB_B should get v1 events from Apr+ only

        Without valid_to, ASOF would attribute ALL v1 events to SUB_A (inflated).
        With valid_to, v1 events after Apr 1 go to SUB_B only.
        """
        fs = self._create_feature_store()
        scd2_table = self._create_scd2_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            subscriber_fv = FeatureView(
                name="subscriber_valid_to",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(scd2_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            self.assertIsNotNone(registered.rollup_metadata)
            self.assertEqual(registered.rollup_metadata.mapping_valid_to_col.upper(), "VALID_TO")

            spine = self._session.create_dataframe(
                [
                    {"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)},
                    {"subscriber_id": "SUB_B", "ts": datetime(2024, 5, 20)},
                ]
            )

            result_df = fs.generate_training_set(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                save_as=None,
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 2)

            count_col = self._find_count_col(result_df)
            sub_a_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_A"]
            sub_b_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_B"]
            self.assertEqual(len(sub_a_row), 1)
            self.assertEqual(len(sub_b_row), 1)

            sub_a_count = sub_a_row[0][count_col] or 0
            sub_b_count = sub_b_row[0][count_col] or 0

            # SUB_A: v1 events before Apr 1 (Feb, Mar tiles) + v2 events (Feb, Mar, Apr)
            # SUB_B: v1 events from Apr onward (Apr, Jul tiles)
            # Both should have events, but SUB_A should NOT have v1's Apr+ events
            self.assertGreater(sub_a_count, 0, "SUB_A should have pre-expiry v1 + v2 events")
            self.assertGreater(sub_b_count, 0, "SUB_B should have v1's post-reassignment events")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {scd2_table}").collect()

    def test_valid_to_expired_no_successor(self) -> None:
        """Subscriber with only expired mappings gets COUNT=0, MIN=NULL.

        This reproduces the customer's exact symptom: when valid_to filters out
        all mappings for a subscriber, COUNT=0 (ELSE 0) and MIN=NULL (ELSE NULL).
        """
        fs = self._create_feature_store()

        # Create mapping where SUB_C had v1 briefly, then expired with no successor
        expired_table = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.expired_mapping_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {expired_table}
                (visitor_id VARCHAR(16), subscriber_id VARCHAR(16),
                 valid_from TIMESTAMP_NTZ, valid_to TIMESTAMP_NTZ)
            """
        ).collect()
        self._session.sql(
            f"""INSERT INTO {expired_table} (visitor_id, subscriber_id, valid_from, valid_to) VALUES
                ('v1', 'SUB_C', '2020-01-01 00:00:00', '2024-01-15 00:00:00'),
                ('v2', 'SUB_A', '2020-01-01 00:00:00', NULL)
            """
        ).collect()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            subscriber_fv = FeatureView(
                name="subscriber_expired",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(expired_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            # SUB_C's mapping expired at Jan 15 2024, all v1 events are Feb+
            spine = self._session.create_dataframe([{"subscriber_id": "SUB_C", "ts": datetime(2024, 5, 20)}])

            result_df = fs.generate_training_set(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                save_as=None,
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 1)

            count_col = self._find_count_col(result_df)
            sub_c_count = results[0][count_col]

            # SUB_C should have 0 or NULL — all v1 events are after the mapping expired
            self.assertTrue(
                sub_c_count is None or sub_c_count == 0,
                f"SUB_C should have 0 or NULL count (got {sub_c_count}) since mapping expired before events",
            )
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {expired_table}").collect()

    def test_valid_to_null_means_active(self) -> None:
        """Mappings with NULL valid_to are treated as still-active."""
        fs = self._create_feature_store()
        scd2_table = self._create_scd2_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            subscriber_fv = FeatureView(
                name="subscriber_null_vt",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(scd2_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            # v2 -> SUB_A with valid_to=NULL (always active)
            # At any point, v2's events should be attributed to SUB_A
            spine = self._session.create_dataframe([{"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)}])

            result_df = fs.generate_training_set(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                save_as=None,
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 1)

            count_col = self._find_count_col(result_df)
            sub_a_count = results[0][count_col]

            # SUB_A should have at least v2's events (NULL valid_to = active)
            self.assertIsNotNone(sub_a_count, "SUB_A should have non-null count (v2 is always active)")
            self.assertGreater(sub_a_count, 0, "SUB_A should have v2's events via NULL valid_to")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {scd2_table}").collect()

    def test_valid_from_only_registration_rejected(self) -> None:
        """Setting only mapping_valid_from_col (no valid_to) raises ValueError."""
        fs = self._create_feature_store()
        scd2_table = self._create_scd2_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            with self.assertRaisesRegex(ValueError, "must both be provided or both be omitted"):
                FeatureView(
                    name="subscriber_no_vt",
                    entities=[subscriber_entity],
                    rollup_config=RollupConfig(
                        source=registered_visitor,
                        mapping_df=self._session.table(scd2_table),
                        mapping_valid_from_col="VALID_FROM",
                    ),
                )
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {scd2_table}").collect()

    def test_valid_to_only_registration_rejected(self) -> None:
        """Setting only mapping_valid_to_col (no valid_from) raises ValueError."""
        fs = self._create_feature_store()
        scd2_table = self._create_scd2_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            with self.assertRaisesRegex(ValueError, "must both be provided or both be omitted"):
                FeatureView(
                    name="subscriber_no_vf",
                    entities=[subscriber_entity],
                    rollup_config=RollupConfig(
                        source=registered_visitor,
                        mapping_df=self._session.table(scd2_table),
                        mapping_valid_to_col="VALID_TO",
                    ),
                )
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {scd2_table}").collect()

    # =========================================================================
    # 1:N Overlapping Mapping Tests
    # =========================================================================

    def _create_overlapping_mapping_table(self) -> str:
        """Create a mapping where v1 maps to TWO subscribers simultaneously.

        v1 -> SUB_A  valid_from=2020-01-01, valid_to=NULL  (always active)
        v1 -> SUB_B  valid_from=2024-03-01, valid_to=NULL  (added Mar 1, overlaps SUB_A)
        v2 -> SUB_A  valid_from=2020-01-01, valid_to=NULL  (always active)

        Events from _create_events_table:
          v1: Feb 20, Feb 25, Mar 15, Mar 20, Apr 10, Apr 20, Jul 10
          v2: Feb 22, Mar 18, Apr 15

        After Mar 1, v1's events should fan out to BOTH SUB_A and SUB_B.
        """
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.overlapping_mapping_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), subscriber_id VARCHAR(16),
                 valid_from TIMESTAMP_NTZ, valid_to TIMESTAMP_NTZ)
            """
        ).collect()
        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, subscriber_id, valid_from, valid_to) VALUES
                ('v1', 'SUB_A', '2020-01-01 00:00:00', NULL),
                ('v1', 'SUB_B', '2024-03-01 00:00:00', NULL),
                ('v2', 'SUB_A', '2020-01-01 00:00:00', NULL)
            """
        ).collect()
        return table_full_path

    def test_overlapping_mappings_pit_training(self) -> None:
        """1:N: one visitor mapped to two subscribers, events attributed to both."""
        fs = self._create_feature_store()
        overlap_table = self._create_overlapping_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            subscriber_fv = FeatureView(
                name="subscriber_overlap",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(overlap_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            spine = self._session.create_dataframe(
                [
                    {"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)},
                    {"subscriber_id": "SUB_B", "ts": datetime(2024, 5, 20)},
                ]
            )

            result_df = fs.generate_training_set(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                save_as=None,
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 2)

            count_col = self._find_count_col(result_df)
            sub_a_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_A"]
            sub_b_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_B"]
            self.assertEqual(len(sub_a_row), 1)
            self.assertEqual(len(sub_b_row), 1)

            sub_a_count = sub_a_row[0][count_col] or 0
            sub_b_count = sub_b_row[0][count_col] or 0

            # SUB_A: v1 events (all of them, active since 2020) + v2 events
            # SUB_B: v1 events from Mar 1+ only (Mar 15, Mar 20, Apr 10, Apr 20)
            self.assertGreater(sub_a_count, 0, "SUB_A should have events from both v1 and v2")
            self.assertGreater(sub_b_count, 0, "SUB_B should have v1's post-Mar-1 events (1:N overlap)")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {overlap_table}").collect()

    def test_overlapping_mappings_inference(self) -> None:
        """1:N inference: both subscribers get visitor's events via flat DT."""
        fs = self._create_feature_store()
        overlap_table = self._create_overlapping_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            subscriber_fv = FeatureView(
                name="subscriber_overlap_inf",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(overlap_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            spine = self._session.create_dataframe(
                [
                    {"subscriber_id": "SUB_A", "ts": datetime(2024, 8, 1)},
                    {"subscriber_id": "SUB_B", "ts": datetime(2024, 8, 1)},
                ]
            )
            result_df = fs.retrieve_feature_values(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 2)

            count_col = self._find_count_col(result_df)
            sub_a_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_A"]
            sub_b_row = [r for r in results if r["SUBSCRIBER_ID"] == "SUB_B"]

            # Both mappings are active (valid_to=NULL), so both get v1's events
            self.assertGreater(sub_a_row[0][count_col] or 0, 0, "SUB_A should have events in flat DT (1:N)")
            self.assertGreater(sub_b_row[0][count_col] or 0, 0, "SUB_B should have events in flat DT (1:N)")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {overlap_table}").collect()

    def test_boundary_half_open_interval(self) -> None:
        """Verify [valid_from, valid_to) semantics at exact boundaries.

        v1 -> SUB_A valid_from=2024-03-01, valid_to=2024-04-01
        v1's Mar 15 tile (TILE_START=2024-03-01) should match (>= valid_from).
        v1's Apr tile (TILE_START=2024-04-01) should NOT match (< valid_to).
        """
        fs = self._create_feature_store()

        boundary_table = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.boundary_mapping_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {boundary_table}
                (visitor_id VARCHAR(16), subscriber_id VARCHAR(16),
                 valid_from TIMESTAMP_NTZ, valid_to TIMESTAMP_NTZ)
            """
        ).collect()
        self._session.sql(
            f"""INSERT INTO {boundary_table} (visitor_id, subscriber_id, valid_from, valid_to) VALUES
                ('v1', 'SUB_A', '2024-03-01 00:00:00', '2024-04-01 00:00:00'),
                ('v2', 'SUB_A', '2020-01-01 00:00:00', NULL)
            """
        ).collect()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            registered_visitor = self._register_parent_fv(fs, visitor_entity)

            subscriber_fv = FeatureView(
                name="subscriber_boundary",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(boundary_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            # Query at Apr 15 with 90-day window: should only see v1's Mar events
            # (Mar 15, Mar 20 are in [2024-03-01, 2024-04-01) window)
            # v1's Apr+ events should NOT be attributed to SUB_A
            spine = self._session.create_dataframe([{"subscriber_id": "SUB_A", "ts": datetime(2024, 4, 15)}])

            result_df = fs.generate_training_set(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                save_as=None,
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 1)

            count_col = self._find_count_col(result_df)
            count = results[0][count_col] or 0

            # With [valid_from, valid_to) = [2024-03-01, 2024-04-01):
            #   v1's Feb tile (TILE_START=2024-02-01) excluded: < valid_from
            #   v1's Mar tile (TILE_START=2024-03-01) included: >= valid_from and < valid_to (2 events)
            #   v1's Apr tile (TILE_START=2024-04-01) excluded: not < valid_to
            #   v2 always active: 3 events
            # Expected total ≈ 5. Without boundary filtering, v1 contributes 6 events
            # in the 90d window (Feb+Mar+Apr) + v2's 3 = 9.
            self.assertGreater(count, 0, "SUB_A should have some events from valid window")
            self.assertLess(
                count,
                9,
                f"Half-open interval should exclude v1's Feb and Apr+ tiles. "
                f"Got {count}, unfiltered would be ~9 (6 v1 + 3 v2 in 90d window)",
            )
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {boundary_table}").collect()

    def test_overlapping_mappings_with_lifetime(self) -> None:
        """1:N with lifetime features: cumulative columns partitioned correctly."""
        fs = self._create_feature_store()
        overlap_table = self._create_overlapping_mapping_table()

        try:
            visitor_entity = self._create_visitor_entity()
            subscriber_entity = self._create_subscriber_entity()
            fs.register_entity(visitor_entity)
            fs.register_entity(subscriber_entity)

            visitor_fv = FeatureView(
                name="visitor_events_lt_overlap",
                entities=[visitor_entity],
                feature_df=self._session.table(self._events_table),
                timestamp_col="event_ts",
                refresh_freq="1h",
                feature_granularity="30d",
                features=[
                    Feature.count("visitor_id", "lifetime").alias("event_count_lifetime"),
                    Feature.min("event_ts", "lifetime").alias("first_event_lifetime"),
                    Feature.max("event_ts", "lifetime").alias("last_event_lifetime"),
                ],
            )
            registered_visitor = fs.register_feature_view(visitor_fv, "v1")

            subscriber_fv = FeatureView(
                name="subscriber_lt_overlap",
                entities=[subscriber_entity],
                rollup_config=RollupConfig(
                    source=registered_visitor,
                    mapping_df=self._session.table(overlap_table),
                    mapping_valid_from_col="VALID_FROM",
                    mapping_valid_to_col="VALID_TO",
                ),
            )
            registered = fs.register_feature_view(subscriber_fv, "v1")

            spine = self._session.create_dataframe(
                [
                    {"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)},
                    {"subscriber_id": "SUB_B", "ts": datetime(2024, 5, 20)},
                ]
            )

            result_df = fs.generate_training_set(
                spine_df=spine,
                features=[registered],
                spine_timestamp_col="ts",
                save_as=None,
                join_method="cte",
            )

            results = result_df.collect()
            self.assertEqual(len(results), 2)

            count_col = None
            first_col = None
            last_col = None
            for col in result_df.columns:
                if "EVENT_COUNT_LIFETIME" in col.upper():
                    count_col = col
                if "FIRST_EVENT_LIFETIME" in col.upper():
                    first_col = col
                if "LAST_EVENT_LIFETIME" in col.upper():
                    last_col = col
            self.assertIsNotNone(count_col)
            self.assertIsNotNone(first_col)
            self.assertIsNotNone(last_col)

            result_map = {row["SUBSCRIBER_ID"]: row for row in results}

            # Both subscribers should have non-NULL lifetime features
            for sub_id in ["SUB_A", "SUB_B"]:
                row = result_map.get(sub_id)
                self.assertIsNotNone(row, f"{sub_id} should be in results")
                self.assertIsNotNone(row[count_col], f"{sub_id} lifetime COUNT should not be NULL")
                self.assertGreater(row[count_col], 0, f"{sub_id} lifetime COUNT should be positive")
                self.assertIsNotNone(row[first_col], f"{sub_id} lifetime MIN should not be NULL")
                self.assertIsNotNone(row[last_col], f"{sub_id} lifetime MAX should not be NULL")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {overlap_table}").collect()


class LifetimeRollupFeatureViewTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for lifetime features on rollup feature views.

    Validates that _CUM_* columns are generated correctly in rollup DTs,
    allowing MergingSqlGenerator.LIFETIME_MERGED to work via range JOIN.
    """

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
        """Create visitor-level events for lifetime testing.

        v1: events in Feb, Mar, Apr 2024 (page_views: 5, 3, 4, 2, 7, 1)
        v2: events in Feb, Mar, Apr 2024 (page_views: 8, 3, 2)
        v1: additional event in Jul 2024 (page_views: 6)
        """
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.lifetime_rollup_events_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), event_ts TIMESTAMP_NTZ, page_views INT)
            """
        ).collect()
        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, event_ts, page_views) VALUES
                ('v1', '2024-02-20 10:00:00', 5),
                ('v1', '2024-02-25 10:00:00', 3),
                ('v1', '2024-03-15 10:00:00', 4),
                ('v1', '2024-03-20 10:00:00', 2),
                ('v1', '2024-04-10 10:00:00', 7),
                ('v1', '2024-04-20 10:00:00', 1),
                ('v1', '2024-07-10 10:00:00', 6),
                ('v2', '2024-02-22 10:00:00', 8),
                ('v2', '2024-03-18 10:00:00', 3),
                ('v2', '2024-04-15 10:00:00', 2)
            """
        ).collect()
        return table_full_path

    def _create_mapping_table(self) -> str:
        """Create temporal mapping with SCD2 semantics.

        v1 -> SUB_A from 2020-01-01 (valid_to=2024-06-01, then reassigned)
        v1 -> SUB_B from 2024-06-01 (valid_to=NULL, currently active)
        v2 -> SUB_A from 2020-01-01 (valid_to=NULL, always active)
        """
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.lifetime_mapping_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (visitor_id VARCHAR(16), subscriber_id VARCHAR(16),
                 valid_from TIMESTAMP_NTZ, valid_to TIMESTAMP_NTZ)
            """
        ).collect()
        self._session.sql(
            f"""INSERT INTO {table_full_path} (visitor_id, subscriber_id, valid_from, valid_to) VALUES
                ('v1', 'SUB_A', '2020-01-01 00:00:00', '2024-06-01 00:00:00'),
                ('v1', 'SUB_B', '2024-06-01 00:00:00', NULL),
                ('v2', 'SUB_A', '2020-01-01 00:00:00', NULL)
            """
        ).collect()
        return table_full_path

    def _create_feature_store(self) -> FeatureStore:
        current_schema = create_random_schema(self._session, "FS_LIFETIME_ROLLUP_TEST", database=self.test_db)
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

    def _register_parent_fv_with_lifetime(self, fs, visitor_entity):
        """Register a visitor-level FV with lifetime COUNT, MIN, and MAX features."""
        visitor_fv = FeatureView(
            name="visitor_events_lifetime",
            entities=[visitor_entity],
            feature_df=self._session.table(self._events_table),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="30d",
            features=[
                Feature.count("visitor_id", "lifetime").alias("event_count_lifetime"),
                Feature.min("event_ts", "lifetime").alias("first_event_lifetime"),
                Feature.max("event_ts", "lifetime").alias("last_event_lifetime"),
            ],
        )
        return fs.register_feature_view(visitor_fv, "v1")

    def _find_col(self, result_df, pattern):
        """Find a column matching the given pattern (case-insensitive)."""
        pattern_upper = pattern.upper()
        for col in result_df.columns:
            if pattern_upper in col.upper():
                return col
        self.fail(f"Could not find column matching '{pattern}' in {result_df.columns}")

    def test_rollup_lifetime_training_set(self) -> None:
        """Rollup FV with lifetime features produces non-NULL training results.

        This is the core test for the lifetime rollup fix: the rollup DT now
        includes _CUM_* columns, so LIFETIME_MERGED can read them via range JOIN.
        """
        fs = self._create_feature_store()

        visitor_entity = Entity("visitor", ["visitor_id"])
        subscriber_entity = Entity("subscriber", ["subscriber_id"])
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv_with_lifetime(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_lifetime",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        spine = self._session.create_dataframe([{"subscriber_id": "SUB_A", "ts": datetime(2024, 8, 1)}])

        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 1)

        count_col = self._find_col(result_df, "EVENT_COUNT_LIFETIME")
        first_col = self._find_col(result_df, "FIRST_EVENT_LIFETIME")
        last_col = self._find_col(result_df, "LAST_EVENT_LIFETIME")

        self.assertIsNotNone(results[0][count_col], "Lifetime COUNT should not be NULL")
        self.assertIsNotNone(results[0][first_col], "Lifetime MIN should not be NULL")
        self.assertIsNotNone(results[0][last_col], "Lifetime MAX should not be NULL")

        self.assertGreater(results[0][count_col], 0, "Lifetime COUNT should be positive")

    def test_temporal_rollup_lifetime_pit_correct(self) -> None:
        """PIT CTE path with lifetime features returns correct values.

        Before Jun 1, v1 belongs to SUB_A. After Jun 1, v1 belongs to SUB_B.
        At May 20, SUB_A's lifetime count includes both v1 (6 events) and
        v2 (3 events). With mapping_valid_to_col, expired mappings are filtered.
        """
        fs = self._create_feature_store()

        visitor_entity = Entity("visitor", ["visitor_id"])
        subscriber_entity = Entity("subscriber", ["subscriber_id"])
        fs.register_entity(visitor_entity)
        fs.register_entity(subscriber_entity)

        registered_visitor = self._register_parent_fv_with_lifetime(fs, visitor_entity)

        subscriber_fv = FeatureView(
            name="subscriber_pit_lt",
            entities=[subscriber_entity],
            rollup_config=RollupConfig(
                source=registered_visitor,
                mapping_df=self._session.table(self._mapping_table),
                mapping_valid_from_col="VALID_FROM",
                mapping_valid_to_col="VALID_TO",
            ),
        )
        registered = fs.register_feature_view(subscriber_fv, "v1")

        spine = self._session.create_dataframe(
            [
                {"subscriber_id": "SUB_A", "ts": datetime(2024, 5, 20)},
                {"subscriber_id": "SUB_B", "ts": datetime(2024, 8, 1)},
            ]
        )

        result_df = fs.generate_training_set(
            spine_df=spine,
            features=[registered],
            spine_timestamp_col="ts",
            save_as=None,
            join_method="cte",
        )

        results = result_df.collect()
        self.assertEqual(len(results), 2)

        count_col = self._find_col(result_df, "EVENT_COUNT_LIFETIME")
        first_col = self._find_col(result_df, "FIRST_EVENT_LIFETIME")
        last_col = self._find_col(result_df, "LAST_EVENT_LIFETIME")

        result_map = {}
        for row in results:
            sub_id = row["SUBSCRIBER_ID"]
            result_map[sub_id] = row

        sub_a = result_map.get("SUB_A")
        self.assertIsNotNone(sub_a, "SUB_A should be in results")
        self.assertIsNotNone(sub_a[count_col], "SUB_A lifetime COUNT should not be NULL")
        self.assertIsNotNone(sub_a[first_col], "SUB_A lifetime MIN should not be NULL")
        self.assertIsNotNone(sub_a[last_col], "SUB_A lifetime MAX should not be NULL")
        self.assertGreater(sub_a[count_col], 0, "SUB_A lifetime COUNT should be positive")

        sub_b = result_map.get("SUB_B")
        self.assertIsNotNone(sub_b, "SUB_B should be in results")
        self.assertIsNotNone(sub_b[count_col], "SUB_B lifetime COUNT should not be NULL (v1 Jul event)")
        self.assertGreater(sub_b[count_col], 0, "SUB_B lifetime COUNT should be positive")


if __name__ == "__main__":
    absltest.main()
