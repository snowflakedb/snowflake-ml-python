"""Performance test for Feature Store dataset generation with duplicate spine rows.

This test validates that the CTE method can handle large-scale datasets with:
- Multiple feature views with different join keys
- Duplicate spine rows (multiple subscribers per company at same timestamp)
- Millions of rows in both spine and feature views
"""

import time
from typing import Optional
from uuid import uuid4

from absl.testing import absltest
from common_utils import (
    FS_INTEG_TEST_DB,
    cleanup_temporary_objects,
    create_random_schema,
)

from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

# ============================================================================
# CONFIGURABLE TEST PARAMETERS
# ============================================================================

# Warehouse configuration
TEST_WAREHOUSE = "FS_PERF_TEST_4XL"  # Dedicated 4XL warehouse for this performance test

# Number of feature views to create
NUM_COMPANY_FVS = 5  # Feature views with company_id join key
NUM_SUBSCRIBER_FVS = 5  # Feature views with subscriber_id join key

# Feature view schema configuration
NUM_FEATURES_PER_FV = 2  # Number of feature columns in each feature view

# Data sizes (in millions)
FV_ROWS_MILLIONS = 2
SPINE_ROWS_MILLIONS = 1

# Derived parameters
FV_ROWS = FV_ROWS_MILLIONS * 1_000_000
SPINE_ROWS = SPINE_ROWS_MILLIONS * 1_000_000

# Number of unique companies and subscribers
NUM_COMPANIES = 100
NUM_SUBSCRIBERS = 100

# Timestamp range (in seconds)
MIN_TIMESTAMP = 1000
MAX_TIMESTAMP = 2000

# ============================================================================


class FeatureStoreDatasetPerfTest(absltest.TestCase):
    """Performance tests for Feature Store with large datasets and duplicates."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
        cls._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        cleanup_temporary_objects(cls._session)
        cls._active_feature_store = []
        cls._test_tables = []

        # Create Snowpark-optimized 4XL warehouse for this test if it doesn't exist
        try:
            cls._session.sql(
                f"CREATE WAREHOUSE IF NOT EXISTS {TEST_WAREHOUSE} "
                f"WITH WAREHOUSE_SIZE='4X-LARGE' "
                f"WAREHOUSE_TYPE='SNOWPARK-OPTIMIZED' "
                f"AUTO_SUSPEND=60 "
                f"AUTO_RESUME=TRUE"
            ).collect()

            # Activate the warehouse in the session
            cls._session.use_warehouse(TEST_WAREHOUSE)
        except Exception as e:
            print(f"Warning: Could not create/use warehouse {TEST_WAREHOUSE}: {e}")

        cls._test_warehouse_name = TEST_WAREHOUSE

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test environment."""
        for fs in cls._active_feature_store:
            try:
                fs._clear(dryrun=False)
                cls._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
            except Exception as e:
                print(f"Warning: Failed to clean up feature store: {e}")

        # Clean up test tables
        for table in cls._test_tables:
            try:
                cls._session.sql(f"DROP TABLE IF EXISTS {table}").collect()
            except Exception as e:
                print(f"Warning: Failed to drop table {table}: {e}")

        cls._session.close()

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        """Create a feature store for testing."""
        current_schema = create_random_schema(self._session, "FS_PERF_TEST") if name is None else name

        # Use the configured test warehouse (4XL for large-scale tests)
        print(f"Using warehouse: {TEST_WAREHOUSE}")

        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            default_warehouse=TEST_WAREHOUSE,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        return fs

    def _create_large_table(
        self,
        fs: FeatureStore,
        table_name: str,
        num_rows: int,
        join_key: str,
        num_unique_keys: int,
        num_features: int = 1,
    ) -> str:
        """Create a large table with synthetic data.

        Args:
            fs: Feature store instance (to get schema)
            table_name: Name of the table to create
            num_rows: Total number of rows
            join_key: Name of the join key column (company_id or subscriber_id)
            num_unique_keys: Number of unique values for the join key
            num_features: Number of feature columns to create (default: 1)

        Returns:
            Full path to the created table
        """
        full_table_name = f"{fs._config.full_schema_path}.{table_name}"
        self._test_tables.append(full_table_name)

        # Use 4XL warehouse for data generation
        self._session.sql(f"USE WAREHOUSE {TEST_WAREHOUSE}").collect()

        # Build column definitions for multiple features
        feature_columns = [f"feature_{i} FLOAT" for i in range(num_features)]
        columns_def = f"{join_key} INT, ts INT, " + ", ".join(feature_columns)

        # Create table
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {full_table_name} (
                {columns_def}
            )
        """
        ).collect()

        # Build SELECT clause for multiple features
        feature_selects = [f"UNIFORM(0, 1000, RANDOM())::FLOAT AS feature_{i}" for i in range(num_features)]
        select_clause = (
            f"UNIFORM(1, {num_unique_keys}, RANDOM()) AS {join_key}, "
            f"UNIFORM({MIN_TIMESTAMP}, {MAX_TIMESTAMP}, RANDOM()) AS ts, " + ", ".join(feature_selects)
        )

        # Insert data using generator
        self._session.sql(
            f"""
            INSERT INTO {full_table_name}
            SELECT {select_clause}
            FROM TABLE(GENERATOR(ROWCOUNT => {num_rows}))
        """
        ).collect()

        return full_table_name

    def test_large_scale_dataset_generation_with_duplicates(self) -> None:
        """Test dataset generation with large scale data and duplicate spine rows.

        This test creates:
        - Multiple feature views with company_id (each with configurable rows and features)
        - Multiple feature views with subscriber_id (each with configurable rows and features)
        - 1 spine with configurable rows containing duplicate (company_id, timestamp) combinations

        The spine has duplicates because multiple subscribers belong to the same company
        at the same timestamp.
        """
        print(
            f"\nPerformance Test: {NUM_COMPANY_FVS + NUM_SUBSCRIBER_FVS} FVs, "
            f"{(NUM_COMPANY_FVS + NUM_SUBSCRIBER_FVS) * NUM_FEATURES_PER_FV} features, "
            f"{SPINE_ROWS:,} spine rows"
        )

        fs = self._create_feature_store()

        # Create entities
        e_company = Entity("company", ["company_id"])
        fs.register_entity(e_company)

        e_subscriber = Entity("subscriber", ["subscriber_id"])
        fs.register_entity(e_subscriber)

        # Create company feature views
        print(f"Creating {NUM_COMPANY_FVS} company FVs...")
        company_fvs = []
        for i in range(NUM_COMPANY_FVS):
            table_name = f"COMPANY_FV_DATA_{i}_{uuid4().hex.upper()[:8]}"
            table_path = self._create_large_table(
                fs=fs,
                table_name=table_name,
                num_rows=FV_ROWS,
                join_key="company_id",
                num_unique_keys=NUM_COMPANIES,
                num_features=NUM_FEATURES_PER_FV,
            )

            # Build SELECT clause for all feature columns
            feature_cols = ", ".join([f"feature_{j} AS company_fv{i}_feature_{j}" for j in range(NUM_FEATURES_PER_FV)])

            fv = FeatureView(
                name=f"company_fv_{i}",
                entities=[e_company],
                feature_df=self._session.sql(
                    f"""
                    SELECT
                        company_id,
                        ts,
                        {feature_cols}
                    FROM {table_path}
                """
                ),
                timestamp_col="ts",
                refresh_freq="DOWNSTREAM",
            )
            fv = fs.register_feature_view(feature_view=fv, version="v1")
            company_fvs.append(fv)

        # Create subscriber feature views
        print(f"Creating {NUM_SUBSCRIBER_FVS} subscriber FVs...")
        subscriber_fvs = []
        for i in range(NUM_SUBSCRIBER_FVS):
            table_name = f"SUBSCRIBER_FV_DATA_{i}_{uuid4().hex.upper()[:8]}"
            table_path = self._create_large_table(
                fs=fs,
                table_name=table_name,
                num_rows=FV_ROWS,
                join_key="subscriber_id",
                num_unique_keys=NUM_SUBSCRIBERS,
                num_features=NUM_FEATURES_PER_FV,
            )

            # Build SELECT clause for all feature columns
            feature_cols = ", ".join(
                [f"feature_{j} AS subscriber_fv{i}_feature_{j}" for j in range(NUM_FEATURES_PER_FV)]
            )

            fv = FeatureView(
                name=f"subscriber_fv_{i}",
                entities=[e_subscriber],
                feature_df=self._session.sql(
                    f"""
                    SELECT
                        subscriber_id,
                        ts,
                        {feature_cols}
                    FROM {table_path}
                """
                ),
                timestamp_col="ts",
                refresh_freq="DOWNSTREAM",
            )
            fv = fs.register_feature_view(feature_view=fv, version="v1")
            subscriber_fvs.append(fv)

        # Create spine with DUPLICATES
        print(f"Creating spine with {SPINE_ROWS:,} rows...")
        spine_table_name = f"SPINE_DATA_{uuid4().hex.upper()[:8]}"
        spine_table_path = f"{fs._config.full_schema_path}.{spine_table_name}"
        self._test_tables.append(spine_table_path)

        start_time = time.time()
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {spine_table_path} (
                company_id INT,
                subscriber_id INT,
                ts INT
            )
        """
        ).collect()

        # Generate spine with duplicates: multiple subscribers per company at same timestamp
        self._session.sql(
            f"""
            INSERT INTO {spine_table_path}
            SELECT
                UNIFORM(1, {NUM_COMPANIES}, RANDOM()) AS company_id,
                UNIFORM(1, {NUM_SUBSCRIBERS}, RANDOM()) AS subscriber_id,
                UNIFORM({MIN_TIMESTAMP}, {MAX_TIMESTAMP}, RANDOM()) AS ts
            FROM TABLE(GENERATOR(ROWCOUNT => {SPINE_ROWS}))
        """
        ).collect()

        elapsed = time.time() - start_time

        # Check for duplicates in spine
        duplicate_check = self._session.sql(
            f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT company_id, ts) as unique_company_ts,
                COUNT(*) - COUNT(DISTINCT company_id, ts) as duplicate_company_ts
            FROM {spine_table_path}
        """
        ).collect()[0]

        dup_factor = duplicate_check["TOTAL_ROWS"] / max(duplicate_check["UNIQUE_COMPANY_TS"], 1)
        print(f"Spine created: {duplicate_check['DUPLICATE_COMPANY_TS']:,} duplicates ({dup_factor:.2f}x factor)")

        spine_df = self._session.table(spine_table_path)
        all_fvs = company_fvs + subscriber_fvs

        # Test both CTE and Sequential methods
        results = {}

        for join_method in ["cte", "sequential"]:
            print(f"\nTesting {join_method.upper()} method...")

            dataset_name = f"PERF_TEST_DATASET_{join_method.upper()}_{uuid4().hex.upper()}"

            start_time = time.time()
            result_df = fs.generate_dataset(
                name=dataset_name,
                spine_df=spine_df,
                features=all_fvs,
                spine_timestamp_col="ts",
                output_type="table",
                join_method=join_method,
            )
            elapsed = time.time() - start_time

            # Get the query ID from the last query executed
            query_id = None
            try:
                query_history = self._session.sql("SELECT LAST_QUERY_ID()").collect()
                if query_history:
                    query_id = query_history[0][0]
            except Exception:
                pass

            # Verify results
            result_count = result_df.count()

            # Verify no join explosion occurred
            self.assertEqual(
                result_count,
                SPINE_ROWS,
                f"Dataset should have {SPINE_ROWS:,} rows (same as spine), "
                f"but got {result_count:,}. Join explosion detected in {join_method} method!",
            )

            # Verify all expected columns are present
            result_columns = set(result_df.columns)
            expected_columns = {"COMPANY_ID", "SUBSCRIBER_ID", "TS"}

            # Add company feature columns (NUM_FEATURES_PER_FV features per FV)
            for i in range(NUM_COMPANY_FVS):
                for j in range(NUM_FEATURES_PER_FV):
                    expected_columns.add(f"COMPANY_FV{i}_FEATURE_{j}")

            # Add subscriber feature columns (NUM_FEATURES_PER_FV features per FV)
            for i in range(NUM_SUBSCRIBER_FVS):
                for j in range(NUM_FEATURES_PER_FV):
                    expected_columns.add(f"SUBSCRIBER_FV{i}_FEATURE_{j}")

            missing_columns = expected_columns - result_columns
            self.assertEqual(
                len(missing_columns), 0, f"Missing expected columns in {join_method} method: {missing_columns}"
            )

            throughput = SPINE_ROWS / elapsed
            print(
                f"  Result: {result_count:,} rows, {elapsed:.2f}s, {throughput:,.0f} rows/s"
                + (f", query_id={query_id}" if query_id else "")
            )

            # Store results for comparison
            results[join_method] = {
                "time": elapsed,
                "throughput": throughput,
                "row_count": result_count,
                "query_id": query_id,
            }

        # Print comparison
        cte_time = results["cte"]["time"]
        seq_time = results["sequential"]["time"]
        speedup = seq_time / cte_time if cte_time > 0 else 0

        print(
            f"\nPerformance: CTE={cte_time:.2f}s, Sequential={seq_time:.2f}s, "
            + f"CTE is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        )


if __name__ == "__main__":
    absltest.main()
