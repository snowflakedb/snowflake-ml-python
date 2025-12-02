import time
from typing import Literal, Optional, cast
from uuid import uuid4

from absl.testing import absltest, parameterized
from common_utils import FS_INTEG_TEST_DATASET_SCHEMA, create_random_schema
from fs_dataset_provisioner import ensure_canonical_datasets
from fs_integ_test_base import FeatureStoreIntegTestBase
from pandas.testing import assert_frame_equal

from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
)
from snowflake.ml.feature_store._internal.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from snowflake.snowpark import DataFrame, functions as F


class FeatureStoreLargeScaleTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._active_feature_store = []

    def tearDown(self) -> None:
        for fs in self._active_feature_store:
            fs._clear(dryrun=False)
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
        super().tearDown()

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        current_schema = (
            create_random_schema(self._session, "FS_LARGE_SCALE_TEST", database=self.test_db) if name is None else name
        )
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        return fs

    def test_cron_scheduling(self) -> None:
        fs = self._create_feature_store()

        datasets = ensure_canonical_datasets(self._session, self.test_db, FS_INTEG_TEST_DATASET_SCHEMA)
        wine_data = datasets["wine_quality"]  # type: ignore[index]
        # Copy into a temp table for mutations even if source is a view; ensure it lives in FS schema
        cloned_wine_data = f"wine_quality_data_{uuid4().hex.upper()}"
        cloned_wine_data_fqn = f"{fs._config.full_schema_path}.{cloned_wine_data}"
        self._session.sql(f"CREATE OR REPLACE TABLE {cloned_wine_data_fqn} AS SELECT * FROM {wine_data}").collect(
            block=True
        )

        entity = Entity(name="wine", join_keys=["wine_id"])
        fs.register_entity(entity)

        def addIdColumn(df: DataFrame, id_column_name: str) -> DataFrame:
            # Add id column to dataframe
            columns = df.columns
            new_df = df.withColumn(id_column_name, F.monotonically_increasing_id())
            return cast(DataFrame, new_df[[id_column_name] + columns])

        source_df = self._session.table(cloned_wine_data_fqn)
        feature_df = addIdColumn(source_df, "wine_id")
        fv = FeatureView(
            name="wine_features",
            entities=[entity],
            feature_df=feature_df,
            refresh_freq="* * * * * America/Los_Angeles",
            desc="wine features",
        )
        # Ensure dynamic table resolves source in FS schema
        self._session.sql(f"USE SCHEMA {fs._config.full_schema_path}").collect()
        fv = fs.register_feature_view(feature_view=fv, version="v1", block=True)
        self.assertEqual(fv.refresh_freq, "DOWNSTREAM")
        initial_count = self._session.sql(f"SELECT COUNT(*) AS C FROM {cloned_wine_data_fqn}").collect()[0]["C"]
        self.assertEqual(len(fs.read_feature_view(fv).collect()), initial_count)

        # Insert synthetic data into the table
        generator = SyntheticDataGenerator(
            self._session, self._session.get_current_database(), self._session.get_current_schema(), cloned_wine_data
        )
        generator.trigger(batch_size=10, num_batches=10, freq=1)

        # wait for 90s so feature view will be refreshed at least once
        time.sleep(90)
        self.assertEqual(len(fs.read_feature_view(fv).collect()), initial_count + 100)

    @parameterized.parameters(  # type: ignore[misc]
        {"join_method": "cte"},
        {"join_method": "sequential"},
    )
    def test_external_table(self, join_method: Literal["sequential", "cte"]) -> None:
        current_schema = create_random_schema(self._session, "TEST_EXTERNAL_TABLE", database=self.test_db)
        fs = self._create_feature_store(current_schema)

        e_loc = Entity("LOCATION", ["PULOCATIONID"])
        fs.register_entity(e_loc)

        datasets = ensure_canonical_datasets(self._session, self.test_db, FS_INTEG_TEST_DATASET_SCHEMA)
        raw_dataset = datasets["yellow_trip"]  # type: ignore[index]

        feature_df = self._session.sql(
            f"""SELECT PULOCATIONID, AVG(TIP_AMOUNT) AS F_AVG_TIP, AVG(TOTAL_AMOUNT) AS F_AVG_TOTAL_AMOUNT
                    FROM {raw_dataset}
                    GROUP BY PULOCATIONID"""
        )
        location_features = FeatureView(
            name="FV_LOCATION_FEATURES",
            entities=[e_loc],
            feature_df=feature_df,
            refresh_freq="1 minute",
            desc="location features",
        )

        location_features = fs.register_feature_view(feature_view=location_features, version="V1")

        def create_select_query(start: str, end: str) -> str:
            return f"""
            SELECT
                DATE_TRUNC('second', TO_TIMESTAMP(TO_VARCHAR(TPEP_DROPOFF_DATETIME))) AS DROPOFF_TIME,
                PULOCATIONID,
                TIP_AMOUNT,
                TOTAL_AMOUNT
            FROM {raw_dataset}
            WHERE DATE_TRUNC('second', TO_TIMESTAMP(TO_VARCHAR(TPEP_DROPOFF_DATETIME))) >= '{start}'
                AND DATE_TRUNC('second', TO_TIMESTAMP(TO_VARCHAR(TPEP_DROPOFF_DATETIME))) < '{end}'
            """

        spine_df_1 = self._session.sql(create_select_query("2016-01-01 00:00:00", "2016-01-03 00:00:00"))

        dataset_name = f"FS_INTEG_TEST_{uuid4().hex.upper()}"
        dataset_version = "test_version"

        fv_slice = location_features.slice(["F_AVG_TIP", "F_AVG_TOTAL_AMOUNT"])
        ds0 = fs.generate_dataset(
            spine_df=spine_df_1,
            features=[fv_slice],
            name=dataset_name,
            version=dataset_version,
            spine_timestamp_col="DROPOFF_TIME",
            spine_label_cols=None,
            output_type="dataset",
            join_method=join_method,
        )

        # verify dataset metadata is correct
        dsv0 = ds0.selected_version
        dsv0_meta = dsv0._get_metadata()
        self.assertEqual(
            dsv0.url(), f"snow://dataset/{self.test_db}.{current_schema}.{dataset_name}/versions/{dataset_version}/"
        )
        self.assertIsNotNone(dsv0_meta.properties)
        self.assertEqual(len(dsv0_meta.properties.compact_feature_views), 1)
        deserialized_fv_slice = FeatureView._load_from_compact_repr(
            self._session,
            dsv0_meta.properties.compact_feature_views[0],
        )
        # verify dataset rows count equal to spine df rows count
        df1_row_count = len(spine_df_1.collect())
        ds0_df = ds0.read.to_snowpark_dataframe()
        self.assertEqual(len(ds0_df.collect()), df1_row_count)

        self.assertEqual(deserialized_fv_slice, fv_slice)
        self.assertIsNone(dsv0_meta.label_cols)

        # verify materialized table value is correct
        actual_pdf = (
            ds0_df.select(["PULOCATIONID", "F_AVG_TIP", "F_AVG_TOTAL_AMOUNT"])
            .to_pandas()
            .sort_values(by="PULOCATIONID")
            .reset_index(drop=True)
        )
        expected_pdf = (
            spine_df_1.join(feature_df, on="PULOCATIONID", how="left")
            .select(["PULOCATIONID", "F_AVG_TIP", "F_AVG_TOTAL_AMOUNT"])
            .to_pandas()
            .sort_values(by="PULOCATIONID")
            .reset_index(drop=True)
        )
        assert_frame_equal(expected_pdf, actual_pdf, check_dtype=False)


if __name__ == "__main__":
    absltest.main()
