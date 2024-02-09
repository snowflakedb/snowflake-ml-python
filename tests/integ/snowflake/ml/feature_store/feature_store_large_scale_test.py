import time
from typing import Optional, cast
from uuid import uuid4

from absl.testing import absltest
from common_utils import (
    FS_INTEG_TEST_DATASET_SCHEMA,
    FS_INTEG_TEST_DB,
    FS_INTEG_TEST_WINE_QUALITY_DATA,
    FS_INTEG_TEST_YELLOW_TRIP_DATA,
    cleanup_temporary_objects,
    create_random_schema,
    get_test_warehouse_name,
)
from pandas.testing import assert_frame_equal

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    Entity,
    FeatureStore,
    FeatureView,
    FeatureViewSlice,
)
from snowflake.ml.feature_store._internal.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session, functions as F


class FeatureStoreLargeScaleTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        cleanup_temporary_objects(self._session)
        self._active_feature_store = []
        self._test_warehouse_name = get_test_warehouse_name(self._session)

    @classmethod
    def tearDownClass(self) -> None:
        for fs in self._active_feature_store:
            fs.clear()
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
        self._session.close()

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        current_schema = create_random_schema(self._session, "FS_LARGE_SCALE_TEST") if name is None else name
        fs = FeatureStore(
            self._session,
            FS_INTEG_TEST_DB,
            current_schema,
            self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        return fs

    def test_cron_scheduling(self) -> None:
        fs = self._create_feature_store()

        wine_data = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.{FS_INTEG_TEST_WINE_QUALITY_DATA}"
        cloned_wine_data = f"wine_quality_data_{uuid4().hex.upper()}"
        self._session.sql(f"CREATE TABLE {cloned_wine_data} CLONE {wine_data}").collect(block=True)

        entity = Entity(name="wine", join_keys=["wine_id"])
        fs.register_entity(entity)

        def addIdColumn(df: DataFrame, id_column_name: str) -> DataFrame:
            # Add id column to dataframe
            columns = df.columns
            new_df = df.withColumn(id_column_name, F.monotonically_increasing_id())
            return cast(DataFrame, new_df[[id_column_name] + columns])

        source_df = self._session.table(cloned_wine_data)
        feature_df = addIdColumn(source_df, "wine_id")
        fv = FeatureView(
            name="wine_features",
            entities=[entity],
            feature_df=feature_df,
            refresh_freq="* * * * * America/Los_Angeles",
            desc="wine features",
        )
        fv = fs.register_feature_view(feature_view=fv, version="v1", block=True)
        self.assertEqual(fv.refresh_freq, "DOWNSTREAM")
        self.assertEqual(len(fs.read_feature_view(fv).collect()), 1599)

        # Insert synthetic data into the table
        temp_session = Session.builder.configs(SnowflakeLoginOptions()).create()
        generator = SyntheticDataGenerator(
            temp_session, self._session.get_current_database(), self._session.get_current_schema(), cloned_wine_data
        )
        generator.trigger(batch_size=10, num_batches=10, freq=1)

        # wait for 90s so feature view will be refreshed at least once
        time.sleep(90)
        self.assertEqual(len(fs.read_feature_view(fv).collect()), 1699)

        self._session.sql(f"DROP TABLE {cloned_wine_data}").collect()

    def test_external_table(self) -> None:
        current_schema = create_random_schema(self._session, "TEST_EXTERNAL_TABLE")
        fs = self._create_feature_store(current_schema)

        e_loc = Entity("LOCATION", ["PULOCATIONID"])
        fs.register_entity(e_loc)

        raw_dataset = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.{FS_INTEG_TEST_YELLOW_TRIP_DATA}"

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

        location_features = fs.register_feature_view(
            feature_view=location_features,
            version="V1",
            block=True,
        )

        def create_select_query(start: str, end: str) -> str:
            return f"""SELECT DISTINCT DATE_TRUNC('second', TO_TIMESTAMP(TPEP_DROPOFF_DATETIME)) AS DROPOFF_TIME,
                    PULOCATIONID, TIP_AMOUNT, TOTAL_AMOUNT
                FROM {raw_dataset}
                WHERE DROPOFF_TIME >= '{start}' AND DROPOFF_TIME < '{end}'
            """

        spine_df_1 = self._session.sql(create_select_query("2016-01-01 00:00:00", "2016-01-03 00:00:00"))

        result_table_name = f"FS_INTEG_TEST_{uuid4().hex.upper()}"

        fv_slice = location_features.slice(["F_AVG_TIP", "F_AVG_TOTAL_AMOUNT"])
        ds0 = fs.generate_dataset(
            spine_df=spine_df_1,
            features=[fv_slice],
            materialized_table=result_table_name,
            spine_timestamp_col="DROPOFF_TIME",
            spine_label_cols=None,
            save_mode="merge",
        )

        # verify dataset metadata is correct
        self.assertEqual(ds0.materialized_table, f"{FS_INTEG_TEST_DB}.{current_schema}.{result_table_name}")
        self.assertIsNotNone(ds0.feature_store_metadata)
        self.assertEqual(len(ds0.feature_store_metadata.features), 1)
        deserialized_fv_slice = FeatureViewSlice.from_json(ds0.feature_store_metadata.features[0], self._session)
        # verify snapshot 0 rows count equal to spine df rows count
        df1_row_count = len(spine_df_1.collect())
        self.assertEqual(self._session.sql(f"SELECT COUNT(*) FROM {ds0.snapshot_table}").collect()[0][0], df1_row_count)
        self.assertEqual(len(ds0.df.collect()), df1_row_count)

        self.assertEqual(deserialized_fv_slice, fv_slice)
        self.assertIsNone(ds0.label_cols)
        self.assertDictEqual(
            ds0.feature_store_metadata.connection_params,
            {
                "database": SqlIdentifier(FS_INTEG_TEST_DB).identifier(),
                "schema": SqlIdentifier(current_schema).identifier(),
            },
        )

        # verify materialized table value is correct
        actual_pdf = (
            self._session.sql(f"SELECT PULOCATIONID, F_AVG_TIP, F_AVG_TOTAL_AMOUNT FROM {ds0.materialized_table}")
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
        assert_frame_equal(expected_pdf, actual_pdf, check_dtype=True)

        # generate another dataset and merge with original materialized table
        spine_df_2 = self._session.sql(create_select_query("2016-01-04 00:00:00", "2016-01-05 00:00:00"))
        ds1 = fs.generate_dataset(
            spine_df=spine_df_2,
            features=[fv_slice],
            materialized_table=result_table_name,
            spine_timestamp_col="DROPOFF_TIME",
            spine_label_cols=None,
            save_mode="merge",
        )

        df2_row_count = len(spine_df_2.collect())
        # verify snapshot 1 rows count equal to 2x of spine df rows count (as it appends same amount of rows)
        self.assertEqual(
            self._session.sql(f"SELECT COUNT(*) FROM {ds1.snapshot_table}").collect()[0][0],
            df1_row_count + df2_row_count,
        )
        self.assertEqual(len(ds1.df.collect()), df1_row_count + df2_row_count)
        # verify snapshort 0 is not impacted after new data is merged with materialized table.
        self.assertEqual(self._session.sql(f"SELECT COUNT(*) FROM {ds0.snapshot_table}").collect()[0][0], df1_row_count)
        self.assertEqual(len(ds0.df.collect()), df1_row_count)


if __name__ == "__main__":
    absltest.main()
