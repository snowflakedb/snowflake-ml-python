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

import datetime
import decimal
import json
import logging
import math
import os
import time
import uuid
from typing import Optional

from absl.testing import absltest
from common_utils import FS_INTEG_TEST_DATASET_SCHEMA
from feature_store_streaming_fv_integ_base import (
    StreamingFeatureViewIntegTestBase,
    identity_transform,
)

from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StorageConfig,
    StorageFormat,
    StoreType,
)
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.snowpark import functions as snowpark_functions


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
        self._iceberg_external_volumes: list[str] = []
        self._iceberg_tables: list[str] = []
        self._test_tables: list[str] = []
        self._udf_functions: list[str] = []
        self._udf_stages: list[str] = []

    def tearDown(self) -> None:
        # Delete tracked FVs first so Dynamic Iceberg Tables release the external
        # volume (same order as FeatureStoreTest.tearDown). The bundle schema is
        # shared and must not be dropped.
        super().tearDown()
        if os.environ.get("SKIP_FV_TEARDOWN"):
            return
        for table in getattr(self, "_test_tables", []):
            self._session.sql(f"DROP TABLE IF EXISTS {table}").collect()
        for table in getattr(self, "_iceberg_tables", []):
            self._session.sql(f"DROP ICEBERG TABLE IF EXISTS {table}").collect()
        for volume in getattr(self, "_iceberg_external_volumes", []):
            self._evm.drop_external_volume(volume, if_exists=True)
        for function_sig in getattr(self, "_udf_functions", []):
            self._session.sql(f"DROP FUNCTION IF EXISTS {function_sig}").collect()
        for stage in getattr(self, "_udf_stages", []):
            self._session.sql(f"DROP STAGE IF EXISTS {stage}").collect()

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

    def _create_batch_source_table(
        self,
        fs: FeatureStore,
        suffix: str,
        entity_key: str,
        amount: float,
        *,
        event_time_type: str = "TIMESTAMP_NTZ",
    ) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_ONLINE_SRC_{suffix}"
        self._test_tables.append(table_name)
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME {event_time_type},
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
        self._test_tables.append(table_name)
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
        # Anchor to UTC day boundaries so tiles align with the online store's UTC window.
        utc_now_ntz = "CONVERT_TIMEZONE('UTC', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ"
        utc_yesterday = f"DATEADD('day', -1, DATE_TRUNC('day', {utc_now_ntz}))"
        self._session.sql(
            f"""
            INSERT INTO {table_name}
            SELECT column1, column2, column3
            FROM VALUES
                ({entity_key!r}, DATEADD('hour', 1, {utc_yesterday}), {amounts[0]}),
                ({entity_key!r}, DATEADD('hour', 2, {utc_yesterday}), {amounts[1]}),
                ({entity_key!r}, DATEADD('hour', 3, {utc_yesterday}), {amounts[2]})
        """
        ).collect()
        return table_name, sum(amounts), len(amounts)

    # =========================================================================
    # E2E: Batch non-tiled (timeseries) — registration -> online read
    # =========================================================================

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

    def _create_iceberg_storage_config(self) -> StorageConfig:
        """Create a unique AWS Iceberg external volume + StorageConfig for this test."""
        volume_name = f"MLPLATFORMTEST_ICEBERG_AWS_S3_{uuid.uuid4().hex[:8].upper()}"
        storage_location_sql = """
                (
                    NAME                 = 'prod-iceberg-s3'
                    STORAGE_PROVIDER     = 'S3'
                    STORAGE_BASE_URL     = 's3://mlplatform-iceberg-test/ml-platform/'
                    STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::736112632310:role/MLPlatformTestIcebergRole'
                    STORAGE_AWS_EXTERNAL_ID = 'MLPLATFORMTEST_SFCRole=MLPlatformExternalVolume='
                )
            """
        self._evm.create_external_volume(volume_name, storage_location_sql)
        self._iceberg_external_volumes.append(volume_name)
        return StorageConfig(
            format=StorageFormat.ICEBERG,
            external_volume=volume_name,
            base_location=f"test_{uuid.uuid4().hex}/",
        )

    def _assert_storage_format(self, fs: FeatureStore, table_name: str, *, expect_iceberg: bool) -> None:
        """Assert whether ``table_name`` (schema-local identifier) is an Iceberg table.

        Args:
            fs: Feature store whose schema is searched.
            table_name: Unqualified table identifier as stored in Snowflake.
            expect_iceberg: Whether the table must appear in ``SHOW ICEBERG TABLES``.
        """
        rows = self._session.sql(f"SHOW ICEBERG TABLES IN SCHEMA {fs._config.full_schema_path}").collect()
        iceberg_names = {str(row["name"]) for row in rows}
        is_iceberg = table_name in iceberg_names
        self.assertEqual(
            is_iceberg,
            expect_iceberg,
            f"{table_name} iceberg={is_iceberg}, expected={expect_iceberg}; iceberg tables={sorted(iceberg_names)}",
        )

    @absltest.skip("Iceberg feature views are not supported with online storage enabled.")  # type: ignore[misc]
    def test_iceberg_batch_fv_spec_oft_online_read_by_key(self) -> None:
        """Dynamic Iceberg Table batch FV with Postgres OFT: register, wait, online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"ICEBERG_BATCH_OFT_{s}"
        batch_key = f"U_ICEBERG_{s}"
        expected_amount = 777.0

        # Iceberg TIMESTAMP_NTZ max scale is 6; Snowflake default TIMESTAMP_NTZ is (9).
        src_table = self._create_batch_source_table(
            fs, s, batch_key, expected_amount, event_time_type="TIMESTAMP_NTZ(6)"
        )
        feature_df = self._session.table(src_table)
        iceberg_storage_config = self._create_iceberg_storage_config()

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            storage_config=iceberg_storage_config,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertFalse(registered.is_streaming)
        self.assertTrue(registered.online)
        assert registered.online_config is not None
        self.assertEqual(registered.online_config.store_type, OnlineStoreType.POSTGRES)
        assert registered.storage_config is not None
        self.assertEqual(registered.storage_config.format, StorageFormat.ICEBERG)

        online_name = registered.fully_qualified_online_table_name()
        self.assertIsNotNone(online_name)
        self.assertIn("$ONLINE", online_name)

        self._wait_offline_dt_rows(fs, fv_name, "v1")
        self._assert_amount_round_trip(fs, fv_name, "v1", src_table, batch_key, expected_amount)

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate, desc="iceberg batch postgres oft"
        )

        # Reverse-ETL: insert into the source and let the pipeline propagate on its own lag
        # (Dynamic Iceberg Table at refresh_freq, then Postgres OFT at target_lag). Nothing is
        # refreshed by hand, so this exercises the automatic source -> DIT -> Postgres path.
        new_key = f"U_ICEBERG_NEW_{s}"
        new_amount = 888.0
        self._session.sql(
            f"""
            INSERT INTO {src_table} VALUES
            ({new_key!r}, DATEADD('minute', -1, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ(6)), {new_amount})
            """
        ).collect()

        def _validate_new(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), new_amount)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[new_key]],
            validate_fn=_validate_new,
            desc="iceberg dit reverse-etl postgres oft",
        )
        # Online served the new row, so the DIT behind it is materialized; confirm offline agrees.
        self._assert_amount_round_trip(fs, fv_name, "v1", src_table, new_key, new_amount)

    def _create_iceberg_table_from_snowflake_table(self, fs: FeatureStore, snowflake_table: str, suffix: str) -> str:
        """Create a Snowflake-managed Iceberg table (not a Dynamic Iceberg Table) from a Snowflake table."""
        iceberg_storage_config = self._create_iceberg_storage_config()
        iceberg_table = f"{self.test_db}.{fs._config.schema.identifier()}.ICEBERG_FROM_SF_{suffix}"
        self._session.sql(
            f"""
            CREATE ICEBERG TABLE {iceberg_table}
                CATALOG = 'SNOWFLAKE'
                EXTERNAL_VOLUME = {iceberg_storage_config.external_volume}
                BASE_LOCATION = '{iceberg_storage_config.base_location}'
            AS SELECT * FROM {snowflake_table}
            """
        ).collect()
        self._iceberg_tables.append(iceberg_table)
        return iceberg_table

    def _assert_amount_round_trip(
        self,
        fs: FeatureStore,
        fv_name: str,
        version: str,
        source_table: str,
        batch_key: str,
        expected_amount: float,
    ) -> None:
        """Assert the inserted AMOUNT is unchanged on the source table and on the offline FV read.

        Args:
            fs: Feature store client.
            fv_name: Feature view name.
            version: Feature view version.
            source_table: Fully qualified table the FV is registered over.
            batch_key: Entity key of the inserted row.
            expected_amount: Amount written into ``source_table``.
        """
        source_rows = self._session.sql(f"SELECT AMOUNT FROM {source_table} WHERE USER_ID = {batch_key!r}").collect()
        self.assertEqual(len(source_rows), 1, f"expected one inserted row in {source_table}")
        inserted_amount = float(source_rows[0]["AMOUNT"])
        self.assertEqual(inserted_amount, expected_amount)

        fv_live = fs.get_feature_view(fv_name, version)
        offline_pdf = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE).to_pandas()
        self.assertGreater(len(offline_pdf), 0)
        self.assertIn("AMOUNT", offline_pdf.columns)
        if "USER_ID" in offline_pdf.columns:
            offline_pdf = offline_pdf[offline_pdf["USER_ID"].astype(str) == batch_key]
            self.assertGreater(len(offline_pdf), 0, f"no offline row for key {batch_key!r}")
        self.assertEqual(float(offline_pdf.iloc[0]["AMOUNT"]), inserted_amount)

    def _write_snowflake_table_via_udf(
        self,
        fs: FeatureStore,
        source_table: str,
        suffix: str,
    ) -> str:
        """Write a Snowflake table by projecting a permanent scalar UDF over ``source_table``.

        Args:
            fs: Feature store used for schema placement.
            source_table: Fully qualified source table with USER_ID, EVENT_TIME, AMOUNT.
            suffix: Unique suffix for UDF, stage, and destination names.

        Returns:
            Fully qualified destination table name.
        """
        schema_path = fs._config.full_schema_path
        stage_name = f"{schema_path}.UDF_WRITE_STAGE_{suffix}"
        udf_name = f"{schema_path}.TIMES_TWO_{suffix}"
        dest_table = f"{self.test_db}.{fs._config.schema.identifier()}.UDF_WRITE_SRC_{suffix}"
        self._test_tables.append(dest_table)

        self._session.sql(f"CREATE OR REPLACE STAGE {stage_name}").collect()
        self._udf_stages.append(stage_name)

        @snowpark_functions.udf(  # type: ignore[misc, arg-type]
            name=udf_name,
            session=self._session,
            is_permanent=True,
            stage_location=f"@{stage_name}",
            replace=True,
        )
        def times_two(x: float) -> float:
            return x * 2.0

        self._udf_functions.append(f"{udf_name}(DOUBLE)")

        self._session.table(source_table).select(
            snowpark_functions.col("USER_ID"),
            snowpark_functions.col("EVENT_TIME"),
            snowpark_functions.call_udf(udf_name, snowpark_functions.col("AMOUNT")).alias("AMOUNT"),
        ).write.save_as_table(dest_table, mode="overwrite")
        return dest_table

    def test_udf_write_snowflake_table_batch_fv_spec_oft_online_read_by_key(self) -> None:
        """Batch FV over a Snowflake table written via UDF, with Postgres OFT online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"UDF_WRITE_BATCH_OFT_{s}"
        batch_key = f"U_UDF_WRITE_{s}"
        source_amount = 333.0
        expected_amount = source_amount * 2.0

        src_table = self._create_batch_source_table(fs, s, batch_key, source_amount)
        dest_table = self._write_snowflake_table_via_udf(fs, src_table, s)
        feature_df = self._session.table(dest_table)

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
        assert registered.online_config is not None
        self.assertEqual(registered.online_config.store_type, OnlineStoreType.POSTGRES)

        online_name = registered.fully_qualified_online_table_name()
        self.assertIsNotNone(online_name)
        self.assertIn("$ONLINE", online_name)

        self._wait_offline_dt_rows(fs, fv_name, "v1")
        self._assert_amount_round_trip(fs, fv_name, "v1", dest_table, batch_key, expected_amount)

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[batch_key]],
            validate_fn=_validate,
            desc="udf write snowflake table postgres oft",
        )

    def _register_passthrough_streaming_fv(
        self,
        fs: FeatureStore,
        fv_name: str,
        suffix: str,
        source_table: str,
        *,
        storage_config: Optional[StorageConfig] = None,
    ) -> FeatureView:
        """Register a passthrough SFV whose backfill is ``source_table`` and wait for UDF backfill.

        Args:
            fs: Feature store client.
            fv_name: Feature view name.
            suffix: Unique suffix for the stream source name.
            source_table: Fully qualified backfill table (Iceberg or Snowflake).
            storage_config: Optional Iceberg (or other) storage config for the registered FV.

        Returns:
            The registered streaming feature view.
        """
        stream = f"TXN_{suffix}"
        self._make_stream_source(fs, stream)
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            stream_config=StreamConfig(
                stream_source=stream,
                transformation_fn=identity_transform,
                backfill_df=self._session.table(source_table),
            ),
            timestamp_col="EVENT_TIME",
            refresh_freq="1 minute",
            storage_config=storage_config,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_streaming)
        self.assertTrue(registered.online)
        assert registered.online_config is not None
        self.assertEqual(registered.online_config.store_type, OnlineStoreType.POSTGRES)

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        self._wait_udf_and_backfill(
            fq_udf,
            feature_store=fs,
            streaming_fv_metadata_name=str(registered.name),
            streaming_fv_version=str(registered.version),
        )
        self._wait_offline_dt_rows(fs, fv_name, "v1")
        return registered

    def test_iceberg_table_from_snowflake_streaming_fv_spec_oft_online_read_by_key(self) -> None:
        """SFV backfill from a Snowflake-managed Iceberg table (CTAS, not DIT); Postgres OFT."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"ICEBERG_TBL_STREAM_OFT_{s}"
        batch_key = f"U_ICEBERG_SFV_{s}"
        expected_amount = 555.0

        src_table = self._create_batch_source_table(
            fs, s, batch_key, expected_amount, event_time_type="TIMESTAMP_NTZ(6)"
        )
        iceberg_table = self._create_iceberg_table_from_snowflake_table(fs, src_table, s)
        self._register_passthrough_streaming_fv(fs, fv_name, s, iceberg_table)
        self._assert_amount_round_trip(fs, fv_name, "v1", iceberg_table, batch_key, expected_amount)

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[batch_key]],
            validate_fn=_validate,
            desc="iceberg table from snowflake streaming postgres oft",
        )

    def test_udf_write_snowflake_table_streaming_fv_spec_oft_online_read_by_key(self) -> None:
        """SFV backfill from a Snowflake table written via UDF, with Postgres OFT online read."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"UDF_WRITE_STREAM_OFT_{s}"
        batch_key = f"U_UDF_WRITE_SFV_{s}"
        source_amount = 222.0
        expected_amount = source_amount * 2.0

        src_table = self._create_batch_source_table(fs, s, batch_key, source_amount)
        dest_table = self._write_snowflake_table_via_udf(fs, src_table, s)
        self._register_passthrough_streaming_fv(fs, fv_name, s, dest_table)
        self._assert_amount_round_trip(fs, fv_name, "v1", dest_table, batch_key, expected_amount)

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[batch_key]],
            validate_fn=_validate,
            desc="udf write snowflake table streaming postgres oft",
        )

    @absltest.skip("Iceberg feature views are not supported with online storage enabled.")  # type: ignore[misc]
    def test_iceberg_backfill_streaming_fv_spec_oft_online_read_by_key(self) -> None:
        """SFV whose offline object is a Dynamic Iceberg Table, through to an online read.

        ``$UDF_TRANSFORMED`` / ``$BACKFILL`` stay regular Snowflake tables with
        Iceberg-compatible timestamp scale (6). Postgres OFT hydrates from
        ``$UDF_TRANSFORMED``.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"ICEBERG_BF_STREAM_OFT_{s}"
        batch_key = f"U_ICEBERG_BF_{s}"
        expected_amount = 333.0

        # Iceberg TIMESTAMP_NTZ max scale is 6.
        src_table = self._create_batch_source_table(
            fs, s, batch_key, expected_amount, event_time_type="TIMESTAMP_NTZ(6)"
        )
        iceberg_storage_config = self._create_iceberg_storage_config()
        registered = self._register_passthrough_streaming_fv(
            fs, fv_name, s, src_table, storage_config=iceberg_storage_config
        )
        assert registered.storage_config is not None
        self.assertEqual(registered.storage_config.format, StorageFormat.ICEBERG)

        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)

        self._assert_storage_format(fs, physical_name.resolved(), expect_iceberg=True)
        self._assert_storage_format(fs, udf_table.resolved(), expect_iceberg=False)

        self._assert_amount_round_trip(fs, fv_name, "v1", src_table, batch_key, expected_amount)

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[batch_key]],
            validate_fn=_validate,
            desc="iceberg backfill streaming postgres oft",
        )

        ingested_key = f"U_ICEBERG_INGEST_{s}"
        ingested_amount = 777.0
        ingested_event_time = datetime.datetime(2024, 6, 1, 12, 0, 0)
        self._stream_ingest_with_retry(
            fs,
            f"TXN_{s}",
            {
                "USER_ID": ingested_key,
                "AMOUNT": ingested_amount,
                "EVENT_TIME": ingested_event_time,
            },
        )

        def _validate_ingested(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertEqual(float(pdf.iloc[0]["AMOUNT"]), ingested_amount)

        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[ingested_key]],
            validate_fn=_validate_ingested,
            desc="iceberg streaming ingest postgres oft",
        )

    def test_batch_fv_online_read_negotiates_http2(self) -> None:
        """Soft assertion: confirm the Online Service negotiates HTTP/2.

        Wraps ``httpx.Client.post`` for one online read, captures ``Response.http_version``
        from successful responses, and asserts ``HTTP/2`` when at least one negotiation
        succeeded. Skips (non-failing) if the server downgraded to HTTP/1.1 so the test
        does not flake against environments that have not yet enabled h2.
        """
        import httpx

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_HTTP2_FV_{s}"
        batch_key = f"U_HTTP2_{s}"
        expected_amount = 123.0

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
        fs.register_feature_view(fv, "v1")
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        # Wait until rows are visible online before instrumenting httpx, so the captured
        # negotiation reflects the real query path rather than an empty/poll response.
        self._poll_online_read(
            fs,
            fv_name,
            "v1",
            keys=[[batch_key]],
            validate_fn=lambda pdf: self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount, places=3),
            desc="batch http2 warmup",
        )

        captured: list[str] = []
        original_post = httpx.Client.post

        def _spy_post(self_client, url, *args, **kwargs):  # type: ignore[no-untyped-def]
            resp = original_post(self_client, url, *args, **kwargs)
            try:
                if 200 <= int(resp.status_code) < 300:
                    captured.append(str(getattr(resp, "http_version", "unknown")))
            except (TypeError, ValueError):
                logging.debug("batch online read http2 spy could not parse status_code", exc_info=True)
            return resp

        httpx.Client.post = _spy_post  # type: ignore[method-assign]
        try:
            fv_live = fs.get_feature_view(fv_name, "v1")
            # Retry inside the patched window so a transient 404 doesn't fail the measurement.
            pdf = self._read_online_with_retry(fs, fv_live, keys=[[batch_key]])
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount, places=3)
        finally:
            httpx.Client.post = original_post  # type: ignore[method-assign]

        self.assertGreater(len(captured), 0, "httpx.Client.post was never invoked during the online read.")
        if all(v != "HTTP/2" for v in captured):
            self.skipTest(
                f"Online Service did not negotiate HTTP/2 in this environment; captured versions={captured!r}. "
                "This is a soft assertion: the server is reachable but negotiated HTTP/1.1."
            )
        self.assertIn("HTTP/2", captured)

    # =========================================================================
    # E2E: Batch tiled (timeseries) — registration -> online read
    # =========================================================================

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
    # E2E: Batch tiled approx_count_distinct — registration -> online read
    # =========================================================================

    def test_batch_tiled_approx_count_distinct_online_read(self) -> None:
        """Tiled batch FV with approx_count_distinct on a STRING column: online read returns HLL estimate."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_HLL_FV_{s}"
        key_a = f"U_HLL_A_{s}"
        key_b = f"U_HLL_B_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_HLL_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                CATEGORY VARCHAR
            )
        """
        ).collect()
        # Anchor to UTC day boundaries so tiles align with the online store's UTC window.
        yesterday = "DATEADD('day', -1, DATE_TRUNC('day', CONVERT_TIMEZONE('UTC', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ))"
        self._session.sql(
            f"""
            INSERT INTO {table_name}
            SELECT column1, column2, column3 FROM VALUES
                ({key_a!r}, DATEADD('hour', 1, {yesterday}), 'electronics'),
                ({key_a!r}, DATEADD('hour', 2, {yesterday}), 'books'),
                ({key_a!r}, DATEADD('hour', 3, {yesterday}), 'electronics'),
                ({key_b!r}, DATEADD('hour', 1, {yesterday}), 'toys')
        """
        ).collect()

        features = [Feature.approx_count_distinct("CATEGORY", "2d").alias("UNIQUE_CATS_2D")]
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.table(table_name),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_tiled)
        self.assertTrue(registered.online)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        # key_a has categories {electronics, books} within the window -> exactly 2
        # distinct. The online store returns a rounded integer estimate; assert an
        # exact integer value (no float coercion, no tolerance) for parity.
        def _validate(pdf):
            self.assertIn("UNIQUE_CATS_2D", pdf.columns)
            self.assert_long_feature(
                pdf.iloc[0]["UNIQUE_CATS_2D"], expected=2, msg="batch online approx_count_distinct"
            )

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[key_a]], validate_fn=_validate, desc="batch tiled approx_count_distinct"
        )

    # =========================================================================
    # E2E: Batch tiled (timeseries) + secondary key — registration -> online read
    # =========================================================================

    def test_batch_tiled_fv_spec_oft_secondary_key_online_read_by_key(self) -> None:
        """Tiled batch FV with secondary keys: per-key SUM/COUNT arrays compose across multiple tiles in the window."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_TILED_SK_FV_{s}"
        batch_key = f"U_BATCH_SK_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_TILED_SK_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AD_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()

        # Anchor to UTC day boundaries so tiles align with the online store's UTC window.
        utc_now_ntz = "CONVERT_TIMEZONE('UTC', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ"
        yesterday_midnight = f"DATEADD('day', -1, DATE_TRUNC('day', {utc_now_ntz}))"
        two_days_ago_midnight = f"DATEADD('day', -2, DATE_TRUNC('day', {utc_now_ntz}))"
        self._session.sql(
            f"""
            INSERT INTO {table_name}
            SELECT column1, column2, column3, column4
            FROM VALUES
                ({batch_key!r}, DATEADD('hour', 1, {two_days_ago_midnight}), 'ad_a', 10.0),
                ({batch_key!r}, DATEADD('hour', 2, {two_days_ago_midnight}), 'ad_b', 50.0),
                ({batch_key!r}, DATEADD('hour', 1, {yesterday_midnight}), 'ad_a', 20.0),
                ({batch_key!r}, DATEADD('hour', 2, {yesterday_midnight}), 'ad_c', 70.0)
        """
        ).collect()
        feature_df = self._session.table(table_name)

        features = [
            Feature.sum("AMOUNT", "3d").alias("AMOUNT_SUM_3D"),
            Feature.count("AMOUNT", "3d").alias("TXN_COUNT_3D"),
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
            aggregation_secondary_keys=["AD_ID"],
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertFalse(registered.is_streaming)
        self.assertTrue(registered.is_tiled)
        self.assertTrue(registered.online)
        self.assertEqual(registered.aggregation_secondary_keys, ["AD_ID"])

        fv_live = fs.get_feature_view(fv.name, "v1")
        self.assertEqual(fv_live.name, fv.name)
        self.assertFalse(fv_live.is_streaming)
        self.assertTrue(fv_live.is_tiled)
        self.assertTrue(fv_live.online)
        self.assertEqual(fv_live.aggregation_secondary_keys, ["AD_ID"])

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            self.assertIn("AD_ID_KEYS_3D", pdf.columns)
            self.assertIn("AMOUNT_SUM_3D", pdf.columns)
            self.assertIn("TXN_COUNT_3D", pdf.columns)
            row = pdf.iloc[0]
            # Keys are ARRAY_AGG(AD_ID) WITHIN GROUP (ORDER BY AD_ID) — alphabetical;
            # value arrays must be co-ordered with the key array.
            self.assertEqual(list(row["AD_ID_KEYS_3D"]), ["ad_a", "ad_b", "ad_c"])
            self.assertEqual(list(row["AMOUNT_SUM_3D"]), [30.0, 50.0, 70.0])
            self.assertEqual(list(row["TXN_COUNT_3D"]), [2, 1, 1])

        self._poll_online_read(fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate, desc="batch tiled sk")

    # =========================================================================
    # E2E: Batch non-timeseries (with refresh_freq) — registration -> online read
    # =========================================================================

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

    def test_batch_static_fv_spec_oft_enables_source_view_change_tracking(self) -> None:
        """Static batch (view-backed) POSTGRES FV enables CHANGE_TRACKING on its source view.

        A spec-backed OFT can only resolve to an incremental refresh when the object it reads from
        exposes change tracking. Static FVs are backed by a plain View (change tracking off by
        default), so without this the backend silently downgrades the OFT to a FULL refresh.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_STATIC_CT_FV_{s}"
        entity_key = f"U_STATIC_CT_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_STATIC_CT_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {table_name} VALUES ({entity_key!r}, 12.5)").collect()
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

        # Deterministic assertion: the client should have enabled change tracking on the source
        # View as part of OFT creation. This depends only on the client emitting the ALTER VIEW.
        source_view_name = registered.fully_qualified_name()
        view_rows = self._session.sql(
            f"SHOW VIEWS LIKE '%{fv_name.upper()}%' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertGreaterEqual(len(view_rows), 1)
        self.assertTrue(
            all(row["change_tracking"] == "ON" for row in view_rows),
            f"expected CHANGE_TRACKING=ON on source view {source_view_name}, got "
            f"{[(row['name'], row['change_tracking']) for row in view_rows]}",
        )

        # The OFT should not be downgraded to FULL now that the source view exposes change tracking.
        # With the default AUTO mode the backend resolves to INCREMENTAL.
        list_result = fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        online_config = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertEqual(online_config["refresh_mode"], "INCREMENTAL")

    def test_batch_static_fv_explicit_incremental_enables_change_tracking(self) -> None:
        """Static FV with explicit INCREMENTAL refresh_mode enables change tracking and OFT is INCREMENTAL."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_STATIC_INCR_FV_{s}"
        entity_key = f"U_STATIC_INCR_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_STATIC_INCR_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {table_name} VALUES ({entity_key!r}, 42.0)").collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            refresh_mode="INCREMENTAL",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)
        self.assertIsNone(registered.refresh_freq)

        view_rows = self._session.sql(
            f"SHOW VIEWS LIKE '%{fv_name.upper()}%' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertGreaterEqual(len(view_rows), 1)
        self.assertTrue(
            all(row["change_tracking"] == "ON" for row in view_rows),
            f"expected CHANGE_TRACKING=ON for INCREMENTAL refresh_mode, got "
            f"{[(row['name'], row['change_tracking']) for row in view_rows]}",
        )

        list_result = fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        online_cfg = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertEqual(online_cfg["refresh_mode"], "INCREMENTAL")

    def test_batch_static_fv_explicit_full_skips_change_tracking(self) -> None:
        """Static FV with explicit FULL refresh_mode does not enable change tracking."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_STATIC_FULL_FV_{s}"
        entity_key = f"U_STATIC_FULL_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_STATIC_FULL_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {table_name} VALUES ({entity_key!r}, 55.0)").collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            refresh_mode="FULL",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)

        view_rows = self._session.sql(
            f"SHOW VIEWS LIKE '%{fv_name.upper()}%' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertGreaterEqual(len(view_rows), 1)
        self.assertTrue(
            all(row["change_tracking"] == "OFF" for row in view_rows),
            f"expected CHANGE_TRACKING=OFF for FULL refresh_mode, got "
            f"{[(row['name'], row['change_tracking']) for row in view_rows]}",
        )

        list_result = fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        online_cfg = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertEqual(online_cfg["refresh_mode"], "FULL")

    def _run_managed_fv_refresh_mode_test(self, input_refresh_mode: str, expected_oft_refresh_mode: str) -> None:
        """Helper: register a managed (DT-backed) FV with the given refresh_mode and verify the
        OFT resolves to the expected refresh mode.

        Args:
            input_refresh_mode: The refresh_mode to set on the FeatureView.
            expected_oft_refresh_mode: The expected resolved refresh_mode on the OFT.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_MANAGED_{input_refresh_mode}_FV_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_MANAGED_{input_refresh_mode}_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {table_name} VALUES ('user1', 99.0)").collect()
        feature_df = self._session.table(table_name)

        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=feature_df,
            refresh_freq="1h",
            refresh_mode=input_refresh_mode,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)

        list_result = fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        online_cfg = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertEqual(online_cfg["refresh_mode"], expected_oft_refresh_mode)

    def test_batch_managed_fv_full_refresh_resolves_to_full(self) -> None:
        """A managed FV with FULL refresh mode resolves the OFT to FULL."""
        self._run_managed_fv_refresh_mode_test("FULL", "FULL")

    def test_batch_managed_fv_auto_refresh_resolves_to_incremental(self) -> None:
        """A managed FV with AUTO refresh mode resolves the OFT to INCREMENTAL."""
        self._run_managed_fv_refresh_mode_test("AUTO", "INCREMENTAL")

    def test_batch_managed_fv_incremental_refresh_resolves_to_incremental(self) -> None:
        """A managed FV with INCREMENTAL refresh mode resolves the OFT to INCREMENTAL."""
        self._run_managed_fv_refresh_mode_test("INCREMENTAL", "INCREMENTAL")

    def test_batch_static_fv_auto_refresh_change_tracking_failure_resolves_to_full(self) -> None:
        """Static FV with AUTO refresh_mode falls back to FULL when change tracking cannot be enabled.

        Uses a non-incrementalizable source: a view on top of another view that contains an
        aggregation (AVG). Snowflake cannot enable change tracking on a view built on an
        aggregated view, so the ALTER VIEW SET CHANGE_TRACKING = TRUE fails and the OFT
        gracefully falls back to FULL refresh.
        """
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        schema_path = f"{self.test_db}.{fs._config.schema.identifier()}"

        base_table = f"{schema_path}.CT_FAIL_BASE_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {base_table} (
                USER_ID VARCHAR,
                AMOUNT FLOAT
            )
        """
        ).collect()
        self._session.sql(f"INSERT INTO {base_table} VALUES ('u1', 10.0), ('u1', 20.0)").collect()

        agg_view = f"{schema_path}.CT_FAIL_AGG_V_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE VIEW {agg_view} AS
                SELECT USER_ID, AVG(AMOUNT) AS AVG_AMOUNT FROM {base_table} GROUP BY USER_ID
        """
        ).collect()

        outer_view = f"{schema_path}.CT_FAIL_OUTER_V_{s}"
        self._session.sql(f"CREATE OR REPLACE VIEW {outer_view} AS SELECT * FROM {agg_view}").collect()

        feature_df = self._session.table(outer_view)
        fv = FeatureView(
            name=f"BATCH_STATIC_AUTO_CT_FAIL_{s}",
            entities=[self.user_entity],
            feature_df=feature_df,
            refresh_mode="AUTO",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )

        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.online)

        fv_name = f"BATCH_STATIC_AUTO_CT_FAIL_{s}"
        list_result = fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        online_cfg = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertEqual(online_cfg["refresh_mode"], "FULL")

    # =========================================================================
    # E2E: Batch tiled NULL-handling parity — online read of all-NULL data
    # =========================================================================

    def _build_all_aggregation_features(self, *, numeric_col: str, category_col: str, window: str, n: int = 3) -> list:
        """Build one feature per aggregation the Postgres online store supports (aliases uppercase to match columns)."""
        return [
            Feature.sum(numeric_col, window).alias("F_SUM"),
            Feature.count(numeric_col, window).alias("F_COUNT"),
            Feature.avg(numeric_col, window).alias("F_AVG"),
            Feature.min(numeric_col, window).alias("F_MIN"),
            Feature.max(numeric_col, window).alias("F_MAX"),
            Feature.stddev(numeric_col, window).alias("F_STDDEV"),
            Feature.var(numeric_col, window).alias("F_VAR"),
            Feature.approx_count_distinct(category_col, window).alias("F_ACD"),
            Feature.last_n(category_col, window, n=n).alias("F_LAST_N"),
            Feature.first_n(category_col, window, n=n).alias("F_FIRST_N"),
            Feature.last_distinct_n(category_col, window, n=n).alias("F_LAST_DISTINCT_N"),
            Feature.first_distinct_n(category_col, window, n=n).alias("F_FIRST_DISTINCT_N"),
        ]

    @staticmethod
    def _online_list(value) -> list:
        """Normalize an online list-aggregation column (JSON string or array) to a Python list."""
        if isinstance(value, str):
            return json.loads(value)
        return list(value)

    def test_batch_tiled_null_handling_online_read(self) -> None:
        """Tiled online read of NULL data: an all-NULL key follows the NULL/zero contract while a key whose
        only value lives in one tile has that value drive every aggregation. One FV serves both keys."""
        import pandas as pd

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_TILED_NULL_{s}"
        key_all_null = f"U_NULL_ALL_{s}"
        key_one_value = f"U_NULL_ONE_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BATCH_NULL_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR, EVENT_TIME TIMESTAMP_NTZ, AMOUNT FLOAT, CATEGORY VARCHAR
            )
        """
        ).collect()
        # Anchor to UTC day boundaries so every 1d tile lands inside the online 4d window.
        utc_now_ntz = "CONVERT_TIMEZONE('UTC', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ"
        yesterday = f"DATEADD('day', -1, DATE_TRUNC('day', {utc_now_ntz}))"
        two_days_ago = f"DATEADD('day', -2, DATE_TRUNC('day', {utc_now_ntz}))"
        three_days_ago = f"DATEADD('day', -3, DATE_TRUNC('day', {utc_now_ntz}))"
        # key_all_null: two tiles, two all-NULL events each -> every tile NULL.
        # key_one_value: two all-NULL tiles plus one tile holding a NULL and a single real value.
        self._session.sql(
            f"""
            INSERT INTO {table_name} (USER_ID, EVENT_TIME, AMOUNT, CATEGORY) VALUES
                ({key_all_null!r}, DATEADD('hour', 1, {two_days_ago}), NULL, NULL),
                ({key_all_null!r}, DATEADD('hour', 2, {two_days_ago}), NULL, NULL),
                ({key_all_null!r}, DATEADD('hour', 1, {yesterday}), NULL, NULL),
                ({key_all_null!r}, DATEADD('hour', 2, {yesterday}), NULL, NULL),
                ({key_one_value!r}, DATEADD('hour', 1, {three_days_ago}), NULL, NULL),
                ({key_one_value!r}, DATEADD('hour', 2, {three_days_ago}), NULL, NULL),
                ({key_one_value!r}, DATEADD('hour', 1, {two_days_ago}), NULL, NULL),
                ({key_one_value!r}, DATEADD('hour', 2, {two_days_ago}), NULL, NULL),
                ({key_one_value!r}, DATEADD('hour', 1, {yesterday}), NULL, NULL),
                ({key_one_value!r}, DATEADD('hour', 2, {yesterday}), 42.0, 'cat1')
        """
        ).collect()

        features = self._build_all_aggregation_features(numeric_col="AMOUNT", category_col="CATEGORY", window="4d")
        fv = FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._session.table(table_name),
            timestamp_col="EVENT_TIME",
            refresh_mode="FULL",
            refresh_freq="1 minute",
            feature_granularity="1d",
            features=features,
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_tiled)
        self.assertTrue(registered.online)

        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate_all_null(pdf):
            row = pdf.iloc[0]
            self.assertTrue(pd.isna(row["F_SUM"]), f"F_SUM={row['F_SUM']!r}")
            self.assert_long_feature(row["F_COUNT"], expected=0, msg="count")
            self.assertTrue(pd.isna(row["F_AVG"]), f"F_AVG={row['F_AVG']!r}")
            self.assertTrue(pd.isna(row["F_MIN"]), f"F_MIN={row['F_MIN']!r}")
            self.assertTrue(pd.isna(row["F_MAX"]), f"F_MAX={row['F_MAX']!r}")
            self.assertTrue(pd.isna(row["F_STDDEV"]), f"F_STDDEV={row['F_STDDEV']!r}")
            self.assertTrue(pd.isna(row["F_VAR"]), f"F_VAR={row['F_VAR']!r}")
            # Offline is the source of truth: for an all-NULL key it returns 0 for approx_count_distinct and
            # NULL for the list aggregations, and online serving matches.
            self.assert_long_feature(row["F_ACD"], expected=0, msg="approx_count_distinct")
            for col in ("F_LAST_N", "F_FIRST_N", "F_LAST_DISTINCT_N", "F_FIRST_DISTINCT_N"):
                self.assertTrue(pd.isna(row[col]), f"{col}={row[col]!r}")

        def _validate_one_value(pdf):
            row = pdf.iloc[0]
            # The single non-NULL value drives every aggregation.
            self.assertAlmostEqual(float(row["F_SUM"]), 42.0, places=2)
            self.assert_long_feature(row["F_COUNT"], expected=1, msg="count")
            self.assertAlmostEqual(float(row["F_AVG"]), 42.0, places=2)
            self.assertAlmostEqual(float(row["F_MIN"]), 42.0, places=2)
            self.assertAlmostEqual(float(row["F_MAX"]), 42.0, places=2)
            self.assertAlmostEqual(float(row["F_STDDEV"]), 0.0, places=2)  # single value -> population std 0
            self.assertAlmostEqual(float(row["F_VAR"]), 0.0, places=2)  # single value -> population variance 0
            self.assert_long_feature(row["F_ACD"], expected=1, msg="approx_count_distinct")
            for col in ("F_LAST_N", "F_FIRST_N", "F_LAST_DISTINCT_N", "F_FIRST_DISTINCT_N"):
                self.assertEqual(self._online_list(row[col]), ["cat1"], f"{col}={row[col]!r}")

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[key_one_value]], validate_fn=_validate_one_value, desc="tiled null all-but-one"
        )
        self._poll_online_read(
            fs, fv_name, "v1", keys=[[key_all_null]], validate_fn=_validate_all_null, desc="tiled null all events"
        )

    # =========================================================================
    # E2E: Multi-entity batch FV — registration -> online read
    # =========================================================================

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

    def test_batch_fv_high_precision_decimal_online_read(self) -> None:
        """A ``NUMBER(38,37)`` value survives an online read with full precision (no float truncation)."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"BATCH_HP_DECIMAL_{s}"
        entity_key = f"U_HPDEC_{s}"

        # 1 integer digit + 37 fractional digits = 38 significant digits, i.e. the NUMBER(38,37) limit.
        high_precision = "3.1415926535897932384626433832795028841"
        expected = decimal.Decimal(high_precision)

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.HP_DECIMAL_SRC_{s}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                BALANCE NUMBER(38,37)
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ({entity_key!r}, DATEADD('minute', -5, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), {high_precision})
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
            value = pdf.iloc[0]["BALANCE"]
            self.assertIsInstance(value, decimal.Decimal)
            self.assertEqual(value, expected)
            # A float round-trip would collapse the value to ~16 significant digits; ensure it did not.
            self.assertNotEqual(value, decimal.Decimal(str(float(high_precision))))

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[entity_key]], validate_fn=_validate, desc="high-precision decimal BFV"
        )

    # =========================================================================
    # as_pandas fast path: wiring, dtype parity, and offline rejection
    # =========================================================================

    def test_as_pandas_offline_rejected(self) -> None:
        """``as_pandas=True`` with ``StoreType.OFFLINE`` must raise INVALID_ARGUMENT."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"AS_PANDAS_OFFLINE_REJECT_{s}"
        entity_key = f"U_REJECT_{s}"
        expected_amount = 11.0

        src_table = self._create_batch_source_table(fs, s, entity_key, expected_amount)
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

        with self.assertRaises(ValueError) as ctx:
            fs.read_feature_view(registered, keys=[[entity_key]], store_type=StoreType.OFFLINE, as_pandas=True)
        self.assertIn("(2110)", str(ctx.exception))
        self.assertIn("as_pandas=True", str(ctx.exception))
        self.assertIn("OFFLINE", str(ctx.exception))

    def test_as_pandas_postgres_online_reuses_http_client(self) -> None:
        """Two consecutive Postgres online reads must reuse the same ``fs._online_http_client`` instance."""
        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"AS_PANDAS_REUSE_{s}"
        batch_key = f"U_REUSE_{s}"
        expected_amount = 12.0

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
        fs.register_feature_view(fv, "v1")
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        def _validate(pdf):
            self.assertIn("AMOUNT", pdf.columns)
            self.assertAlmostEqual(float(pdf.iloc[0]["AMOUNT"]), expected_amount, places=3)

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[batch_key]], validate_fn=_validate, desc="as_pandas reuse warmup"
        )

        first_client = fs._online_http_client
        self.assertIsNotNone(first_client, "Postgres online read must populate fs._online_http_client.")

        fv_live = fs.get_feature_view(fv_name, "v1")
        # Retry a transient 404; it's a response through the existing client, so reuse holds.
        pdf2 = self._read_online_with_retry(fs, fv_live, keys=[[batch_key]])
        self.assertIs(fs._online_http_client, first_client, "Second read must reuse the same HTTP client.")
        import pandas as pd

        self.assertIsInstance(pdf2, pd.DataFrame)

    def test_as_pandas_parity_all_supported_types(self) -> None:
        """``as_pandas=True`` must match ``.to_pandas()`` on the Snowpark path (column order + dtypes)."""
        import pandas as pd

        fs = self._create_feature_store()
        s = uuid.uuid4().hex[:8]
        fv_name = f"AS_PANDAS_PARITY_{s}"
        entity_key = f"U_PARITY_{s}"

        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.ALL_TYPES_PARITY_SRC_{s}"
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
            refresh_freq="2 minutes",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        fs.register_feature_view(fv, "v1")
        self._wait_offline_dt_rows(fs, fv_name, "v1")

        # Online reads drop timestamp_col (EVENT_TIME) from the schema.
        compared_cols = ("USER_ID", "SCORE", "RANK", "PRICE", "IS_ACTIVE")

        def _validate_snowpark(pdf):
            # Require every compared column to be present AND non-null so the warmup poll waits for
            # the online row to be fully materialized before the measured parity reads run. A
            # transient null would otherwise flip pandas dtype inference (bool -> object, float ->
            # nan) and flake the dtype/value parity assertions below.
            for col in compared_cols:
                self.assertIn(col, pdf.columns)
                self.assertFalse(
                    bool(pd.isna(pdf.iloc[0][col])),
                    f"online value for {col} not yet materialized (null)",
                )

        self._poll_online_read(
            fs, fv_name, "v1", keys=[[entity_key]], validate_fn=_validate_snowpark, desc="as_pandas parity warmup"
        )

        fv_live = fs.get_feature_view(fv_name, "v1")
        # Retry both parity arms: a transient online-serving 404 can surface on a single measured
        # read even after the warmup poll observed rows. require_non_null_cols also retries when the
        # online row is present but a value is still null, so both arms observe the same
        # fully-materialized row (a transient null flips dtype inference and skews the SCORE value).
        non_null_cols = list(compared_cols)
        pdf_sp = self._read_online_with_retry(
            fs, fv_live, keys=[[entity_key]], as_pandas=False, require_non_null_cols=non_null_cols
        ).to_pandas()
        pdf_fast = self._read_online_with_retry(
            fs, fv_live, keys=[[entity_key]], as_pandas=True, require_non_null_cols=non_null_cols
        )

        def _log_parity_diagnostics() -> None:
            # Domain-only diagnostics so a future failure shows whether the online value was still
            # null (a materialization race) or the two read paths genuinely diverged on type/value.
            logging.error(
                "as_pandas parity mismatch for %s/v1 key=%s\n"
                "  fast row=%r\n  fast dtypes=%s\n"
                "  snowpark row=%r\n  snowpark dtypes=%s",
                fv_name,
                entity_key,
                pdf_fast.iloc[0].to_dict(),
                pdf_fast.dtypes.to_dict(),
                pdf_sp.iloc[0].to_dict(),
                pdf_sp.dtypes.to_dict(),
            )
            try:
                offline_pdf = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE).to_pandas()
                offline_row = offline_pdf.iloc[0].to_dict() if len(offline_pdf) else "<no offline rows>"
                logging.error("  offline row=%r", offline_row)
            except Exception:
                logging.error("  offline row unavailable", exc_info=True)
            try:
                logging.error("  online service status=%r", fs.get_online_service_status())
            except Exception:
                logging.error("  online service status unavailable", exc_info=True)

        try:
            self.assertIsInstance(pdf_fast, pd.DataFrame)
            self.assertEqual(list(pdf_fast.columns), list(pdf_sp.columns))
            # Both arms are gated on non-null values above; assert it explicitly so any residual
            # race surfaces as "value still null" rather than a confusing nan/dtype-kind failure.
            for col in compared_cols:
                self.assertFalse(
                    bool(pd.isna(pdf_fast.iloc[0][col])), f"fast-path value for {col} still null after retries"
                )
                self.assertFalse(
                    bool(pd.isna(pdf_sp.iloc[0][col])), f"snowpark value for {col} still null after retries"
                )
            # NUMBER columns: fast path keeps Decimal (object), Snowpark Arrow downcasts to narrow numeric.
            decimal_skew_cols = {"RANK", "PRICE"}
            for col in pdf_sp.columns:
                if col in decimal_skew_cols:
                    continue
                self.assertEqual(
                    pdf_fast[col].dtype.kind,
                    pdf_sp[col].dtype.kind,
                    f"dtype-kind mismatch on {col}: fast={pdf_fast[col].dtype} vs sp={pdf_sp[col].dtype}",
                )
            self.assertEqual(pdf_fast.iloc[0]["USER_ID"], pdf_sp.iloc[0]["USER_ID"])
            self.assertAlmostEqual(float(pdf_fast.iloc[0]["SCORE"]), float(pdf_sp.iloc[0]["SCORE"]), places=5)
            self.assertEqual(int(pdf_fast.iloc[0]["RANK"]), int(pdf_sp.iloc[0]["RANK"]))
            # Snowpark Arrow rounds NUMBER(10,2) to a narrow int; compare via float+round.
            self.assertEqual(round(float(pdf_fast.iloc[0]["PRICE"])), round(float(pdf_sp.iloc[0]["PRICE"])))
            self.assertEqual(bool(pdf_fast.iloc[0]["IS_ACTIVE"]), bool(pdf_sp.iloc[0]["IS_ACTIVE"]))
        except AssertionError:
            _log_parity_diagnostics()
            raise

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
