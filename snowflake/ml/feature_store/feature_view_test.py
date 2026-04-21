"""Unit tests for feature_view module."""

from __future__ import annotations

import datetime
import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from snowflake.ml.feature_store.feature_store import FeatureStore
    from snowflake.ml.feature_store.stream_config import StreamConfig

from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StorageConfig,
    StorageFormat,
)
from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    StoreType,
    TableType,
)
from snowflake.snowpark import Row
from snowflake.snowpark.types import (
    DataType,
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)


class FeatureViewValidationTest(parameterized.TestCase):
    """Unit tests for FeatureView validation logic."""

    def _create_mock_feature_view_with_specs(self, specs: list[AggregationSpec]) -> FeatureView:
        """Create a FeatureView with mocked DataFrame for testing validation."""
        from snowflake.ml.feature_store.entity import Entity

        mock_df = MagicMock()
        mock_df.columns = ["user_id", "event_ts", "amount"]
        mock_df.queries = {"queries": ["SELECT * FROM source"]}

        # Create a real entity with a join key that matches the DataFrame
        entity = Entity(name="user", join_keys=["user_id"])

        # Create FV - pass specs via _kwargs to bypass Feature.to_spec()
        return FeatureView(
            name="test_fv",
            entities=[entity],
            feature_df=mock_df,
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            _aggregation_specs=specs,  # Pass directly, not via features
        )

    def test_duplicate_feature_alias_raises_error(self) -> None:
        """Test that duplicate feature aliases raise ValueError."""
        specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="amount",
                window="2h",
                output_column="TOTAL",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="amount",
                window="2h",
                output_column="TOTAL",  # Duplicate!
            ),
        ]

        with self.assertRaises(ValueError) as cm:
            self._create_mock_feature_view_with_specs(specs)

        self.assertIn("Duplicate feature alias", str(cm.exception))
        self.assertIn("TOTAL", str(cm.exception))

    def test_duplicate_alias_case_insensitive(self) -> None:
        """Test that duplicate aliases are detected case-insensitively."""
        specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="amount",
                window="2h",
                output_column="Total",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="amount",
                window="2h",
                output_column="TOTAL",  # Same when uppercased
            ),
        ]

        with self.assertRaises(ValueError) as cm:
            self._create_mock_feature_view_with_specs(specs)

        self.assertIn("Duplicate feature alias", str(cm.exception))


class OnlineConfigTest(parameterized.TestCase):
    """Unit tests for OnlineConfig class."""

    @parameterized.parameters(  # type: ignore[misc]
        "10s",
        "5m",
        "2h",
        "1d",
        "10 seconds",
        "5 minutes",
        "2 hours",
        "1 day",
        "1 sec",  # Snowflake supports this
        "30 mins",  # Snowflake supports this
    )
    def test_online_config_valid_target_lag(self, target_lag: str) -> None:
        """Test OnlineConfig accepts valid target_lag values."""
        config = OnlineConfig(enable=True, target_lag=target_lag)
        # Just verify it doesn't throw an error and stores the value as-is (after trim)
        self.assertEqual(config.target_lag, target_lag.strip())

    @parameterized.parameters(  # type: ignore[misc]
        ("  10s  ", "10s"),
        ("\t5m\t", "5m"),
        ("\n2h\n", "2h"),
        ("  10 seconds  ", "10 seconds"),
    )
    def test_online_config_whitespace_handling(self, input_val: str, expected: str) -> None:
        """Test OnlineConfig trims whitespace from target_lag."""
        config = OnlineConfig(enable=True, target_lag=input_val)
        self.assertEqual(config.target_lag, expected)

    @parameterized.parameters(  # type: ignore[misc]
        "",
        "   ",
        "\t",
        "\n",
        123,
        10.5,
    )
    def test_online_config_invalid_target_lag(self, invalid_target_lag: object) -> None:
        """Test OnlineConfig rejects empty/invalid target_lag values."""
        with self.assertRaises(ValueError) as cm:
            OnlineConfig(enable=True, target_lag=invalid_target_lag)  # type: ignore[arg-type]

        error_msg = str(cm.exception)
        self.assertIn("non-empty string", error_msg)

    # ---- OnlineStoreType tests ----

    def test_online_config_default_store_type(self) -> None:
        """Test OnlineConfig defaults to HYBRID_TABLE store type."""
        config = OnlineConfig(enable=True, target_lag="10s")
        self.assertEqual(config.store_type, OnlineStoreType.HYBRID_TABLE)

    def test_online_config_postgres_store_type(self) -> None:
        """Test OnlineConfig accepts POSTGRES store type."""
        config = OnlineConfig(
            enable=True,
            target_lag="30s",
            store_type=OnlineStoreType.POSTGRES,
        )
        self.assertEqual(config.store_type, OnlineStoreType.POSTGRES)
        self.assertEqual(config.target_lag, "30s")
        self.assertTrue(config.enable)

    def test_online_config_to_json_with_store_type(self) -> None:
        """Test OnlineConfig serializes store_type correctly."""
        config = OnlineConfig(
            enable=True,
            target_lag="30s",
            store_type=OnlineStoreType.POSTGRES,
        )
        json_str = config.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["store_type"], "postgres")
        self.assertEqual(data["enable"], True)
        self.assertEqual(data["target_lag"], "30s")

    def test_online_config_to_json_default_store_type(self) -> None:
        """Test OnlineConfig serializes default HYBRID_TABLE store_type."""
        config = OnlineConfig(enable=True, target_lag="10s")
        json_str = config.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["store_type"], "hybrid_table")

    def test_online_config_from_json_with_store_type(self) -> None:
        """Test OnlineConfig deserializes store_type correctly."""
        json_str = '{"enable": true, "target_lag": "30s", "store_type": "postgres"}'
        config = OnlineConfig.from_json(json_str)
        self.assertEqual(config.store_type, OnlineStoreType.POSTGRES)
        self.assertEqual(config.target_lag, "30s")
        self.assertTrue(config.enable)

    def test_online_config_from_json_backward_compat(self) -> None:
        """Test OnlineConfig deserializes old configs without store_type."""
        json_str = '{"enable": true, "target_lag": "10s"}'
        config = OnlineConfig.from_json(json_str)
        self.assertEqual(config.store_type, OnlineStoreType.HYBRID_TABLE)
        self.assertEqual(config.target_lag, "10s")
        self.assertTrue(config.enable)

    def test_online_config_from_json_backward_compat_disabled(self) -> None:
        """Test backward compat for old disabled config without store_type."""
        json_str = '{"enable": false, "target_lag": null}'
        config = OnlineConfig.from_json(json_str)
        self.assertEqual(config.store_type, OnlineStoreType.HYBRID_TABLE)
        self.assertFalse(config.enable)
        self.assertIsNone(config.target_lag)

    def test_online_config_roundtrip_json(self) -> None:
        """Test OnlineConfig round-trips through JSON correctly."""
        original = OnlineConfig(
            enable=True,
            target_lag="15s",
            store_type=OnlineStoreType.POSTGRES,
        )
        restored = OnlineConfig.from_json(original.to_json())
        self.assertEqual(restored.enable, original.enable)
        self.assertEqual(restored.target_lag, original.target_lag)
        self.assertEqual(restored.store_type, original.store_type)

    def test_online_config_roundtrip_json_hybrid_table(self) -> None:
        """Test round-trip for default HYBRID_TABLE config."""
        original = OnlineConfig(enable=True, target_lag="10s")
        restored = OnlineConfig.from_json(original.to_json())
        self.assertEqual(restored.store_type, OnlineStoreType.HYBRID_TABLE)
        self.assertEqual(restored.target_lag, original.target_lag)

    def test_online_store_type_enum_values(self) -> None:
        """Test OnlineStoreType enum values."""
        self.assertEqual(OnlineStoreType.HYBRID_TABLE.value, "hybrid_table")
        self.assertEqual(OnlineStoreType.POSTGRES.value, "postgres")

    def test_online_config_frozen(self) -> None:
        """Test OnlineConfig is immutable (frozen dataclass)."""
        config = OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES)
        with self.assertRaises(AttributeError):
            config.store_type = OnlineStoreType.HYBRID_TABLE  # type: ignore[misc]


class StorageConfigTest(parameterized.TestCase):
    """Unit tests for StorageConfig class."""

    def test_storage_config_to_json_iceberg(self) -> None:
        """Test StorageConfig serialization for Iceberg format."""
        config = StorageConfig(
            format=StorageFormat.ICEBERG,
            external_volume="MY_VOLUME",
            base_location="path/to/data",
        )
        json_str = config.to_json()
        self.assertIn('"format": "iceberg"', json_str)
        self.assertIn('"external_volume": "MY_VOLUME"', json_str)
        self.assertIn('"base_location": "path/to/data"', json_str)

    def test_storage_config_from_json_iceberg(self) -> None:
        """Test StorageConfig deserialization for Iceberg format."""
        json_str = '{"format": "iceberg", "external_volume": "VOL", "base_location": "loc"}'
        config = StorageConfig.from_json(json_str)
        self.assertEqual(config.format, StorageFormat.ICEBERG)
        self.assertEqual(config.external_volume, "VOL")
        self.assertEqual(config.base_location, "loc")

    def test_storage_config_default_snowflake(self) -> None:
        """Test StorageConfig defaults to Snowflake format."""
        config = StorageConfig()
        self.assertEqual(config.format, StorageFormat.SNOWFLAKE)
        self.assertIsNone(config.external_volume)
        self.assertIsNone(config.base_location)


class BuildBatchFeatureViewSpecTest(absltest.TestCase):
    """Tests for FeatureStore._build_batch_feature_view_spec."""

    def _make_mock_feature_store(self, database: str = "TEST_DB", schema: str = "TEST_SCHEMA") -> MagicMock:
        """Create a mock FeatureStore with _config set."""
        from snowflake.ml.feature_store.feature_store import (
            FeatureStore,
            _FeatureStoreConfig,
        )

        mock_fs = MagicMock(spec=FeatureStore)
        mock_fs._config = _FeatureStoreConfig(
            database=SqlIdentifier(database),
            schema=SqlIdentifier(schema),
        )
        # Bind the real method to our mock
        mock_fs._build_batch_feature_view_spec = FeatureStore._build_batch_feature_view_spec.__get__(mock_fs)
        return mock_fs

    def _make_feature_view(
        self,
        *,
        columns: list[str],
        column_types: Optional[list[DataType]] = None,
        entity_keys: list[str],
        timestamp_col: Optional[str] = None,
        feature_granularity: Optional[str] = None,
        aggregation_specs: Optional[list[AggregationSpec]] = None,
    ) -> FeatureView:
        """Create a FeatureView with mocked DataFrame."""
        if column_types is None:
            column_types = [DoubleType()] * len(columns)

        schema = StructType([StructField(c, t) for c, t in zip(columns, column_types)])

        mock_df = MagicMock()
        mock_df.columns = columns
        mock_df.queries = {"queries": ["SELECT * FROM source"]}
        mock_df.schema = schema

        entity = Entity(name="test_entity", join_keys=entity_keys)

        return FeatureView(
            name="test_fv",
            entities=[entity],
            feature_df=mock_df,
            timestamp_col=timestamp_col,
            refresh_freq="1h",
            feature_granularity=feature_granularity,
            _aggregation_specs=aggregation_specs,
        )

    # ------------------------------------------------------------------ #
    # Non-tiled batch FV
    # ------------------------------------------------------------------ #

    def test_non_tiled_basic(self) -> None:
        """Non-tiled batch FV: passthrough features for non-entity columns."""
        fv = self._make_feature_view(
            columns=["USER_ID", "AMOUNT", "SCORE"],
            column_types=[StringType(), DoubleType(), DoubleType()],
            entity_keys=["USER_ID"],
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(fv, SqlIdentifier("DT_NAME"), "v1", "30s")

        # Kind and metadata
        self.assertEqual(spec.kind, FeatureViewKind.BatchFeatureView)
        self.assertEqual(spec.metadata.database, "TEST_DB")
        self.assertEqual(spec.metadata.schema_, "TEST_SCHEMA")
        self.assertEqual(spec.metadata.name, "TEST_FV")
        self.assertEqual(spec.metadata.version, "v1")

        # Online store type
        self.assertEqual(spec.online_store_type, StoreType.POSTGRES)

        # Offline config
        self.assertEqual(len(spec.offline_configs), 1)
        self.assertEqual(spec.offline_configs[0].table_type, TableType.BATCH_SOURCE)
        self.assertEqual(spec.offline_configs[0].table, "DT_NAME")

        # Properties
        self.assertEqual(spec.spec.ordered_entity_column_names, ["USER_ID"])
        self.assertIsNone(spec.spec.timestamp_field)
        self.assertIsNone(spec.spec.feature_granularity_sec)
        self.assertIsNone(spec.spec.feature_aggregation_method)
        self.assertEqual(spec.spec.target_lag_sec, 30)

        # Passthrough features: AMOUNT and SCORE (USER_ID excluded as entity)
        self.assertEqual(len(spec.spec.features), 2)
        feat_names = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(feat_names, ["AMOUNT", "SCORE"])
        for feat in spec.spec.features:
            self.assertEqual(feat.source_column, feat.output_column)
            self.assertIsNone(feat.function)
            self.assertIsNone(feat.window_sec)

        # No sources for non-tiled batch
        self.assertEqual(spec.spec.sources, [])
        self.assertIsNone(spec.spec.udf)

    def test_non_tiled_with_timestamp(self) -> None:
        """Non-tiled batch FV with timestamp: timestamp column excluded from features."""
        fv = self._make_feature_view(
            columns=["USER_ID", "EVENT_TIME", "AMOUNT", "SCORE"],
            column_types=[StringType(), TimestampType(), DoubleType(), DoubleType()],
            entity_keys=["USER_ID"],
            timestamp_col="EVENT_TIME",
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(fv, SqlIdentifier("DT_NAME"), "v1", "1 minute")

        # Timestamp set
        self.assertEqual(spec.spec.timestamp_field, "EVENT_TIME")

        # Passthrough features: AMOUNT and SCORE (USER_ID + EVENT_TIME excluded)
        feat_names = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(feat_names, ["AMOUNT", "SCORE"])
        for feat in spec.spec.features:
            self.assertIsNone(feat.function)

    def test_non_tiled_without_timestamp(self) -> None:
        """Non-tiled batch FV without timestamp: all non-entity columns become features."""
        fv = self._make_feature_view(
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            entity_keys=["USER_ID"],
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(fv, SqlIdentifier("MY_VIEW"), "v2", "60s")

        self.assertIsNone(spec.spec.timestamp_field)
        self.assertEqual(len(spec.spec.features), 1)
        self.assertEqual(spec.spec.features[0].output_column.name, "AMOUNT")

    def test_non_tiled_multiple_entities(self) -> None:
        """Non-tiled batch FV with multiple entity join keys: all excluded from features."""
        entity1 = Entity(name="user", join_keys=["USER_ID"])
        entity2 = Entity(name="merchant", join_keys=["MERCHANT_ID"])

        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("MERCHANT_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("SCORE", DoubleType()),
            ]
        )
        mock_df = MagicMock()
        mock_df.columns = ["USER_ID", "MERCHANT_ID", "AMOUNT", "SCORE"]
        mock_df.queries = {"queries": ["SELECT * FROM source"]}
        mock_df.schema = schema

        fv = FeatureView(
            name="multi_entity_fv",
            entities=[entity1, entity2],
            feature_df=mock_df,
            refresh_freq="1h",
        )

        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(fv, SqlIdentifier("DT_NAME"), "v1", "30s")

        self.assertEqual(spec.spec.ordered_entity_column_names, ["USER_ID", "MERCHANT_ID"])
        feat_names = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(feat_names, ["AMOUNT", "SCORE"])

    def test_non_tiled_preserves_column_types(self) -> None:
        """Non-tiled batch FV: passthrough features preserve column types."""
        fv = self._make_feature_view(
            columns=["USER_ID", "AMOUNT", "COUNT_VAL"],
            column_types=[StringType(), DoubleType(), DecimalType(18, 0)],
            entity_keys=["USER_ID"],
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(fv, SqlIdentifier("DT_NAME"), "v1", "30s")

        self.assertEqual(len(spec.spec.features), 2)

        amount_feat = spec.spec.features[0]
        self.assertEqual(amount_feat.output_column.name, "AMOUNT")
        self.assertEqual(amount_feat.output_column.type, "DoubleType")

        count_feat = spec.spec.features[1]
        self.assertEqual(count_feat.output_column.name, "COUNT_VAL")
        self.assertEqual(count_feat.output_column.type, "DecimalType")
        self.assertEqual(count_feat.output_column.precision, 18)
        self.assertEqual(count_feat.output_column.scale, 0)

    # ------------------------------------------------------------------ #
    # Tiled batch FV
    # ------------------------------------------------------------------ #

    def test_tiled_basic(self) -> None:
        """Tiled batch FV: aggregation features with function and window."""
        agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            ),
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_COUNT_24H",
            ),
        ]
        # Tiled FVs: offline_configs describe the materialized DT (from Snowflake), not feature_df.
        fv = self._make_feature_view(
            columns=["USER_ID", "EVENT_TIME", "AMOUNT"],
            column_types=[StringType(), TimestampType(), DoubleType()],
            entity_keys=["USER_ID"],
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            aggregation_specs=agg_specs,
        )
        tiled_dt_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
                StructField("_PARTIAL_COUNT_AMOUNT", LongType()),
            ]
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(
            fv,
            SqlIdentifier("DT_TILED"),
            "v1",
            "30s",
            offline_materialized_schema=tiled_dt_schema,
        )

        # Kind
        self.assertEqual(spec.kind, FeatureViewKind.BatchFeatureView)
        self.assertEqual(spec.offline_configs[0].table_type, TableType.BATCH_SOURCE)

        # Properties
        self.assertEqual(spec.spec.ordered_entity_column_names, ["USER_ID"])
        self.assertEqual(spec.spec.timestamp_field, "EVENT_TIME")
        self.assertEqual(spec.spec.feature_granularity_sec, 3600)
        self.assertEqual(spec.spec.feature_aggregation_method, FeatureAggregationMethod.TILES)
        self.assertEqual(spec.spec.target_lag_sec, 30)

        # Features
        self.assertEqual(len(spec.spec.features), 2)

        sum_feat = spec.spec.features[0]
        self.assertEqual(sum_feat.source_column.name, "AMOUNT")
        self.assertEqual(sum_feat.output_column.name, "AMOUNT_SUM_24H")
        self.assertEqual(sum_feat.output_column.type, "DoubleType")  # SUM preserves source type
        self.assertEqual(sum_feat.function, "sum")
        self.assertEqual(sum_feat.window_sec, 86400)

        count_feat = spec.spec.features[1]
        self.assertEqual(count_feat.source_column.name, "AMOUNT")
        self.assertEqual(count_feat.output_column.name, "AMOUNT_COUNT_24H")
        self.assertEqual(count_feat.output_column.type, "DecimalType")  # COUNT always integer
        self.assertEqual(count_feat.function, "count")
        self.assertEqual(count_feat.window_sec, 86400)

        # BATCH source is builder-internal — must not appear in output
        self.assertEqual(spec.spec.sources, [])

    def test_tiled_without_timestamp_raises(self) -> None:
        """Tiled FVs require a timestamp column — FeatureView validation catches this."""
        agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            ),
        ]
        # FeatureView itself raises when feature_granularity is set without timestamp_col
        with self.assertRaises(ValueError):
            self._make_feature_view(
                columns=["USER_ID", "AMOUNT_SUM_24H"],
                column_types=[StringType(), DoubleType()],
                entity_keys=["USER_ID"],
                # timestamp_col not set
                feature_granularity="1h",
                aggregation_specs=agg_specs,
            )

    def test_tiled_multiple_agg_types(self) -> None:
        """Tiled FV with mixed aggregation types: SUM, AVG, LAST."""
        agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="1h",
                output_column="AMOUNT_SUM_1H",
            ),
            AggregationSpec(
                function=AggregationType.AVG,
                source_column="AMOUNT",
                window="1h",
                output_column="AMOUNT_AVG_1H",
            ),
            AggregationSpec(
                function=AggregationType.MAX,
                source_column="AMOUNT",
                window="1h",
                output_column="AMOUNT_MAX_1H",
            ),
        ]
        fv = self._make_feature_view(
            columns=["USER_ID", "EVENT_TIME", "AMOUNT"],
            column_types=[StringType(), TimestampType(), DoubleType()],
            entity_keys=["USER_ID"],
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            aggregation_specs=agg_specs,
        )
        tiled_dt_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
                StructField("_PARTIAL_COUNT_AMOUNT", LongType()),
                StructField("_PARTIAL_MAX_AMOUNT", DoubleType()),
            ]
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(
            fv,
            SqlIdentifier("DT_TILED"),
            "v1",
            "30s",
            offline_materialized_schema=tiled_dt_schema,
        )

        self.assertEqual(len(spec.spec.features), 3)
        functions = [f.function for f in spec.spec.features]
        self.assertEqual(functions, ["sum", "avg", "max"])

    # ------------------------------------------------------------------ #
    # Spec serialization sanity check
    # ------------------------------------------------------------------ #

    def test_spec_to_json_roundtrip(self) -> None:
        """Spec serializes to JSON and can be parsed back."""
        fv = self._make_feature_view(
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            entity_keys=["USER_ID"],
        )
        fs = self._make_mock_feature_store()
        spec = fs._build_batch_feature_view_spec(fv, SqlIdentifier("DT_NAME"), "v1", "30s")

        json_str = spec.to_json()
        parsed = json.loads(json_str)

        self.assertEqual(parsed["kind"], "BatchFeatureView")
        self.assertEqual(parsed["metadata"]["name"], "TEST_FV")
        self.assertEqual(parsed["metadata"]["version"], "v1")
        self.assertEqual(len(parsed["spec"]["features"]), 1)
        self.assertEqual(parsed["spec"]["features"][0]["output_column"]["name"], "AMOUNT")

    # ------------------------------------------------------------------ #
    # Case-sensitive identifier handling
    # ------------------------------------------------------------------ #

    def test_case_sensitive_identifiers_use_resolved_names(self) -> None:
        """Case-sensitive SqlIdentifiers produce resolved names (no SQL quotes) in spec JSON.

        Database, schema, FV name, and table name can be case-sensitive
        (created with double-quoted SQL identifiers). The spec JSON must
        contain the *resolved* form (e.g. ``myDb``) not the SQL identifier
        form (e.g. ``"myDb"`` with literal double-quote characters).
        """
        # Case-sensitive DB/schema via quoted identifiers
        fs = self._make_mock_feature_store(database='"myDb"', schema='"mySchema"')

        # Standard uppercase columns (as Snowpark returns them)
        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType()),
                StructField("AMOUNT", DoubleType()),
            ]
        )
        mock_df = MagicMock()
        mock_df.columns = ["USER_ID", "EVENT_TIME", "AMOUNT"]
        mock_df.queries = {"queries": ["SELECT * FROM source"]}
        mock_df.schema = schema

        entity = Entity(name="user", join_keys=["USER_ID"])
        fv = FeatureView(
            name='"myFv"',  # quoted → case-sensitive
            entities=[entity],
            feature_df=mock_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="1h",
        )

        # Case-sensitive table name
        table_name = SqlIdentifier('"myTable"')

        spec = fs._build_batch_feature_view_spec(fv, table_name, "v1", "30s")

        # All names should be the *resolved* form — no literal double-quote characters
        self.assertEqual(spec.metadata.database, "myDb")
        self.assertEqual(spec.metadata.schema_, "mySchema")
        self.assertEqual(spec.metadata.name, "myFv")
        self.assertEqual(spec.offline_configs[0].database, "myDb")
        self.assertEqual(spec.offline_configs[0].schema_, "mySchema")
        self.assertEqual(spec.offline_configs[0].table, "myTable")
        self.assertEqual(spec.spec.timestamp_field, "EVENT_TIME")

        # Entity column is resolved via jk.resolved()
        self.assertEqual(spec.spec.ordered_entity_column_names, ["USER_ID"])

        # Verify the serialized JSON uses resolved names, not SQL identifier form
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metadata"]["database"], "myDb")
        self.assertEqual(parsed["metadata"]["schema"], "mySchema")
        self.assertEqual(parsed["metadata"]["name"], "myFv")
        # Ensure no stray literal double-quotes leaked into string values
        for key in ("database", "schema", "name"):
            self.assertFalse(
                parsed["metadata"][key].startswith('"'),
                f"metadata.{key} should not start with a double-quote: {parsed['metadata'][key]}",
            )


class CreateOnlineFeatureTableTest(absltest.TestCase):
    """Tests for FeatureStore._create_online_feature_table SQL assembly."""

    def _make_mock_feature_store(
        self,
        database: str = "TEST_DB",
        schema: str = "TEST_SCHEMA",
        *,
        postgres_online_service_running: bool = False,
    ) -> MagicMock:
        """Create a mock FeatureStore with methods needed for _create_online_feature_table."""
        from snowflake.ml.feature_store.feature_store import (
            FeatureStore,
            _FeatureStoreConfig,
        )

        mock_fs = MagicMock(spec=FeatureStore)
        mock_fs._config = _FeatureStoreConfig(
            database=SqlIdentifier(database),
            schema=SqlIdentifier(schema),
        )
        mock_fs._default_warehouse = None
        mock_fs._telemetry_stmp = {}

        # _get_fully_qualified_name: replicate real behaviour
        def _get_fqn(name: object) -> str:
            return f"{database}.{schema}.{name}"

        mock_fs._get_fully_qualified_name = _get_fqn

        # Bind real methods
        unwrapped = FeatureStore._create_online_feature_table.__wrapped__  # type: ignore[attr-defined]
        mock_fs._create_online_feature_table = unwrapped.__get__(mock_fs)
        mock_fs._build_batch_feature_view_spec = FeatureStore._build_batch_feature_view_spec.__get__(mock_fs)

        mock_fs._session = MagicMock()
        if postgres_online_service_running:
            status_json = json.dumps(
                {
                    "status": "RUNNING",
                    "message": "ok",
                    "endpoints": [{"name": "query", "url": "https://example.com/query"}],
                }
            )

            def sql_side_effect(query: str, *args: object, **kwargs: object) -> MagicMock:
                mock_result = MagicMock()
                qn = query.replace("\n", " ")
                loc = f"{SqlIdentifier(database)}.{SqlIdentifier(schema)}"
                if f"SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('{loc}')" in qn:
                    mock_result.collect.return_value = [Row(status_json)]
                else:
                    mock_result.collect.return_value = []
                return mock_result

            mock_fs._session.sql.side_effect = sql_side_effect
        else:
            mock_fs._session.sql.return_value.collect.return_value = []

        return mock_fs

    def _first_online_feature_table_sql(self, mock_fs: MagicMock) -> str:
        for call in mock_fs._session.sql.call_args_list:
            q = call[0][0]
            if isinstance(q, str) and "ONLINE FEATURE TABLE" in q.upper():
                return q
        raise AssertionError("No CREATE ONLINE FEATURE TABLE SQL found in mock session calls")

    def _make_feature_view(
        self,
        *,
        entity_keys: list[str],
        columns: list[str],
        column_types: Optional[list[DataType]] = None,
        timestamp_col: Optional[str] = None,
        store_type: OnlineStoreType = OnlineStoreType.HYBRID_TABLE,
    ) -> FeatureView:
        """Create a non-tiled FeatureView with mocked DataFrame."""
        if column_types is None:
            column_types = [DoubleType()] * len(columns)
        schema = StructType([StructField(c, t) for c, t in zip(columns, column_types)])
        mock_df = MagicMock()
        mock_df.columns = columns
        mock_df.queries = {"queries": ["SELECT * FROM source"]}
        mock_df.schema = schema

        entity = Entity(name="test_entity", join_keys=entity_keys)
        return FeatureView(
            name="test_fv",
            entities=[entity],
            feature_df=mock_df,
            timestamp_col=timestamp_col,
            refresh_freq="1h",
            online_config=OnlineConfig(enable=True, target_lag="30s", store_type=store_type),
        )

    # ------------------------------------------------------------------ #
    # HYBRID_TABLE path (regression)
    # ------------------------------------------------------------------ #

    def test_hybrid_table_sql_uses_from_clause(self) -> None:
        """HYBRID_TABLE path: SQL uses FROM <source_table>."""
        fv = self._make_feature_view(
            entity_keys=["USER_ID"],
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            store_type=OnlineStoreType.HYBRID_TABLE,
        )
        fs = self._make_mock_feature_store()

        fs._create_online_feature_table(fv, SqlIdentifier("DT_NAME"), version="v1")

        # Inspect the SQL that was executed
        sql_call = fs._session.sql.call_args_list[0]
        query = sql_call[0][0]
        self.assertIn("FROM TEST_DB.TEST_SCHEMA.DT_NAME", query)
        self.assertNotIn("FROM SPECIFICATION", query)
        self.assertIn("PRIMARY KEY", query)
        self.assertIn("TARGET_LAG='30s'", query)

    def test_hybrid_table_sql_overwrite(self) -> None:
        """HYBRID_TABLE with overwrite: SQL contains OR REPLACE."""
        fv = self._make_feature_view(
            entity_keys=["USER_ID"],
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            store_type=OnlineStoreType.HYBRID_TABLE,
        )
        fs = self._make_mock_feature_store()

        fs._create_online_feature_table(fv, SqlIdentifier("DT_NAME"), version="v1", overwrite=True)

        query = fs._session.sql.call_args_list[0][0][0]
        self.assertIn("CREATE OR REPLACE ONLINE FEATURE TABLE", query)

    # ------------------------------------------------------------------ #
    # POSTGRES path
    # ------------------------------------------------------------------ #

    def test_postgres_sql_uses_specification_clause(self) -> None:
        """POSTGRES path: SQL uses FROM SPECIFICATION $$...$$."""
        fv = self._make_feature_view(
            entity_keys=["USER_ID"],
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            store_type=OnlineStoreType.POSTGRES,
        )
        fs = self._make_mock_feature_store(postgres_online_service_running=True)

        fs._create_online_feature_table(fv, SqlIdentifier("DT_NAME"), version="v1")

        query = self._first_online_feature_table_sql(fs)
        self.assertIn("FROM SPECIFICATION $$", query)
        self.assertIn("$$", query)
        self.assertNotIn("FROM TEST_DB.TEST_SCHEMA.DT_NAME", query)
        self.assertIn("PRIMARY KEY", query)
        self.assertIn("TARGET_LAG='30s'", query)

        # Verify the spec JSON is embedded
        self.assertIn('"BatchFeatureView"', query)

    def test_postgres_sql_contains_valid_spec_json(self) -> None:
        """POSTGRES path: embedded spec is valid JSON."""
        fv = self._make_feature_view(
            entity_keys=["USER_ID"],
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            store_type=OnlineStoreType.POSTGRES,
        )
        fs = self._make_mock_feature_store(postgres_online_service_running=True)

        fs._create_online_feature_table(fv, SqlIdentifier("DT_NAME"), version="v1")

        query = self._first_online_feature_table_sql(fs)

        # Extract JSON between $$ delimiters
        start = query.index("$$") + 2
        end = query.index("$$", start)
        spec_json = query[start:end]

        parsed = json.loads(spec_json)
        self.assertEqual(parsed["kind"], "BatchFeatureView")
        self.assertIn("features", parsed["spec"])

    # ------------------------------------------------------------------ #
    # $$ injection guard
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Version flows correctly
    # ------------------------------------------------------------------ #

    def test_version_passed_to_spec_builder(self) -> None:
        """Version parameter flows through to the spec builder."""
        fv = self._make_feature_view(
            entity_keys=["USER_ID"],
            columns=["USER_ID", "AMOUNT"],
            column_types=[StringType(), DoubleType()],
            store_type=OnlineStoreType.POSTGRES,
        )
        fs = self._make_mock_feature_store(postgres_online_service_running=True)

        fs._create_online_feature_table(fv, SqlIdentifier("DT_NAME"), version="v42")

        query = self._first_online_feature_table_sql(fs)
        # Extract and verify the version in the embedded spec
        start = query.index("$$") + 2
        end = query.index("$$", start)
        parsed = json.loads(query[start:end])
        self.assertEqual(parsed["metadata"]["version"], "v42")


class PostgresOnlineLocalRowCoercionTest(absltest.TestCase):
    """``Session.create_dataframe`` literal rules for Postgres Query API rows."""

    def test_coerce_int_to_float_for_double(self) -> None:
        from snowflake.ml.feature_store.feature_store import (
            _coerce_row_values_for_snowpark_local_schema,
        )

        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
            ]
        )
        row = {"USER_ID": "k1", "AMOUNT": 999}
        out = _coerce_row_values_for_snowpark_local_schema(row, schema)
        self.assertEqual(out["USER_ID"], "k1")
        self.assertIsInstance(out["AMOUNT"], float)
        self.assertEqual(out["AMOUNT"], 999.0)

    def test_coerce_int_to_decimal_for_count_column(self) -> None:
        from snowflake.ml.feature_store.feature_store import (
            _coerce_row_values_for_snowpark_local_schema,
        )

        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TXN_COUNT_2D", DecimalType(38, 0)),
            ]
        )
        row = {"USER_ID": "k1", "TXN_COUNT_2D": 3}
        out = _coerce_row_values_for_snowpark_local_schema(row, schema)
        self.assertEqual(out["USER_ID"], "k1")
        self.assertIsInstance(out["TXN_COUNT_2D"], Decimal)
        self.assertEqual(out["TXN_COUNT_2D"], Decimal("3"))

    def test_coerce_iso_timestamp_string_for_ntz(self) -> None:
        from snowflake.ml.feature_store.feature_store import (
            _coerce_row_values_for_snowpark_local_schema,
        )

        schema = StructType([StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ))])
        row = {"EVENT_TIME": "2024-06-01T12:00:00Z"}
        out = _coerce_row_values_for_snowpark_local_schema(row, schema)
        self.assertIsInstance(out["EVENT_TIME"], datetime.datetime)
        self.assertIsNone(out["EVENT_TIME"].tzinfo)


class FeatureAggregationMethodTest(absltest.TestCase):
    """Tests for the feature_aggregation_method parameter on FeatureView."""

    def _make_mock_backfill_df(self) -> MagicMock:
        mock_df = MagicMock()
        mock_df.queries = {"queries": ["SELECT * FROM SRC"]}
        mock_df.columns = ["USER_ID", "AMOUNT", "EVENT_TIME"]
        mock_df.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        return mock_df

    def _make_stream_config(self) -> StreamConfig:
        from snowflake.ml.feature_store.stream_config import StreamConfig

        def _identity(df: Any) -> Any:
            return df

        return StreamConfig(
            stream_source="txn_events",
            transformation_fn=_identity,
            backfill_df=self._make_mock_backfill_df(),
        )

    def test_streaming_tiled_defaults_to_tiles(self) -> None:
        """Tiled streaming FV without explicit agg method defaults to TILES."""
        from snowflake.ml.feature_store.feature import Feature

        fv = FeatureView(
            name="test_fv",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            stream_config=self._make_stream_config(),
            timestamp_col="EVENT_TIME",
            feature_granularity="1d",
            features=[Feature.sum("AMOUNT", "2d")],
        )
        self.assertEqual(fv.feature_aggregation_method, FeatureAggregationMethod.TILES)

    def test_streaming_tiled_continuous_accepted(self) -> None:
        """Tiled streaming FV with CONTINUOUS: accepted and stored."""
        from snowflake.ml.feature_store.feature import Feature

        fv = FeatureView(
            name="test_fv",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            stream_config=self._make_stream_config(),
            timestamp_col="EVENT_TIME",
            feature_granularity="1d",
            features=[Feature.sum("AMOUNT", "2d")],
            feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
        )
        self.assertEqual(fv.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)

    def test_streaming_tiled_tiles_explicit(self) -> None:
        """Tiled streaming FV with explicit TILES."""
        from snowflake.ml.feature_store.feature import Feature

        fv = FeatureView(
            name="test_fv",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            stream_config=self._make_stream_config(),
            timestamp_col="EVENT_TIME",
            feature_granularity="1d",
            features=[Feature.sum("AMOUNT", "2d")],
            feature_aggregation_method=FeatureAggregationMethod.TILES,
        )
        self.assertEqual(fv.feature_aggregation_method, FeatureAggregationMethod.TILES)

    def test_streaming_non_tiled_rejects_agg_method(self) -> None:
        """Non-tiled streaming FV with feature_aggregation_method raises ValueError."""
        with self.assertRaisesRegex(ValueError, "feature_aggregation_method requires"):
            FeatureView(
                name="test_fv",
                entities=[Entity(name="user", join_keys=["USER_ID"])],
                stream_config=self._make_stream_config(),
                timestamp_col="EVENT_TIME",
                feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
            )

    def test_streaming_non_tiled_has_none(self) -> None:
        """Non-tiled streaming FV: feature_aggregation_method is None."""
        fv = FeatureView(
            name="test_fv",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            stream_config=self._make_stream_config(),
            timestamp_col="EVENT_TIME",
        )
        self.assertIsNone(fv.feature_aggregation_method)

    def test_batch_rejects_agg_method(self) -> None:
        """Batch FV with feature_aggregation_method raises ValueError."""
        mock_df = MagicMock()
        mock_df.columns = ["USER_ID", "AMOUNT"]
        mock_df.queries = {"queries": ["SELECT * FROM SRC"]}

        with self.assertRaisesRegex(ValueError, "only supported for streaming"):
            FeatureView(
                name="test_fv",
                entities=[Entity(name="user", join_keys=["USER_ID"])],
                feature_df=mock_df,
                feature_aggregation_method=FeatureAggregationMethod.TILES,
            )

    def test_rollup_rejects_agg_method(self) -> None:
        """Rollup FV with feature_aggregation_method raises ValueError.

        The check happens before RollupConfig validation, so we can use a
        lightweight mock for rollup_config.
        """
        mock_rollup = MagicMock()

        with self.assertRaisesRegex(ValueError, "only supported for streaming"):
            FeatureView(
                name="rollup_fv",
                entities=[Entity(name="dept", join_keys=["DEPT_ID"])],
                rollup_config=mock_rollup,
                feature_aggregation_method=FeatureAggregationMethod.TILES,
            )


class UnicodeColumnSqlGenerationTest(absltest.TestCase):
    """Verify SQL generation properly quotes Unicode (Japanese) column names."""

    def _create_feature_store(self) -> FeatureStore:
        from snowflake.ml.feature_store.feature_store import (
            FeatureStore,
            _FeatureStoreConfig,
        )

        fs = object.__new__(FeatureStore)
        fs._config = _FeatureStoreConfig(
            database=SqlIdentifier("TEST_DB"),
            schema=SqlIdentifier("TEST_SCHEMA"),
        )
        return fs

    def _create_mock_fv(self, join_key_str: str, timestamp_col_str: str, fv_name: str = "jp_fv") -> MagicMock:
        entity = MagicMock()
        entity.join_keys = [SqlIdentifier(join_key_str)]

        fv = MagicMock()
        fv.entities = [entity]
        fv.timestamp_col = SqlIdentifier(timestamp_col_str)
        fv.is_tiled = False
        fv.version = "v1"
        fv.name = fv_name
        fv.aggregation_specs = None
        fv.fully_qualified_name.return_value = "TEST_DB.TEST_SCHEMA.JP_FV$v1"
        return fv

    def test_cte_query_quotes_japanese_columns(self) -> None:
        """_build_cte_query must produce properly quoted Japanese identifiers."""
        fs = self._create_feature_store()
        fv = self._create_mock_fv('"顧客ID"', '"記録時刻"')

        feature_columns = ['"色", "高さ", "重さ"']

        query = fs._build_cte_query(
            feature_views=[fv],
            feature_columns=feature_columns,
            spine_ref="SELECT * FROM spine_table",
            spine_timestamp_col=SqlIdentifier('"記録時刻"'),
        )

        for ident in ['"色"', '"高さ"', '"重さ"', '"顧客ID"', '"記録時刻"']:
            self.assertIn(ident, query, f"Missing quoted identifier {ident} in generated SQL")

        for bare in ['""色""', '""高さ""', '""重さ""', '""顧客ID""', '""記録時刻""']:
            self.assertNotIn(bare, query, f"Double-quoted identifier {bare} found in generated SQL")

    def test_cte_query_quotes_japanese_columns_no_timestamp(self) -> None:
        """_build_cte_query LEFT JOIN path (no timestamp) must quote Japanese identifiers."""
        fs = self._create_feature_store()
        fv = self._create_mock_fv('"顧客ID"', '"記録時刻"')
        fv.timestamp_col = None

        feature_columns = ['"色", "高さ"']

        query = fs._build_cte_query(
            feature_views=[fv],
            feature_columns=feature_columns,
            spine_ref="SELECT * FROM spine_table",
            spine_timestamp_col=None,
        )

        self.assertIn('"顧客ID"', query)
        self.assertIn('"色"', query)
        self.assertNotIn('""顧客ID""', query)

    def test_feature_columns_use_identifier_not_resolved(self) -> None:
        """Verify SqlIdentifier.identifier() is used for feature_columns, not resolved()."""
        col = SqlIdentifier('"色"')
        self.assertEqual(col.identifier(), '"色"')
        self.assertEqual(col.resolved(), "色")

        cols = [SqlIdentifier('"色"'), SqlIdentifier('"高さ"'), SqlIdentifier('"重さ"')]
        result = ", ".join(c.identifier() for c in cols)
        self.assertEqual(result, '"色", "高さ", "重さ"')


if __name__ == "__main__":
    absltest.main()
