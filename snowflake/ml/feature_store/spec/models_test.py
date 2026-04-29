"""Unit tests for spec.models module."""

import json

from absl.testing import absltest, parameterized

from snowflake.ml.feature_store.spec.enums import (
    FeatureViewKind,
    SourceType,
    StoreType,
    TableType,
)
from snowflake.ml.feature_store.spec.models import (
    Feature,
    FeatureViewSpec,
    FSColumn,
    Metadata,
    OfflineTableConfig,
    Source,
    Spec,
    _columns_from_struct_type,
    _make_fs_column,
    _sanitize_json_for_dollar_quoting,
    validate_schema_types,
    validate_spec_oft_offline_table_schema,
)
from snowflake.snowpark.types import (
    ArrayType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)


class MakeFSColumnTest(parameterized.TestCase):
    """Tests for _make_fs_column helper."""

    def test_string_type(self) -> None:
        col = _make_fs_column("name", StringType())
        self.assertEqual(col.name, "name")
        self.assertEqual(col.type, "StringType")
        self.assertIsNone(col.length)

    def test_string_type_with_length(self) -> None:
        col = _make_fs_column("code", StringType(10))
        self.assertEqual(col.type, "StringType")
        self.assertEqual(col.length, 10)

    def test_double_type(self) -> None:
        col = _make_fs_column("amount", DoubleType())
        self.assertEqual(col.type, "DoubleType")
        self.assertIsNone(col.precision)
        self.assertIsNone(col.scale)

    def test_long_type(self) -> None:
        col = _make_fs_column("user_id", LongType())
        self.assertEqual(col.type, "LongType")
        self.assertIsNone(col.precision)

    def test_decimal_type(self) -> None:
        col = _make_fs_column("price", DecimalType(10, 2))
        self.assertEqual(col.type, "DecimalType")
        self.assertEqual(col.precision, 10)
        self.assertEqual(col.scale, 2)

    def test_timestamp_type_ntz(self) -> None:
        col = _make_fs_column("ts", TimestampType(TimestampTimeZone.NTZ))
        self.assertEqual(col.type, "TimestampType")
        self.assertIsNone(col.timezone)

    def test_timestamp_type_tz_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_NTZ"):
            _make_fs_column("ts", TimestampType(TimestampTimeZone.TZ))

    def test_timestamp_type_ltz_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_NTZ"):
            _make_fs_column("ts", TimestampType(TimestampTimeZone.LTZ))

    def test_unsupported_type_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported type.*ArrayType"):
            _make_fs_column("arr", ArrayType(StringType()))


class ColumnsFromStructTypeTest(absltest.TestCase):
    """Tests for _columns_from_struct_type utility."""

    def test_basic_schema(self) -> None:
        schema = StructType(
            [
                StructField("USER_ID", LongType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        cols = _columns_from_struct_type(schema)
        self.assertEqual(len(cols), 3)
        self.assertEqual(cols[0].name, "USER_ID")
        self.assertEqual(cols[0].type, "LongType")
        self.assertEqual(cols[1].name, "AMOUNT")
        self.assertEqual(cols[1].type, "DoubleType")
        self.assertEqual(cols[2].name, "EVENT_TIME")
        self.assertEqual(cols[2].type, "TimestampType")

    def test_empty_schema(self) -> None:
        schema = StructType([])
        self.assertEqual(_columns_from_struct_type(schema), [])


class SourceModelTest(absltest.TestCase):
    """Tests for Source model."""

    def test_features_source_with_selected(self) -> None:
        src = Source(
            name="upstream_fv",
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="SCORE", type="FloatType")],
            source_version="v1",
        )
        d = src.model_dump(exclude_none=True)
        self.assertEqual(d["source_version"], "v1")
        self.assertEqual([c["name"] for c in d["columns"]], ["SCORE"])
        self.assertNotIn("selected_features", d)


class FeatureModelTest(absltest.TestCase):
    """Tests for spec-internal Feature model."""

    def test_aggregated_feature(self) -> None:
        feat = Feature(
            source_column=FSColumn(name="AMT", type="FloatType"),
            output_column=FSColumn(name="AMT_SUM_24H", type="FloatType"),
            function="SUM",
            window_sec=86400,
        )
        d = feat.model_dump(exclude_none=True)
        self.assertEqual(d["function"], "SUM")
        self.assertEqual(d["window_sec"], 86400)

    def test_source_name_and_version_round_trip(self) -> None:
        """FG features carry source_name + source_version through serialization."""
        feat = Feature(
            source_column=FSColumn(name="SCORE", type="DoubleType"),
            output_column=FSColumn(name="SCORE", type="DoubleType"),
            source_name="USER_FV",
            source_version="v1",
        )
        d = feat.model_dump(exclude_none=True)
        self.assertEqual(d["source_name"], "USER_FV")
        self.assertEqual(d["source_version"], "v1")
        self.assertNotIn("function", d)
        self.assertNotIn("window_sec", d)

    def test_source_name_and_version_omitempty_default(self) -> None:
        """source_name / source_version are omitted by default (Stream/Batch/RTFV)."""
        feat = Feature(
            source_column=FSColumn(name="SCORE", type="DoubleType"),
            output_column=FSColumn(name="SCORE", type="DoubleType"),
        )
        self.assertIsNone(feat.source_name)
        self.assertIsNone(feat.source_version)
        d = feat.model_dump(exclude_none=True)
        self.assertNotIn("source_name", d)
        self.assertNotIn("source_version", d)


class OfflineTableConfigTest(absltest.TestCase):
    """Tests for OfflineTableConfig model with alias."""

    def test_schema_alias(self) -> None:
        """schema_ maps to 'schema' in JSON output."""
        config = OfflineTableConfig(
            store_type=StoreType.SNOWFLAKE,
            table_type=TableType.UDF_TRANSFORMED,
            database="DB",
            schema="SCH",
            table="TBL",
            columns=[FSColumn(name="X", type="FloatType")],
        )
        d = config.model_dump(exclude_none=True, by_alias=True)
        self.assertIn("schema", d)
        self.assertNotIn("schema_", d)
        self.assertEqual(d["schema"], "SCH")

    def test_populate_by_name(self) -> None:
        """Can construct using alias 'schema' and access via field name schema_."""
        config = OfflineTableConfig(
            store_type=StoreType.SNOWFLAKE,
            table_type=TableType.TILED,
            database="DB",
            schema="SCH",
            table="TBL",
            columns=[],
        )
        self.assertEqual(config.schema_, "SCH")


class MetadataTest(absltest.TestCase):
    """Tests for Metadata model with alias."""

    def test_schema_alias(self) -> None:
        meta = Metadata(
            database="DB",
            schema="SCH",
            name="FV",
            version="v1",
            spec_format_version="1",
            internal_data_version="1",
            client_version="1.0.0",
        )
        d = meta.model_dump(by_alias=True)
        self.assertIn("schema", d)
        self.assertNotIn("schema_", d)


class FeatureViewSpecRootTest(absltest.TestCase):
    """Tests for FeatureViewSpec root model serialization."""

    def test_full_serialization(self) -> None:
        root = FeatureViewSpec(
            kind=FeatureViewKind.StreamingFeatureView,
            metadata=Metadata(
                database="DB",
                schema="SCH",
                name="FV",
                version="v1",
                spec_format_version="1",
                internal_data_version="1",
                client_version="1.0.0",
            ),
            offline_configs=[
                OfflineTableConfig(
                    store_type=StoreType.SNOWFLAKE,
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="TBL",
                    columns=[FSColumn(name="X", type="FloatType")],
                )
            ],
            spec=Spec(
                ordered_entity_column_names=["USER_ID"],
                sources=[],
                features=[],
            ),
        )
        d = root.model_dump(exclude_none=True, by_alias=True)
        self.assertEqual(d["kind"], FeatureViewKind.StreamingFeatureView)
        self.assertEqual(d["metadata"]["schema"], "SCH")
        self.assertNotIn("online_store_type", d)

    def test_online_store_type_included(self) -> None:
        """online_store_type is present in output when explicitly set."""
        root = FeatureViewSpec(
            kind=FeatureViewKind.BatchFeatureView,
            metadata=Metadata(
                database="DB",
                schema="SCH",
                name="FV",
                version="v1",
                spec_format_version="1",
                internal_data_version="1",
                client_version="1.0.0",
            ),
            offline_configs=[],
            spec=Spec(
                ordered_entity_column_names=[],
                sources=[],
                features=[],
            ),
            online_store_type=StoreType.POSTGRES,
        )
        d = root.model_dump(exclude_none=True, by_alias=True)
        self.assertEqual(d["online_store_type"], StoreType.POSTGRES)

    def test_to_dict(self) -> None:
        """to_dict() returns the same result as manual .model_dump() with correct flags."""
        root = FeatureViewSpec(
            kind=FeatureViewKind.BatchFeatureView,
            metadata=Metadata(
                database="DB",
                schema="SCH",
                name="FV",
                version="v1",
                spec_format_version="1",
                internal_data_version="1",
                client_version="1.0.0",
            ),
            offline_configs=[],
            spec=Spec(
                ordered_entity_column_names=[],
                sources=[],
                features=[],
            ),
            online_store_type=StoreType.POSTGRES,
        )
        d = root.to_dict()
        # Aliases resolved
        self.assertIn("schema", d["metadata"])
        self.assertNotIn("schema_", d["metadata"])
        # omitempty applied
        self.assertNotIn("timestamp_field", d["spec"])
        # Values correct
        self.assertEqual(d["kind"], "BatchFeatureView")
        self.assertEqual(d["online_store_type"], "postgres")

    def test_to_json(self) -> None:
        """to_json() returns a valid JSON string with aliases and omitempty."""
        import json

        root = FeatureViewSpec(
            kind=FeatureViewKind.BatchFeatureView,
            metadata=Metadata(
                database="DB",
                schema="SCH",
                name="FV",
                version="v1",
                spec_format_version="1",
                internal_data_version="1",
                client_version="1.0.0",
            ),
            offline_configs=[],
            spec=Spec(
                ordered_entity_column_names=[],
                sources=[],
                features=[],
            ),
        )
        raw = root.to_json()
        parsed = json.loads(raw)
        self.assertIn("schema", parsed["metadata"])
        self.assertNotIn("schema_", parsed["metadata"])
        self.assertNotIn("online_store_type", parsed)

    def test_to_yaml(self) -> None:
        """to_yaml() returns a valid YAML string with aliases and omitempty."""
        import yaml

        root = FeatureViewSpec(
            kind=FeatureViewKind.BatchFeatureView,
            metadata=Metadata(
                database="DB",
                schema="SCH",
                name="FV",
                version="v1",
                spec_format_version="1",
                internal_data_version="1",
                client_version="1.0.0",
            ),
            offline_configs=[],
            spec=Spec(
                ordered_entity_column_names=[],
                sources=[],
                features=[],
            ),
            online_store_type=StoreType.POSTGRES,
        )
        raw = root.to_yaml()
        parsed = yaml.safe_load(raw)
        # Aliases resolved
        self.assertIn("schema", parsed["metadata"])
        self.assertNotIn("schema_", parsed["metadata"])
        # omitempty applied
        self.assertNotIn("timestamp_field", parsed["spec"])
        # Values correct
        self.assertEqual(parsed["kind"], "BatchFeatureView")
        self.assertEqual(parsed["online_store_type"], "postgres")


# ============================================================================
# Schema Validation Tests
# ============================================================================


class ValidateSchemaTypesTest(parameterized.TestCase):
    """Tests for validate_schema_types utility."""

    def test_valid_schema_passes(self) -> None:
        """Schema with all supported types passes without error."""
        schema = StructType(
            [
                StructField("NAME", StringType()),
                StructField("ID", LongType()),
                StructField("SCORE", DoubleType()),
                StructField("PRICE", DecimalType(10, 2)),
                StructField("ACTIVE", BooleanType()),
                StructField("TS", TimestampType()),
            ]
        )
        # Should not raise
        validate_schema_types(schema)

    def test_empty_schema_passes(self) -> None:
        """Empty schema passes without error."""
        validate_schema_types(StructType([]))

    def test_unsupported_type_rejected(self) -> None:
        """Schema with unsupported type raises ValueError."""
        schema = StructType(
            [
                StructField("NAME", StringType()),
                StructField("TAGS", ArrayType(StringType())),
            ]
        )
        with self.assertRaisesRegex(ValueError, "Unsupported column types.*TAGS.*ArrayType"):
            validate_schema_types(schema)

    def test_multiple_unsupported_types_all_reported(self) -> None:
        """All unsupported columns are listed in the error, not just the first."""
        schema = StructType(
            [
                StructField("NAME", StringType()),
                StructField("TAGS", ArrayType(StringType())),
                StructField("BIRTHDAY", DateType()),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            validate_schema_types(schema)
        msg = str(ctx.exception)
        self.assertIn("TAGS", msg)
        self.assertIn("BIRTHDAY", msg)
        self.assertIn("ArrayType", msg)
        self.assertIn("DateType", msg)

    def test_date_type_rejected(self) -> None:
        """DateType is not supported."""
        schema = StructType([StructField("D", DateType())])
        with self.assertRaisesRegex(ValueError, "DateType"):
            validate_schema_types(schema)

    def test_float_type_rejected(self) -> None:
        """FloatType (Snowpark) is not supported — use DoubleType instead."""
        schema = StructType([StructField("F", FloatType())])
        with self.assertRaisesRegex(ValueError, "FloatType"):
            validate_schema_types(schema)

    def test_error_message_includes_supported_types(self) -> None:
        """Error message lists the supported types for user guidance."""
        schema = StructType([StructField("X", ArrayType(StringType()))])
        with self.assertRaises(ValueError) as ctx:
            validate_schema_types(schema)
        msg = str(ctx.exception)
        self.assertIn("Supported types:", msg)
        self.assertIn("LongType", msg)
        self.assertIn("DoubleType", msg)

    def test_timestamp_ntz_passes(self) -> None:
        """Explicit TIMESTAMP_NTZ passes validation."""
        schema = StructType([StructField("TS", TimestampType(TimestampTimeZone.NTZ))])
        validate_schema_types(schema)

    def test_timestamp_ltz_rejected(self) -> None:
        """TIMESTAMP_LTZ is rejected with guidance to cast to NTZ."""
        schema = StructType([StructField("TS", TimestampType(TimestampTimeZone.LTZ))])
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_NTZ"):
            validate_schema_types(schema)

    def test_timestamp_tz_rejected(self) -> None:
        """TIMESTAMP_TZ is rejected with guidance to cast to NTZ."""
        schema = StructType([StructField("TS", TimestampType(TimestampTimeZone.TZ))])
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_NTZ"):
            validate_schema_types(schema)


class ValidateSpecOftOfflineTableSchemaTest(absltest.TestCase):
    """``validate_spec_oft_offline_table_schema`` matches OFT offline table / FSColumn rules."""

    def test_same_rejection_as_validate_schema_types(self) -> None:
        schema = StructType([StructField("X", ArrayType(StringType()))])
        with self.assertRaises(ValueError) as ctx_pg:
            validate_spec_oft_offline_table_schema(schema)
        with self.assertRaises(ValueError) as ctx_base:
            validate_schema_types(schema)
        self.assertEqual(str(ctx_pg.exception), str(ctx_base.exception))


# ============================================================================
# Dollar-Quoting Sanitization Tests
# ============================================================================


class SanitizeJsonForDollarQuotingTest(absltest.TestCase):
    """Tests for _sanitize_json_for_dollar_quoting."""

    def test_no_special_chars(self) -> None:
        """Payload without $$ passes through unchanged."""
        payload = json.dumps({"udf_body": "def compute(x):\n    return x + 1"})
        result = _sanitize_json_for_dollar_quoting(payload)
        self.assertEqual(result, payload)
        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], "def compute(x):\n    return x + 1")

    def test_single_dollar_unchanged(self) -> None:
        """A single $ is not special and passes through."""
        payload = json.dumps({"udf_body": 'x = "$100"'})
        result = _sanitize_json_for_dollar_quoting(payload)
        self.assertEqual(result, payload)
        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], 'x = "$100"')

    def test_dollar_dollar_sanitized_and_roundtrips(self) -> None:
        """$$ in the payload is replaced; json.loads recovers original."""
        original = 'def compute():\n    return "price is $$5.00"'
        payload = json.dumps({"udf_body": original})
        result = _sanitize_json_for_dollar_quoting(payload)

        self.assertNotIn("$$", result)

        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], original)

    def test_triple_dollar_sanitized_and_roundtrips(self) -> None:
        """$$$ contains $$; must be sanitized and round-trip correctly."""
        original = 'x = "$$$"'
        payload = json.dumps({"udf_body": original})
        result = _sanitize_json_for_dollar_quoting(payload)

        self.assertNotIn("$$", result)

        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], original)

    def test_multiple_dollar_dollar_occurrences(self) -> None:
        """Multiple $$ occurrences are all sanitized."""
        original = "$$start$$ middle $$end$$"
        payload = json.dumps({"value": original})
        result = _sanitize_json_for_dollar_quoting(payload)

        self.assertNotIn("$$", result)

        parsed = json.loads(result)
        self.assertEqual(parsed["value"], original)

    def test_literal_unicode_escape_in_source_no_collision(self) -> None:
        r"""Source containing literal \u0024 must not collide with sanitization."""
        original = 'x = "\\u0024"'
        payload = json.dumps({"udf_body": original})
        result = _sanitize_json_for_dollar_quoting(payload)

        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], original)

    def test_empty_string_payload(self) -> None:
        """Empty JSON string value."""
        payload = json.dumps({"udf_body": ""})
        result = _sanitize_json_for_dollar_quoting(payload)
        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], "")

    def test_unicode_characters(self) -> None:
        """Unicode content passes through without issue."""
        original = "def compute():\n    return '日本語'"
        payload = json.dumps({"udf_body": original}, ensure_ascii=False)
        result = _sanitize_json_for_dollar_quoting(payload)
        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], original)

    def test_backslashes_in_source(self) -> None:
        r"""Backslashes (e.g., regex patterns) round-trip correctly."""
        original = "import re\npattern = r'\\d+'"
        payload = json.dumps({"udf_body": original})
        result = _sanitize_json_for_dollar_quoting(payload)
        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], original)

    def test_quotes_in_source(self) -> None:
        """Single and double quotes round-trip correctly."""
        original = 'def compute():\n    return "it\'s done"'
        payload = json.dumps({"udf_body": original})
        result = _sanitize_json_for_dollar_quoting(payload)
        parsed = json.loads(result)
        self.assertEqual(parsed["udf_body"], original)


if __name__ == "__main__":
    absltest.main()
