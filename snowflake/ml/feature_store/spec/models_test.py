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
)
from snowflake.snowpark.types import (
    ArrayType,
    DecimalType,
    FloatType,
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

    def test_float_type(self) -> None:
        col = _make_fs_column("amount", FloatType())
        self.assertEqual(col.type, "FloatType")
        self.assertIsNone(col.precision)
        self.assertIsNone(col.scale)

    def test_decimal_type(self) -> None:
        col = _make_fs_column("price", DecimalType(10, 2))
        self.assertEqual(col.type, "DecimalType")
        self.assertEqual(col.precision, 10)
        self.assertEqual(col.scale, 2)

    def test_timestamp_type_ntz(self) -> None:
        col = _make_fs_column("ts", TimestampType(TimestampTimeZone.NTZ))
        self.assertEqual(col.type, "TimestampType")
        self.assertIsNone(col.timezone)

    def test_timestamp_type_with_tz(self) -> None:
        col = _make_fs_column("ts", TimestampType(TimestampTimeZone.TZ))
        self.assertEqual(col.type, "TimestampType")
        self.assertEqual(col.timezone, str(TimestampTimeZone.TZ))

    def test_unsupported_type_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported type.*ArrayType"):
            _make_fs_column("arr", ArrayType(StringType()))


class ColumnsFromStructTypeTest(absltest.TestCase):
    """Tests for _columns_from_struct_type utility."""

    def test_basic_schema(self) -> None:
        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", FloatType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        cols = _columns_from_struct_type(schema)
        self.assertEqual(len(cols), 3)
        self.assertEqual(cols[0].name, "USER_ID")
        self.assertEqual(cols[0].type, "StringType")
        self.assertEqual(cols[1].name, "AMOUNT")
        self.assertEqual(cols[1].type, "FloatType")
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
            selected_features=["SCORE"],
        )
        d = src.dict(exclude_none=True)  # type: ignore[deprecation]
        self.assertEqual(d["source_version"], "v1")
        self.assertEqual(d["selected_features"], ["SCORE"])


class FeatureModelTest(absltest.TestCase):
    """Tests for spec-internal Feature model."""

    def test_aggregated_feature(self) -> None:
        feat = Feature(
            source_column=FSColumn(name="AMT", type="FloatType"),
            output_column=FSColumn(name="AMT_SUM_24H", type="FloatType"),
            function="SUM",
            window_sec=86400,
        )
        d = feat.dict(exclude_none=True)  # type: ignore[deprecation]
        self.assertEqual(d["function"], "SUM")
        self.assertEqual(d["window_sec"], 86400)


class OfflineTableConfigTest(absltest.TestCase):
    """Tests for OfflineTableConfig model with alias."""

    def test_schema_alias(self) -> None:
        """schema_ maps to 'schema' in JSON output."""
        config = OfflineTableConfig(
            store_type=StoreType.SNOWFLAKE,
            table_type=TableType.UDF_TRANSFORMED,
            database="DB",
            schema_="SCH",
            table="TBL",
            columns=[FSColumn(name="X", type="FloatType")],
        )
        d = config.dict(exclude_none=True, by_alias=True)  # type: ignore[deprecation]
        self.assertIn("schema", d)
        self.assertNotIn("schema_", d)
        self.assertEqual(d["schema"], "SCH")

    def test_populate_by_name(self) -> None:
        """Can construct using Pythonic name schema_."""
        config = OfflineTableConfig(
            store_type=StoreType.SNOWFLAKE,
            table_type=TableType.TILED,
            database="DB",
            schema_="SCH",
            table="TBL",
            columns=[],
        )
        self.assertEqual(config.schema_, "SCH")


class MetadataTest(absltest.TestCase):
    """Tests for Metadata model with alias."""

    def test_schema_alias(self) -> None:
        meta = Metadata(
            database="DB",
            schema_="SCH",
            name="FV",
            version="v1",
            spec_format_version="1",
            internal_data_version="1",
            client_version="1.0.0",
        )
        d = meta.dict(by_alias=True)  # type: ignore[deprecation]
        self.assertIn("schema", d)
        self.assertNotIn("schema_", d)


class FeatureViewSpecRootTest(absltest.TestCase):
    """Tests for FeatureViewSpec root model serialization."""

    def test_full_serialization(self) -> None:
        root = FeatureViewSpec(
            kind=FeatureViewKind.StreamingFeatureView,
            metadata=Metadata(
                database="DB",
                schema_="SCH",
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
                    schema_="SCH",
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
        d = root.dict(exclude_none=True, by_alias=True)  # type: ignore[deprecation]
        self.assertEqual(d["kind"], FeatureViewKind.StreamingFeatureView)
        self.assertEqual(d["metadata"]["schema"], "SCH")
        self.assertNotIn("online_store_type", d)

    def test_online_store_type_included(self) -> None:
        """online_store_type is present in output when explicitly set."""
        root = FeatureViewSpec(
            kind=FeatureViewKind.BatchFeatureView,
            metadata=Metadata(
                database="DB",
                schema_="SCH",
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
        d = root.dict(exclude_none=True, by_alias=True)  # type: ignore[deprecation]
        self.assertEqual(d["online_store_type"], StoreType.POSTGRES)

    def test_to_dict(self) -> None:
        """to_dict() returns the same result as manual .dict() with correct flags."""
        root = FeatureViewSpec(
            kind=FeatureViewKind.BatchFeatureView,
            metadata=Metadata(
                database="DB",
                schema_="SCH",
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
                schema_="SCH",
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
                schema_="SCH",
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
