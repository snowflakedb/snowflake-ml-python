"""Unit tests for stream_source module."""

from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.stream_source import (
    _STREAM_SOURCE_NAME_LENGTH_LIMIT,
    StreamSource,
    _schema_from_dict,
    _schema_to_dict,
)
from snowflake.snowpark.types import (
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
    TimeType,
)


class StreamSourceValidationTest(parameterized.TestCase):
    """Unit tests for StreamSource validation logic."""

    def _default_schema(self) -> StructType:
        return StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", FloatType()),
                StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
            ]
        )

    def test_valid_construction(self) -> None:
        """Test that a valid StreamSource can be created."""
        ss = StreamSource(
            name="txn_events",
            schema=self._default_schema(),
            desc="Test stream source",
        )
        self.assertEqual(ss.name, SqlIdentifier("txn_events"))
        self.assertEqual(ss.desc, "Test stream source")
        self.assertIsNone(ss.owner)

    def test_name_is_sql_identifier(self) -> None:
        """Test that name is stored as SqlIdentifier and uppercased."""
        ss = StreamSource("my_source", self._default_schema())
        self.assertIsInstance(ss.name, SqlIdentifier)
        self.assertEqual(ss.name.resolved(), "MY_SOURCE")

    def test_name_exceeds_length_limit(self) -> None:
        """Test that name exceeding the length limit raises ValueError."""
        long_name = "a" * (_STREAM_SOURCE_NAME_LENGTH_LIMIT + 1)
        with self.assertRaisesRegex(ValueError, "exceeds maximum length"):
            StreamSource(long_name, self._default_schema())

    def test_name_at_length_limit(self) -> None:
        """Test that name exactly at the length limit is accepted."""
        name = "a" * _STREAM_SOURCE_NAME_LENGTH_LIMIT
        ss = StreamSource(name, self._default_schema())
        self.assertIsNotNone(ss)

    def test_empty_schema_raises(self) -> None:
        """Test that an empty schema raises ValueError."""
        with self.assertRaisesRegex(ValueError, "at least one field"):
            StreamSource("src", StructType([]))

    @parameterized.parameters(  # type: ignore[misc]
        StringType,
        FloatType,
        DecimalType,
        BooleanType,
        TimestampType,
        DateType,
        TimeType,
    )
    def test_supported_scalar_types(self, type_cls: type) -> None:
        """Test that all supported scalar types are accepted in schema."""
        schema = StructType(
            [
                StructField("col", type_cls()),
            ]
        )
        ss = StreamSource("src", schema)
        self.assertIsNotNone(ss)

    def test_unsupported_type_raises(self) -> None:
        """Test that an unsupported type in schema raises ValueError."""
        from snowflake.snowpark.types import ArrayType

        schema = StructType(
            [
                StructField("bad_col", ArrayType(StringType())),
            ]
        )
        with self.assertRaisesRegex(ValueError, "Unsupported type"):
            StreamSource("src", schema)

    def test_timestamp_ntz_accepted(self) -> None:
        """Test that TimestampType(NTZ) is accepted."""
        schema = StructType([StructField("ts", TimestampType(TimestampTimeZone.NTZ))])
        ss = StreamSource("src", schema)
        self.assertIsNotNone(ss)

    def test_timestamp_default_accepted(self) -> None:
        """Test that TimestampType() (DEFAULT) is accepted."""
        schema = StructType([StructField("ts", TimestampType())])
        ss = StreamSource("src", schema)
        self.assertIsNotNone(ss)

    @parameterized.parameters(TimestampTimeZone.LTZ, TimestampTimeZone.TZ)  # type: ignore[misc]
    def test_timestamp_ltz_tz_rejected(self, tz: TimestampTimeZone) -> None:
        """Test that TimestampType with LTZ or TZ raises ValueError."""
        schema = StructType([StructField("ts", TimestampType(tz))])
        with self.assertRaisesRegex(ValueError, "Only TIMESTAMP_NTZ is supported"):
            StreamSource("src", schema)

    def test_desc_defaults_to_empty(self) -> None:
        """Test that desc defaults to empty string."""
        ss = StreamSource("src", self._default_schema())
        self.assertEqual(ss.desc, "")


class StreamSourceSerializationTest(absltest.TestCase):
    """Unit tests for StreamSource serialization/deserialization."""

    def _default_schema(self) -> StructType:
        return StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", FloatType()),
                StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
            ]
        )

    def test_to_dict_structure(self) -> None:
        """Test that _to_dict produces the expected structure."""
        ss = StreamSource("txn", self._default_schema(), desc="Test")
        ss.owner = "ROLE_1"
        d = ss._to_dict()

        self.assertEqual(d["name"], "TXN")
        self.assertEqual(d["desc"], "Test")
        self.assertEqual(d["owner"], "ROLE_1")
        self.assertIsInstance(d["schema"], list)
        self.assertEqual(len(d["schema"]), 3)
        self.assertNotIn("timestamp_col", d)

    def test_to_dict_owner_none(self) -> None:
        """Test that _to_dict handles None owner."""
        ss = StreamSource("txn", self._default_schema())
        d = ss._to_dict()
        self.assertEqual(d["owner"], "")

    def test_roundtrip_without_owner(self) -> None:
        """Test serialization roundtrip with owner=None."""
        ss = StreamSource("txn", self._default_schema(), desc="Test")
        d = ss._to_dict()
        ss2 = StreamSource._from_dict(d)
        self.assertEqual(ss, ss2)

    def test_roundtrip_with_owner(self) -> None:
        """Test serialization roundtrip with owner set."""
        ss = StreamSource("txn", self._default_schema(), desc="Test")
        ss.owner = "ROLE_1"
        d = ss._to_dict()
        ss2 = StreamSource._from_dict(d)
        self.assertEqual(ss, ss2)
        self.assertEqual(ss2.owner, "ROLE_1")

    def test_from_dict_missing_owner(self) -> None:
        """Test _from_dict handles missing owner gracefully."""
        d = {
            "name": "SRC",
            "schema": [{"name": "ts", "type": "TimestampType"}],
            "desc": "",
        }
        ss = StreamSource._from_dict(d)
        self.assertIsNone(ss.owner)

    def test_roundtrip_unquoted_name_uppercased(self) -> None:
        """Test that unquoted stream source name is uppercased and preserved through roundtrip."""
        ss = StreamSource("mySource", self._default_schema(), desc="Test")
        d = ss._to_dict()
        # Unquoted identifiers are uppercased by SqlIdentifier
        self.assertEqual(d["name"], "MYSOURCE")

        ss2 = StreamSource._from_dict(d)
        self.assertEqual(ss, ss2)
        self.assertEqual(ss2.name.resolved(), "MYSOURCE")

    def test_roundtrip_quoted_name_preserves_case(self) -> None:
        """Test that quoted stream source name preserves mixed case through roundtrip."""
        ss = StreamSource('"myMixedCase"', self._default_schema(), desc="Test")
        d = ss._to_dict()
        # Quoted identifiers keep the quotes and preserve case
        self.assertIn("myMixedCase", d["name"])

        ss2 = StreamSource._from_dict(d)
        self.assertEqual(ss, ss2)
        self.assertEqual(ss2.name.resolved(), "myMixedCase")

    def test_roundtrip_column_name_case_preserved(self) -> None:
        """Test that column names survive StreamSource _to_dict/_from_dict roundtrip.

        Snowpark's StructField uppercases unquoted names, so mixed-case inputs
        become uppercase before our code sees them. We verify that the uppercased
        names are preserved faithfully through the roundtrip.
        """
        schema = StructType(
            [
                StructField("userId", StringType()),
                StructField("EventTime", TimestampType()),
            ]
        )
        ss = StreamSource("src", schema)
        d = ss._to_dict()
        ss2 = StreamSource._from_dict(d)

        # StructField uppercases the names; roundtrip preserves them
        self.assertEqual(ss2.schema.fields[0].name, "USERID")
        self.assertEqual(ss2.schema.fields[1].name, "EVENTTIME")

    def test_to_dict_name_preserves_quoted_format(self) -> None:
        """Test that _to_dict stores the str(SqlIdentifier) which preserves quoting for reconstruction."""
        # Unquoted: str(SqlIdentifier("mySource")) = "MYSOURCE"
        ss_unquoted = StreamSource("mySource", self._default_schema())
        d1 = ss_unquoted._to_dict()
        self.assertEqual(d1["name"], "MYSOURCE")

        # Quoted: str(SqlIdentifier('"myMixed"')) = '"myMixed"' (includes quote chars)
        ss_quoted = StreamSource('"myMixed"', self._default_schema())
        d2 = ss_quoted._to_dict()
        self.assertEqual(d2["name"], '"myMixed"')

        # Roundtrip of quoted name: _from_dict reconstructs the SqlIdentifier correctly
        ss_reconstructed = StreamSource._from_dict(d2)
        self.assertEqual(ss_reconstructed.name.resolved(), "myMixed")

    def test_construct_stream_source(self) -> None:
        """Test _construct_stream_source factory method."""
        schema = self._default_schema()
        ss = StreamSource._construct_stream_source(
            name="SRC",
            schema=schema,
            desc="desc",
            owner="ADMIN",
        )
        self.assertEqual(ss.name, SqlIdentifier("SRC"))
        self.assertEqual(ss.owner, "ADMIN")
        self.assertEqual(ss.desc, "desc")


class SchemaSerializationTest(parameterized.TestCase):
    """Unit tests for _schema_to_dict and _schema_from_dict."""

    @parameterized.parameters(  # type: ignore[misc]
        ("col", StringType()),
        ("col", FloatType()),
        ("col", DecimalType()),
        ("col", BooleanType()),
        ("col", TimestampType(TimestampTimeZone.NTZ)),
        ("col", DateType()),
        ("col", TimeType()),
    )
    def test_roundtrip_single_type(self, col_name: str, dtype: DataType) -> None:
        """Test schema roundtrip for each supported type."""
        schema = StructType([StructField(col_name, dtype)])
        d = _schema_to_dict(schema)
        schema2 = _schema_from_dict(d)
        self.assertEqual(schema, schema2)

    def test_roundtrip_decimal_with_precision(self) -> None:
        """Test DecimalType preserves precision and scale through roundtrip."""
        schema = StructType([StructField("amount", DecimalType(10, 2))])
        d = _schema_to_dict(schema)
        self.assertEqual(d[0]["precision"], 10)
        self.assertEqual(d[0]["scale"], 2)

        schema2 = _schema_from_dict(d)
        self.assertEqual(schema, schema2)
        self.assertEqual(schema2.fields[0].datatype.precision, 10)
        self.assertEqual(schema2.fields[0].datatype.scale, 2)

    def test_roundtrip_string_with_length(self) -> None:
        """Test StringType preserves length through roundtrip."""
        schema = StructType([StructField("name", StringType(50))])
        d = _schema_to_dict(schema)
        self.assertEqual(d[0]["length"], 50)

        schema2 = _schema_from_dict(d)
        self.assertEqual(schema, schema2)
        self.assertEqual(schema2.fields[0].datatype.length, 50)

    def test_roundtrip_string_without_length(self) -> None:
        """Test StringType without explicit length (max/unlimited) roundtrips correctly."""
        schema = StructType([StructField("name", StringType())])
        d = _schema_to_dict(schema)
        self.assertNotIn("length", d[0])

        schema2 = _schema_from_dict(d)
        self.assertEqual(schema, schema2)
        self.assertIsNone(schema2.fields[0].datatype.length)

    def test_roundtrip_timestamp_ntz_explicit(self) -> None:
        """Test TimestampType(NTZ) roundtrips correctly and no 'tz' is stored."""
        schema = StructType([StructField("ts", TimestampType(TimestampTimeZone.NTZ))])
        d = _schema_to_dict(schema)
        self.assertNotIn("tz", d[0])

        schema2 = _schema_from_dict(d)
        self.assertEqual(schema2.fields[0].datatype.tz, TimestampTimeZone.NTZ)
        self.assertEqual(schema, schema2)

    def test_roundtrip_timestamp_default_normalized_to_ntz(self) -> None:
        """Test TimestampType() (DEFAULT) is deserialized as NTZ."""
        schema = StructType([StructField("ts", TimestampType())])
        d = _schema_to_dict(schema)
        self.assertNotIn("tz", d[0])

        schema2 = _schema_from_dict(d)
        # DEFAULT is normalized to NTZ on deserialization
        self.assertEqual(schema2.fields[0].datatype.tz, TimestampTimeZone.NTZ)

    def test_from_dict_legacy_tz_value_ignored(self) -> None:
        """Test _schema_from_dict ignores any stored 'tz' value (backward compatibility)."""
        # Legacy data may have "tz": "default" or "tz": "ntz"
        for tz_val in ["default", "ntz", "ltz"]:
            d = [{"name": "TS", "type": "TimestampType", "tz": tz_val}]
            schema = _schema_from_dict(d)
            self.assertEqual(schema.fields[0].datatype.tz, TimestampTimeZone.NTZ)

    def test_from_dict_missing_tz_defaults_to_ntz(self) -> None:
        """Test _schema_from_dict handles missing 'tz' key (backward compatibility)."""
        d = [{"name": "TS", "type": "TimestampType"}]
        schema = _schema_from_dict(d)
        self.assertEqual(schema.fields[0].datatype.tz, TimestampTimeZone.NTZ)

    def test_from_dict_missing_length_defaults_to_none(self) -> None:
        """Test _schema_from_dict handles missing 'length' key (backward compatibility)."""
        d = [{"name": "NAME", "type": "StringType"}]
        schema = _schema_from_dict(d)
        self.assertIsNone(schema.fields[0].datatype.length)

    def test_roundtrip_multi_field(self) -> None:
        """Test schema roundtrip with multiple fields."""
        schema = StructType(
            [
                StructField("A", StringType()),
                StructField("B", FloatType()),
                StructField("C", TimestampType(TimestampTimeZone.NTZ)),
                StructField("D", BooleanType()),
            ]
        )
        d = _schema_to_dict(schema)
        schema2 = _schema_from_dict(d)
        self.assertEqual(schema, schema2)

    def test_column_name_uppercased_by_snowpark(self) -> None:
        """Test that column names are uppercased by Snowpark's StructField and preserved through roundtrip.

        Snowpark's StructField normalizes unquoted column names to uppercase,
        matching Snowflake SQL's identifier behavior. Our serialization faithfully
        preserves whatever StructField.name gives us.
        """
        schema = StructType(
            [
                StructField("userId", StringType()),
                StructField("EventTime", TimestampType()),
                StructField("AMOUNT", FloatType()),
                StructField("is_active", BooleanType()),
            ]
        )

        # StructField uppercases names (Snowflake SQL identifier behavior)
        self.assertEqual(schema.fields[0].name, "USERID")
        self.assertEqual(schema.fields[1].name, "EVENTTIME")
        self.assertEqual(schema.fields[2].name, "AMOUNT")
        self.assertEqual(schema.fields[3].name, "IS_ACTIVE")

        # Verify our serialization preserves the (uppercased) names through roundtrip
        d = _schema_to_dict(schema)
        self.assertEqual(d[0]["name"], "USERID")
        self.assertEqual(d[1]["name"], "EVENTTIME")
        self.assertEqual(d[2]["name"], "AMOUNT")
        self.assertEqual(d[3]["name"], "IS_ACTIVE")

        schema2 = _schema_from_dict(d)
        self.assertEqual(schema2.fields[0].name, "USERID")
        self.assertEqual(schema2.fields[1].name, "EVENTTIME")
        self.assertEqual(schema2.fields[2].name, "AMOUNT")
        self.assertEqual(schema2.fields[3].name, "IS_ACTIVE")

    def test_to_dict_unsupported_type_raises(self) -> None:
        """Test that _schema_to_dict raises for unsupported types."""
        from snowflake.snowpark.types import ArrayType

        schema = StructType([StructField("bad", ArrayType(StringType()))])
        with self.assertRaisesRegex(ValueError, "Unsupported type"):
            _schema_to_dict(schema)

    def test_from_dict_unsupported_type_raises(self) -> None:
        """Test that _schema_from_dict raises for unsupported types."""
        d = [{"name": "bad", "type": "MapType"}]
        with self.assertRaisesRegex(ValueError, "Unsupported type"):
            _schema_from_dict(d)

    def test_to_dict_field_structure(self) -> None:
        """Test the structure of each field dict."""
        schema = StructType([StructField("MY_COL", FloatType())])
        d = _schema_to_dict(schema)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["name"], "MY_COL")
        self.assertEqual(d[0]["type"], "FloatType")
        self.assertNotIn("nullable", d[0])


class StreamSourceEqualityTest(absltest.TestCase):
    """Unit tests for StreamSource __eq__ and __repr__."""

    def _default_schema(self) -> StructType:
        return StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
            ]
        )

    def test_equal_instances(self) -> None:
        """Test that two identical StreamSources are equal."""
        ss1 = StreamSource("src", self._default_schema(), desc="d")
        ss2 = StreamSource("src", self._default_schema(), desc="d")
        self.assertEqual(ss1, ss2)

    def test_not_equal_different_name(self) -> None:
        """Test that StreamSources with different names are not equal."""
        ss1 = StreamSource("src1", self._default_schema())
        ss2 = StreamSource("src2", self._default_schema())
        self.assertNotEqual(ss1, ss2)

    def test_not_equal_different_schema(self) -> None:
        """Test that StreamSources with different schemas are not equal."""
        schema1 = StructType(
            [
                StructField("A", StringType()),
                StructField("TS", TimestampType(TimestampTimeZone.NTZ)),
            ]
        )
        schema2 = StructType(
            [
                StructField("B", FloatType()),
                StructField("TS", TimestampType(TimestampTimeZone.NTZ)),
            ]
        )
        ss1 = StreamSource("src", schema1)
        ss2 = StreamSource("src", schema2)
        self.assertNotEqual(ss1, ss2)

    def test_not_equal_different_desc(self) -> None:
        """Test that StreamSources with different desc are not equal."""
        ss1 = StreamSource("src", self._default_schema(), desc="a")
        ss2 = StreamSource("src", self._default_schema(), desc="b")
        self.assertNotEqual(ss1, ss2)

    def test_not_equal_different_owner(self) -> None:
        """Test that StreamSources with different owners are not equal."""
        ss1 = StreamSource("src", self._default_schema())
        ss1.owner = "ROLE_A"
        ss2 = StreamSource("src", self._default_schema())
        ss2.owner = "ROLE_B"
        self.assertNotEqual(ss1, ss2)

    def test_not_equal_to_non_stream_source(self) -> None:
        """Test that StreamSource is not equal to non-StreamSource objects."""
        ss = StreamSource("src", self._default_schema())
        self.assertNotEqual(ss, "not a stream source")
        self.assertNotEqual(ss, 42)
        self.assertNotEqual(ss, None)

    def test_case_insensitive_name_equality(self) -> None:
        """Test that names are compared case-insensitively (via SqlIdentifier)."""
        ss1 = StreamSource("src", self._default_schema())
        ss2 = StreamSource("SRC", self._default_schema())
        self.assertEqual(ss1, ss2)

    def test_repr(self) -> None:
        """Test that __repr__ returns a useful string."""
        ss = StreamSource("src", self._default_schema(), desc="Test")
        r = repr(ss)
        self.assertIn("StreamSource(", r)
        self.assertIn("name=", r)
        self.assertIn("schema=", r)


if __name__ == "__main__":
    absltest.main()
