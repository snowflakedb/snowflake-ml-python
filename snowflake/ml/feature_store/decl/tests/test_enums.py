"""Tests for decl/enums.py — FSBaseType, TYPE_ALIASES, normalize_type()."""

from snowflake.ml.feature_store.decl.enums import (
    TYPE_ALIASES,
    FeatureAggregationMethod,
    FeatureViewKind,
    FSBaseType,
    OpKind,
    SourceType,
    normalize_type,
)
from snowflake.ml.test_utils import pytest_driver


class TestFSBaseType:
    def test_all_values(self) -> None:
        assert FSBaseType.StringType == "StringType"
        assert FSBaseType.LongType == "LongType"
        assert FSBaseType.DoubleType == "DoubleType"
        assert FSBaseType.DecimalType == "DecimalType"
        assert FSBaseType.BooleanType == "BooleanType"
        assert FSBaseType.BinaryType == "BinaryType"
        assert FSBaseType.TimestampType == "TimestampType"

    def test_is_string_subclass(self) -> None:
        for member in FSBaseType:
            assert isinstance(member, str)

    def test_seven_members(self) -> None:
        # ``BinaryType`` was added to align FSBaseType with
        # ``spec.models._SUPPORTED_TYPES`` and ``stream_source._TYPE_NAME_TO_CLASS``.
        assert len(list(FSBaseType)) == 7


class TestFeatureViewKind:
    def test_all_values(self) -> None:
        assert FeatureViewKind.StreamingFeatureView == "StreamingFeatureView"
        assert FeatureViewKind.RealtimeFeatureView == "RealtimeFeatureView"
        assert FeatureViewKind.BatchFeatureView == "BatchFeatureView"
        assert FeatureViewKind.FeatureGroup == "FeatureGroup"

    def test_four_members(self) -> None:
        # ``FeatureGroup`` joined the spec.enums vocabulary on this branch.
        assert len(list(FeatureViewKind)) == 4


class TestSourceType:
    def test_canonical_values(self) -> None:
        # Member names follow spec.enums (UPPERCASE).
        assert SourceType.STREAM.value == "Stream"
        assert SourceType.REQUEST.value == "Request"
        assert SourceType.FEATURES.value == "Features"
        assert SourceType.BATCH.value == "Batch"

    def test_no_sqldatasource(self) -> None:
        # ``SQLDataSource`` was removed during the decl/spec consolidation.
        assert not hasattr(SourceType, "SQLDataSource")

    def test_no_legacy_batchsource_member(self) -> None:
        # The PascalCase legacy member name ``BatchSource`` is gone; the
        # canonical value is ``Batch`` and the member name is ``BATCH``.
        assert not hasattr(SourceType, "BatchSource")


class TestFeatureAggregationMethod:
    def test_canonical_values(self) -> None:
        # Member names follow spec.enums (UPPERCASE); values are unchanged.
        assert FeatureAggregationMethod.TILES.value == "tiles"
        assert FeatureAggregationMethod.CONTINUOUS.value == "continuous"


class TestOpKind:
    def test_create_ops(self) -> None:
        assert OpKind.CREATE_ENTITY == "CREATE_ENTITY"
        assert OpKind.CREATE_SOURCE == "CREATE_SOURCE"
        assert OpKind.CREATE_FV == "CREATE_FV"
        assert OpKind.CREATE_FG == "CREATE_FG"

    def test_update_ops(self) -> None:
        assert OpKind.UPDATE_SOURCE == "UPDATE_SOURCE"
        assert OpKind.UPDATE_FV == "UPDATE_FV"
        assert OpKind.RECREATE_FV == "RECREATE_FV"

    def test_drop_ops(self) -> None:
        assert OpKind.DROP_ENTITY == "DROP_ENTITY"
        assert OpKind.DROP_SOURCE == "DROP_SOURCE"
        assert OpKind.DROP_FV == "DROP_FV"
        assert OpKind.DROP_FG == "DROP_FG"


class TestTypeAliases:
    def test_string_aliases(self) -> None:
        assert TYPE_ALIASES["str"] == "StringType"
        assert TYPE_ALIASES["string"] == "StringType"

    def test_integer_aliases(self) -> None:
        assert TYPE_ALIASES["int"] == "LongType"
        assert TYPE_ALIASES["integer"] == "LongType"

    def test_float_aliases(self) -> None:
        assert TYPE_ALIASES["float"] == "DoubleType"
        assert TYPE_ALIASES["number"] == "DoubleType"

    def test_decimal_alias(self) -> None:
        assert TYPE_ALIASES["decimal"] == "DecimalType"

    def test_boolean_aliases(self) -> None:
        assert TYPE_ALIASES["bool"] == "BooleanType"
        assert TYPE_ALIASES["boolean"] == "BooleanType"

    def test_binary_aliases(self) -> None:
        assert TYPE_ALIASES["binary"] == "BinaryType"
        assert TYPE_ALIASES["bytes"] == "BinaryType"

    def test_timestamp_aliases(self) -> None:
        assert TYPE_ALIASES["datetime"] == "TimestampType"
        assert TYPE_ALIASES["timestamp"] == "TimestampType"

    def test_all_values_are_fsbasetype_values(self) -> None:
        canonical_values = {t.value for t in FSBaseType}
        for alias, canonical in TYPE_ALIASES.items():
            assert canonical in canonical_values, f"Alias '{alias}' maps to unknown '{canonical}'"

    def test_alias_count(self) -> None:
        from snowflake.ml.feature_store.decl.enums import TYPE_ALIASES

        # 13 pre-existing aliases + ``binary`` + ``bytes`` for the new
        # ``BinaryType`` member; bumps to 15 once BinaryType is supported.
        assert len(TYPE_ALIASES) == 15


class TestNormalizeType:
    def test_passthrough_canonical_names(self) -> None:
        for member in FSBaseType:
            assert normalize_type(member.value) == member.value

    def test_str_alias(self) -> None:
        assert normalize_type("str") == "StringType"

    def test_string_alias(self) -> None:
        assert normalize_type("string") == "StringType"

    def test_int_alias(self) -> None:
        assert normalize_type("int") == "LongType"

    def test_integer_alias(self) -> None:
        assert normalize_type("integer") == "LongType"

    def test_float_alias(self) -> None:
        assert normalize_type("float") == "DoubleType"

    def test_number_alias(self) -> None:
        assert normalize_type("number") == "DoubleType"

    def test_decimal_alias(self) -> None:
        assert normalize_type("decimal") == "DecimalType"

    def test_bool_alias(self) -> None:
        assert normalize_type("bool") == "BooleanType"

    def test_boolean_alias(self) -> None:
        assert normalize_type("boolean") == "BooleanType"

    def test_binary_alias(self) -> None:
        assert normalize_type("binary") == "BinaryType"

    def test_bytes_alias(self) -> None:
        assert normalize_type("bytes") == "BinaryType"

    def test_datetime_alias(self) -> None:
        assert normalize_type("datetime") == "TimestampType"

    def test_timestamp_alias(self) -> None:
        assert normalize_type("timestamp") == "TimestampType"

    def test_case_insensitive_aliases(self) -> None:
        assert normalize_type("STR") == "StringType"
        assert normalize_type("INT") == "LongType"
        assert normalize_type("FLOAT") == "DoubleType"
        assert normalize_type("BOOL") == "BooleanType"

    def test_unknown_type_passthrough(self) -> None:
        assert normalize_type("MyCustomType") == "MyCustomType"
        assert normalize_type("unknown_xyz") == "unknown_xyz"


if __name__ == "__main__":
    pytest_driver.main()
