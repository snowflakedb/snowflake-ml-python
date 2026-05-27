"""Unit tests for spec.table_schema_evolution module."""

from typing import Any

from absl.testing import absltest, parameterized

from snowflake.ml.feature_store.spec.models import _SUPPORTED_TYPES
from snowflake.ml.feature_store.spec.table_schema_evolution import (
    get_table_schema_evolution_extend_only_commands,
)
from snowflake.snowpark.types import (
    ArrayType,
    BooleanType,
    DataType,
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    TimestampTimeZone,
    TimestampType,
)

_TABLE = "DB.SCH.T"

# Sample instances for identity evolution tests — must define one per entry in ``_SUPPORTED_TYPES``.
_IDENTITY_EVOLUTION_SAMPLES: dict[type, DataType] = {
    BooleanType: BooleanType(),
    DecimalType: DecimalType(10, 2),
    DoubleType: DoubleType(),
    LongType: LongType(),
    StringType: StringType(8),
    TimestampType: TimestampType(TimestampTimeZone.NTZ),
}

# (case_name, old_schema, new_schema, expected_forward, expected_rollback) — call must succeed
# and return the listed DDL commands (forward = ADD COLUMN in declaration order; rollback = LIFO).
_SCHEMA_EVOLUTION_SUCCESS_CASES: tuple[
    tuple[str, dict[str, DataType], dict[str, DataType], list[str], list[str]],
    ...,
] = (
    (
        "append_one_column",
        {"X": StringType()},
        {"X": StringType(), "Y": LongType()},
        [f'ALTER TABLE {_TABLE} ADD COLUMN "Y" NUMBER(38,0)'],
        [f'ALTER TABLE {_TABLE} DROP COLUMN "Y"'],
    ),
    (
        "append_multiple_columns",
        {"X": StringType()},
        {"X": StringType(), "Y": BooleanType(), "Z": DoubleType()},
        [
            f'ALTER TABLE {_TABLE} ADD COLUMN "Y" BOOLEAN',
            f'ALTER TABLE {_TABLE} ADD COLUMN "Z" FLOAT',
        ],
        [
            f'ALTER TABLE {_TABLE} DROP COLUMN "Z"',
            f'ALTER TABLE {_TABLE} DROP COLUMN "Y"',
        ],
    ),
)

# (case_name, old_schema, new_schema, exception_type, message_regex)
_SCHEMA_EVOLUTION_FAILURE_CASES: tuple[
    tuple[str, dict[str, DataType], dict[str, DataType], type[Exception], str],
    ...,
] = (
    (
        "unsupported_type_in_old_schema",
        {"X": ArrayType(StringType())},
        {"X": StringType()},
        ValueError,
        r"Existing table schema.*ArrayType",
    ),
    (
        "unsupported_type_in_new_schema",
        {"X": StringType()},
        {"X": StringType(), "Y": ArrayType(StringType())},
        ValueError,
        r"New source schema.*ArrayType",
    ),
    (
        "dropped_column",
        {"OLD": StringType(), "KEEP": LongType()},
        {"KEEP": LongType()},
        ValueError,
        r"extend-only evolution cannot drop columns",
    ),
    (
        "reordered_columns",
        {"A": StringType(), "B": LongType()},
        {"B": LongType(), "A": StringType()},
        ValueError,
        r"Column at position 0 mismatch",
    ),
    (
        "drop_middle_column_and_append_new",
        {
            "ENTITY": StringType(),
            "TS": TimestampType(TimestampTimeZone.NTZ),
            "A": LongType(),
            "B": LongType(),
            "C": LongType(),
        },
        {
            "ENTITY": StringType(),
            "TS": TimestampType(TimestampTimeZone.NTZ),
            "A": LongType(),
            "C": LongType(),
            "D": LongType(),
        },
        ValueError,
        r"Column at position 3 mismatch: old='B', new='C'",
    ),
    (
        "old_column_replaced_with_new_at_same_position",
        {"OLD": StringType()},
        {"NEW": StringType()},
        ValueError,
        r"Column at position 0 mismatch",
    ),
    (
        "string_widening_rejected",
        {"X": StringType(10)},
        {"X": StringType(20)},
        ValueError,
        r"Column 'X' type changed.*extend-only evolution does not support type changes",
    ),
    (
        "string_narrowing_rejected",
        {"S": StringType(100)},
        {"S": StringType(10)},
        ValueError,
        r"Column 'S' type changed.*extend-only evolution does not support type changes",
    ),
    (
        "decimal_widening_rejected",
        {"D": DecimalType(10, 2)},
        {"D": DecimalType(18, 2)},
        ValueError,
        r"Column 'D' type changed.*extend-only evolution does not support type changes",
    ),
    (
        "long_to_double_rejected",
        {"N": LongType()},
        {"N": DoubleType()},
        ValueError,
        r"Column 'N' type changed.*extend-only evolution does not support type changes",
    ),
    (
        "long_to_string_rejected",
        {"X": LongType()},
        {"X": StringType()},
        ValueError,
        r"Column 'X' type changed.*extend-only evolution does not support type changes",
    ),
    (
        "timestamp_to_string_rejected",
        {"T": TimestampType(TimestampTimeZone.NTZ)},
        {"T": StringType()},
        ValueError,
        r"Column 'T' type changed.*extend-only evolution does not support type changes",
    ),
)


def _schema_evolution_success_named_parameters() -> tuple[tuple[Any, ...], ...]:
    """Build absl ``named_parameters`` rows: explicit cases plus identity rows per supported type."""
    rows: list[tuple[Any, ...]] = [(*case,) for case in _SCHEMA_EVOLUTION_SUCCESS_CASES]
    for col_type in sorted(_IDENTITY_EVOLUTION_SAMPLES.keys(), key=lambda c: c.__name__):
        sample = _IDENTITY_EVOLUTION_SAMPLES[col_type]
        # Identity (old == new) must produce no commands.
        rows.append((f"identity_{col_type.__name__}", {"C": sample}, {"C": sample}, [], []))
    return tuple(rows)


class GetTableSchemaEvolutionExtendOnlyCommandsTest(parameterized.TestCase):
    """Tests for get_table_schema_evolution_extend_only_commands."""

    def test_identity_samples_cover_supported_types(self) -> None:
        self.assertEqual(
            set(_IDENTITY_EVOLUTION_SAMPLES.keys()),
            set(_SUPPORTED_TYPES),
            msg="Add a sample instance for each type in _SUPPORTED_TYPES.",
        )

    @parameterized.named_parameters(*_schema_evolution_success_named_parameters())  # type: ignore[misc]
    def test_evolution_commands_success(
        self,
        old_schema: dict[str, DataType],
        new_schema: dict[str, DataType],
        expected_forward: list[str],
        expected_rollback: list[str],
    ) -> None:
        forward, rollback = get_table_schema_evolution_extend_only_commands(_TABLE, old_schema, new_schema)
        self.assertEqual(forward, expected_forward)
        self.assertEqual(rollback, expected_rollback)

    @parameterized.named_parameters(*_SCHEMA_EVOLUTION_FAILURE_CASES)  # type: ignore[misc]
    def test_evolution_commands_failure(
        self,
        old_schema: dict[str, DataType],
        new_schema: dict[str, DataType],
        expect_exc: type[Exception],
        message_regex: str,
    ) -> None:
        with self.assertRaisesRegex(expect_exc, message_regex):
            get_table_schema_evolution_extend_only_commands(_TABLE, old_schema, new_schema)

    def test_required_old_columns_present_succeeds(self) -> None:
        """When all required columns exist in old_schema the call proceeds normally."""
        old_schema = {"ENTITY": StringType(), "TS": TimestampType(TimestampTimeZone.NTZ)}
        new_schema = {
            "ENTITY": StringType(),
            "TS": TimestampType(TimestampTimeZone.NTZ),
            "FEATURE": LongType(),
        }
        forward, rollback = get_table_schema_evolution_extend_only_commands(
            _TABLE, old_schema, new_schema, required_old_columns=["ENTITY", "TS"]
        )
        self.assertEqual(forward, [f'ALTER TABLE {_TABLE} ADD COLUMN "FEATURE" NUMBER(38,0)'])
        self.assertEqual(rollback, [f'ALTER TABLE {_TABLE} DROP COLUMN "FEATURE"'])

    def test_required_old_columns_match_is_case_insensitive(self) -> None:
        """Required-column matching uppercases both sides so user-supplied casing is irrelevant."""
        old_schema = {"entity": StringType(), "ts": TimestampType(TimestampTimeZone.NTZ)}
        new_schema = dict(old_schema)
        forward, rollback = get_table_schema_evolution_extend_only_commands(
            _TABLE, old_schema, new_schema, required_old_columns=["ENTITY", "Ts"]
        )
        self.assertEqual(forward, [])
        self.assertEqual(rollback, [])

    def test_required_old_columns_missing_raises_with_all_missing_names(self) -> None:
        """Missing required columns raise a domain-friendly error listing every missing column.

        The required-columns check fires *before* the structural prefix/type comparison,
        so callers get a "missing required columns" error rather than the generic
        position-mismatch error that the structural check would otherwise emit.
        """
        old_schema = {"FEATURE": LongType()}
        new_schema = {
            "ENTITY": StringType(),
            "TS": TimestampType(TimestampTimeZone.NTZ),
            "FEATURE": LongType(),
        }
        with self.assertRaisesRegex(ValueError, rf"{_TABLE} is missing required columns: ENTITY, TS\."):
            get_table_schema_evolution_extend_only_commands(
                _TABLE, old_schema, new_schema, required_old_columns=["ENTITY", "TS"]
            )

    def test_required_old_columns_none_skips_check(self) -> None:
        """``required_old_columns=None`` is the update-path contract: no semantic check applied."""
        old_schema = {"X": StringType()}
        new_schema = {"X": StringType(), "Y": LongType()}
        forward, _ = get_table_schema_evolution_extend_only_commands(
            _TABLE, old_schema, new_schema, required_old_columns=None
        )
        self.assertEqual(forward, [f'ALTER TABLE {_TABLE} ADD COLUMN "Y" NUMBER(38,0)'])


if __name__ == "__main__":
    absltest.main()
