"""Unit tests for stream_config module."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.feature_store.stream_config import (
    ALLOWED_MODULES,
    StreamConfig,
    _infer_structtype_from_pandas,
    _snowpark_type_to_sql,
    _validate_imports,
)
from snowflake.snowpark.types import (
    BooleanType,
    DecimalType,
    DoubleType,
    StringType,
    TimestampType,
)

# ============================================================================
# Test transformation functions (must be module-level named functions)
# ============================================================================


def valid_transform(df: pd.DataFrame) -> pd.DataFrame:
    """A simple valid transformation function."""
    df["doubled"] = df["amount"] * 2
    return df


def transform_with_numpy(df: pd.DataFrame) -> pd.DataFrame:
    """A valid transformation using numpy (allowed module)."""
    import numpy as np

    df["log_amount"] = np.log1p(df["amount"])
    return df


def transform_with_pandas_import(df: pd.DataFrame) -> pd.DataFrame:
    """A valid transformation with explicit pandas import."""
    import pandas as pd

    df["flag"] = pd.Series([True] * len(df))
    return df


def transform_with_os(df: pd.DataFrame) -> pd.DataFrame:
    """An INVALID transformation using os (disallowed module)."""
    import os

    df["cwd"] = os.getcwd()
    return df


def transform_with_subprocess(df: pd.DataFrame) -> pd.DataFrame:
    """An INVALID transformation using subprocess."""
    from subprocess import (  # noqa: F401 — intentionally unused, testing import rejection
        run,
    )

    return df


def transform_with_re(df: pd.DataFrame) -> pd.DataFrame:
    """A valid transformation using re (allowed module)."""
    import re

    df["cleaned"] = df["name"].apply(lambda x: re.sub(r"[^a-zA-Z]", "", str(x)))
    return df


def transform_with_copy(df: pd.DataFrame) -> pd.DataFrame:
    """A valid transformation using copy (allowed module)."""
    import copy

    result = copy.deepcopy(df)
    result["new_col"] = 1
    return result


# ============================================================================
# StreamConfig validation tests
# ============================================================================


class StreamConfigValidationTest(parameterized.TestCase):
    """Unit tests for StreamConfig construction-time validation."""

    def _mock_df(self) -> MagicMock:
        return MagicMock()  # Snowpark DataFrame mock

    def test_valid_construction(self) -> None:
        """Test that a valid StreamConfig can be created."""
        config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=valid_transform,
            backfill_df=self._mock_df(),
        )
        self.assertEqual(config.get_function_name(), "valid_transform")
        self.assertEqual(config.get_stream_source_name(), "txn_events")

    def test_valid_with_backfill_start_time(self) -> None:
        """Test construction with backfill_start_time."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=valid_transform,
            backfill_df=self._mock_df(),
            backfill_start_time=datetime.datetime(2024, 6, 1),
        )
        self.assertEqual(config.backfill_start_time, datetime.datetime(2024, 6, 1))

    def test_valid_with_numpy(self) -> None:
        """Test that numpy imports are allowed."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=transform_with_numpy,
            backfill_df=self._mock_df(),
        )
        self.assertIn("numpy", config.get_function_source())

    def test_valid_with_pandas(self) -> None:
        """Test that pandas imports are allowed."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=transform_with_pandas_import,
            backfill_df=self._mock_df(),
        )
        self.assertIn("pandas", config.get_function_source())

    def test_valid_with_re(self) -> None:
        """Test that re imports are allowed."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=transform_with_re,
            backfill_df=self._mock_df(),
        )
        self.assertIn("re", config.get_function_source())

    def test_valid_with_copy(self) -> None:
        """Test that copy imports are allowed."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=transform_with_copy,
            backfill_df=self._mock_df(),
        )
        self.assertIn("copy", config.get_function_source())

    def test_reject_lambda(self) -> None:
        """Test that lambdas are rejected."""
        with self.assertRaisesRegex(ValueError, "named function"):
            StreamConfig(
                stream_source="src",
                transformation_fn=lambda df: df,
                backfill_df=self._mock_df(),
            )

    def test_reject_non_callable(self) -> None:
        """Test that non-callables are rejected."""
        with self.assertRaisesRegex(ValueError, "callable"):
            StreamConfig(
                stream_source="src",
                transformation_fn="not a function",  # type: ignore[arg-type]
                backfill_df=self._mock_df(),
            )

    def test_reject_disallowed_import_os(self) -> None:
        """Test that os import is rejected."""
        with self.assertRaisesRegex(ValueError, "not allowed"):
            StreamConfig(
                stream_source="src",
                transformation_fn=transform_with_os,
                backfill_df=self._mock_df(),
            )

    def test_reject_disallowed_from_import(self) -> None:
        """Test that 'from subprocess import ...' is rejected."""
        with self.assertRaisesRegex(ValueError, "not allowed"):
            StreamConfig(
                stream_source="src",
                transformation_fn=transform_with_subprocess,
                backfill_df=self._mock_df(),
            )

    def test_reject_no_backfill_df(self) -> None:
        """Test that missing backfill_df is rejected."""
        with self.assertRaisesRegex(ValueError, "backfill_df"):
            StreamConfig(
                stream_source="src",
                transformation_fn=valid_transform,
                backfill_df=None,
            )

    def test_get_function_source(self) -> None:
        """Test that get_function_source returns dedented source code."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=valid_transform,
            backfill_df=self._mock_df(),
        )
        source = config.get_function_source()
        self.assertIn("def valid_transform", source)
        self.assertIn('df["doubled"]', source)

    def test_get_function_name(self) -> None:
        """Test that get_function_name returns the function's __name__."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=valid_transform,
            backfill_df=self._mock_df(),
        )
        self.assertEqual(config.get_function_name(), "valid_transform")

    def test_get_stream_source_name_string(self) -> None:
        """Test get_stream_source_name with a string source."""
        config = StreamConfig(
            stream_source="my_source",
            transformation_fn=valid_transform,
            backfill_df=self._mock_df(),
        )
        self.assertEqual(config.get_stream_source_name(), "my_source")

    def test_frozen_dataclass(self) -> None:
        """Test that StreamConfig is immutable (frozen)."""
        config = StreamConfig(
            stream_source="src",
            transformation_fn=valid_transform,
            backfill_df=self._mock_df(),
        )
        with self.assertRaises(AttributeError):
            config.stream_source = "other"  # type: ignore[misc]


# ============================================================================
# _validate_imports tests
# ============================================================================


class ValidateImportsTest(parameterized.TestCase):
    """Unit tests for _validate_imports."""

    def test_no_imports(self) -> None:
        """Test source with no imports passes."""
        _validate_imports("def f(x): return x", ALLOWED_MODULES)

    @parameterized.parameters(  # type: ignore[misc]
        "import numpy",
        "import pandas",
        "import re",
        "import copy",
        "import numpy as np",
        "from numpy import array",
        "from pandas import DataFrame",
        "import numpy.linalg",
    )
    def test_allowed_imports(self, source: str) -> None:
        """Test that allowed imports pass."""
        _validate_imports(f"def f():\n    {source}", ALLOWED_MODULES)

    @parameterized.parameters(  # type: ignore[misc]
        "import os",
        "import sys",
        "import subprocess",
        "from os import path",
        "from subprocess import run",
        "import socket",
    )
    def test_disallowed_imports(self, source: str) -> None:
        """Test that disallowed imports are rejected."""
        with self.assertRaisesRegex(ValueError, "not allowed"):
            _validate_imports(f"def f():\n    {source}", ALLOWED_MODULES)

    @parameterized.parameters(  # type: ignore[misc]
        "__import__('os')",
        "eval('1+1')",
        "exec('import os')",
        "compile('pass', '<string>', 'exec')",
    )
    def test_dangerous_builtins_rejected(self, call_expr: str) -> None:
        """Test that __import__, eval, exec, compile calls are rejected."""
        with self.assertRaisesRegex(ValueError, "not allowed"):
            _validate_imports(f"def f():\n    {call_expr}", ALLOWED_MODULES)


# ============================================================================
# _infer_structtype_from_pandas tests
# ============================================================================


class InferStructTypeTest(parameterized.TestCase):
    """Unit tests for _infer_structtype_from_pandas."""

    def test_integer_column(self) -> None:
        """Test int64 maps to DecimalType(38, 0)."""
        pdf = pd.DataFrame({"col": [1, 2, 3]})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertEqual(len(schema.fields), 1)
        # StructField uppercases names by default
        self.assertEqual(schema.fields[0].name, "COL")
        self.assertIsInstance(schema.fields[0].datatype, DecimalType)
        self.assertEqual(schema.fields[0].datatype.precision, 38)
        self.assertEqual(schema.fields[0].datatype.scale, 0)

    def test_float_column(self) -> None:
        """Test float64 maps to DoubleType."""
        pdf = pd.DataFrame({"col": [1.0, 2.5, 3.7]})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, DoubleType)

    def test_bool_column(self) -> None:
        """Test bool maps to BooleanType (checked before integer)."""
        pdf = pd.DataFrame({"col": [True, False, True]})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, BooleanType)

    def test_datetime_column(self) -> None:
        """Test datetime64 maps to TimestampType."""
        pdf = pd.DataFrame({"col": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, TimestampType)

    def test_string_column(self) -> None:
        """Test object/string maps to StringType() with no length."""
        pdf = pd.DataFrame({"col": ["a", "b", "c"]})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, StringType)
        self.assertIsNone(schema.fields[0].datatype.length)

    def test_multiple_columns(self) -> None:
        """Test inference of multiple columns with different types."""
        pdf = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "amount": [100.0, 200.0],
                "count": [1, 2],
                "is_active": [True, False],
                "event_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        schema = _infer_structtype_from_pandas(pdf)
        self.assertEqual(len(schema.fields), 5)

        self.assertIsInstance(schema.fields[0].datatype, StringType)  # user_id
        self.assertIsInstance(schema.fields[1].datatype, DoubleType)  # amount
        self.assertIsInstance(schema.fields[2].datatype, DecimalType)  # count
        self.assertIsInstance(schema.fields[3].datatype, BooleanType)  # is_active
        self.assertIsInstance(schema.fields[4].datatype, TimestampType)  # event_time

    def test_empty_dataframe_no_columns(self) -> None:
        """Test that an empty DataFrame (no columns) raises ValueError."""
        pdf = pd.DataFrame()
        with self.assertRaisesRegex(ValueError, "no columns"):
            _infer_structtype_from_pandas(pdf)

    def test_empty_dataframe_with_columns(self) -> None:
        """Test that an empty DataFrame with columns succeeds."""
        pdf = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertEqual(len(schema.fields), 1)
        self.assertIsInstance(schema.fields[0].datatype, DecimalType)

    def test_nullable_integer(self) -> None:
        """Test that pandas nullable Int64 maps to DecimalType."""
        pdf = pd.DataFrame({"col": pd.array([1, None, 3], dtype="Int64")})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, DecimalType)

    def test_nullable_float(self) -> None:
        """Test that pandas nullable Float64 maps to DoubleType."""
        pdf = pd.DataFrame({"col": pd.array([1.0, None, 3.0], dtype="Float64")})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, DoubleType)

    def test_nullable_boolean(self) -> None:
        """Test that pandas nullable boolean maps to BooleanType."""
        pdf = pd.DataFrame({"col": pd.array([True, None, False], dtype="boolean")})
        schema = _infer_structtype_from_pandas(pdf)
        self.assertIsInstance(schema.fields[0].datatype, BooleanType)


# ============================================================================
# _snowpark_type_to_sql tests
# ============================================================================


class SnowparkTypeToSqlTest(parameterized.TestCase):
    """Unit tests for _snowpark_type_to_sql shared utility."""

    def test_string_type(self) -> None:
        self.assertEqual(_snowpark_type_to_sql(StringType(16777216)), "VARCHAR(16777216)")

    def test_string_type_no_length(self) -> None:
        self.assertEqual(_snowpark_type_to_sql(StringType()), "VARCHAR")

    def test_decimal_type(self) -> None:
        self.assertEqual(_snowpark_type_to_sql(DecimalType(38, 0)), "NUMBER(38,0)")

    def test_double_type(self) -> None:
        self.assertEqual(_snowpark_type_to_sql(DoubleType()), "FLOAT")

    def test_boolean_type(self) -> None:
        self.assertEqual(_snowpark_type_to_sql(BooleanType()), "BOOLEAN")

    def test_timestamp_type(self) -> None:
        self.assertEqual(_snowpark_type_to_sql(TimestampType()), "TIMESTAMP_NTZ")


class SnowparkTypeToSqlExtendedTest(parameterized.TestCase):
    """Additional _snowpark_type_to_sql tests for types not yet covered."""

    def test_long_type(self) -> None:
        from snowflake.snowpark.types import LongType

        self.assertEqual(_snowpark_type_to_sql(LongType()), "NUMBER(38,0)")

    def test_integer_type(self) -> None:
        from snowflake.snowpark.types import IntegerType

        self.assertEqual(_snowpark_type_to_sql(IntegerType()), "NUMBER(38,0)")

    def test_float_type(self) -> None:
        from snowflake.snowpark.types import FloatType

        self.assertEqual(_snowpark_type_to_sql(FloatType()), "FLOAT")

    def test_date_type(self) -> None:
        from snowflake.snowpark.types import DateType

        self.assertEqual(_snowpark_type_to_sql(DateType()), "DATE")

    def test_time_type(self) -> None:
        from snowflake.snowpark.types import TimeType

        self.assertEqual(_snowpark_type_to_sql(TimeType()), "TIME")


class StreamConfigStreamSourceObjectTest(absltest.TestCase):
    """Test get_stream_source_name with a StreamSource object (not just string)."""

    def test_get_stream_source_name_with_object(self) -> None:
        from snowflake.ml.feature_store.stream_source import StreamSource as SS
        from snowflake.snowpark.types import StructField, StructType

        source = SS(
            name="my_source",
            schema=StructType([StructField("col", StringType())]),
        )
        config = StreamConfig(
            stream_source=source,
            transformation_fn=valid_transform,
            backfill_df=MagicMock(),
        )
        self.assertEqual(config.get_stream_source_name(), "MY_SOURCE")


if __name__ == "__main__":
    absltest.main()
