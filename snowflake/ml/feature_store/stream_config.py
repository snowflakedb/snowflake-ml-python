"""StreamConfig — configuration for streaming feature views.

This module provides:
- ``StreamConfig``: User-facing dataclass for defining streaming feature view configuration.
- ``_validate_imports``: AST-based validation of transformation function imports.
- ``_infer_structtype_from_pandas``: Pandas dtype to Snowpark StructType inference.
- ``_snowpark_type_to_sql``: Snowpark DataType to SQL DDL string conversion (shared utility).
- ``ALLOWED_MODULES``: Set of modules permitted in transformation functions.
"""

from __future__ import annotations

import ast
import datetime
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import pandas as pd

from snowflake.snowpark.types import (
    BooleanType,
    DataType,
    DecimalType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

ALLOWED_MODULES: set[str] = {"numpy", "pandas", "re", "copy"}


_DANGEROUS_BUILTINS = frozenset({"__import__", "eval", "exec", "compile"})


def _validate_imports(source: str, allowed: set[str]) -> None:
    """Best-effort validation that imports in source code are from the allowed set.

    Uses AST parsing to walk the source code and check every ``import`` and
    ``from ... import`` statement.  Only the top-level package name is checked
    (e.g., ``import numpy.linalg`` checks ``numpy``).

    Also detects dynamic import/execution mechanisms: ``__import__``,
    ``importlib``, ``eval``, ``exec``, and ``compile``.

    Note: This is a heuristic check based on static analysis. It cannot catch
    all possible import mechanisms (e.g., ``getattr(builtins, '__import__')``
    or obfuscated code). It is intended as a guardrail, not a security sandbox.

    Args:
        source: Plain-text Python source code.
        allowed: Set of permitted top-level module names.

    Raises:
        ValueError: If a disallowed import or dangerous builtin call is found.
    """
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in allowed:
                    raise ValueError(
                        f"Import '{alias.name}' is not allowed in transformation_fn. "
                        f"Allowed modules: {sorted(allowed)}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module not in allowed:
                    raise ValueError(
                        f"Import from '{node.module}' is not allowed in transformation_fn. "
                        f"Allowed modules: {sorted(allowed)}"
                    )
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in _DANGEROUS_BUILTINS:
                raise ValueError(
                    f"Use of '{func_name}()' is not allowed in transformation_fn. "
                    f"Dynamic code execution and imports are forbidden."
                )


def _infer_structtype_from_pandas(pdf: pd.DataFrame) -> StructType:
    """Infer a Snowpark StructType from a pandas DataFrame.

    Maps pandas dtypes directly to Snowflake-native types for exact round-trip
    fidelity.  This avoids type promotion gaps that occur when Snowpark's
    ``LongType`` is stored as Snowflake ``NUMBER(38,0)`` and read back as
    ``DecimalType(38, 0)``.

    Mapping:
        - ``int64`` / ``Int64``      -> ``DecimalType(38, 0)``   = ``NUMBER(38,0)``
        - ``float64`` / ``Float64``  -> ``DoubleType()``         = ``FLOAT``
        - ``bool`` / ``boolean``     -> ``BooleanType()``        = ``BOOLEAN``
        - ``datetime64[ns]``         -> ``TimestampType()``      = ``TIMESTAMP_NTZ``
        - ``object`` / everything else -> ``StringType(16777216)`` = ``VARCHAR``

    Args:
        pdf: A pandas DataFrame whose dtypes will be inspected.

    Returns:
        A Snowpark StructType matching the DataFrame's columns and types.

    Raises:
        ValueError: If the DataFrame has no columns.
    """
    if pdf.columns.empty:
        raise ValueError("Cannot infer schema from an empty DataFrame (no columns).")

    fields: list[StructField] = []
    for col_name, dtype in pdf.dtypes.items():
        sf_type: DataType
        if pd.api.types.is_bool_dtype(dtype):
            # Check bool BEFORE integer — pandas bool is also integer-like
            sf_type = BooleanType()
        elif pd.api.types.is_integer_dtype(dtype):
            sf_type = DecimalType(38, 0)
        elif pd.api.types.is_float_dtype(dtype):
            sf_type = DoubleType()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sf_type = TimestampType()
        else:
            sf_type = StringType(16777216)
        fields.append(StructField(str(col_name), sf_type))
    return StructType(fields)


def _snowpark_type_to_sql(dt: Any) -> str:
    """Convert a Snowpark DataType to a SQL type string for DDL.

    Shared utility used by streaming registration (empty table creation)
    and feature_store.py (OFT column defs).

    Args:
        dt: A Snowpark DataType instance.

    Returns:
        SQL type string (e.g., ``VARCHAR``, ``NUMBER(38,0)``, ``FLOAT``).
    """
    from snowflake.snowpark import types as sp_types

    if isinstance(dt, sp_types.StringType):
        return "VARCHAR" if dt.length is None else f"VARCHAR({dt.length})"
    if isinstance(dt, sp_types.DecimalType):
        return f"NUMBER({dt.precision},{dt.scale})"
    if isinstance(dt, (sp_types.LongType, sp_types.IntegerType, sp_types.ShortType, sp_types.ByteType)):
        return "NUMBER(38,0)"
    if isinstance(dt, (sp_types.DoubleType, sp_types.FloatType)):
        return "FLOAT"
    if isinstance(dt, sp_types.BooleanType):
        return "BOOLEAN"
    if isinstance(dt, sp_types.TimestampType):
        return "TIMESTAMP_NTZ"
    if isinstance(dt, sp_types.DateType):
        return "DATE"
    if isinstance(dt, sp_types.TimeType):
        return "TIME"
    return str(dt)


@dataclass(frozen=True, eq=False)
class StreamConfig:
    """Configuration for streaming feature views.

    A ``StreamConfig`` bundles a registered ``StreamSource`` (or its name), a
    transformation function, and backfill data into a single configuration.  The
    function is validated at construction time: its source code must be extractable
    via ``inspect.getsource`` and may only import from the allowed set
    (``numpy``, ``pandas``, ``re``, ``copy``).

    Args:
        stream_source: A ``StreamSource`` object or the string name of a
            registered stream source.
        transformation_fn: A **named** Python function with signature
            ``(pd.DataFrame) -> pd.DataFrame``.  Lambdas, callable classes,
            and interactively defined functions are not supported.
        backfill_df: Snowpark DataFrame containing historical data for backfill.
            The ``transformation_fn`` will be applied to this data server-side
            via ``map_in_pandas`` and written to the udf_transformed table
            asynchronously.
        backfill_start_time: Optional timestamp to filter backfill data.  When
            provided, only rows where ``timestamp_col >= backfill_start_time``
            are included in the backfill.

    Raises:
        ValueError: If ``transformation_fn`` is not a named function, its source
            code cannot be extracted, or it imports disallowed modules.

    Example::

        >>> def normalize_txn(df: pd.DataFrame) -> pd.DataFrame:
        ...     df["amount_cents"] = (df["amount"] * 100).astype(int)
        ...     df["is_large"] = df["amount"] > 1000
        ...     return df
        >>>
        >>> stream_config = StreamConfig(
        ...     stream_source="transaction_events",
        ...     transformation_fn=normalize_txn,
        ...     backfill_df=session.table("historical_transactions"),
        ...     backfill_start_time=datetime(2024, 6, 1),
        ... )
        >>>
        >>> fv = FeatureView(
        ...     name="realtime_txn_features",
        ...     entities=[user_entity],
        ...     stream_config=stream_config,
        ...     timestamp_col="event_time",
        ...     refresh_freq="1 hour",
        ... )
    """

    stream_source: Union[str, Any]  # Union[StreamSource, str] — Any avoids circular import
    transformation_fn: Callable[[pd.DataFrame], pd.DataFrame]
    backfill_df: Any  # DataFrame — Any avoids circular import
    backfill_start_time: Optional[datetime.datetime] = None

    def __post_init__(self) -> None:
        fn = self.transformation_fn

        # 1. Must be callable
        if not callable(fn):
            raise ValueError("transformation_fn must be callable.")

        # 2. Must be a named function (not lambda, not callable class)
        if not hasattr(fn, "__name__") or fn.__name__ == "<lambda>":
            raise ValueError(
                "transformation_fn must be a named function. Lambdas and callable objects are not supported."
            )

        # 3. Source code must be extractable via inspect
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError) as e:
            raise ValueError(
                "Cannot extract source code from transformation_fn. "
                "It must be a named function defined at module level. "
                "Interactively defined or dynamically generated functions "
                f"are not supported: {e}"
            ) from e

        # 4. Validate imports against allowed modules
        _validate_imports(source, ALLOWED_MODULES)

        # 5. backfill_df must be provided
        if self.backfill_df is None:
            raise ValueError("backfill_df is required.")

    def get_function_source(self) -> str:
        """Return the dedented plain-text source code of ``transformation_fn``."""
        return textwrap.dedent(inspect.getsource(self.transformation_fn))

    def get_function_name(self) -> str:
        """Return the ``__name__`` of ``transformation_fn``."""
        return self.transformation_fn.__name__

    def get_stream_source_name(self) -> str:
        """Return the stream source name as a string.

        If ``stream_source`` is a string, returns it directly.
        If it is a ``StreamSource`` object, returns ``str(stream_source.name)``.

        Returns:
            The stream source name.
        """
        if isinstance(self.stream_source, str):
            return self.stream_source
        # StreamSource.name is a SqlIdentifier
        return str(self.stream_source.name)
