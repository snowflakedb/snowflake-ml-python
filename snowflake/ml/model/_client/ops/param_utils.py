"""Utility functions for model parameter validation and resolution."""

import datetime
from typing import Any, Optional, Sequence

from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._signatures import core
from snowflake.snowpark._internal import utils as snowpark_utils


def format_param_value_for_sql(value: Any) -> str:
    """Format a parameter value as a valid SQL literal expression.

    This function converts Python values to their SQL literal representations
    for use in both MANIFEST files (model creation) and runtime SQL execution.

    Args:
        value: The parameter value to format.

    Returns:
        A string representation suitable for SQL.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        # SQL booleans are lowercase; check before int since bool is subclass of int
        return "true" if value else "false"
    if isinstance(value, str):
        # SQL string literals use single quotes; escape any single quotes in the value
        escaped = snowpark_utils.escape_single_quotes(value)  # type: ignore[no-untyped-call]
        return f"'{escaped}'"
    if isinstance(value, bytes):
        # Convert bytes to hex literal for SQL
        hex_str = value.hex()
        return f"X'{hex_str}'"
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        # Format as SQL date literal (check before datetime since datetime is subclass of date)
        return f"'{value.isoformat()}'::DATE"
    if isinstance(value, datetime.datetime):
        # Format as SQL timestamp literal
        iso_str = value.isoformat(sep=" ")
        return f"'{iso_str}'::TIMESTAMP_NTZ"
    if isinstance(value, list):
        # Use str() for lists - Python's repr uses single quotes for strings,
        # which SQL interprets as string literals (not identifiers like double quotes)
        return str(value)
    # Numeric types (int, float) and other values can be converted directly
    return str(value)


def format_param_value_for_table_function_sql(value: Any) -> str:
    """Format a parameter value for table function SQL invocation with explicit type casts.

    Table functions enforce strict argument type matching. Snowflake infers bare float
    literals like ``1.0`` as ``NUMBER``, not ``FLOAT``, causing type mismatch errors.

    Args:
        value: The parameter value to format.

    Returns:
        A string representation suitable for table function SQL invocation.
    """
    base = format_param_value_for_sql(value)
    if isinstance(value, float):
        return f"{base}::FLOAT"
    return base


def validate_params(
    params: Optional[dict[str, Any]],
    signature_params: Optional[Sequence[core.BaseParamSpec]],
) -> None:
    """Validate user-provided params against signature params.

    Args:
        params: User-provided parameter dictionary (runtime values).
        signature_params: Parameter specifications from the model signature.

    Raises:
        SnowflakeMLException: If params are provided but signature has no params,
            or if unknown params are provided, or if param types are invalid,
            or if duplicate params are provided with different cases.
    """
    # Params provided but signature has no params defined
    if params and not signature_params:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Parameters were provided ({sorted(params.keys())}), "
                "but this method does not accept any parameters."
            ),
        )

    if not signature_params or not params:
        return

    # Case-insensitive lookup: normalized_name -> param_spec
    param_spec_lookup = {ps.name.upper(): ps for ps in signature_params}

    # Check for duplicate params with different cases (e.g., "temperature" and "TEMPERATURE")
    normalized_names = [name.upper() for name in params]
    if len(normalized_names) != len(set(normalized_names)):
        # Find the duplicate params to raise an error
        param_seen: dict[str, list[str]] = {}
        for param_name in params:
            param_seen.setdefault(param_name.upper(), []).append(param_name)
        duplicate_param_names = [param_names for param_names in param_seen.values() if len(param_names) > 1]
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Duplicate parameter(s) provided with different cases: {duplicate_param_names}. "
                "Parameter names are case-insensitive."
            ),
        )

    # Validate user-provided params exist (case-insensitive)
    invalid_params = [name for name in params if name.upper() not in param_spec_lookup]
    if invalid_params:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Unknown parameter(s): {sorted(invalid_params)}. "
                f"Valid parameters are: {sorted(ps.name for ps in signature_params)}"
            ),
        )

    # Validate types for each provided param
    for param_name, default_value in params.items():
        param_spec = param_spec_lookup[param_name.upper()]
        if isinstance(param_spec, core.ParamSpec):
            core.ParamSpec._validate_default_value(param_spec.dtype, default_value, param_spec.shape)


def resolve_params(
    params: Optional[dict[str, Any]],
    signature_params: Sequence[core.BaseParamSpec],
) -> list[tuple[sql_identifier.SqlIdentifier, Any]]:
    """Resolve final method parameters by applying user-provided params over signature defaults.

    Args:
        params: User-provided parameter dictionary (runtime values).
        signature_params: Parameter specifications from the model signature.

    Returns:
        List of tuples (SqlIdentifier, value) for method invocation.
    """
    # Case-insensitive lookup: normalized_name -> param_spec
    param_spec_lookup = {ps.name.upper(): ps for ps in signature_params}

    # Start with defaults from signature
    final_params: dict[str, Any] = {}
    for param_spec in signature_params:
        if hasattr(param_spec, "default_value"):
            final_params[param_spec.name] = param_spec.default_value

    # Override with provided runtime parameters (using signature's original param names)
    if params:
        for param_name, override_value in params.items():
            canonical_name = param_spec_lookup[param_name.upper()].name
            final_params[canonical_name] = override_value

    return [(sql_identifier.SqlIdentifier(param_name), param_value) for param_name, param_value in final_params.items()]


def validate_and_resolve_params(
    params: Optional[dict[str, Any]],
    signature_params: Optional[Sequence[core.BaseParamSpec]],
) -> Optional[list[tuple[sql_identifier.SqlIdentifier, Any]]]:
    """Validate user-provided params against signature params and return method parameters.

    Args:
        params: User-provided parameter dictionary (runtime values).
        signature_params: Parameter specifications from the model signature.

    Returns:
        List of tuples (SqlIdentifier, value) for method invocation, or None if no params.
    """
    validate_params(params, signature_params)

    if not signature_params:
        return None

    return resolve_params(params, signature_params)
