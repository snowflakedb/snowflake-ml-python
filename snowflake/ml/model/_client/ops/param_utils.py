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
    if isinstance(value, dict):
        # OBJECT_CONSTRUCT_KEEP_NULL preserves NULL values and maintains type fidelity.
        # Nested dicts recurse into nested OBJECT_CONSTRUCT_KEEP_NULL calls.
        parts = []
        for k, v in value.items():
            parts.append(f"'{k}'")
            parts.append(format_param_value_for_sql(v))
        return f"OBJECT_CONSTRUCT_KEEP_NULL({', '.join(parts)})"
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


def _validate_param_group_value(
    group_spec: core.ParamGroupSpec,
    value: Any,
    param_path: str,
) -> None:
    """Recursively validate a value against a ParamGroupSpec.

    Args:
        group_spec: The ParamGroupSpec defining the expected structure.
        value: The user-provided value to validate.
        param_path: Dot-separated path for error messages (e.g., "config.sampling").
    """
    if value is None:
        return

    # Shaped groups (e.g. shape=(2,)) represent a list of dicts.
    # Validate by recursively unwrapping list dimensions first,
    # then checking each leaf dict against the spec's children.
    if group_spec.shape is not None:
        _validate_shaped_param_group_value(group_spec, value, param_path, group_spec.shape)
        return

    # Unshaped groups represent a single dict — validate directly.
    _validate_param_group_dict(group_spec, value, param_path)


def _validate_shaped_param_group_value(
    group_spec: core.ParamGroupSpec,
    value: Any,
    param_path: str,
    remaining_shape: tuple[int, ...],
) -> None:
    """Validate a shaped ParamGroupSpec value by unwrapping list dimensions.

    Each dimension in the shape corresponds to one level of list nesting.
    Once all dimensions are unwrapped, each leaf is validated as a dict.

    Args:
        group_spec: The ParamGroupSpec defining the expected structure.
        value: The user-provided value to validate.
        param_path: Path for error messages (e.g., "config.items[0][2]").
            May include both dot-separated segments and bracket indices.
        remaining_shape: Remaining shape dimensions to unwrap.

    Raises:
        SnowflakeMLException: If the value structure doesn't match the shape.
    """
    # All shape dimensions consumed — validate the leaf value as a dict.
    if not remaining_shape:
        _validate_param_group_dict(group_spec, value, param_path)
        return

    # Current dimension expects a list wrapper.
    if not isinstance(value, list):
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Parameter '{param_path}' expected a list (shape={group_spec.shape}), " f"got {type(value).__name__}."
            ),
        )

    # -1 means variable length (skip length check); otherwise enforce exact match.
    expected_len = remaining_shape[0]
    if expected_len != -1 and len(value) != expected_len:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Parameter '{param_path}' expected length {expected_len}, got {len(value)}."
            ),
        )

    # Recurse into each element with the next dimension.
    for i, elem in enumerate(value):
        _validate_shaped_param_group_value(group_spec, elem, f"{param_path}[{i}]", remaining_shape[1:])


def _validate_param_group_dict(
    group_spec: core.ParamGroupSpec,
    value: Any,
    param_path: str,
) -> None:
    """Validate that a value is a dict matching the group spec's children.

    Args:
        group_spec: The ParamGroupSpec defining the expected structure.
        value: The user-provided value to validate.
        param_path: Dot-separated path for error messages (e.g., "config.sampling").

    Raises:
        SnowflakeMLException: If the value is not a dict, contains unknown keys,
            non-string keys, duplicate case-insensitive keys, or child values
            fail type validation.
    """
    if not isinstance(value, dict):
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(f"Parameter '{param_path}' expected a dict, got {type(value).__name__}."),
        )

    # Case-insensitive key matching
    spec_lookup = {spec.name.upper(): spec for spec in group_spec.specs}

    # Track seen keys to detect duplicates
    seen_upper: set[str] = set()

    for key, child_value in value.items():
        # Validate the key is a string
        if not isinstance(key, str):
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=TypeError(
                    f"Parameter '{param_path}' has non-string key: {key}. " "Dict parameter keys must be strings."
                ),
            )

        # Reject duplicate keys that differ only by case (e.g., {"foo": 1, "FOO": 2})
        key_upper = key.upper()
        if key_upper in seen_upper:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Parameter '{param_path}' has duplicate case-insensitive key '{key}'. "
                    "Dict parameter keys are case-insensitive."
                ),
            )
        seen_upper.add(key_upper)

        # Reject keys not declared in the spec
        if key_upper not in spec_lookup:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Unknown key(s) '{key}' in parameter '{param_path}'."),
            )

        # Validate child value against its spec (recurse for nested groups)
        child_spec = spec_lookup[key_upper]
        child_path = f"{param_path}.{key}"
        if isinstance(child_spec, core.ParamSpec):
            # Reuse ParamSpec's existing validation for scalar/array type and shape checking
            core.ParamSpec._validate_default_value(child_spec.dtype, child_value, child_spec.shape)
        elif isinstance(child_spec, core.ParamGroupSpec):
            # Recursively validate the child value against the child spec
            _validate_param_group_value(child_spec, child_value, child_path)


def _deep_merge_param_group(
    group_spec: core.ParamGroupSpec,
    default: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge a user-provided partial dict into a ParamGroupSpec's default dict.

    For each key in the group spec:
    - If the key is in override and its child spec is a ParamGroupSpec, recurse.
    - If the key is in override and its child spec is a ParamSpec, use the override value.
    - Otherwise, use the default value.

    Args:
        group_spec: The ParamGroupSpec defining the structure.
        default: The full default dict from group_spec.default_value.
        override: The user-provided partial dict.

    Returns:
        A merged dict with all keys from the spec.
    """
    # Case-insensitive key matching to be forgiving of user key casing
    override_lookup = {k.upper(): v for k, v in override.items()}

    # Iterate over the spec (not the override) so every declared key appears in the output.
    # This ensures unspecified keys retain their defaults.
    merged: dict[str, Any] = {}
    for spec in group_spec.specs:
        key_upper = spec.name.upper()
        if key_upper in override_lookup:
            # Get the override value for the key
            override_value = override_lookup[key_upper]
            if isinstance(spec, core.ParamGroupSpec) and isinstance(override_value, dict):
                # Nested group: recurse to preserve defaults at deeper levels
                merged[spec.name] = _deep_merge_param_group(spec, default.get(spec.name, {}), override_value)
            else:
                # Use the override value for the key
                merged[spec.name] = override_value
        else:
            # Use the default value for the key
            merged[spec.name] = default.get(spec.name)

    return merged


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
    for param_name, param_value in params.items():
        param_spec = param_spec_lookup[param_name.upper()]
        if isinstance(param_spec, core.ParamSpec):
            core.ParamSpec._validate_default_value(param_spec.dtype, param_value, param_spec.shape)
        elif isinstance(param_spec, core.ParamGroupSpec):
            _validate_param_group_value(param_spec, param_value, param_name)


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

    # Start with defaults from signature, coercing types (e.g. int -> float for DOUBLE)
    final_params: dict[str, Any] = {}
    for param_spec in signature_params:
        if hasattr(param_spec, "default_value"):
            value = param_spec.default_value
            if isinstance(param_spec, core.ParamSpec):
                value = core.coerce_param_value(param_spec, value)
            final_params[param_spec.name] = value

    # Override with provided runtime parameters (using signature's original param names)
    if params:
        for param_name, override_value in params.items():
            spec = param_spec_lookup[param_name.upper()]
            canonical_name = spec.name
            if isinstance(spec, core.ParamGroupSpec) and isinstance(override_value, dict):
                # Deep merge: user provides partial dict {"temperature": 0.5},
                # we fill in missing keys from the spec's defaults {"temperature": 0.5, "top_k": 50}
                default_dict = final_params.get(canonical_name, {})
                if isinstance(default_dict, dict):
                    final_params[canonical_name] = _deep_merge_param_group(spec, default_dict, override_value)
                else:
                    final_params[canonical_name] = override_value
            else:
                if isinstance(spec, core.ParamSpec):
                    override_value = core.coerce_param_value(spec, override_value)
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
