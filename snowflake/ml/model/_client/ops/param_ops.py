# Canonical location: snowflake/ml/model/_client/ops/param_ops.py
# Copybara copies this file into inference server and batch inference image
# build contexts (see model_container_services_deployment/ci/copy.bara.sky).
"""Shared parameter operations for validation, merging, and resolution.

This module is decoupled from snowflake.ml.model._signatures.core. It uses
duck typing (e.g. ``spec.dtype._numpy_type``) instead of concrete type imports
so it can run in any environment regardless of installed snowflake-ml-python version.

All functions raise plain ValueError/TypeError. Callers are responsible
for catching and re-raising as their environment's exception type.
"""

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt


def is_group_spec(spec: Any) -> bool:
    """Check whether a param spec is a group (has child specs) using duck typing.

    ParamGroupSpec has a .specs attribute (list of child specs).
    ParamSpec does not. Both have .dtype, so .dtype is not a reliable discriminator.

    Args:
        spec: A parameter specification object (duck-typed).

    Returns:
        True if spec has a .specs attribute that is a list, False otherwise.
    """
    return hasattr(spec, "specs") and isinstance(spec.specs, list)


# ---------------------------------------------------------------------------
# Dtype helpers (duck-typed — no DataType import)
# ---------------------------------------------------------------------------

_FLOAT_NUMPY_TYPES: frozenset[npt.DTypeLike] = frozenset({np.float32, np.float64})

_INT_NUMPY_TYPES: frozenset[npt.DTypeLike] = frozenset(
    {np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64}
)


def _is_float_dtype(spec: Any) -> bool:
    """Return True if *spec.dtype* is a float/double type (duck-typed)."""
    numpy_type = getattr(getattr(spec, "dtype", None), "_numpy_type", None)
    return numpy_type in _FLOAT_NUMPY_TYPES


def _is_int_dtype(spec: Any) -> bool:
    """Return True if *spec.dtype* is an integer type (duck-typed)."""
    numpy_type = getattr(getattr(spec, "dtype", None), "_numpy_type", None)
    return numpy_type in _INT_NUMPY_TYPES


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def _coerce_int_to_float(value: Any) -> Any:
    """Recursively convert int values to float in nested structures."""
    if isinstance(value, list):
        return [_coerce_int_to_float(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_coerce_int_to_float(v) for v in value)
    # bool is a subclass of int in Python — exclude it to avoid coercing True/False to 1.0/0.0
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return float(value)
    return value


def coerce_param_value(spec: Any, value: Any) -> Any:
    """Coerce a parameter value to match the spec's dtype.

    Converts int values to float when dtype is FLOAT/DOUBLE so resolved values
    have the correct Python type.

    Args:
        spec: The parameter specification (duck-typed: needs ``dtype._numpy_type``).
        value: The value to coerce.

    Returns:
        The coerced value.
    """
    # None values pass through unchanged
    if value is None:
        return None
    # Only float/double dtypes need int-to-float coercion
    if _is_float_dtype(spec):
        return _coerce_int_to_float(value)
    return value


# ---------------------------------------------------------------------------
# Leaf validation (numpy-based, mirrors core.ParamSpec._validate_default_value)
# ---------------------------------------------------------------------------


def _get_leaf_values(value: Any) -> list[Any]:
    """Extract leaf values from potentially nested lists/tuples."""
    # Recursively flatten nested containers to get scalar elements
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_get_leaf_values(item))
        return result
    # Base case: value is already a scalar
    return [value]


def _validate_param_python_types(spec: Any, value: Any, *, is_array: bool) -> None:
    """Validate that Python types of value elements are compatible with dtype.

    Prevents silent type coercions (e.g. str->int, float->int) that numpy allows.

    Args:
        spec: A parameter specification object (duck-typed: needs .dtype).
        value: The value to validate.
        is_array: Whether the value is an array (True) or scalar (False).

    Raises:
        ValueError: When types are incompatible.
    """
    # For array params, flatten to get all leaf scalars; for scalars, wrap in a list.
    # If a scalar spec receives a list/tuple, skip — numpy will catch the shape mismatch later.
    if is_array:
        elements = _get_leaf_values(value)
    else:
        if isinstance(value, (list, tuple)):
            return
        elements = [value]

    for elem in elements:
        # bool is a subclass of int — reject it for numeric types to avoid silent coercion
        if _is_int_dtype(spec):
            if isinstance(elem, bool) or not isinstance(elem, (int, np.integer)):
                raise ValueError(
                    f"Value {repr(elem)} (type: {type(elem).__name__}) is not compatible with "
                    f"dtype {spec.dtype}. Expected int."
                )
        elif _is_float_dtype(spec):
            # Accept int for float dtypes (will be coerced later), but reject str/bool/etc.
            if isinstance(elem, bool) or not isinstance(elem, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"Value {repr(elem)} (type: {type(elem).__name__}) is not compatible with "
                    f"dtype {spec.dtype}. Expected int or float."
                )


def validate_leaf_param_value(
    spec: Any,
    value: Any,
    param_path: str,
) -> None:
    """Validate a leaf (non-group) parameter value against its spec's dtype and shape.

    Mirrors core.ParamSpec._validate_default_value but raises plain
    ValueError/TypeError instead of SnowflakeMLException.

    Args:
        spec: The parameter specification (duck-typed: needs ``dtype._numpy_type``, ``shape``).
        value: The user-provided value to validate.
        param_path: Path for error messages (e.g., "temperature" or "config.temperature").

    Raises:
        ValueError: If value is incompatible with the expected type or shape.
    """
    # None is always valid — represents an unset parameter
    if value is None:
        return

    # Duck-type: extract the numpy type from spec.dtype._numpy_type.
    # If the spec has no numpy type info, skip validation (unknown/custom dtype).
    numpy_type = getattr(getattr(spec, "dtype", None), "_numpy_type", None)
    if numpy_type is None:
        return

    # Check Python types first to catch silent coercions (e.g. str -> int)
    # before numpy silently converts them.
    _validate_param_python_types(spec, value, is_array=spec.shape is not None)

    try:
        # Use numpy to validate dtype compatibility and array shape
        arr = np.array(value, dtype=numpy_type)

        if spec.shape is None:
            # Scalar param: numpy array should be 0-dimensional
            if arr.ndim != 0:
                raise ValueError(f"Parameter '{param_path}': expected scalar value, got array with shape {arr.shape}")
        else:
            # Array param: check dimensionality matches
            if arr.ndim != len(spec.shape):
                raise ValueError(
                    f"Parameter '{param_path}': expected {len(spec.shape)}-dimensional value, "
                    f"got {arr.ndim}-dimensional"
                )
            # Check each dimension; -1 means variable-length (skip check)
            for i, (expected_dim, actual_dim) in enumerate(zip(spec.shape, arr.shape)):
                if expected_dim != -1 and expected_dim != actual_dim:
                    raise ValueError(
                        f"Parameter '{param_path}': dimension {i}: expected {expected_dim}, got {actual_dim}"
                    )

    except (ValueError, TypeError, OverflowError) as e:
        # Re-raise with context about which parameter failed
        raise ValueError(
            f"Parameter '{param_path}': value {repr(value)} (type: {type(value).__name__}) "
            f"is not compatible with dtype {spec.dtype} and shape {spec.shape}. {e}"
        ) from e


# ---------------------------------------------------------------------------
# Group validation
# ---------------------------------------------------------------------------


def deep_merge_param_group(
    group_spec: Any,
    default: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge a user-provided partial dict into a ParamGroupSpec's default dict.

    For each key in the group spec:
    - If the key is in override and its child spec is a group, recurse.
    - If the key is in override and its child spec is scalar, use the override.
    - Otherwise, use the default value.

    Args:
        group_spec: The ParamGroupSpec defining the structure (duck-typed: needs .specs).
        default: The full default dict from the spec's default_value.
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
            override_value = override_lookup[key_upper]
            if is_group_spec(spec) and isinstance(override_value, dict):
                # Nested group: recurse to preserve defaults at deeper levels
                merged[spec.name] = deep_merge_param_group(spec, default.get(spec.name, {}), override_value)
            else:
                merged[spec.name] = override_value
        else:
            merged[spec.name] = default.get(spec.name)

    return merged


def validate_param_group_value(
    group_spec: Any,
    value: Any,
    param_path: str,
) -> None:
    """Recursively validate a value against a group param spec.

    Args:
        group_spec: The ParamGroupSpec (duck-typed: needs .specs, .shape).
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

    # Unshaped groups represent a single dict -- validate directly.
    _validate_param_group_dict(group_spec, value, param_path)


def _validate_shaped_param_group_value(
    group_spec: Any,
    value: Any,
    param_path: str,
    remaining_shape: tuple[int, ...],
) -> None:
    """Validate a shaped ParamGroupSpec value by unwrapping list dimensions.

    Each dimension in the shape corresponds to one level of list nesting.
    Once all dimensions are unwrapped, each leaf is validated as a dict.

    Args:
        group_spec: The ParamGroupSpec (duck-typed: needs .specs, .shape).
        value: The user-provided value to validate.
        param_path: Path for error messages (e.g., "config.items[0][2]").
        remaining_shape: Remaining shape dimensions to unwrap.

    Raises:
        ValueError: If the value structure doesn't match the shape.
    """
    # All shape dimensions consumed -- validate the leaf value as a dict.
    if not remaining_shape:
        _validate_param_group_dict(group_spec, value, param_path)
        return

    # Current dimension expects a list wrapper.
    if not isinstance(value, list):
        raise ValueError(
            f"Parameter '{param_path}' expected a list (shape={group_spec.shape}), got {type(value).__name__}."
        )

    # -1 means variable length (skip length check); otherwise enforce exact match.
    expected_len = remaining_shape[0]
    if expected_len != -1 and len(value) != expected_len:
        raise ValueError(f"Parameter '{param_path}' expected length {expected_len}, got {len(value)}.")

    # Recurse into each element with the next dimension.
    for i, elem in enumerate(value):
        _validate_shaped_param_group_value(group_spec, elem, f"{param_path}[{i}]", remaining_shape[1:])


def _validate_param_group_dict(
    group_spec: Any,
    value: Any,
    param_path: str,
) -> None:
    """Validate that a value is a dict matching the group spec's children.

    Checks: value is a dict, all keys are strings, no duplicate case-insensitive
    keys, no unknown keys. Recurses for nested groups, validates leaf values inline.

    Args:
        group_spec: The ParamGroupSpec (duck-typed: needs .specs).
        value: The user-provided value to validate.
        param_path: Dot-separated path for error messages.

    Raises:
        ValueError: If value is not a dict, has unknown/duplicate keys.
        TypeError: If value has non-string keys.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Parameter '{param_path}' expected a dict, got {type(value).__name__}.")

    # Case-insensitive key matching
    spec_lookup = {spec.name.upper(): spec for spec in group_spec.specs}
    # Track seen keys to detect duplicates
    seen_upper: set[str] = set()

    for key, child_value in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Parameter '{param_path}' has non-string key: {key} (type: {type(key).__name__}). "
                "Dict parameter keys must be strings."
            )

        # Reject duplicate keys that differ only by case (e.g., {"foo": 1, "FOO": 2})
        key_upper = key.upper()
        if key_upper in seen_upper:
            raise ValueError(
                f"Parameter '{param_path}' has duplicate case-insensitive key '{key}'. "
                "Dict parameter keys are case-insensitive."
            )
        seen_upper.add(key_upper)

        # Reject keys not declared in the spec
        if key_upper not in spec_lookup:
            raise ValueError(f"Unknown key(s) '{key}' in parameter '{param_path}'.")

        # Validate child value against its spec (recurse for nested groups)
        child_spec = spec_lookup[key_upper]
        child_path = f"{param_path}.{key}"
        if is_group_spec(child_spec):
            validate_param_group_value(child_spec, child_value, child_path)
        else:
            validate_leaf_param_value(child_spec, child_value, child_path)


# ---------------------------------------------------------------------------
# Top-level validate / resolve
# ---------------------------------------------------------------------------


def validate_params(
    params: Optional[dict[str, Any]],
    signature_params: Optional[Sequence[Any]],
) -> None:
    """Validate user-provided params against signature param specs.

    Checks: params provided when signature has none, unknown params,
    duplicate case-insensitive param names, type validation per spec.

    Args:
        params: User-provided parameter dictionary (runtime values).
        signature_params: Parameter specifications from the model signature.

    Raises:
        ValueError: If validation fails.
    """
    # Params provided but signature has no params defined
    if params and not signature_params:
        raise ValueError(
            f"Parameters were provided ({sorted(params.keys())}), " "but this method does not accept any parameters."
        )

    if not signature_params or not params:
        return

    # Case-insensitive lookup: normalized_name -> param_spec
    param_spec_lookup = {ps.name.upper(): ps for ps in signature_params}

    # Check for duplicate params with different cases (e.g., "temperature" and "TEMPERATURE")
    normalized_names = [name.upper() for name in params]
    if len(normalized_names) != len(set(normalized_names)):
        param_seen: dict[str, list[str]] = {}
        for param_name in params:
            param_seen.setdefault(param_name.upper(), []).append(param_name)
        duplicate_param_names = [names for names in param_seen.values() if len(names) > 1]
        raise ValueError(
            f"Duplicate parameter(s) provided with different cases: {duplicate_param_names}. "
            "Parameter names are case-insensitive."
        )

    # Validate user-provided params exist (case-insensitive)
    invalid_params = [name for name in params if name.upper() not in param_spec_lookup]
    if invalid_params:
        raise ValueError(
            f"Unknown parameter(s): {sorted(invalid_params)}. "
            f"Valid parameters are: {sorted(ps.name for ps in signature_params)}"
        )

    # Validate types for each provided param
    for param_name, param_value in params.items():
        param_spec = param_spec_lookup[param_name.upper()]
        if is_group_spec(param_spec):
            validate_param_group_value(param_spec, param_value, param_name)
        else:
            validate_leaf_param_value(param_spec, param_value, param_name)


def resolve_params(
    params: Optional[dict[str, Any]],
    signature_params: Sequence[Any],
) -> dict[str, Any]:
    """Resolve final parameters by applying user overrides over signature defaults.

    Coerces scalar values (e.g., int -> float for FLOAT/DOUBLE dtypes).

    Args:
        params: User-provided parameter dictionary (runtime values).
        signature_params: Parameter specifications from the model signature.

    Returns:
        Dict of canonical param name -> resolved value.
    """
    # Case-insensitive lookup: normalized_name -> param_spec
    param_spec_lookup = {ps.name.upper(): ps for ps in signature_params}

    # Start with defaults from signature (coerce types, e.g. int -> float for DOUBLE)
    final_params: dict[str, Any] = {}
    for param_spec in signature_params:
        if hasattr(param_spec, "default_value"):
            value = param_spec.default_value
            if not is_group_spec(param_spec):
                value = coerce_param_value(param_spec, value)
            final_params[param_spec.name] = value

    # Override with provided runtime parameters (using signature's original param names)
    if params:
        for param_name, override_value in params.items():
            spec = param_spec_lookup[param_name.upper()]
            # Use the canonical name from the spec (preserves original casing)
            canonical_name = spec.name
            if is_group_spec(spec) and isinstance(override_value, dict):
                # Deep merge: user provides partial dict {"temperature": 0.5},
                # we fill in missing keys from the spec's defaults
                default_dict = final_params.get(canonical_name, {})
                if isinstance(default_dict, dict):
                    final_params[canonical_name] = deep_merge_param_group(spec, default_dict, override_value)
                else:
                    # Default is not a dict (e.g. None) — use override as-is
                    final_params[canonical_name] = override_value
            else:
                # Leaf param: coerce types (e.g. int -> float for DOUBLE dtype)
                if not is_group_spec(spec):
                    override_value = coerce_param_value(spec, override_value)
                final_params[canonical_name] = override_value

    return final_params
