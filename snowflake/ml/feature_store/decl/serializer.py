"""Serialization utilities for decl spec model instances.

Converts Pydantic spec models to plain dicts, YAML, and JSON.
All functions are self-contained — no imports from snowflake.ml.feature_store.spec,
snowflake.snowpark, or snowflake.connector.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from typing import Any, Callable

import yaml

from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureView,
    FeatureViewRef,
    SpecBase,
    StreamingSource,
)

# Mapping from a SpecBase ``kind`` to the canonical ``source_type`` value
# emitted on ``SourceRef`` dicts. Values match
# :class:`snowflake.ml.feature_store.spec.enums.SourceType` (``Stream``,
# ``Batch``, etc.). The decl/spec consolidation collapsed the previous
# decl-only ``SQLDataSource`` kind, and the legacy ``BatchSource`` source
# type value was retired in favour of the canonical ``Batch`` (matching
# the imperative SourceType enum). The exporter's
# ``_SOURCE_TYPE_TO_KIND`` still accepts the legacy ``BatchSource``
# string for plan-file backward compatibility.
_SOURCE_KIND_TO_TYPE: dict[str, str] = {
    "StreamingSource": "Stream",
    "BatchSource": "Batch",
}


def callable_to_source(func: Callable) -> str:  # type: ignore[type-arg]
    """Extract Python source code from a callable, stripping decorator lines.

    Args:
        func: A callable whose source code should be extracted.

    Returns:
        Dedented source string starting from the ``def`` line.
    """
    source = inspect.getsource(func)
    lines = source.splitlines(keepends=True)

    # Walk past decorator lines (handles multi-line decorators via paren depth)
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].lstrip()
        if stripped.startswith("@"):
            depth = stripped.count("(") - stripped.count(")")
            idx += 1
            while depth > 0 and idx < len(lines):
                depth += lines[idx].count("(") - lines[idx].count(")")
                idx += 1
        elif stripped.startswith("def ") or stripped.startswith("async def "):
            break
        else:
            idx += 1

    return textwrap.dedent("".join(lines[idx:]))


def spec_to_dict(spec: SpecBase) -> dict[str, Any]:
    """Convert a spec model instance to a plain dict.

    Rules applied during conversion:
    - ``None`` values are excluded from the output.
    - Empty lists and dicts are excluded from the output.
    - ``False`` boolean values are excluded.
    - ``function_definition`` callables are replaced with their source code.
    - ``StreamingSource`` / ``BatchSource`` instances in a ``sources`` list are
      collapsed to ``SourceRef`` dicts (``{"name": ..., "source_type": ...}``).
    - ``Entity`` objects in ``entities`` are expanded to their join-key
      column name strings.
    - ``FeatureView`` instances in ``feature_views`` are collapsed to name-only
      ref dicts (``{"name": ...}``).

    The output dict carries the **authoring** key names (``entities``,
    ``timestamp_col``) that match the Pydantic model fields and the
    imperative ``FeatureView`` constructor.  The wire-form names
    (``ordered_entity_column_names``, ``timestamp_field``) only appear
    on the output side of :func:`spec_compiler.compile_to_spec` and on
    the input side of the ``DESCRIBE ... TYPE = SPECIFICATION``
    recovery in :mod:`state` — i.e. exclusively on the GS-bound contract.

    Args:
        spec: A ``SpecBase`` subclass instance to serialize.

    Returns:
        A plain dict suitable for JSON/YAML serialization.
    """
    raw = spec.model_dump()
    return _clean_dict(spec, raw)


def _clean_dict(spec: SpecBase, raw: dict[str, Any]) -> dict[str, Any]:
    """Walk the raw model_dump() result and apply serialization rules."""
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if value is False:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, dict) and not value:
            continue

        # Special handling for known spec fields
        if key == "function_definition":
            # Get the original object to check if it's a callable
            orig_value = _get_nested_attr(spec, key)
            if callable(orig_value):
                result[key] = callable_to_source(orig_value)
            else:
                result[key] = value
            continue

        if key == "entities" and isinstance(value, list):
            # ``Entity`` objects in ``entities`` are expanded to plain
            # join-key column-name strings so the downstream compiler
            # consumes a flat list of strings regardless of the
            # authoring shape.  The key name stays as the authoring
            # ``entities`` — the authoring → wire-form translation
            # happens inside :func:`spec_compiler.compile_to_spec`.
            result[key] = _expand_entities(spec, value)
            continue

        if key == "sources" and isinstance(value, list):
            result[key] = _collapse_sources(spec, value)
            continue

        if key == "feature_views" and isinstance(value, list):
            result[key] = _collapse_feature_views(spec, value)
            continue

        if isinstance(value, dict):
            result[key] = _clean_nested_dict(value)
        elif isinstance(value, list):
            result[key] = [_clean_value(item) for item in value]
        else:
            result[key] = value

    return result


def _get_nested_attr(spec: SpecBase, key: str) -> Any:
    """Retrieve a direct attribute from the spec object."""
    return getattr(spec, key, None)


def _clean_nested_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively clean a nested dict, excluding None/False/empty values."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if v is False:
            continue
        if isinstance(v, list) and not v:
            continue
        if isinstance(v, dict) and not v:
            result[k] = _clean_nested_dict(v)
            continue
        # Callables (e.g. function_definition inside udf) → extract source
        if k == "function_definition" and callable(v):
            result[k] = callable_to_source(v)
            continue
        if isinstance(v, dict):
            result[k] = _clean_nested_dict(v)
        elif isinstance(v, list):
            result[k] = [_clean_value(item) for item in v]
        else:
            result[k] = v
    return result


def _clean_value(item: Any) -> Any:
    """Clean a single list element."""
    if isinstance(item, dict):
        return _clean_nested_dict(item)
    return item


def _expand_entities(spec: SpecBase, raw_list: list[Any]) -> list[str]:
    """Expand ``Entity`` objects in ``entities`` to join-key column names.

    The authoring field name was renamed from ``ordered_entity_column_names``
    to ``entities`` to match the imperative API, so we look up the original
    list under the new attribute name; the wire-form key
    (``ordered_entity_column_names``) is set by the caller in
    :func:`_clean_dict`.

    Args:
        spec: The original spec object — used for accurate type checking
            (``raw_list`` has already been passed through
            :meth:`pydantic.BaseModel.model_dump`, which converts
            ``Entity`` objects to plain dicts).
        raw_list: The dumped list value; used as a fallback when ``spec``
            does not expose the new attribute (e.g., a ``SpecBase`` that
            is not a ``FeatureView``).

    Returns:
        The flat list of join-key column name strings, with any ``Entity``
        object expanded to its constituent join-key names.
    """
    original_list = getattr(spec, "entities", raw_list)
    names: list[str] = []
    for item in original_list:
        if isinstance(item, Entity):
            names.extend(jk.name for jk in item.join_keys)
        else:
            names.append(str(item))
    return names


def _collapse_sources(spec: SpecBase, raw_list: list[Any]) -> list[dict[str, Any]]:
    """Collapse source objects in sources list to SourceRef dicts."""
    original_list = getattr(spec, "sources", raw_list)
    result = []
    for item in original_list:
        if isinstance(item, (StreamingSource, BatchSource)):
            source_type = _SOURCE_KIND_TO_TYPE.get(item.kind, item.kind)
            result.append({"name": item.name, "source_type": source_type})
        elif isinstance(item, dict) and "name" in item and "source_type" in item:
            result.append({"name": item["name"], "source_type": item["source_type"]})
        elif isinstance(item, dict):
            result.append(_clean_nested_dict(item))
        elif hasattr(item, "name") and hasattr(item, "source_type"):
            # SourceRef or similar model instance
            result.append({"name": item.name, "source_type": item.source_type})
        else:
            result.append(item)
    return result


def _collapse_feature_views(spec: SpecBase, raw_list: list[Any]) -> list[dict[str, Any]]:
    """Collapse ``feature_views`` entries to ``FeatureViewRef`` dicts.

    Accepts heterogeneous shapes for backward compatibility: a
    ``FeatureViewRef`` instance (preserves ``version``, ``slice_columns``,
    ``alias``), a legacy ``FeatureView`` instance (collapses to
    ``{"name": ..., "version": ...}``), or a raw dict (passes through after
    the standard ``None`` / empty / ``False`` filtering).

    Args:
        spec: Owning spec model (used to recover the original
            ``feature_views`` attribute when it carries typed
            ``FeatureViewRef`` instances rather than dicts).
        raw_list: Fallback list passed in when ``spec`` lacks a
            ``feature_views`` attribute (defensive — keeps the helper
            usable from raw-dict callers).

    Returns:
        List of authoring-shape ``FeatureViewRef`` dicts. Each dict
        always carries ``name`` and ``version``; ``slice_columns`` /
        ``alias`` are preserved when set on the source model.
    """
    original_list = getattr(spec, "feature_views", raw_list)
    result: list[dict[str, Any]] = []
    for item in original_list:
        if isinstance(item, FeatureViewRef):
            ref: dict[str, Any] = {"name": item.name, "version": item.version}
            if item.slice_columns is not None:
                ref["slice_columns"] = list(item.slice_columns)
            if item.alias is not None:
                # Preserve alias="" (semantically: "no prefix").
                ref["alias"] = item.alias
            result.append(ref)
        elif isinstance(item, FeatureView):
            entry: dict[str, Any] = {"name": item.name}
            if item.version:
                entry["version"] = item.version
            result.append(entry)
        elif isinstance(item, dict):
            result.append(_clean_nested_dict(item))
        else:
            result.append(item)
    return result


def spec_to_yaml(spec: SpecBase) -> str:
    """Convert a spec model instance to a YAML string.

    Args:
        spec: A ``SpecBase`` subclass instance to serialize.

    Returns:
        YAML representation of the spec.
    """
    d = spec_to_dict(spec)
    return yaml.dump(d, default_flow_style=False, sort_keys=False, allow_unicode=True)


def spec_to_json(spec: SpecBase) -> str:
    """Convert a spec model instance to a JSON string.

    Args:
        spec: A ``SpecBase`` subclass instance to serialize.

    Returns:
        JSON representation of the spec.
    """
    d = spec_to_dict(spec)
    return json.dumps(d, indent=2, ensure_ascii=False)
