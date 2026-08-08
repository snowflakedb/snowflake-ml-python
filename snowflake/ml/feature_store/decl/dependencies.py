"""Dependency graph extraction and topological sort.

All functions are pure — no database connections, no file I/O.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Kind ordinals used to break ties in topological sorting so that
# Entity → Source → FeatureView → FeatureGroup ordering is stable.
_KIND_ORDER: dict[str, int] = {
    "Entity": 0,
    "StreamingSource": 1,
    "BatchSource": 1,
    "StreamingFeatureView": 2,
    "RealtimeFeatureView": 2,
    "BatchFeatureView": 2,
    "FeatureGroup": 3,
}


def extract_dependencies(spec: dict[str, Any]) -> list[str]:
    """Return names of objects this spec depends on.

    - ``FeatureView``: entity column names and source names.
    - ``FeatureGroup``: feature view names.
    - ``Entity`` / ``Source``: no dependencies.

    Args:
        spec: A normalized spec dict.

    Returns:
        List of dependency name strings.
    """
    kind = spec.get("kind", "")
    deps: list[str] = []

    if "FeatureView" in kind:
        for col in spec.get("entities", []):
            if isinstance(col, str) and col:
                deps.append(col)
        for src in spec.get("sources", []):
            src_name = src.get("name", "") if isinstance(src, dict) else ""
            if src_name:
                deps.append(src_name)

    elif kind == "FeatureGroup":
        for fv_ref in spec.get("feature_views", []):
            fv_name = fv_ref.get("name", "") if isinstance(fv_ref, dict) else ""
            if fv_name:
                deps.append(fv_name)

    return deps


def topological_sort(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort specs by dependency order.

    Returns entities first, then sources, then feature views (respecting
    inter-FV dependencies), then feature groups.  Uses Kahn's algorithm
    with stable tie-breaking by ``_KIND_ORDER``.

    Args:
        specs: List of normalized spec dicts.

    Returns:
        Topologically-sorted list of spec dicts.
    """
    if not specs:
        return []

    name_to_spec: dict[str, dict[str, Any]] = {s.get("name", ""): s for s in specs}
    spec_names: set[str] = set(name_to_spec.keys())

    # Build in-degree map and adjacency list (only within the batch).
    in_degree: dict[str, int] = {n: 0 for n in spec_names}
    dependents: dict[str, list[str]] = defaultdict(list)

    for name, spec in name_to_spec.items():
        for dep in extract_dependencies(spec):
            if dep in spec_names:
                in_degree[name] += 1
                dependents[dep].append(name)

    # Kahn's algorithm: process nodes with in-degree 0.
    # Use a list sorted by kind order for stable output.
    def _kind_order(name: str) -> int:
        return _KIND_ORDER.get(name_to_spec[name].get("kind", ""), 99)

    ready = sorted((n for n, d in in_degree.items() if d == 0), key=_kind_order)
    result: list[dict[str, Any]] = []

    while ready:
        node = ready.pop(0)
        result.append(name_to_spec[node])
        for dep_name in sorted(dependents[node], key=_kind_order):
            in_degree[dep_name] -= 1
            if in_degree[dep_name] == 0:
                # Insert in kind-order position
                inserted = False
                for i, r in enumerate(ready):
                    if _kind_order(r) > _kind_order(dep_name):
                        ready.insert(i, dep_name)
                        inserted = True
                        break
                if not inserted:
                    ready.append(dep_name)

    # Append any remaining nodes (handles disconnected / external deps).
    remaining = [name_to_spec[n] for n in spec_names if name_to_spec[n] not in result]
    remaining.sort(key=lambda s: _KIND_ORDER.get(s.get("kind", ""), 99))
    result.extend(remaining)

    return result
