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

    # Nodes are keyed by list index (a stable per-spec identity), NOT by
    # ``name`` — object identity is ``(name, version)`` for versioned kinds
    # (mirrored by ``invariants.spec_key`` / ``state._build_spec_key``), so
    # two versions of one FeatureView name are distinct objects that must
    # both survive the sort.  Keying ``name_to_spec`` by name alone silently
    # dropped one of them (the later spec overwrote the earlier), which made
    # a still-authored v1 vanish from the batch and the planner orphan-DROP
    # its deployed counterpart.  Dependency edges are still resolved by name
    # (``extract_dependencies`` yields entity / source / FV names, which are
    # unversioned references), so a name may map to more than one provider id.
    node_ids: list[int] = list(range(len(specs)))
    spec_by_id: dict[int, dict[str, Any]] = dict(enumerate(specs))
    ids_by_name: dict[str, list[int]] = defaultdict(list)
    for node_id, spec in spec_by_id.items():
        ids_by_name[spec.get("name", "")].append(node_id)

    # Build in-degree map and adjacency list (only within the batch).
    in_degree: dict[int, int] = {node_id: 0 for node_id in node_ids}
    dependents: dict[int, list[int]] = defaultdict(list)

    for node_id, spec in spec_by_id.items():
        for dep in extract_dependencies(spec):
            # A dependency name may resolve to multiple provider ids when
            # several versions of one FV name are present; add an edge from
            # every provider so this node lands after all of them.
            for dep_id in ids_by_name.get(dep, []):
                if dep_id == node_id:
                    continue  # ignore a self-reference (would deadlock Kahn)
                in_degree[node_id] += 1
                dependents[dep_id].append(node_id)

    # Kahn's algorithm: process nodes with in-degree 0.
    # Use a list sorted by kind order for stable output.
    def _kind_order(node_id: int) -> int:
        return _KIND_ORDER.get(spec_by_id[node_id].get("kind", ""), 99)

    ready = sorted((n for n, d in in_degree.items() if d == 0), key=_kind_order)
    result: list[dict[str, Any]] = []
    resolved: set[int] = set()

    while ready:
        node = ready.pop(0)
        result.append(spec_by_id[node])
        resolved.add(node)
        for dep_id in sorted(dependents[node], key=_kind_order):
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                # Insert in kind-order position
                inserted = False
                for i, r in enumerate(ready):
                    if _kind_order(r) > _kind_order(dep_id):
                        ready.insert(i, dep_id)
                        inserted = True
                        break
                if not inserted:
                    ready.append(dep_id)

    # Append any remaining nodes (handles disconnected / external deps and
    # any cycle survivors).  Keyed by id so distinct versions are preserved.
    remaining = [spec_by_id[node_id] for node_id in node_ids if node_id not in resolved]
    remaining.sort(key=lambda s: _KIND_ORDER.get(s.get("kind", ""), 99))
    result.extend(remaining)

    return result
