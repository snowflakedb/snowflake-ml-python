"""Tests for decl/dependencies.py — dependency extraction, topo sort, cycle detection.

``detect_cycles`` has no runtime callers (plan-time dependency validation lives in
``invariants._check_dependencies``), so it is defined here as a test-local helper
rather than shipped in the production module.
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store.decl.dependencies import (
    extract_dependencies,
    topological_sort,
)

# ---------------------------------------------------------------------------
# Test-local helpers (moved out of production ``dependencies.py`` — no runtime callers)
# ---------------------------------------------------------------------------


def detect_cycles(specs: list[dict[str, Any]]) -> list[list[str]]:
    """Return list of cycles found in the dependency graph.

    Each cycle is a list of spec names forming the cycle.  Returns an
    empty list if the graph is acyclic.  Uses DFS with path tracking to
    find all back-edges.

    Args:
        specs: List of normalized spec dicts.

    Returns:
        List of cycles; each cycle is a list of name strings.
    """
    name_to_spec: dict[str, dict[str, Any]] = {s.get("name", ""): s for s in specs}
    spec_names: set[str] = set(name_to_spec.keys())

    # Build adjacency list restricted to the batch.
    graph: dict[str, list[str]] = {n: [] for n in spec_names}
    for name, spec in name_to_spec.items():
        for dep in extract_dependencies(spec):
            if dep in spec_names:
                graph[name].append(dep)

    visited: set[str] = set()
    path_set: set[str] = set()
    cycles: list[list[str]] = []

    def _dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        path_set.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor, path)
            elif neighbor in path_set:
                # Found a back-edge → cycle
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])

        path.pop()
        path_set.discard(node)

    for name in spec_names:
        if name not in visited:
            _dfs(name, [])

    return cycles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(name: str = "user") -> dict[str, Any]:
    return {
        "kind": "Entity",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "join_keys": [{"name": "user_id", "type": "StringType"}],
    }


def _source(name: str = "clicks") -> dict[str, Any]:
    return {"kind": "StreamingSource", "name": name, "database": "DB", "schema": "SCH", "columns": []}


def _fv(
    name: str = "click_fv",
    entity_cols: list[Any] | None = None,
    sources: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "version": "V1",
        "entities": entity_cols or ["user"],
        "sources": sources or [{"name": "clicks", "source_type": "Stream"}],
        "features": [],
    }


def _fg(name: str = "my_group", fv_names: list[Any] | None = None) -> dict[str, Any]:
    return {
        "kind": "FeatureGroup",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "feature_views": [{"name": n} for n in (fv_names or ["click_fv"])],
    }


# ---------------------------------------------------------------------------
# extract_dependencies tests
# ---------------------------------------------------------------------------


class TestExtractDependencies:
    def test_entity_has_no_dependencies(self) -> None:
        deps = extract_dependencies(_entity())
        assert deps == []

    def test_source_has_no_dependencies(self) -> None:
        deps = extract_dependencies(_source())
        assert deps == []

    def test_fv_depends_on_entity_and_source(self) -> None:
        fv = _fv(entity_cols=["user_id"], sources=[{"name": "clicks", "source_type": "Stream"}])
        deps = extract_dependencies(fv)
        assert "user_id" in deps
        assert "clicks" in deps

    def test_fv_multiple_sources(self) -> None:
        fv = _fv(
            sources=[
                {"name": "src_a", "source_type": "Stream"},
                {"name": "src_b", "source_type": "Batch"},
            ]
        )
        deps = extract_dependencies(fv)
        assert "src_a" in deps
        assert "src_b" in deps

    def test_fg_depends_on_feature_views(self) -> None:
        fg = _fg(fv_names=["fv1", "fv2"])
        deps = extract_dependencies(fg)
        assert "fv1" in deps
        assert "fv2" in deps


# ---------------------------------------------------------------------------
# topological_sort tests
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_entities_come_before_fv(self) -> None:
        entity = _entity()
        fv = _fv()
        specs = [fv, entity]
        sorted_specs = topological_sort(specs)
        names = [s["name"] for s in sorted_specs]
        assert names.index("user") < names.index("click_fv")

    def test_sources_come_before_fv(self) -> None:
        source = _source()
        fv = _fv()
        specs = [fv, source]
        sorted_specs = topological_sort(specs)
        names = [s["name"] for s in sorted_specs]
        assert names.index("clicks") < names.index("click_fv")

    def test_fv_comes_before_fg(self) -> None:
        fv = _fv()
        fg = _fg()
        specs = [fg, fv]
        sorted_specs = topological_sort(specs)
        names = [s["name"] for s in sorted_specs]
        assert names.index("click_fv") < names.index("my_group")

    def test_full_ordering(self) -> None:
        entity = _entity()
        source = _source()
        fv = _fv()
        fg = _fg()
        specs = [fg, fv, source, entity]
        sorted_specs = topological_sort(specs)
        names = [s["name"] for s in sorted_specs]
        assert names.index("user") < names.index("click_fv")
        assert names.index("clicks") < names.index("click_fv")
        assert names.index("click_fv") < names.index("my_group")

    def test_single_spec_returns_unchanged(self) -> None:
        specs = [_entity()]
        result = topological_sort(specs)
        assert len(result) == 1

    def test_empty_list_returns_empty(self) -> None:
        result = topological_sort([])
        assert result == []


# ---------------------------------------------------------------------------
# detect_cycles tests
# ---------------------------------------------------------------------------


class TestDetectCycles:
    def test_no_cycles_returns_empty(self) -> None:
        specs = [_entity(), _source(), _fv()]
        cycles = detect_cycles(specs)
        assert cycles == []

    def test_self_cycle_detected(self) -> None:
        # A FV that depends on itself (via sources list with its own name)
        fv = {
            "kind": "StreamingFeatureView",
            "name": "cyclic_fv",
            "sources": [{"name": "cyclic_fv", "source_type": "Stream"}],
            "entities": [],
            "features": [],
        }
        cycles = detect_cycles([fv])
        assert len(cycles) >= 1

    def test_linear_chain_no_cycles(self) -> None:
        fv1 = _fv(name="fv1")
        fv2 = _fv(name="fv2", sources=[{"name": "fv1", "source_type": "Features"}])
        cycles = detect_cycles([fv1, fv2])
        assert cycles == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
