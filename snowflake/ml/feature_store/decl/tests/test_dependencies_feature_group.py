"""Phase 1 RED tests — version-aware FG dependency extraction.

The FeatureGroup spec model now requires ``feature_views[].version``.  The
dependency extractor must continue to surface ``fv_name`` strings (so the
topological sort places FG after its source FVs) but also expose the
``(name, version)`` pair via a sibling helper for downstream diagnostics
(used by Phase 2 invariants).
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store.decl.dependencies import (
    extract_dependencies,
    topological_sort,
)


def _entity(name: str = "customer") -> dict[str, Any]:
    return {
        "kind": "Entity",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "join_keys": [{"name": "customer_id", "type": "StringType"}],
    }


def _fv(name: str = "click_fv") -> dict[str, Any]:
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "version": "V1",
        "entities": ["customer_id"],
        "sources": [{"name": "clicks", "source_type": "Stream"}],
        "features": [],
    }


def _fg(
    name: str = "MY_FG",
    fv_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "FeatureGroup",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "feature_views": fv_refs or [{"name": "click_fv", "version": "V1"}],
    }


# ---------------------------------------------------------------------------
# extract_dependencies — FG with version-aware refs
# ---------------------------------------------------------------------------


class TestExtractDependenciesFeatureGroupVersionAware:
    def test_returns_fv_names_only(self) -> None:
        fg = _fg(fv_refs=[{"name": "fv1", "version": "V1"}, {"name": "fv2", "version": "V1"}])
        deps = extract_dependencies(fg)
        # The dependency edge is on the fv_name; topo sort matches by name.
        assert "fv1" in deps
        assert "fv2" in deps

    def test_handles_slice_columns_and_alias_in_refs(self) -> None:
        fg = _fg(
            fv_refs=[
                {
                    "name": "fv1",
                    "version": "V1",
                    "slice_columns": ["A", "B"],
                    "alias": "x",
                },
                {"name": "fv2", "version": "V1"},
            ]
        )
        deps = extract_dependencies(fg)
        assert deps == ["fv1", "fv2"]

    def test_same_name_different_version_emits_one_dep(self) -> None:
        # Topo sort cares about the FV node by name; two refs to (fv1, V1) and
        # (fv1, V2) still produce a dependency on "fv1".  De-duplication is
        # acceptable but not required (sort is stable either way).
        fg = _fg(
            fv_refs=[
                {"name": "fv1", "version": "V1"},
                {"name": "fv1", "version": "V2"},
            ]
        )
        deps = extract_dependencies(fg)
        assert "fv1" in deps


# ---------------------------------------------------------------------------
# topological_sort — FG still placed after its source FV
# ---------------------------------------------------------------------------


class TestTopologicalSortFeatureGroupVersionAware:
    def test_fg_after_fv(self) -> None:
        fv = _fv("click_fv")
        fg = _fg("MY_FG", fv_refs=[{"name": "click_fv", "version": "V1"}])
        sorted_specs = topological_sort([fg, fv])
        names = [s["name"] for s in sorted_specs]
        assert names.index("click_fv") < names.index("MY_FG")

    def test_full_chain(self) -> None:
        entity = _entity("customer")
        fv = _fv("click_fv")
        fg = _fg("MY_FG", fv_refs=[{"name": "click_fv", "version": "V1"}])
        sorted_specs = topological_sort([fg, fv, entity])
        names = [s["name"] for s in sorted_specs]
        assert names.index("customer") < names.index("click_fv")
        assert names.index("click_fv") < names.index("MY_FG")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
