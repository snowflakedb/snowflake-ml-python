"""Phase 1 RED tests — version-aware FG dependency extraction.

The FeatureGroup spec model now requires ``feature_views[].version``.  The
dependency extractor must continue to surface ``fv_name`` strings (so the
topological sort places FG after its source FVs) but also expose the
``(name, version)`` pair via a sibling helper for downstream diagnostics
(used by Phase 2 invariants).
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.dependencies import (
    extract_dependencies,
    order_specs_for_drop,
    topological_sort,
)
from snowflake.ml.test_utils import pytest_driver


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


# ---------------------------------------------------------------------------
# order_specs_for_drop — reverse-topological teardown order
# ---------------------------------------------------------------------------


class TestOrderSpecsForDrop:
    """``order_specs_for_drop`` is the safe teardown order — the reverse of
    ``topological_sort``.  A FeatureGroup must precede its member FeatureViews
    so Snowflake's online FG table (which references each member FV's online
    table) is torn down before the members it depends on.
    """

    def test_fg_before_member_fv(self) -> None:
        fv = _fv("click_fv")
        fg = _fg("MY_FG", fv_refs=[{"name": "click_fv", "version": "V1"}])
        ordered = order_specs_for_drop([fv, fg])
        names = [s["name"] for s in ordered]
        assert names.index("MY_FG") < names.index("click_fv")

    def test_full_chain_reverse(self) -> None:
        entity = _entity("customer")
        fv = _fv("click_fv")
        fg = _fg("MY_FG", fv_refs=[{"name": "click_fv", "version": "V1"}])
        ordered = order_specs_for_drop([fg, fv, entity])
        names = [s["name"] for s in ordered]
        # FeatureGroup → FeatureView → Entity (reverse of create order).
        assert names.index("MY_FG") < names.index("click_fv")
        assert names.index("click_fv") < names.index("customer")

    def test_empty_input(self) -> None:
        assert order_specs_for_drop([]) == []

    def test_two_versions_of_one_fv_name_both_survive(self) -> None:
        # (name, version) identity — both versions must survive the sort so
        # neither DROP is silently lost (6b1 index-keyed nodes).
        fv_v1 = {"kind": "BatchFeatureView", "name": "FV", "version": "V1"}
        fv_v2 = {"kind": "BatchFeatureView", "name": "FV", "version": "V2"}
        ordered = order_specs_for_drop([fv_v1, fv_v2])
        assert len(ordered) == 2
        versions = {s["version"] for s in ordered}
        assert versions == {"V1", "V2"}


if __name__ == "__main__":
    pytest_driver.main()
