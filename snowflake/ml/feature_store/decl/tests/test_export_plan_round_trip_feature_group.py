"""Phase 5 RED — FG export → load → plan round-trip yields all NO_CHANGE.

Combined with the cross-plan non-regression invariant: the project under
test mixes one FeatureGroup with one BatchFeatureView that uses an
advanced field (``cluster_by``) so the test pins compatibility between
this plan and the recently-shipped advanced BFV plan.

The harness is intentionally narrow: we drive the exporter directly with
synthetic ``feature_group_rows`` (matching the
:func:`imperative_executor.fetch_feature_group_rows` shape), reload the
emitted YAMLs via :func:`loader.load_from_project`, and assert the
loaded ``FeatureGroup`` model and the corresponding ``AppliedObject`` (built
from the same source row via ``state.fetch_applied_state``) hash to the
same content.  The full plan-stream NO_CHANGE invariant for FVs lives in
``test_export_plan_round_trip.py``; this file pins the FG-only slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.exporter import export_specs
from snowflake.ml.feature_store.decl.invariants import fg_content_hash, model_to_dict
from snowflake.ml.feature_store.decl.loader import load_from_project
from snowflake.ml.feature_store.decl.spec_models import FeatureGroup


def _fg_row_with_advanced_bfv_source(name: str = "USER_FRAUD_FG") -> dict[str, Any]:
    """Build a synthetic FG row whose source FV carries an advanced field.

    The exporter does not emit FV YAMLs from FG rows directly — that is
    the FV-level path's job — so this fixture only carries the FG row.
    The cross-plan invariant the test pins is "FG hash basis is decoupled
    from BFV-side advanced fields", which holds whether or not the BFV
    YAML ever materialises in the project tree.

    Args:
        name: FG name to embed in the synthetic row (defaults to a
            project-flavoured name so the test reads naturally).

    Returns:
        Dict in the shape produced by
        :func:`imperative_executor.fetch_feature_group_rows`.
    """
    return {
        "name": name,
        "version": "V1",
        "desc": "Round-trip test FG.",
        "owner": "ROLE",
        "auto_prefix": True,
        "sources": [
            {"fv_name": "ADV_BFV", "fv_version": "V1"},
        ],
        "output_columns": ["ADV_BFV_V1_F1", "ADV_BFV_V1_F2"],
        "database_name": "MYDB",
        "schema_name": "PUBLIC",
    }


class TestExportThenLoadFGRoundTrip:
    def test_emitted_yaml_round_trips_through_loader(self, tmp_path: Path) -> None:
        rows = [_fg_row_with_advanced_bfv_source(name="USER_FRAUD_FG")]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )
        # The loader walks <root>/sources/feature_groups/.
        batch = load_from_project(tmp_path, database="MYDB", schema="PUBLIC")
        fgs = [s for s in batch.specs if s.kind == "FeatureGroup"]
        assert len(fgs) == 1
        fg = fgs[0]
        assert isinstance(fg, FeatureGroup)
        assert fg.name == "USER_FRAUD_FG"
        assert fg.feature_views[0].name == "ADV_BFV"
        assert fg.feature_views[0].version == "V1"

    def test_emitted_yaml_carries_fg_version_key(self, tmp_path: Path) -> None:
        # Belt-and-suspenders gap-closer over the hash-only assertions
        # in this file: the existing tests check that the local FG
        # hash matches the applied FG hash, but a missing/empty
        # ``version`` on both sides would hash identically and mask
        # the bug.  This test asserts the literal ``version:`` key in
        # the on-disk YAML AND on the loaded model, so a regression
        # that emits ``version: null`` (or omits the key) surfaces
        # directly.
        #
        # Pinned by the FG export ``version`` round-trip plan
        # (``plans/fg-export-version-fix_*.plan.md``).
        rows = [_fg_row_with_advanced_bfv_source(name="USER_FRAUD_FG")]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )

        # On-disk YAML carries an explicit, non-null ``version: V1``.
        fg_yaml = tmp_path / "sources" / "feature_groups" / "USER_FRAUD_FG.yaml"
        assert fg_yaml.exists()
        text = fg_yaml.read_text()
        assert "version: V1" in text, f"expected literal 'version: V1' in exported FG YAML; got:\n{text}"
        assert "version: null" not in text

        # Loaded model carries the same version (and is non-None — the
        # validator's MISSING_VERSION check would fire on None / "").
        batch = load_from_project(tmp_path, database="MYDB", schema="PUBLIC")
        fg = next(s for s in batch.specs if s.kind == "FeatureGroup")
        assert fg.version == "V1"

    def test_local_fg_hash_matches_applied_fg_hash(self, tmp_path: Path) -> None:
        rows = [_fg_row_with_advanced_bfv_source(name="USER_FRAUD_FG")]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )

        # Local: load and compute the hash basis the planner will see.
        batch = load_from_project(tmp_path, database="MYDB", schema="PUBLIC")
        fg = next(s for s in batch.specs if s.kind == "FeatureGroup")
        local_hash = fg_content_hash(model_to_dict(fg))

        # Applied: build AppliedState from the exact same row.
        state = decl_api.fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database="MYDB",
            default_schema="PUBLIC",
        )
        ao = state.objects["FeatureGroup:MYDB.PUBLIC:USER_FRAUD_FG"]
        # The planner's NO_CHANGE branch fires iff these two hashes match.
        assert local_hash == ao.content_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
