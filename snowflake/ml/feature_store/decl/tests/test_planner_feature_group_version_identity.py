"""End-to-end regression — FG applied-state key must carry its version.

Reproduces the reviewer's symptom through the *real* applied-state builder
(``state.fetch_applied_state(feature_group_rows=...)`` →
``_build_feature_group_object``) rather than a hand-built ``AppliedObject``.

``FeatureGroup`` is a versioned kind, so :func:`invariants.spec_key` yields
``FeatureGroup:DB.SCH:MY_FG:V1`` for the local batch. If the applied-state
builder emits a version-less key (``FeatureGroup:DB.SCH:MY_FG``), the two keys
never collide and the planner treats an unchanged, still-deployed FG as both:

* ``CREATE_FG`` (diff loop: batch key misses in ``applied_state.objects``), and
* ``DROP_FG`` (orphan pass under ``full_directory_mode``: applied key absent
  from ``batch_keys``) — a destructive op gated by ``--allow-recreate``.

A clean re-plan of an unchanged FG must be a single ``NO_CHANGE`` with no
``CREATE_FG`` and no ``DROP_FG``. This file is additive so it propagates
forward without conflict and passes on ``6d1+`` (already fixed); it fails on
the buggy builder in ``6b1``.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import FeatureGroup
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.feature_store.decl.types import PlanOptions, SpecBatch
from snowflake.ml.test_utils import pytest_driver


def _fg_model() -> FeatureGroup:
    """Local FG whose hash basis matches the applied row reconstructed below."""
    return FeatureGroup.model_validate(
        {
            "kind": "FeatureGroup",
            "name": "MY_FG",
            "database": "DB",
            "schema_": "SCH",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [{"name": "FV_A", "version": "V1"}],
        }
    )


def _fg_row() -> dict[str, Any]:
    """FG metadata row that reconstructs to the same declarative shape as ``_fg_model``."""
    return {
        "name": "MY_FG",
        "version": "V1",
        "desc": "",
        "owner": "ROLE",
        "auto_prefix": True,
        "sources": [{"fv_name": "FV_A", "fv_version": "V1"}],
        "output_columns": None,
        "database_name": "DB",
        "schema_name": "SCH",
    }


class TestFeatureGroupVersionIdentity:
    def test_unchanged_deployed_fg_is_no_change_not_recreate(self) -> None:
        applied = fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=[_fg_row()],
            default_database="DB",
            default_schema="SCH",
        )
        batch = SpecBatch(specs=[_fg_model()], source_files=[])

        plan = generate_plan(batch, applied, PlanOptions(full_directory_mode=True))

        fg_ops = [op for op in plan.ops if op.name == "MY_FG"]
        op_summary = [(op.kind.value, op.destructive) for op in plan.ops]

        assert not any(op.kind == OpKind.CREATE_FG for op in plan.ops), (
            "Unchanged deployed FG must not be re-created; the applied-state key omitted "
            f"its version so the batch key never matched. Plan ops: {op_summary}"
        )
        assert not any(op.kind == OpKind.DROP_FG for op in plan.ops), (
            "Unchanged deployed FG must not be dropped as an orphan; the applied-state key "
            f"omitted its version so it fell out of batch_keys. Plan ops: {op_summary}"
        )
        assert len(fg_ops) == 1 and fg_ops[0].kind == OpKind.NO_CHANGE, (
            "A clean re-plan of an unchanged FG must be exactly one NO_CHANGE op; got "
            f"{[(op.kind.value, op.destructive) for op in fg_ops]}"
        )


if __name__ == "__main__":
    pytest_driver.main()
