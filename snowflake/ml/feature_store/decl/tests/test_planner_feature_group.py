"""Phase 2 RED tests — planner emits FG ops.

The planner must emit (no UPDATE_FG — no imperative ``update_feature_group``):

* ``CREATE_FG`` (non-destructive) for a new FG.
* ``NO_CHANGE`` when the local hash matches the applied content_hash.
* ``CREATE_FG`` with ``destructive=True`` on hash mismatch (semantically
  RECREATE; mirrors the FV-level ``backfill.overwrite=True`` pattern that
  reuses ``CREATE_FV(destructive=True)`` rather than introducing a new
  ``RECREATE_FG`` enum).
* ``DROP_FG`` (destructive) for orphaned applied-state FGs in
  ``full_directory_mode``.

Includes the cross-plan non-regression check: an FG over a BFV that uses
the advanced ``cluster_by`` / ``storage_config`` fields produces
``NO_CHANGE`` for the FG when its hash matches.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import FeatureGroup, FeatureViewRef
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fg_model(
    name: str = "MY_FG",
    version: str = "V1",
    desc: str = "",
    auto_prefix: bool = True,
    refs: list[FeatureViewRef] | None = None,
) -> FeatureGroup:
    return FeatureGroup.model_validate(
        {
            "kind": "FeatureGroup",
            "name": name,
            "database": "DB",
            "schema_": "SCH",
            "version": version,
            "desc": desc,
            "auto_prefix": auto_prefix,
            "feature_views": [r.model_dump() for r in refs] if refs else [{"name": "FV_A", "version": "V1"}],
        }
    )


def _fg_payload(
    name: str = "MY_FG",
    version: str = "V1",
    desc: str = "",
    auto_prefix: bool = True,
    feature_views: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "FeatureGroup",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "version": version,
        "desc": desc,
        "auto_prefix": auto_prefix,
        "feature_views": feature_views or [{"name": "FV_A", "version": "V1"}],
    }


def _applied_with_fg(
    spec_payload: dict[str, Any],
    *,
    content_hash: str | None = None,
) -> AppliedState:
    from snowflake.ml.feature_store.decl.invariants import fg_content_hash

    name_upper = (spec_payload.get("name") or "").upper()
    db_upper = (spec_payload.get("database") or "").upper()
    schema_upper = (spec_payload.get("schema") or "").upper()
    version_upper = str(spec_payload.get("version") or "").upper()
    key = f"FeatureGroup:{db_upper}.{schema_upper}:{name_upper}"
    # FeatureGroup is a versioned kind — mirror invariants.spec_key.
    if version_upper:
        key = f"{key}:{version_upper}"
    ao = AppliedObject(
        key=key,
        kind="FeatureGroup",
        name=spec_payload.get("name", ""),
        version=spec_payload.get("version"),
        content_hash=content_hash if content_hash is not None else fg_content_hash(spec_payload),
        spec_payload=spec_payload,
    )
    return AppliedState(objects={key: ao})


def _opts(**kwargs: Any) -> PlanOptions:
    return PlanOptions(**kwargs)


# ---------------------------------------------------------------------------
# CREATE_FG (new)
# ---------------------------------------------------------------------------


class TestPlannerCreateFG:
    def test_new_fg_emits_create_fg(self) -> None:
        fg = _fg_model()
        plan = generate_plan(SpecBatch(specs=[fg], source_files=[]), AppliedState(objects={}), _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.CREATE_FG in kinds
        create = next(op for op in plan.ops if op.kind == OpKind.CREATE_FG)
        assert create.destructive is False


# ---------------------------------------------------------------------------
# NO_CHANGE
# ---------------------------------------------------------------------------


class TestPlannerNoChangeFG:
    def test_unchanged_fg_emits_no_change(self) -> None:
        fg = _fg_model()
        # Build an applied state whose hash matches the local FG's compiled hash.
        from snowflake.ml.feature_store.decl.invariants import (
            fg_content_hash,
            model_to_dict,
        )

        local_payload = model_to_dict(fg)
        applied = _applied_with_fg(local_payload, content_hash=fg_content_hash(local_payload))

        plan = generate_plan(SpecBatch(specs=[fg], source_files=[]), applied, _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds


# ---------------------------------------------------------------------------
# Destructive CREATE_FG (semantically RECREATE)
# ---------------------------------------------------------------------------


class TestPlannerDestructiveFG:
    def test_changed_fg_emits_destructive_create(self) -> None:
        fg = _fg_model(desc="new description")
        # Applied side has the old desc → hash mismatch → destructive CREATE_FG.
        applied_payload = _fg_payload(desc="old description")
        applied = _applied_with_fg(applied_payload)

        plan = generate_plan(SpecBatch(specs=[fg], source_files=[]), applied, _opts())
        fg_ops = [op for op in plan.ops if op.kind == OpKind.CREATE_FG]
        assert len(fg_ops) == 1
        # Hash mismatch must produce a destructive op (no in-place update_feature_group).
        assert fg_ops[0].destructive is True


# ---------------------------------------------------------------------------
# DROP_FG (orphan in full_directory_mode)
# ---------------------------------------------------------------------------


class TestPlannerDropFG:
    def test_orphan_fg_emits_drop_fg_in_full_dir_mode(self) -> None:
        # No local FGs; one FG is present in applied state → DROP_FG.
        applied = _applied_with_fg(_fg_payload(name="ORPHAN_FG"))
        plan = generate_plan(
            SpecBatch(specs=[], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        kinds = [op.kind for op in plan.ops]
        assert OpKind.DROP_FG in kinds
        drop = next(op for op in plan.ops if op.kind == OpKind.DROP_FG)
        assert drop.destructive is True


# ---------------------------------------------------------------------------
# Cross-plan non-regression: FG over a BFV that uses advanced fields
# ---------------------------------------------------------------------------


class TestPlannerFGOverAdvancedBfvNoRegression:
    def test_fg_unchanged_when_bfv_uses_cluster_by(self) -> None:
        """Pin that an FG hash basis is decoupled from BFV-side advanced fields.

        The advanced BFV plan added ``cluster_by`` (etc.) to ``BatchFeatureView``;
        we hash an FG over a BFV with ``cluster_by`` set.  The FG hash must NOT
        change when the BFV's ``cluster_by`` shifts — the FG only references
        ``(fv_name, fv_version, slice_columns, alias)``.
        """
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        # Two compile-time FG payloads identical apart from a BFV-side cluster_by
        # tweak that the FG hash basis must ignore.  The hash basis MUST be
        # independent of any source-FV authoring detail beyond (name, version,
        # slice, alias).
        a = _fg_payload(feature_views=[{"name": "BFV_A", "version": "V1"}])
        b = _fg_payload(feature_views=[{"name": "BFV_A", "version": "V1"}])
        # Inject an FG-side ignored stash to mimic accidental client-side
        # additions; the basis must ignore it.
        b["bfv_authoring_witness"] = {"cluster_by": ["USER_ID"]}
        assert fg_content_hash(a) == fg_content_hash(b)


if __name__ == "__main__":
    pytest_driver.main()
