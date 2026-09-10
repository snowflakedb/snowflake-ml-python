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
from snowflake.ml.feature_store.decl.invariants import (
    fg_content_hash,
    model_to_dict,
    spec_key,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import (
    FeatureGroup,
    FeatureView,
    FeatureViewRef,
)
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


def _applied_fv_object(name: str, version: str = "V1") -> tuple[str, AppliedObject]:
    """Build an applied ``BatchFeatureView`` object (orphan-drop candidate)."""
    key = f"FeatureView:DB.SCH:{name}:{version}"
    return key, AppliedObject(
        key=key,
        kind="BatchFeatureView",
        name=name,
        version=version,
        content_hash="fvhash",
        spec_payload={"kind": "BatchFeatureView", "name": name, "version": version},
    )


def _applied_fg_object(
    name: str,
    member_fv_names: list[str],
    version: str = "V1",
) -> tuple[str, AppliedObject]:
    """Build an applied ``FeatureGroup`` object referencing ``member_fv_names``."""
    from snowflake.ml.feature_store.decl.invariants import fg_content_hash

    payload = _fg_payload(
        name=name,
        version=version,
        feature_views=[{"name": m, "version": "V1"} for m in member_fv_names],
    )
    key = f"FeatureGroup:DB.SCH:{name}:{version}"
    return key, AppliedObject(
        key=key,
        kind="FeatureGroup",
        name=name,
        version=version,
        content_hash=fg_content_hash(payload),
        spec_payload=payload,
    )


class TestPlannerDropFGOrdering:
    """Pin the referential-integrity contract from
    ``plans/bug_orphan_drop_pass_no_fg_member_dependency_order.md``: the
    orphan-drop pass must emit ``DROP_FG`` for a FeatureGroup *before* the
    ``DROP_FV`` for any of its member FeatureViews, regardless of the order
    in which those objects appear in ``applied_state.objects``.
    """

    def _live_graph_applied(self) -> AppliedState:
        # Members inserted FIRST, FG LAST — the failing dict order this bug
        # documents (``USER_FRAUD_FG_DECL`` over ``USER_AMOUNTS_FG_DECL`` /
        # ``USER_CLICKS_FG_DECL``).
        objects: dict[str, AppliedObject] = {}
        for k, o in (
            _applied_fv_object("USER_AMOUNTS_FG_DECL"),
            _applied_fv_object("USER_CLICKS_FG_DECL"),
            _applied_fg_object(
                "USER_FRAUD_FG_DECL",
                ["USER_AMOUNTS_FG_DECL", "USER_CLICKS_FG_DECL"],
            ),
        ):
            objects[k] = o
        return AppliedState(objects=objects)

    def test_drop_fg_before_member_drops_members_first_in_dict(self) -> None:
        applied = self._live_graph_applied()
        plan = generate_plan(
            SpecBatch(specs=[], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        fg_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FG)
        fv_idxs = [i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FV]
        assert len(fv_idxs) == 2
        assert all(fg_idx < i for i in fv_idxs), (
            "DROP_FG must precede every member DROP_FV; "
            f"got FG at {fg_idx}, FVs at {fv_idxs}: "
            f"{[(op.kind, op.name) for op in plan.ops]}"
        )

    def test_drop_fg_before_member_fg_first_in_dict(self) -> None:
        # Control: even when the FG is inserted FIRST, ordering holds.
        objects: dict[str, AppliedObject] = {}
        for k, o in (
            _applied_fg_object(
                "USER_FRAUD_FG_DECL",
                ["USER_AMOUNTS_FG_DECL", "USER_CLICKS_FG_DECL"],
            ),
            _applied_fv_object("USER_AMOUNTS_FG_DECL"),
            _applied_fv_object("USER_CLICKS_FG_DECL"),
        ):
            objects[k] = o
        applied = AppliedState(objects=objects)
        plan = generate_plan(
            SpecBatch(specs=[], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        fg_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FG)
        fv_idxs = [i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FV]
        assert len(fv_idxs) == 2
        assert all(fg_idx < i for i in fv_idxs)

    def test_independent_orphan_fv_still_dropped(self) -> None:
        # An orphan FV with no owning FG still gets its DROP_FV.
        k, o = _applied_fv_object("LONELY_FV")
        applied = AppliedState(objects={k: o})
        plan = generate_plan(
            SpecBatch(specs=[], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        drop_names = {op.name for op in plan.ops if op.kind == OpKind.DROP_FV}
        assert drop_names == {"LONELY_FV"}

    def test_local_create_precedes_all_orphan_drops(self) -> None:
        # A local CREATE_ENTITY plus orphan FG+members: the CREATE stays
        # ahead of every DROP (rename atomicity), and among the DROPs the FG
        # still precedes its members.
        from snowflake.ml.feature_store.decl.spec_models import Entity

        entity = Entity.model_validate(
            {
                "kind": "Entity",
                "name": "NEW_ENTITY",
                "database": "DB",
                "schema_": "SCH",
                "join_keys": [{"name": "USER_ID", "type": "StringType"}],
            }
        )
        applied = self._live_graph_applied()
        plan = generate_plan(
            SpecBatch(specs=[entity], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        create_idxs = [i for i, op in enumerate(plan.ops) if op.kind == OpKind.CREATE_ENTITY]
        drop_idxs = [i for i, op in enumerate(plan.ops) if op.kind in (OpKind.DROP_FG, OpKind.DROP_FV)]
        assert create_idxs, "expected a CREATE_ENTITY op"
        assert max(create_idxs) < min(drop_idxs), "CREATE ops must precede all DROP ops"
        fg_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FG)
        fv_idxs = [i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FV]
        assert all(fg_idx < i for i in fv_idxs)


def _member_fv_model(name: str = "MEMBER_FV", version: str = "V1", output_name: str = "event") -> FeatureView:
    """A FeatureView eligible for RECREATE_FV (structural-fingerprint path)."""
    return FeatureView.model_validate(
        {
            "kind": "StreamingFeatureView",
            "name": name,
            "database": "DB",
            "schema_": "SCH",
            "version": version,
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": output_name, "type": "StringType"},
                }
            ],
        }
    )


def _applied_fv_from_model(model: FeatureView, *, content_hash: str) -> tuple[str, AppliedObject]:
    """Build an applied FV keyed by ``spec_key`` (structural-fingerprint path).

    Args:
        model: The FeatureView model whose ``spec_key`` the applied object
            must share so a batch spec of the same model collides with it.
        content_hash: The applied content hash (set to a stale value to force
            a ``RECREATE_FV`` diff).

    Returns:
        A ``(key, AppliedObject)`` tuple for insertion into ``AppliedState``.
    """
    normalized = model_to_dict(model)
    key = spec_key(normalized)
    return key, AppliedObject(
        key=key,
        kind=normalized.get("kind", ""),
        name=normalized.get("name", ""),
        version=normalized.get("version"),
        content_hash=content_hash,
        spec_payload=normalized,
        from_specification=False,
    )


def _no_change_fg_applied(fg: FeatureGroup) -> tuple[str, AppliedObject]:
    """Build an applied FG whose content hash matches ``fg``.

    Args:
        fg: The local FeatureGroup model; the applied object's ``content_hash``
            is computed so the planner emits ``NO_CHANGE`` for it.

    Returns:
        A ``(key, AppliedObject)`` tuple for insertion into ``AppliedState``.
    """
    payload = model_to_dict(fg)
    state = _applied_with_fg(payload, content_hash=fg_content_hash(payload))
    ((key, obj),) = state.objects.items()
    return key, obj


class TestPlannerFGMemberStillReferenced:
    """Authored-spec rule: a member FV ``DROP_FV`` / ``RECREATE_FV`` must not
    tear down or recreate a hash-matched FeatureGroup that still lists it.  The
    planner refuses the member op with ``FG_MEMBER_STILL_REFERENCED`` and
    leaves the FG as ``NO_CHANGE`` (never invents ``DROP_FG`` / ``CREATE_FG``).
    """

    def test_no_change_fg_blocks_member_drop_fv(self) -> None:
        fg = _fg_model(name="MY_FG", refs=[FeatureViewRef(name="MEMBER_FV", version="V1")])
        objects: dict[str, AppliedObject] = {}
        fg_key, fg_obj = _no_change_fg_applied(fg)
        objects[fg_key] = fg_obj
        m_key, m_obj = _applied_fv_object("MEMBER_FV")
        objects[m_key] = m_obj
        applied = AppliedState(objects=objects)

        plan = generate_plan(
            SpecBatch(specs=[fg], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        kinds = [op.kind for op in plan.ops]
        assert OpKind.DROP_FV not in kinds
        assert OpKind.DROP_FG not in kinds
        assert OpKind.CREATE_FG not in kinds
        assert any(op.kind == OpKind.NO_CHANGE and op.name == "MY_FG" for op in plan.ops)
        assert any(e.code == "FG_MEMBER_STILL_REFERENCED" for e in plan.errors)

    def test_no_change_fg_blocks_member_recreate_fv(self) -> None:
        fg = _fg_model(name="MY_FG", refs=[FeatureViewRef(name="MEMBER_FV", version="V1")])
        member = _member_fv_model(name="MEMBER_FV")
        objects: dict[str, AppliedObject] = {}
        fg_key, fg_obj = _no_change_fg_applied(fg)
        objects[fg_key] = fg_obj
        m_key, m_obj = _applied_fv_from_model(member, content_hash="stale-hash-forces-recreate")
        objects[m_key] = m_obj
        applied = AppliedState(objects=objects)

        plan = generate_plan(
            SpecBatch(specs=[fg, member], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        kinds = [op.kind for op in plan.ops]
        assert OpKind.RECREATE_FV not in kinds
        assert any(op.kind == OpKind.NO_CHANGE and op.name == "MY_FG" for op in plan.ops)
        assert any(e.code == "FG_MEMBER_STILL_REFERENCED" for e in plan.errors)

    def test_incremental_mode_fg_only_no_error(self) -> None:
        # full_directory_mode=False → no orphan pass; member only in applied.
        fg = _fg_model(name="MY_FG", refs=[FeatureViewRef(name="MEMBER_FV", version="V1")])
        objects: dict[str, AppliedObject] = {}
        fg_key, fg_obj = _no_change_fg_applied(fg)
        objects[fg_key] = fg_obj
        m_key, m_obj = _applied_fv_object("MEMBER_FV")
        objects[m_key] = m_obj
        applied = AppliedState(objects=objects)

        plan = generate_plan(
            SpecBatch(specs=[fg], source_files=[]),
            applied,
            _opts(full_directory_mode=False),
        )
        assert OpKind.DROP_FV not in [op.kind for op in plan.ops]
        assert not any(e.code == "FG_MEMBER_STILL_REFERENCED" for e in plan.errors)

    def test_orphan_fg_drops_before_member_recreate(self) -> None:
        # Applied FG orphaned (absent from batch) + local member RECREATE_FV →
        # DROP_FG ordered before RECREATE_FV; no refusal (no batch FG lists it).
        member = _member_fv_model(name="MEMBER_FV")
        objects: dict[str, AppliedObject] = {}
        fg_key, fg_obj = _applied_fg_object("MY_FG", ["MEMBER_FV"])
        objects[fg_key] = fg_obj
        m_key, m_obj = _applied_fv_from_model(member, content_hash="stale-hash")
        objects[m_key] = m_obj
        applied = AppliedState(objects=objects)

        plan = generate_plan(
            SpecBatch(specs=[member], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        kinds = [op.kind for op in plan.ops]
        assert OpKind.RECREATE_FV in kinds
        assert OpKind.DROP_FG in kinds
        assert not any(e.code == "FG_MEMBER_STILL_REFERENCED" for e in plan.errors)
        fg_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_FG)
        rc_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.RECREATE_FV)
        assert fg_idx < rc_idx

    def test_membership_change_create_fg_before_member_recreate(self) -> None:
        # Local FG no longer lists MEMBER_FV (lists OTHER_FV) → destructive
        # CREATE_FG; member changed → RECREATE_FV.  CREATE_FG precedes
        # RECREATE_FV and the member op is NOT refused.
        fg = _fg_model(name="MY_FG", refs=[FeatureViewRef(name="OTHER_FV", version="V1")])
        member = _member_fv_model(name="MEMBER_FV")
        objects: dict[str, AppliedObject] = {}
        fg_key, fg_obj = _applied_fg_object("MY_FG", ["MEMBER_FV"])
        objects[fg_key] = fg_obj
        m_key, m_obj = _applied_fv_from_model(member, content_hash="stale-hash")
        objects[m_key] = m_obj
        applied = AppliedState(objects=objects)

        plan = generate_plan(
            SpecBatch(specs=[fg, member], source_files=[]),
            applied,
            _opts(full_directory_mode=True),
        )
        kinds = [op.kind for op in plan.ops]
        assert OpKind.CREATE_FG in kinds
        assert OpKind.RECREATE_FV in kinds
        assert not any(e.code == "FG_MEMBER_STILL_REFERENCED" for e in plan.errors)
        create_fg_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.CREATE_FG)
        rc_idx = next(i for i, op in enumerate(plan.ops) if op.kind == OpKind.RECREATE_FV)
        assert create_fg_idx < rc_idx


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
