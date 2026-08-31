"""Planner regression: two authored versions of one FeatureView name coexist.

Object identity is ``(name, version)`` (``invariants.spec_key`` /
``state._build_spec_key``).  When both a v1 and a v2 of the same FV name are
authored locally and only v1 is deployed, the plan must keep v1
(``NO_CHANGE``) and add v2 (``CREATE_FV``) — never orphan-``DROP`` the still
authored v1.

The regression this pins: ``dependencies.topological_sort`` keyed its
internal ``name_to_spec`` map by name alone, so the two versions collapsed
(the later one won) and the v1 spec never reached the planner's diff loop.
The deployed v1 then looked like an orphan (its key was absent from
``batch_keys``) and was ``DROP``ped while v2 was ``CREATE``d — exactly the
symptom seen in the failing ``feature_plan`` file.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    model_to_dict,
    spec_key,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import FeatureView
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


def _fv_model(name: str = "STAYS_FV", version: str = "V1") -> FeatureView:
    """A minimal streaming FeatureView keyed by ``(name, version)``."""
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
                    "output_column": {"name": "event", "type": "StringType"},
                }
            ],
        }
    )


def _applied_object(model: Any) -> tuple[str, AppliedObject]:
    normalized = model_to_dict(model)
    key = spec_key(normalized)
    ao = AppliedObject(
        key=key,
        kind=normalized.get("kind", ""),
        name=normalized.get("name", ""),
        version=normalized.get("version"),
        content_hash=structural_fingerprint_hash(normalized),
        spec_payload=normalized,
    )
    return key, ao


def _applied_with(*models: Any) -> AppliedState:
    objects = dict(_applied_object(m) for m in models)
    return AppliedState(objects=objects)


def _batch(*models: Any) -> SpecBatch:
    return SpecBatch(specs=list(models), source_files=[])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVersionedFeatureViewCoexistence:
    def test_adding_v2_keeps_v1_and_creates_v2(self) -> None:
        """Deployed: v1.  Authored locally: v1 + v2.  Plan must keep v1
        (NO_CHANGE) and add v2 (CREATE_FV), with no DROP of v1."""
        fv_v1 = _fv_model(version="V1")
        fv_v2 = _fv_model(version="V2")
        applied = _applied_with(fv_v1)
        batch = _batch(fv_v1, fv_v2)

        plan = generate_plan(batch, applied, PlanOptions(full_directory_mode=True))

        v1_ops = [op for op in plan.ops if op.name == "STAYS_FV" and (op.payload or {}).get("version") == "V1"]
        v2_ops = [op for op in plan.ops if op.name == "STAYS_FV" and (op.payload or {}).get("version") == "V2"]

        assert len(v1_ops) == 1, (
            "The still-authored v1 must produce exactly one op; got "
            f"{[(op.kind.value, (op.payload or {}).get('version')) for op in plan.ops]}"
        )
        assert v1_ops[0].kind == OpKind.NO_CHANGE, (
            "The deployed, still-authored v1 must be NO_CHANGE — not dropped. "
            f"Got {v1_ops[0].kind.value} (reason={v1_ops[0].reason!r})."
        )
        assert len(v2_ops) == 1 and v2_ops[0].kind == OpKind.CREATE_FV, (
            "The newly authored v2 must be CREATE_FV; got "
            f"{[(op.kind.value, (op.payload or {}).get('version')) for op in v2_ops]}"
        )

    def test_v1_is_not_dropped_when_v2_is_added(self) -> None:
        """No DROP_FV may target STAYS_FV when v1 is still authored locally."""
        fv_v1 = _fv_model(version="V1")
        fv_v2 = _fv_model(version="V2")
        applied = _applied_with(fv_v1)
        batch = _batch(fv_v1, fv_v2)

        plan = generate_plan(batch, applied, PlanOptions(full_directory_mode=True))

        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_FV]
        assert drop_ops == [], (
            "A version that is still authored locally must never be orphan-DROPped. "
            f"Got drops: {[(op.name, (op.payload or {}).get('version')) for op in drop_ops]}"
        )


if __name__ == "__main__":
    pytest_driver.main()
