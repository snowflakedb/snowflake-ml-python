"""Red tests for the BUG_BASH §11 ``UPDATE_ENTITY USER_ID`` failure.

Plan: ``plans/step11_update_entity_bug_*.plan.md`` (Solution B).

The bug-bash doc step 10 edits ``$DECL/entities/USER_ID.yaml``'s
``description:`` field, then step 11 expects ``snow feature plan`` to emit a
single non-destructive ``UPDATE_ENTITY USER_ID`` row.  Today the planner
emits ``NO_CHANGE`` because three layered defects all silence the diff:

- H2 — :func:`invariants._structural_fingerprint` reduces an Entity
  dict to ``{name, version, columns}`` and never reads
  ``description`` / ``join_keys``, so two entity dicts that differ only
  in ``description`` hash identically.
- H3 — :func:`state._build_entity_object` reads the deployed tag's
  ``COMMENT`` into ``details["comment"]`` but never plumbs it into
  ``spec_payload["description"]``, so even if the fingerprint were
  widened, the applied side would always look like
  ``description=None`` and a clean re-plan would emit a phantom
  ``UPDATE_ENTITY``.

(H1 — the BUG_BASH doc / verify_bug_bash.sh wrote ``desc:``, which the
Pydantic ``Entity`` model silently dropped because the field is
``description`` with no alias — is fixed at the doc/script layer in
Solution B; the test here pins the canonical ``description:`` shape so a
future regression of the loader is caught.)

These tests are red on ``main`` and turn green when the chosen solution
ships.  See the plan for the truth table that explains why a partial fix
is worse than the current symptom.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    spec_key,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import Entity
from snowflake.ml.feature_store.decl.state import _build_entity_object
from snowflake.ml.feature_store.decl.types import (
    AppliedState,
    ObjectKind,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.feature_store.spec.enums import (
    ENTITY_TAG_PREFIX as _ENTITY_TAG_PREFIX,
)

_DB = "JKEW_DB"
_SCHEMA = "JKEW_SCHEMA"
_NAME = "USER_ID"


# ---------------------------------------------------------------------------
# Helpers — keep the test file standalone so it can be deleted independently
# ---------------------------------------------------------------------------


def _entity_yaml_with_description(description: str) -> str:
    """Canonical authoring YAML for the bug-bash USER_ID entity.

    Solution B fixes BUG_BASH.md and scripts/verify_bug_bash.sh to use
    ``description:`` (matching the SpecBase model and the exporter); the
    YAML below mirrors that canonical shape.

    Args:
        description: Value to embed under the ``description:`` key.

    Returns:
        A YAML document string for the canonical USER_ID entity.
    """
    return (
        "kind: Entity\n"
        f"name: {_NAME}\n"
        f'description: "{description}"\n'
        "join_keys:\n"
        f"  - name: {_NAME}\n"
        "    type: StringType\n"
    )


def _entity_dict_with_description(description: str | None) -> dict[str, Any]:
    """Authoring-format dict for an Entity, optionally carrying description."""
    spec: dict[str, Any] = {
        "kind": "Entity",
        "name": _NAME,
        "database": _DB,
        "schema": _SCHEMA,
        "join_keys": [{"name": _NAME, "type": "StringType"}],
    }
    if description is not None:
        spec["description"] = description
    return spec


def _show_tags_row(comment: str | None) -> dict[str, Any]:
    """A single SHOW TAGS row in the shape ``_build_entity_object`` consumes."""
    row: dict[str, Any] = {
        "name": f"{_ENTITY_TAG_PREFIX}{_NAME}",
        "database_name": _DB,
        "schema_name": _SCHEMA,
        "allowed_values": f'["{_NAME}"]',
    }
    if comment is not None:
        row["comment"] = comment
    return row


def _applied_state_for_entity(
    *,
    deployed_description: str | None,
) -> AppliedState:
    """Build an :class:`AppliedState` whose only entry is the deployed entity.

    After Solution B lands, ``_build_entity_object`` plumbs the SHOW TAGS
    ``comment`` into ``spec_payload["description"]``, and the Entity-aware
    branch of ``_structural_fingerprint`` includes it in the hash.  The
    helper here just delegates to ``_build_entity_object`` so the test
    state stays bit-identical to what the live read path produces.

    Args:
        deployed_description: Value to surface as the deployed tag's
            ``COMMENT`` (i.e., the description on the applied side).

    Returns:
        Applied state containing exactly one entity object.
    """
    row = _show_tags_row(deployed_description)
    obj = _build_entity_object(row, default_db=_DB, default_schema=_SCHEMA)
    assert obj is not None, "test fixture: _build_entity_object returned None"
    return AppliedState(objects={obj.key: obj})


def _batch_with_entity(local_description: str | None) -> SpecBatch:
    """Build a :class:`SpecBatch` containing only the local entity spec."""
    spec_dict = _entity_dict_with_description(local_description)
    # The loader runs ``_dict_to_spec`` which calls ``Entity.model_validate``;
    # reproduce that path here so the batch shape matches a live load.
    model = Entity.model_validate(spec_dict)
    return SpecBatch(specs=[model], source_files=[])


# ---------------------------------------------------------------------------
# H1 supplemental — authoring contract for ``description:``
# ---------------------------------------------------------------------------


class TestEntityYamlAuthoringContract:
    """Pin the BUG_BASH §10 authoring shape after Solution B's doc fix.

    The YAML key ``description:`` is the canonical authoring shape (it
    matches the ``SpecBase.description`` field and round-trips through the
    exporter).  Any regression that drops the value silently (as
    Pydantic v2 does for unknown keys) is caught here.
    """

    def test_yaml_description_populates_model_description(self) -> None:
        raw = yaml.safe_load(_entity_yaml_with_description("Bug-bash user identifier."))
        entity = Entity.model_validate(raw)
        assert entity.description == "Bug-bash user identifier."

    def test_edited_yaml_description_round_trips_through_model(self) -> None:
        raw = yaml.safe_load(_entity_yaml_with_description("Bug-bash user identifier — updated."))
        entity = Entity.model_validate(raw)
        assert entity.description == "Bug-bash user identifier — updated."


# ---------------------------------------------------------------------------
# H2 — Entity structural fingerprint must include ``description``
# ---------------------------------------------------------------------------


class TestEntityFingerprintIncludesDescription:
    """``structural_fingerprint_hash`` for Entity kinds must hash description.

    Today the helper reduces every Entity dict to ``{name, version="",
    columns=[]}``, so two entities that differ only in ``description``
    produce identical hashes and the planner reports ``NO_CHANGE`` for a
    desc-only edit.  Solution B widens the Entity branch of
    ``_structural_fingerprint`` to include ``description`` (and the
    ``join_keys`` schema) so a non-destructive ``UPDATE_ENTITY`` op is
    emitted instead.
    """

    def test_description_only_diff_changes_entity_hash(self) -> None:
        before = _entity_dict_with_description("Bug-bash user identifier.")
        after = _entity_dict_with_description("Bug-bash user identifier — updated.")
        before_hash = structural_fingerprint_hash(before)
        after_hash = structural_fingerprint_hash(after)
        assert before_hash != after_hash, (
            "Editing only Entity.description must change the structural "
            "fingerprint so the planner emits UPDATE_ENTITY instead of "
            "NO_CHANGE.  Got identical hashes — fingerprint is too coarse "
            "for entities."
        )

    def test_join_keys_diff_changes_entity_hash(self) -> None:
        """Adding a join key must change the hash (regression guard)."""
        before = _entity_dict_with_description("d")
        after = _entity_dict_with_description("d")
        after["join_keys"] = [
            {"name": _NAME, "type": "StringType"},
            {"name": "TENANT_ID", "type": "StringType"},
        ]
        assert structural_fingerprint_hash(before) != structural_fingerprint_hash(after)

    def test_no_description_hashes_equal_to_empty_string_description(self) -> None:
        """Absent ``description`` and empty ``description`` are the same.

        Treat a missing description as equivalent to an empty one — the
        deployed tag's ``COMMENT`` may be unset or ``''`` and we want both
        to round-trip cleanly through the round-trip green guard below.
        """
        without = _entity_dict_with_description(None)
        with_empty = _entity_dict_with_description("")
        assert structural_fingerprint_hash(without) == structural_fingerprint_hash(with_empty)


# ---------------------------------------------------------------------------
# H3 — applied state must surface the deployed COMMENT as ``description``
# ---------------------------------------------------------------------------


class TestBuildEntityObjectPopulatesDescription:
    """``_build_entity_object`` must put the tag ``COMMENT`` into ``description``.

    Today the helper writes ``comment`` into ``details["comment"]`` only
    and the ``spec_payload`` lacks any description field.  Once H2 widens
    the fingerprint to read ``description``, the applied side has to
    surface the deployed comment under that key — otherwise an unedited
    re-plan emits a phantom ``UPDATE_ENTITY`` (local has description,
    applied has none → hashes diverge).
    """

    def test_comment_populates_spec_payload_description(self) -> None:
        row = _show_tags_row("Bug-bash user identifier.")
        obj = _build_entity_object(row, default_db=_DB, default_schema=_SCHEMA)
        assert obj is not None
        assert obj.spec_payload.get("description") == "Bug-bash user identifier."

    def test_missing_comment_leaves_description_unset_or_empty(self) -> None:
        row = _show_tags_row(None)
        obj = _build_entity_object(row, default_db=_DB, default_schema=_SCHEMA)
        assert obj is not None
        assert obj.spec_payload.get("description", "") == ""

    def test_content_hash_includes_description(self) -> None:
        """Pin the symmetry: same desc on both sides → same hash."""
        row = _show_tags_row("Bug-bash user identifier.")
        obj = _build_entity_object(row, default_db=_DB, default_schema=_SCHEMA)
        local = _entity_dict_with_description("Bug-bash user identifier.")
        assert obj is not None
        assert obj.content_hash == structural_fingerprint_hash(local), (
            "Applied entity content_hash must match the local "
            "structural_fingerprint_hash for the same description so a "
            "clean re-plan stays NO_CHANGE."
        )


# ---------------------------------------------------------------------------
# End-to-end — desc-edited entity YAML produces UPDATE_ENTITY
# ---------------------------------------------------------------------------


class TestDescriptionEditPlansUpdateEntity:
    """BUG_BASH §10–11: editing ``description:`` produces ``UPDATE_ENTITY``."""

    def test_description_edit_emits_single_update_entity_op(self) -> None:
        """The local YAML's description differs from the deployed tag's
        comment → exactly one ``UPDATE_ENTITY USER_ID`` op, no others."""
        applied = _applied_state_for_entity(deployed_description="Bug-bash user identifier.")
        batch = _batch_with_entity("Bug-bash user identifier — updated.")

        plan = generate_plan(
            batch,
            applied,
            PlanOptions(),
            database=_DB,
            schema=_SCHEMA,
        )

        update_ops = [op for op in plan.ops if op.kind == OpKind.UPDATE_ENTITY]
        non_no_change = [op for op in plan.ops if op.kind != OpKind.NO_CHANGE]
        assert len(update_ops) == 1, (
            "Editing Entity.description must emit exactly one "
            f"UPDATE_ENTITY op; got plan ops "
            f"{[(op.kind.value, op.name) for op in plan.ops]!r}"
        )
        assert update_ops[0].name == _NAME
        assert update_ops[0].destructive is False
        assert non_no_change == update_ops, (
            "Editing only Entity.description must not emit any other op; "
            f"got {[(op.kind.value, op.name) for op in non_no_change]!r}"
        )

    def test_unedited_yaml_emits_no_change(self) -> None:
        """Round-trip green guard: when local ``description`` matches the
        deployed tag ``comment``, every op must be ``NO_CHANGE``.

        Without this guard, a fix that only widens the fingerprint (H2)
        but doesn't symmetrize the applied side (H3) would silently
        regress into a phantom ``UPDATE_ENTITY`` on every clean re-plan.
        """
        applied = _applied_state_for_entity(deployed_description="Bug-bash user identifier.")
        batch = _batch_with_entity("Bug-bash user identifier.")

        plan = generate_plan(
            batch,
            applied,
            PlanOptions(),
            database=_DB,
            schema=_SCHEMA,
        )

        non_no_change = [op for op in plan.ops if op.kind != OpKind.NO_CHANGE]
        assert non_no_change == [], (
            "Local description matches deployed tag comment — every op "
            f"must be NO_CHANGE; got {[(op.kind.value, op.name) for op in non_no_change]!r}"
        )


# ---------------------------------------------------------------------------
# Bug-bash applied-state helper symmetry
# ---------------------------------------------------------------------------


class TestAppliedStateKeyMatchesPlannerLookup:
    """The planner's ``spec_key`` lookup must hit the applied entity.

    This is a regression guard: if a future refactor changes the key
    format on either side, the planner would fall through the
    ``applied is None`` branch and emit ``CREATE_ENTITY`` for an
    already-deployed tag.  The end-to-end tests above would fail with a
    misleading error — this one isolates the key-derivation half.
    """

    def test_planner_finds_deployed_entity_in_applied_state(self) -> None:
        applied = _applied_state_for_entity(deployed_description="d")
        local = _entity_dict_with_description("d")
        key = spec_key(local, database=_DB, schema=_SCHEMA)
        assert key in applied.objects, (
            f"spec_key={key!r} not found in applied_state keys " f"{list(applied.objects.keys())!r}"
        )
        # And the lookup yields an Entity AppliedObject.
        obj = applied.objects[key]
        assert obj.kind == ObjectKind.ENTITY
        assert obj.name == _NAME


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
