"""Captured-DESCRIBE golden replay for the planner re-validation contract.

The existing :class:`TestBugBashGoldenSpecRoundTrip` (in
``test_planner_revalidate_identical_spec.py``) only asserts
``_full_spec_hash`` equality between the captured golden and
``compile_to_spec(local)``.  It does NOT exercise the
validator/planner against a ``from_specification=True`` AppliedObject
built from the golden.  As a consequence, a regression that re-broke
the nested-features lookup in ``_check_fv_column_evolution`` /
``_check_destructive`` or the strict-``<`` semantics in
``_check_versions`` would silently keep that hash test green while
re-creating the BUG_BASH §9 / §11 / §14 cascade in the live env.

This module pins that gap by replaying
``snowflake/ml/feature_store/decl/tests/golden_specs/USER_CLICK_STATS_DECL.json``
— a frozen capture of ``DESCRIBE ONLINE FEATURE TABLE ... TYPE =
SPECIFICATION`` for the BUG_BASH §5 FV — through the public validator
and planner entry points.  Each test asserts the contract that a
re-plan against an unchanged spec must be a clean ``NO_CHANGE``
(zero ERRORRs, zero ``COLUMN_ADDED`` warnings, exactly one ``NO_CHANGE``
plan op for the FV).  See ``plans/planner_revalidate_identical_spec.plan.md``
for the cascade write-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    spec_key,
    structural_fingerprint_hash,
    validate_specs,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import (
    Entity,
    FSColumn,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.tests.test_planner_revalidate_identical_spec import (
    _BUG_BASH_DB,
    _BUG_BASH_FV_NAME,
    _BUG_BASH_FV_VERSION,
    _BUG_BASH_SCHEMA,
    _bug_bash_fv_model,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    ObjectKind,
    PlanOptions,
    SpecBatch,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Golden fixture loading
# ---------------------------------------------------------------------------


_GOLDEN_DIR = Path(__file__).parent / "golden_specs"
_BUG_BASH_GOLDEN = _GOLDEN_DIR / "USER_CLICK_STATS_DECL.json"


def _build_full_batch() -> SpecBatch:
    """Return the full BUG_BASH §5 batch (entity + source + FV).

    Mirrors what ``loader.load_specs`` returns for the BUG_BASH §5
    ``$DECL/{entities,datasources,feature_views}/`` tree, but built
    from in-memory model constructors so the test stays hermetic (no
    tmp_path round-trip).  Including the source spec in the batch is
    required so ``_check_dependencies`` resolves the FV's
    ``sources[].name`` reference against ``batch_names`` rather than
    raising ``MISSING_SOURCE`` — the ``Datasource`` kind in the
    applied state alone does not satisfy the dependency check
    (``_check_dependencies`` looks for ``StreamingSource`` /
    ``BatchSource`` literally).

    Returns:
        SpecBatch carrying USER_ID entity + CLICKSTREAM_EVENTS source +
        the BUG_BASH FV.
    """
    entity = Entity(
        name="USER_ID",
        database=_BUG_BASH_DB,
        schema_=_BUG_BASH_SCHEMA,
        description="Bug-bash user identifier.",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    source = StreamingSource(
        name="CLICKSTREAM_EVENTS",
        database=_BUG_BASH_DB,
        schema_=_BUG_BASH_SCHEMA,
        type="REST",
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="SESSION_ID", type="StringType"),
            FSColumn(name="PAGE_URL", type="StringType"),
            FSColumn(name="EVENT_TYPE", type="StringType"),
            FSColumn(name="TIMESTAMP", type="TimestampType"),
            FSColumn(name="TIME_ON_PAGE_SECONDS", type="DoubleType"),
        ],
    )
    return SpecBatch(specs=[entity, source, _bug_bash_fv_model()], source_files=[])


def _load_golden() -> dict[str, Any]:
    """Load the captured DESCRIBE TYPE = SPECIFICATION payload.

    Returns:
        The parsed golden dict — the verbatim shape Snowflake returns
        from ``DESCRIBE ONLINE FEATURE TABLE ... TYPE = SPECIFICATION``
        for the BUG_BASH §5 FV (features nested under ``spec.features``,
        with Snowflake-stamped ``metadata.oft_id`` / ``offline_configs`` /
        ``online_store_type`` keys present).
    """
    golden: dict[str, Any] = json.loads(_BUG_BASH_GOLDEN.read_text())
    return golden


def _build_full_applied_state(
    *,
    use_fv_hash: str | None = None,
) -> AppliedState:
    """Build an :class:`AppliedState` for the BUG_BASH flow off the golden.

    Mirrors what ``state.fetch_applied_state`` produces in the live env
    after step 6 deploy, but with the FV's ``spec_payload`` sourced from
    the captured golden instead of synthesised via ``compile_to_spec``.
    This is the closer-to-production input shape that catches regressions
    the synthetic-payload tests miss.

    The applied state contains:

    - **Entity** (USER_ID): structural-fingerprint hash, ``from_specification=False``.
    - **Datasource** (CLICKSTREAM_EVENTS): structural-fingerprint hash,
      ``from_specification=True`` (recovered from FV ``spec.sources[]``).
    - **StreamingFeatureView** (USER_CLICK_STATS_DECL): ``content_hash =
      _full_spec_hash(golden)``, ``from_specification=True``,
      ``spec_payload = golden``.

    Args:
        use_fv_hash: Optional override for the FV's ``content_hash`` —
            pass ``"deadbeef" * 8`` to force ``_check_idempotency`` to
            MISS and exercise the fall-through validator path (this is
            the regression guard against re-broken nested-features
            lookups in ``_check_fv_column_evolution`` / ``_check_destructive``).

    Returns:
        AppliedState ready to feed into ``validate_specs`` /
        ``generate_plan``.
    """
    objects: dict[str, AppliedObject] = {}
    golden = _load_golden()

    entity_payload = {
        "kind": ObjectKind.ENTITY,
        "name": "USER_ID",
        "database": _BUG_BASH_DB,
        "schema": _BUG_BASH_SCHEMA,
        "description": "Bug-bash user identifier.",
        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
    }
    entity_key = f"{ObjectKind.ENTITY}:{_BUG_BASH_DB}.{_BUG_BASH_SCHEMA}:USER_ID"
    objects[entity_key] = AppliedObject(
        key=entity_key,
        kind=ObjectKind.ENTITY,
        name="USER_ID",
        content_hash=structural_fingerprint_hash(entity_payload),
        spec_payload=entity_payload,
        from_specification=False,
        details={"join_keys": ["USER_ID"], "comment": "Bug-bash user identifier."},
    )

    source_columns = [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "SESSION_ID", "type": "StringType"},
        {"name": "PAGE_URL", "type": "StringType"},
        {"name": "EVENT_TYPE", "type": "StringType"},
        {"name": "TIMESTAMP", "type": "TimestampType"},
        {"name": "TIME_ON_PAGE_SECONDS", "type": "DoubleType"},
    ]
    source_payload = {
        "kind": ObjectKind.DATASOURCE,
        "name": "CLICKSTREAM_EVENTS",
        "database": _BUG_BASH_DB,
        "schema": _BUG_BASH_SCHEMA,
        "source_type": "Stream",
        "columns": source_columns,
    }
    source_key = f"{ObjectKind.DATASOURCE}:{_BUG_BASH_DB}.{_BUG_BASH_SCHEMA}:CLICKSTREAM_EVENTS"
    objects[source_key] = AppliedObject(
        key=source_key,
        kind=ObjectKind.DATASOURCE,
        name="CLICKSTREAM_EVENTS",
        content_hash=structural_fingerprint_hash(source_payload),
        spec_payload=source_payload,
        from_specification=True,
        details={"source_type": "Stream", "column_count": len(source_columns)},
    )

    fv_dict = {
        "kind": "StreamingFeatureView",
        "name": _BUG_BASH_FV_NAME,
        "version": _BUG_BASH_FV_VERSION,
        "database": _BUG_BASH_DB,
        "schema": _BUG_BASH_SCHEMA,
    }
    fv_key = spec_key(fv_dict, database=_BUG_BASH_DB, schema=_BUG_BASH_SCHEMA)
    objects[fv_key] = AppliedObject(
        key=fv_key,
        kind="StreamingFeatureView",
        name=_BUG_BASH_FV_NAME,
        version=_BUG_BASH_FV_VERSION,
        content_hash=use_fv_hash if use_fv_hash is not None else _full_spec_hash(golden),
        spec_payload=golden,
        from_specification=True,
    )

    return AppliedState(objects=objects)


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _BUG_BASH_GOLDEN.exists(),
    reason="Golden spec USER_CLICK_STATS_DECL.json not yet captured.",
)
class TestValidateSpecsAgainstGoldenDescribe:
    """``validate_specs`` against the captured DESCRIBE payload must be clean."""

    def test_validate_specs_against_golden_describe_no_errors(self) -> None:
        """A re-plan against the captured payload produces zero validation
        ERROR results.

        This is the end-to-end shape of BUG_BASH §9 — the live
        ``snow feature plan`` after step 6 deploy.  A regression that
        re-broke the strict-``<`` ``_check_versions`` semantics or the
        nested-features lookup would surface here as a non-empty
        ERROR list (typically ``VERSION_CONFLICT``).
        """
        applied_state = _build_full_applied_state()
        batch = _build_full_batch()

        results = validate_specs(
            batch,
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        errors = [r for r in results if r.severity == "ERROR"]
        assert errors == [], (
            "Re-plan against captured DESCRIBE payload must produce zero "
            f"ERROR results; got {[(r.code, r.message) for r in errors]!r}"
        )

    def test_validate_specs_against_golden_describe_no_column_added(self) -> None:
        """The re-plan must emit zero ``COLUMN_ADDED`` warnings.

        Pins the H1 nested-features fix
        (``_check_fv_column_evolution`` reading from
        ``spec_payload['spec']['features']`` when
        ``from_specification=True``) against the golden's actual nested
        shape.  A regression to ``spec_payload.get('features', [])``
        would surface here as one ``COLUMN_ADDED`` warning per
        ``features[].output_column``.
        """
        applied_state = _build_full_applied_state()
        batch = _build_full_batch()

        results = validate_specs(
            batch,
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        column_added = [r for r in results if r.code == "COLUMN_ADDED"]
        assert column_added == [], (
            "Re-plan against captured DESCRIBE payload must produce zero "
            f"COLUMN_ADDED warnings; got {[r.message for r in column_added]!r}"
        )


@pytest.mark.skipif(
    not _BUG_BASH_GOLDEN.exists(),
    reason="Golden spec USER_CLICK_STATS_DECL.json not yet captured.",
)
class TestGeneratePlanAgainstGoldenDescribe:
    """``generate_plan`` against the captured DESCRIBE payload must be NO_CHANGE."""

    def test_generate_plan_against_golden_describe_is_no_change(self) -> None:
        """The planner must emit exactly one op for the FV, and it must
        be ``NO_CHANGE``.

        This is the planner-side counterpart to
        :meth:`TestValidateSpecsAgainstGoldenDescribe.test_validate_specs_against_golden_describe_no_errors`.
        Even if the validator stays clean, a regression in
        ``compute_local_spec_hash`` / ``_full_spec_hash`` would surface
        here as a phantom ``RECREATE_FV`` (the planner's destructive op
        for hash mismatches on FV kinds).
        """
        applied_state = _build_full_applied_state()
        batch = _build_full_batch()

        plan = generate_plan(
            batch,
            applied_state,
            PlanOptions(),
            database=_BUG_BASH_DB,
            schema=_BUG_BASH_SCHEMA,
        )

        fv_ops = [op for op in plan.ops if op.name == _BUG_BASH_FV_NAME]
        assert (
            len(fv_ops) == 1
        ), f"expected exactly one op for {_BUG_BASH_FV_NAME}; got {[op.kind.value for op in fv_ops]!r}"
        assert fv_ops[0].kind == OpKind.NO_CHANGE, (
            "Re-plan against captured DESCRIBE payload must produce a "
            f"NO_CHANGE op for {_BUG_BASH_FV_NAME}; got "
            f"kind={fv_ops[0].kind.value!r} reason={fv_ops[0].reason!r}"
        )


@pytest.mark.skipif(
    not _BUG_BASH_GOLDEN.exists(),
    reason="Golden spec USER_CLICK_STATS_DECL.json not yet captured.",
)
class TestValidateSpecsWithDivergentHash:
    """Force ``_check_idempotency`` MISS and pin the fall-through path.

    When the live ``DESCRIBE`` payload's ``content_hash`` does not match
    ``compute_local_spec_hash(local)`` for any reason (extra
    Snowflake-stamped fields, normalization order, format-version bump),
    ``_check_idempotency`` returns ``(False, [])`` and validation falls
    through to ``_check_versions`` + ``_check_fv_column_evolution`` +
    ``_check_destructive``.  Without the H1 / H1-extended / H3 fixes
    those three rules cascade into the BUG_BASH §9 symptom even though
    the spec is genuinely unchanged at the column / feature / version
    level.

    This class fires the divergent-hash path explicitly and asserts the
    fall-through validator surface stays clean.
    """

    _DIVERGENT_HASH = "deadbeef" * 8

    def test_validate_specs_with_divergent_hash_no_errors(self) -> None:
        """Divergent ``content_hash`` must not push validation into ERROR.

        Reproduces the live BUG_BASH §9 condition under which the H3
        ``VERSION_CONFLICT`` regression surfaced.  Pins the strict-``<``
        ``_check_versions`` semantics — equal versions on identical
        content remain a NO_CHANGE / no-ERROR outcome regardless of
        hash drift.
        """
        applied_state = _build_full_applied_state(use_fv_hash=self._DIVERGENT_HASH)
        batch = _build_full_batch()

        results: list[ValidationResult] = validate_specs(
            batch,
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        errors = [r for r in results if r.severity == "ERROR"]
        assert errors == [], (
            "Divergent-hash re-plan against captured DESCRIBE payload "
            "must produce zero ERROR results; got "
            f"{[(r.code, r.message) for r in errors]!r}"
        )

    def test_validate_specs_with_divergent_hash_no_column_added(self) -> None:
        """Divergent ``content_hash`` must not push column evolution into
        ``COLUMN_ADDED``.

        Pins the H1 nested-features fix in ``_check_fv_column_evolution``
        on the divergent-hash code path — which is the path the live
        BUG_BASH §9 cascade actually exercises.
        """
        applied_state = _build_full_applied_state(use_fv_hash=self._DIVERGENT_HASH)
        batch = _build_full_batch()

        results = validate_specs(
            batch,
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        column_added = [r for r in results if r.code == "COLUMN_ADDED"]
        assert column_added == [], (
            "Divergent-hash re-plan against captured DESCRIBE payload "
            "must produce zero COLUMN_ADDED warnings; got "
            f"{[r.message for r in column_added]!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
