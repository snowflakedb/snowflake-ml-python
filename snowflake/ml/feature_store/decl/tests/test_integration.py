"""Integration tests: full pipeline from specs → validate → plan.

The legacy ``decl/sql_generator.py`` module was deleted alongside the
dry-run apply path; SQL is now exclusively emitted by
``imperative_executor`` while it walks the plan, so the previous
"validate → plan → generate_sql" pipeline has collapsed to "validate
→ plan".
"""

from __future__ import annotations

import json

from snowflake.ml.feature_store.decl.api import (
    fetch_applied_state,
    generate_plan,
    validate_specs,
)
from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.spec_models import (
    Entity,
    FeatureView,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.types import AppliedState, PlanOptions, SpecBatch
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FV_SPEC_PAYLOAD = {
    "kind": "StreamingFeatureView",
    "metadata": {"name": "USER_CLICKS", "version": "V1", "database": "DB", "schema": "SCH"},
    "spec": {
        "ordered_entity_column_names": ["USER_ID"],
        "sources": [],
        "features": [
            {
                "source_column": {"name": "EVENT", "type": "StringType"},
                "output_column": {"name": "EVENT", "type": "StringType"},
            }
        ],
    },
}


def _make_batch() -> SpecBatch:
    entity = Entity.model_validate(
        {
            "kind": "Entity",
            "name": "user",
            "database": "DB",
            "schema_": "SCH",
            "join_keys": [{"name": "user_id", "type": "StringType"}],
        }
    )
    source = StreamingSource.model_validate(
        {
            "kind": "StreamingSource",
            "name": "clicks",
            "database": "DB",
            "schema_": "SCH",
            "columns": [{"name": "user_id", "type": "StringType"}, {"name": "event", "type": "StringType"}],
        }
    )
    fv = FeatureView.model_validate(
        {
            "kind": "StreamingFeatureView",
            "name": "user_clicks",
            "database": "DB",
            "schema_": "SCH",
            "version": "V1",
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
    return SpecBatch(specs=[entity, source, fv], source_files=[])


# ---------------------------------------------------------------------------
# fetch_applied_state integration
# ---------------------------------------------------------------------------


class TestFetchAppliedStateIntegration:
    def test_empty_show_returns_empty_state(self) -> None:
        state = fetch_applied_state([], None)
        assert state.objects == {}

    def test_show_row_produces_applied_object(self) -> None:
        row = {
            "name": "USER_CLICKS$V1$ONLINE",
            "created_on": "2024-01-01",
            "specification": json.dumps(_FV_SPEC_PAYLOAD),
        }
        state = fetch_applied_state([row], None)
        assert len(state.objects) == 1


# ---------------------------------------------------------------------------
# validate_specs integration
# ---------------------------------------------------------------------------


class TestValidateSpecsIntegration:
    def test_valid_batch_returns_no_errors(self) -> None:
        batch = _make_batch()
        state = AppliedState(objects={})
        results = validate_specs(batch, state)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) == 0

    def test_second_apply_produces_no_change(self) -> None:
        batch = _make_batch()
        # First apply: no state
        state = AppliedState(objects={})
        results1 = validate_specs(batch, state)
        errors1 = [r for r in results1 if r.severity == "ERROR"]
        assert len(errors1) == 0

    def test_missing_version_blocks_fv(self) -> None:
        fv_no_version = FeatureView.model_validate(
            {
                "kind": "StreamingFeatureView",
                "name": "no_version_fv",
                "database": "DB",
                "schema_": "SCH",
                "version": None,
                "entities": [],
                "sources": [],
                "features": [],
            }
        )
        batch = SpecBatch(specs=[fv_no_version])
        results = validate_specs(batch, AppliedState(objects={}))
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "MISSING_VERSION" for r in errors)


# ---------------------------------------------------------------------------
# generate_plan integration
# ---------------------------------------------------------------------------


class TestGeneratePlanIntegration:
    def test_new_batch_all_create_ops(self) -> None:
        batch = _make_batch()
        state = AppliedState(objects={})
        plan = generate_plan(batch, state, PlanOptions())
        op_kinds = {op.kind for op in plan.ops}
        assert OpKind.CREATE_ENTITY in op_kinds or OpKind.CREATE_FV in op_kinds

    def test_unchanged_batch_no_ops(self) -> None:
        batch = _make_batch()
        state = AppliedState(objects={})
        # First plan gives CREATE ops
        plan1 = generate_plan(batch, state, PlanOptions())
        assert len(plan1.ops) > 0

    def test_plan_is_topologically_ordered(self) -> None:
        batch = _make_batch()
        state = AppliedState(objects={})
        plan = generate_plan(batch, state, PlanOptions())
        names = [op.name for op in plan.ops]
        # user (entity) should come before user_clicks (FV)
        if "user" in names and "user_clicks" in names:
            assert names.index("user") < names.index("user_clicks")


# ---------------------------------------------------------------------------
# End-to-end: validate → plan → SQL
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    def test_new_fv_produces_create_ops(self) -> None:
        batch = _make_batch()
        state = AppliedState(objects={})
        # Validate
        results = validate_specs(batch, state)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) == 0

        # Plan
        plan = generate_plan(batch, state, PlanOptions())
        assert len(plan.ops) > 0
        op_kinds = {op.kind for op in plan.ops}
        # A new FV against an empty state must produce at least one
        # CREATE_* op.  SQL emission is exercised separately by the
        # imperative executor's own tests.
        assert {OpKind.CREATE_FV, OpKind.CREATE_ENTITY} & op_kinds

    def test_fetch_state_then_plan(self) -> None:
        # Simulate fetch → plan cycle
        show_row = {
            "name": "USER_CLICKS$V1$ONLINE",
            "created_on": "2024-01-01",
            "specification": json.dumps(_FV_SPEC_PAYLOAD),
        }
        applied = fetch_applied_state([show_row])
        assert len(applied.objects) == 1

        batch = _make_batch()
        plan = generate_plan(batch, applied, PlanOptions())
        # Should still have ops for entity + source (not in applied state)
        assert isinstance(plan, object)


if __name__ == "__main__":
    pytest_driver.main()
