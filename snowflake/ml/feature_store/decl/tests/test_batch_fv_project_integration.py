"""Integration test: load minimal batch FV project and generate CREATE plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.manifest import FSTarget
from snowflake.ml.feature_store.decl.types import AppliedState, PlanOptions
from snowflake.ml.test_utils import pytest_driver

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "batch_fv_minimal"
_BUGBASH_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "batch_fv_bugbash_names"


def test_batch_fv_project_load_plan_create() -> None:
    target = FSTarget(
        name="DEV",
        account_identifier="ORG-ACCOUNT",
        database="BFV_DB",
        schema="BFV_SC",
        role="TESTROLE",
    )
    batch = decl_api.load_project(_FIXTURE_ROOT, target=target)
    decl_api.resolve_datasource_columns(batch)

    fv_spec: Any = next(s for s in batch.specs if getattr(s, "name", "") == "BATCH_SIMPLE")
    src0 = fv_spec.sources[0]
    table = src0.table if hasattr(src0, "table") else src0.get("table")
    assert table == "RAW_EVENTS", "Batch FV source must inherit table from BatchSource for apply/plan payloads"

    results = decl_api.validate_specs(
        batch,
        AppliedState(objects={}),
        target_database=target.database,
        target_schema=target.schema,
    )
    errors = [r for r in results if r.severity == "ERROR"]
    assert not errors, [r.message for r in errors]

    plan = decl_api.generate_plan(
        batch,
        AppliedState(objects={}),
        PlanOptions(),
        database=target.database,
        schema=target.schema,
    )
    kinds = {op.kind.value for op in plan.ops}
    assert "CREATE_FV" in kinds
    fv_create = [op for op in plan.ops if op.kind.value == "CREATE_FV" and op.name == "BATCH_SIMPLE"]
    assert len(fv_create) == 1
    assert fv_create[0].payload.get("kind") == "BatchFeatureView"
    pl_src = fv_create[0].payload.get("sources") or []
    assert pl_src and pl_src[0].get("table") == "RAW_EVENTS"

    raw = decl_api.serialize_plan(
        plan,
        target.database,
        target.schema,
        list(batch.source_files or []),
        target_name=target.name,
    )
    round_file = decl_api.deserialize_plan(raw)
    assert len(round_file.plan.ops) == len(plan.ops)
    rt_create = [op for op in round_file.plan.ops if op.kind.value == "CREATE_FV" and op.name == "BATCH_SIMPLE"]
    assert len(rt_create) == 1
    rt_src = (rt_create[0].payload.get("sources") or [])[0]
    assert rt_src.get("table") == "RAW_EVENTS"


def test_batch_fv_bugbash_doc_names_resolve_plan_and_roundtrip() -> None:
    """Doc/BATCH_FV_BUG_BASH.md object names: name+source_type FV refs inherit table."""
    target = FSTarget(
        name="DEV",
        account_identifier="ORG-ACCOUNT",
        database="BFV_DB",
        schema="BFV_SC",
        role="TESTROLE",
    )
    batch = decl_api.load_project(_BUGBASH_FIXTURE, target=target)
    decl_api.resolve_datasource_columns(batch)

    fv_spec: Any = next(s for s in batch.specs if getattr(s, "name", "") == "MY_BATCH_FV_BATCH_DECL")
    src0 = fv_spec.sources[0]
    table = src0.table if hasattr(src0, "table") else src0.get("table")
    assert table == "RAW_EVENTS_BATCH_DECL"

    results = decl_api.validate_specs(
        batch,
        AppliedState(objects={}),
        target_database=target.database,
        target_schema=target.schema,
    )
    errors = [r for r in results if r.severity == "ERROR"]
    assert not errors, [r.message for r in errors]

    plan = decl_api.generate_plan(
        batch,
        AppliedState(objects={}),
        PlanOptions(),
        database=target.database,
        schema=target.schema,
    )
    fv_create = [op for op in plan.ops if op.kind.value == "CREATE_FV" and op.name == "MY_BATCH_FV_BATCH_DECL"]
    assert len(fv_create) == 1
    payload = fv_create[0].payload
    assert payload.get("online") is True, "Bug-bash BFV authors online: true for snow feature query"
    # W-D: the bug-bash fixture must use a short refresh cadence (1 minute) so the
    # ``snow feature query`` poll in ``scripts/verify_batch_fv_bug_bash.sh`` §6
    # has time to observe non-empty rows within ``BATCH_BUGBASH_MATERIALIZE_TIMEOUT_SEC``
    # (default 180s).  Pin both surfaces so a future regression of the YAML to
    # ``1 hour`` is caught at unit-test time.
    target_lag_sec = payload.get("target_lag_sec")
    if target_lag_sec is None and payload.get("target_lag") is not None:
        tl = payload["target_lag"]
        target_lag_sec = int(tl) if isinstance(tl, int) else None
    assert target_lag_sec == 60, (
        f"bug-bash fixture must set target_lag=1 minute (60s); got {target_lag_sec!r}. "
        f"Update fixtures/batch_fv_bugbash_names/sources/feature_views/MY_BATCH_FV_BATCH_DECL.yaml."
    )
    assert payload.get("refresh_freq") == "1 minute", (
        f"bug-bash fixture must set refresh_freq=1 minute; got {payload.get('refresh_freq')!r}. "
        f"See docs/BATCH_FV_BUG_BASH.md §5."
    )
    pl_src = payload.get("sources") or []
    assert pl_src[0].get("source_type") == "Batch"
    assert pl_src[0].get("table") == "RAW_EVENTS_BATCH_DECL"

    raw = decl_api.serialize_plan(
        plan,
        target.database,
        target.schema,
        list(batch.source_files or []),
        target_name=target.name,
    )
    round_file = decl_api.deserialize_plan(raw)
    rt = [op for op in round_file.plan.ops if op.kind.value == "CREATE_FV" and op.name == "MY_BATCH_FV_BATCH_DECL"]
    assert len(rt) == 1
    rt0 = (rt[0].payload.get("sources") or [])[0]
    assert rt0.get("table") == "RAW_EVENTS_BATCH_DECL"
    assert rt0.get("source_type") == "Batch"


def test_resolve_datasource_columns_batch_source_case_insensitive() -> None:
    """FV source name matches BatchSource case-insensitively (H4)."""
    target = FSTarget(
        name="DEV",
        account_identifier="ORG-ACCOUNT",
        database="D",
        schema="S",
        role="R",
    )
    batch = decl_api.load_project(_BUGBASH_FIXTURE, target=target)
    fv: Any = next(s for s in batch.specs if getattr(s, "name", "") == "MY_BATCH_FV_BATCH_DECL")
    src = fv.sources[0]
    if hasattr(src, "model_copy"):
        fv.sources[0] = src.model_copy(update={"name": "events_batch_decl"})
    elif isinstance(src, dict):
        src["name"] = "events_batch_decl"
    decl_api.resolve_datasource_columns(batch)
    src0 = fv.sources[0]
    table = src0.table if hasattr(src0, "table") else src0.get("table")
    assert table == "RAW_EVENTS_BATCH_DECL"


if __name__ == "__main__":
    pytest_driver.main()
