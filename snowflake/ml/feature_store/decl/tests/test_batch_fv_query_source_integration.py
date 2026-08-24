"""End-to-end integration test for query-backed BatchSource fixtures.

Exercises the full Phase 1-7 pipeline against the
``batch_fv_query_source`` fixture tree:

* Two BatchSources — one with an inline ``query: |`` block and one with
  a ``query_file: EVENTS_FILE.sql`` sidecar.
* Two BatchFeatureViews referencing them.
* One Entity providing the ``USER_ID`` join key.

Verified contracts (matching plan §8 Pre/Post):

1. **Initial CREATE plan** — ``load → compile → resolve → validate →
   plan`` produces ``CREATE_FV`` for both FVs and the planner payloads
   carry ``query`` (normalised, sidecar inlined) on ``sources[0]``.
2. **Round-trip NO_CHANGE** — synthesising an ``AppliedState`` via the
   Phase-B3 ``FV_SOURCE_REFS`` metadata recovery path (a
   ``feature_view_rows[].source_refs`` payload carrying the
   operator-authored source name + columns + query body) and replanning
   produces ``NO_CHANGE`` for both FVs.  Replaces the pre-Phase-B
   ``_classify_dt_body`` + synthetic ``<FV>__SOURCE`` shim — the
   ``FV_SOURCE_REFS`` metadata row is now authoritative and the
   recovery side passes the authored ``BatchSource.name`` straight
   through.
3. **Sidecar mutation** — overwriting ``EVENTS_FILE.sql`` with a
   semantically different body and replanning produces a
   ``RECREATE_FV`` / ``UPDATE_FV`` op for ``FV_FILE`` while
   ``FV_INLINE`` stays ``NO_CHANGE``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.compiler import normalize_sql_whitespace
from snowflake.ml.feature_store.decl.manifest import FSTarget
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.feature_store.decl.types import AppliedState, PlanOptions

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "batch_fv_query_source"


def _target() -> FSTarget:
    return FSTarget(
        name="DEV",
        account_identifier="ORG-ACCOUNT",
        database="BFV_DB",
        schema="BFV_SC",
        role="TESTROLE",
    )


def _load_and_validate(project_root: Path) -> Any:
    target = _target()
    batch = decl_api.load_project(project_root, target=target)
    decl_api.resolve_datasource_columns(batch)
    results = decl_api.validate_specs(
        batch,
        AppliedState(objects={}),
        target_database=target.database,
        target_schema=target.schema,
    )
    errors = [r for r in results if r.severity == "ERROR"]
    assert not errors, [r.message for r in errors]
    return batch, target


def _generate_plan(batch: Any, applied: AppliedState, target: FSTarget) -> Any:
    return decl_api.generate_plan(
        batch,
        applied,
        PlanOptions(),
        database=target.database,
        schema=target.schema,
    )


def _fv_spec_payload_from_plan(plan: Any, fv_name: str) -> dict[str, Any]:
    # Pull a single FV's CREATE-time payload out of the plan ops list. The
    # CREATE_FV payload mirrors the post-compile SPECIFICATION-shape payload
    # that the imperative executor will hand to register_feature_view, so it
    # is the right starting point for synthesising the deployed state.
    ops = [op for op in plan.ops if op.kind.value == "CREATE_FV" and op.name == fv_name]
    assert len(ops) == 1, f"expected exactly one CREATE_FV for {fv_name}; got {[o.kind.value for o in plan.ops]}"
    return copy.deepcopy(ops[0].payload)


def _build_applied_state(
    batch: Any,
    fv_names: list[str],
    target: FSTarget,
) -> AppliedState:
    # Build an AppliedState covering the supplied FVs + their entity tags.
    #
    # Compiles each local FV authoring dict via spec_compiler.compile_to_spec
    # (the same compile generate_plan uses for the full-spec hash) so the
    # recovered SPECIFICATION shape carries every derived field —
    # target_lag_sec, feature_granularity_sec, online_store_type, etc —
    # matching what DESCRIBE … TYPE = SPECIFICATION returns from a live OFT.
    #
    # Then threads a ``feature_view_rows`` entry whose ``source_refs``
    # cell carries the operator-authored source name + columns + query
    # body — the Phase-B3 contract.  ``state._inject_batch_fv_source_from_metadata``
    # reads ``source_refs`` from this row and injects it onto
    # ``spec.sources[]`` BEFORE hashing, so the resulting AppliedObject
    # hashes identically to the local-compile spec and the planner
    # resolves NO_CHANGE.
    #
    # Hands the recovered payloads to state.fetch_applied_state so the
    # Entity / FV AppliedObjects pick up the same structural fingerprinting
    # logic the planner reads against.
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    show_rows = []
    spec_map: dict[str, Any] = {}
    feature_view_rows: list[dict[str, Any]] = []
    seen_entity_keys: set[str] = set()
    entity_rows = []

    for fv_name in fv_names:
        fv_spec = next(s for s in batch.specs if getattr(s, "name", "") == fv_name)
        authoring_dict = fv_spec.model_dump(exclude_none=True)
        if "schema_" in authoring_dict and "schema" not in authoring_dict:
            authoring_dict["schema"] = authoring_dict.pop("schema_")
        compiled = compile_to_spec(authoring_dict, target.database, target.schema)
        version = compiled.get("metadata", {}).get("version") or authoring_dict.get("version") or "V1"
        oft_name = f"{fv_name.upper()}${version.upper()}$ONLINE"
        show_rows.append(
            {
                "name": oft_name,
                "database_name": target.database,
                "schema_name": target.schema,
                "scheduling_state": "ACTIVE",
            }
        )

        recovered = copy.deepcopy(compiled)
        spec_root = cast(
            "dict[str, Any]", recovered.get("spec") if isinstance(recovered.get("spec"), dict) else recovered
        )
        sources = spec_root.get("sources") or []
        assert sources, f"compile_to_spec produced no sources for {fv_name}: {compiled!r}"
        src0 = sources[0]
        query = src0.get("query")
        assert isinstance(query, str) and query, (
            f"FV {fv_name}: compile_to_spec must propagate sources[0].query for "
            "query-backed BatchSources; got {src0!r}"
        )
        spec_map[oft_name] = recovered

        # Phase B3 metadata-roundtrip contract: the operator-authored
        # source name + columns + query body ride on ``FV_SOURCE_REFS``.
        # The recovery side reads this list verbatim and writes it onto
        # ``spec.sources[]`` (overriding whatever was on the
        # SPECIFICATION JSON), so the AppliedObject hashes identically
        # to the local-compile spec.
        feature_view_rows.append(
            {
                "name": fv_name.upper(),
                "version": str(version).upper(),
                "database_name": target.database,
                "schema_name": target.schema,
                "kind": "BATCH",
                "entities": list(spec_root.get("ordered_entity_column_names", [])),
                "online_enabled": False,
                "target_lag": "",
                "refresh_freq": "",
                "warehouse": "",
                "cluster_by": "",
                "refresh_mode": "",
                "desc": "",
                "physical_dt_name": f"{fv_name.upper()}${str(version).upper()}",
                "source_refs": [
                    {
                        "name": src0.get("name") or "",
                        "source_type": src0.get("source_type") or "Batch",
                        "query": query,
                        "columns": src0.get("columns") or [],
                    }
                ],
            }
        )

        for col_name in spec_root.get("ordered_entity_column_names", []):
            tag_name = f"SNOWML_FEATURE_STORE_ENTITY_{str(col_name).upper()}"
            if tag_name in seen_entity_keys:
                continue
            seen_entity_keys.add(tag_name)
            entity_rows.append(
                {
                    "name": tag_name,
                    "database_name": target.database,
                    "schema_name": target.schema,
                    "allowed_values": f'["{str(col_name).upper()}"]',
                }
            )

    return fetch_applied_state(
        show_rows,
        None,
        specification_map=spec_map,
        entity_rows=entity_rows,
        feature_view_rows=feature_view_rows,
        default_database=target.database,
        default_schema=target.schema,
    )


def test_load_compile_resolve_validate_plan_emits_create_for_both_query_fvs() -> None:
    """Phase 8 case 1: an unseeded apply emits CREATE_FV for both FVs.

    Both source shapes (``query: |`` inline and ``query_file:`` sidecar)
    must reach the planner with the SQL inlined and whitespace-
    normalised; the resulting CREATE_FV payloads carry ``query`` (not
    ``table``) on ``sources[0]``.
    """
    batch, target = _load_and_validate(_FIXTURE_ROOT)

    plan = _generate_plan(batch, AppliedState(objects={}), target)

    create_fv_names = sorted(op.name for op in plan.ops if op.kind.value == "CREATE_FV")
    assert (
        "FV_INLINE" in create_fv_names and "FV_FILE" in create_fv_names
    ), f"both query-backed FVs must produce CREATE_FV ops; got: {create_fv_names!r}"

    for fv_name in ("FV_INLINE", "FV_FILE"):
        payload = _fv_spec_payload_from_plan(plan, fv_name)
        spec_root = cast("dict[str, Any]", payload.get("spec") if isinstance(payload.get("spec"), dict) else payload)
        sources = spec_root.get("sources") or []
        assert sources, f"{fv_name} CREATE_FV payload missing sources: {payload!r}"
        src0 = sources[0]
        assert src0.get("source_type") == "Batch"
        assert src0.get("query"), (
            f"{fv_name}.sources[0] must carry inlined query; got {src0!r}. "
            "compile_to_spec / inline_query_source must propagate the BatchSource.query "
            "into the CREATE_FV payload."
        )
        assert "query_file" not in src0, (
            f"{fv_name}.sources[0] must NOT carry query_file at planner stage — "
            "the compiler is supposed to inline the sidecar before plan generation."
        )
        # The compiler runs normalize_sql_whitespace; verify by checking the
        # query is single-spaced (no embedded \n / multi-space runs).
        query = src0["query"]
        assert query == normalize_sql_whitespace(query), (
            f"{fv_name}.sources[0].query must be whitespace-normalised before reaching " f"the planner; got {query!r}"
        )


def test_query_backed_fvs_round_trip_to_no_change_on_replan() -> None:
    """Phase 8 case 2: re-applying an unchanged tree emits NO_CHANGE for both FVs.

    Reconstructs the deployed AppliedState via the Phase-4 DT-text
    recovery path (sources[0] uses synthetic ``<FV>__SOURCE`` name plus
    the normalised query body) and verifies the planner reports zero
    drift — exercising the Phase-5 hash convergence rule end-to-end.
    """
    batch, target = _load_and_validate(_FIXTURE_ROOT)
    applied = _build_applied_state(batch, ["FV_INLINE", "FV_FILE"], target)
    replan = _generate_plan(batch, applied, target)

    fv_ops = [op for op in replan.ops if op.name in ("FV_INLINE", "FV_FILE")]
    non_no_change = [op for op in fv_ops if op.kind.value != "NO_CHANGE"]
    assert not non_no_change, (
        "round-trip from query-backed AppliedState must produce NO_CHANGE for both FVs; "
        f"got: {[(o.kind.value, o.name, o.reason) for o in non_no_change]!r}"
    )


def test_mutating_query_file_triggers_recreate_for_fv_file_only(tmp_path: Path) -> None:
    # Phase 8 case 3: editing EVENTS_FILE.sql produces a non-NO_CHANGE op
    # for FV_FILE. Copies the fixture into tmp_path so the in-tree fixture
    # stays pristine, then overwrites EVENTS_FILE.sql with a semantically
    # different query and re-plans against the original AppliedState.
    # Expected:
    #   * FV_FILE -> RECREATE_FV (or UPDATE_FV per current planner rules)
    #     because the query body changed.
    #   * FV_INLINE -> NO_CHANGE because the inline query was untouched.
    import shutil

    project = tmp_path / "batch_fv_query_source"
    shutil.copytree(_FIXTURE_ROOT, project)

    batch_orig, target = _load_and_validate(project)
    applied = _build_applied_state(batch_orig, ["FV_INLINE", "FV_FILE"], target)

    sql_path = project / "sources" / "datasources" / "EVENTS_FILE.sql"
    sql_path.write_text(
        "SELECT\n"
        "    USER_ID,\n"
        "    EVENT_TS,\n"
        "    METRIC_VAL\n"
        "FROM RAW_EVENTS\n"
        "WHERE EVENT_TS > '2025-06-01'\n"
    )

    batch_edited, _ = _load_and_validate(project)
    replan = _generate_plan(batch_edited, applied, target)

    fv_inline_ops = [op for op in replan.ops if op.name == "FV_INLINE"]
    fv_file_ops = [op for op in replan.ops if op.name == "FV_FILE"]
    assert len(fv_inline_ops) == 1
    assert len(fv_file_ops) == 1

    assert fv_inline_ops[0].kind.value == "NO_CHANGE", (
        f"FV_INLINE was untouched; expected NO_CHANGE, got "
        f"{fv_inline_ops[0].kind.value} (reason={fv_inline_ops[0].reason!r})"
    )
    assert fv_file_ops[0].kind.value in ("RECREATE_FV", "UPDATE_FV"), (
        f"FV_FILE's sidecar SQL changed semantically; expected RECREATE_FV / UPDATE_FV, got "
        f"{fv_file_ops[0].kind.value} (reason={fv_file_ops[0].reason!r})"
    )


@pytest.mark.parametrize("fv_name", ["FV_INLINE", "FV_FILE"])
def test_query_is_inlined_at_compile_time(fv_name: str) -> None:
    # Pin the contract that query_file never reaches the loader's spec dict.
    # Compile-time inlining (compiler.inline_query_source) is the contract by
    # which the rest of the pipeline (planner, exporter, invariants) only
    # ever sees query. This is a regression guard against a future refactor
    # that might delay the inlining to plan-execution time.
    batch, _target = _load_and_validate(_FIXTURE_ROOT)
    fv = next(s for s in batch.specs if getattr(s, "name", "") == fv_name)
    src0 = fv.sources[0]
    assert getattr(src0, "query_file", None) is None, (
        f"{fv_name}.sources[0].query_file must be None after load+compile; "
        "the compiler must inline query_file -> query before specs are returned to the loader. "
        f"got: {src0!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
