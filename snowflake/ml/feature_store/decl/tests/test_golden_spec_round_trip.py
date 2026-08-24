"""Golden-spec round-trip regression tests pinned to live ``DESCRIBE`` output.

These fixtures (under ``tests/golden_specs/``) are byte-for-byte captures
of the JSON returned by ``DESCRIBE ONLINE FEATURE TABLE <name> TYPE =
SPECIFICATION`` for representative FVs in the live JKEW_DB.JKEW_SCHEMA
environment.  Each test parametrises over every captured kind:

- BatchFeatureView (``USER_PROFILE_INFO_BATCH``)
- StreamingFeatureView with UDF (``USER_CLICK_STATS``)
- Continuous-flagged StreamingFeatureView (``USER_CLICK_STATS_M2_CONTINUOUS``)

If a future Snowflake release stamps a new field into the
SPECIFICATION JSON that the round-trip can't symmetrize, these tests
fail with a structured JSON diff so the regression is caught before it
hits operators.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.exporter import export_specs
from snowflake.ml.feature_store.decl.invariants import (
    _VOLATILE_METADATA_KEYS,
    _full_spec_hash,
)
from snowflake.ml.feature_store.decl.loader import load_specs
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.feature_store.decl.types import PlanOptions

GOLDEN_DIR = Path(__file__).parent / "golden_specs"


def _golden_files() -> list[Path]:
    return sorted(GOLDEN_DIR.glob("*.json"))


def _golden_id(path: Path) -> str:
    return path.stem


GOLDEN_PARAMS = [pytest.param(p, id=_golden_id(p)) for p in _golden_files()]


def _show_row_from_golden(spec_payload: dict[str, Any]) -> dict[str, Any]:
    md = spec_payload["metadata"]
    return {
        "name": f"{md['name']}${md['version']}$ONLINE",
        "database_name": md["database"],
        "schema_name": md["schema"],
        "scheduling_state": "RUNNING",
    }


def _entity_rows_from_golden(spec_payload: dict[str, Any]) -> list[dict[str, Any]]:
    md = spec_payload["metadata"]
    rows = []
    for col in spec_payload.get("spec", {}).get("ordered_entity_column_names", []) or []:
        rows.append(
            {
                "name": f"SNOWML_FEATURE_STORE_ENTITY_{col.upper()}",
                "database_name": md["database"],
                "schema_name": md["schema"],
                "allowed_values": f'["{col.upper()}"]',
            }
        )
    return rows


def _strip_volatile(spec: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy *spec* with ``_VOLATILE_METADATA_KEYS`` stripped from metadata.

    Args:
        spec: Spec payload to sanitize.

    Returns:
        dict: deep copy with volatile metadata keys removed.
    """
    cleaned = copy.deepcopy(spec)
    md = cleaned.get("metadata") if isinstance(cleaned, dict) else None
    if isinstance(md, dict):
        for key in _VOLATILE_METADATA_KEYS:
            md.pop(key, None)
    return cleaned


# ---------------------------------------------------------------------------
# Test 1 — Hash symmetry: live DESCRIBE → export → reload → compile, equal hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("golden_path", GOLDEN_PARAMS)
def test_golden_spec_hash_symmetry(golden_path: Any, tmp_path: Path) -> None:
    """Hash(compile_to_spec(load(export(applied)))) == hash(applied) for every golden.

    Args:
        golden_path: Path to a golden DESCRIBE JSON fixture.
        tmp_path: pytest tmp dir used for the export staging.
    """
    spec_payload = json.loads(golden_path.read_text())
    md = spec_payload["metadata"]
    db = md["database"]
    schema = md["schema"]

    show_rows = [_show_row_from_golden(spec_payload)]
    spec_map = {show_rows[0]["name"]: spec_payload}

    export_specs(
        show_rows=show_rows,
        describe_rows_by_oft={},
        output_dir=str(tmp_path),
        database=db,
        schema=schema,
        specification_map=spec_map,
    )

    export_root = tmp_path / f"{db}.{schema}"
    batch = load_specs([f"{export_root}/..."])

    fv_kind = spec_payload["kind"]
    fv_specs = [s for s in batch.specs if getattr(s, "kind", "") == fv_kind]
    assert fv_specs, (
        f"loader returned no spec of kind {fv_kind!r} from {export_root}; "
        f"got kinds={[getattr(s, 'kind', '?') for s in batch.specs]!r}"
    )
    loaded_dict = fv_specs[0].model_dump(exclude_none=True)
    if "schema_" in loaded_dict:
        loaded_dict["schema"] = loaded_dict.pop("schema_")

    compiled = compile_to_spec(loaded_dict, db, schema)

    if _full_spec_hash(compiled) != _full_spec_hash(spec_payload):
        expected = json.dumps(_strip_volatile(spec_payload), sort_keys=True, indent=2)
        actual = json.dumps(_strip_volatile(compiled), sort_keys=True, indent=2)
        pytest.fail(
            f"Golden spec {_golden_id(golden_path)!r}: "
            "compile_to_spec(loaded YAML) hash != applied hash.\n"
            "Live env will hit VERSION_CONFLICT on a clean round-trip.\n\n"
            f"--- expected (applied, volatile stripped) ---\n{expected}\n\n"
            f"--- actual (compiled, volatile stripped) ---\n{actual}\n"
        )


# ---------------------------------------------------------------------------
# Test 2 — End-to-end: export → load → validate+plan, every op NO_CHANGE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("golden_path", GOLDEN_PARAMS)
def test_golden_spec_export_then_plan_no_change(golden_path: Any, tmp_path: Path) -> None:
    """Export → load → plan emits zero non-NO_CHANGE ops for every golden.

    Args:
        golden_path: Path to a golden DESCRIBE JSON fixture.
        tmp_path: pytest tmp dir used for the export staging.
    """
    spec_payload = json.loads(golden_path.read_text())
    md = spec_payload["metadata"]
    db = md["database"]
    schema = md["schema"]

    show_rows = [_show_row_from_golden(spec_payload)]
    spec_map = {show_rows[0]["name"]: spec_payload}
    entity_rows = _entity_rows_from_golden(spec_payload)

    applied_state = fetch_applied_state(
        show_rows,
        None,
        specification_map=spec_map,
        entity_rows=entity_rows,
        default_database=db,
        default_schema=schema,
    )

    export_specs(
        show_rows=show_rows,
        describe_rows_by_oft={},
        output_dir=str(tmp_path),
        database=db,
        schema=schema,
        specification_map=spec_map,
    )

    export_root = tmp_path / f"{db}.{schema}"
    batch = load_specs([f"{export_root}/..."])

    decl_api.resolve_datasource_columns(batch)
    errors = [
        r
        for r in decl_api.validate_specs(
            batch,
            applied_state,
            target_database=db,
            target_schema=schema,
        )
        if r.severity == "ERROR"
    ]
    assert errors == [], (
        f"Golden {_golden_id(golden_path)!r}: round-trip plan reports " f"validation_failed; errors={errors!r}"
    )

    plan = decl_api.generate_plan(
        batch,
        applied_state,
        PlanOptions(),
        database=db,
        schema=schema,
    )
    bad_ops = [op for op in plan.ops if op.kind.value != "NO_CHANGE"]
    assert not bad_ops, f"Golden {_golden_id(golden_path)!r}: non-NO_CHANGE ops emitted: {bad_ops!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
