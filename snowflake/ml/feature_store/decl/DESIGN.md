# Design: `snowflake.ml.feature_store.decl`

## Package Purpose

`decl` is the **declarative authoring library** for the Snowflake Online Feature Store.
It provides the API the `snowflake-cli` feature plugin calls to:

1. Load spec files (YAML, JSON, Python) into Pydantic models
2. Validate specs against invariant rules and applied state
3. Generate a dependency-ordered execution plan
4. Execute the plan via lazy-imported `FeatureStore` imperative calls

Distributed as a **standalone lightweight wheel** (`snowflake-ml-feature-store-decl`) with
three runtime dependencies: `pydantic`, `pyyaml`, `jinja2`.

## Isolation Rules

Every module must obey:

1. **No imports from `snowflake.ml.*`** outside `decl/` — except `spec.enums`
   (stdlib-only, shared enum vocabulary). `spec.models` and `spec.builder` remain
   forbidden (they pull in `snowflake.snowpark.types`).
2. **No `snowflake.snowpark.*` imports**
3. **No `snowflake.connector.*` imports**
4. **Only stdlib + `pydantic`, `pyyaml`, `jinja2`** at runtime
5. **SQL generators return strings** — they do not execute
6. **State fetchers accept raw query results** — they do not hold connections

Enforced by `tests/test_wheel_isolation.py::TestNarrowedSpecIsolation`.

## Module Map

| Module | Responsibility |
|--------|---------------|
| `__init__.py` | Public API re-exports |
| `api.py` | Top-level facade — the **only** entry point the CLI imports |
| `enums.py` | Re-exports `spec.enums` vocabulary + `OpKind` (plan operation kinds) |
| `spec_models.py` | Pydantic v2 models (`Entity`, `FeatureView`, `FeatureGroup`, sources, etc.) |
| `types.py` | Pipeline types: `SpecBatch`, `AppliedState`, `Plan`, `PlanOp`, `ValidationResult` |
| `errors.py` | Domain errors: `SpecLoadError`, `ValidationError`, `FeatureStoreNotInitializedError`, … |
| `queries.py` | SQL string factories — OFT `SHOW` + per-OFT `DESCRIBE … TYPE = SPECIFICATION` |
| `loader.py` | Multi-format spec loading (`.py`, `.yaml`, `.json`); project directory walker |
| `compiler.py` / `spec_compiler.py` | Authoring format → `FROM SPECIFICATION` JSON |
| `templating.py` | Jinja2 template rendering with StrictUndefined |
| `serializer.py` | `to_dict()`, `to_yaml()`, `to_json()` for all spec types |
| `invariants.py` | Validation; hashes (`_full_spec_hash`, `structural_fingerprint_hash`, `fg_content_hash`) |
| `planner.py` | Diffs `SpecBatch` vs `AppliedState` → `Plan` (ordered `PlanOp` list) |
| `state.py` | Builds `AppliedState` from raw `SHOW`/`DESCRIBE` rows and imperative FS rows |
| `exporter.py` | Reconstructs authoring YAML from `AppliedState` (`snow feature init`) |
| `dependencies.py` | Topological sort; cycle detection |
| `udf_loader.py` | UDF source-string → callable (satisfies `StreamConfig.__post_init__` inspection guard) |
| `imperative_executor.py` | **Only Snowflake I/O in decl/** — lazy bridge to `FeatureStore`; executes `PlanOp`s |
| `tests/` | Unit and integration tests (co-located) |

## Key Contracts

### `imperative_executor.py`

The only module allowed to side-effect Snowflake or import from `snowflake.ml.feature_store`.
All `FeatureStore` / `FeatureView` / `Entity` imports are lazy (inside functions).

- **Entity ops:** `fs.register_entity` / `fs.delete_entity` / `fs.update_entity(name, desc=...)` —
  no raw `CREATE TAG` / `ALTER TAG` DDL. Locked by `tests/test_no_entity_tag_sql_in_decl.py`.
- **FV ops:** routed through `_build_feature_view`, which resolves entity names via `fs.get_entity(name)`.
- **Source ops:** dispatch by `payload["kind"]`. `BatchSource` ops are virtual no-ops (no API).
- **`--allow-recreate` gate:** `execute_plan` short-circuits before any DDL when destructive ops
  are present but `PlanOptions.allow_recreate=False`. Returns `status="refused"` for the whole
  plan (atomic — no partial apply).

### `planner.py`

Uses `_full_spec_hash` when `applied.from_specification=True`, falling back to
`structural_fingerprint_hash`. Both strip volatile metadata keys before hashing to prevent
phantom `RECREATE` ops on a clean round-trip.

Source ops use a four-way decision: `NO_CHANGE` (virtual override for sources whose FVs
are unchanged), `CREATE_SOURCE`, `UPDATE_SOURCE` (desc-only, non-destructive), or
`RECREATE_SOURCE` (structural, `--allow-recreate` gated).

### `state.py`

All names are uppercased to match the planner's `spec_key` normalisation. `Datasource`
applied state merges two paths: runtime-registered `StreamingSource` rows (authoritative,
beat FV-derived on key collision) and FV-derived entries from `DESCRIBE` spec output.

### `api.py`

`serialize_plan` / `deserialize_plan` form the canonical `plan → apply` handoff.
The apply flow is exclusively a plan-file consumer — `generate_plan` and `validate_specs`
are never re-invoked at apply time. This eliminates the parity-bug class (plan UI vs.
plan file divergence).

### Init-first invariant

All `snow feature` commands except `init` call `assert_feature_store_initialized(...)` before
any read or write. Uninitialised schemas raise `FeatureStoreNotInitializedError`; there is no
silent fallback. Pinned by `tests/test_init_first_hypotheses.py` (H1–H5, H7, H8).

## Authoring vs Internal Format

| Aspect | `spec/` (imperative) | `decl/` (authoring) |
|--------|---------------------|---------------------|
| Purpose | Internal serialization for Go backend | Human authoring format |
| Column types | Snowpark `DataType` objects | String-based (`"str"`, `"int"`, `"StringType"`, …) |
| Durations | Integer seconds (`_sec` fields) | Human strings (`"5m"`, `"1h"`, `"7d"`) |
| Wheel | `snowflake-ml-python` | `snowflake-ml-feature-store-decl` |
| CLI-installable | No (snowpark dependency) | Yes (pydantic + pyyaml + jinja2) |

`decl/compiler.py` transforms authoring → `FROM SPECIFICATION` JSON without depending on
snowpark. Duration parsing delegates to `interval_utils.interval_to_seconds` (stdlib-only).

### Python Authoring Form

Specs can be authored as `.py` files using the Pydantic model classes directly
(`from snowflake.ml.feature_store.decl import BatchFeatureView, Entity, …`). The YAML
and Python paths share one Pydantic validation graph.

The loader applies **spec-first, UDF-fallback** to every `.py` in a spec dir:
exec the file; if it yields ≥1 spec instance, collect them; if it yields 0 instances,
treat as a UDF body and skip; if exec fails and the file is a known UDF companion, swallow
the error; otherwise raise `SpecLoadError`.

### FV Backfill (operational, not structural)

`backfill` is excluded from `_full_spec_hash` (`_OPERATIONAL_FV_KEYS`). The planner emits
`NO_CHANGE` for backfill-only edits, except `backfill.overwrite=True` on a batch FV, which
produces a destructive `CREATE_FV` (requires `--allow-recreate`). The exporter does not
recover `backfill:` — it is write-only.

| FV `kind` | Backfill field | Imperative target |
|-----------|---------------|------------------|
| `StreamingFeatureView` | `backfill.table` | `StreamConfig(backfill_df=session.table(<table>))` |
| `StreamingFeatureView` | `backfill.start_time` | `StreamConfig(backfill_start_time=<datetime>)` |
| `BatchFeatureView` | `backfill.overwrite` | `register_feature_view(overwrite=<bool>)` |
| `BatchFeatureView` | `backfill.initialize` | `FeatureView(initialize="ON_CREATE"\|"ON_SCHEDULE")` |

## How to Add a New Spec Kind

1. Add enum value to `enums.py` (if needed)
2. Add Pydantic subclass to `spec_models.py`; use `Optional[...]` for new fields
3. Add validation in `invariants.py`; wire into `_check_idempotency`
4. Add compilation in `compiler.py`
5. Add execution in `imperative_executor.py` — **imperative API calls only, no raw DDL**
6. Re-export from `__init__.py`
7. Add the class to `loader._dict_to_spec`'s `kind_map` and `load_python_file`'s `known_types`
8. Add tests and update `docs/CHANGES.md`

**`FeatureGroup` note:** No `UPDATE_FG` op exists — every FG edit is destructive recreate
(`CREATE_FG(destructive=True)`, `--allow-recreate` gated). Do not introduce `_OPERATIONAL_FG_KEYS`.

### How to Add a New BFV Field

Same five touchpoints every time:

1. **`spec_models.py`** — `Optional[...]` field; use `Literal[...]` for enum-valued fields
2. **`spec_compiler.py`** — structural fields flow into the compiled `spec` dict
   (contribute to hash); operational fields (e.g. `warehouse`) do not
3. **`invariants.py`** — structural → `_BATCH_FV_STRUCTURAL_INNER_KEYS`; operational → `_OPERATIONAL_FV_KEYS`
4. **`imperative_executor.py`** — translate payload value to `FeatureView(**kwargs)`;
   mirror operational changes in `_execute_update_feature_view`
5. **`exporter.py`** — recover from SPECIFICATION JSON, `SHOW` row, or DT DDL text (in preference order)

Add a row to `tests/test_advanced_bvt_fields.py` and remove the `xfail` marker.

## Public API (`api.py`)

```python
# Planning
load_specs(files, config) → SpecBatch
validate_specs(batch, applied_state) → list[ValidationResult]
generate_plan(batch, applied_state, options, *, database, schema) → Plan
serialize_plan(plan) → str
deserialize_plan(json_str) → Plan

# State
fetch_applied_state(
    show_rows, table_rows,
    *, describe_map=None, specification_map=None, entity_rows=None,
    dt_text_map=None, feature_view_rows=None, feature_group_rows=None,
    stream_source_rows=None, default_database=None, default_schema=None,
) → AppliedState
enrich_list_results(*, oft_show_rows, entity_show_rows, specification_map, describe_map=None) → list[dict]
export_specs(
    show_rows, describe_rows_by_oft, output_dir, database, schema,
    *, specification_map=None, entity_rows=None,
) → dict[str, list[str]]

# SQL factories (no SDK needed)
state_queries(database, schema) → dict[str, str]
list_state_queries(database, schema) → dict[str, str]
describe_specification_query(database, schema, name) → str
parse_specification_rows(rows) → dict | None

# Imperative reads (require session)
fetch_entity_rows(session, database, schema, warehouse="") → list[dict]
fetch_feature_view_rows(session, database, schema, warehouse="") → list[dict]
fetch_feature_group_rows(session, database, schema, warehouse="") → list[dict]
fetch_stream_source_rows(session, database, schema, warehouse="") → list[dict]
```

All OFT SQL lives inside the library. Entity and FeatureView reads delegate to
`FeatureStore.list_entities()` / `list_feature_views()` with no raw-SQL fallback.

## Reference Documents

- `docs/ARCHITECTURE.md` — System-level architecture and data flow
- `docs/CHANGES.md` — Change log
- `plans/HIGH_LEVEL_EXECUTION_PLAN.md` — Module-level design decisions
