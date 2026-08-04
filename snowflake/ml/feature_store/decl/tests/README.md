# Tests: `snowflake.ml.feature_store.decl`

<!-- markdownlint-disable MD013 -->

## How to Run Tests

From the snowml repo root:

```bash
cd /path/to/snowml
python -m pytest snowflake/ml/feature_store/decl/tests/ -v
```

Focused batch FV pytest slice (no Snowflake):

```bash
bash /path/to/snowcli_fs/scripts/verify_batch_fv_bug_bash.sh --pytest-only
```

Live walkthrough of the operator doc `docs/BATCH_FV_BUG_BASH.md` at the snowcli_fs
workspace root: run `bash …/scripts/verify_batch_fv_bug_bash.sh` (no env required
unless you need overrides); the script reads **CURRENT_DATABASE** /
**CURRENT_SCHEMA** from the connection for `snow sql`. Use `--pytest-only` for
unit tests only. On failure it may inject HTML TODO markers into that doc only.

Run a specific test file:

```bash
python -m pytest snowflake/ml/feature_store/decl/tests/test_enums.py -v
```

Run with coverage:

```bash
python -m pytest snowflake/ml/feature_store/decl/tests/ -v --cov=snowflake.ml.feature_store.decl --cov-report=term-missing
```

## Test File Naming Convention

Each test file is named `test_<module>.py` where `<module>` is the module it covers:

| Test File | Module Covered | Description |
|-----------|---------------|-------------|
| `test_enums.py` | `enums.py` | Enum values, TYPE_ALIASES, normalize_type() |
| `test_enums_consolidation.py` | `enums.py` ↔ `spec.enums` | Identity-parity; OpKind is decl-local |
| `test_spec_models.py` | `spec_models.py` | Pydantic model construction, field defaults, validation |
| `test_types.py` | `types.py` | Plan pipeline types; `PlanFile`; serialize/deserialize round-trips |
| `test_errors.py` | `errors.py` | Exception classes and attributes |
| `test_loader.py` | `loader.py` | Multi-format loading (Python, YAML, JSON) *(Phase 1)*; `TestIsQueryCompanionSql` mirrors `_is_udf_companion_py` for `BatchSource.query_file` sidecars |
| `test_compiler.py` | `compiler.py` | Type normalization, duration parsing, UDF inlining *(Phase 1)*; `TestNormalizeSqlWhitespace` + `TestInlineQuerySource` cover the BatchSource `query` / `query_file` compile-time inlining and the idempotent whitespace canonicalisation |
| `test_templating.py` | `templating.py` | Jinja2 rendering, StrictUndefined, config sources *(Phase 1)* |
| `test_serializer.py` | `serializer.py` | to_dict, to_yaml, to_json round-trips *(Phase 1)* |
| `test_invariants.py` | `invariants.py` | All invariant rules from DevExAndSchemaAndCICD.md *(Phase 1)* |
| `test_dependencies.py` | `dependencies.py` | Topo sort, cycle detection, two-phase resolution *(Phase 1)* |
| `test_state.py` | `state.py` | Applied state parsing from raw SHOW/DESCRIBE results *(Phase 1)* |
| `test_planner.py` | `planner.py` | Plan generation: CREATE, UPDATE, RECREATE, NO_CHANGE *(Phase 1)* |
| `test_planner_batch_feature_view.py` | `planner.py` | Batch FV plan ops (`UPDATE_FV`, `RECREATE_FV`, drift) — also covers `_batch_fv_operational_drift` recovery on `compile_to_spec` failure and the `refresh_freq + target_lag` co-set case from BUG_BASH §7 |
| `test_batch_feature_view_validation.py` | `invariants.py` | Batch FV validation (no UDF, sources, tiles) |
| `test_imperative_executor_entity_resolution.py` | `imperative_executor.py` | FV entity ref: name vs join-key fallback |
| `test_imperative_executor_update_batch_fv.py` | `imperative_executor.py` | Batch UPDATE_FV → `update_feature_view`; `OnlineConfig.target_lag` propagation from `target_lag` and `target_lag_sec` (BUG_BASH §6 latent fix) |
| `test_batch_fv_project_integration.py` | cross-module | Load → validate → plan → serialize round-trip; pins fixture `target_lag` / `refresh_freq` shape from BUG_BASH §5 |
| `test_batch_fv_query_source_integration.py` | cross-module | End-to-end coverage for `BatchSource.query` / `query_file`: load → compile → resolve → validate → plan emits CREATE_FV with inlined whitespace-normalised `query`; AppliedState reconstructed via the Phase-4 DT-text recovery path round-trips to NO_CHANGE; mutating `EVENTS_FILE.sql` triggers RECREATE_FV / UPDATE_FV for FV_FILE while FV_INLINE stays NO_CHANGE |
| `test_export_plan_round_trip.py` | `exporter.py` (+ cross-module) | Re-export-then-replan invariant; `TestExtractQueryToSqlFile` pins the BatchSource sidecar contract (mirrors `_extract_udf_to_py_file`); `TestExportSpecsQueryBackedDatasource` + `TestExporterRoundTripThroughLoader` exercise the exporter end-to-end (query body → `<source_name>.sql` sidecar → loader → compiler inline → equal to original) |
| `test_queries.py` | `queries.py` | `state_queries` / `list_state_queries` SQL factories + new `dynamic_tables_query` (BUG_BASH §7/§8 DT-text recovery) |
| `test_state.py` | `state.py` | Applied-state parsing; `TestFetchAppliedStateWithDtTextMap` + `TestExtractSourceTableFromDtText` cover the DT-DDL → `BatchFV.sources[0].table` injection; `TestExtractDtBodyFromText` + `TestClassifyDtBody` + `TestInjectBatchFvSourceQueryShape` cover the `query:`-shape recovery path that emits a synthetic `<FV>__SOURCE` source name when the deployed body is anything other than a flat `SELECT * FROM <single qualified table>` |
| `test_invariants.py` | `invariants.py` | `TestBatchFvFullSpecHashParity` pins the BatchFV hash/structural-equivalent normalisation: `spec.sources` projected to sorted `[{binding}]`, 1:1 auto-derived features stripped, explicit aggregation surfaces preserved |
| `test_object_hashing.py` | `invariants.py` (+ `spec_compiler.py`) | Canonical hash table per object kind. `TestThreeShapeParity` pins authoring ↔ compiled ↔ applied hash equality for each of the 11 kind variants (Entity, two Source flavours, two StreamingFV variants, three BatchFV variants, RealtimeFV, FeatureGroup). `TestEditSensitivity` is the parametrised MUST-bump / MUST-NOT-bump matrix — every documented edit (UDF body, source-table swap, advanced BFV fields, FG slice/alias) is exercised once and the reason text surfaces in the failure message. `TestVolatileMetadataStripped` pins that bumping `client_version` / `oft_id` / `spec_format_version` on the applied side does not change the hash, so an operator upgrade cannot trigger a recreate storm. Live counterpart: `declarative_feature_store/tests/verify_roundtrip.sh` |
| `test_sql_generator.py` | `sql_generator.py` | SQL DDL string generation from Plan *(Phase 1)* |
| `test_integration.py` | cross-module | End-to-end pipeline: loader → compiler → invariants → planner *(Phase 1)* |

## What Each Test File Covers

### `test_enums.py`

- All `FSBaseType` values map to expected string names
- `FeatureViewKind` carries the four post-consolidation members
  (`StreamingFeatureView`, `RealtimeFeatureView`, `BatchFeatureView`,
  `FeatureGroup`)
- `SourceType` uses the canonical UPPERCASE member names (`STREAM`,
  `REQUEST`, `FEATURES`, `BATCH`) with `BATCH.value == "Batch"`; the
  decl-only `SQLDataSource` member has been removed and the legacy
  PascalCase `BatchSource` member name no longer exists
- `FeatureAggregationMethod` uses the canonical UPPERCASE member names
  (`TILES`, `CONTINUOUS`)
- `TYPE_ALIASES` maps human-friendly names to canonical FSBaseType values
- `normalize_type()` resolves aliases and passes through canonical names unchanged
- `normalize_type()` returns unknown types unchanged

### `test_spec_models.py`

- `FSColumn` construction with required and optional fields
- `FSColumn` serializes correctly via `model_dump()`
- `Entity` with `join_keys` list
- `StreamingSource` with `columns` list
- `BatchSource` with source location fields
- `SQLDataSource` is intentionally absent — removed during the
  decl/spec enum consolidation; `test_spec_models.py::TestSQLDataSourceRemoved`
  guards against accidental re-introduction
- `FeatureView` with all optional fields
- `FeatureGroup` with `feature_views` list
- All models serialize to plain dicts (no Snowpark types, no unserializable objects)

### `test_types.py`

- `AppliedObject` construction and serialization
- `AppliedState` with `objects` dict keyed by `kind:db.schema:name`
- `PlanOp` with all required fields
- `Plan` with `ops` list and `warnings`
- `ValidationResult` with severity literals
- `PlanOptions` defaults
- `SpecBatch` with `specs` and `source_files`
- `PlanFile` construction with defaults (`version="1"`, empty `plan`, empty `summary`)
- `serialize_plan()`: returns JSON with `version`, `target_database`, `target_schema`,
  `source_files`, ISO-8601 `created_at`, op-kind `summary` counts
- `deserialize_plan()`: round-trips database, schema, ops, warnings, source_files;
  raises `ValueError` on invalid JSON
- All types serialize cleanly to JSON via `model_dump()`

### `test_errors.py`

- `SpecLoadError` is an `Exception`
- `ValidationError` has `results: list[ValidationResult]`
- `DependencyError` is an `Exception`
- `StateDriftError` has `object_name`, `expected_version`, `found_version`

## How to Add Tests for New Invariants

1. Identify the invariant in `docs/DevExAndSchemaAndCICD.md`
2. Add a test case in `test_invariants.py` that:
   - Constructs a `SpecBatch` that triggers the invariant
   - Calls `validate_specs(batch, applied_state)`
   - Asserts the returned `ValidationResult` has the correct `severity` and `code`
3. Follow the pattern `test_<invariant_name>_blocking` for ERROR-level invariants
4. Follow the pattern `test_<invariant_name>_warning` for WARNING-level invariants

Example:

```python
def test_new_output_column_without_default_is_blocking():
    """New output column on deployed FV must have a default."""
    # Arrange: create an existing applied FV and a new spec with added column
    ...
    # Act
    results = validate_specs(batch, applied_state)
    # Assert
    errors = [r for r in results if r.severity == "ERROR"]
    assert any(r.code == "COLUMN_MISSING_DEFAULT" for r in errors)
```

## TDD Protocol

All test files for Phase 1+ follow the red-green-refactor cycle:

1. Write the test file (tests fail — red)
2. Commit: `test: add failing tests for decl/<module>`
3. Implement the module
4. Run tests to confirm they pass (green)
5. Commit: `feat: implement decl/<module>`
6. Refactor if needed, then commit: `refactor: clean up decl/<module>`
