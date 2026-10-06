# Design: `snowflake.ml.feature_store.decl`

## Package Purpose

`decl` is the **declarative authoring library** for the Snowflake Online Feature Store.
It provides the Python API that the `snowflake-cli` feature plugin calls to:

1. Load and parse spec files (YAML, JSON, Python) into structured Pydantic models
2. Validate specs against invariant rules and applied state
3. Generate a dependency-ordered execution plan (CREATE/UPDATE/RECREATE/NO_CHANGE/DROP operations)
4. Produce SQL DDL strings from the plan (NO_CHANGE ops are included in the plan for display but excluded from SQL)

The package is distributed as a **standalone lightweight wheel**
(`snowflake-ml-feature-store-decl`) with only three runtime dependencies:
`pydantic`, `pyyaml`, and `jinja2`.

## Isolation Rules

These rules must be obeyed by every module in this package:

1. **No imports from `snowflake.ml.*`** outside of `decl/` itself, with one
   explicit exception: `snowflake.ml.feature_store.spec.enums` is permitted
   (it is stdlib-only and is the canonical home for the FS enum vocabulary
   shared between the imperative and declarative paths). Heavy
   spec submodules (`snowflake.ml.feature_store.spec.models` and
   `snowflake.ml.feature_store.spec.builder`) remain forbidden because
   they import from `snowflake.snowpark.types`.
2. **No imports from `snowflake.snowpark.*`**
3. **No imports from `snowflake.connector.*`**
4. **Only stdlib + `pydantic`, `pyyaml`, `jinja2`** as runtime dependencies
5. **SQL generator returns strings only** — it does not execute them
6. **State fetcher accepts raw query results** — it does not hold connections

The narrowed exception in rule 1 is enforced by
`decl/tests/test_wheel_isolation.py::TestNarrowedSpecIsolation`: `spec.enums`
is permitted, while `spec.models` / `spec.builder` / `snowflake.snowpark` /
`snowflake.connector` continue to be forbidden. Violating any of rules 2–4
would make the wheel uninstallable in CLI environments that do not have the
snowpark SDK.

## Module Map

| Module | Responsibility |
|--------|---------------|
| `__init__.py` | Public API re-exports (`api`, `types`, `errors`, `enums`, `spec_models`) |
| `enums.py` | Re-exports `spec.enums` vocabulary + the decl-local `OpKind` (plan operation kinds) |
| `spec_models.py` | Pydantic v2 authoring models (`Entity`, `FeatureView`, `FeatureGroup`, sources, etc.) |
| `types.py` | Pipeline types: `SpecBatch`, `AppliedState`, `Plan`, `PlanOp`, `ValidationResult` |
| `errors.py` | Domain errors: `SpecLoadError`, `ValidationError`, `FeatureStoreNotInitializedError`, … |
| `api.py` | Top-level facade — query factories, state, planning, enrichment, export, plan-file (de)serialisation |
| `queries.py` | SQL string factories — OFT `SHOW` + per-OFT `DESCRIBE … TYPE = SPECIFICATION` |
| `imperative_executor.py` | **Only Snowflake I/O in decl/** — lazy bridge to `FeatureStore`; executes `PlanOp`s |
| `udf_loader.py` | UDF source-string → callable (satisfies `StreamConfig.__post_init__` inspection guard) |
| `loader.py` | Multi-format spec loading (`.py`, `.yaml`, `.json`); project directory walker |
| `compiler.py` / `spec_compiler.py` | Authoring format → `FROM SPECIFICATION` JSON compilation |
| `templating.py` | Jinja2 template rendering with StrictUndefined |
| `serializer.py` | `to_dict()`, `to_yaml()`, `to_json()` for all spec types |
| `invariants.py` | Validation rules; hashes (`_full_spec_hash`, `structural_fingerprint_hash`, `fg_content_hash`) |
| `dependencies.py` | `topological_sort` (create order); `order_specs_for_drop` (reverse-topo teardown order) |
| `planner.py` | Diffs `SpecBatch` vs `AppliedState` → `Plan` (ordered `PlanOp` list) |
| `state.py` | Builds `AppliedState` from raw `SHOW`/`DESCRIBE` rows and imperative FS rows |
| `exporter.py` | Reconstructs authoring YAML/Python from `AppliedState`; one `_export` driver + `_ExportRenderer` |
| `python_codegen.py` | Renders authoring-format spec dicts as loadable `.py` modules (`export_specs_as_python`) |
| `service.py` | Online-service CLI display helpers (status / describe rendering) |
| `tests/` | Unit and integration tests (co-located with package) |

### Module Notes

- **`api.py`** re-exports:
  - Query factories: `state_queries`, `list_state_queries`,
    `describe_specification_query`, `export_queries`. (The legacy
    `list_entities_query` was removed when the entity read path
    moved into `imperative_executor.fetch_entity_rows`.)
  - State: `fetch_applied_state`, `fetch_entity_rows`,
    `fetch_feature_view_rows`, `fetch_feature_group_rows`,
    `fetch_stream_source_rows`, `parse_specification_rows`,
    `enrich_list_results`. The
    `fetch_entity_rows(session, db, schema, warehouse="")` facade is
    a thin wrapper around `imperative_executor.fetch_entity_rows`,
    keeping the lazy-import of `snowflake.ml.feature_store` inside
    the executor module; `fetch_stream_source_rows(...)` is the
    parallel facade for `imperative_executor.fetch_stream_source_rows`,
    so the CLI manager can read registered `StreamingSource`
    metadata without importing the executor directly.
    `fetch_applied_state(...)` accepts a `stream_source_rows=...`
    keyword that propagates the runtime rows through to
    `state.fetch_applied_state`, where they reify into runtime-authoritative
    `Datasource` AppliedObjects via `state._build_stream_source_object`
    and beat the FV-derived path on key collision.
  - Planning: `resolve_datasource_columns`, `generate_plan`,
    `serialize_plan`, `deserialize_plan`.
  - Resolution: `resolve_oft_name(show_rows, name, version=None)` maps a
    user-supplied `(name, version)` onto a single deployed
    `<base>$<version>$ONLINE` name from `SHOW ONLINE FEATURE TABLES` rows,
    returning `(oft_name, error)`. It owns the OFT naming convention so
    `snow feature describe` can offer an optional `--version` without the CLI
    knowing the `$…$ONLINE` shape; a bare name that matches multiple versions
    yields an actionable "specify --version" error listing the versions.
  - Export: `export_specs`.
- **`queries.py`** — every multi-query bundle (`state_queries`,
  `list_state_queries`, `export_queries`) includes a
  `describe_specification_template` (a `{name}`-format string) so
  callers can issue per-OFT
  `DESCRIBE ONLINE FEATURE TABLE … TYPE = SPECIFICATION` queries
  without ever building SQL themselves.
- **`invariants.py`** — `structural_fingerprint_hash` is the
  pre-existing fallback. `_full_spec_hash` and
  `compute_local_spec_hash` implement full-spec diffing; both strip
  the volatile metadata keys (`client_version`,
  `spec_format_version`, `internal_data_version`) before hashing so
  they don't trigger spurious `RECREATE`s. `_check_idempotency`
  uses the **same hash strategy as the planner** so the validator
  and planner cannot disagree: when `applied.from_specification` is
  true and `kind` is in `_FV_RECREATE_KINDS`
  (`StreamingFeatureView`, `RealtimeFeatureView`,
  `BatchFeatureView`) it computes
  `compute_local_spec_hash(spec, target_database, target_schema,
  entity_join_keys=...)` (the same `build_entity_join_key_map` the
  planner threads, so a FeatureView whose entity **name** differs
  from its join-key **column** still hashes to `NO_CHANGE`)
  and compares against `applied.content_hash`; otherwise it falls
  back to `structural_fingerprint_hash`.  For `kind: FeatureGroup`,
  `_check_idempotency` instead routes to `fg_content_hash(spec)`
  — a SHA-256 over the FG's authored surface (`name`, `version`,
  `desc`, `auto_prefix`, and a sorted tuple of source
  `(fv_name, fv_version, slice_columns, alias)`).  The FG hash
  intentionally excludes the derived `output_columns` field so it is
  a pure function of the authoring YAML; the same hash is what the
  planner emits as `applied.content_hash` for FGs (built from the
  imperative row by `_build_feature_group_object`), so the
  no-change branch fires whenever the local YAML and the deployed FG
  agree on those five fields.   `spec_key` normalizes
  the two concrete source kinds (`StreamingSource`, `BatchSource`)
  to the generic `Datasource` via
  `_SOURCE_KIND_ALIASES` so a loaded YAML's `kind: StreamingSource`
  and the applied state's `kind: Datasource` resolve to the same
  lookup key — without this, every re-applied datasource looked like
  a new object on round-trip.
- **`planner.py`** — when `applied.from_specification` is true and
  the object is a FeatureView, the planner compares
  `compute_local_spec_hash(local_model)` against
  `applied.content_hash` (which is the full-spec hash from
  `state.py`). On a mismatch it emits `RECREATE_FV` with a reason
  that names the full-spec path; the structural fingerprint is the
  fallback when SPECIFICATION is unavailable. `Plan.warnings`
  records which path was used per FV.  For `FeatureGroup` specs,
  the planner uses `fg_content_hash` (see `invariants.py` notes
  above) — a hash match emits `NO_CHANGE`, a new key emits
  non-destructive `CREATE_FG`, a hash mismatch emits
  `CREATE_FG(destructive=True)` (gated by `--allow-recreate`), and
  an orphan FG in `full_directory_mode` emits `DROP_FG`.  There is
  no `UPDATE_FG` op — the imperative side has no
  `fs.update_feature_group` API and every FG edit materialises as
  "delete then register" (see "Why `FeatureGroup` skips the
  operational/structural split" below).
  After the diff and orphan passes, `generate_plan` runs two
  post-processing steps: (1) **Authored-spec FG/member gate** — a
  member `DROP_FV` / `RECREATE_FV` is refused when a still-authored
  batch FeatureGroup lists that `(name, version)` (`FG_MEMBER_STILL_REFERENCED`
  on `Plan.errors`; a hash-matched FeatureGroup is never promoted to
  `DROP_FG` / destructive `CREATE_FG`); (2) **Teardown banding** —
  remaining ops are stably reordered so FeatureGroup teardown
  (`DROP_FG`, destructive `CREATE_FG`) precedes member deletes, which
  precede source/entity drops. `Plan.errors` is a blocking, plan-time
  list distinct from `validate_specs` output; a non-empty list means
  the plan must not be written or applied.
- **`state.py`** populates three kinds of `AppliedObject`:
  - `FeatureView` — **`FeatureStore.list_feature_views()` (via the
    `feature_view_rows` kwarg) is the authoritative discovery
    source**: every registered FV enters applied state from its list
    row regardless of whether an OFT exists.
    `DESCRIBE … TYPE = SPECIFICATION` (via `specification_map`)
    remains authoritative only for the *spec payload* of OFT-backed
    online kinds — when the same FV surfaces via both the OFT path
    and `list_feature_views()`, the OFT-derived `spec_payload` (the
    server-stamped SPECIFICATION JSON) wins on key collision.  Every
    OFT-less FV — offline-only `BatchFeatureView`s (`online: false`)
    **and** FVs whose OFT was dropped but whose backing Dynamic Table
    is still listable — gets a reconstructed SPECIFICATION-equivalent
    `spec_payload` built by `_build_offline_fv_object` from the
    imperative row + DT text + entity metadata, with
    `from_specification=True` so the planner's full-spec hash
    branches resolve normally.  This list-driven discovery closes the
    two "invisible FV" bug classes (see
    `plans/done.bug_offline_bfv_invisible_after_apply.md` and
    `plans/done.bug_exporter_fv_without_oft_invisible.md`): a
    `snow feature plan` after a clean apply no longer re-emits a
    spurious `CREATE_FV` "New object: not found in applied state.",
    and `snow feature sync` no longer silently drops an OFT-less FV.
    `SHOW ONLINE FEATURE TABLES` is otherwise a **diagnostic-only**
    side channel — `orphaned_oft_warnings(...)` flags any OFT with no
    matching listed FV/FG (backing DT dropped) rather than treating
    it as a recovery path.      Cost: one `get_feature_view()` round-trip
    per OFT-less FV (N+1 for that subset; acceptable for CLI plan/init).
    `cluster_by` column identifiers recovered from `list_feature_views()`
    rows are normalised through `identifier.resolve_identifier()` — via
    `_resolve_cluster_column`, which now delegates to the shared
    `invariants._resolve_hash_identifier` so recovery and hashing use one
    identifier resolver — so a quoted identifier with an internal quote
    (`"FOO""BAR"`) resolves to its canonical form instead of the mangled
    `FOO""BAR` a naive `.strip('"')` would yield.
  - `Entity` — from rows produced by
    `imperative_executor.fetch_entity_rows`, which delegates strictly
    to `FeatureStore.list_entities()`. There is no raw-SQL fallback —
    uninitialised schemas raise `FeatureStoreNotInitializedError`,
    which the CLI surfaces as an actionable "run `snow feature init`"
    message. `list_entities()` is `SHOW TAGS … .select(…)`, so Snowpark
    runs the trailing `.select` as a **warehouse-bound**
    `SELECT … FROM TABLE(RESULT_SCAN(…))` — the `SHOW TAGS` alone would
    not touch a warehouse, but the `.select` does, so the `.collect()`
    can lose a race against warehouse auto-suspend ("Warehouse … was
    suspended while SQL was waiting to be scheduled"). `fetch_entity_rows`
    wraps the collect in `_collect_with_retry` (retry re-runs the
    collect, which resumes the warehouse; no `session.sql` is issued so
    the zero-entity-tag-SQL contract still holds), and on a persistent
    failure the exception propagates rather than degrading to an empty
    list — an empty applied-state entity set would spuriously produce
    `MISSING_ENTITY` for every referencing FV. The CLI seam
    (`manager._fetch_entity_rows`) mirrors this: it raises a `CliError`
    on a surviving failure instead of soft-failing to `[]` (see
    `plans/bug_cleanup_sweep_list_entities_warehouse_suspend_race.md`).
    The imperative row shape is normalised to the legacy
    `SHOW TAGS` row shape here, so `_build_entity_object` does not
    need to know about the imperative encoding. Names are uppercased and `content_hash`
    is computed via `structural_fingerprint_hash(spec_payload)`
    so the canonical key and hash align with the planner's
    `spec_key` normalization.
  - `Datasource` — populated from two complementary paths and merged
    runtime-authoritative on key collision.  `_build_stream_source_object`
    reifies registered `StreamingSource` rows produced by
    `imperative_executor.fetch_stream_source_rows` into
    `AppliedObject(kind="Datasource", spec_payload={"source_type": "Stream", ...})`
    entries; `_datasource_objects_from_specs` derives `Datasource`
    entries by unioning `spec.sources[]` across all recovered FV specs
    (deduplicated by name).  The merge step in `fetch_applied_state`
    inserts the runtime entries into `state.objects` first; the
    FV-derived pass then skips any key already present so the runtime
    row's authoritative `description` survives on collision.  Names
    are uppercased and hashed via `structural_fingerprint_hash` on
    both sides for the same reason as the entity path — without this
    normalization, every re-applied datasource produced a spurious
    `CREATE_*` op on round-trip even when nothing had changed.
    `BatchSource` entries continue to come exclusively from the
    FV-derived path because snowml-core has no `list_batch_sources`
    API; their applied state is recovered from a deployed BatchFV's
    Dynamic Table body via `_inject_batch_fv_source_from_dt_text`.

  `parse_specification_rows(rows)` lifts a single DESCRIBE result
  into a spec dict for callers. All Entity / Datasource state is
  authoritative — there is no fallback that synthesizes rows from
  PK columns or `source`-string parsing. `DESCRIBE … TYPE =
  SPECIFICATION` is enabled by default at the account level; no
  client-side session priming is issued.
- **`api.py.enrich_list_results`** produces the CLI's
  `snow feature list` rows — FeatureView rows from
  `oft_show_rows` (with subkind from `specification_map`), Entity
  rows from `entity_show_rows`, FeatureGroup rows from
  `feature_group_rows` (Phase 5b — one row per imperative
  `list_feature_groups()` entry; `details` carries `source_count`
  and an ordered ``"<fv_name>:<fv_version>"`` summary), and
  Datasource rows from `spec.sources[]` recovered via
  `specification_map`.  Display order is FV → Entity → FeatureGroup
  → Datasource so deployed objects group above the derived
  Datasource block.  Legacy two-arg calls
  (`enrich_list_results(show_rows, describe_map)` with no kwargs)
  keep producing FV-only output for backward compat with
  pre-existing CLI plugins.
- **`exporter.py`** —
  `export_specs(applied_state, *, specification_map, entity_rows)` emits
  full-fidelity YAML (UDF source, full sources block, features with
  windows / offsets / aggregations, granularity, target_lag,
  refresh_mode, etc.) by routing every FeatureView through
  `_build_full_fidelity_fv`. The function requires a non-empty entry
  in `specification_map` for every FV `AppliedObject` and raises a
  clear error naming the missing OFT(s) otherwise — every emitted
  YAML is full-fidelity or the call fails. Callers obtain the map
  by running `export_queries(...)` and
  `parse_specification_rows(...)` per OFT (the CLI does this via
  `_fetch_oft_state`).

  Entity emission is driven by `entity_rows` — the legacy SHOW TAGS
  shape returned by `fetch_entity_rows()`. Every tag (referenced
  *or* orphan) becomes a YAML, with `description` recovered from
  the tag's `comment` and `join_keys` from `allowed_values`. The
  invariant `FV.ordered_entity_column_names ⊆ (join-key columns of
  entity_rows)` is enforced leniently when `entity_rows` is non-empty
  (the shared helper `_orphaned_entity_column_warnings` decodes each
  tag's join-key **columns** via `_join_keys_from_row` /
  `allowed_values` — **never** the entity **name** via
  `_entity_name_from_row`, since an entity's name and its join-key
  column are frequently and validly different strings, e.g.
  `NOTEBOOK_SYNC_USER` / `USER_ID`): an OFT referencing an entity
  column with no matching registered join-key column is a stale
  artifact (typically an imperative-API registration predating the
  2026-07-31 join-key-immutability decision).  The FV is
  nonetheless a real deployed object whose definition is fully
  recoverable, so it is **warned but still exported** — the message
  surfaced in the returned `warnings` list — rather than skipped.
  Skipping its YAML would leave the planner seeing the deployed FV
  with no local spec and emitting a spurious `DROP_FV`; exporting it
  keeps an unmodified init → plan cycle at `NO_CHANGE` because
  `validate_specs` short-circuits on the content-hash idempotency
  check before `_check_dependencies`, so the unregistered column does
  not raise `MISSING_ENTITY`.  (A *modified* orphaned FV still fails to
  re-apply — it has no registered entity — hence the warning.)
  `entity_rows` values of `None` and `[]` are treated identically —
  no entity YAMLs are emitted and the subset check is skipped, so
  callers that opt out of entity emission (or hit a permission gap
  downstream of `fetch_entity_rows`) get a clean degraded export
  rather than a crash. Schemas with only entities (no FVs) still
  emit YAMLs as long as `entity_rows` is non-empty, closing the
  export → plan round-trip invariant for entity-only configurations.

  **Python-form overwrite guard.** `export_specs_as_python` derives each
  file name from the object name (`<FV_NAME>.py`), which collides with a
  UDF body named after the same FV (the `udf.file:` sidecar a sibling
  YAML references, e.g. `USER_CLICK_BACKFILL_DECL.py`). A blind write
  would replace the UDF function with the FV stub and break the next
  apply. Every Python-form `.py` write therefore routes through
  `_guarded_write_py`, which skips (and records a `warnings` entry) when
  the destination already exists and is **not** an exporter-generated
  spec module. The classifier `_py_is_spec_module` inspects the
  file's AST for a module-level `NAME = <Constructor>(...)` assignment
  where `<Constructor>` is one of `_SPEC_CONSTRUCTORS` (`Entity`,
  `BatchSource`, `StreamingSource`, `BatchFeatureView`,
  `StreamingFeatureView`, `RealtimeFeatureView`, `FeatureGroup`); a UDF
  body has only `def` blocks, so it is preserved. Genuine
  exporter-generated stubs (which carry that assignment) are still
  refreshed on re-export. The YAML-mode `export_specs` path is unaffected
  because it writes the recovered UDF body (correct content), not a stub.

  After the per-OFT FV loop and the entity emission, the exporter
  additionally emits one `datasources/<name>.yaml` per unique source
  name unioned across every FV `spec.sources[]` (column-superset,
  deduplicated by name in first-seen order). The emission is
  hybrid: FV YAMLs retain their inline `sources[]`, and the new
  files are the normalized cross-FV view.   Source-type →
  authoring-kind mapping (`Stream` → `StreamingSource`,
  `Batch` → `BatchSource` (canonical),
  `BatchSource` → `BatchSource` (legacy plan-file alias),
  `OfflineTable` → `BatchSource`) lives in
  `_SOURCE_TYPE_TO_KIND`; `Request` / `Features` are silently
  skipped (synthetic realtime sources). Unknown source-type values
  raise `ValueError`. Column-type conflict across FVs is a strict
  failure — the error names the datasource, column, both FVs, and
  both observed types.

## Authoring Format vs Internal Format

The `decl/` package uses an **authoring format** designed for human ergonomics:

- String-based type identifiers (`"StringType"`, `"str"`, `"int"`) instead of Snowpark
  `DataType` objects
- Human-friendly duration strings (`"5m"`, `"1h"`, `"7d"`) normalized to
  integer seconds during compilation. Duration string parsing is delegated to
  `snowflake.ml.feature_store.interval_utils.interval_to_seconds` (the canonical, stdlib-only
  parser shared with the imperative aggregation layer); `decl/compiler.py:parse_duration_to_seconds`
  is a thin shim around it that adds `None` / `int` / `float` passthrough. The `"lifetime"`
  sentinel maps to `-1` on both paths.
- Rich source/entity references (inline objects or name strings) resolved during compilation

Entity join keys are immutable after create. `spec_compiler.build_entity_join_key_map`
resolves the FV wire field `ordered_entity_column_names` from **applied** join keys for
already-deployed entities (batch keys only fill in new, not-yet-deployed entities), so an
unappliable YAML join-key edit does not flip a dependent FV to `RECREATE_FV`. The edit is
rejected up front by `invariants._check_entity_join_keys_immutable`
(`ENTITY_JOIN_KEY_IMMUTABLE`), which compares the **ordered** join-key names (so a reorder
is caught) and runs in `validate_specs` **before** the idempotency skip.

`validate_specs` builds the same `build_entity_join_key_map` and threads it into
`_check_idempotency` → `compute_local_spec_hash`, mirroring the planner. Without it a
FeatureView whose entity **name** differs from its join-key **column** hashes on the
authored name and never matches the column-based applied hash, so `NO_CHANGE` never
short-circuits for exactly those projects.

### Python authoring form

In addition to the YAML / JSON file forms, every spec kind can be authored
as a Python `.py` file using the class-based form from
`snowflake.ml.feature_store.decl`. The constructor surface IS the
`spec_models` Pydantic models — no new dataclass module — so the YAML and
Python paths share one Pydantic validation graph (Q1 of
`plans/python_form/python_authoring_form.md`).

Three FV subclasses (`StreamingFeatureView`, `BatchFeatureView`,
`RealtimeFeatureView`) pin `kind` to their class name as a Pydantic
class default so authors discriminate kinds by **class type** instead
of by string (Q2). `loader._dict_to_spec` maps the matching YAML
`kind:` strings to the same three subclasses (Q8), so a YAML / JSON
load and a Python load produce identical `isinstance` chains for every
downstream consumer.

Cross-spec references are equally first-class for **name strings**
(matching YAML) and **Python objects** (Q5):

- `serializer._collapse_sources` collapses `StreamingSource` /
  `BatchSource` instances in a `FeatureView.sources` list to
  `{"name": ..., "source_type": ...}` dicts at serialization time.
- `FeatureGroup._coerce_feature_view_objects` (a `model_validator(mode="before")`
  on `FeatureGroup`) collapses any `FeatureView` instance under
  `feature_views=[...]` to a `FeatureViewRef(name=..., version=...)`
  before Pydantic finishes validating the typed `list[FeatureViewRef]`
  schema.

Inline UDF callables (Q4) replace the YAML+sidecar pair: authors pass a
plain Python `def` as `UDF.function_definition`; the existing
`serializer.callable_to_source` (`inspect.getsource`) helper extracts
the function source at load time, so the on-wire compiled spec carries
the same `function_definition` string the YAML+sidecar path produces.

The loader applies a **spec-first with UDF fallback** rule to every
`.py` in a spec subdir (Q9):

1. Try to exec the file as a Python spec module.
2. If exec succeeds AND yields ≥1 module-level spec instance, those
   become specs in the batch.
3. If exec succeeds with zero spec instances, the file is treated as a
   pure UDF body — silently dropped from BOTH the spec list and
   `source_files`. (A sibling YAML's `udf.file:` mechanism still reads
   the same `.py` as text downstream.)
4. If exec fails AND `_is_udf_companion_py(filepath)` returns `True`,
   the failure is silently swallowed (the YAML+sidecar case where the
   `.py` body legitimately can't import at spec-load time).
5. Otherwise, raise `SpecLoadError` naming the file.

`_is_udf_companion_py` was broadened as part of the Python form: it
now scans **every** sibling YAML in the directory, not only the same-stem
peer, so UDF bodies may be named independently of their owning FV's
YAML and the loader still recognises the companion relationship via
the YAML's top-level `udf.file:` field.

The `spec/models.py` (existing internal package) uses an **internal serialization format**
designed for the Go backend:

- Snowpark `DataType` objects for column types
- Integer seconds for all durations
- Already-resolved source/entity references

`decl/compiler.py` transforms authoring format → internal JSON payload that
`CREATE ONLINE FEATURE TABLE ... FROM SPECIFICATION $$...$$` accepts. It produces the
same JSON structure as `spec/models.py` would, but without depending on snowpark.

### FV-level backfill (operational, not structural)

`FeatureView.backfill` is a kind-aware Pydantic block (`Backfill` model) that
maps directly onto the imperative library's two backfill surfaces:

| FV `kind`            | Backfill field      | Imperative target                                         |
|----------------------|---------------------|-----------------------------------------------------------|
| StreamingFeatureView | `backfill.table`    | `StreamConfig(backfill_df=session.table(<qualified table>))` |
| StreamingFeatureView | `backfill.start_time` | `StreamConfig(backfill_start_time=<datetime>)`          |
| BatchFeatureView     | `backfill.overwrite`| `FeatureStore.register_feature_view(overwrite=<bool>)`    |
| BatchFeatureView     | `backfill.initialize` | `FeatureView(initialize="ON_CREATE" \| "ON_SCHEDULE")` |

Cross-kind misuse (`overwrite` on a streaming FV, `table` on a batch FV)
is rejected by a `model_validator(mode="after")` on `FeatureView`. An empty
`backfill: {}` block is valid and is a no-op (streaming falls back to the
existing synthesized one-row sentinel DataFrame; batch falls back to the
imperative defaults).

**Target-schema qualification for `backfill.table`.**
`imperative_executor._build_streaming_backfill_df` qualifies an *unqualified*
`backfill.table` name with the FeatureStore's target `<database>.<schema>`
before calling `session.table(...)`. Without this the unqualified name resolves
in the Snowpark session's connection-profile default schema, which points the
backfill lookup at the wrong schema whenever the apply target schema differs
from the connection profile — either failing (`Object … does not exist`) or,
worse, silently reading a same-named table in an unintended schema. A
*fully-qualified* name (containing a `.`) is passed through verbatim so
cross-schema backfill tables remain valid. Only the `session.table()`
resolution is qualified — the authored value stamped onto
`StreamConfig.backfill_table` (and persisted as `StreamingMetadata.backfill_table`
for round-trip) stays exactly as authored, so a clean re-plan is still
`NO_CHANGE`.

**Operational semantics.** `backfill` is intentionally excluded from the
structural identity:

- `invariants._OPERATIONAL_FV_KEYS` strips it from `_full_spec_hash`.
- The planner emits `NO_CHANGE` for backfill-only edits on existing FVs,
  except that a batch FV with `backfill.overwrite=True` produces a
  destructive `CREATE_FV` op so that plain `snow feature apply` refuses
  the plan and `--allow-recreate` must be passed explicitly.
- The exporter does not recover `backfill:` from a deployed FV (it is a
  write-only authoring surface, mirroring the imperative ingest-time
  knobs); `_FV_TOP_LEVEL_ORDER` lists it for ordering when an authored
  spec round-trips through `model_dump`.

**Migration from `StreamingSource.backfill_table`.** The legacy
source-level field has been removed. A `model_validator(mode="before")`
on `StreamingSource` raises a `ValidationError` whose message points
operators at the new `FeatureView.backfill: { table: <FQN> }` block.

## Relationship to `spec/`

| Aspect | `spec/` (existing) | `decl/` (new) |
|--------|-------------------|---------------|
| Purpose | Internal serialization for Go backend | Human authoring format |
| Column types | Snowpark `DataType` objects | String-based `FSBaseType` enum |
| Durations | Integer seconds (`_sec` fields) | Human strings (`"5m"`, `"1h"`) |
| Part of wheel | `snowflake-ml-python` | `snowflake-ml-feature-store-decl` |
| CLI-installable | No (snowpark dependency) | Yes (pydantic + pyyaml + jinja2 only) |

## How to Add a New Spec Kind

1. Add a new value to the relevant enum in `enums.py` (e.g., a new `FeatureViewKind`)
2. Add a new `SpecBase` subclass in `spec_models.py` with the required fields.
   **Class-based discrimination (Q2 of the Python authoring form):** if
   the kind is a *variant* of an existing top-level kind (e.g.
   `StreamingFeatureView` / `BatchFeatureView` / `RealtimeFeatureView`
   all inherit `FeatureView`), make the subclass set `kind: str = "<NewKind>"`
   as a Pydantic class default and add a matching entry to
   `loader._dict_to_spec`'s `kind_map` so YAML / JSON dicts carrying
   `kind: "<NewKind>"` validate against the same subclass.  The
   serializer is unchanged — `model_dump()` produces the same dict for
   the subclass as for the base class with the matching `kind=` string.
3. Add corresponding validation rules in `invariants.py`
4. Add compilation logic in `compiler.py`
5. Add execution logic in `imperative_executor.py` — the executor is the only
   `decl/` module that talks to Snowflake, so any new op kind must be
   implemented as **imperative-API calls** there. Raw entity-tag DDL
   (`CREATE TAG` / `DROP TAG` / `ALTER TAG` / `SHOW TAGS`) is forbidden — the
   contract is locked in by `tests/test_no_entity_tag_sql_in_decl.py` (AST
   scan + behavioural assertions). (There is no `sql_generator.py`; SQL
   string generation as a planning artefact has been removed.)
6. Re-export the new class from `__init__.py`. If the kind is a
   top-level authoring target, it must be importable as
   `from snowflake.ml.feature_store.decl import <NewKind>` so Python
   authoring files (`.py` specs) can construct it directly.
7. Add `load_python_file`'s `known_types` tuple. Any new top-level kind
   that authors should be able to declare in a `.py` MUST appear in the
   tuple — the loader walks each module-level symbol and emits it as a
   spec via `isinstance(obj, known_types)`. Subclasses of an existing
   kind in the tuple are picked up automatically because `isinstance`
   follows MRO.
8. Add unit tests in `tests/test_spec_models.py` and `tests/test_invariants.py`
9. Update `docs/CHANGES.md`

### Worked Example — `FeatureGroup`

The `FeatureGroup` kind landed in v1 of FG support; treat the layout below as
the canonical "How to Add a New Spec Kind" template (it is more representative
than the per-field BFV walkthrough below — FG touched every layer in the
package because it is a new top-level kind, not just a new field on an
existing kind).  The numbered entries map onto the eight steps above:

1. **`enums.py`** — `FeatureGroup` reuses the existing top-level kind
   discriminator string `"FeatureGroup"`; no new enum value was required.
2. **`spec_models.py`** — added `FeatureGroup` (`name` / `version` / `desc` /
   `auto_prefix` / `feature_views`) and enriched the `FeatureViewRef`
   reference shape with `version` (required), `slice_columns` (optional), and
   `alias` (optional, with the literal `alias=""` semantically meaning "no
   prefix").  Validators reject empty `feature_views` lists, duplicate
   `(name, version)` pairs, empty `slice_columns`, and `$` in the FG name.
3. **`invariants.py`** — added `fg_content_hash(spec)`: a stable SHA-256 over
   the FG's name / version / desc / auto_prefix / sources tuple
   (`(fv_name, fv_version, slice_columns, alias)`).  `output_columns` is
   intentionally excluded so the hash is a pure function of the *authored*
   surface; `alias=""` is preserved distinct from `alias=None`.  Added
   `_check_feature_group_sources` to validate that referenced FVs exist (in
   either local batch or applied state), that `(name, version)` pairs are
   unique within the FG, and that source FVs are `online: true` with
   `store_type: POSTGRES` (with a soft-pass when `store_type` is not
   declared in the local spec).  `_check_idempotency` was extended to use
   `fg_content_hash` when `kind == "FeatureGroup"`.
4. **`loader.py`** — appended `"feature_groups"` to `_PROJECT_SUBDIRS` so
   `<root>/sources/feature_groups/` is discovered by `load_from_project`.
   No compiler logic — FG specs are passed through to the planner verbatim.
5. **`imperative_executor.py`** — added `fetch_feature_group_rows(...)` (read
   path; delegates to `FeatureStore.list_feature_groups()`) and
   `_build_feature_group(fs, payload, version)` (write path).  The build
   helper hydrates each source FV via `fs.get_feature_view(name, version)` —
   never `FeatureView(...)` — so the seven advanced BFV fields (and any future
   FV authoring fields) ride along automatically.  CREATE_FG and DROP_FG
   ops dispatch through `fs.register_feature_group` and
   `fs.delete_feature_group`; destructive CREATE_FG (the FG-side replacement
   for "update") issues a best-effort `delete_feature_group` followed by
   `register_feature_group`.  Honoured the existing `--allow-recreate` gate.
6. **`__init__.py`** — `FeatureGroup` and `FeatureViewRef` are re-exported
   alongside the other top-level spec models.
7. **Tests** — new files: `tests/test_spec_models_feature_group.py`,
   `tests/test_loader_feature_groups.py`,
   `tests/test_dependencies_feature_group.py`,
   `tests/test_invariants_feature_group.py`,
   `tests/test_planner_feature_group.py`,
   `tests/test_imperative_executor_fetch_fg.py`,
   `tests/test_state_feature_group.py`,
   `tests/test_imperative_executor_feature_group.py`,
   `tests/test_exporter_feature_group.py`,
   `tests/test_export_plan_round_trip_feature_group.py`,
   `tests/test_enrich_list_results_feature_group.py`.
8. **`docs/CHANGES.md`** — per-phase headings landed in Phase 8 of the FG plan.

### Why `FeatureGroup` skips the operational/structural split

The "How to Add a New BFV Operational or Structural Field" walkthrough below
introduces `_OPERATIONAL_FV_KEYS` and `_BATCH_FV_STRUCTURAL_INNER_KEYS` so a
batch-FV authoring edit can route to either `UPDATE_FV` (warehouse, etc.) or
`RECREATE_FV` (cluster_by, etc.) without spuriously rebuilding the deployed
Dynamic Table.  **`FeatureGroup` does not honour this split.**  The
imperative side has no `fs.update_feature_group` API today — every FG edit
materialises as "delete then register" — so any non-structural authoring
edit on the FG would silently degrade to a destructive recreate anyway.
The planner reflects this: a hash mismatch on `fg_content_hash` always
emits `CREATE_FG(destructive=True)` (gated by `--allow-recreate`), and there
is no FG analogue of `UPDATE_FV`.  Treat the `fg_content_hash` field set as
purely structural; do not introduce an `_OPERATIONAL_FG_KEYS` mirror.

### How to Add a New BFV Operational or Structural Field

When extending an existing `FeatureView` kind (most commonly `BatchFeatureView`)
with a new authoring knob that already exists on the imperative
`snowflake.ml.feature_store.FeatureView` constructor, every TDD pass touches
the same five points in a fixed order.  Worker agents from the per-field
phases (Phases 1–6 of the advanced BFV plan) follow this template verbatim:

1. **`spec_models.py`** — add the field to the `FeatureView` Pydantic model
   (or a nested model when the field is itself a struct, e.g. `storage_config`).
   Use `Optional[...]` so existing YAMLs that omit the field still validate.
   When the field has an enum value space, prefer `Literal[...]` so unknown
   strings raise a `ValidationError` at load time.
2. **`spec_compiler.py:compile_to_spec`** — decide whether the field belongs
   inside the compiled inner `spec` dict.  *Structural* fields (those that
   change the deployed Dynamic Table identity — `cluster_by`, `refresh_mode`,
   `initialize`, `storage_config`, `aggregation_secondary_keys`, `append_only`)
   MUST flow
   through so they contribute to `_full_spec_hash` and so a drift surfaces
   as `RECREATE_FV`.  *Operational* fields (`warehouse`) MUST NOT flow
   through the structural hash; instead, list them in
   `invariants._OPERATIONAL_FV_KEYS` so the hash strips them and the planner
   emits `UPDATE_FV` via `FeatureStore.update_feature_view`.  **CRON
   `refresh_freq`:** a CRON cadence (e.g. `"0 0 * * * UTC"`, required by
   `append_only` BFVs) drives a companion Task via `TARGET_LAG = 'DOWNSTREAM'`,
   so it never populates the wire `target_lag_sec` — gate the
   `BatchFeatureView` duration parse with `_is_interval_duration_refresh_freq(...)`
   (True only for values the narrower `interval_utils` grammar can consume) or
   `parse_duration_to_seconds` raises `Invalid interval format` and the planner
   loops on a spurious `RECREATE_FV`.  Do not gate it with `_is_cron_refresh_freq`:
   that helper now delegates to the core `pytimeparse` classifier
   (`feature_view_refresh_freq._is_cron_refresh_freq`) for append-only parity and
   accepts durations (`"2 weeks"` / `"1.5h"`) the narrower `interval_utils` parser
   cannot handle.
3. **`invariants.py`** — for structural fields, add the key to
   `_BATCH_FV_STRUCTURAL_INNER_KEYS` so
   `batch_feature_view_structural_equivalent` notices it (without this, a
   structural edit would mistakenly route to `UPDATE_FV` instead of `RECREATE_FV`).
   For operational fields, add the key to `_OPERATIONAL_FV_KEYS`.  Add any
   field-specific authoring validator (e.g. "secondary keys require tiled
   features") as a `_check_*` helper invoked from `validate_specs`.
4. **`imperative_executor.py:_build_feature_view`** — translate the payload
   value into the corresponding `FeatureView(**kwargs)` kwarg.  For nested
   shapes (e.g. `storage_config`), construct the snowml-core dataclass
   (`StorageConfig(format=StorageFormat(...), external_volume=..., ...)`)
   via a lazy import inside the function (the wheel-isolation rule forbids
   top-level imports of `feature_view.py`).  For operational fields, mirror
   the change in `_execute_update_feature_view` so the `UPDATE_FV` plan op
   forwards the kwarg to `fs.update_feature_view`.  **Kind-gate any Dynamic
   Table-only operational kwarg.**  `refresh_freq` and `warehouse` are DT
   refresh knobs, so `_execute_update_feature_view` forwards them only for a
   FV that materialises as a managed DT: `materializes_as_dt = is_batch or
   (is_tiled_streaming and payload carries refresh_freq)`.  A streaming FV
   without `refresh_freq` (non-tiled/continuous, or tiled-without-`refresh_freq`)
   is a zero-lag VIEW (`FeatureViewStatus.STATIC`); forwarding `warehouse` or
   `refresh_freq` to it makes `fs.update_feature_view` fail with error 2110.
   **`online_config` is likewise kind-gated: a `StreamingFeatureView`
   `UPDATE_FV` never forwards the kwarg.**  A streaming FV is always online
   by design, so the full authoring payload always carries `online: true` —
   the default authored value, not an in-place toggle (the OFT was created
   at `CREATE_FV` time).  Forwarding `online_config` would run
   `_create_online_feature_table`, whose streaming branch asserts
   `feature_view.stream_config is not None`; a streaming FV recovered from
   applied state as a zero-lag VIEW carries no `stream_config`, so the
   assertion fails and surfaces as the empty-message `(1300) Update feature
   view <NAME>/V1 failed:` (see `plans/done.bug_update_fv_error_1300.md`).
   The executor therefore omits `online_config` for streaming and still
   applies the other operational edits; an `online: true`-only payload is a
   no-op (empty kwargs, no imperative call).  It does **not** refuse the op
   just because `online` is present.  Genuinely changing a streaming FV's
   online routing requires a destructive `RECREATE_FV`. `BatchFeatureView`
   and `RealtimeFeatureView` still forward `online_config`.
5. **`exporter.py:_build_full_fidelity_fv`** — recover the field value from
   the deployed state so `snow feature init` produces a round-trippable YAML.
   Recovery sources (in preference order):
   - The compiled `spec` dict in `DESCRIBE … TYPE = SPECIFICATION` — works
     for fields that snowml-core stamps into SPECIFICATION JSON
     (`aggregation_secondary_keys`, possibly `feature_granularity_sec`).
   - The `SHOW ONLINE FEATURE TABLES` row (e.g. the JSON `storage_config`
     column added by snowml-core — see `feature_store.py:297`).
   - The offline Dynamic Table's DDL text (`CLUSTER BY (...)`,
     `REFRESH_MODE = '...'`, `INITIALIZE = '...'`, `WAREHOUSE = ...`,
     `EXTERNAL_VOLUME = ...`, `BASE_LOCATION = '...'`) — parsed by
     `state.py`-side helpers analogous to `_inject_batch_fv_source_from_dt_text`.

   `exporter._FV_TOP_LEVEL_ORDER` already reserves a stable position for
   every advanced field; never re-order it.

   **Operational-field round-trip (planner-side detection).** For an
   operational field the planner's `_split_operational_vs_structural` must see
   the *same* value on both sides after export, or a clean re-plan emits a
   spurious `UPDATE_FV`. Two closed gaps illustrate the two failure shapes
   (`plans/bug_operational_field_drift_streaming_bfvs.md`):
   - `description`: the applied side recovers the deployed `desc` at the top
     level of `spec_payload` via `state._inject_fv_desc_from_list_row` (invoked
     for BatchFV *and* StreamingFV), and `exporter._build_full_fidelity_fv`
     re-emits it as the authoring-shape `description`. Skipping either half made
     a described FV drift (`_desc_drifted`) on every re-plan.
   - `online_config` for always-online kinds: online routing is *derived* (from
     the account online store), not operator-authored, and
     `update_feature_view` cannot change it on `StreamingFeatureView` /
     `RealtimeFeatureView`. A streaming FV deployed with `online_enabled=False`
     recovers no applied `online_store_type`, but `compile_to_spec` always
     derives one, so `planner._online_config_drifted` now treats an
     absent/empty applied store type on an always-online kind as `NO_CHANGE`
     (a genuinely divergent store type is still drift).
   - `warehouse` for online-only BFVs (root cause B): the refresh `warehouse`
     is a Dynamic Table property, so `DESCRIBE … TYPE = SPECIFICATION` never
     carries it. Offline-only BFVs recover via
     `imperative_executor._serialize_batch_fv_spec`, which injects the list-row
     warehouse into `spec.warehouse`; but *online-only* BFVs recover through the
     OFT DESCRIBE path (`state.py`), whose
     `state._inject_batch_fv_fields_from_list_row` historically injected
     `desc`/`cluster_by`/`refresh_mode`/`append_only`/`initialize` but **not**
     `warehouse`. So an online-only BFV that authored a warehouse recovered an
     absent `spec.warehouse` and `planner._warehouse_drifted` reported a
     permanent authored-vs-absent drift → a spurious `UPDATE_FV` every re-plan
     (the 11 online-only ops in the AIMLDEV plan capture). The fix injects the
     `list_feature_views()` row `warehouse` in
     `_inject_batch_fv_fields_from_list_row` (additive — a no-op when the
     offline path already populated it), mirroring the `cluster_by`/`refresh_mode`
     injections. `_warehouse_drifted` stays case-insensitive and still fires on
     a genuine warehouse edit; a local-omitted warehouse is a no-op by contract
     (`update_feature_view` cannot clear a deployed warehouse). Pinned by
     `tests/test_online_warehouse_drift_idempotency.py` with live QA6 goldens
     (`golden_specs/ONLINE_WAREHOUSE_BFV.json` +
     `fixtures/ONLINE_WAREHOUSE_BFV_ROW.json`).
     Streaming half (`6e1`, customer triage): `warehouse` is *also* operational
     for a tiled `StreamingFeatureView`, but `state.fetch_applied_state` only
     ran the injection on the `BatchFeatureView` branch, so a streaming FV
     authoring `warehouse:` looped on the same spurious `UPDATE_FV`. The batch
     injection was extracted into a shared, kind-agnostic
     `state._inject_fv_warehouse_from_list_row` and the streaming recovery
     branch now calls it too. Realtime stays excluded (OFT-only, no DT;
     `warehouse` is not in its operational set). Pinned by the streaming cases
     added to `tests/test_online_warehouse_drift_idempotency.py`.

After all five touchpoints land, add a row per field to the per-field
`test_advanced_bvt_fields.py` matrix (remove the `xfail` marker) and to the
ADVANCED bug-bash walkthrough in `docs/ADVANCED_BVT_BUGBASH.md`.

### Secondary-key recovery pool (RECREATE_FV-loop fix)

A tiled BFV's `aggregation_secondary_keys` are synthesized into a
`_SECONDARY_KEY_ARRAY` aggregation spec whose `source_column` is the secondary
key, so the state-recovery resolution pool — rebuilt from the stamped
`FV_SOURCE_REFS` columns in `_serialize_batch_fv_spec` — MUST contain every
secondary key. The operator is not required to list the secondary key in the
authored `BatchSource.columns`; when they omit it, the stamped columns lack the
key, `_build_batch_feature_view_spec` raises `Column '<SK>' not found in
resolution pool`, `_serialize_batch_fv_spec` returns `None`, and the planner
loops on `RECREATE_FV` forever (task `fe8e6603`). Two `imperative_executor.py`
helpers close the loop with no destructive recreate and no core-snowml change:

- `_augment_schema_with_secondary_keys(raw_schema, secondary_keys, fallback_schemas)`
  — unions each missing secondary key into the rebuilt raw-source schema,
  typing it from the first fallback that carries it (the materialized DT schema,
  then the offline materialized schema — both physically contain the key).
  Wired into `_serialize_batch_fv_spec` just before it substitutes
  `fv._feature_df`, so recovery heals *already-deployed* FVs on the next plan.
- `_augment_source_ref_columns_with_secondary_keys(source_refs, secondary_keys, schema)`
  — unions the secondary keys into the stamped `FV_SOURCE_REFS` columns (typed
  from the resolved `feature_df` schema) in `_build_feature_view`, so *new*
  deployments persist a complete recovery pool.

`invariants._check_batch_feature_view_constraints` also emits a soft
`BATCH_FV_SECONDARY_KEY_ABSENT_FROM_SOURCE_COLUMNS` **WARNING** (never an ERROR
— existing YAML that omits the key must keep planning) nudging authors to list
the secondary key on the `BatchSource`. The `_serialize_batch_fv_spec` fallback
log no longer promises a "one-time RECREATE_FV will converge" (it can recur;
the recreate re-stamps the same columns). Pinned by
`tests/test_secondary_key_recovery_pool.py` and the
`test_advanced_bvt_secondary_keys.py` validator/stamping cases.

### Secondary-key hash canonicalization (`_strip_secondary_key_derivations`)

Separate from the recovery-pool fix above (which heals state recovery), a clean
apply-then-plan on a **secondary-key** FV still looped on `RECREATE_FV` because
core's `FROM SPECIFICATION` builders rewrite the deployed spec in three ways the
local decl compiler never mirrors: the SK is appended to
`ordered_entity_column_names`, every real aggregation `output_column` becomes
`ArrayType` with a server-derived `element_type` (a pure function of the
aggregation function + source type — e.g. `approx_count_distinct` →
`LongType`), and one synthesized `<SK>_KEYS_<window>S` array feature is added per
distinct window (sourced from the SK, no `function`).
`invariants._strip_secondary_key_derivations`, invoked from
`_normalize_for_full_spec_hash`, canonicalizes all three symmetrically so the
operator-authored local compile (SK-free entity list, scalar authored outputs,
no SK-array features) and the applied recovery converge on the same hash. The SK
identity and every semantic field (`function` / `window_sec` / `function_params`
/ `source_column` / `output_column.name`) survive, so a genuine SK / feature /
window / entity edit still recreates.

The helper is invoked for **both** `BatchFeatureView` and
`StreamingFeatureView` and is **SK-gated** (a no-op when the spec carries no
secondary key). This scoping preserves the streaming detection-safety contract:
a **non-SK** streaming `ArrayType` FV keeps its `element_type` hash-significant
(the inner type is operator-meaningful there, not server-derived). Both
back-end shapes are covered by live-captured goldens — a tiled online BatchFV
(`golden_specs/CONTENT_AGG_BATCH_SK.json`, client `1.52.0`, which additionally
recovers the tiled `cluster_by` default `[<entities>, TILE_START]`) and a
continuous online StreamingFV (`golden_specs/CONTENT_AGG_STREAM_SK.json`, client
`1.52.0`). Note the older `SK_ALL_AGGREGATIONS.json` streaming golden (client
`1.30.0`) *stripped* the SK from `ordered_entity_column_names`, whereas `1.52.0`
*retains* it; the symmetric entity-strip normalizes either shape to the same
hash, so both goldens are kept for cross-version regression coverage. For
streaming the planner compares with `_normalize_for_full_spec_hash` only — the
asymmetric `_normalize_applied_bfv_for_hash` (`refresh_mode` strip) is
`BatchFeatureView`-scoped. Pinned by
`tests/test_secondary_key_replan_idempotency.py` (batch + streaming
clean-roundtrip + over-strip guards),
`tests/test_streaming_fv_array_element_type_roundtrip.py`, and
`tests/test_streaming_fv_python_form_drift.py` (element_type detection on a
non-SK golden + SK semantic-change detection).

### SQL-identifier hash canonicalization (`_canonicalise_spec_identifiers`)

`compile_to_spec` copies authored identifiers into the wire spec verbatim, but
snowml-core resolves every identifier through `SqlIdentifier`: an **unquoted**
name folds to UPPER, a `"quoted"` name preserves case. So a project authoring a
lower-/mixed-case entity join key, `cluster_by`, `timestamp_col`, feature
column, or metadata identity applies cleanly (core uppercases on `CREATE`) and
then loops forever on destructive `RECREATE_FV` — byte-for-byte the customer
triage's `spec.cluster_by, spec.features, spec.ordered_entity_column_names,
spec.timestamp_field changed` reason strings (and, for lowercase metadata
identity, the generic full-spec fallback).

`invariants._normalize_for_full_spec_hash` calls `_canonicalise_spec_identifiers`
**first** (right after the JSON deep copy, before any downstream `.upper()`
comparison), routing every identifier-bearing field through
`_resolve_hash_identifier` — a thin wrapper over
`snowflake.ml._internal.utils.identifier.resolve_identifier`, the exact rule
core applies. Because normalization runs on **both** the local compile and the
applied recovery, authored casing and core-resolved casing collapse to one
canonical form. `compile_to_spec` output and the authored YAML are left
untouched, so uppercase golden hashes stay byte-stable. Fields canonicalized:
`metadata.{database,schema,name,version}` and, inside `spec`,
`ordered_entity_column_names`, `cluster_by`, `aggregation_secondary_keys`,
`ordered_secondary_key_column_names`, `timestamp_field`, each feature's
`source_column`/`output_column` name, `output_columns[].name`, and
`sources[].columns[].name`. A **quoted** identifier keeps its case, so a genuine
`"foo"` → `"FOO"` rename still `RECREATE_FV`s. `state._resolve_cluster_column`
delegates to the same `_resolve_hash_identifier`, so recovery and hashing share
one resolver. Pinned by `tests/test_identifier_case_canonicalization.py`
(per-field symmetry, full lowercase archetype end-to-end `NO_CHANGE`,
genuine-rename recreate guards, uppercase byte-stability) and the class-level
`tests/test_customer_drift_repro.py`.

### Exporter kind-filter + source-less BatchFV skip (init on legacy schemas)

Both FV write loops (`export_specs` YAML form, `export_specs_as_python` Python
form) filter each OFT by its recovered `kind` before writing anything under
`feature_views/`:

- **FeatureGroup OFTs are skipped.** A FeatureGroup is a real Online Feature
  Table, so `DESCRIBE … TYPE = SPECIFICATION` returns `kind == "FeatureGroup"`
  for it. Rendering it through the FV path emitted a broken
  `FeatureGroup(feature_views=[])` stub under `feature_views/` (the FeatureGroup
  renderer reads `spec["feature_views"]`, which the FV loop never populates),
  duplicating and shadowing the correct file emitted from the
  `feature_group_rows` → `feature_groups/` path and crashing `snow feature plan`
  with `FeatureGroup '<name>' must reference at least one FeatureView`. Any OFT
  whose `kind ∉ _FV_APPLIED_KINDS` is now `continue`d past. FeatureGroups reach
  the tree **only** via `feature_group_rows`; an FG OFT absent from that
  metadata is genuinely unrecoverable (its member refs are not in the
  SPECIFICATION) and is warned + skipped.
- **Source-less BatchFVs are skipped.** `_fv_doc_sources_recovered(fv_doc,
  full_spec)` returns `False` for a `BatchFeatureView` that carries neither a
  non-empty `fv_doc["sources"]` (from `FV_SOURCE_REFS` recovery or the
  SPECIFICATION) nor a `BatchSource` binding in `offline_configs`. Legacy BFVs
  registered by older imperative clients lack `FV_SOURCE_REFS` and expose
  `spec.sources: []`, so writing them would produce a non-round-trippable,
  source-less file; the exporter warns + skips instead. A passthrough BFV whose
  offline store *is* the source table (a `BatchSource` entry in
  `offline_configs`, e.g. the `USER_PROFILE_INFO_BATCH` golden) has a
  recoverable source and is still exported.

All skips surface an actionable message through the export envelope's
`warnings` list (lifted to the CLI `Warnings:` block). Backwards compatible:
streaming/realtime FVs, sourced BatchFVs, and passthrough BatchFVs are
unaffected. Pinned by `tests/test_exporter.py` classes
`TestExporterFeatureGroupOftLeak` and `TestExporterLegacyBatchFvMissingSources`.

**`name_filter` scopes datasource collection, not just its output.**
`_collect_datasources` takes a keyword-only `name_filter` and skips non-matching
sources at the top of the inner source loop, *before* validating `source_type`
or cross-FV conflicts — mirroring the FV/entity/FG write loops that filter on
identity first. Filtering the collected output instead let an unrelated FV's
malformed source (unrecognised `source_type`, cross-FV column-type conflict)
abort `sync --name GOOD_FV` after the matching file was already on disk. `None`
collects schema-wide with the strict conflict checks intact. Pinned by
`tests/test_exporter_sync_filter.py::TestExportSpecsNameFilterIsolatesInvalid`.

**`name_filter` scopes the orphan-OFT diagnostic too.**
`_oft_name_matches_filter(raw_name, name_filter)` is the shared OFT-identity
filter (parsed `<base>$<version>$ONLINE`, case-insensitive) used by the FV write
loop, the orphan-OFT diagnostic (`orphaned_oft_warnings`), and
`_orphaned_entity_column_warnings`, so a filtered export only *warns* about OFTs
it may actually emit — `sync --name GOOD_FV` no longer reports an unrelated
`GHOST_FV` as orphaned. Only the checked rows are filtered; the known-FV set fed
to `orphaned_oft_warnings` stays schema-wide, so a filtered healthy OFT is judged
against the full registry rather than a one-row slice. Pinned by
`tests/test_exporter_sync_filter.py::TestExportSpecsNameFilterScopesWarnings`.

**`append_only: true` is companion-gated on export.**
`_append_only_export_blocker` mirrors `spec_models._validate_append_only`
against the document about to be written (`BatchFeatureView`, `refresh_mode:
FULL`, a CRON `refresh_freq`, a `timestamp_col`, non-tiled). When unmet, the
flag is dropped with a warning rather than writing a spec `loader._dict_to_spec`
would reject and skip (a skipped file becomes a spurious `DROP_FV`). The
duration `refresh_freq` fallback (`"<n> seconds"` from `target_lag_sec`) is
likewise suppressed for append-only BFVs, whose CRON cadence stamps no
`target_lag_sec`.

**Kind directories are created on first write.** `_export` mkdirs
`entities/` / `feature_views/` / `feature_groups/` immediately before the
first write of that kind (not up front), so a no-match `name_filter` returns
the same empty envelope as an empty schema (`directory: ""`, no scaffolded
dirs). Pinned by `tests/test_exporter_sync_filter.py::TestExportSpecsNameFilterNoMatch`.

**A UDF without `function_definition` is unrecoverable in Python export.**
`_render_feature_view` raises rather than emitting `function_definition=,`
(a `SyntaxError` on load). YAML `_extract_udf_to_py_file` may skip a missing
body; the Python renderer must not. The same raise already covers a body with
no `def`. Pinned by
`tests/test_python_codegen.py::TestPythonCodegenCleanups.test_udf_without_function_definition_raises`.

**`aggregation_secondary_keys` is not tiled-only.** It is a structural BFV
knob (`RECREATE_FV`, max length 1) that is valid on **both** tiled and
non-tiled BFVs — on a tiled FV it adds a secondary group-by to each
aggregation; on a non-tiled (passthrough) FV it is still a spec/OFT identity
column (folded into `entity_columns` / `secondary_key_columns`, and the
POSTGRES OFT primary key). Do not re-add a tiled-only authoring gate: the
imperative side persists non-tiled SK and the exporter re-emits it, so a
tiled-only reject would break the `snow feature init` round-trip. Only the
length-1 cap (`BATCH_FV_SECONDARY_KEYS_MAX_LENGTH`) is enforced.

## GS Migration Boundaries

`api.py` is the only entry point. The top-level functions have stable signatures:

```python
load_specs(files, config) → SpecBatch
validate_specs(batch, applied_state) → list[ValidationResult]
generate_plan(batch, applied_state, options, *, database, schema) → Plan
fetch_applied_state(
    show_rows, table_rows,
    *, describe_map=None, specification_map=None, entity_rows=None,
    dt_text_map=None, feature_view_rows=None, feature_group_rows=None,
    stream_source_rows=None,
    default_database=None, default_schema=None,
) → AppliedState
enrich_list_results(*, oft_show_rows, entity_show_rows, specification_map, describe_map=None) → list[dict]
export_specs(
    show_rows, describe_rows_by_oft, output_dir, database, schema,
    *, specification_map=None, entity_rows=None,
) → dict[str, list[str]]
```

Query factories (also re-exported from `api.py`) keep all OFT SQL inside the library:

```python
state_queries(database, schema) → dict[str, str]            # apply-time bundle (OFTs only)
list_state_queries(database, schema) → dict[str, str]       # snow feature list bundle (OFTs only)
describe_specification_query(database, schema, name) → str
resolve_oft_name(show_rows, name, version=None) → (oft_name | None, error | None)  # describe resolution
parse_specification_rows(rows) → dict | None
fetch_entity_rows(session, database, schema, warehouse="", *, on_progress=None) → list[dict]         # entities
fetch_feature_view_rows(session, database, schema, warehouse="", *, on_progress=None) → list[dict]   # FVs incl. offline-only
fetch_feature_group_rows(session, database, schema, warehouse="", *, on_progress=None) → list[dict]  # FeatureGroups
fetch_stream_source_rows(session, database, schema, warehouse="", *, on_progress=None) → list[dict]  # registered StreamingSources
```

Each `fetch_*` facade accepts an optional keyword-only
`on_progress: Callable[[int, int, str], None]` used by the CLI to render a
human-facing progress bar. The library **never prints** — it only *invokes*
the callback the caller supplies (stdlib `Callable` only; no Rich import in
`decl/`, preserving the wheel-isolation rules). The contract is: once, right
after the listing `collect()` returns, `on_progress(0, n, "")` fires with
`n = len(listed)` — the moment `total` first exists; then `on_progress(i, n,
name)` fires after each translated row (1-based `i`; for BatchFVs the tick
fires *after* the per-FV `get_feature_view` serialization so it reflects the
real per-FV cost). An empty listing still emits the single `(0, 0, "")` call.
`on_progress=None` (the default) is a no-op, so every existing caller is
unchanged. See `snowflake-cli` `_plugins/feature/DESIGN.md` → "State-fetch
progress bar" for the CLI-side rendering and the `--silent`/structured-output
gating that keeps machine-readable stdout clean.

Entity rows are no longer fetched via a SQL constant: `decl_api.fetch_entity_rows`
delegates strictly into `imperative_executor.fetch_entity_rows` (lazy
`FeatureStore.list_entities()` import — no raw-SQL fallback), so the declarative
library does not duplicate the imperative encoding of the entity tag.
Uninitialised schemas surface `FeatureStoreNotInitializedError` instead of
returning an empty list.

`decl_api.fetch_feature_view_rows` is the parallel of `fetch_entity_rows` for
FeatureView enumeration: it lazy-imports `FeatureStore.list_feature_views()`
inside `imperative_executor.fetch_feature_view_rows`, applies the same
init-first guard, and translates each row into the narrow Phase-1 contract
(`name`, `version`, `database_name`, `schema_name`, `kind`, `entities`,
`online_enabled`, `target_lag`, `refresh_freq`, `warehouse`, `desc`,
`physical_dt_name`).  The new arrow surfaces offline-only `BatchFeatureView`s
(`online: false`) that `SHOW ONLINE FEATURE TABLES` cannot enumerate so
`fetch_applied_state` can build an `AppliedObject` for them.

Materialisation of `CREATE_ENTITY` / `DROP_ENTITY` /
`UPDATE_ENTITY` is similarly handled in `imperative_executor.py` via
`fs.register_entity` / `fs.delete_entity` / `fs.update_entity(name, desc=...)`
(desc-only — entity join keys are immutable after create per the 2026-07-31
design decision, so join-key edits are rejected at plan time by
`ENTITY_JOIN_KEY_IMMUTABLE` rather than routed through `update_entity`)
— no raw entity-tag DDL ever. `imperative_executor.py` is now the only SQL
emitter in the entire `decl/` subtree (the old `sql_generator.py` no-op shim has
been removed along with the dry-run apply path).

### Init-first invariant

All `snow feature` entry points except `init` call
`decl_api.assert_feature_store_initialized(session, db, schema, warehouse)`
before any read or write SQL. The helper constructs
`FeatureStore(creation_mode=FAIL_IF_NOT_EXIST)` and rewraps the snowml-core
`NOT_FOUND` ("Feature store internal tag … does not exist") as
`decl_api.FeatureStoreNotInitializedError(database, schema, wrapped)`. The CLI
catches the wrapper at the command boundary and surfaces it as a top-level
`ClickException`. There is no read-only fallback: `fetch_entity_rows` raises
rather than returning an empty list. The invariant is pinned by
`tests/test_init_first_hypotheses.py` (H1-H5, H7, H8),
`tests/test_imperative_executor.py::test_execute_plan_raises_feature_store_not_initialized_when_tags_missing`
(NT4), the matching read-path test (NT5),
`snowflake-cli/tests/feature/test_uninitialized_schema_errors.py` (NT6), and
the live `scripts/verify_uninitialized_schema.sh` harness.

When GS gains capabilities, individual functions can be replaced with GS-backed
implementations (REST calls instead of local logic) without changing calling code in the
CLI plugin. The prototype's `DESCRIBE … TYPE = SPECIFICATION` call is the proxy for the
future GS config fetcher; the spec JSON it returns already matches the `GetFVConfig`
contract (see `docs/FUTURE_ARCHITECTURE.md`).

## Reference Documents

- `docs/ARCHITECTURE.md` — System-level architecture and data flow
- `docs/DevExAndSchemaAndCICD.md` — Invariant rules specification
- `docs/ofs_dataclass_schema.py` — Original dataclass prototype this package replaces
- `plans/HIGH_LEVEL_EXECUTION_PLAN.md` — Execution plan and module-level design decisions
