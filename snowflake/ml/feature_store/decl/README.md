# Declarative feature store (`snowflake.ml.feature_store.decl`)

Authoring-time library used by `snow feature plan` / `apply`: load YAML from a
manifest project, validate invariants, compile to SPECIFICATION-shaped JSON, diff
against deployed state, and execute plans through the imperative `FeatureStore`
API.

- **Tests:** [tests/README.md](tests/README.md)
- **Batch FVs (no UDF):** see the snowcli_fs workspace doc
  `docs/BATCH_FV_BUG_BASH.md` (operator walkthrough and `UPDATE_FV` vs
  `RECREATE_FV` expectations).
- **FV-level `backfill:`:** see the snowcli_fs workspace docs
  `docs/CHANGES.md` and `docs/LIMITATIONS.md` for the contract; the
  `Backfill` Pydantic model lives in `spec_models.py` and the imperative
  mapping (streaming `StreamConfig.backfill_df` / `backfill_start_time`,
  batch `register_feature_view(overwrite=...)` / `FeatureView(initialize=...)`)
  lives in `imperative_executor.py`. The legacy
  `StreamingSource.backfill_table` field has been removed; loaders now
  raise a migration error pointing at `FeatureView.backfill.table`.
