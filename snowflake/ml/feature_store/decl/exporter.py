"""Export spec reconstruction for the declarative feature store library.

Reconstructs YAML spec files from pre-fetched SHOW results plus the full
original spec JSON returned by ``DESCRIBE ONLINE FEATURE TABLE <name>
TYPE = SPECIFICATION``.  No Snowflake connection access — accepts data
that the caller has already fetched and converted to plain dicts.

Export is strict: every OFT row in *show_rows* must have a corresponding
non-empty entry in *specification_map*.  Missing entries abort the export
with a :class:`ValueError`.

Entity emission is driven by *entity_rows* — the legacy ``SHOW TAGS``-shape
rows returned by :func:`decl_api.fetch_entity_rows` (which delegates to
``FeatureStore.list_entities()``).  This is the only way orphan entities
(registered tags not referenced by any FV) make it into the exported
YAML.  Passing ``entity_rows=None`` or ``entity_rows=[]`` means "no
entities registered" — no entity YAMLs are written and the strict
``FV.entities`` subset check is skipped (callers that want
authoritative entity emission must forward the output of
:func:`decl_api.fetch_entity_rows`).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from snowflake.ml.feature_store.decl.state import _ENTITY_TAG_PREFIX, _parse_oft_name

logger = logging.getLogger(__name__)


# YAML keys that should appear at the top of a feature view document, in this
# order, when present.  Any keys not listed here are appended after these.
#
# Order policy: identity → routing → schema → semantics → advanced runtime
# knobs → operational metadata.  The "advanced runtime knobs" cluster
# (``warehouse`` / ``cluster_by`` / ``refresh_mode`` / ``initialize`` /
# ``storage_config`` / ``aggregation_secondary_keys``) is a stable
# placement contract; each per-field TDD phase fills the corresponding
# slot in ``_build_full_fidelity_fv`` without re-ordering this tuple.
_FV_TOP_LEVEL_ORDER: tuple[str, ...] = (
    "kind",
    "name",
    "version",
    "database",
    "schema",
    "online",
    # Authoring-side keys (match the imperative FeatureView constructor).
    # The exporter translates the wire-form ``ordered_entity_column_names``
    # / ``timestamp_field`` keys recovered from DESCRIBE into these
    # authoring names in ``_build_full_fidelity_fv``.
    "entities",
    "timestamp_col",
    "feature_granularity_sec",
    "feature_aggregation_method",
    # ``refresh_freq`` is the authoring-form Dynamic Table refresh
    # cadence — it sits next to the tile-aggregation knobs because they
    # always co-author together for tiled feature views.  Same name on
    # both sides: the imperative ``FeatureView(refresh_freq=...)``
    # constructor kwarg and the declarative authoring key match.
    "refresh_freq",
    # ``target_lag`` is the authoring-form OFT staleness — online-only
    # by validator contract; rare in practice (most online BFVs accept
    # the imperative default ``"10 seconds"``) but exported when present.
    "target_lag",
    "sources",
    "features",
    "udf",
    # Advanced BFV authoring knobs (one slot per Phase 1–6 field, in plan
    # order).  Slots are reserved up front so a recovered FV YAML keeps
    # the same stable layout across phases — implementations only fill
    # values, never re-order the schema.
    "warehouse",
    "cluster_by",
    "refresh_mode",
    "initialize",
    "storage_config",
    "aggregation_secondary_keys",
    # ``backfill`` is operational metadata authored by the user — it never
    # appears in DESCRIBE … TYPE = SPECIFICATION output, so the exporter
    # cannot emit it from a recovered spec.  Listed here so any future code
    # path (or test) that hand-attaches a backfill block to the export dict
    # ends up with a stable, last-position placement under the FV doc.
    "backfill",
)


def _ordered_fv_doc(fv_spec: dict[str, Any]) -> dict[str, Any]:
    """Return *fv_spec* re-ordered with well-known keys first.

    Python ``dict`` preserves insertion order, and PyYAML respects it when
    serializing — so this gives the emitted YAML a predictable layout while
    leaving any unknown keys to trail at the end.

    Args:
        fv_spec: Feature view dict to re-order.

    Returns:
        New dict with the same keys, ordered with well-known keys first.
    """
    ordered: dict[str, Any] = {}
    for key in _FV_TOP_LEVEL_ORDER:
        if key in fv_spec:
            ordered[key] = fv_spec[key]
    for key, value in fv_spec.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _build_full_fidelity_fv(
    full_spec: dict[str, Any],
    *,
    fallback_name: str,
    fallback_version: str,
    fallback_database: str,
    fallback_schema: str,
) -> dict[str, Any]:
    """Build a YAML-ready feature view dict from a parsed spec JSON document.

    Args:
        full_spec: The spec JSON returned by ``DESCRIBE ... TYPE = SPECIFICATION``.
            Expected shape: ``{kind, metadata, offline_configs, spec, ...}``.
        fallback_name: Name to use if ``metadata.name`` is absent.
        fallback_version: Version to use if ``metadata.version`` is absent.
        fallback_database: Database to use if ``metadata.database`` is absent.
        fallback_schema: Schema to use if ``metadata.schema`` is absent.

    Returns:
        Dict ready to be passed through :func:`_ordered_fv_doc` and serialized.
    """
    kind = full_spec.get("kind", "StreamingFeatureView")
    metadata = full_spec.get("metadata", {}) if isinstance(full_spec.get("metadata"), dict) else {}
    inner = full_spec.get("spec", {}) if isinstance(full_spec.get("spec"), dict) else {}

    fv_doc: dict[str, Any] = {
        "kind": kind,
        "name": metadata.get("name", fallback_name),
        "version": metadata.get("version", fallback_version),
        "database": metadata.get("database", fallback_database),
        "schema": metadata.get("schema", fallback_schema),
    }

    # ``StreamingFeatureView`` and ``RealtimeFeatureView`` are always
    # online by design — the implicit-online contract on
    # ``spec_models.FeatureView`` defaults ``online=True`` and rejects
    # an explicit ``online=False`` for those kinds — so emitting the
    # field here would be redundant noise in the exported YAML.
    # ``BatchFeatureView`` keeps the field first-class because batch
    # FVs can legitimately be offline-only; infer ``online: true |
    # false`` from whether the recovered SPECIFICATION carries an
    # ``online_store_type`` (stripped from hashes but present on
    # DESCRIBE).
    if kind == "BatchFeatureView":
        fv_doc["online"] = bool(full_spec.get("online_store_type"))

    # Translate the wire-form keys recovered from DESCRIBE ... TYPE =
    # SPECIFICATION into the authoring keys the Pydantic models accept.
    # ``snow feature plan`` re-validates this YAML through the renamed
    # FeatureView model, so emitting the legacy keys would surface the
    # migration error in ``spec_models.FeatureView._reject_legacy_authoring_keys``
    # for every exported FV — defeating the round-trip.
    if "ordered_entity_column_names" in inner:
        fv_doc["entities"] = list(inner["ordered_entity_column_names"])

    if "timestamp_field" in inner and inner["timestamp_field"] is not None:
        fv_doc["timestamp_col"] = inner["timestamp_field"]
    if "feature_granularity_sec" in inner and inner["feature_granularity_sec"] is not None:
        fv_doc["feature_granularity_sec"] = inner["feature_granularity_sec"]
    if "feature_aggregation_method" in inner and inner["feature_aggregation_method"] is not None:
        fv_doc["feature_aggregation_method"] = inner["feature_aggregation_method"]
    # ``refresh_freq`` (authoring-form DT refresh) is the canonical
    # YAML-side surface for the offline Dynamic Table cadence — same
    # name as the imperative ``FeatureView(refresh_freq=...)``
    # constructor kwarg.  Prefer the explicit ``refresh_freq`` value on
    # the applied inner spec (``state._inject_fv_refresh_freq_from_list_row``
    # populates it from the deployed DT's ``REFRESH_FREQ``) and fall
    # back to deriving ``"<n> seconds"`` from the wire-form
    # ``target_lag_sec`` when only that key is present.  Streaming and
    # realtime kinds are excluded entirely: the runtime stamps
    # ``target_lag_sec: 0`` regardless of the authored value, and the
    # spec-validator ``FeatureView._reject_refresh_freq_on_stream_or_realtime``
    # now rejects authoring ``refresh_freq`` on those kinds entirely.
    # The wire-form ``target_lag_sec`` is no longer emitted at the top
    # level — ``refresh_freq`` is the single authoring surface for the
    # DT refresh cadence after the
    # ``feature_granularity`` / ``refresh_freq`` / ``target_lag``
    # decoupling.
    if kind == "BatchFeatureView":
        rf = inner.get("refresh_freq")
        if isinstance(rf, str) and rf:
            fv_doc["refresh_freq"] = rf
        elif "target_lag_sec" in inner and inner["target_lag_sec"] is not None:
            fv_doc["refresh_freq"] = f"{int(inner['target_lag_sec'])} seconds"

    sources = inner.get("sources")
    if isinstance(sources, list):
        fv_doc["sources"] = [dict(s) if isinstance(s, dict) else s for s in sources]

    features = inner.get("features")
    if isinstance(features, list):
        fv_doc["features"] = [dict(f) if isinstance(f, dict) else f for f in features]

    udf = inner.get("udf")
    if isinstance(udf, dict) and udf:
        # The live SPECIFICATION JSON already uses the authoring-shape
        # keys ``name`` / ``engine``; only ``function_definition`` needs to
        # become the authoring alias ``source`` for the YAML to round-trip
        # cleanly through :func:`spec_compiler._compile_udf`.  Note: the
        # source body is later extracted to a sibling ``.py`` file by
        # :func:`_extract_udf_to_py_file`, so the YAML actually carries
        # ``udf.file`` rather than an inline ``udf.source``.
        fv_doc["udf"] = dict(udf)

    # Advanced BFV authoring knobs recovered into ``inner.*`` by
    # ``state._inject_advanced_bfv_fields_from_dt_text``.  Each is copied
    # through verbatim into the YAML doc so ``snow feature init`` against
    # a deployed BFV produces a re-applicable spec.  Phase 2 wires
    # ``cluster_by``; later phases will add ``refresh_mode`` /
    # ``initialize`` / ``storage_config`` / ``aggregation_secondary_keys``
    # at this same block.
    cluster_by = inner.get("cluster_by")
    if isinstance(cluster_by, list) and cluster_by:
        fv_doc["cluster_by"] = list(cluster_by)

    refresh_mode = inner.get("refresh_mode")
    if isinstance(refresh_mode, str) and refresh_mode:
        fv_doc["refresh_mode"] = refresh_mode.upper()

    initialize = inner.get("initialize")
    if isinstance(initialize, str) and initialize:
        fv_doc["initialize"] = initialize.upper()

    storage_config = inner.get("storage_config")
    if isinstance(storage_config, dict) and storage_config:
        fv_doc["storage_config"] = {k: v for k, v in storage_config.items() if v is not None}

    secondary_keys = inner.get("aggregation_secondary_keys")
    if isinstance(secondary_keys, list) and secondary_keys:
        fv_doc["aggregation_secondary_keys"] = list(secondary_keys)

    # ``backfill:`` block re-emission (Plan B8 — closes LIMITATIONS L2
    # export half).  Phase A's metadata-roundtrip extensions persist the
    # operator's authored streaming-backfill table
    # (:attr:`StreamingMetadata.backfill_table` from A3) and carry the
    # existing ``backfill_start_time`` (already in ``STREAM_CONFIG``)
    # into the recovered ``spec_payload``.  Surface them back into the
    # YAML so ``snow feature init`` produces a re-applicable spec.
    #
    # ``initialize`` is intentionally NOT folded into the backfill block:
    # it has been promoted to a first-class top-level FV field
    # (:class:`FeatureView.initialize` per ``spec_models.py`` Phase 4)
    # and is already emitted above as ``fv_doc["initialize"]``.  Per the
    # plan note that field is "already in DT tag metadata" — listed
    # alongside the new backfill keys for completeness, not because it
    # needs to be re-folded.
    #
    # ``backfill.overwrite`` is intentionally NOT re-emitted.  This is
    # the authoring-only contract pinned by Plan B8 / LIMITATIONS L2:
    # round-tripping the operational one-shot flag would silently force
    # re-overwrites on every ``snow feature plan``.  A3 deliberately
    # does NOT persist it; the exporter's matching contract is to
    # never surface it even if some upstream payload happens to carry
    # the key.
    backfill_block: dict[str, Any] = {}
    backfill_table = inner.get("backfill_table")
    if isinstance(backfill_table, str) and backfill_table:
        backfill_block["table"] = backfill_table
    backfill_start_time = inner.get("backfill_start_time")
    if isinstance(backfill_start_time, str) and backfill_start_time:
        backfill_block["start_time"] = backfill_start_time
    if backfill_block:
        fv_doc["backfill"] = backfill_block

    return fv_doc


# ``file_stem`` comes from attacker-influenceable metadata; restrict it to a
# strict allowlist so a crafted name cannot escape the export dir (CWE-22).
_SAFE_FILE_STEM_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _safe_sidecar_path(base_dir: Path, file_stem: str, suffix: str) -> Path:
    """Build a sidecar path under *base_dir*, rejecting path traversal.

    Args:
        base_dir: Directory the sidecar must be written inside.
        file_stem: Bare filename stem derived from object metadata.
        suffix: File extension including the dot (e.g. ``".py"``).

    Returns:
        The resolved absolute path to the sidecar file.

    Raises:
        ValueError: If *file_stem* is not allowlist-clean or the resolved
            path escapes *base_dir*.
    """
    if not isinstance(file_stem, str) or not _SAFE_FILE_STEM_RE.fullmatch(file_stem):
        raise ValueError(f"unsafe sidecar file stem {file_stem!r}: only [A-Za-z0-9_-] permitted")
    resolved_base = base_dir.resolve()
    candidate = (resolved_base / f"{file_stem}{suffix}").resolve()
    if candidate.parent != resolved_base:
        raise ValueError(f"sidecar path {candidate} escapes export directory {resolved_base}")
    return candidate


def _extract_udf_to_py_file(
    fv_doc: dict[str, Any],
    fv_dir: Path,
    file_stem: str,
) -> Optional[str]:
    """Externalize ``fv_doc['udf']['function_definition']`` to a sibling ``.py`` file.

    Mutates *fv_doc* in place: when a non-empty ``function_definition`` string
    is present, it is removed from the ``udf`` block and replaced with a
    ``file`` key holding the bare filename (relative to *fv_dir*).  The source
    is written to ``<fv_dir>/<file_stem>.py``.  When ``function_definition``
    is missing, empty, or non-string, no file is written and the ``udf`` block
    is left untouched.

    The bare-filename convention matches what
    :func:`compiler.inline_udf_source` resolves (it joins ``udf.file`` with
    the spec file's directory) — so the exported YAML round-trips through
    ``snow feature apply`` without manual edits.

    Args:
        fv_doc: Feature view dict being assembled for YAML serialization.
        fv_dir: Directory in which the FV YAML will be written.
        file_stem: Base filename (typically the FV name) used for the ``.py``
            sidecar.

    Returns:
        Absolute path to the written ``.py`` file as a string, or ``None`` if
        no extraction occurred.
    """
    udf = fv_doc.get("udf")
    if not isinstance(udf, dict):
        return None
    source = udf.get("function_definition")
    if not isinstance(source, str) or not source:
        return None

    udf_path = _safe_sidecar_path(fv_dir, file_stem, ".py")
    udf_path.write_text(source)
    udf.pop("function_definition", None)
    udf["file"] = udf_path.name
    return str(udf_path)


def _extract_query_to_sql_file(
    ds_doc: dict[str, Any],
    ds_dir: Path,
    file_stem: str,
) -> Optional[str]:
    """Externalize ``ds_doc['query']`` to a sibling ``.sql`` file.

    Mirrors :func:`_extract_udf_to_py_file` for the BatchSource ``query`` /
    ``query_file`` sidecar pair.  Mutates *ds_doc* in place: when a
    non-empty ``query`` string is present it is removed from the
    datasource doc and replaced with a ``query_file`` key holding the
    bare filename (relative to *ds_dir*).  The SQL body is written to
    ``<ds_dir>/<file_stem>.sql``.

    No length threshold — every recovered query goes to a sidecar so
    the YAML stays thin and the on-disk shape matches what
    ``snow feature init`` scaffolds (Plan §7 OS2 default).  Empty,
    missing, or non-string ``query`` values are no-ops; ``query_file``
    is left untouched in those cases (the helper is idempotent against
    a doc that already carries ``query_file``).

    The bare-filename convention matches what
    :func:`compiler.inline_query_source` resolves on load (it joins
    ``query_file`` with the spec file's directory) — so the exported
    YAML round-trips through ``snow feature apply`` without manual
    edits.

    Args:
        ds_doc: BatchSource datasource dict being assembled for YAML
            serialization.
        ds_dir: Directory in which the datasource YAML will be written
            (the ``.sql`` sidecar lands beside it).
        file_stem: Base filename (typically the source name) used for
            the ``.sql`` sidecar.

    Returns:
        Absolute path to the written ``.sql`` file as a string, or
        ``None`` if no extraction occurred.
    """
    query = ds_doc.get("query")
    if not isinstance(query, str) or not query:
        return None

    sql_path = _safe_sidecar_path(ds_dir, file_stem, ".sql")
    sql_path.write_text(query)
    ds_doc.pop("query", None)
    ds_doc["query_file"] = sql_path.name
    return str(sql_path)


# Mapping from SPECIFICATION JSON ``source_type`` (as emitted by Snowflake)
# to the authoring-format top-level ``kind`` for the datasource YAML.  The
# ``Request`` and ``Features`` source kinds are intentionally excluded —
# they describe synthetic realtime-FV inputs that have no external system
# to materialize as a top-level datasource.
# ``Batch`` is the canonical ``source_type`` value emitted by the
# imperative side (:data:`snowflake.ml.feature_store.spec.enums.SourceType.BATCH`)
# and by the consolidated decl serializer.  ``BatchSource`` is the
# pre-consolidation decl-side value retained for plan-file backward
# compatibility, and ``OfflineTable`` is a Go-backend value carried over
# from earlier spec emissions.  The ``SQLDataSource`` entry was removed
# during the decl/spec enum consolidation.
_SOURCE_TYPE_TO_KIND: dict[str, str] = {
    "Stream": "StreamingSource",
    "Batch": "BatchSource",
    "BatchSource": "BatchSource",
    "OfflineTable": "BatchSource",
}

_REALTIME_SYNTHETIC_SOURCE_TYPES: frozenset[str] = frozenset({"Request", "Features"})

_DS_TOP_LEVEL_ORDER: tuple[str, ...] = (
    "kind",
    "name",
    "version",
    "type",
    "source_database",
    "source_schema",
    "table",
    "query",
    "query_file",
    "columns",
)

# Fields copied verbatim from the FV inline source dict to the emitted
# datasource YAML, when present on the source.  ``type`` is only honoured for
# StreamingSource (see :func:`_collect_datasources`); for non-streaming kinds
# any stray ``type`` key on the FV source dict is ignored.
_DS_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "source_database",
    "source_schema",
    "table",
    "query",
    "version",
)


def _ordered_ds_doc(ds_spec: dict[str, Any]) -> dict[str, Any]:
    """Return *ds_spec* re-keyed in the canonical authoring order.

    Mirrors :func:`_ordered_fv_doc`: well-known keys come first in a fixed
    order; any unknown keys retain their insertion order at the end.

    Args:
        ds_spec: Datasource dict to re-order.

    Returns:
        New dict with the same keys, ordered with well-known keys first.
    """
    ordered: dict[str, Any] = {}
    for key in _DS_TOP_LEVEL_ORDER:
        if key in ds_spec:
            ordered[key] = ds_spec[key]
    for key, value in ds_spec.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _collect_datasources(
    specification_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build datasource dicts from the union of every FV's ``spec.sources[]``.

    Groups by lowercase source name (so case differences across FVs collapse
    to one entry) and merges columns into a superset (deduped by lowercase
    column name).  The first occurrence's case is preserved for the emitted
    name and column names; the returned dict is keyed by that case-preserved
    emit name so callers can use it directly as the YAML filename.

    Strict-fail on conflict: if two FVs reference the same source with the
    same column but disagree on its ``type``, raises :class:`ValueError`
    naming the datasource, the column, and both contributing FVs.  The same
    rule applies to top-level passthrough fields (``source_database``,
    ``source_schema``, ``table``, ``query``, ``version``) and to the
    StreamingSource ``type`` field — any disagreement is a hard error.

    Realtime synthetic source types (``Request``, ``Features``) are silently
    skipped — they have no external system to materialize as a top-level
    datasource.  Any other unrecognised ``source_type`` value raises
    :class:`ValueError` naming the offending value.

    Args:
        specification_map: Map of OFT name → parsed spec JSON, as returned by
            ``DESCRIBE ONLINE FEATURE TABLE <name> TYPE = SPECIFICATION``.

    Returns:
        Dict keyed by case-preserved emit name → datasource spec dict, each
        ready for :func:`_ordered_ds_doc` and YAML serialization.

    Raises:
        ValueError: On any conflict between two FVs (column type, passthrough
            field, or kind), or on an unrecognised ``source_type`` value.
    """
    by_name: dict[str, dict[str, Any]] = {}
    name_case: dict[str, str] = {}
    col_index: dict[str, dict[str, tuple[str, Optional[str], str]]] = {}
    meta_index: dict[str, dict[str, tuple[Any, str]]] = {}

    for oft_name, full_spec in specification_map.items():
        if not isinstance(full_spec, dict):
            continue
        _md = full_spec.get("metadata")
        metadata = _md if isinstance(_md, dict) else {}
        fv_name = metadata.get("name") or oft_name or "<unknown>"
        inner = full_spec.get("spec") if isinstance(full_spec.get("spec"), dict) else {}
        sources = inner.get("sources") if isinstance(inner, dict) else None
        if not isinstance(sources, list):
            continue

        for src in sources:
            if not isinstance(src, dict):
                continue
            source_type = src.get("source_type", "")
            if source_type in _REALTIME_SYNTHETIC_SOURCE_TYPES:
                continue
            kind = _SOURCE_TYPE_TO_KIND.get(source_type)
            if kind is None:
                raise ValueError(
                    f"datasource export: source '{src.get('name', '<unknown>')}' in feature view "
                    f"'{fv_name}' has unrecognised source_type '{source_type}'."
                )
            raw_name = src.get("name")
            if not raw_name:
                continue
            lower_name = str(raw_name).lower()

            if lower_name not in by_name:
                emit_name = str(raw_name)
                ds: dict[str, Any] = {"kind": kind, "name": emit_name}
                if kind == "StreamingSource":
                    ds["type"] = src.get("type") or "REST"
                for fld in _DS_PASSTHROUGH_FIELDS:
                    if src.get(fld) is not None:
                        ds[fld] = src[fld]
                ds["columns"] = []
                by_name[lower_name] = ds
                name_case[lower_name] = emit_name
                col_index[lower_name] = {}
                meta_index[lower_name] = {}
                if kind == "StreamingSource":
                    meta_index[lower_name]["type"] = (ds["type"], fv_name)
                for fld in _DS_PASSTHROUGH_FIELDS:
                    if fld in ds:
                        meta_index[lower_name][fld] = (ds[fld], fv_name)
            else:
                ds = by_name[lower_name]
                if ds["kind"] != kind:
                    prev_fv = next(iter(meta_index[lower_name].values()))[1] if meta_index[lower_name] else "<unknown>"
                    raise ValueError(
                        f"datasource export: source '{name_case[lower_name]}' has conflicting "
                        f"kinds: '{ds['kind']}' (from feature view '{prev_fv}') vs "
                        f"'{kind}' (from feature view '{fv_name}')."
                    )
                check_fields: tuple[str, ...] = _DS_PASSTHROUGH_FIELDS
                if kind == "StreamingSource":
                    check_fields = ("type",) + _DS_PASSTHROUGH_FIELDS
                for fld in check_fields:
                    src_val = src.get(fld)
                    if src_val is None:
                        continue
                    current = meta_index[lower_name]
                    if fld in current:
                        prev_val, prev_fv = current[fld]
                        if prev_val != src_val:
                            raise ValueError(
                                f"datasource export: source '{name_case[lower_name]}' field "
                                f"'{fld}' conflicts: '{prev_val}' (from feature view "
                                f"'{prev_fv}') vs '{src_val}' (from feature view '{fv_name}')."
                            )
                    else:
                        current[fld] = (src_val, fv_name)
                        ds[fld] = src_val

            cols = src.get("columns")
            if not isinstance(cols, list):
                continue
            for col in cols:
                if not isinstance(col, dict):
                    continue
                col_name = col.get("name")
                if not col_name:
                    continue
                lower_col = str(col_name).lower()
                col_type = col.get("type")
                if lower_col in col_index[lower_name]:
                    prev_col_name, prev_type, prev_fv = col_index[lower_name][lower_col]
                    if col_type != prev_type:
                        raise ValueError(
                            f"datasource export: source '{name_case[lower_name]}' column "
                            f"'{prev_col_name}' type conflicts: '{prev_type}' (from feature "
                            f"view '{prev_fv}') vs '{col_type}' (from feature view "
                            f"'{fv_name}')."
                        )
                else:
                    col_index[lower_name][lower_col] = (str(col_name), col_type, fv_name)
                    by_name[lower_name]["columns"].append(dict(col))

    return {name_case[lower_name]: ds for lower_name, ds in by_name.items()}


def _entity_cols_from_full_spec(full_spec: dict[str, Any]) -> list[str]:
    """Return the join-key column names from a full spec JSON document.

    Args:
        full_spec: Spec JSON returned by ``DESCRIBE ... TYPE = SPECIFICATION``.

    Returns:
        List of entity column names, or an empty list if absent.
    """
    inner = full_spec.get("spec", {}) if isinstance(full_spec.get("spec"), dict) else {}
    cols = inner.get("ordered_entity_column_names")
    if isinstance(cols, list):
        return [str(c) for c in cols]
    return []


def _entity_name_from_row(row: dict[str, Any]) -> Optional[str]:
    """Return the prefix-stripped entity name from a SHOW TAGS row.

    Mirrors :func:`state._build_entity_object` so the exporter and the
    applied-state parser produce identical canonical names.  Rows whose
    ``name`` does not carry the ``SNOWML_FEATURE_STORE_ENTITY_`` prefix
    are skipped (defensive — a non-entity tag should not appear in the
    list returned by :func:`decl_api.fetch_entity_rows`).

    Args:
        row: One row dict from
            :func:`decl_api.fetch_entity_rows`.

    Returns:
        The uppercase entity name, or ``None`` if the row is malformed.
    """
    raw_name = row.get("name") or row.get("NAME") or ""
    if not isinstance(raw_name, str) or not raw_name:
        return None
    if not raw_name.startswith(_ENTITY_TAG_PREFIX):
        return None
    return raw_name[len(_ENTITY_TAG_PREFIX) :].upper()


def _join_keys_from_row(row: dict[str, Any]) -> list[str]:
    """Decode the join-key list from a SHOW TAGS row's ``allowed_values``.

    Mirrors the parsing in :func:`state._build_entity_object`: prefer a
    list value verbatim, otherwise treat ``allowed_values`` as JSON, with
    a final fallback to a comma-split form for legacy SHOW TAGS output.

    Args:
        row: One row dict from
            :func:`decl_api.fetch_entity_rows`.

    Returns:
        List of join-key column names (preserving case).  Empty when
        ``allowed_values`` is absent or unparsable.
    """
    raw = row.get("allowed_values") or row.get("ALLOWED_VALUES") or row.get("allowed_values_list")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [v.strip() for v in raw.strip("[]").split(",") if v.strip()]
    return []


def _entity_yaml_from_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Build a full-fidelity entity authoring dict from a SHOW TAGS row.

    The shape mirrors what :func:`state._build_entity_object` emits as
    ``spec_payload`` (minus the ``database`` / ``schema`` fields, which
    the loader fills in from the directory layout).  Per-join-key types
    are not carried by tag rows, so every join key is normalised to
    ``StringType`` — matching the applied-state path so the structural
    fingerprints align (see :func:`invariants._structural_fingerprint`).

    The ``description`` field carries the tag's ``comment`` so the
    user-authored description round-trips back into authoring YAML.

    Args:
        row: One row dict from
            :func:`decl_api.fetch_entity_rows`.

    Returns:
        Authoring-format entity dict ready for YAML serialization, or
        ``None`` if the row is missing a recognisable entity name.
    """
    entity_name = _entity_name_from_row(row)
    if not entity_name:
        return None

    join_keys = _join_keys_from_row(row)
    if not join_keys:
        # Defensive: an entity tag with no allowed_values is a degenerate
        # state, but we still want to emit a YAML so the planner sees it
        # (and produces NO_CHANGE on round-trip).  Use the entity name
        # itself as the implicit join-key column.
        join_keys = [entity_name]

    spec: dict[str, Any] = {
        "kind": "Entity",
        "name": entity_name,
        "join_keys": [{"name": jk.upper(), "type": "StringType"} for jk in join_keys],
    }
    comment = row.get("comment") or row.get("COMMENT")
    if isinstance(comment, str) and comment:
        spec["description"] = comment
    return spec


_VALID_LAYOUTS = ("db_schema", "sources")


# Canonical authoring-format key order for FeatureGroup YAML.  Mirrors
# the ``_FV_TOP_LEVEL_ORDER`` policy: identity first, then schema-shaping
# fields, then sources.  Any keys not listed here are appended in their
# natural (insertion) order so future fields land predictably without
# requiring a re-order of this tuple.
_FG_TOP_LEVEL_ORDER: tuple[str, ...] = (
    "kind",
    "name",
    "version",
    "database",
    "schema",
    "desc",
    "auto_prefix",
    "feature_views",
)


def _fg_source_to_authoring_dict(source: dict[str, Any]) -> dict[str, Any]:
    """Translate one imperative FG source row into authoring-shape dict.

    The imperative row carries ``fv_name`` / ``fv_version`` (the keys
    used by :class:`FeatureGroupMetadata.FvSourceRef` on the imperative
    side); the authoring YAML uses ``name`` / ``version`` (matching
    :class:`spec_models.FeatureViewRef`).  ``slice_columns`` is
    omitted when ``None`` or empty (matching the validator's "absent
    means no slice"); ``alias`` is preserved as an explicit key
    whenever present in the row (including ``alias=""``, which is
    distinct from "absent").

    Strict-fail on missing/empty ``fv_name`` / ``fv_version``: emitting
    a YAML with ``name: null`` / ``version: null`` (or omitted keys)
    would re-load as ``FeatureViewRef(version="")``, which the loader
    validator rejects with a less-actionable message.  Raising here
    points the operator at the upstream metadata problem in the row
    instead of the downstream loader symptom.

    Args:
        source: One element of the row's ``sources`` list.

    Returns:
        Dict in canonical authoring shape ready for YAML serialisation.

    Raises:
        ValueError: If ``fv_name`` or ``fv_version`` is missing or empty.
    """
    fv_name = source.get("fv_name") or ""
    fv_version = source.get("fv_version") or ""
    if not fv_name:
        raise ValueError(
            f"FeatureGroup source ref is missing 'fv_name' "
            f"(row: {source!r}) — refusing to emit a YAML feature_views[] "
            f"entry that would re-validate as missing-name"
        )
    if not fv_version:
        raise ValueError(
            f"FeatureGroup source ref '{fv_name}' is missing 'fv_version' "
            f"(row: {source!r}) — refusing to emit a YAML feature_views[] "
            f"entry that would re-validate as MISSING_VERSION"
        )
    out: dict[str, Any] = {
        "name": fv_name,
        "version": fv_version,
    }
    slice_cols = source.get("slice_columns")
    if slice_cols:
        out["slice_columns"] = list(slice_cols)
    if "alias" in source:
        out["alias"] = source["alias"]
    return out


def _ordered_fg_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Reorder an FG authoring dict by ``_FG_TOP_LEVEL_ORDER``.

    Args:
        doc: Authoring-format FG dict (kind/name/version/...).

    Returns:
        New dict with stable key order suitable for ``yaml.dump``.
    """
    ordered: dict[str, Any] = {}
    for key in _FG_TOP_LEVEL_ORDER:
        if key in doc:
            ordered[key] = doc[key]
    for key, val in doc.items():
        if key not in ordered:
            ordered[key] = val
    return ordered


def _build_fg_yaml_doc(row: dict[str, Any]) -> dict[str, Any]:
    """Build a canonical authoring-format FG dict from an imperative row.

    The row shape mirrors :func:`imperative_executor.fetch_feature_group_rows`
    output: ``{name, version, desc, owner, auto_prefix, sources,
    output_columns, database_name, schema_name}``.  ``output_columns``
    is intentionally NOT emitted into the YAML — it is a derived field
    on the imperative side and is excluded from the FG content hash for
    exactly the same reason; round-tripping it would create a spurious
    diff under different declarative orderings of the same FV refs.

    Strict-fail on missing/empty ``name`` / ``version``: ``yaml.dump``
    would otherwise emit ``version: null`` (or omit the key entirely),
    and the next ``snow feature plan`` would re-validate the file as
    ``MISSING_VERSION`` — a confusing downstream symptom of the real
    upstream metadata problem.  Raising here surfaces the row that's
    actually broken so the operator can fix the source instead of
    debugging the loader.  Mirrors the OFT strict-fail pattern further
    down in this module (DESCRIBE … TYPE = SPECIFICATION).

    Args:
        row: One row dict from
            :func:`imperative_executor.fetch_feature_group_rows`.

    Returns:
        Authoring-format FG dict ready for YAML serialisation.

    Raises:
        ValueError: If ``name`` or ``version`` is missing or empty.
    """
    name = row.get("name") or ""
    version = row.get("version") or ""
    if not name:
        raise ValueError(
            f"FeatureGroup export row is missing 'name' (row: {row!r}) — " f"refusing to write a YAML with no identity"
        )
    if not version:
        raise ValueError(
            f"FeatureGroup '{name}' export row is missing 'version' "
            f"(row: {row!r}) — refusing to write a YAML that would "
            f"re-validate as MISSING_VERSION on the next snow feature plan"
        )
    sources = row.get("sources") or []
    feature_views = [_fg_source_to_authoring_dict(s) for s in sources]
    doc: dict[str, Any] = {
        "kind": "FeatureGroup",
        "name": name,
        "version": version,
        "desc": row.get("desc", "") or "",
        "auto_prefix": bool(row.get("auto_prefix", True)),
        "feature_views": feature_views,
    }
    return doc


# Canonical FV-kind discriminator values used to filter ``AppliedState``
# down to the subset the exporter knows how to render.  Keep in sync
# with :data:`state._FV_KIND_MAP` — entities / datasources /
# feature-groups travel through their dedicated kwargs.
_FV_APPLIED_KINDS: frozenset[str] = frozenset({"BatchFeatureView", "StreamingFeatureView", "RealtimeFeatureView"})


def _overlay_applied_state_on_export_inputs(
    show_rows: list[dict[str, Any]],
    specification_map: Optional[dict[str, dict[str, Any]]],
    applied_state: Optional[Any],
    *,
    default_database: str,
    default_schema: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Overlay an :class:`AppliedState` snapshot onto raw exporter inputs.

    Returns the ``(show_rows, spec_map)`` pair the rest of
    :func:`export_specs` consumes, with two transformations applied
    when *applied_state* is provided:

    1. **Recovered-payload precedence.**  For every FV-kind
       ``AppliedObject`` whose synthesised OFT name matches an entry
       in *show_rows* / *specification_map*, the object's
       ``spec_payload`` overwrites the raw ``specification_map`` entry
       — this is how recovered BatchFV ``sources`` (injected from DT
       text by :func:`state._inject_batch_fv_source_from_dt_text`,
       transitioning under Plan B-state to the
       ``FV_SOURCE_REFS``-driven
       :func:`state._inject_batch_fv_source_from_metadata`) and
       advanced BFV fields (``cluster_by``, ``refresh_mode``,
       ``initialize`` — injected by
       :func:`state._inject_advanced_bfv_fields_from_dt_text`) make
       their way into the exported YAML.  Without this rule, the
       lossy raw SPECIFICATION JSON would always win.

       FVs absent from *applied_state* (legacy / fresh-init case) keep
       their raw ``specification_map`` entry untouched so the export
       still produces a YAML for them — soft-fallback per Plan B8.
    2. **Offline-only synthesis.**  When an ``AppliedObject`` has no
       matching *show_rows* entry (because it was surfaced by
       :func:`state._build_offline_fv_object` from
       ``feature_view_rows`` rather than ``SHOW ONLINE FEATURE
       TABLES``), a synthetic *show_row* is appended so the FV
       reaches the YAML-emission loop.  Only the keys the exporter
       actually consumes (``name`` / ``database_name`` /
       ``schema_name``) are populated — runtime metadata such as
       ``scheduling_state`` is never authoring surface and is no
       longer round-tripped into the YAML.

    Args:
        show_rows: Caller-provided rows from ``SHOW ONLINE FEATURE
            TABLES``.  Returned as a new list (the original is never
            mutated) extended with offline-only synthesised rows.
        specification_map: Caller-provided OFT → spec_payload map.
            Returned as a new dict (never mutated) with recovered
            payloads from *applied_state* overlaid.
        applied_state: Optional :class:`AppliedState` snapshot.
            ``None`` short-circuits — the function returns
            ``(show_rows, specification_map or {})`` unchanged so
            legacy callers keep working byte-stably.
        default_database: Fallback database for synthesised
            show-rows when an ``AppliedObject``'s ``spec_payload.
            metadata.database`` is empty.
        default_schema: Fallback schema for synthesised show-rows
            when an ``AppliedObject``'s
            ``spec_payload.metadata.schema`` is empty.

    Returns:
        Tuple ``(show_rows, spec_map)`` ready for the rest of
        :func:`export_specs`.  Both are fresh objects when overlay
        was performed; otherwise ``spec_map`` is ``specification_map
        or {}`` and ``show_rows`` is the input reference.
    """
    if applied_state is None:
        return show_rows, specification_map or {}

    objects = getattr(applied_state, "objects", None)
    if not objects:
        return show_rows, specification_map or {}

    overlaid_spec_map: dict[str, dict[str, Any]] = dict(specification_map or {})
    existing_oft_names: set[str] = {str(r.get("name", "")) for r in show_rows if r.get("name")}
    synthesised_rows: list[dict[str, Any]] = []

    # TODO(gate-b cross-stream): the BatchFV ``Source`` bindings inside
    # ``spec_payload["spec"]["sources"]`` are produced today by
    # :func:`state._inject_batch_fv_source_from_dt_text`.  Plan B-state
    # is rewriting that helper as
    # :func:`state._inject_batch_fv_source_from_metadata` driven by the
    # ``FV_SOURCE_REFS`` row introduced in Plan A1.  The exporter reads
    # the *result* (``spec_payload["spec"]["sources"]``) regardless of
    # which injector produced it — but Gate B's full-test cross-check
    # must confirm B-state still surfaces the recovered list under the
    # same key path.  If B-state moves the bindings (e.g. to a
    # top-level ``spec_payload["sources_metadata"]``), this overlay
    # function and ``_build_full_fidelity_fv`` need a parallel update.
    for obj in objects.values():
        kind = getattr(obj, "kind", "")
        if kind not in _FV_APPLIED_KINDS:
            continue
        spec_payload = getattr(obj, "spec_payload", None)
        if not isinstance(spec_payload, dict) or not spec_payload:
            continue
        name = getattr(obj, "name", "") or ""
        version = getattr(obj, "version", "") or ""
        if not name or not version:
            continue
        # OFT-naming contract: ``<base>$<version>$ONLINE`` — must match
        # what :func:`state._parse_oft_name` expects on the inverse
        # parse so synthesised show-rows are indistinguishable from
        # the real SHOW output.
        oft_name = f"{name}${version}$ONLINE"

        # Precedence rule: applied_state's recovered payload always
        # wins for collisions — that is the entire point of the
        # overlay.  Raw specification_map entries survive only for
        # FVs that applied_state does not know about (defensive: the
        # plan path normally populates every OFT).
        overlaid_spec_map[oft_name] = spec_payload

        if oft_name in existing_oft_names:
            continue
        _md = spec_payload.get("metadata")
        metadata = _md if isinstance(_md, dict) else {}
        synth_db = metadata.get("database") or default_database
        synth_schema = metadata.get("schema") or default_schema
        synthesised_rows.append(
            {
                "name": oft_name,
                "database_name": synth_db,
                "schema_name": synth_schema,
            }
        )
        existing_oft_names.add(oft_name)

    if not synthesised_rows:
        return list(show_rows), overlaid_spec_map
    return list(show_rows) + synthesised_rows, overlaid_spec_map


def export_specs(
    show_rows: list[dict[str, Any]],
    describe_rows_by_oft: dict[str, list[dict[str, Any]]],
    output_dir: str,
    database: str,
    schema: str,
    *,
    specification_map: Optional[dict[str, dict[str, Any]]] = None,
    entity_rows: Optional[list[dict[str, Any]]] = None,
    feature_group_rows: Optional[list[dict[str, Any]]] = None,
    applied_state: Optional[Any] = None,
    layout: str = "db_schema",
) -> dict[str, Any]:
    """Reconstruct YAML specs from raw Snowflake query results and write to disk.

    Strict mode: every row in ``SHOW ONLINE FEATURE TABLES`` must have a
    matching non-empty entry in *specification_map* (the parsed JSON
    returned by ``DESCRIBE ONLINE FEATURE TABLE <name> TYPE =
    SPECIFICATION``).  When that requirement is met, each YAML preserves
    UDF source, source column schemas, feature aggregations
    (function/window/offset), granularity, and target lag — enabling a
    round-trip ``snow feature apply`` without manual edits.

    Entity YAML emission is driven by *entity_rows* (the legacy SHOW
    TAGS-shape rows returned by :func:`decl_api.fetch_entity_rows`).
    Every entity tag is materialised to YAML — including orphan tags
    not referenced by any FV — and every FV's ``entities`` (the
    authoring-side name; recovered from the wire-form
    ``ordered_entity_column_names`` key in DESCRIBE output) is
    cross-checked against the provided rows so a deployed FV
    referencing an unknown entity is a hard error.  ``entity_rows=None`` and ``entity_rows=[]`` are
    treated identically: no entity YAMLs are emitted and the strict
    subset check is skipped (the caller has opted out of authoritative
    entity emission).  Callers that want entity YAMLs on disk must
    forward the output of :func:`decl_api.fetch_entity_rows`.

    Datasource YAML files are written for the deduped column-superset
    of every FV's ``spec.sources[]`` entry.

    Args:
        show_rows: Rows from ``SHOW ONLINE FEATURE TABLES`` (list of dicts).
        describe_rows_by_oft: Map of OFT name → ``DESCRIBE`` rows (list of
            dicts).  Retained for forward compatibility; not consumed in
            strict mode.
        output_dir: Base output directory.  Under the default
            ``layout="db_schema"`` a ``<database>.<schema>`` subdirectory
            is created inside it; under ``layout="sources"`` the YAMLs
            land directly in ``<output_dir>/sources/`` (mirroring the
            manifest project tree scaffolded by ``snow feature init``).
        database: Connection database name, used as the directory prefix and
            as a fallback when a row lacks one.
        schema: Connection schema name, used as the directory prefix and as
            a fallback when a row lacks one.
        specification_map: Map of OFT name → parsed spec JSON returned by
            ``DESCRIBE ... TYPE = SPECIFICATION``.  Required for every OFT
            in *show_rows*; missing or empty entries abort the export.
        entity_rows: Rows from
            ``SHOW TAGS LIKE 'SNOWML_FEATURE_STORE_ENTITY_%'`` (or the
            equivalent imperative ``FeatureStore.list_entities()``
            output).  When non-empty, these rows are the authoritative
            source of truth for entity emission: every row produces a
            YAML stub, and every FV ``entities`` entry (recovered from
            the wire-form ``ordered_entity_column_names`` key) is
            validated against the row set.  ``None`` and ``[]`` are
            treated identically — no entity YAMLs are written and the
            subset check is skipped.
        feature_group_rows: Rows produced by
            :func:`decl_api.fetch_feature_group_rows` (the imperative
            ``FeatureStore.list_feature_groups()`` output, normalised
            into the executor's row shape).  When non-empty, every row
            materialises a YAML in
            ``<base>/feature_groups/<NAME>.yaml``, including FGs whose
            source FVs reference advanced batch fields (``cluster_by``,
            ``storage_config``, etc.) — those advanced fields live on
            the FV YAML, not the FG, so the FG export path is
            decoupled from the BFV authoring surface.  ``None`` and
            ``[]`` are treated identically — no FG YAMLs are written.
        applied_state: Optional :class:`AppliedState` snapshot produced
            by :func:`fetch_applied_state` (the same surface the plan
            path consumes).  When present, the exporter overlays its
            FV-kind ``AppliedObject.spec_payload`` entries on top of
            *specification_map* — the recovered payload wins for every
            OFT also present in *show_rows*, and any FV in
            ``applied_state.objects`` that does not have a matching
            *show_rows* entry (offline-only BatchFVs surfaced through
            :func:`state._build_offline_fv_object`) is synthesised
            into a show-row so it still gets a YAML.  This is the
            init-export unification surface: with it,
            ``snow feature init`` no longer drops the BatchFV
            ``sources[]`` / advanced fields / offline-only entries
            the SPECIFICATION JSON loses on round-trip.  When
            ``None``, the legacy ``specification_map``-only codepath
            runs unchanged (back-compat for downstream callers that
            haven't migrated to the applied-state surface).
        layout: ``"db_schema"`` (default, back-compat) writes to
            ``<output_dir>/<database>.<schema>/{entities,datasources,
            feature_views}/``.  ``"sources"`` writes to
            ``<output_dir>/sources/{entities,datasources,feature_views}/``
            so the result drops straight into the manifest project tree
            consumed by ``snow feature plan`` / ``apply``.

    Returns:
        Dict with keys:

        - ``status``: ``"exported"``
        - ``directory``: Absolute path to the export base directory
          (``<output_dir>/<database>.<schema>`` for ``layout="db_schema"``
          or ``<output_dir>/sources`` for ``layout="sources"``), or
          ``""`` if both *show_rows* and *entity_rows* are empty.
        - ``files``: List of absolute paths to the written YAML files.

    Raises:
        ValueError: If *layout* is not one of ``"db_schema"`` /
            ``"sources"``; if any OFT in *show_rows* has no
            corresponding entry in *specification_map* (or maps to an
            empty/non-dict value); if a FV's ``entities`` (recovered
            from the wire-form ``ordered_entity_column_names`` key)
            references an entity name absent from *entity_rows*; if
            two FVs disagree on a datasource column type, kind, or
            passthrough field; or if a FV source carries an
            unrecognised ``source_type`` value.
    """
    del describe_rows_by_oft  # retained in signature for forward compatibility

    if layout not in _VALID_LAYOUTS:
        raise ValueError(f"export_specs: unknown layout {layout!r}; expected one of " f"{_VALID_LAYOUTS!r}")

    # Normalise the caller-facing ``Optional[list]`` shape into a list so
    # the rest of the function can iterate without re-checking for None.
    # Empty (None or []) means "no entities registered" — no entity YAMLs
    # are written and the strict subset check below is skipped.
    entity_rows = entity_rows or []
    feature_group_rows = feature_group_rows or []

    # When the caller hands us an ``AppliedState`` (the same surface the
    # plan path consumes), overlay its recovered FV payloads on top of
    # the raw ``specification_map`` and synthesise show-rows for any
    # offline-only FVs that the SHOW pipeline cannot enumerate.  This is
    # the init-export unification: without it, BatchFV YAMLs land with
    # ``sources: []`` because the raw SPECIFICATION JSON drops the
    # offline-DT source binding, and offline-only BFVs silently
    # disappear from the exported tree.  Keep the overlay scoped to FV
    # kinds — entities and feature-groups still flow through their
    # dedicated kwargs.
    show_rows, specification_map = _overlay_applied_state_on_export_inputs(
        show_rows,
        specification_map,
        applied_state,
        default_database=database,
        default_schema=schema,
    )

    # Empty input means there is genuinely nothing to export (no FVs,
    # no entity tags, no FGs).  Preserved so a fresh-schema export call
    # returns a noop envelope without creating empty subdirectories.
    if not show_rows and not entity_rows and not feature_group_rows:
        return {"status": "exported", "directory": "", "files": []}

    spec_map = specification_map or {}

    if layout == "sources":
        base = Path(output_dir) / "sources"
    else:
        base = Path(output_dir) / f"{database}.{schema}"
    entities_dir = base / "entities"
    fv_dir = base / "feature_views"
    for d in (entities_dir, fv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-validate FV ↔ entity_rows consistency before writing any files.
    # When entity_rows is non-empty, it is authoritative: every
    # FV-referenced entity must appear in the row set or the deployed
    # runtime is in an inconsistent state and we abort cleanly to avoid
    # emitting a half-baked tree that won't re-apply.  An empty
    # entity_rows means the caller has opted out of entity emission; the
    # subset check is vacuous in that case and is skipped.
    # ------------------------------------------------------------------
    if entity_rows:
        known_entity_names: set[str] = set()
        for row in entity_rows:
            ent_name = _entity_name_from_row(row)
            if ent_name:
                known_entity_names.add(ent_name)

        for row in show_rows:
            raw_name = row.get("name", "")
            full_spec = spec_map.get(raw_name)
            if not isinstance(full_spec, dict) or not full_spec:
                continue  # downstream loop will raise the strict error
            for col_name in _entity_cols_from_full_spec(full_spec):
                if col_name.upper() not in known_entity_names:
                    raise ValueError(
                        f"OFT '{raw_name}' references entity column "
                        f"'{col_name}' which is not present in entity_rows. "
                        "The deployed runtime is inconsistent; aborting export "
                        "before any partial state is written.  Re-run after "
                        "verifying the entity tag exists, or omit entity_rows "
                        "to opt out of entity YAML emission entirely."
                    )

    seen_entities: set[str] = set()
    created: list[str] = []

    for row in show_rows:
        raw_name = row.get("name", "")
        db = row.get("database_name", database)
        sch = row.get("schema_name", schema)

        full_spec = spec_map.get(raw_name)
        if not isinstance(full_spec, dict) or not full_spec:
            raise ValueError(
                f"DESCRIBE … TYPE = SPECIFICATION returned no spec for OFT "
                f"'{raw_name}'. Cannot export without authoritative specification."
            )

        fallback_name, fallback_version = _parse_oft_name(raw_name)
        fv_doc = _build_full_fidelity_fv(
            full_spec,
            fallback_name=fallback_name,
            fallback_version=fallback_version,
            fallback_database=db,
            fallback_schema=sch,
        )
        file_stem = str(fv_doc.get("name") or fallback_name)

        # Extract UDF source into a sibling ``.py`` file before serializing the
        # YAML, so the YAML carries a ``file:`` reference instead of inlining
        # the source.  This mirrors the authoring format (``UDF.file`` in
        # ``spec_models.py``) and is reversed by ``compiler.inline_udf_source``
        # when the YAML is later loaded for ``snow feature apply``.
        udf_py_path = _extract_udf_to_py_file(fv_doc, fv_dir, file_stem)

        fv_path = fv_dir / f"{file_stem}.yaml"
        fv_path.write_text(yaml.dump(_ordered_fv_doc(fv_doc), default_flow_style=False, sort_keys=False))
        created.append(str(fv_path))
        if udf_py_path is not None:
            created.append(udf_py_path)

    # Authoritative entity emission: drive directly from entity_rows.
    # Every tag (referenced or orphan) gets a YAML.  Naming + casing
    # match :func:`state._build_entity_object` so structural
    # fingerprints align and the planner reports NO_CHANGE on an
    # unmodified round-trip.  When ``entity_rows`` was normalised to
    # ``[]`` above (caller passed ``None`` or ``[]``), the loop is a
    # noop and no entity YAMLs are written.
    for row in entity_rows:
        entity_spec = _entity_yaml_from_row(row)
        if entity_spec is None:
            continue
        ent_name = entity_spec["name"]
        if ent_name in seen_entities:
            continue
        seen_entities.add(ent_name)
        entity_path = entities_dir / f"{ent_name}.yaml"
        entity_path.write_text(yaml.dump(entity_spec, default_flow_style=False, sort_keys=False))
        created.append(str(entity_path))

    # FeatureGroup emission: drive directly from feature_group_rows.
    # Every row produces a ``feature_groups/<NAME>.yaml`` stub in the
    # canonical authoring shape.  When ``feature_group_rows`` was
    # normalised to ``[]`` (caller passed ``None`` or ``[]``), the
    # loop is a noop and no FG YAMLs (or directory) are created.
    if feature_group_rows:
        feature_groups_dir = base / "feature_groups"
        feature_groups_dir.mkdir(parents=True, exist_ok=True)
        seen_fg_files: set[str] = set()
        for row in feature_group_rows:
            fg_doc = _build_fg_yaml_doc(row)
            file_stem = str(fg_doc["name"])
            if file_stem in seen_fg_files:
                continue
            seen_fg_files.add(file_stem)
            fg_path = feature_groups_dir / f"{file_stem}.yaml"
            fg_path.write_text(yaml.dump(_ordered_fg_doc(fg_doc), default_flow_style=False, sort_keys=False))
            created.append(str(fg_path))

    # Datasource emission must come AFTER all FV/entity files are written but
    # BEFORE the ``datasources/`` directory is created — a ValueError raised by
    # ``_collect_datasources`` (column-type or kind/passthrough conflict, or an
    # unrecognised ``source_type``) must propagate cleanly without leaving any
    # partially-written datasource files on disk.
    datasources = _collect_datasources(spec_map)
    if datasources:
        datasources_dir = base / "datasources"
        datasources_dir.mkdir(parents=True, exist_ok=True)
        for emit_name, ds_spec in datasources.items():
            # Externalise any inline ``query:`` body into a sibling ``.sql``
            # sidecar before serialising — mirrors the UDF emission path
            # above so the YAML stays thin and the on-disk shape matches
            # what ``snow feature init`` scaffolds.  ``compiler.inline_
            # query_source`` reverses the move on load.
            sql_sidecar_path = _extract_query_to_sql_file(ds_spec, datasources_dir, emit_name)
            ds_path = datasources_dir / f"{emit_name}.yaml"
            ds_path.write_text(yaml.dump(_ordered_ds_doc(ds_spec), default_flow_style=False, sort_keys=False))
            created.append(str(ds_path))
            if sql_sidecar_path is not None:
                created.append(sql_sidecar_path)

    return {"status": "exported", "directory": str(base), "files": created}
