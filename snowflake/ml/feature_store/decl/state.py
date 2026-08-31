"""Applied-state parser for the declarative feature store library.

Parses raw Snowflake query results into ``AppliedState``.
No connection access — accepts pre-fetched data.

**No SQL/DT-text parsing anywhere in ``decl/``** (Phase B1).  Every
recovery path consumes authoritative metadata:

* Source bindings (``sources[]``) come from ``FvSourceRefsMetadata``
  surfaced as the ``source_refs`` column on
  :meth:`FeatureStore.list_feature_views` rows (plan section A1).
* ``cluster_by`` and ``refresh_mode`` come from the dedicated columns on
  the same list-FV row.
* ``initialize`` comes from the rehydrated ``FeatureView`` object
  returned by :meth:`FeatureStore.get_feature_view` (fetched via
  :func:`decl.imperative_executor.fetch_feature_view_object`).

The legacy ``_build_datasources_by_table`` name-lookup shim is retained
as a fallback ONLY for FVs registered before the ``FV_SOURCE_REFS``
metadata row existed; when it engages, a once-per-FV ``logger.warning``
recommends re-apply so the operator can recover the authoritative
source bindings.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence, cast

from snowflake.ml._internal.utils import identifier
from snowflake.ml.feature_store.decl.invariants import (
    _VERSIONED_KINDS,
    _full_spec_hash,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    ObjectKind,
)
from snowflake.ml.feature_store.spec.enums import ENTITY_TAG_PREFIX

logger = logging.getLogger(__name__)

# OFT name separator character
_SEP = "$"
# Suffix appended to all online feature table names
_ONLINE_SUFFIX = "ONLINE"

# Legacy alias.  The canonical value lives in
# :data:`snowflake.ml.feature_store.spec.enums.ENTITY_TAG_PREFIX` (imported
# above).  Existing internal callers
# (``decl/exporter.py``, ``decl/api.py``, tests) reference the
# leading-underscore name; keeping this alias as a thin re-export means we do
# not have to chase every callsite, while the cross-module identity test
# (``spec/enums_test.py``) still passes because both names point at the same
# string object.
_ENTITY_TAG_PREFIX = ENTITY_TAG_PREFIX


# ---------------------------------------------------------------------------
# Metadata-backed source-binding recovery (Phase B1 + B3).
#
# Replaces the legacy DT-text regex parsers.  The operator-authored
# ``SourceRef`` list rides on the ``source_refs`` column of
# :meth:`FeatureStore.list_feature_views` rows (populated from the
# ``FV_SOURCE_REFS`` metadata row — see plan section A1), so the recovery
# layer reads it directly instead of string-matching the DT body.
#
# ``cluster_by`` and ``refresh_mode`` come from dedicated columns on the
# same list-FV row.  ``initialize`` comes from the rehydrated
# :class:`FeatureView` object returned by
# :meth:`FeatureStore.get_feature_view` (fetched once per tiled BFV via
# :func:`decl.imperative_executor.fetch_feature_view_object`).
# ---------------------------------------------------------------------------


# Canonical ``BatchSource`` JSON keys preserved verbatim on the
# spec_payload.  No transformation: the metadata write path encodes the
# operator-authored ``SourceRef`` shape and the recovery side keeps the
# same shape so a clean round-trip hashes identically.
_SOURCE_REF_PASS_THROUGH_KEYS: tuple[str, ...] = (
    "name",
    "source_type",
    "table",
    "query",
    "source_database",
    "source_schema",
    "columns",
)


def _normalize_source_ref(raw: Any) -> Optional[dict[str, Any]]:
    """Coerce a single ``source_refs[i]`` entry to a canonical dict.

    The metadata write path stores ``SourceRef`` entries via the same
    JSON shape :meth:`SourceRef.model_dump` / ``SourceRef.to_dict()``
    produces (``name``, ``source_type``, optional ``table`` / ``query``
    / ``columns`` / ``source_database`` / ``source_schema``).  The
    read path here keeps the shape unchanged so the planner / hash
    comparison treats a metadata-recovered source the same as a
    locally-compiled one.

    Args:
        raw: A single entry from the JSON-decoded ``source_refs`` cell.

    Returns:
        A dict with only the canonical keys (no unknown extras), or
        ``None`` when *raw* is not a mapping or carries no ``name``.
    """
    if not isinstance(raw, dict):
        return None
    name = raw.get("name") or ""
    if not isinstance(name, str) or not name:
        return None
    result: dict[str, Any] = {}
    for key in _SOURCE_REF_PASS_THROUGH_KEYS:
        if key in raw and raw[key] is not None:
            result[key] = raw[key]
    return result


def _coerce_source_refs(raw: Any) -> list[dict[str, Any]]:
    """Decode a list-FV row's ``source_refs`` cell into a list of dicts.

    Accepts the cell in any of the three shapes the read path may
    surface — already-decoded ``list``, JSON-string, or ``None`` —
    and normalises each entry via :func:`_normalize_source_ref`.

    Args:
        raw: The raw cell value (list, JSON string, or ``None``).

    Returns:
        A list of canonical source-ref dicts.  Empty list when the
        cell is missing, malformed JSON, or contains no valid
        entries.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        decoded: list[Any] = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if not isinstance(decoded, list):
            return []
    else:
        return []
    out: list[dict[str, Any]] = []
    for entry in decoded:
        norm = _normalize_source_ref(entry)
        if norm is not None:
            out.append(norm)
    return out


def _inject_batch_fv_source_from_metadata(
    spec_payload: dict[str, Any],
    source_refs: Optional[Sequence[Any]],
) -> bool:
    """Populate ``spec.sources[]`` from authoritative ``FV_SOURCE_REFS``
    metadata.

    The operator-authored ``SourceRef`` list rides on the new
    ``source_refs`` column of :meth:`FeatureStore.list_feature_views`
    rows (plan section A1).  Each entry is a JSON-friendly dict
    matching the same shape :meth:`SourceRef.model_dump` produces, so
    the recovery side passes the values through unchanged — logical
    ``name``, ``columns``, ``table``/``query`` binding, optional
    ``source_database`` / ``source_schema``.  Tiled vs non-tiled
    shape is irrelevant; the metadata path carries the operator's
    authored source regardless of how the offline DT body was
    generated.

    Args:
        spec_payload: A BatchFV spec dict (the ``specification_map``
            value or a ``_build_offline_fv_object`` output), mutated
            in place.
        source_refs: The decoded ``source_refs`` cell from the
            list-FV row.  ``None`` or empty signals a legacy
            deployment with no metadata row — caller falls back to
            the once-per-FV warning + legacy shim path.

    Returns:
        ``True`` when ``spec.sources[]`` was populated from metadata,
        ``False`` when no metadata was available (caller should
        engage the legacy fallback).
    """
    inner = spec_payload.get("spec") if isinstance(spec_payload.get("spec"), dict) else None
    if inner is None:
        return False
    decoded = _coerce_source_refs(source_refs)
    if not decoded:
        return False
    inner["sources"] = decoded
    return True


def _inject_batch_fv_fields_from_list_row(
    spec_payload: dict[str, Any],
    row: dict[str, Any],
    fv_obj: Any = None,
) -> None:
    """Inject BatchFV structural / operational fields from metadata.

    Populates the spec's ``cluster_by``, ``refresh_mode``, ``initialize``,
    and top-level ``desc`` keys from the authoritative sources —
    replacing the legacy DT-text regex parsers (Phase B1) and closing
    the L3 desc round-trip gap:

    * ``cluster_by`` ← ``row["cluster_by"]`` (list-FV column).
    * ``refresh_mode`` ← ``row["refresh_mode"]`` (list-FV column).
    * ``initialize`` ← ``fv_obj.initialize`` when *fv_obj* is supplied
      (the rehydrated :class:`FeatureView` from
      :func:`decl.imperative_executor.fetch_feature_view_object`).
    * ``desc`` (top-level of *spec_payload*) ← ``row["desc"]``
      (list-FV column, mirrors ``list_feature_views().desc``).  The
      planner's :func:`planner._resolve_applied_desc` probes the
      top-level key first, so the deployed description must land
      there for L3 ``UPDATE_FV`` → ``NO_CHANGE`` re-plan parity after
      ``FeatureStore.update_feature_view(desc=...)``.  Without this
      injection, an ``ALTER DYNAMIC TABLE … SET COMMENT = '<new>'``
      apply persists on the Snowflake side but the next plan still
      reads the spec_payload's empty desc → spurious ``UPDATE_FV``
      every replan (Phase E live verify, gap 4).

    Each injection is additive (skipped when the key already exists)
    so a spec_payload that already carries the field — e.g. via the
    enrichment in
    :func:`decl.imperative_executor._serialize_batch_fv_spec` — keeps
    its existing value.

    Args:
        spec_payload: A BatchFV spec dict, mutated in place.
        row: The list-FV row for this FV (see
            :func:`decl.imperative_executor.fetch_feature_view_rows`
            contract).  Carries ``cluster_by`` (string,
            possibly comma-separated), ``refresh_mode`` (string), and
            ``desc`` (string).
        fv_obj: Optional rehydrated :class:`FeatureView` object.
            When supplied, ``initialize`` is read from it.
    """
    desc_cell = row.get("desc") if isinstance(row, dict) else None
    if isinstance(desc_cell, str) and desc_cell.strip() and "desc" not in spec_payload:
        spec_payload["desc"] = desc_cell.strip()

    inner = spec_payload.get("spec") if isinstance(spec_payload.get("spec"), dict) else None
    if inner is None:
        return

    cluster_by_cell = row.get("cluster_by") if isinstance(row, dict) else None
    cluster_by = _parse_cluster_by_list(cluster_by_cell)
    if cluster_by is not None and "cluster_by" not in inner:
        inner["cluster_by"] = cluster_by

    refresh_cell = row.get("refresh_mode") if isinstance(row, dict) else None
    if isinstance(refresh_cell, str) and refresh_cell.strip() and "refresh_mode" not in inner:
        inner["refresh_mode"] = refresh_cell.strip().upper()

    if fv_obj is not None and "initialize" not in inner:
        initialize = getattr(fv_obj, "initialize", None)
        if initialize:
            inner["initialize"] = str(initialize).strip().upper() or None


def _spec_has_aggregation_windows(inner: dict[str, Any]) -> bool:
    """Whether an inner spec dict describes a tiled FV — i.e. any feature
    declares an aggregation window.

    Mirrors the compiler's ``has_windows`` check and the imperative
    ``FeatureView.is_tiled``: a tiled FV materialises its tiles as a
    managed Dynamic Table whose ``TARGET_LAG`` is driven by
    ``refresh_freq``.

    Args:
        inner: The inner ``spec`` dict (``spec_payload["spec"]``).

    Returns:
        ``True`` if any ``features[]`` entry carries ``window`` or
        ``window_sec``, else ``False``.
    """
    return any(
        isinstance(f, dict) and (f.get("window_sec") is not None or f.get("window") is not None)
        for f in (inner.get("features") or [])
    )


def _inject_fv_refresh_freq_from_list_row(
    spec_payload: dict[str, Any],
    row: dict[str, Any],
) -> None:
    """Inject the deployed Dynamic Table cadence onto an applied-state payload.

    Plumbs ``row["refresh_freq"]`` (the ``REFRESH_FREQ`` column of the
    deployed Dynamic Table, surfaced by
    :func:`decl.imperative_executor.fetch_feature_view_rows`) onto the
    recovered ``AppliedObject.spec_payload`` as
    ``spec.refresh_freq`` so the planner's
    :func:`_refresh_freq_drifted` helper can compare local authoring
    against the deployed cadence symmetrically.

    Without this plumbing the planner falls back to
    ``spec.target_lag_sec``, which for OFT-backed FVs carries the OFT
    staleness (``0`` for streaming, ``_BATCH_OFT_TARGET_LAG`` for
    online BFV without authored ``target_lag``) — the original BACKFILL
    re-plan invariant break.

    Kind-aware: realtime and **non-tiled** streaming spec_payloads are
    skipped entirely.  A non-tiled streaming FV materialises to a
    zero-lag VIEW and a realtime FV computes on lookup — neither has an
    offline Dynamic Table to recover a cadence from, and the spec
    validator (``FeatureView._reject_refresh_freq_on_stream_or_realtime``)
    rejects ``refresh_freq`` authoring on those shapes.  Without this
    skip the runtime-stamped ``target_lag_sec=0`` would round-trip into
    YAML as ``refresh_freq: "0 seconds"`` and the next ``snow feature
    plan`` would reject the file on load.  A **tiled** streaming FV
    (aggregation windows) DOES schedule an offline tile Dynamic Table, so
    its ``REFRESH_FREQ`` is recovered here — otherwise the planner
    compares the local cadence against the OFT ``target_lag_sec=0``
    sentinel and emits a spurious ``UPDATE_FV`` on every replan.

    Additive: an existing ``spec.refresh_freq`` (e.g. one populated via
    a pre-enrichment path) is preserved.

    Args:
        spec_payload: The ``AppliedObject.spec_payload`` dict, mutated
            in place.  May omit an inner ``spec`` dict (legacy
            embedded-specification flatten path) — the helper
            early-returns rather than mutating the top-level shape.
        row: A list-FV row dict (the
            :func:`decl.imperative_executor.fetch_feature_view_rows`
            contract).  Must carry ``refresh_freq`` for the injection
            to fire; rows that omit the key or carry an empty string
            are tolerated.
    """
    kind = spec_payload.get("kind") if isinstance(spec_payload, dict) else None
    if kind == "RealtimeFeatureView":
        return

    inner = spec_payload.get("spec") if isinstance(spec_payload.get("spec"), dict) else None
    if inner is None:
        return

    # Non-tiled streaming FVs have no offline DT (zero-lag VIEW), so skip
    # them; tiled streaming and batch always schedule a DT.
    if kind == "StreamingFeatureView" and not _spec_has_aggregation_windows(inner):
        return

    if "refresh_freq" in inner:
        return

    raw_value = row.get("refresh_freq") if isinstance(row, dict) else None
    if not isinstance(raw_value, str) or not raw_value.strip():
        return

    inner["refresh_freq"] = raw_value


def _resolve_cluster_column(value: Any) -> Optional[str]:
    """Normalise a single ``cluster_by`` column value to a resolved identifier.

    Uses :func:`identifier.resolve_identifier` so a quoted identifier with an
    internal quote (Snowflake returns it as ``"FOO""BAR"``) resolves to the
    canonical ``"FOO""BAR"`` rather than the mangled ``FOO""BAR`` a naive
    ``.strip('"')`` would produce.  Values the SQL parser would reject fall
    back to the legacy stripped form so state recovery never raises.

    Args:
        value: A single column cell from the ``cluster_by`` payload.

    Returns:
        The resolved identifier, or ``None`` when the value is empty.
    """
    text = str(value).strip()
    if not text:
        return None
    try:
        return identifier.resolve_identifier(text)
    except ValueError:
        return text.strip('"').strip() or None


def _parse_cluster_by_list(raw: Any) -> Optional[list[str]]:
    """Coerce a ``list_feature_views`` ``cluster_by`` cell to a column list.

    Snowpark / cursor paths surface this column in **three** observed
    shapes and we normalise all of them to a list of bare column
    identifiers so the planner's structural fingerprint hashes
    symmetrically across local-compile and applied-state halves:

    1. Python ``list`` — Snowpark's native ARRAY-of-string shape.
       Example: ``["USER_ID", "SESSION_ID"]``.
    2. Comma-separated ``str`` — legacy cursor surface.
       Example: ``'USER_ID, SESSION_ID'``.
    3. JSON-encoded ``str`` — the live deployed DT's ``cluster_by``
       column round-trips through Snowflake as a JSON-array literal
       and the Snowpark driver hands us a single-element list whose
       sole entry is the literal ``'["USER_ID"]'``.  Without the
       JSON-decode branch the recovered list reads
       ``['["USER_ID"]']`` and the structural hash never matches the
       local-compile ``['USER_ID']`` — Phase E live-verify symptom
       on offline tiled BFVs like ``MY_ADV_BFV_DECL``.

    Args:
        raw: The ``cluster_by`` cell from a list-FV row (``None``,
            ``list``, or ``str``).

    Returns:
        An ordered list of bare column identifiers, or ``None`` when
        no usable value is present.
    """
    if raw is None:
        return None
    # Phase E gap 1 — try JSON-decode FIRST before any comma-splitting.
    # ``feature_store._extract_feature_view_info`` emits this column as
    # ``json.dumps(self._extract_cluster_by_columns(...))`` (see
    # ``feature_store.py:7607``), so the canonical applied-state shape
    # is a JSON-array literal — either a bare string
    # ``'["USER_ID","TILE_START"]'`` or a single-element list wrapping
    # that string ``['["USER_ID","TILE_START"]']``.  Splitting the JSON
    # literal on commas BEFORE decoding shreds the quotes and brackets
    # into garbage (``'["USER_ID'`` / ``'TILE_START"]'``), which is the
    # symptom Phase E gap 1 originally observed for the multi-key
    # tiled-BFV default ``[entities..., TILE_START]``.  The decode
    # branch handles both wrappings (bare-string + single-element
    # list) and falls through to the legacy list / comma-split paths
    # when the input does not look like JSON.
    import json

    candidate: Any = raw
    if isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], str):
        stripped = candidate[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            candidate = stripped
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
                if isinstance(decoded, list):
                    resolved = [_resolve_cluster_column(x) for x in decoded if x is not None]
                    cols = [c for c in resolved if c is not None]
                    return cols or None
            except (ValueError, TypeError):
                # Fall through to the legacy comma-split path below.
                pass
    if isinstance(raw, list):
        resolved = [_resolve_cluster_column(c) for c in raw if c is not None]
    elif isinstance(raw, str) and raw.strip():
        resolved = [_resolve_cluster_column(c) for c in raw.split(",") if c.strip()]
    else:
        return None
    cols = [c for c in resolved if c is not None]
    return cols or None


def _build_datasources_by_table(specs: Sequence[Any]) -> dict[str, Any]:
    """Build a physical-table → logical-source-name lookup from local specs.

    **Legacy fallback only** (Phase B4).  Retained for FVs registered
    before the ``FV_SOURCE_REFS`` metadata row existed; engaged from
    :func:`fetch_applied_state` ONLY when a list-FV row carries no
    ``source_refs`` payload, accompanied by a once-per-FV
    ``logger.warning`` recommending re-apply for full source recovery.

    Walks every spec in *specs* and indexes every ``BatchSource``
    (skipping query- / query_file-backed sources, which have no
    physical table identity) by the uppercased unqualified last segment
    of its ``table`` attribute.

    Multi-match (two ``BatchSource`` YAMLs declaring the same physical
    table) is signalled by storing the list of conflicting logical
    names.

    Args:
        specs: Iterable of loaded specs (the ``SpecBatch.specs`` list
            from :func:`loader.load_from_project` is the typical
            input).  Non-``BatchSource`` entries are walked through
            without effect on the result.

    Returns:
        Dict keyed by the uppercased unqualified table ident.  Values
        are either ``str`` (unique match — the operator's authored
        logical name) or ``list[str]`` (multi-match collision marker).
    """
    lookup: dict[str, Any] = {}
    for spec in specs:
        if getattr(spec, "kind", None) != "BatchSource":
            continue
        table = getattr(spec, "table", None)
        if not isinstance(table, str) or not table:
            continue
        # Uppercased, unqualified last segment so the lookup keys
        # collide on identical objects regardless of YAML-side casing.
        key = table.split(".")[-1].strip().strip('"').upper()
        if not key:
            continue
        name = getattr(spec, "name", "") or ""
        if not name:
            continue
        existing = lookup.get(key)
        if existing is None:
            lookup[key] = name
            continue
        if isinstance(existing, str):
            if existing == name:
                continue
            lookup[key] = [existing, name]
            continue
        if isinstance(existing, list):
            if name not in existing:
                existing.append(name)
    return lookup


# Map between the imperative ``list_feature_views().kind`` discriminator
# values (``BATCH`` / ``STREAMING`` / ``REALTIME``) and the canonical
# spec-payload ``kind`` strings the planner branches on.  The full set
# is enumerated here so an unknown value falls through to ``None``
# (caller skips the row) instead of synthesising an invalid kind.
_FV_KIND_MAP: dict[str, str] = {
    "BATCH": "BatchFeatureView",
    "STREAMING": "StreamingFeatureView",
    "REALTIME": "RealtimeFeatureView",
}


def _build_offline_fv_object(
    fv_row: dict[str, Any],
    *,
    default_database: str,
    default_schema: str,
    datasources_by_table: Optional[dict[str, Any]] = None,
    fv_obj_provider: Any = None,
    dt_text_map: Any = None,
) -> Optional[AppliedObject]:
    """Build an ``AppliedObject`` for an offline-only FV from a list-FV row.

    Reconstructs the SPECIFICATION-equivalent ``spec_payload`` from
    the imperative ``list_feature_views()`` row (Phase 1 contract) +
    authoritative ``FvSourceRefsMetadata`` (via the row's new
    ``source_refs`` cell — plan section A1).  No DT-text parsing
    anywhere: source bindings, ``cluster_by``, ``refresh_mode``, and
    ``initialize`` all ride on metadata columns or on the rehydrated
    :class:`FeatureView` object returned by *fv_obj_provider*.

    The reconstruction satisfies the hash-equivalence invariant
    ``_full_spec_hash(applied_payload) == compute_local_spec_hash(local)``
    on a clean round-trip — the gating condition for ``NO_CHANGE``
    detection on the second plan after a successful offline-only
    apply.

    Args:
        fv_row: A row dict in the Phase-1 contract shape.  Must carry
            ``name``, ``version``, ``database_name``, ``schema_name``,
            ``kind``, ``entities``, ``physical_dt_name``, and
            ``target_lag``.  Optional new keys carry metadata-backed
            authoring fields: ``source_refs`` (plan A1),
            ``cluster_by`` (column), ``refresh_mode`` (column),
            ``spec_text`` (pre-enriched SPECIFICATION dict from
            :func:`decl.imperative_executor._serialize_batch_fv_spec`).
        default_database: Fallback database when the row omits one.
        default_schema: Fallback schema when the row omits one.
        datasources_by_table: Optional physical → logical source-name
            lookup, consumed ONLY when ``source_refs`` is absent (the
            legacy fallback path — plan section B4).  A
            ``logger.warning`` is emitted once per FV when this path
            engages, so the operator notices that source recovery is
            degraded and that re-apply will restore the metadata
            row.
        fv_obj_provider: Optional zero-argument callable returning a
            rehydrated :class:`FeatureView` (see
            :func:`decl.imperative_executor.fetch_feature_view_object`).
            Called lazily — only for BatchFVs and only when the
            metadata-driven enrichment needs the ``initialize`` field.
        dt_text_map: **Deprecated back-compat slot** (Plan B1).  Before
            Phase B the offline DT's ``CREATE … AS SELECT`` text was
            parsed for source / cluster_by / refresh_mode / initialize
            recovery; all four fields now ride on
            ``FvSourceRefsMetadata`` (sources) and the
            ``list_feature_views()`` row columns (cluster_by,
            refresh_mode, initialize via *fv_obj_provider*).  The
            argument is accepted (and silently discarded) so legacy
            callers in ``test_decoupled_refresh_contract`` and the
            offline-BFV bug-bash suites keep importing cleanly; a
            future cleanup will drop the slot.

    Returns:
        An ``AppliedObject(from_specification=True)`` carrying a
        BatchFV-shaped ``spec_payload``, or ``None`` when the row is
        missing required fields or the kind is unrecognised.
    """
    from snowflake.ml.feature_store.decl.compiler import parse_duration_to_seconds

    del dt_text_map  # Plan B1: DT-text parsing removed; slot kept for back-compat only.

    name = fv_row.get("name") or ""
    version = fv_row.get("version") or ""
    if not name or not version:
        logger.debug("Skipping offline FV row with missing name/version: %r", fv_row)
        return None

    kind_raw = (fv_row.get("kind") or "").upper()
    kind = _FV_KIND_MAP.get(kind_raw)
    if kind is None:
        logger.debug("Skipping offline FV row with unknown kind=%r: %s/%s", kind_raw, name, version)
        return None

    db = fv_row.get("database_name") or default_database
    schema_val = fv_row.get("schema_name") or default_schema
    physical_dt = fv_row.get("physical_dt_name") or f"{name}{_SEP}{version}"

    entities_raw = fv_row.get("entities") or []
    if isinstance(entities_raw, str):
        try:
            entities_list = json.loads(entities_raw)
        except (TypeError, ValueError):
            entities_list = []
    elif isinstance(entities_raw, list):
        entities_list = entities_raw
    else:
        entities_list = []
    entities = [str(e) for e in entities_list if e is not None]

    target_lag_sec: Optional[int] = None
    target_lag_raw: Optional[str] = None
    # The imperative ``list_feature_views()`` row reports the deployed DT
    # cadence under ``REFRESH_FREQ`` (the column ``SHOW DYNAMIC TABLES``
    # populates).  Older callers shipped ``target_lag`` directly on the
    # row.  Accept both, preferring an explicit ``target_lag`` when set
    # — the live shape always populates ``refresh_freq`` and leaves
    # ``target_lag`` empty, so without this fallback ``_build_offline_fv_object``
    # would emit a payload missing the DT cadence and the planner
    # would surface a phantom ``UPDATE_FV`` on every replan after a clean
    # offline-only apply.
    raw_value = fv_row.get("target_lag") or fv_row.get("refresh_freq")
    if raw_value:
        target_lag_raw = str(raw_value)
        try:
            target_lag_sec = parse_duration_to_seconds(raw_value)
        except Exception:  # noqa: BLE001 — defensive: malformed cadence string
            target_lag_sec = None

    inner_spec: dict[str, Any] = {
        "ordered_entity_column_names": entities,
        "sources": [],
        "features": [],
    }
    # Surface the deployed DT cadence in BOTH keys on the applied side
    # so downstream consumers see what they expect:
    # * ``target_lag_sec`` — the wire-form key the planner's full-spec
    #   hash already understands (``_full_spec_hash`` reads it).  Kept
    #   for hash-equivalence with deployed SPECIFICATION payloads.
    # * ``refresh_freq`` — the authoring-form key the exporter writes
    #   to YAML and ``_batch_fv_operational_drift`` compares against
    #   the local authoring spec.  Same name as the imperative
    #   ``FeatureView(refresh_freq=...)`` constructor kwarg.  Without
    #   this, an exported BFV YAML would lose its DT-refresh cadence on
    #   round-trip.  Only emitted for BFV — streaming and realtime kinds
    #   reject ``refresh_freq`` in the spec validator, so a recovered
    #   payload that carried the field would fail to load back.
    if target_lag_sec is not None:
        inner_spec["target_lag_sec"] = target_lag_sec
    if target_lag_raw is not None and kind == "BatchFeatureView":
        inner_spec["refresh_freq"] = target_lag_raw

    # When the imperative-side enrichment supplied a full FeatureViewSpec
    # dict (``spec_text``), use it verbatim — it carries the same shape
    # ``DESCRIBE … TYPE = SPECIFICATION`` would have returned for the
    # online subset (entities, sources, features, timestamp_field,
    # feature_granularity_sec, aggregation_secondary_keys, cluster_by,
    # refresh_mode, initialize, target_lag_sec).  This closes the hash
    # round-trip gap for offline-only BatchFVs that otherwise hash an
    # empty ``features`` list against a fully-populated local-compile.
    spec_text = fv_row.get("spec_text") if isinstance(fv_row, dict) else None
    if isinstance(spec_text, dict) and spec_text:
        spec_payload: dict[str, Any] = dict(spec_text)
        # Ensure ``offline_configs[0].table`` matches the physical DT name
        # so the metadata-driven recovery has a stable join key.  The
        # enrichment path already populates this from
        # ``_build_batch_fv_spec``; we set it defensively here for the
        # corner case where the metadata is absent.
        if not spec_payload.get("offline_configs"):
            spec_payload["offline_configs"] = [
                {
                    "store_type": "snowflake",
                    "table_type": "BatchTable",
                    "database": db,
                    "schema": schema_val,
                    "table": physical_dt,
                    "columns": [],
                }
            ]
    else:
        # Reconstruct ``offline_configs[0].table = <physical_dt_name>`` so
        # downstream consumers have a stable physical-DT pointer.
        # ``offline_configs`` is stripped before hashing by
        # ``_DERIVED_TOP_LEVEL_KEYS`` so it never contributes to
        # ``content_hash``.
        spec_payload = {
            "kind": kind,
            "metadata": {
                "database": db,
                "schema": schema_val,
                "name": name,
                "version": version,
                "spec_format_version": "1",
                "internal_data_version": "1",
                "client_version": "0.1.0",
            },
            "offline_configs": [
                {
                    "store_type": "snowflake",
                    "table_type": "BatchTable",
                    "database": db,
                    "schema": schema_val,
                    "table": physical_dt,
                    "columns": [],
                }
            ],
            "spec": inner_spec,
        }

    if kind == "BatchFeatureView":
        # B3 — authoritative source-binding recovery from FV_SOURCE_REFS
        # metadata; falls through to the legacy shim with a once-per-FV
        # warning when the row carries no ``source_refs`` payload.
        source_refs = fv_row.get("source_refs") if isinstance(fv_row, dict) else None
        injected = _inject_batch_fv_source_from_metadata(spec_payload, source_refs)
        if not injected:
            _engage_legacy_source_shim(
                spec_payload,
                fv_name=name,
                fv_version=version,
                datasources_by_table=datasources_by_table,
            )
        # B1 — cluster_by / refresh_mode / initialize from metadata.
        fv_obj = fv_obj_provider() if callable(fv_obj_provider) else None
        _inject_batch_fv_fields_from_list_row(spec_payload, fv_row, fv_obj=fv_obj)

    content_hash = _full_spec_hash(spec_payload)
    # Pass the authoritative row version as a top-level fallback so the
    # key carries the ``:VERSION`` segment even when the enriched
    # spec_text branch did not stamp ``metadata.version``.
    key = _build_spec_key(kind, {**spec_payload, "version": version})
    return AppliedObject(
        key=key,
        kind=kind,
        name=name,
        version=version,
        content_hash=content_hash,
        spec_payload=spec_payload,
        columns=[],
        from_specification=True,
    )


def _engage_legacy_source_shim(
    spec_payload: dict[str, Any],
    *,
    fv_name: str,
    fv_version: str,
    datasources_by_table: Optional[dict[str, Any]] = None,
) -> None:
    """Legacy fallback for FVs lacking ``FV_SOURCE_REFS`` metadata (Phase B4).

    Emits a once-per-FV ``logger.warning`` recommending re-apply for full
    source recovery, then leaves ``spec.sources`` empty.  The planner's
    ``_normalise_fv_sources_for_hash`` projects both sides to ``[]`` so
    a re-apply that does not change the source still resolves to
    ``NO_CHANGE`` — the warning is the operator-visible signal that the
    metadata row is missing and a re-apply will close the gap.

    The legacy ``_build_datasources_by_table`` lookup (a physical-table
    → logical-name map) is preserved on the import surface for any
    out-of-tree callers that already construct it, but is intentionally
    NOT consulted here: without DT-text parsing there is no recovered
    physical-table ident to look up.

    Args:
        spec_payload: The BatchFV spec dict; ``spec.sources`` is left
            empty when the legacy shim engages.
        fv_name: FV name (used in the warning message).
        fv_version: FV version (used in the warning message).
        datasources_by_table: Reserved for future use; accepted on the
            signature for back-compat but unused.
    """
    del datasources_by_table  # legacy back-compat slot, intentionally unused
    logger.warning(
        "decl.state: feature view %s/%s has no FV_SOURCE_REFS metadata; "
        "source bindings cannot be recovered authoritatively. Re-apply "
        "the FV (snow feature apply) to populate the metadata row and "
        "restore full source recovery on the next replan.",
        fv_name,
        fv_version,
    )


def _parse_oft_name(name: str) -> tuple[str, str]:
    """Extract ``(base_name, version)`` from an OFT name.

    The naming convention is ``<base_name>$<version>$ONLINE``.
    Falls back to splitting on the last ``$`` if the ``$ONLINE``
    suffix is absent (handles malformed or legacy names gracefully).

    Args:
        name: The raw OFT name string from Snowflake.

    Returns:
        Tuple of ``(base_name, version)``.
    """
    parts = name.split(_SEP)
    if len(parts) >= 3 and parts[-1] == _ONLINE_SUFFIX:
        # Standard: BASE$VERSION$ONLINE
        version = parts[-2]
        base = _SEP.join(parts[:-2])
        return base, version
    if len(parts) >= 2:
        # Fallback: BASE$VERSION
        version = parts[-1]
        base = _SEP.join(parts[:-1])
        return base, version
    # No separator — treat whole name as base, empty version
    return name, ""


def _extract_spec_from_oft(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract and parse the embedded JSON specification from an OFT row.

    Args:
        row: A single row from ``SHOW ONLINE FEATURE TABLES`` results.

    Returns:
        Parsed spec dict, or ``None`` if the specification column is missing
        or contains invalid JSON.
    """
    raw = row.get("specification", "")
    if not raw:
        return None
    try:
        return cast("dict[str, Any] | None", json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        logger.debug("Could not parse specification JSON for row: %s", row.get("name", "?"))
        return None


def _build_spec_key(kind: str, spec: dict[str, Any]) -> str:
    """Build a canonical key from spec metadata.

    Versioned kinds key as ``kind:DATABASE.SCHEMA:NAME:VERSION`` and
    unversioned kinds as ``kind:DATABASE.SCHEMA:NAME``.
    Mirrors :func:`snowflake.ml.feature_store.decl.invariants.spec_key` so
    the planner's batch-side keys (uppercased) and the applied-state side
    keys collide on identical objects.  Without this normalisation,
    full-directory mode treats every FV as both "new" *and* "orphaned"
    when the deployed name (e.g. ``user_profile_info``) and the local YAML
    name (e.g. ``USER_PROFILE_INFO``) differ only in case, which produces
    spurious ``CREATE_FV`` + ``DROP_FV`` ops on a clean round-trip.

    Args:
        kind: The object kind string.
        spec: The parsed spec dict (may contain a ``metadata`` sub-dict).

    Returns:
        Canonical key string with database, schema, and name uppercased.
    """
    metadata = spec.get("metadata", spec)
    db = (metadata.get("database", "") or spec.get("database", "") or "").upper()
    schema = (metadata.get("schema", "") or spec.get("schema", "") or "").upper()
    name = (metadata.get("name", "") or spec.get("name", "") or "").upper()
    qualifier = f"{db}.{schema}" if (db or schema) else ""
    key = f"{kind}:{qualifier}:{name}"
    # Mirror :func:`invariants.spec_key`: versioned kinds carry a trailing
    # ``:VERSION`` segment so two versions of one name occupy two distinct
    # applied-state slots instead of one shadowing the other.
    version = metadata.get("version", "") or spec.get("version", "")
    if kind in _VERSIONED_KINDS and version:
        key = f"{key}:{str(version).upper()}"
    return key


def _describe_feature_cols(desc_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract non-primary-key columns from ``DESCRIBE ONLINE FEATURE TABLE`` rows.

    Primary key columns are the entity/join-key columns.  The remaining columns
    are the feature output columns that form the structural fingerprint.

    Args:
        desc_rows: Rows from ``DESCRIBE ONLINE FEATURE TABLE``.

    Returns:
        List of ``{"name": col_name, "type": col_type}`` dicts for feature cols.
    """
    cols: list[dict[str, str]] = []
    for col in desc_rows:
        is_pk = False
        for pk_key in ("primary key", "PRIMARY KEY", "primary_key", "PRIMARY_KEY"):
            val = col.get(pk_key, "")
            if val and str(val).upper() in ("Y", "YES", "TRUE", "1"):
                is_pk = True
                break
        # Positional fallback: 5th value (index 4) == "Y" (mirrors exporter.py).
        if not is_pk:
            raw_vals = list(col.values())
            if len(raw_vals) >= 5 and raw_vals[4] == "Y":
                is_pk = True

        if not is_pk:
            col_name = col.get("name") or col.get("NAME") or ""
            col_type = col.get("type") or col.get("TYPE") or ""
            if col_name:
                cols.append({"name": col_name, "type": col_type})
    return cols


def parse_specification_rows(
    rows: Optional[list[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    """Parse rows returned by ``DESCRIBE ... TYPE = SPECIFICATION``.

    The new SQL primitive returns a single row carrying the original spec
    JSON used to create the Online Feature Table.  Different Snowflake
    cursor adapters expose the column under different names
    (``specification``, ``SPECIFICATION``, or simply the first string
    column), so this helper checks all of them and falls back to scanning
    for a single string value that parses as JSON.

    Args:
        rows: Raw rows (list of dicts) from the SQL execution.

    Returns:
        Parsed spec JSON dict, or ``None`` if no row contained a parseable
        spec.
    """
    if not rows:
        return None

    for row in rows:
        if not isinstance(row, dict):
            continue
        # Try the obvious column names first.
        candidates: list[Any] = []
        for col in ("specification", "SPECIFICATION", "Specification"):
            if col in row and row[col]:
                candidates.append(row[col])
        # Fall back to the first string-valued cell.
        if not candidates:
            for value in row.values():
                if isinstance(value, str) and value.strip():
                    candidates.append(value)
                    break
        for raw in candidates:
            if not isinstance(raw, str):
                continue
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _build_entity_object(
    row: dict[str, Any],
    default_db: str,
    default_schema: str,
) -> Optional[AppliedObject]:
    """Build an Entity ``AppliedObject`` from a ``SHOW TAGS`` row."""
    raw_name = row.get("name") or row.get("NAME") or ""
    if not raw_name:
        return None
    if not raw_name.startswith(_ENTITY_TAG_PREFIX):
        return None
    # Entity names match the join-key column.  Uppercase for round-trip
    # parity with :func:`invariants.spec_key` (which uppercases names so
    # casing differences between authoring YAML and Snowflake identifiers
    # collapse to a single canonical key).
    entity_name = raw_name[len(_ENTITY_TAG_PREFIX) :].upper()
    db = (row.get("database_name") or row.get("DATABASE_NAME") or default_db or "").upper()
    schema_val = (row.get("schema_name") or row.get("SCHEMA_NAME") or default_schema or "").upper()
    qualifier = f"{db}.{schema_val}" if (db or schema_val) else ""
    key = f"{ObjectKind.ENTITY}:{qualifier}:{entity_name}"

    join_keys: list[str] = []
    raw_allowed = row.get("allowed_values") or row.get("ALLOWED_VALUES") or row.get("allowed_values_list")
    if raw_allowed:
        if isinstance(raw_allowed, list):
            join_keys = [str(v) for v in raw_allowed]
        elif isinstance(raw_allowed, str):
            try:
                parsed = json.loads(raw_allowed)
                if isinstance(parsed, list):
                    join_keys = [str(v) for v in parsed]
            except (json.JSONDecodeError, TypeError):
                join_keys = [v.strip() for v in raw_allowed.strip("[]").split(",") if v.strip()]

    details: dict[str, Any] = {"join_keys": join_keys}
    comment = row.get("comment") or row.get("COMMENT") or ""
    if comment:
        details["comment"] = comment

    spec_payload: dict[str, Any] = {
        "kind": ObjectKind.ENTITY,
        "name": entity_name,
        "database": db,
        "schema": schema_val,
        "join_keys": [{"name": jk, "type": "StringType"} for jk in join_keys],
    }
    # Surface the deployed tag's COMMENT under ``description`` so the
    # Entity-aware branch of :func:`_structural_fingerprint` (which folds
    # description into the hash) sees the same value the local YAML's
    # ``description:`` produces.  Without this an unedited re-plan would
    # emit a phantom ``UPDATE_ENTITY`` after the fingerprint widen — see
    # ``plans/step11_update_entity_bug_*.plan.md``.
    if comment:
        spec_payload["description"] = comment

    # Match the planner's diff strategy: it hashes Entity/Source specs via
    # :func:`structural_fingerprint_hash`, so the applied side must use the
    # same so a re-applied YAML hashes identically and emits ``NO_CHANGE``.
    content_hash = structural_fingerprint_hash(spec_payload)

    return AppliedObject(
        key=key,
        kind=ObjectKind.ENTITY,
        name=entity_name,
        version=None,
        content_hash=content_hash,
        spec_payload=spec_payload,
        columns=[],
        from_specification=False,
        details=details,
    )


def _build_stream_source_object(
    row: dict[str, Any],
    default_db: str,
    default_schema: str,
) -> AppliedObject:
    """Build a Datasource ``AppliedObject`` from a runtime stream-source row.

    Companion to :func:`_datasource_objects_from_specs`: that function
    derives ``Datasource`` entries by unioning ``spec.sources[]`` across
    every recovered FV; this one reifies a single row produced by
    :func:`imperative_executor.fetch_stream_source_rows` (i.e. by
    ``FeatureStore.list_stream_sources()``).  The returned object's
    ``key`` and ``spec_payload`` shape match the FV-derived counterpart
    so that a downstream merge step in :func:`fetch_applied_state` can
    dedup the two paths by ``AppliedObject.key`` without any field-level
    coercion.  See ``plans/stream_source_contract.md`` §5a for the
    pinned contract.

    The ``"type"`` (REST vs other producer-protocol) key is deliberately
    omitted because the runtime metadata does not record it.  The diff
    helper ``invariants.compute_source_diff_kind`` strips ``"type"``
    from the local payload before the structural-fingerprint comparison
    so a clean round-trip still resolves to ``"no_change"`` (see
    contract §2 and §5a).

    Args:
        row: A single canonical row dict with keys ``name`` (str),
            ``schema`` (list of ``{"name", "type"}`` dicts), ``desc``
            (str), ``owner`` (str).  Missing ``desc`` is treated as
            empty string.
        default_db: Database to use for the canonical key (uppercased).
            Sourced from the active connection context by every other
            applied-state builder.
        default_schema: Schema to use for the canonical key
            (uppercased).

    Returns:
        ``AppliedObject(kind="Datasource")`` whose ``spec_payload``
        matches the shape :func:`_datasource_objects_from_specs`
        produces for the same logical source.
    """
    name_upper = (row.get("name") or "").upper()
    db_upper = (default_db or "").upper()
    schema_upper = (default_schema or "").upper()
    columns = row.get("schema") or []
    description = row.get("desc", "") or ""

    spec_payload: dict[str, Any] = {
        "kind": ObjectKind.DATASOURCE,
        "name": name_upper,
        "database": db_upper,
        "schema": schema_upper,
        "source_type": "Stream",
        "columns": columns,
        "description": description,
    }
    content_hash = structural_fingerprint_hash(spec_payload)
    key = f"{ObjectKind.DATASOURCE}:{db_upper}.{schema_upper}:{name_upper}"
    details: dict[str, Any] = {
        "source_type": "Stream",
        "column_count": len(columns) if isinstance(columns, list) else 0,
    }
    return AppliedObject(
        key=key,
        kind=ObjectKind.DATASOURCE,
        name=name_upper,
        version=None,
        content_hash=content_hash,
        spec_payload=spec_payload,
        columns=[],
        from_specification=True,
        details=details,
    )


def _datasource_objects_from_specs(
    fv_specs: list[dict[str, Any]],
    default_db: str,
    default_schema: str,
    *,
    known_fv_names: Optional[set[str]] = None,
) -> list[AppliedObject]:
    """Derive Datasource AppliedObjects by unioning ``spec.sources[]``.

    Datasources are virtual: they are not separately registered in
    Snowflake.  The runtime contract for them lives inside each FV's
    ``spec.sources[]`` block.  This function deduplicates by name and emits
    one ``AppliedObject(kind="Datasource")`` per unique source.

    FG-backed BFVs emit a ``sources[0].name`` that equals the FV name itself
    (an internal executor artifact, not an operator-authored source).  Any
    source whose canonical name appears in *known_fv_names* is skipped so
    that the planner does not schedule a spurious ``DROP_SOURCE`` for it on
    every subsequent plan run.

    Args:
        fv_specs: Parsed feature view spec JSON documents (the return
            value of :func:`parse_specification_rows` for each OFT).
        default_db: Database to use when a source's metadata is missing.
        default_schema: Schema to use when a source's metadata is missing.
        known_fv_names: Optional set of uppercased FeatureView names already
            recovered into the applied state.  Source entries whose canonical
            name is in this set are treated as internal executor artifacts
            (FG-backed BFV pattern) and are not promoted to Datasource
            AppliedObjects.  ``None`` disables the filter (legacy behaviour).

    Returns:
        List of ``AppliedObject`` instances, one per unique datasource.
    """
    by_name: dict[str, AppliedObject] = {}
    for fv_spec in fv_specs:
        if not isinstance(fv_spec, dict):
            continue
        inner = fv_spec.get("spec") if isinstance(fv_spec.get("spec"), dict) else fv_spec
        sources = inner.get("sources", []) if isinstance(inner, dict) else []
        if not isinstance(sources, list):
            continue
        _md = fv_spec.get("metadata")
        metadata = _md if isinstance(_md, dict) else {}
        db = (metadata.get("database") or default_db or "").upper()
        schema_val = (metadata.get("schema") or default_schema or "").upper()
        qualifier = f"{db}.{schema_val}" if (db or schema_val) else ""

        for src in sources:
            if not isinstance(src, dict):
                continue
            src_name = src.get("name", "")
            if not src_name:
                continue
            # Uppercase for round-trip parity with :func:`invariants.spec_key`
            # — the same source name in two FVs (one upper, one lower)
            # canonicalises to one AppliedObject and one key.
            canonical_name = src_name.upper()
            # FG-backed BFVs set sources[0].name == fv.name (executor
            # artifact).  Skip these so the planner does not schedule a
            # spurious DROP_SOURCE for an operator-absent source.
            if known_fv_names and canonical_name in known_fv_names:
                continue
            dedupe_key = canonical_name
            if dedupe_key in by_name:
                continue
            source_type = src.get("source_type") or src.get("type") or ""
            columns = src.get("columns", [])
            details: dict[str, Any] = {"source_type": source_type}
            if isinstance(columns, list):
                details["column_count"] = len(columns)

            # ``table`` / ``query`` are preserved on the Datasource
            # AppliedObject so a downstream ``BatchSource`` YAML with a
            # matching ``table:`` / ``query:`` round-trips to
            # ``NO_CHANGE``.  Phase B3: BatchFV source bindings now ride
            # on ``FvSourceRefsMetadata`` (via the list-FV row's
            # ``source_refs`` cell — plan section A1), so the values
            # below come from the metadata-injected ``spec.sources[]``
            # rather than from DT-text parsing.  Streaming sources
            # preserve them on the wire spec already.
            table = src.get("table") if isinstance(src.get("table"), str) else ""
            query = src.get("query") if isinstance(src.get("query"), str) else ""

            spec_payload: dict[str, Any] = {
                "kind": ObjectKind.DATASOURCE,
                "name": canonical_name,
                "database": db,
                "schema": schema_val,
                "source_type": source_type,
                "columns": columns,
            }
            if table:
                spec_payload["table"] = table
            if query:
                spec_payload["query"] = query
            # Match the planner's diff strategy: structural fingerprint, not
            # raw payload hash, so a re-applied YAML matches and emits
            # ``NO_CHANGE`` instead of a spurious recreate.
            content_hash = structural_fingerprint_hash(spec_payload)
            key = f"{ObjectKind.DATASOURCE}:{qualifier}:{canonical_name}"

            by_name[dedupe_key] = AppliedObject(
                key=key,
                kind=ObjectKind.DATASOURCE,
                name=canonical_name,
                version=None,
                content_hash=content_hash,
                spec_payload=spec_payload,
                columns=[],
                from_specification=True,
                details=details,
            )
    return list(by_name.values())


def fetch_applied_state(
    raw_show_results: list[dict[str, Any]],
    raw_table_results: Optional[list[dict[str, Any]]] = None,
    describe_map: Optional[dict[str, list[dict[str, Any]]]] = None,
    *,
    specification_map: Optional[dict[str, dict[str, Any]]] = None,
    entity_rows: Optional[list[dict[str, Any]]] = None,
    dt_text_map: Optional[dict[str, str]] = None,
    feature_view_rows: Optional[list[dict[str, Any]]] = None,
    feature_group_rows: Optional[list[dict[str, Any]]] = None,
    stream_source_rows: Optional[list[dict[str, Any]]] = None,
    datasources_by_table: Optional[dict[str, Any]] = None,
    default_database: str = "",
    default_schema: str = "",
) -> AppliedState:
    """Parse raw Snowflake query results into an ``AppliedState`` snapshot.

    Steps:

    1. Parse ``SHOW ONLINE FEATURE TABLES`` rows → extract name/version from
       the ``$version$ONLINE`` naming convention.
    2. If ``specification_map[name]`` is present, use the full spec JSON as
       the canonical ``spec_payload`` and mark
       ``from_specification=True``.  This enables full-spec diffs.
    3. Otherwise try to extract embedded spec JSON from the
       ``specification`` column on the SHOW row (legacy path), then fall
       back to reconstructing state from ``describe_map`` columns
       (structural-fingerprint path).
    4. If ``entity_rows`` is provided, build one ``AppliedObject`` per
       entity tag.
    5. **BatchFV source-binding recovery** (Phase B3): when a list-FV
       row carries authoritative ``FV_SOURCE_REFS`` metadata (via the
       ``source_refs`` column added in plan section A1), populate
       ``spec.sources[]`` directly from it — operator-authored logical
       names, columns, table / query bindings, tiled-vs-non-tiled
       irrelevant.  Legacy FVs that pre-date the metadata row engage
       the ``_build_datasources_by_table`` fallback with a once-per-FV
       ``logger.warning`` recommending re-apply for full source recovery.
    6. Derive ``Datasource`` AppliedObjects from a runtime-authoritative
       merge of two sources (see ``plans/stream_source_contract.md``
       §5b): first, one ``Datasource`` per row in ``stream_source_rows``
       via :func:`_build_stream_source_object` (registered stream
       sources from :func:`FeatureStore.list_stream_sources`); then
       the FV-derived entries from
       :func:`_datasource_objects_from_specs` are inserted only when
       their key is not already taken (runtime wins on collision).
       ``stream_source_rows=None`` falls back to the FV-derived-only
       behaviour so every existing call site that does not yet
       thread the new kwarg works unchanged.

    Args:
        raw_show_results: Rows from ``SHOW ONLINE FEATURE TABLES``.
        raw_table_results: Rows from ``SHOW TABLES`` for offline table
            existence checks (currently unused in v0, reserved for future
            use).
        describe_map: Optional mapping of OFT name → ``DESCRIBE ONLINE
            FEATURE TABLE`` rows (column-only).  Used as the fallback when
            a specification is unavailable.
        specification_map: Optional mapping of OFT name → parsed spec JSON
            from ``DESCRIBE ... TYPE = SPECIFICATION``.  When present, this
            is the authoritative source for the FV's spec_payload.
        entity_rows: Optional rows from
            ``SHOW TAGS LIKE 'SNOWML_FEATURE_STORE_ENTITY_%'``.  Each row
            becomes an ``AppliedObject(kind="Entity")``.
        dt_text_map: **Back-compat slot only** — Phase B1 removed the
            DT-text recovery path.  Existing CLI manager call sites
            still thread this kwarg, but it is no longer consumed.
            Source bindings now come from ``FvSourceRefsMetadata``
            (surfaced via the list-FV row's ``source_refs`` cell).
        feature_view_rows: Optional list of FV rows in the Phase-1
            contract shape (see
            :func:`imperative_executor.fetch_feature_view_rows` and
            ``plans/offline_bfv_state_fix_b9da0006.plan.md`` Section 7).
            Surfaces offline-only ``BatchFeatureView``s that
            ``SHOW ONLINE FEATURE TABLES`` cannot enumerate, AND
            provides the authoritative ``source_refs`` / ``cluster_by``
            / ``refresh_mode`` metadata for OFT-backed BatchFVs whose
            ``DESCRIBE TYPE = SPECIFICATION`` payload does not include
            them.  When the same FV appears in both
            ``raw_show_results`` and ``feature_view_rows``, the
            OFT-derived ``spec_payload`` is authoritative and the
            ``feature_view_rows`` entry contributes only the
            metadata-enrichment overlay.
        feature_group_rows: Optional list of FG rows in the contract
            shape returned by
            :func:`imperative_executor.fetch_feature_group_rows`.
            Each row becomes an ``AppliedObject(kind="FeatureGroup")``.
            ``None`` and ``[]`` are equivalent (no FG entries
            contributed to the snapshot).  The kind is fully
            imperative — there is no ``SHOW FEATURE GROUPS`` SQL —
            so without this kwarg an FG that exists in the runtime
            never makes it into the applied-state snapshot and the
            planner re-emits a spurious ``CREATE_FG`` on every plan.
        stream_source_rows: Optional list of registered stream-source
            rows in the canonical shape returned by
            :func:`imperative_executor.fetch_stream_source_rows`
            (``{"name", "schema", "desc", "owner"}``).  Each row is
            reified as one ``AppliedObject(kind="Datasource")`` via
            :func:`_build_stream_source_object` and merged with the
            FV-derived ``Datasource`` entries from
            :func:`_datasource_objects_from_specs`.  The merge is
            runtime-authoritative: when both paths surface the same
            ``Datasource:DB.SCHEMA:NAME`` key, the runtime row wins
            and the FV-derived shape is skipped.  ``None`` (the
            default) falls back to the FV-derived-only behaviour for
            back-compat with every existing call site.  ``None`` and
            ``[]`` are equivalent — both contribute zero runtime
            ``Datasource`` entries.
        datasources_by_table: Optional mapping of uppercased
            unqualified physical table ident → operator-authored
            logical ``BatchSource.name``. Built by
            :func:`_build_datasources_by_table` from the local
            ``sources/datasources/`` tree. When provided, the BatchFV
            source-binding recovery (step 5) prefers the local logical
            name over the recovered physical table identifier — closing
            the export round-trip where operators saw ``MISSING_SOURCE``
            on every re-plan after ``snow feature init`` rewrote
            ``sources[0].name`` to the underlying table name.  ``None``
            (the default) falls back to the legacy table-as-name
            behaviour, preserving cold-start ``init`` against a
            never-seen schema.
        default_database: Database to use when a row does not include one.
        default_schema: Schema to use when a row does not include one.

    Returns:
        AppliedState snapshot with all deployed objects (FVs + Entities +
        derived Datasources).
    """
    # ``dt_text_map`` is accepted on the signature for back-compat with
    # existing CLI manager call sites but is no longer consumed — Phase
    # B1 removed all DT-text parsing.  Discard explicitly to make the
    # no-op contract visible in code review.
    del dt_text_map
    objects: dict[str, AppliedObject] = {}
    spec_payloads_for_datasources: list[dict[str, Any]] = []

    # Build a (name_upper, version_str) → list-FV row index so the OFT
    # loop below can look up the authoritative ``source_refs`` /
    # ``cluster_by`` / ``refresh_mode`` / ``fv_obj_provider`` for an
    # OFT-backed BatchFV.
    fv_row_by_name_version: dict[tuple[str, str], dict[str, Any]] = {}
    if feature_view_rows:
        for fv_row in feature_view_rows:
            fv_name = fv_row.get("name") or ""
            fv_version = fv_row.get("version") or ""
            if fv_name and fv_version:
                fv_row_by_name_version[(str(fv_name).upper(), str(fv_version))] = fv_row

    for row in raw_show_results:
        oft_name = row.get("name", "")
        if not oft_name:
            continue

        base_name, version = _parse_oft_name(oft_name)

        # Path 1: Full spec JSON via DESCRIBE TYPE = SPECIFICATION.
        spec = None
        from_spec = False
        if specification_map is not None:
            spec = specification_map.get(oft_name)
            if spec is not None:
                from_spec = True

        # Path 2: Legacy embedded specification column on the SHOW row.
        if spec is None:
            spec = _extract_spec_from_oft(row)

        if spec is not None:
            kind = spec.get("kind", "StreamingFeatureView")
            metadata = spec.get("metadata", {}) if isinstance(spec.get("metadata"), dict) else {}
            db = metadata.get("database", "") or row.get("database_name", "") or default_database
            schema_val = metadata.get("schema", "") or row.get("schema_name", "") or default_schema

            spec_payload: dict[str, Any]
            if from_spec:
                spec_payload = dict(spec)
                if "metadata" not in spec_payload:
                    spec_payload["metadata"] = {
                        "database": db,
                        "schema": schema_val,
                        "name": base_name,
                        "version": version,
                    }
                # BatchFV source-binding recovery (Phase B3) — read
                # authoritative ``FV_SOURCE_REFS`` metadata from the
                # matching ``feature_view_rows`` entry and inject it
                # BEFORE hashing so the resulting ``content_hash``
                # reflects the recovered ``sources[]``.
                if kind == "BatchFeatureView":
                    matched_row = fv_row_by_name_version.get((str(base_name).upper(), str(version)))
                    if matched_row is not None:
                        source_refs = matched_row.get("source_refs")
                        injected = _inject_batch_fv_source_from_metadata(spec_payload, source_refs)
                        if not injected:
                            _engage_legacy_source_shim(
                                spec_payload,
                                fv_name=base_name,
                                fv_version=version,
                                datasources_by_table=datasources_by_table,
                            )
                        _inject_batch_fv_fields_from_list_row(spec_payload, matched_row, fv_obj=None)
                        # Plumb the deployed DT cadence onto the BFV
                        # ``spec.refresh_freq`` so the planner's
                        # ``_refresh_freq_drifted`` helper has an
                        # authoring-form value to compare against.
                        # No-op for streaming / realtime kinds (the
                        # helper short-circuits internally).
                        _inject_fv_refresh_freq_from_list_row(spec_payload, matched_row)
                elif kind == "StreamingFeatureView":
                    # A tiled StreamingFeatureView schedules an offline
                    # tile Dynamic Table whose ``TARGET_LAG`` is
                    # ``refresh_freq``.  Recover that cadence onto
                    # ``spec.refresh_freq`` so a genuine cadence edit is
                    # detected by ``_refresh_freq_drifted`` (which, for
                    # streaming, refuses to read the OFT ``target_lag_sec=0``
                    # sentinel).  The helper self-guards: it is a no-op for
                    # a non-tiled streaming FV (zero-lag VIEW, no DT).
                    matched_row = fv_row_by_name_version.get((str(base_name).upper(), str(version)))
                    if matched_row is not None:
                        _inject_fv_refresh_freq_from_list_row(spec_payload, matched_row)
                content_hash = _full_spec_hash(spec_payload)
                spec_payloads_for_datasources.append(spec_payload)
            else:
                # Legacy embedded-specification path → flatten for fingerprint.
                spec_payload = {
                    "kind": kind,
                    "name": base_name,
                    "version": version,
                    "database": db,
                    "schema": schema_val,
                }
                inner_spec = spec.get("spec", {})
                if isinstance(inner_spec, dict):
                    spec_payload.update(inner_spec)
                content_hash = structural_fingerprint_hash(spec_payload)

            key = _build_spec_key(
                kind,
                {
                    "database": db,
                    "schema": schema_val,
                    "name": base_name,
                    "version": version,
                },
            )
            objects[key] = AppliedObject(
                key=key,
                kind=kind,
                name=base_name,
                version=version,
                content_hash=content_hash,
                spec_payload=spec_payload,
                columns=[],
                from_specification=from_spec,
            )
            continue

        # Path 3: DESCRIBE-only fallback (structural fingerprint).
        if describe_map is None:
            logger.debug("Skipping OFT row with no parseable specification: %s", oft_name)
            continue
        desc_rows = describe_map.get(oft_name)
        if not desc_rows:
            logger.debug(
                "Skipping OFT row: no specification and no describe_map entry for %s",
                oft_name,
            )
            continue

        db = row.get("database_name", "") or default_database
        schema_val = row.get("schema_name", "") or default_schema

        feature_cols = _describe_feature_cols(desc_rows)
        fp_spec: dict[str, Any] = {
            "name": base_name,
            "version": version,
            "features": [{"output_column": c} for c in feature_cols],
        }
        content_hash = structural_fingerprint_hash(fp_spec)
        kind = "StreamingFeatureView"
        key = _build_spec_key(
            kind,
            {"database": db, "schema": schema_val, "name": base_name, "version": version},
        )

        objects[key] = AppliedObject(
            key=key,
            kind=kind,
            name=base_name,
            version=version,
            content_hash=content_hash,
            spec_payload=fp_spec,
            columns=[],
            from_specification=False,
        )

    # Offline FV AppliedObjects from FeatureStore.list_feature_views().
    #
    # Surfaces FVs that have no Online Feature Table — primarily
    # offline-only ``BatchFeatureView``s (``online: false``).  The OFT
    # loop above already populated ``objects`` for online FVs; entries
    # here are skipped on key-collision so the OFT-derived
    # ``spec_payload`` (the authoritative DESCRIBE-TYPE-SPECIFICATION
    # JSON) always wins.
    if feature_view_rows:
        for fv_row in feature_view_rows:
            offline_obj = _build_offline_fv_object(
                fv_row,
                default_database=default_database,
                default_schema=default_schema,
                datasources_by_table=datasources_by_table,
            )
            if offline_obj is None:
                continue
            if offline_obj.key in objects:
                continue
            objects[offline_obj.key] = offline_obj
            spec_payloads_for_datasources.append(offline_obj.spec_payload)

    # Entity AppliedObjects from SHOW TAGS.
    if entity_rows:
        for entity_row in entity_rows:
            ent_obj = _build_entity_object(entity_row, default_database, default_schema)
            if ent_obj is not None:
                objects[ent_obj.key] = ent_obj

    # FeatureGroup AppliedObjects from FeatureStore.list_feature_groups().
    #
    # Each row is reified as one ``AppliedObject(kind="FeatureGroup")``
    # whose ``content_hash`` matches the planner's ``fg_content_hash``
    # by construction (decl-shape spec_payload built from the imperative
    # row, with FvSourceRef ``fv_name`` / ``fv_version`` translated to the
    # declarative ``name`` / ``version`` keys).  ``from_specification``
    # is False — FG state lives in the metadata table, not in an OFT
    # specification.
    if feature_group_rows:
        for fg_row in feature_group_rows:
            fg_obj = _build_feature_group_object(fg_row, default_database, default_schema)
            if fg_obj is not None:
                objects[fg_obj.key] = fg_obj

    # Datasource AppliedObjects — runtime-authoritative merge per
    # ``plans/stream_source_contract.md`` §5b.
    #
    # Step 1: insert one ``AppliedObject`` per registered stream-source
    # row (``FeatureStore.list_stream_sources()``) FIRST so its
    # ``content_hash`` / ``spec_payload`` wins on key collision.  The
    # runtime row carries metadata the FV-derived path cannot recover
    # (notably the registered ``description``), so it is the source
    # of truth whenever both sides surface the same Datasource key.
    for runtime_row in stream_source_rows or []:
        runtime_obj = _build_stream_source_object(
            runtime_row,
            default_database,
            default_schema,
        )
        objects[runtime_obj.key] = runtime_obj

    # Step 2: walk the FV-derived ``Datasource`` entries unioned from
    # ``spec.sources[]`` across every recovered FV.  Skip on key
    # collision so the runtime entry remains authoritative; insert
    # otherwise so sources that no FV references still emerge (and
    # so do FV-only sources that have no registered runtime row,
    # e.g. virtual ``BatchSource`` entries derived from BatchFV DT
    # text).
    #
    # Pass the set of recovered FV names so FG-backed BFV source
    # entries (whose name == fv.name) are suppressed — they are
    # internal executor artifacts, not operator-authored sources.
    _FV_KINDS = {"BatchFeatureView", "StreamingFeatureView", "RealtimeFeatureView"}
    known_fv_names = {obj.name.upper() for obj in objects.values() if obj.kind in _FV_KINDS}
    for ds_obj in _datasource_objects_from_specs(
        spec_payloads_for_datasources, default_database, default_schema, known_fv_names=known_fv_names
    ):
        if ds_obj.key in objects:
            continue
        objects[ds_obj.key] = ds_obj

    return AppliedState(objects=objects)


def _build_feature_group_object(
    fg_row: dict[str, Any],
    default_database: str,
    default_schema: str,
) -> Optional[AppliedObject]:
    """Reify one FG row from :func:`imperative_executor.fetch_feature_group_rows`
    as an ``AppliedObject(kind="FeatureGroup")``.

    The applied-state spec_payload is built in **declarative shape**
    (``feature_views[].name`` / ``.version`` rather than the imperative
    ``fv_name`` / ``fv_version``) so the planner's hash basis matches a
    local YAML by construction.  Both ``name`` / ``version`` and the
    optional ``slice_columns`` / ``alias`` are translated; ``alias = ""``
    survives as the literal empty string (semantically: "no prefix").

    Args:
        fg_row: A row dict in the shape produced by
            :func:`imperative_executor.fetch_feature_group_rows`.
        default_database: Database to use when the row does not include one.
        default_schema: Schema to use when the row does not include one.

    Returns:
        An :class:`AppliedObject` with ``kind="FeatureGroup"`` and a
        content_hash matching :func:`invariants.fg_content_hash`, or
        ``None`` when ``fg_row`` is missing a name / version.
    """
    # Lazy import to avoid widening the import surface of this module
    # (matches the rest of the per-kind builder helpers in this file).
    from snowflake.ml.feature_store.decl.invariants import fg_content_hash

    name = fg_row.get("name") or ""
    version = fg_row.get("version") or ""
    if not name or not version:
        return None

    db = fg_row.get("database_name") or default_database
    schema = fg_row.get("schema_name") or default_schema

    decl_sources: list[dict[str, Any]] = []
    for src in fg_row.get("sources") or []:
        if not isinstance(src, dict):
            continue
        fv_name = src.get("fv_name") or src.get("name") or ""
        fv_version = src.get("fv_version") or src.get("version") or ""
        if not fv_name or not fv_version:
            continue
        decl_src: dict[str, Any] = {"name": fv_name, "version": fv_version}
        if "slice_columns" in src and src["slice_columns"] is not None:
            decl_src["slice_columns"] = list(src["slice_columns"])
        if "alias" in src and src["alias"] is not None:
            # Preserve alias="" as the literal empty string.
            decl_src["alias"] = src["alias"]
        decl_sources.append(decl_src)

    spec_payload: dict[str, Any] = {
        "kind": "FeatureGroup",
        "name": name,
        "version": version,
        "database": db,
        "schema": schema,
        "desc": fg_row.get("desc") or "",
        "auto_prefix": bool(fg_row.get("auto_prefix", True)),
        "feature_views": decl_sources,
    }

    content_hash = fg_content_hash(spec_payload)
    key = _build_spec_key(
        "FeatureGroup",
        {"database": db, "schema": schema, "name": name, "version": version},
    )

    return AppliedObject(
        key=key,
        kind="FeatureGroup",
        name=name,
        version=version,
        content_hash=content_hash,
        spec_payload=spec_payload,
        columns=[],
        from_specification=False,
        details={},
    )
