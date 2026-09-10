"""Compiler: authoring-format spec dict → imperative FeatureViewSpec format dict.

Transforms a loaded/compiled spec dict (human-friendly YAML authoring format,
with UDF source already inlined by ``compiler.py``) into a dict matching the
``FeatureViewSpec.to_dict()`` output structure expected by
``CREATE ONLINE FEATURE TABLE ... FROM SPECIFICATION``.

All functions are self-contained — no imports from snowflake.ml.feature_store.spec,
snowflake.snowpark, or snowflake.connector.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from snowflake.ml.feature_store.decl.compiler import parse_duration_to_seconds
from snowflake.ml.feature_store.decl.types import AppliedState, ObjectKind

_CLIENT_VERSION = "0.1.0"

# ``feature_aggregation_method: continuous`` streaming FVs are allowed to
# omit ``feature_granularity`` in the authoring YAML.  The Snowflake
# runtime always post-defaults the deployed SPECIFICATION JSON to 60
# seconds for this case (mirrors snowml-core's
# ``feature_view._DEFAULT_CONTINUOUS_FEATURE_GRANULARITY = "1m"`` —
# see :func:`snowflake.ml.feature_store.feature_view.FeatureView._resolve_tiled_config`),
# so the local compiler must stamp the same value or
# :func:`~snowflake.ml.feature_store.decl.invariants._full_spec_hash`
# diverges between authoring YAML and deployed runtime, surfacing as a
# spurious ``RECREATE_FV`` on every ``snow feature plan`` against an
# unchanged remote state.  Scoped strictly to ``StreamingFeatureView``:
# ``BatchFeatureView`` rejects ``feature_aggregation_method`` entirely
# (the decl-side ``BATCH_FV_TILING_GRANULARITY`` invariant is the
# surfacing layer), and tiles streaming FVs raise from snowml-core when
# the granularity is missing — that imperative error is the desired UX.
_DEFAULT_CONTINUOUS_FEATURE_GRANULARITY_SEC = 60


def sanitize_json_for_dollar_quoting(payload: str) -> str:
    """Replace ``$$`` with ``$\\u0024`` to make JSON safe for SQL ``$$`` quoting.

    ``\\u0024`` is the JSON unicode escape for ``$``; every conformant JSON
    parser decodes it back to ``$`` automatically.

    Args:
        payload: A JSON-encoded string.

    Returns:
        The same string with every ``$$`` replaced by ``$\\u0024``.
    """
    return payload.replace("$$", "$\\u0024")


def _resolve_duration(obj: dict[str, Any], key: str) -> int | None:
    """Read ``key`` or ``key_sec`` from *obj* and return integer seconds.

    If ``key`` is present (string/int) it is converted via
    :func:`~snowflake.ml.feature_store.decl.compiler.parse_duration_to_seconds`.
    If ``key_sec`` is already present (post-normalize_durations pass) it is
    returned directly.  Returns ``None`` if neither key is found.

    Args:
        obj: A dict that may contain ``key`` or ``{key}_sec``.
        key: The base key name (without ``_sec`` suffix).

    Returns:
        Integer seconds or ``None``.
    """
    if key in obj:
        return parse_duration_to_seconds(obj[key])
    sec_key = f"{key}_sec"
    if sec_key in obj:
        val = obj[sec_key]
        return int(val) if val is not None else None
    return None


def _compile_features(raw_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert feature dicts, normalizing ``window``/``offset`` to ``*_sec`` ints.

    Args:
        raw_features: List of feature dicts in authoring or normalized format.

    Returns:
        New list of feature dicts with ``window_sec`` / ``offset_sec`` integers
        and the original string keys removed.
    """
    result = []
    for feat in raw_features:
        compiled: dict[str, Any] = {}
        for k, v in feat.items():
            if k in ("window", "offset"):
                compiled[f"{k}_sec"] = parse_duration_to_seconds(v)
            elif k in ("window_sec", "offset_sec"):
                compiled[k] = int(v) if v is not None else v
            else:
                compiled[k] = v
        result.append(compiled)
    return result


def _compile_udf(udf: dict[str, Any]) -> dict[str, Any]:
    """Build the ``spec.udf`` sub-dict from an authoring-format UDF dict.

    The live ``DESCRIBE ONLINE FEATURE TABLE ... TYPE = SPECIFICATION``
    payload uses the authoring-shape keys ``name`` and ``engine`` directly
    (verified against captured fixtures in
    ``tests/golden_specs/USER_CLICK_STATS.json``).  An earlier symmetric
    rename to ``function_name`` / ``language`` inverted the round-trip
    invariant — the compiled local hash never matched the deployed hash —
    so we now keep ``name`` and ``engine`` as-is.  Only ``source`` is
    renamed to ``function_definition`` (the legacy alias the live payload
    uses for the inlined Python source).  ``file`` is dropped because the
    YAML loader has already inlined it as ``function_definition``.

    Args:
        udf: Authoring-format UDF dict.

    Returns:
        Compiled UDF dict matching the SPECIFICATION JSON shape.
    """
    rename = {"source": "function_definition"}
    compiled: dict[str, Any] = {}
    for k, v in udf.items():
        if k == "file":
            continue  # already inlined; drop the path
        compiled[rename.get(k, k)] = v
    return compiled


def _normalize_sources_for_imperative_json(sources: list[dict[str, Any]], *, fv_kind: str) -> list[dict[str, Any]]:
    """Normalise authoring ``sources[]`` entries toward imperative SPECIFICATION JSON.

    Batch feature views require ``source_type: Batch`` (the value of
    :class:`~snowflake.ml.feature_store.spec.enums.SourceType.BATCH`) on
    each ``spec.sources[]`` entry so the compiled payload matches what
    Snowflake returns from ``DESCRIBE … TYPE = SPECIFICATION``.

    Args:
        sources: Authoring ``sources`` list from the feature view spec.
        fv_kind: Feature view kind string (e.g. ``BatchFeatureView``).

    Returns:
        A shallow-copied list with ``source_type`` normalised where applicable.
    """
    out: list[dict[str, Any]] = []
    for raw in sources:
        entry = dict(raw)
        if fv_kind == "BatchFeatureView":
            st = (entry.get("source_type") or "").strip()
            if st in ("", "BatchSource", "OfflineTable"):
                entry["source_type"] = "Batch"
        out.append(entry)
    return out


def _entity_join_keys_from_payload(payload: Any) -> list[str]:
    """Extract ordered join-key column names from an Entity spec dict/payload.

    Accepts the authoring/applied Entity shape where ``join_keys`` is a list
    of ``{"name": ..., "type": ...}`` dicts (or bare column-name strings) and
    returns the ordered column names.  Returns an empty list for any shape it
    does not recognise.

    Args:
        payload: The Entity spec dict (authoring or applied ``spec_payload``).

    Returns:
        The ordered join-key column names, or ``[]`` when none are found.
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("join_keys")
    if not isinstance(raw, list):
        return []
    cols: list[str] = []
    for jk in raw:
        if isinstance(jk, dict) and jk.get("name"):
            cols.append(str(jk["name"]))
        elif isinstance(jk, str) and jk:
            cols.append(jk)
    return cols


def build_entity_join_key_map(
    spec_dicts: list[dict[str, Any]],
    applied_state: Optional[AppliedState] = None,
) -> dict[str, list[str]]:
    """Build an entity-name → ordered join-key-columns map.

    The wire field ``ordered_entity_column_names`` must carry entity
    **join-key columns**, mirroring core
    ``FeatureStore._build_batch_feature_view_spec`` (which iterates
    ``entity.join_keys``).  This map lets :func:`compile_to_spec` translate an
    FV's authored entity **names** into those columns.

    Sources, in precedence order (**applied wins for already-deployed
    entities** — join keys are immutable, so the FV must hash against the keys
    the executor will actually reproduce, not an unappliable YAML edit; a
    join-key edit is rejected separately by ``ENTITY_JOIN_KEY_IMMUTABLE``):

    1. ``Entity`` objects in ``applied_state`` (the deployed key set; also
       covers incremental single-file plans that omit the Entity spec).
    2. ``Entity`` spec dicts in ``spec_dicts`` (the current batch) — fills in
       new entities not yet deployed.

    Both the authored casing and an upper-cased alias are recorded so a
    case-variant reference still resolves.

    Args:
        spec_dicts: The batch spec dicts (read-only; not mutated).
        applied_state: Optional applied-state snapshot; the authoritative
            source of entity → join-key mappings for deployed entities.

    Returns:
        A mapping from entity name (and its upper-cased alias) to the ordered
        join-key column names.
    """
    entity_map: dict[str, list[str]] = {}

    def _record(name: Any, cols: list[str]) -> None:
        if not name or not cols:
            return
        key = str(name)
        entity_map.setdefault(key, cols)
        entity_map.setdefault(key.upper(), cols)

    if applied_state is not None:
        for obj in applied_state.objects.values():
            if obj.kind == ObjectKind.ENTITY:
                _record(obj.name, _entity_join_keys_from_payload(obj.spec_payload))

    for data in spec_dicts:
        if isinstance(data, dict) and data.get("kind") == ObjectKind.ENTITY:
            _record(data.get("name"), _entity_join_keys_from_payload(data))

    return entity_map


def resolve_ordered_entity_columns(
    entities: list[Any],
    entity_join_keys: Optional[Mapping[str, list[str]]],
) -> list[str]:
    """Resolve authored entity references to ordered join-key columns.

    Each reference is looked up in ``entity_join_keys`` (case-insensitively)
    and expanded to its join-key columns, de-duplicated case-insensitively in
    first-seen order.  A reference absent from the map is used verbatim.

    To keep the overwhelmingly common ``name == join key`` shape byte-stable
    (and avoid rewriting the authored casing), the authored list is returned
    unchanged whenever the resolved columns match it case-insensitively.  When
    ``entity_join_keys`` is empty/``None`` the authored names are returned
    verbatim (historical behaviour).

    Args:
        entities: The authored ``entities`` list (column-name strings or
            ``{"name": ...}`` dicts).
        entity_join_keys: Optional entity-name → join-key-columns map from
            :func:`build_entity_join_key_map`.

    Returns:
        The ordered join-key column names for the wire field.
    """
    authored: list[str] = []
    for ref in entities:
        ref_name = ref.get("name") if isinstance(ref, dict) else ref
        if ref_name is None:
            continue
        authored.append(str(ref_name))

    if not entity_join_keys:
        return authored

    resolved: list[str] = []
    seen: set[str] = set()
    for ref_name in authored:
        cols = entity_join_keys.get(ref_name) or entity_join_keys.get(ref_name.upper()) or [ref_name]
        for col in cols:
            if col.upper() not in seen:
                seen.add(col.upper())
                resolved.append(col)

    if not resolved:
        return authored
    if [c.upper() for c in resolved] == [a.upper() for a in authored]:
        return authored
    return resolved


def compile_to_spec(
    spec_dict: dict[str, Any],
    database: str,
    schema: str,
    *,
    entity_join_keys: Optional[Mapping[str, list[str]]] = None,
) -> dict[str, Any]:
    """Compile a YAML authoring-format spec dict into imperative FeatureViewSpec format.

    Accepts the spec dict produced by ``decl/loader.py`` + ``compiler.py``
    (i.e., with UDF source already inlined as ``udf.function_definition`` and
    type aliases normalized, but duration strings may still be raw strings or
    already converted ``*_sec`` integers).

    This is the single point that translates the authoring key names
    (``entities`` / ``timestamp_col`` — matching the imperative
    ``FeatureView(...)`` constructor) into the wire-form key names
    (``ordered_entity_column_names`` / ``timestamp_field``) carried by
    the compiled SPECIFICATION JSON sent to Snowflake.  The wire-form
    names are owned by GS and persisted in
    ``DESCRIBE ... TYPE = SPECIFICATION`` output; the authoring names
    are owned by the declarative authoring surface.

    Args:
        spec_dict: The loaded/compiled spec dict from the authoring YAML.
        database: Snowflake database name (from connection context).
        schema: Snowflake schema name (from connection context).
        entity_join_keys: Optional entity-name → ordered join-key-columns map
            (see :func:`build_entity_join_key_map`).  When provided, the FV's
            authored entity **names** are resolved to their join-key columns
            for the wire field ``ordered_entity_column_names``.  When omitted,
            the authored ``entities`` list is copied verbatim — preserving the
            historical behaviour and keeping golden hashes stable for callers
            that do not supply the map.

    Returns:
        Dict matching ``FeatureViewSpec.to_dict()`` output, ready for
        JSON serialization in ``FROM SPECIFICATION $$ ... $$``.
    """
    name: str = spec_dict["name"]
    version: str = spec_dict["version"]
    kind: str = spec_dict.get("kind", "StreamingFeatureView")

    # 1. metadata -------------------------------------------------------
    metadata: dict[str, Any] = {
        "database": database,
        "schema": schema,
        "name": name,
        "version": version,
        "spec_format_version": "1",
        "internal_data_version": "1",
        "client_version": _CLIENT_VERSION,
    }

    # 2. offline_configs ------------------------------------------------
    udf = spec_dict.get("udf")
    offline_configs: list[dict[str, Any]] = []
    if kind != "BatchFeatureView" and udf and isinstance(udf, dict):
        table_name = f"{name.upper()}${version.upper()}$UDF_TRANSFORMED"
        offline_configs.append(
            {
                "store_type": "snowflake",
                "table_type": "UDFTransformed",
                "database": database,
                "schema": schema,
                "table": table_name,
                "columns": [dict(col) for col in udf.get("output_columns", [])],
            }
        )

    # 3. spec -----------------------------------------------------------
    raw_features: list[dict[str, Any]] = spec_dict.get("features", [])
    compiled_features = _compile_features(raw_features)

    feature_granularity_sec = _resolve_duration(spec_dict, "feature_granularity")

    raw_sources = [dict(s) for s in spec_dict.get("sources", [])]
    # ``entities`` / ``timestamp_col`` are the authoring-side names that
    # match the imperative ``FeatureView(...)`` constructor; they are
    # translated to the wire-form ``ordered_entity_column_names`` /
    # ``timestamp_field`` here so the compiled SPECIFICATION JSON keeps
    # the GS-owned contract intact.
    # ``ordered_entity_column_names`` must carry the entity **join-key
    # columns**, not the authored entity *names* — the applied side recovers
    # these columns from ``entity.join_keys`` (core
    # ``_build_batch_feature_view_spec``).  When the caller supplies an
    # ``entity_join_keys`` map, resolve each FV's entity references to their
    # ordered join-key columns so a BFV whose entity name differs from its
    # join key does not loop on ``RECREATE_FV``.  When the map is omitted, the
    # authored ``entities`` list is copied verbatim (the historical behaviour,
    # which keeps the name==join-key case and existing golden hashes stable).
    if entity_join_keys:
        ordered_entity_column_names = resolve_ordered_entity_columns(
            list(spec_dict.get("entities", [])), entity_join_keys
        )
    else:
        ordered_entity_column_names = list(spec_dict.get("entities", []))
    spec: dict[str, Any] = {
        "ordered_entity_column_names": ordered_entity_column_names,
        "sources": _normalize_sources_for_imperative_json(raw_sources, fv_kind=kind),
        "features": compiled_features,
    }

    if spec_dict.get("timestamp_col"):
        spec["timestamp_field"] = spec_dict["timestamp_col"]

    # Compile-time default for ``feature_aggregation_method: continuous``
    # on a ``StreamingFeatureView`` with windowed features.  Mirrors the
    # imperative-side ``_DEFAULT_CONTINUOUS_FEATURE_GRANULARITY = "1m"``
    # default so the local-compile hash matches the runtime-stamped
    # SPECIFICATION JSON.  Strictly scoped: ``BatchFeatureView`` and
    # ``tiles`` SFVs both intentionally fall through to the existing
    # surfacing layers (BFV invariant + imperative ValueError
    # respectively).  Skipped entirely when the spec carries no windowed
    # features (a non-aggregated SFV has no place for the default).
    has_windows = any("window_sec" in f for f in compiled_features)
    if (
        feature_granularity_sec is None
        and kind == "StreamingFeatureView"
        and has_windows
        and (spec_dict.get("feature_aggregation_method") or "").lower() == "continuous"
    ):
        feature_granularity_sec = _DEFAULT_CONTINUOUS_FEATURE_GRANULARITY_SEC

    if feature_granularity_sec is not None:
        spec["feature_granularity_sec"] = feature_granularity_sec

    # ``target_lag_sec`` on the wire ``spec`` is the **offline Dynamic
    # Table refresh cadence** in seconds — it maps to
    # ``CREATE DYNAMIC TABLE … TARGET_LAG = '<n> seconds'`` on the
    # offline materialisation side.  After the
    # ``feature_granularity`` / ``refresh_freq`` / ``target_lag``
    # decoupling, the canonical authoring source is ``refresh_freq``
    # (which matches the imperative ``FeatureView(refresh_freq=...)``
    # constructor kwarg); the authoring ``target_lag`` field is OFT
    # staleness only and **does NOT** populate this wire field.
    #
    # Streaming and realtime kinds are excluded entirely: the Snowflake
    # runtime always stamps ``target_lag_sec: 0`` onto streaming /
    # realtime SPECIFICATIONs regardless of the authored value, the
    # Pydantic validator
    # (``FeatureView._reject_refresh_freq_on_stream_or_realtime``)
    # rejects authored ``refresh_freq`` on those kinds entirely, and
    # emitting the key here would create a phantom drift against the
    # runtime-stamped ``0`` (the planner's ``_full_spec_hash`` strips
    # ``target_lag_sec`` via ``_RUNTIME_STAMPED_SPEC_KEYS`` from the
    # deployed side, so stripping it from the authored side is the
    # matching half of that hash-symmetry contract).
    if kind == "BatchFeatureView" and spec_dict.get("refresh_freq"):
        spec["target_lag_sec"] = parse_duration_to_seconds(spec_dict["refresh_freq"])
    elif kind == "StreamingFeatureView" and has_windows and spec_dict.get("refresh_freq"):
        # A tiled streaming FV schedules an offline tile Dynamic Table
        # whose ``TARGET_LAG`` is ``refresh_freq``.  Carry the
        # authoring-form ``refresh_freq`` (NOT ``target_lag_sec`` — that
        # is the OFT ingest lag the runtime stamps to ``0`` and
        # ``_RUNTIME_STAMPED_SPEC_KEYS`` strips) so the local-vs-applied
        # round-trip is symmetric: the applied side recovers the same
        # authoring-form key from the deployed DT ``REFRESH_FREQ`` via
        # ``state._inject_fv_refresh_freq_from_list_row``.  ``refresh_freq``
        # is operational (``_OPERATIONAL_FV_KEYS``), so it is stripped
        # from the structural hash on both sides.
        spec["refresh_freq"] = spec_dict["refresh_freq"]

    if has_windows:
        method = spec_dict.get("feature_aggregation_method", "tiles")
        spec["feature_aggregation_method"] = method

    # Advanced BFV structural knobs.  ``cluster_by`` is the only one wired
    # up so far (Phase 2 of the advanced BFV plan); subsequent phases will
    # add ``refresh_mode``, ``initialize``, ``storage_config``, and
    # ``aggregation_secondary_keys`` at this same insertion point.  Each
    # must live inside the inner ``spec`` dict so it contributes to
    # ``_full_spec_hash`` and an edit lands as ``RECREATE_FV``.
    cluster_by = spec_dict.get("cluster_by")
    if isinstance(cluster_by, list) and cluster_by:
        spec["cluster_by"] = [str(c) for c in cluster_by]

    refresh_mode = spec_dict.get("refresh_mode")
    if isinstance(refresh_mode, str) and refresh_mode:
        spec["refresh_mode"] = refresh_mode.upper()

    # ``initialize`` resolution with the Phase 4 back-compat alias:
    # prefer the top-level key, fall back to the legacy nested
    # ``backfill.initialize``.  The compiled inner spec carries a single
    # ``initialize`` value so the structural hash is unambiguous.
    initialize = spec_dict.get("initialize")
    if initialize is None:
        bf = spec_dict.get("backfill")
        if isinstance(bf, dict):
            initialize = bf.get("initialize")
    if isinstance(initialize, str) and initialize:
        spec["initialize"] = initialize.upper()

    storage_config = spec_dict.get("storage_config")
    if isinstance(storage_config, dict) and storage_config:
        # Strip ``None`` values so a partially-populated authoring block
        # (``format: snowflake`` with no Iceberg fields) doesn't bloat
        # the structural hash with explicit-null noise.
        spec["storage_config"] = {k: v for k, v in storage_config.items() if v is not None}

    secondary_keys = spec_dict.get("aggregation_secondary_keys")
    if isinstance(secondary_keys, list) and secondary_keys:
        spec["aggregation_secondary_keys"] = [str(c) for c in secondary_keys]

    if kind != "BatchFeatureView" and udf and isinstance(udf, dict):
        spec["udf"] = _compile_udf(udf)

    # Enrich Stream sources with column schema if missing.
    # Only use columns that were already resolved (e.g. from datasource YAML
    # loaded by the pipeline). Do NOT fall back to UDF output_columns here —
    # those are the transformed output schema, not the raw input schema.
    enriched_sources = []
    for src in spec["sources"]:
        enriched_sources.append(dict(src))
    spec["sources"] = enriched_sources

    online_enabled = bool(spec_dict.get("online", kind != "BatchFeatureView"))
    online_store: str | None = "postgres" if online_enabled else None

    # 4. Assemble root --------------------------------------------------
    result: dict[str, Any] = {
        "kind": kind,
        "metadata": metadata,
        "offline_configs": offline_configs,
        "spec": spec,
    }
    if online_store is not None:
        result["online_store_type"] = online_store
    return result


def to_json(spec_result: dict[str, Any]) -> str:
    """Serialize a compiled spec dict to JSON safe for SQL ``$$`` quoting.

    Args:
        spec_result: Output of :func:`compile_to_spec`.

    Returns:
        JSON string with ``$$`` sequences replaced by ``$\\u0024``.
    """
    return sanitize_json_for_dollar_quoting(json.dumps(spec_result))
