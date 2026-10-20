"""Invariant engine for the declarative feature store library.

Pure functions only — no database connections, no file I/O.
Entry point: ``validate_specs(batch, applied_state) -> list[ValidationResult]``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Literal, Mapping

from snowflake.ml._internal.utils import identifier
from snowflake.ml.feature_store.decl.compiler import (
    canonicalize_sql_for_hash,
    extract_select_output_columns,
    strip_sql_predicates,
    to_comparison_identifier,
)
from snowflake.ml.feature_store.decl.spec_models import SpecBase
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    SpecBatch,
    ValidationResult,
)

# Kinds that must carry an explicit version string.
_VERSIONED_KINDS: frozenset[str] = frozenset(
    {"StreamingFeatureView", "RealtimeFeatureView", "BatchFeatureView", "FeatureGroup"}
)

# FeatureView kinds that, when ``applied.from_specification`` is True, must
# diff against the deployed spec via the full-spec hash path (compile then
# :func:`_full_spec_hash`).  Mirrors :data:`planner._RECREATE_OP` keys so the
# validator's idempotency check stays consistent with
# :func:`planner.generate_plan` — otherwise the validator always disagrees
# with the planner on FVs recovered from ``DESCRIBE ... TYPE = SPECIFICATION``
# (different hash strategy → spurious ``VERSION_CONFLICT`` on round-trip).
_FV_RECREATE_KINDS: frozenset[str] = frozenset({"StreamingFeatureView", "RealtimeFeatureView", "BatchFeatureView"})

# Keys excluded from all hash computations so that storing a hash in the spec
# itself does not change the hash on the next read.
_INTERNAL_KEYS: frozenset[str] = frozenset({"_hash", "_content_hash"})

# FSBaseType → Snowflake SQL type mapping used for structural fingerprinting.
# Mirrors sql_generator._SF_TYPE_MAP so that DESCRIBE-side types (SQL format)
# and spec-side types (FSBaseType format) normalise to the same string.
_FINGERPRINT_TYPE_MAP: dict[str, str] = {
    "StringType": "VARCHAR",
    "IntegerType": "INTEGER",
    "LongType": "INTEGER",
    "FloatType": "FLOAT",
    "DoubleType": "FLOAT",
    "DecimalType": "NUMBER",
    "BooleanType": "BOOLEAN",
    "BinaryType": "BINARY",
    "TimestampType": "TIMESTAMP_NTZ",
}


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


_SOURCE_KIND_ALIASES: frozenset[str] = frozenset({"StreamingSource", "BatchSource"})

# FeatureView subkinds (batch / streaming / realtime) are one physical object.
# Only _check_duplicate_spec_keys collapses them; spec_key keeps the real
# subkind because applied-state lookups and planner routing depend on it.
_FV_DUPLICATE_ALIASES: frozenset[str] = _FV_RECREATE_KINDS


def spec_key(data: dict[str, Any], *, database: str = "", schema: str = "") -> str:
    """Build a unique key for an object's identity.

    Versioned kinds (FeatureView subkinds and FeatureGroup) key as
    ``kind:database.schema:NAME:VERSION`` because their identity is the name
    plus the version; Entity and Datasource key without the version.  Names
    are uppercased so the key is case-stable.

    StreamingSource and BatchSource both collapse to the generic Datasource
    kind, so a freshly-loaded YAML matches the applied-state entry the state
    module builds (it uses the generic kind for display consistency).

    The database and schema arguments are fallbacks, consulted only when the
    spec omits the field.  A spec that names its own database always wins over
    the connection context, while bare YAMLs still get qualified against the
    active connection so their keys collide with the always-qualified
    applied-state keys.

    Args:
        data: Spec dict carrying ``kind``, ``name``, ``database``, and either
            ``schema`` or ``schema_``.
        database: Fallback database when ``data`` has no ``database`` field.
        schema: Fallback schema when ``data`` has no ``schema`` / ``schema_``.

    Returns:
        Canonical key string.
    """
    kind = data.get("kind", "Unknown")
    if kind in _SOURCE_KIND_ALIASES:
        kind = "Datasource"
    name = data.get("name", "unnamed").upper()
    db = (data.get("database", "") or database or "").upper()
    schema = (data.get("schema", "") or data.get("schema_", "") or schema or "").upper()
    qualifier = f"{db}.{schema}" if (db or schema) else ""
    key = f"{kind}:{qualifier}:{name}"
    # Object identity is (name, version) for versioned kinds — append the
    # version so distinct versions of one name key (and therefore plan /
    # validate / apply) independently instead of shadowing each other.
    # Entity / Datasource are schema-level and unversioned, so they keep
    # the name-only key.  A versioned kind that is missing its version
    # keeps the name-only key too; the absence is surfaced as
    # ``MISSING_VERSION`` by :func:`validate_specs`.
    version = data.get("version")
    if kind in _VERSIONED_KINDS and version:
        key = f"{key}:{str(version).upper()}"
    return key


def _spec_hash(data: dict[str, Any]) -> str:
    """SHA-256 over the full spec dict, excluding the internal hash keys."""
    clean = {k: v for k, v in data.items() if k not in _INTERNAL_KEYS}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _content_hash(data: dict[str, Any]) -> str:
    """SHA-256 over the spec dict, excluding the version field."""
    d = {k: v for k, v in data.items() if k not in _INTERNAL_KEYS}
    d = copy.deepcopy(d)
    d.pop("version", None)
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sf_type_for_fp(type_str: str) -> str:
    """Normalise a type string for structural fingerprinting.

    Spec-side names like StringType map to their Snowflake SQL equivalents;
    strings that already come from DESCRIBE are simply uppercased.

    Args:
        type_str: A raw type string from either an authoring spec or DESCRIBE.

    Returns:
        Normalised uppercase SQL type string.
    """
    return _FINGERPRINT_TYPE_MAP.get(type_str, type_str).upper()


def _structural_fingerprint(data: dict[str, Any]) -> dict[str, Any]:
    """Build a structural fingerprint from a spec or DESCRIBE-derived dict.

    Extracts the name, the optional version, and sorted output column
    name/type pairs.  Both the authoring path and the DESCRIBE reconstruction
    path call this, so the two always agree on the fingerprint for a given
    deployed feature view.

    Output columns come from each feature's output column, or from the UDF's
    output columns when one is present.  On the DESCRIBE side those entries
    are pre-populated with the non-primary-key column rows.

    Entity specs also fold in the description and the sorted join-key schema,
    so an edit to any of the three editable surfaces of an entity tag produces
    an update op.  Without that, a description-only edit hashes identically
    and the planner reports no change.

    Args:
        data: Spec dict or DESCRIBE-derived dict.

    Returns:
        Dict with ``name``, ``version`` and ``columns`` keys, plus
        ``description`` and ``join_keys`` for Entity kinds.
    """
    name = (data.get("name") or "").upper()
    version = (data.get("version") or "").upper()

    udf = data.get("udf") or {}
    if udf and "output_columns" in udf:
        raw_cols = list(udf["output_columns"])
    else:
        raw_cols = [f.get("output_column") or {} for f in data.get("features", [])]

    cols = sorted(
        (
            {
                "name": (c.get("name") or "").upper(),
                "type": _sf_type_for_fp(c.get("type") or ""),
            }
            for c in raw_cols
            if c.get("name")
        ),
        key=lambda x: x["name"],
    )
    fingerprint: dict[str, Any] = {"name": name, "version": version, "columns": cols}

    # Entity-specific fields.  An empty / missing ``description`` collapses
    # to "" so ``_build_entity_object`` (no comment) and a YAML lacking a
    # ``description:`` key hash identically — the round-trip after a clean
    # export must still emit ``NO_CHANGE``.
    if data.get("kind") == "Entity":
        fingerprint["description"] = data.get("description") or ""
        fingerprint["join_keys"] = sorted(
            (
                {
                    # Mixed provenance meets here: the authored half is SQL
                    # syntax, the applied half is bare stored form.  Unquote
                    # before case-folding or a re-quoted ``"userId"`` never
                    # matches the deployed ``userId``.
                    "name": to_comparison_identifier(jk["name"]),
                    "type": _sf_type_for_fp(jk.get("type") or ""),
                }
                for jk in (data.get("join_keys") or [])
                if isinstance(jk, dict) and jk.get("name")
            ),
            key=lambda x: x["name"],
        )

    # FeatureView-specific fields: include source bindings (name + table +
    # query) so a BatchSource ``table:`` swap surfaces as a structural diff
    # even when the planner falls back to the structural fingerprint
    # (``applied.from_specification=False``).  BUG_BASH §8 cascade —
    # without this widen, a name-only FV reference whose datasource
    # YAML's ``table:`` was edited hashed identically and the planner
    # silently emitted ``NO_CHANGE``.
    #
    # Conditional contract: the ``sources`` key is added ONLY when the
    # input dict carries at least one source binding with a non-empty
    # ``name``/``table``/``query``.  This keeps backwards-compatibility
    # with describe-only fallback specs (``state.py`` Path 3) and other
    # legacy fingerprint inputs that do not carry sources — both sides
    # of the diff omit the key and the hash continues to match.
    if data.get("kind") in _FV_RECREATE_KINDS:
        sources_raw = data.get("sources") or []
        fp_sources: list[dict[str, str]] = []
        for src in sources_raw:
            if isinstance(src, dict):
                src_name = (src.get("name") or "").upper()
                src_table = (src.get("table") or "").upper()
                src_query = src.get("query") or ""
            else:
                src_name = (getattr(src, "name", "") or "").upper()
                src_table = (getattr(src, "table", "") or "").upper()
                src_query = getattr(src, "query", "") or ""
            if not (src_name or src_table or src_query):
                continue
            # Query-backed sources: the query body IS the binding identity.
            # Local-compile preserves the user's authoring ``name`` while the
            # DT-text recovery path (Phase 4) synthesises ``<FV>__SOURCE``;
            # blanking ``name`` here keeps both halves of the diff hashing
            # identically without losing source-table sensitivity for the
            # table-shape path. Both sides should normalise the query via
            # :func:`compiler.normalize_sql_whitespace`.
            if src_query and not src_table:
                fp_sources.append(
                    {
                        "name": "",
                        "table": "",
                        "query": canonicalize_sql_for_hash(str(src_query)),
                    }
                )
            else:
                fp_sources.append(
                    {
                        "name": src_name,
                        "table": src_table,
                        "query": canonicalize_sql_for_hash(str(src_query)),
                    }
                )
        if fp_sources:
            fingerprint["sources"] = sorted(fp_sources, key=lambda x: (x["name"], x["table"], x["query"]))
    return fingerprint


def structural_fingerprint_hash(data: dict[str, Any]) -> str:
    """SHA-256 of the structural fingerprint.

    The canonical idempotency hash across the pipeline.  Applied state rebuilt
    from DESCRIBE produces the same digest as the authoring spec, so no-change
    detection works without a specification column in SHOW results.

    Args:
        data: Spec dict or DESCRIBE-derived dict.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    fp = _structural_fingerprint(data)
    canonical = json.dumps(fp, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Source-payload diff helper (planner / planner-test surface)
# ---------------------------------------------------------------------------


# Description-style fields that the source diff helper strips before
# structural hashing on BOTH sides.  Edits to these keys never affect
# the structural identity of a source — they route through
# ``OpKind.UPDATE_SOURCE`` (StreamingSource) or a no-op (BatchSource).
_SOURCE_DESC_KEYS: frozenset[str] = frozenset(
    {
        "description",
        "desc",
        "comment",
        "details",
    }
)

# Producer-protocol field carried by the local YAML
# (``StreamingSource.type``) but never stamped onto the runtime row by
# ``FeatureStore.list_stream_sources()``.  Stripped from the LOCAL
# payload only so a clean round-trip matches; the applied side
# generally does not carry the key, but if it ever does (a synthetic
# test fixture or a future runtime change) the key remains in the
# applied stripped payload so a genuine drift surfaces as
# ``"recreate"``.
_SOURCE_LOCAL_VOLATILE_KEYS: frozenset[str] = _SOURCE_DESC_KEYS | frozenset({"type"})

# Map the concrete source ``kind`` aliases to the runtime
# ``SourceType`` value that :func:`state._build_stream_source_object`
# (and :func:`state._datasource_objects_from_specs`) stamp onto the
# applied payload.  The local ``StreamingSource`` / ``BatchSource``
# Pydantic models do NOT expose a top-level ``source_type:`` authoring
# field — it lives on the FV-level :class:`SourceRef` instead — so the
# diff helper derives ``source_type`` from ``kind`` whenever the local
# payload omits it.  Without this derivation, the applied side's
# ``source_type`` (always present) would always look like a structural
# drift and every clean round-trip would emit ``"recreate"``.
_SOURCE_KIND_TO_SOURCE_TYPE: dict[str, str] = {
    "StreamingSource": "Stream",
    "BatchSource": "Batch",
}


def _strip_volatile_source_keys(payload: dict[str, Any], *, strip_type: bool) -> dict[str, Any]:
    """Return a structurally-stable copy of a source payload.

    Drops description-style fields (and optionally the producer-protocol
    ``type``), canonicalises the kind to Datasource, and derives
    ``source_type`` from the original kind alias when the payload omits it.
    After this, a local YAML payload hashes identically to the runtime applied
    row.

    Args:
        payload: The source spec or applied-state dict.  Non-dict input yields
            an empty dict so callers can skip a pre-check.
        strip_type: Also drop the ``type`` field.  Set on the local payload
            only, since the runtime row never carries it; keeping it on the
            applied side lets a genuine runtime drift surface as a recreate.

    Returns:
        A new dict with the volatile keys removed.  An explicit
        ``source_type`` on the input always wins, so an authored mismatch
        still surfaces as structural drift.
    """
    if not isinstance(payload, dict):
        return {}
    keys_to_strip: frozenset[str] = _SOURCE_LOCAL_VOLATILE_KEYS if strip_type else _SOURCE_DESC_KEYS
    out = {k: v for k, v in payload.items() if k not in keys_to_strip}
    original_kind = out.get("kind")
    if original_kind in _SOURCE_KIND_ALIASES:
        out["kind"] = "Datasource"
        if "source_type" not in out and original_kind in _SOURCE_KIND_TO_SOURCE_TYPE:
            out["source_type"] = _SOURCE_KIND_TO_SOURCE_TYPE[original_kind]
    return out


def _source_structural_hash(stripped: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of a stripped source payload.

    Deliberately separate from the structural fingerprint, which projects
    every input down to name, version and columns — so two sources differing
    only in their columns, store type, table or query would fingerprint
    identically.  A source diff needs the full shape, so we hash the stripped
    dict directly.

    Args:
        stripped: A source payload with its volatile keys already removed.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(stripped, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve_source_description(payload: dict[str, Any]) -> str:
    """Pick the description text from a source payload.

    Prefers ``description``, then ``desc``, else empty.  Nulls and empty
    values collapse together so the comparison is symmetric across a missing
    key, a key set to null, and a key set to an empty string.

    Args:
        payload: Source spec or applied-state dict.

    Returns:
        Whitespace-trimmed, case-preserved description text.
    """
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("description") or payload.get("desc") or ""
    return str(raw).strip()


def _canonicalize_columns(columns: Any) -> list[dict[str, str]]:
    """Project a columns list to a stable, sorted, type-stripped form.

    Washes out cosmetic drift that must not trigger a destructive recreate:
    entries sort by uppercased name so YAML order and runtime order agree,
    type strings are trimmed so a padded DESCRIBE row matches a tight YAML
    one, and names are uppercased so a lowercased YAML identifier matches an
    always-uppercase runtime row.

    Args:
        columns: A ``columns`` list value; non-list input returns an empty list
            so callers can skip a pre-check.

    Returns:
        A new sorted list of name/type dicts.  Entries without a name are
        dropped.
    """
    if not isinstance(columns, list):
        return []
    out: list[dict[str, str]] = []
    for c in columns:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").upper()
        if not name:
            continue
        type_str = str(c.get("type") or "").strip()
        out.append({"name": name, "type": type_str})
    return sorted(out, key=lambda x: x["name"])


# Source *location* keys that are implicit in the Datasource ``spec_key``
# qualifier (``kind:database.schema:NAME``) and therefore must NOT drive the
# structural diff computed by :func:`compute_source_diff_kind`.  The applied
# side (``state._datasource_objects_from_specs``) always stamps
# ``database`` / ``schema`` from the FV metadata, while a source referenced
# only by orphaned-OFT FVs is exported as a bare columns-only YAML with none
# of these keys — leaving them in the hash produces a spurious
# ``RECREATE_SOURCE`` after ``snow feature init``.  The physical ``table`` /
# ``query`` binding is deliberately NOT in this set: a genuine relocation must
# still surface as a recreate.
_SOURCE_LOCATION_KEYS: frozenset[str] = frozenset(
    {
        "database",
        "schema",
        "schema_",
        "source_database",
        "source_schema",
    }
)


def compute_source_diff_kind(
    local_payload: dict[str, Any],
    applied_payload: dict[str, Any],
) -> Literal["no_change", "update_desc_only", "recreate"]:
    """Classify a Datasource diff as no_change, update_desc_only, or recreate.

    The planner uses this to pick an op kind for an authored source that
    already has an applied counterpart:

    * ``recreate`` — the structural shape differs: a column added, removed or
      retyped, or a swapped store type, table, query or producer type.  The
      planner marks a StreamingSource recreate destructive (real drop +
      re-register) but a BatchSource recreate non-destructive (executor no-op;
      the consuming FV op is the ``--destructive`` gate).
    * ``update_desc_only`` — the shape matches but the description differs.
    * ``no_change`` — both match.

    Both sides are stripped to a structurally-stable shape before hashing.
    Descriptions are dropped from both; the producer-protocol ``type`` is
    dropped from the local side only, since the runtime row never carries it.
    The target-implicit location keys (``database`` / ``schema`` /
    ``source_database`` / ``source_schema``) are dropped from both sides since
    they are already encoded in the ``Datasource`` ``spec_key``.  Columns are
    canonicalised symmetrically, and dropped from both sides when the applied
    side recovered none — the usual shape for a table-backed batch source,
    where keeping them would force a needless recreate.  Queries are
    canonicalised so a comment or whitespace edit stays cosmetic.  A structural
    mismatch short-circuits the description comparison, since description drift
    is free inside a recreate.

    Args:
        local_payload: The authored YAML payload for the source.
        applied_payload: The applied-state payload for the same source.

    Returns:
        One of ``no_change``, ``update_desc_only``, or ``recreate``.
    """
    stripped_local = _strip_volatile_source_keys(local_payload, strip_type=True)
    stripped_applied = _strip_volatile_source_keys(applied_payload, strip_type=False)

    # Location keys (``database`` / ``schema`` / ``source_database`` /
    # ``source_schema``) are implicit in the Datasource ``spec_key``
    # qualifier and asymmetric across the diff surface: the applied side
    # always stamps ``database`` / ``schema`` while a bare orphaned-FV export
    # carries none.  Drop them from BOTH sides so they never drive a spurious
    # ``RECREATE_SOURCE`` (see ``_SOURCE_LOCATION_KEYS``).  The physical
    # ``table`` / ``query`` binding stays in the hash, so a genuine relocation
    # still surfaces as a recreate.
    for _loc_key in _SOURCE_LOCATION_KEYS:
        stripped_local.pop(_loc_key, None)
        stripped_applied.pop(_loc_key, None)

    # Columns canonicalization (Bug 1 closure — Phase B5).  Pop the
    # raw value off BOTH sides, run each through the same
    # ``_canonicalize_columns`` helper, then either
    #   (a) drop columns entirely from the structural hash on both
    #       sides when the applied side has nothing to compare against
    #       (legacy DT-text recovery for table-backed BatchSources
    #       leaves ``columns: []`` because the recovery path can't
    #       enumerate the source schema), or
    #   (b) put the canonicalised lists back so legitimate add /
    #       remove / type-change diffs continue to surface as
    #       structural drift.
    local_columns_raw = stripped_local.pop("columns", None)
    applied_columns_raw = stripped_applied.pop("columns", None)
    canonical_applied_columns = _canonicalize_columns(applied_columns_raw)
    canonical_local_columns = _canonicalize_columns(local_columns_raw)
    if canonical_applied_columns:
        # Both sides participate in the structural hash via the
        # canonical projection.
        stripped_local["columns"] = canonical_local_columns
        stripped_applied["columns"] = canonical_applied_columns
    # else: applied side has no authoritative columns → both sides
    # omit ``columns`` from the hash so a clean replan does not
    # force ``RECREATE_SOURCE`` (Bug 1 regression pin).

    # Query canonicalization (PR A — A2).  A query-backed BatchSource
    # binds on its SQL body; a comment / keyword-case / whitespace-only
    # edit is cosmetic and must NOT surface as ``recreate`` (which would
    # drop the companion ``$SNAPSHOTS`` for append-only BFVs).  Both
    # sides run through the same canonicaliser so they converge.
    for _stripped in (stripped_local, stripped_applied):
        _q = _stripped.get("query")
        if isinstance(_q, str) and _q.strip():
            _stripped["query"] = canonicalize_sql_for_hash(_q)

    local_hash = _source_structural_hash(stripped_local)
    applied_hash = _source_structural_hash(stripped_applied)

    if local_hash != applied_hash:
        return "recreate"

    if _resolve_source_description(local_payload) == _resolve_source_description(applied_payload):
        return "no_change"
    return "update_desc_only"


# Volatile metadata fields that change with tool versions / build numbers
# but do not represent a change in feature-store semantics.  Stripped before
# computing :func:`_full_spec_hash` so that a client-version bump does not
# falsely flag a deployed spec as changed.
#
# ``oft_id`` is the per-deploy internal ID Snowflake stamps into
# ``DESCRIBE ... TYPE = SPECIFICATION`` output; it changes on every
# ``CREATE`` and is meaningless to the authoring spec, so we treat it the
# same as the other client-version-style stamps.
_VOLATILE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "client_version",
        "spec_format_version",
        "internal_data_version",
        "oft_id",
    }
)

# Top-level keys that Snowflake adds to the ``DESCRIBE ... TYPE =
# SPECIFICATION`` payload but that the authoring layer never produces.
# These are derived runtime metadata (offline-table descriptors,
# online-store routing flags) and are stripped before hashing so a clean
# round-trip does not flag them as drift.
_DERIVED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "offline_configs",
        "online_store_type",
    }
)

# Spec-level keys that Snowflake stamps onto the ``DESCRIBE ... TYPE =
# SPECIFICATION`` payload at deploy time but that the authoring layer
# does not produce when the operator omits them from the YAML.  Stripped
# before hashing so a runtime-side default (e.g. ``target_lag_sec: 0``
# stamped after a streaming-FV CREATE) does not falsely flag the
# deployed spec as changed on the next ``snow feature plan``.
#
# Add a key here when:
#   * the live runtime emits it on DESCRIBE TYPE = SPECIFICATION,
#   * the local-compile path (``spec_compiler.compile_to_spec``) does
#     not emit it for the equivalent local YAML,
#   * its value is a deploy-time default (zero, infinity, or otherwise
#     boilerplate), AND
#   * dropping it from the hash does not lose semantic information
#     (i.e. the local YAML really has no equivalent).
_RUNTIME_STAMPED_SPEC_KEYS: frozenset[str] = frozenset(
    {
        # Snowflake stamps target_lag_sec=0 onto every streaming FV
        # SPECIFICATION post-CREATE; the authoring YAML never sets it
        # because the bug-bash pattern is "use the default lag".
        "target_lag_sec",
    }
)

# snowml-core stamps ``initialize`` onto every BatchFV at create time even
# when the authoring YAML does not author it.  The applied-state side recovers
# it from the deployed DT text via
# :func:`state._inject_batch_fv_fields_from_list_row` while the local-compile
# side leaves it unset, breaking the hash round-trip.  Strip the key from
# both sides when the deployed value matches the snowml-core default so an
# unedited round-trip emits ``NO_CHANGE``.  An operator who authors a
# non-default value keeps the key in the hash.
#
# ``refresh_mode`` is kept out of this dict: its ``AUTO`` sentinel is stripped
# directly in :func:`_strip_default_operational_fields`, and a resolved
# INCREMENTAL / FULL is handled asymmetrically in
# :func:`_normalize_applied_bfv_for_hash` and
# :func:`batch_feature_view_structural_equivalent`.
_BFV_OPERATIONAL_DEFAULTS: dict[str, str] = {
    "initialize": "ON_CREATE",
}

# Snowflake's type system on the deployed offline / online tables widens
# the authoring narrow integer / float types.  Local-compile preserves
# the YAML's authored type names (e.g. ``IntegerType``, ``FloatType``)
# while the deployed ``DESCRIBE … TYPE = SPECIFICATION`` payload reports
# the widened storage type (``LongType``, ``DoubleType``).  Normalise
# both sides to the deployed widened form before hashing so a clean
# round-trip matches.
_BFV_TYPE_PROMOTION: dict[str, str] = {
    "IntegerType": "LongType",
    "FloatType": "DoubleType",
}


def _strip_default_operational_fields(inner: dict[str, Any]) -> None:
    """Drop snowml-core operational defaults from a BatchFV inner spec.

    ``initialize`` is stamped at create time and recovered from the deployed
    table text, but the local side only emits it when the operator authored it.
    Dropping it when it matches the default keeps the two sides hashing
    identically.  ``refresh_mode == "AUTO"`` is stripped symmetrically — it is
    Snowflake's "unset" sentinel (the validator only accepts FULL / INCREMENTAL)
    while a live tiled-online DESCRIBE still reports it; an authored FULL /
    INCREMENTAL stays hash-significant and is handled asymmetrically in
    :func:`_normalize_applied_bfv_for_hash`.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV payload, mutated in place.
            Non-dict input is a no-op.
    """
    if not isinstance(inner, dict):
        return
    for key, default in _BFV_OPERATIONAL_DEFAULTS.items():
        if inner.get(key) == default:
            inner.pop(key, None)
    if str(inner.get("refresh_mode") or "").upper() == "AUTO":
        inner.pop("refresh_mode", None)


def _normalize_applied_bfv_for_hash(
    applied_payload: dict[str, Any],
    local_compiled: dict[str, Any],
) -> dict[str, Any]:
    """Strip Snowflake-resolved fields the operator never authored.

    Snowflake stamps a ``refresh_mode`` of INCREMENTAL or FULL even when nobody
    asked for one, while the local compiled spec omits the key.  Dropping it
    from the applied payload lets the two hashes converge.  Only valid on
    applied payloads, never on local specs.

    Args:
        applied_payload: Full applied-state payload.
        local_compiled: The compiled local spec for the same object.

    Returns:
        A normalised deep copy of the applied payload.
    """
    import json as _json

    local_spec = local_compiled.get("spec")
    local_inner: dict[str, Any] = local_spec if isinstance(local_spec, dict) else {}
    payload: dict[str, Any] = _json.loads(_json.dumps(applied_payload))
    applied_spec = payload.get("spec")
    inner = applied_spec if isinstance(applied_spec, dict) else None
    if inner is not None and "refresh_mode" not in local_inner:
        inner.pop("refresh_mode", None)
    return payload


def _strip_default_cluster_by(inner: dict[str, Any]) -> None:
    """Drop ``cluster_by`` from a BatchFV inner spec when it matches the default.

    snowml-core defaults the cluster list to the ordered entity columns, plus a
    tile-start column for tiled views or the ``timestamp_col`` for a non-tiled
    BFV that has one (the append-only snapshot shape).  YAML that leaves the key
    out compiles without it while the applied side recovers the deployed
    default, so both sides must drop it to hash identically.  An operator who
    authors a non-default list keeps the key, so editing it still forces a
    recreate.

    Tiling is detected from either half of the diff, so the helper works
    symmetrically.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV payload, mutated in place.
    """
    if not isinstance(inner, dict):
        return
    cluster_by = inner.get("cluster_by")
    if not isinstance(cluster_by, list):
        return
    entities = inner.get("ordered_entity_column_names") or []
    if not isinstance(entities, list):
        return
    entities_uc = [str(e).upper() for e in entities]
    cluster_uc = [str(c).upper() for c in cluster_by]
    defaults = [entities_uc, entities_uc + ["TILE_START"]]
    # Non-tiled BFVs with a timestamp column (e.g. every append-only BFV)
    # default to ``[<entities>..., <timestamp_col>]``.  The compiled inner
    # spec carries the timestamp under ``timestamp_field``.
    timestamp_field = inner.get("timestamp_field")
    if timestamp_field:
        defaults.append(entities_uc + [str(timestamp_field).upper()])
    if cluster_uc in defaults:
        inner.pop("cluster_by", None)


def _normalise_feature_column_types(features: Any) -> None:
    """Widen narrow integer and float types in a BatchFV features list.

    Snowflake stores integers as 64-bit and floats as doubles, so the deployed
    specification reports the wider type even when the YAML authored the
    narrower one.  Rewriting both sides to the wide form keeps a clean
    round-trip.

    Args:
        features: The ``features`` list of a BatchFV inner spec, mutated in
            place.  Non-list and non-dict entries are skipped.
    """
    if not isinstance(features, list):
        return
    for feat in features:
        if not isinstance(feat, dict):
            continue
        for col_key in ("source_column", "output_column"):
            col = feat.get(col_key)
            if not isinstance(col, dict):
                continue
            _ctype = col.get("type")
            promoted = _BFV_TYPE_PROMOTION.get(_ctype) if isinstance(_ctype, str) else None
            if promoted is not None:
                col["type"] = promoted


# --- secondary-key FROM SPECIFICATION provenance -----------------------------
# snowml-core's spec builders (``feature_store._build_batch_feature_view_spec``
# and ``streaming_registration._build_streaming_feature_view_spec``) rewrite the
# deployed spec of any BatchFV/StreamingFV carrying ``aggregation_secondary_keys``
# in three ways the local decl compiler never mirrors, so a clean apply-then-plan
# loops on ``RECREATE_FV``.  Confirmed against live ``DESCRIBE ... TYPE =
# SPECIFICATION`` captures of a tiled online BatchFV
# (``golden_specs/CONTENT_AGG_BATCH_SK.json``) and a continuous online StreamingFV
# (``golden_specs/CONTENT_AGG_STREAM_SK.json``):
#
#   1. the SK is appended to ``ordered_entity_column_names``
#      (``["USER_ID","OFFER_KEY"]`` vs local ``["USER_ID"]``);
#   2. every real aggregation ``output_column`` becomes ``ArrayType`` with a
#      server-chosen ``element_type`` (authored ``StringType`` -> ``LongType`` for
#      ``approx_count_distinct``) -- fully derived from (function, source type),
#      never operator-authored;
#   3. one synthesised SK-array feature is added per distinct window
#      (``<SK>_KEYS_<window>S``: sourced from the SK, ``ArrayType`` output, no
#      ``function``) -- pure core derivation from the SK column.
#
# ``_strip_secondary_key_derivations`` canonicalises all three so the authored
# local compile and the ``snow feature export`` recovery converge on the same
# hash, while preserving the SK identity and real aggregation fields
# (``function`` / ``window_sec`` / ``function_params`` / ``source_column`` /
# ``output_column.name``) so genuine edits still bump the hash.
#
# Batch + Streaming are scoped by the caller; SK-gating here keeps the streaming
# detection-safety contract intact for non-SK streaming ArrayType FVs (their
# ``element_type`` stays independently hash-significant -- see
# ``test_streaming_fv_array_element_type_roundtrip``).  For an SK streaming FV the
# ``element_type`` is a pure server derivation and is dropped like batch, with a
# genuine change caught by the retained fields (see
# ``test_element_type_change_still_bumps_hash``).


def _secondary_key_set(inner: dict[str, Any]) -> set[str]:
    """Comparison-folded set of the spec's secondary-key columns, or empty when none.

    Uses ``to_comparison_identifier`` (the same case-insensitive fold the
    exporter's SK filters use) rather than raw ``.upper()``: these are
    comparison sites, not the case-preserving identity sites the stored-form
    canonicalization protects.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV/StreamingFV spec_payload.

    Returns:
        The comparison-folded secondary-key column names, or an empty set when
        the spec declares none.
    """
    sk_raw = inner.get("ordered_secondary_key_column_names") or inner.get("aggregation_secondary_keys") or []
    return {to_comparison_identifier(s) for s in sk_raw if s} if isinstance(sk_raw, list) else set()


def _strip_sk_from_entities(inner: dict[str, Any], sk_set: set[str]) -> None:
    """(1) Drop the SK from the entity-column list (identity kept in the SK field)."""
    entities = inner.get("ordered_entity_column_names")
    if isinstance(entities, list):
        inner["ordered_entity_column_names"] = [c for c in entities if to_comparison_identifier(c) not in sk_set]


def _is_synthesised_sk_array_feature(feat: Any, sk_set: set[str]) -> bool:
    """(3) True for the synthesised SK-array feature: sourced from the SK, an
    ``ArrayType`` output, and no aggregation ``function``.

    Args:
        feat: A single ``features[]`` entry (dict or otherwise).
        sk_set: Comparison-folded secondary-key column names.

    Returns:
        ``True`` if ``feat`` is the pure SK-array derivation, else ``False``.
    """
    if not isinstance(feat, dict):
        return False
    # Bind before isinstance so mypy narrows away the None from dict.get.
    source_column = feat.get("source_column")
    output_column = feat.get("output_column")
    sc = source_column if isinstance(source_column, dict) else {}
    oc = output_column if isinstance(output_column, dict) else {}
    return (
        to_comparison_identifier(sc.get("name")) in sk_set
        and feat.get("function") in (None, "")
        and str(oc.get("type")) == "ArrayType"
    )


def _strip_server_derived_agg_output(feat: Any) -> Any:
    """(2) On a real aggregation feature, drop the server-derived
    ``output_column`` type wrapping (``type`` / ``element_type`` / ``length``,
    redundant with the retained function + source type) and the runtime-default
    VARCHAR ``length`` the deployed spec stamps onto the ``source_column``.

    Args:
        feat: A single ``features[]`` entry, mutated in place when a dict.

    Returns:
        The same ``feat`` (unchanged for non-dict entries).
    """
    if not isinstance(feat, dict):
        return feat
    oc = feat.get("output_column")
    if feat.get("function") not in (None, "") and isinstance(oc, dict):
        for _k in ("type", "element_type", "length"):
            oc.pop(_k, None)
    sc = feat.get("source_column")
    if isinstance(sc, dict):
        _strip_runtime_stamped_columns([sc])
    return feat


def _strip_secondary_key_derivations(inner: dict[str, Any]) -> None:
    """Drop snowml-core's secondary-key ``FROM SPECIFICATION`` derivations so an
    operator-authored SK feature view and its ``snow feature export`` recovery
    hash identically (stops the planner looping on ``RECREATE_FV``).  SK-gated
    no-op; the SK identity and real aggregation fields are preserved so genuine
    edits still bump the hash.  See the module comment above for the three server
    rewrites this reverses.

    The SK-membership comparisons here are the read/hash-path mirror of the
    exporter's write-path SK filters (``exporter._secondary_key_columns`` /
    ``exporter._authoring_features``), and share their canonical
    ``to_comparison_identifier`` fold.  The case-insensitive match is
    intentional (the DESIGN's "comparison form"), not a leftover of the
    ``.upper()`` -> stored-form identity migration: the SK *identity* is
    preserved in stored form via ``aggregation_secondary_keys`` so a genuine
    mixed-case SK rename still bumps the hash.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV/StreamingFV spec_payload,
            mutated in place.  Non-dict inputs and SK-absent specs are no-ops.
    """
    if not isinstance(inner, dict):
        return
    sk_set = _secondary_key_set(inner)
    if not sk_set:
        return
    _strip_sk_from_entities(inner, sk_set)  # (1)
    features = inner.get("features")
    if not isinstance(features, list):
        return
    inner["features"] = [
        _strip_server_derived_agg_output(feat)  # (2) + runtime length
        for feat in features
        if not _is_synthesised_sk_array_feature(feat, sk_set)  # (3)
    ]


def _canonical_function_spelling(fn: Any) -> Any:
    """Map the specification stddev spelling to the imperative wire token.

    ``stddev`` becomes ``std``, case-insensitively; anything else passes
    through, so callers can compare function tokens without treating the two
    spellings as a change.

    Args:
        fn: A raw ``function`` token, or any value.

    Returns:
        ``std`` for ``stddev``, otherwise the value unchanged.
    """
    if isinstance(fn, str) and fn.lower() == "stddev":
        return "std"
    return fn


def _normalise_feature_function_spelling(features: Any) -> None:
    """Canonicalize the aggregation function spelling in a features list.

    The imperative wire token for standard deviation is ``std``, but
    Snowflake's specification output reports the SQL spelling ``stddev``.
    Rewriting it on both halves keeps the round-trip clean without changing the
    wire form.  A no-op on every other token.

    Args:
        features: The ``features`` list of a tiled FV inner spec, mutated in
            place.  Non-list and non-dict entries are skipped.
    """
    if not isinstance(features, list):
        return
    for feat in features:
        if not isinstance(feat, dict):
            continue
        fn = feat.get("function")
        if isinstance(fn, str):
            feat["function"] = _canonical_function_spelling(fn)


def _drop_empty_function_params(features: Any) -> None:
    """Drop empty / ``None`` ``function_params`` (in place) so an authored
    ``{}`` hashes equal to the applied recovery that omits the key. A populated
    dict is preserved so a real parameter change is still detected.

    Args:
        features: The ``features`` list of a tiled FV inner spec (mutated in
            place).  Non-list / non-dict entries are skipped.
    """
    if not isinstance(features, list):
        return
    for feat in features:
        if not isinstance(feat, dict) or "function_params" not in feat:
            continue
        fp = feat.get("function_params")
        if fp is None or (isinstance(fp, dict) and not fp):
            feat.pop("function_params", None)


# Operational FV-level keys that influence registration semantics but
# never appear in the deployed ``DESCRIBE … TYPE = SPECIFICATION``
# payload.  Stripped before hashing so a local YAML that adds (or
# removes) a backfill block does not falsely flag a deployed FV as
# changed — the planner surfaces backfill intent through the
# destructive flag on the plan op instead.
#
# The FV-level ``backfill`` block is the canonical entry; the
# legacy stream-source-level ``backfill_table`` field has been
# removed entirely (any spec carrying it raises a migration error
# at the ``StreamingSource`` validator).  ``backfill_table`` is kept
# in this set as a defensive belt-and-suspenders for any pre-Phase-A
# applied state that still carries it on stream-source rows.
_OPERATIONAL_FV_KEYS: frozenset[str] = frozenset(
    {
        "backfill",
        "backfill_table",
        # ``warehouse`` is an *operational* knob — the imperative
        # ``FeatureStore.update_feature_view(warehouse=...)`` accepts
        # in-place changes, so an isolated warehouse edit must not bump
        # the structural hash.  Drift is detected by the planner's
        # ``_batch_fv_operational_drift`` helper, which routes the
        # change to ``OpKind.UPDATE_FV`` instead of ``RECREATE_FV``.
        "warehouse",
        # DT refresh cadence (``refresh_freq``) and OFT staleness
        # (``target_lag``) are both *operational* after the
        # ``feature_granularity`` / ``refresh_freq`` / ``target_lag``
        # decoupling — each has an in-place imperative path
        # (``FeatureStore.update_feature_view(refresh_freq=…)`` and
        # ``ALTER ONLINE FEATURE TABLE … SET TARGET_LAG`` respectively),
        # and operational drift is detected independently by
        # ``planner._batch_fv_operational_drift``.  The applied-state
        # side carries ``refresh_freq`` as an authoring-form sibling of
        # the wire-form ``target_lag_sec`` (see
        # ``state._inject_fv_refresh_freq_from_list_row``); stripping it
        # from the structural hash keeps the two sides symmetric.
        # ``target_lag`` is the rare authoring shape (most flows
        # normalise to ``target_lag_sec`` upstream), included here for
        # defensive parity.
        "refresh_freq",
        "target_lag",
        # ``desc`` (and the authoring sibling ``description``) is also
        # an *operational* knob — ``FeatureStore.update_feature_view(
        # desc=...)`` rewrites the deployed DT's ``COMMENT`` in place
        # via ``ALTER DYNAMIC TABLE`` (see ``_build_offline_update_queries``).
        # The planner's desc drift detection runs through
        # ``planner._desc_drifted`` (operational fast path), so the
        # structural hash must strip both keys on BOTH sides:
        # ``compile_to_spec`` drops authoring ``description`` entirely
        # so the local side never carries it, but
        # ``state._inject_batch_fv_fields_from_list_row`` now injects
        # the deployed ``desc`` at top level of the applied
        # ``spec_payload`` so ``planner._resolve_applied_desc`` can
        # close the L3 round-trip (``UPDATE_FV`` → ``NO_CHANGE`` on
        # re-plan after ``ALTER DT … SET COMMENT``).  Without this
        # strip, an injected applied-side ``desc`` would bump the
        # structural hash and the planner would emit ``RECREATE_FV``
        # — the canonicalization gap that previously masked the L3
        # operational round-trip.
        "desc",
        "description",
    }
)

# Default ``length`` values Snowflake stamps onto STRING-family columns
# in the SPECIFICATION payload that the local-compile path leaves out.
# Stripped from each ``columns[]`` / ``output_columns[]`` entry before
# hashing.  ``16777216`` is the VARCHAR maximum (2^24) and is the value
# the runtime defaults to for un-sized StringType columns, so its
# presence on the deployed side is not a semantic difference.
_RUNTIME_STAMPED_COLUMN_DEFAULTS: dict[str, object] = {
    "length": 16777216,
}


def _strip_runtime_stamped_columns(columns: object) -> None:
    """Remove runtime-default keys from a columns or output_columns list.

    Walks the list in place, leaving non-dict entries alone.  A key is dropped
    only when its value matches the documented runtime default, so an
    explicitly authored value still contributes to the hash.

    Args:
        columns: The list value of a ``columns`` or ``output_columns`` key.
            Any non-list is a no-op.
    """
    if not isinstance(columns, list):
        return
    for entry in columns:
        if not isinstance(entry, dict):
            continue
        for key, default in _RUNTIME_STAMPED_COLUMN_DEFAULTS.items():
            if entry.get(key) == default:
                entry.pop(key, None)


def _is_auto_derived_feature(feature: Any) -> bool:
    """True when a feature entry is a 1:1 source-to-output pass-through.

    Snowflake auto-derives this shape from the source columns at create time
    for any non-tiled view that omits an explicit ``features:`` block, so the
    deployed specification returns entries the authoring YAML never had.
    Stripping them keeps the local and applied hashes in sync on a clean
    round-trip.  Explicit aggregations carry extra keys, so they survive and
    the planner still sees real semantic edits.

    Args:
        feature: A feature dict from the inner spec.

    Returns:
        True when the source and output column names are both present and
        equal, and the entry carries no other keys.  Any extra key means the
        feature holds semantic content that must influence the hash.
    """
    if not isinstance(feature, dict):
        return False
    oc = feature.get("output_column")
    sc = feature.get("source_column")
    if not (isinstance(oc, dict) and isinstance(sc, dict)):
        return False
    oc_name = (oc.get("name") or "").upper()
    sc_name = (sc.get("name") or "").upper()
    if not oc_name or oc_name != sc_name:
        return False
    # An auto-derived feature carries ONLY ``source_column`` and
    # ``output_column``.  Any extra key — ``function``, ``window_sec``,
    # ``feature_aggregation_method``, ``feature_granularity[_sec]``,
    # ``window``, etc. — represents real semantic content that must
    # bump the hash.  We treat ``output_column.type``  ≠ ``source_column.type``
    # as auto-derived too because snowml-core stamps deployed column types
    # (DoubleType vs FloatType, length defaults, etc.) that the local
    # YAML cannot match — those drift through the per-column runtime
    # stripping already in :func:`_full_spec_hash`.
    extra_keys = set(feature.keys()) - {"source_column", "output_column"}
    return not extra_keys


# FeatureView ``kind`` strings whose ``spec.sources`` / ``spec.features``
# need the BatchFV parity normalisation in :func:`_full_spec_hash`.
# Keeping the set explicit (rather than ``"FeatureView" in kind``) so a
# new FV kind opts in deliberately — the normalisation drops ``name`` /
# ``columns`` / ``source_type`` from each source binding, which is
# tolerable for BatchFV (the deployed SPECIFICATION lost them too) but
# would lose real signal on a kind that preserves them end-to-end.
_FV_HASH_NORMALISE_KINDS: frozenset[str] = frozenset(
    {
        "BatchFeatureView",
        "StreamingFeatureView",
        "RealtimeFeatureView",
    }
)


def _canonicalize_operational_for_drift(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize the operational-drift comparison surface.

    Wraps the imperative canonicalizer behind a lazy import so this package
    does not pull the feature_store module into its initial import graph.  Two
    fields are canonicalized symmetrically across both halves of the diff so a
    clean replan does not emit a phantom update:

    * ``online_store_type`` — mixed-case or padded enum values from the
      deployed specification collapse to the local lowercase form.
    * ``target_lag_sec`` — "5 minutes" and "300 SECONDS" both reduce to a
      number of seconds, so the diff is stable whichever form the table
      happened to return.

    The downstream sentinel used by cron-scheduled tasks is skipped: the
    planner recovers the cron expression from the companion task, and the
    interval parser would crash on it.

    Pure — the input is not mutated, so callers can chain other normalisers.
    If the imperative helper is unavailable the fallback canonicalizes the
    store type only, which is the conservative choice: a missed
    canonicalization shows up as a benign update rather than a silent miss.

    Args:
        payload: A dict carrying any subset of ``online_store_type``,
            ``target_lag`` and ``target_lag_sec``.  Other keys pass through.

    Returns:
        A new dict with the canonicalized values.
    """
    try:
        from snowflake.ml.feature_store.feature_store import (
            _canonicalize_operational_fields,
        )

        return _canonicalize_operational_fields(payload)
    except Exception:  # noqa: BLE001 — defensive fallback; see docstring
        out = dict(payload)
        if isinstance(out.get("online_store_type"), str):
            out["online_store_type"] = out["online_store_type"].strip().lower()
        return out


def _normalise_fv_sources_for_hash(sources: Any) -> list[dict[str, str]]:
    """Project each FV source binding down to its stable identifier.

    The deployed specification either drops ``sources`` entirely, as it does for
    batch views, or keeps a subset of fields that does not line up with the
    local shape.  The stable identity is the binding, preferred in order: the
    table, then the canonicalised query body, then the name.

    A query-backed source binds on its SQL body because the two sides disagree
    on the name — the recovery path synthesises one while the author writes
    their own.  Both sides canonicalise the SQL, so the strings converge.
    Projecting this way keeps the binding in the hash without false positives
    from name or column drift on the deployed side, and the result is sorted so
    ordering differences between recovery and authoring do not matter.

    Args:
        sources: The ``sources`` value, of any type.

    Returns:
        A sorted list of single-key binding dicts, empty when no source carries
        a table, query, or name.
    """
    if not isinstance(sources, list):
        return []
    out: list[dict[str, str]] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        table = src.get("table")
        query = src.get("query")
        name = src.get("name")
        binding = ""
        if isinstance(table, str) and table.strip():
            binding = table.strip().upper()
        elif isinstance(query, str) and query.strip():
            binding = "QUERY:" + canonicalize_sql_for_hash(query)
        elif isinstance(name, str) and name.strip():
            binding = name.strip().upper()
        if not binding:
            continue
        out.append({"binding": binding})
    out.sort(key=lambda s: s["binding"])
    return out


def _resolve_hash_identifier(value: Any) -> Any:
    """Canonicalise a single SQL identifier the way snowml-core does.

    A thin wrapper over :func:`identifier.resolve_identifier` (unquoted folds to
    upper-case, ``"quoted"`` keeps its case).  Non-string / empty / parser-
    rejected values are returned unchanged so hashing never raises.  See the
    "SQL-identifier hash canonicalization" section of ``DESIGN.md``.

    Args:
        value: A single identifier cell (typically a ``str``).

    Returns:
        The core-resolved identifier string, or ``value`` unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return identifier.resolve_identifier(value.strip())
    except ValueError:
        return value


def _to_stored_form(value: Any, *, authored: bool) -> Any:
    """Canonicalise one identifier to Snowflake's bare stored form.

    The authored half is SQL syntax, so it resolves first (``userId`` ->
    ``USERID``, ``"userId"`` -> ``userId``).  The applied half is already
    stored, so it is only unquoted -- never case-folded, which keeps it
    idempotent.  Parser-rejected values pass through so hashing never raises.

    Args:
        value: A single identifier cell (typically a ``str``).
        authored: ``True`` to resolve SQL syntax first, ``False`` to unquote only.

    Returns:
        The bare stored-form identifier, or *value* unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    text = _resolve_hash_identifier(value) if authored else value.strip()
    try:
        return identifier.get_unescaped_names(text)
    except ValueError:
        return text


def _canonicalise_column_list(columns: Any, *, authored: bool) -> None:
    """In-place resolve the ``name`` of every column dict in a list.

    Args:
        columns: A list of ``{"name": ..., "type": ...}`` column dicts.  No-op
            when not a list; non-dict entries (and dicts without a ``name``) are
            skipped.
        authored: Threaded to :func:`_to_stored_form`.
    """
    if not isinstance(columns, list):
        return
    for col in columns:
        if isinstance(col, dict) and "name" in col:
            col["name"] = _to_stored_form(col.get("name"), authored=authored)


def _canonicalise_inner_spec_identifiers(inner: Any, *, authored: bool) -> None:
    """Resolve every identifier-bearing field of an inner ``spec`` block, in place.

    Shared by :func:`_canonicalise_spec_identifiers` (full-spec hash) and the
    structural-equivalence projection in
    :func:`batch_feature_view_structural_equivalent` so both apply the identical
    per-field contract.

    Args:
        inner: The ``spec`` sub-dict (mutated in place).  No-op when not a dict.
        authored: Threaded to :func:`_to_stored_form`.
    """
    if not isinstance(inner, dict):
        return
    for key in (
        "ordered_entity_column_names",
        "cluster_by",
        "aggregation_secondary_keys",
        "ordered_secondary_key_column_names",
    ):
        values = inner.get(key)
        if isinstance(values, list):
            inner[key] = [_to_stored_form(v, authored=authored) for v in values]
    if "timestamp_field" in inner:
        inner["timestamp_field"] = _to_stored_form(inner.get("timestamp_field"), authored=authored)
    features = inner.get("features")
    if isinstance(features, list):
        for feat in features:
            if isinstance(feat, dict):
                _canonicalise_column_list([feat.get("source_column"), feat.get("output_column")], authored=authored)
    _canonicalise_column_list(inner.get("output_columns"), authored=authored)
    sources = inner.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                _canonicalise_column_list(source.get("columns"), authored=authored)
    udf = inner.get("udf")
    if isinstance(udf, dict):
        # ``udf.name`` is a Python function name, not a SQL identifier — the
        # live SPECIFICATION round-trips it lower-case, so only columns resolve.
        _canonicalise_column_list(udf.get("output_columns"), authored=authored)


def _canonicalise_spec_identifiers(cleaned: dict[str, Any], *, authored: bool) -> None:
    """Resolve every identifier-bearing field of a spec dict, in place.

    Must run *first* in :func:`_normalize_for_full_spec_hash`, before any
    downstream ``.upper()`` comparison forms a second, divergent opinion.  The
    authored half is SQL syntax so it resolves to the stored form; the applied
    half is already stored so it only unquotes (see :func:`_to_stored_form`).
    ``metadata.version`` is not an identifier (``FeatureViewVersion`` permits
    ``-`` / ``.``) and folds symmetrically with ``.upper()`` on both halves,
    matching :func:`spec_key`.  See the "SQL-identifier hash canonicalization"
    section of ``DESIGN.md`` for the per-field table.

    Args:
        cleaned: A deep-copied spec dict (mutated in place).  Tolerant of
            missing keys / unexpected shapes so it never raises during hashing.
        authored: ``True`` for the local-compile (authored) half, ``False`` for
            the applied-recovery half.
    """
    if not isinstance(cleaned, dict):
        return
    metadata = cleaned.get("metadata")
    if isinstance(metadata, dict):
        for key in ("database", "schema", "name"):
            if key in metadata:
                metadata[key] = _to_stored_form(metadata.get(key), authored=authored)
        if isinstance(metadata.get("version"), str):
            metadata["version"] = metadata["version"].upper()
    _canonicalise_inner_spec_identifiers(cleaned.get("spec"), authored=authored)


def _normalize_for_full_spec_hash(spec: dict[str, Any], *, authored: bool = False) -> dict[str, Any]:
    """Return a copy of *spec* with hash-irrelevant fields stripped.

    This is the payload the full-spec hash covers.  Diagnostic diffs and any
    other hash-aligned comparison should call this rather than re-implementing
    the skip set, so they report the same fields the planner hashes.

    Volatile metadata is removed so a tool-version bump is not read as a
    change, and spec-level and per-column runtime defaults go too, so a
    deploy-time stamp does not look like drift.  The remaining metadata fields
    (name, version, database, schema) are kept.

    For feature-view kinds the inner source list is projected down to a sorted
    binding shape and auto-derived 1:1 features are dropped, since the deployed
    specification cannot reproduce either faithfully.

    Args:
        spec: A compiled-spec dict (e.g. the output of
            :func:`spec_compiler.compile_to_spec` or the JSON returned by
            ``DESCRIBE ... TYPE = SPECIFICATION``).
        authored: Threaded to :func:`_canonicalise_spec_identifiers` (``True``
            resolves the authored half to stored form; ``False`` unquotes only).

    Returns:
        A deep copy with volatile, derived, operational and runtime-stamped
        fields removed.  Does not mutate the input.
    """
    cleaned: dict[str, Any] = json.loads(json.dumps(spec, default=str))  # deep copy via JSON round-trip
    # Resolve SQL identifiers FIRST, before any downstream ``.upper()``
    # comparison.  The authored half is resolved to the stored form; the
    # applied half is only unquoted (see DESIGN.md).
    _canonicalise_spec_identifiers(cleaned, authored=authored)
    kind = cleaned.get("kind") if isinstance(cleaned, dict) else ""
    metadata = cleaned.get("metadata") if isinstance(cleaned, dict) else None
    if isinstance(metadata, dict):
        for key in _VOLATILE_METADATA_KEYS:
            metadata.pop(key, None)
    if isinstance(cleaned, dict):
        for key in _DERIVED_TOP_LEVEL_KEYS:
            cleaned.pop(key, None)
        # Strip the FV-level ``backfill`` block (and the legacy
        # ``backfill_table`` field) at every nesting level: authoring
        # dicts carry it at the top, compiled / DESCRIBE-derived dicts
        # may carry it (or never) inside ``spec``, and stream-source
        # entries inside ``sources[]`` carried the legacy field.
        for key in _OPERATIONAL_FV_KEYS:
            cleaned.pop(key, None)
    inner = cleaned.get("spec") if isinstance(cleaned, dict) else None
    if isinstance(inner, dict):
        for key in _RUNTIME_STAMPED_SPEC_KEYS:
            inner.pop(key, None)
        for key in _OPERATIONAL_FV_KEYS:
            inner.pop(key, None)
        # Normalise the trivial-default ``storage_config: {format:
        # snowflake}`` away.  The applied-state recovery for an *offline*
        # BatchFV does not emit ``storage_config`` (Phase 5's recovery
        # path is gated on ``SHOW ONLINE FEATURE TABLES``), so a local
        # YAML that authors the default explicitly (or omits it) hashes
        # identically only after we drop the all-default block on both
        # sides.  Iceberg / per-volume configs survive untouched.
        sc = inner.get("storage_config")
        if isinstance(sc, dict):
            # All non-None values must be the trivial default.
            non_none = {k: v for k, v in sc.items() if v is not None}
            if non_none == {"format": "snowflake"} or non_none == {}:
                inner.pop("storage_config", None)
        # ``feature_aggregation_method`` is a decl-side authoring marker
        # (the ``BATCH_FV_TILING_AGG_METHOD`` validator requires it on
        # tiled BFVs) but snowml-core's BFV constructor rejects the
        # same kwarg (``feature_view.py:974-977`` "feature_aggregation_method
        # is only supported for streaming feature views.").  The
        # imperative executor strips it before forwarding, so the
        # field never round-trips through DESCRIBE TYPE = SPECIFICATION
        # for BFVs.  Strip it from the structural hash so a local
        # ``feature_aggregation_method: tiles`` does not diverge from
        # the applied-state hash where it is necessarily absent.
        if kind == "BatchFeatureView":
            inner.pop("feature_aggregation_method", None)
        sources_for_strip = inner.get("sources")
        if isinstance(sources_for_strip, list):
            for src in sources_for_strip:
                if isinstance(src, dict):
                    for key in _OPERATIONAL_FV_KEYS:
                        src.pop(key, None)
        # Strip runtime-default ``length`` (and any future
        # ``_RUNTIME_STAMPED_COLUMN_DEFAULTS`` keys) from every
        # ``columns`` / ``output_columns`` list reachable from
        # ``spec``.  Walk one level deep — sources[].columns and
        # udf.output_columns are the two known carriers; the
        # ``_strip_runtime_stamped_columns`` helper is a no-op on
        # non-list values so the walk stays defensive.
        _strip_runtime_stamped_columns(inner.get("output_columns"))
        sources = inner.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    _strip_runtime_stamped_columns(source.get("columns"))
        udf = inner.get("udf")
        if isinstance(udf, dict):
            _strip_runtime_stamped_columns(udf.get("output_columns"))
        # BatchFV parity normalisation (see docstring).
        if kind in _FV_HASH_NORMALISE_KINDS:
            # Reconcile snowml-core's secondary-key round-trip transforms (SK
            # appended to entity cols, ArrayType-wrapped aggregation outputs,
            # synthesised per-window SK-array features).  SK-gated no-op otherwise.
            if kind in ("BatchFeatureView", "StreamingFeatureView"):
                _strip_secondary_key_derivations(inner)
            inner["sources"] = _normalise_fv_sources_for_hash(inner.get("sources"))
            features = inner.get("features")
            if isinstance(features, list):
                inner["features"] = [f for f in features if not _is_auto_derived_feature(f)]
            # Fold the applied wire key ``ordered_secondary_key_column_names``
            # into the authoring key ``aggregation_secondary_keys`` so both sides
            # hash identically (value preserved, so a real SK change still bumps).
            wire_sk = inner.pop("ordered_secondary_key_column_names", None)
            if wire_sk and "aggregation_secondary_keys" not in inner:
                inner["aggregation_secondary_keys"] = wire_sk
            # Canonicalize the stddev spelling ("stddev" -> "std") across
            # batch + streaming tiled features so the SPECIFICATION spelling
            # does not diverge from the imperative wire token when hashing.
            _normalise_feature_function_spelling(inner.get("features"))
            # Coalesce an explicit empty ``function_params={}`` (kept by
            # ``model_to_dict(exclude_none=True)``) with the applied recovery
            # that omits the key for param-less aggregations (e.g. ``count``),
            # so an unchanged project does not loop on RECREATE_FV.
            _drop_empty_function_params(inner.get("features"))
        # BatchFV-specific operational-default stripping.  snowml-core
        # stamps ``initialize: ON_CREATE``, ``refresh_mode: AUTO``, and a
        # default ``cluster_by`` (entities, or entities + TILE_START for
        # tiled BFVs) onto the deployed Dynamic Table at create time.
        # Recovery from DT text injects them onto the applied side; the
        # local-compile side omits them unless the YAML authored an
        # override.  Strip default values from both sides so the round-
        # trip is idempotent.  Type promotion (IntegerType -> LongType,
        # FloatType -> DoubleType) reflects Snowflake's storage widening
        # and is applied symmetrically on both halves.
        if kind == "BatchFeatureView":
            _strip_default_operational_fields(inner)
            _strip_default_cluster_by(inner)
            _strip_default_append_only(inner)
            _normalise_feature_column_types(inner.get("features"))
    return cleaned


# Back-compat alias: the streaming FV replan drift walker imports this name and
# must hash through the identical normalisation the planner uses, so it aliases
# :func:`_normalize_for_full_spec_hash` rather than re-implementing the skip set.
_canonical_spec_for_hash = _normalize_for_full_spec_hash


def _full_spec_hash(spec: dict[str, Any], *, authored: bool = False) -> str:
    """SHA-256 over the full spec JSON, with stable key ordering.

    Used by the planner for full-spec diffs when the applied object came from a
    specification DESCRIBE.  The hashed payload is the normalised one, with
    volatile metadata, runtime-stamped keys and operational fields removed.

    Args:
        spec: A compiled-spec dict (e.g. the output of
            :func:`spec_compiler.compile_to_spec` or the JSON returned by
            ``DESCRIBE ... TYPE = SPECIFICATION``).
        authored: Threaded to :func:`_normalize_for_full_spec_hash`.  The
            authored and applied hashes match only when the authoring's stored
            form equals the deployed identifier.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(_normalize_for_full_spec_hash(spec, authored=authored), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Authored (local) vs applied half of the full-spec hash
# ---------------------------------------------------------------------------
# The full-spec hash compares two halves: the *authored* (local) spec, freshly
# compiled from the project files, and the *applied* spec recovered from
# ``DESCRIBE ... TYPE = SPECIFICATION``.  On this slice the two halves are
# normalised identically, so these are thin forwarders.  They exist as the
# *single* entry point for each half so that when jkew/decl-fs/6e1.1 lands its
# SQL-identifier canonicalisation (which makes the two halves asymmetric:
# ``authored=True`` resolves SQL syntax to Snowflake's stored form, ``False``
# only unquotes), the authored/applied distinction is threaded **here only** --
# at ``6e-int`` -- rather than silently defaulting to applied semantics on the
# authored side.  Keeping every caller routed through these forwarders is why
# the ``6e-int`` merge surfaces a visible conflict instead of a wrong default.
def _normalize_local_half(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalise the *authored* (local) half of the full-spec hash.

    Args:
        spec: The compiled local spec dict (``compile_to_spec`` output).

    Returns:
        The normalised spec dict ready for hashing / property diffing.
    """
    return _normalize_for_full_spec_hash(spec, authored=True)


def _normalize_applied_half(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalise the *applied* (deployed) half of the full-spec hash.

    Args:
        spec: The applied spec dict recovered from ``DESCRIBE ... TYPE =
            SPECIFICATION``.

    Returns:
        The normalised spec dict ready for hashing / property diffing.
    """
    return _normalize_for_full_spec_hash(spec, authored=False)


def _hash_local_half(spec: dict[str, Any]) -> str:
    """Hash the *authored* (local) half of the full-spec hash.

    Args:
        spec: The compiled local spec dict (``compile_to_spec`` output).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    return _full_spec_hash(spec, authored=True)


def compute_local_spec_hash(
    model_dict: dict[str, Any],
    database: str,
    schema: str,
    *,
    entity_join_keys: Mapping[str, list[str]] | None = None,
) -> str:
    """Compile a local FV spec dict and return its full-spec hash.

    The local dict is run through the spec compiler so it ends up with the same
    shape as a specification DESCRIBE, then hashed — so a deployed spec and a
    local one produce identical digests when their semantics match.

    Only meaningful for FeatureView kinds; Entity and source kinds should keep
    using the structural fingerprint hash.

    Args:
        model_dict: Spec dict from the model serializer.
        database: Snowflake database name.
        schema: Snowflake schema name.
        entity_join_keys: Optional entity-name to join-key-columns map, passed
            to the compiler so the entity column wire field matches the applied
            side.  When omitted the authored entity list compiles verbatim,
            preserving hashes for callers that do not supply the map.

    Returns:
        64-character lowercase hex SHA-256 digest of the compiled spec.
    """
    # Imported locally to avoid widening the import surface of this module
    # at package-load time.
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    compiled = compile_to_spec(model_dict, database, schema, entity_join_keys=entity_join_keys)
    # Route through the authored-half entry point so 6e1.1's identifier
    # canonicalisation is threaded here only (see _hash_local_half).
    return _hash_local_half(compiled)


# Normalized ``spec.*`` property paths whose change *may* be a pure
# *transformation* edit (the source SQL body / table binding) rather than an
# output-schema change.  Membership here no longer *assumes* the edit is
# schema-preserving: a ``spec.sources`` delta is handed to
# :func:`_source_binding_evolution_kind`, which inspects the query SELECT list
# (a drop / reorder / retype projects a different schema and must recreate) and
# only returns ``"preserving"`` for a table retarget or a predicate-only query
# rewrite.  ``spec.sources`` is the compiled binding (projected to a
# ``[{table}]`` / ``[{query}]`` shape by :func:`_normalise_fv_sources_for_hash`
# for the hash), so a comment / whitespace-only edit never reaches here (it is
# already canonicalised to ``NO_CHANGE`` upstream).
_APPEND_ONLY_TRANSFORM_ONLY_PROPS: frozenset[str] = frozenset({"spec.sources"})

# Evolution-kind severity ordering (ascending).  ``_worst_evolution_kind``
# folds the per-section verdicts to the most destructive outcome.
_EVOLUTION_KIND_SEVERITY: dict[str, int] = {"preserving": 0, "extend_only": 1, "breaking": 2}


def _worst_evolution_kind(*kinds: str) -> str:
    """Return the most destructive of *kinds* (``breaking`` > ``extend_only`` > ``preserving``).

    Args:
        *kinds: Evolution-kind labels to fold.

    Returns:
        The label with the highest severity; ``"preserving"`` when no args.
    """
    worst = "preserving"
    for kind in kinds:
        if _EVOLUTION_KIND_SEVERITY.get(kind, 2) > _EVOLUTION_KIND_SEVERITY.get(worst, 0):
            worst = kind
    return worst


def _raw_fv_sources(spec: Any) -> list[dict[str, Any]]:
    """Return the un-normalised ``spec.sources`` list of a compiled / applied spec.

    Reads the raw binding entries (``name`` / ``table`` / ``query`` / ...)
    *before* :func:`_normalise_fv_sources_for_hash` projects them to the
    ``[{binding}]`` hash shape, so the SELECT-list analysis can see the actual
    SQL body.

    Args:
        spec: A compiled-local or applied spec dict.

    Returns:
        The ``spec.sources`` list of dicts, or an empty list when absent.
    """
    inner = spec.get("spec") if isinstance(spec, dict) else None
    sources = inner.get("sources") if isinstance(inner, dict) else None
    if not isinstance(sources, list):
        return []
    return [s for s in sources if isinstance(s, dict)]


def _pair_sources_by_binding(
    local_sources: list[dict[str, Any]],
    applied_sources: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    """Pair local and applied source entries for a one-to-one binding comparison.

    Pairs by uppercased ``name`` when both sides carry a complete, distinct
    name set that matches; otherwise by position when the two lists are the same
    length (the author-side and recovery-side logical names can differ for a
    query-backed source).  Returns ``None`` when neither strategy applies, which
    the caller treats as ``"breaking"``.

    Args:
        local_sources: Raw local ``spec.sources`` entries.
        applied_sources: Raw applied ``spec.sources`` entries.

    Returns:
        A list of ``(local, applied)`` pairs, or ``None`` when the sources
        cannot be aligned.
    """
    if not local_sources or not applied_sources:
        return None

    def _name_index(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        names = [str(s.get("name") or "").strip().upper() for s in sources]
        if all(names) and len(set(names)) == len(names):
            return dict(zip(names, sources))
        return None

    local_by_name = _name_index(local_sources)
    applied_by_name = _name_index(applied_sources)
    if local_by_name is not None and applied_by_name is not None and set(local_by_name) == set(applied_by_name):
        return [(local_by_name[name], applied_by_name[name]) for name in local_by_name]
    if len(local_sources) == len(applied_sources):
        return list(zip(local_sources, applied_sources))
    return None


def _binding_pair_evolution_kind(local_src: dict[str, Any], applied_src: dict[str, Any]) -> str:
    """Classify a single local-vs-applied source binding edit.

    * A table binding on *both* sides is ``"preserving"`` — the new table's
      schema is unknowable at plan time and a retarget is the in-place happy path.
    * A table ↔ query flip is ``"breaking"``: the query side's output schema is
      visible, so it is not waved through like a pure table retarget.
    * Query-vs-query: compare the proven SELECT lists.  Equal → ``"preserving"``;
      the applied list a strict prefix of the local list → ``"extend_only"``;
      any reorder / drop / retype → ``"breaking"``.  When either SELECT list
      cannot be proven, fall back to the predicate-strip proof: identical
      ``WHERE``-free bodies prove the projection is unchanged (``"preserving"``),
      otherwise ``"breaking"``.

    Args:
        local_src: One raw local ``spec.sources`` entry.
        applied_src: The paired raw applied ``spec.sources`` entry.

    Returns:
        One of ``"preserving"``, ``"extend_only"``, ``"breaking"``.
    """
    local_query = local_src.get("query")
    applied_query = applied_src.get("query")
    local_has_query = isinstance(local_query, str) and bool(local_query.strip())
    applied_has_query = isinstance(applied_query, str) and bool(applied_query.strip())
    if local_has_query != applied_has_query:
        return "breaking"  # table ↔ query flip exposes a visible output schema
    if not (isinstance(local_query, str) and local_query.strip()):
        return "preserving"  # table → table retarget (schema unknowable at plan time)
    if not (isinstance(applied_query, str) and applied_query.strip()):
        return "preserving"  # defensive: both sides are queries past the flip check

    local_cols = extract_select_output_columns(local_query)
    applied_cols = extract_select_output_columns(applied_query)
    if local_cols is not None and applied_cols is not None:
        if local_cols == applied_cols:
            return "preserving"
        if len(local_cols) > len(applied_cols) and local_cols[: len(applied_cols)] == applied_cols:
            return "extend_only"
        return "breaking"

    # Unprovable SELECT list — fall back to the predicate-strip equality proof.
    local_stripped = strip_sql_predicates(local_query)
    applied_stripped = strip_sql_predicates(applied_query)
    if local_stripped is not None and local_stripped == applied_stripped:
        return "preserving"
    return "breaking"


def _source_binding_evolution_kind(local_compiled: dict[str, Any], applied_payload: dict[str, Any]) -> str:
    """Classify the ``spec.sources`` delta of an append-only BatchFeatureView edit.

    Reads the raw (pre-hash-normalisation) source bindings from both sides,
    pairs them (:func:`_pair_sources_by_binding`), and folds the per-pair
    verdicts (:func:`_binding_pair_evolution_kind`) to the most destructive
    outcome.  Unalignable source lists are ``"breaking"``.

    Args:
        local_compiled: The compiled local spec dict (``compile_to_spec``).
        applied_payload: The deployed ``AppliedObject.spec_payload``.

    Returns:
        One of ``"preserving"``, ``"extend_only"``, ``"breaking"``.
    """
    local_sources = _raw_fv_sources(local_compiled)
    applied_sources = _raw_fv_sources(applied_payload)
    pairs = _pair_sources_by_binding(local_sources, applied_sources)
    if pairs is None:
        return "breaking"
    return _worst_evolution_kind(*(_binding_pair_evolution_kind(loc, app) for loc, app in pairs))


def _strip_default_append_only(inner: dict[str, Any]) -> None:
    """Drop ``append_only`` from a BatchFV inner spec when it is the default.

    The local compiler only emits ``append_only`` when the operator authored
    ``true``, but a deployed ``DESCRIBE … TYPE = SPECIFICATION`` payload (or a
    ``list_feature_views`` row) may stamp the snowml-core default ``False``.
    Strip the key whenever its value is falsy so a plain BFV hashes identically
    on the local and applied sides, and an append-only BFV (``True``) still
    contributes to the structural hash.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV spec_payload (mutated in
            place).  Non-dict inputs are no-ops.
    """
    if not isinstance(inner, dict):
        return
    if "append_only" in inner and not inner.get("append_only"):
        inner.pop("append_only", None)


def append_only_schema_evolution_kind(
    local_compiled: dict[str, Any],
    applied_payload: dict[str, Any],
) -> str:
    """Classify a hash-differing append-only BatchFeatureView edit.

    Called by the planner *after* the full-spec hash has already been found to
    differ, to decide whether the edit can be applied in place (preserving the
    ``$SNAPSHOTS`` history via ``EVOLVE_FV``) or requires a destructive
    ``RECREATE_FV``.  Both sides are normalised with the same
    :func:`_normalize_for_full_spec_hash` /
    :func:`_normalize_applied_bfv_for_hash` pipeline used for the hash itself,
    then :func:`full_spec_changed_properties` reports which ``spec.*`` sections
    differ:

    * ``"preserving"`` — the difference is limited to the source binding and the
      binding analysis (:func:`_source_binding_evolution_kind`) proves the
      output schema is unchanged: a table retarget, or a query rewrite whose
      SELECT list is identical / a predicate-only edit.
    * ``"extend_only"`` — additionally a query-backed source's SELECT list
      gained new trailing columns (existing columns untouched); the executor
      materialises these because ``session.sql`` yields the new column and core
      extends the ``$SNAPSHOTS`` table in place.
    * ``"breaking"`` — any other structural difference (entities, timestamp,
      refresh mode, cluster-by, secondary keys, ``append_only`` toggle, **any**
      ``spec.features`` edit, **a query SELECT-list drop / reorder / retype**,
      …), or a normalisation failure.  The safe default: the caller emits the
      destructive ``RECREATE_FV``.

    A ``spec.features`` delta is always ``breaking``: a surviving feature entry
    on an append-only BFV is necessarily a rename (``function`` / ``window_sec``
    aggregations are rejected by the authoring validator because append-only
    forbids tiling), and ``imperative_executor._build_features`` drops a
    rename-only entry, so an EVOLVE_FV promise would re-emit on every replan and
    never converge.

    Args:
        local_compiled: The compiled local spec dict (``compile_to_spec``).
        applied_payload: The deployed ``AppliedObject.spec_payload``.

    Returns:
        One of ``"preserving"``, ``"extend_only"``, ``"breaking"``.
    """
    if not isinstance(local_compiled, dict) or not isinstance(applied_payload, dict):
        return "breaking"
    try:
        applied_for_norm = _normalize_applied_bfv_for_hash(applied_payload, local_compiled)
        # Authored vs applied half routed through the single entry points so
        # 6e1.1's identifier canonicalisation is threaded here only.
        local_norm = _normalize_local_half(local_compiled)
        applied_norm = _normalize_applied_half(applied_for_norm)
        changed = set(full_spec_changed_properties(local_norm, applied_norm))
    except Exception:  # noqa: BLE001 — defensive; never break the planner
        return "breaking"
    if not changed:
        # Nothing structural differs after normalisation.  The caller only
        # reaches the classifier on a hash mismatch, so this is a defensive
        # safe-to-evolve outcome (e.g. a pure canonicalisation gap).
        return "preserving"
    # Any section beyond the source binding is structural and always breaking
    # (entities, timestamp, cluster-by, append_only, secondary keys, and
    # ``spec.features``: a surviving feature entry is a rename the EVOLVE_FV
    # executor cannot materialise, so promising it would never converge).
    if changed - _APPEND_ONLY_TRANSFORM_ONLY_PROPS:
        return "breaking"
    # The only remaining difference is the ``spec.sources`` binding: inspect the
    # query SELECT list so a drop / reorder / retype recreates while a table
    # retarget, predicate-only rewrite, or trailing-column append evolves in
    # place.
    return _source_binding_evolution_kind(local_compiled, applied_payload)


def compute_local_spec_and_hash(
    model_dict: dict[str, Any],
    database: str,
    schema: str,
    *,
    entity_join_keys: Mapping[str, list[str]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Compile a local FV spec dict; return ``(compiled_spec, full_spec_hash)``.

    Like :func:`compute_local_spec_hash` but also surfaces the compiled dict
    so callers that need both the hash and the spec for a field-level diff
    avoid calling ``compile_to_spec`` a second time.

    Args:
        model_dict: Spec dict from :func:`model_to_dict`.
        database: Snowflake database name (used as ``metadata.database``).
        schema: Snowflake schema name (used as ``metadata.schema``).
        entity_join_keys: Optional entity-name → join-key-columns map
            forwarded to :func:`compile_to_spec` so the wire field
            ``ordered_entity_column_names`` matches the applied side.

    Returns:
        Tuple of ``(compiled_spec, hash_hex)`` where ``hash_hex`` is the
        64-character lowercase hex SHA-256 digest of the compiled spec.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    compiled = compile_to_spec(model_dict, database, schema, entity_join_keys=entity_join_keys)
    # Route through the authored-half entry point so 6e1.1's identifier
    # canonicalisation is threaded here only (see _hash_local_half).
    return compiled, _hash_local_half(compiled)


def full_spec_changed_properties(local_normalized: dict[str, Any], applied_normalized: dict[str, Any]) -> list[str]:
    """Return sorted dot-path property names that differ between two normalized specs.

    Both inputs must already be processed by :func:`_normalize_for_full_spec_hash`
    so the comparison is over the same canonical shape used for hashing.
    Compares the inner ``spec`` dict key-by-key and returns paths like
    ``"spec.udf"``, ``"spec.features"``, ``"spec.sources"``, ``"spec.entities"``.
    Top-level keys outside ``spec`` that differ are returned without a prefix.

    Args:
        local_normalized: Normalized local compiled spec dict.
        applied_normalized: Normalized applied (deployed) spec dict.

    Returns:
        Sorted list of dot-path property names for sections that differ.
        Empty list when the two dicts are identical after normalization.
    """
    changed: list[str] = []
    local_inner = local_normalized.get("spec") if isinstance(local_normalized, dict) else None
    applied_inner = applied_normalized.get("spec") if isinstance(applied_normalized, dict) else None

    if isinstance(local_inner, dict) and isinstance(applied_inner, dict):
        all_keys = set(local_inner) | set(applied_inner)
        for key in all_keys:
            if local_inner.get(key) != applied_inner.get(key):
                changed.append(f"spec.{key}")
    elif isinstance(local_normalized, dict) and isinstance(applied_normalized, dict):
        non_spec_keys = (set(local_normalized) | set(applied_normalized)) - {"spec"}
        for key in non_spec_keys:
            if local_normalized.get(key) != applied_normalized.get(key):
                changed.append(key)

    return sorted(changed)


def fg_content_hash(spec: dict[str, Any]) -> str:
    """SHA-256 over the FeatureGroup identity and sources tuple.

    The basis is deliberately narrow: the name, the version, the description,
    the auto-prefix flag that controls output column naming, and a sorted tuple
    of each source's name, version, slice columns and alias.  Sorting makes it
    order-independent, so reordering the YAML does not force a destructive
    recreate, while a version edit still does.

    Output columns are derived from the source views at register time and are
    not part of the basis.  Unrelated client-side fields are ignored, since the
    hash reads only the keys above.  An empty alias is preserved as such,
    meaning "no prefix", and stays distinct from an absent one.

    Args:
        spec: An FG spec dict from either authoring or applied state.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    name = (spec.get("name") or "").upper()
    version = (spec.get("version") or "").upper()
    desc = spec.get("desc") or ""
    auto_prefix = bool(spec.get("auto_prefix", True))

    sources: list[tuple[str, str, list[str] | None, str | None]] = []
    for fv_ref in spec.get("feature_views", []) or []:
        if not isinstance(fv_ref, dict):
            continue
        fv_name = (fv_ref.get("name") or "").upper()
        fv_version = (fv_ref.get("version") or "").upper()
        slice_cols = fv_ref.get("slice_columns")
        slice_canon: list[str] | None
        if slice_cols is None:
            slice_canon = None
        else:
            # Preserve the authoring order — slice order is semantically
            # significant on the imperative ``FeatureViewSlice``.
            slice_canon = [str(c) for c in slice_cols]
        alias = fv_ref.get("alias", None)
        # ``""`` survives as ``""``; ``None`` stays ``None`` (auto_prefix path).
        sources.append((fv_name, fv_version, slice_canon, alias))

    # Sort for reorder stability — the imperative side stores ``sources``
    # in declaration order, but a hash mismatch on YAML reorder would be
    # spurious (the FG schema is determined by the set of refs, not their
    # order on the page).  Sort by (fv_name, fv_version) which uniquely
    # identifies each ref under the "no duplicate (name, version)" invariant.
    sources_sorted = sorted(sources, key=lambda t: (t[0], t[1]))

    basis = {
        "name": name,
        "version": version,
        "desc": desc,
        "auto_prefix": auto_prefix,
        "sources": [
            {
                "fv_name": s[0],
                "fv_version": s[1],
                "slice_columns": s[2],
                "alias": s[3],
            }
            for s in sources_sorted
        ],
    }
    canonical = json.dumps(basis, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def model_to_dict(spec_obj: SpecBase) -> dict[str, Any]:
    """Serialize a SpecBase model to a canonical dict for hashing and comparison.

    Renames ``schema_`` to ``schema`` so the key matches the raw dicts the state
    module produces.

    Args:
        spec_obj: A SpecBase (or subclass) model instance.

    Returns:
        Canonical dict with ``schema_`` renamed to ``schema``.
    """
    d = spec_obj.model_dump(exclude_none=True)
    if "schema_" in d:
        d["schema"] = d.pop("schema_")
    return d


def _result(
    severity: str,
    code: str,
    message: str,
    object_name: str = "",
) -> ValidationResult:
    return ValidationResult(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        object_name=object_name,
    )


def _source_column_name(entry: dict[str, Any]) -> str:
    """Return the source-column name from a feature dict, or ``""``.

    Args:
        entry: A single ``features[]`` dict.

    Returns:
        The column name, or empty string when absent.
    """
    raw = entry.get("source_column", "")
    if isinstance(raw, dict):
        return str(raw.get("name") or "").strip()
    return str(raw or "").strip()


def _feature_agg_label(entry: dict[str, Any], index: int) -> str:
    """Human-readable label for an aggregation feature in error messages.

    Args:
        entry: A single ``features[]`` dict.
        index: Position in ``features`` used when no output name is set.

    Returns:
        ``output_column.name`` when present, otherwise ``#{index}``.
    """
    raw = entry.get("output_column")
    if isinstance(raw, dict):
        name = str(raw.get("name") or "").strip()
    else:
        name = str(raw or "").strip()
    return name if name else f"#{index}"


def _check_feature_aggregations(spec: dict[str, Any]) -> list[ValidationResult]:
    """Require complete aggregation rows on FeatureView features.

    A row carrying a function or a window is an aggregation, and needs all three
    of a function, a source-column name, and an integer window in seconds.
    Incomplete rows used to be dropped silently at apply time, so the deployed
    view could end up with fewer features than the spec declared.  Rows with
    none of those keys are 1:1 pass-throughs and are skipped.

    Args:
        spec: Normalized FeatureView spec dict.

    Returns:
        Errors for incomplete aggregation rows; empty when there are none.
    """
    kind = spec.get("kind", "")
    if "FeatureView" not in str(kind):
        return []

    results: list[ValidationResult] = []
    fv_name = spec.get("name", "") or ""
    for index, entry in enumerate(spec.get("features") or []):
        if not isinstance(entry, dict):
            continue
        function_name = entry.get("function")
        window_sec = entry.get("window_sec")
        has_window_sec = isinstance(window_sec, int)
        raw_window = entry.get("window")
        has_raw_window = raw_window is not None
        if not function_name and not has_window_sec and not has_raw_window:
            continue

        label = _feature_agg_label(entry, index)
        prefix = f"{fv_name}: aggregation {label}"

        if not function_name:
            results.append(
                _result(
                    "ERROR",
                    "FEATURE_INCOMPLETE",
                    f"{prefix} requires ``function``.",
                    fv_name,
                )
            )
        if not _source_column_name(entry):
            results.append(
                _result(
                    "ERROR",
                    "FEATURE_INCOMPLETE",
                    f"{prefix} requires ``source_column``.",
                    fv_name,
                )
            )
        if not has_window_sec:
            if has_raw_window:
                results.append(
                    _result(
                        "ERROR",
                        "FEATURE_INCOMPLETE",
                        f"{prefix} has unusable ``window: {raw_window!r}``. "
                        "Use a duration with a unit (``5m``, ``1h``); bare "
                        "numbers like ``300`` and fractional values like "
                        "``1.5m`` are rejected.",
                        fv_name,
                    )
                )
            else:
                results.append(
                    _result(
                        "ERROR",
                        "FEATURE_INCOMPLETE",
                        f"{prefix} requires ``window`` (e.g. ``5m``) or ``window_sec``.",
                        fv_name,
                    )
                )
    return results


# Aggregation functions whose deployed output-column type is predetermined by
# the runtime (e.g. acd -> ``LongType``), keyed by the lower-cased ``function``.
# Mirrors core's ``spec/builder.py::_AGG_PREDETERMINED_OUTPUT`` (parity-tested).
_AGG_REQUIRED_OUTPUT_TYPE: dict[str, str] = {
    "count": "LongType",
    "approx_count_distinct": "LongType",
    "avg": "DoubleType",
    "std": "DoubleType",
    "var": "DoubleType",
    "approx_percentile": "DoubleType",
}


def _check_feature_output_types(spec: dict[str, Any]) -> list[ValidationResult]:
    """Reject aggregation features whose authored output type can never match
    the runtime-deployed type (see :data:`_AGG_REQUIRED_OUTPUT_TYPE`), which
    would otherwise loop every re-plan on a destructive ``RECREATE_FV``. The
    authored type is compared through :data:`_BFV_TYPE_PROMOTION`, so promotable
    spellings (e.g. ``IntegerType`` -> ``LongType``) are accepted.

    Args:
        spec: Normalized FeatureView spec dict (``model_to_dict`` shape).

    Returns:
        ERROR results for mismatched aggregation outputs; empty when none.
    """
    # Scoped to BatchFeatureView: the table reflects offline OFT storage
    # widening; streaming FVs deploy through a different store.
    if spec.get("kind") != "BatchFeatureView":
        return []
    # Skip secondary-key FVs: their aggregation outputs are ArrayType-wrapped by
    # the runtime, reconciled by ``_strip_secondary_key_derivations`` instead.
    for sk_key in ("aggregation_secondary_keys", "ordered_secondary_key_column_names"):
        sk_val = spec.get(sk_key)
        if isinstance(sk_val, list) and sk_val:
            return []

    results: list[ValidationResult] = []
    fv_name = spec.get("name", "") or ""
    for index, entry in enumerate(spec.get("features") or []):
        if not isinstance(entry, dict):
            continue
        function_name = entry.get("function")
        if not function_name:
            continue
        # Canonicalize ``stddev`` -> ``std`` so the table (which mirrors core's
        # ``std`` token) matches either authoring spelling.
        required = _AGG_REQUIRED_OUTPUT_TYPE.get(str(_canonical_function_spelling(function_name)).lower())
        if required is None:
            continue
        out_col = entry.get("output_column")
        actual = out_col.get("type") if isinstance(out_col, dict) else None
        if actual is None:
            continue
        if _BFV_TYPE_PROMOTION.get(str(actual), str(actual)) != required:
            label = _feature_agg_label(entry, index)
            results.append(
                _result(
                    "ERROR",
                    "FEATURE_OUTPUT_TYPE_INVALID",
                    f"{fv_name}: aggregation {label} ({function_name}) requires "
                    f"``output_column.type`` {required}, got {actual}. Author the "
                    f"output as {required} (the runtime-deployed type).",
                    fv_name,
                )
            )
    return results


def _check_feature_param_conflicts(spec: dict[str, Any]) -> list[ValidationResult]:
    """Reject features that set both a ``function_params`` alias and its internal
    key with differing values (an ambiguous authoring shape that would otherwise
    be swallowed by the planner's defensive ``compile_to_spec`` fallbacks).

    Args:
        spec: Normalized FeatureView spec dict (``model_to_dict`` shape).

    Returns:
        ERROR results for conflicting aggregation params; empty when none.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import _FUNCTION_PARAM_ALIASES

    results: list[ValidationResult] = []
    fv_name = spec.get("name", "") or ""
    for index, entry in enumerate(spec.get("features") or []):
        if not isinstance(entry, dict):
            continue
        params = entry.get("function_params")
        if not isinstance(params, dict):
            continue
        aliases = _FUNCTION_PARAM_ALIASES.get(str(entry.get("function")).lower())
        if not aliases:
            continue
        for alias, internal in aliases.items():
            if alias in params and internal in params and params[alias] != params[internal]:
                label = _feature_agg_label(entry, index)
                results.append(
                    _result(
                        "ERROR",
                        "FEATURE_PARAMS_CONFLICT",
                        f"{fv_name}: aggregation {label} sets conflicting params "
                        f"'{alias}'={params[alias]!r} and '{internal}'="
                        f"{params[internal]!r}; provide only one.",
                        fv_name,
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Individual invariant checkers
# ---------------------------------------------------------------------------


def _check_versions(
    spec: dict[str, Any],
    applied: AppliedObject | None,
) -> list[ValidationResult]:
    """Version management invariants.

    Only FeatureView and FeatureGroup kinds must carry an explicit version.
    Entity and source specs are schema-level objects without semantic
    versioning.

    There is no downgrade rule.  Object identity is the name plus the version,
    so the applied object is always either the exact same version or nothing at
    all, and a lower version is a distinct object rather than a conflict.

    Args:
        spec: The normalized spec dict to validate.
        applied: The currently-deployed object, or None for new objects.  Kept
            for signature parity with the other checks; no longer consulted now
            that the downgrade rule is gone.

    Returns:
        Validation results, possibly empty.
    """
    results: list[ValidationResult] = []
    name = spec.get("name", "")
    kind = spec.get("kind", "")
    version = spec.get("version")

    # Version is only mandatory for versioned kinds.
    if kind not in _VERSIONED_KINDS:
        return results

    if not version:
        results.append(
            _result(
                "ERROR",
                "MISSING_VERSION",
                f"{name} has no version. Add an explicit version string.",
                name,
            )
        )
        return results

    # Object identity is (name, version); a versioned kind with a version
    # present is always valid.  There is no cross-version "downgrade" rule:
    # a lower version number is simply a distinct object, not a conflict.
    return results


def _check_idempotency(
    spec: dict[str, Any],
    applied: AppliedObject | None,
    target_database: str = "",
    target_schema: str = "",
    *,
    entity_join_keys: Mapping[str, list[str]] | None = None,
) -> tuple[bool, list[ValidationResult]]:
    """Content-hash idempotency check.

    For FeatureView kinds whose applied object came from a specification
    DESCRIBE, the desired-side hash is compiled and hashed the same way the
    planner and the state module do it.  Everything else compares structural
    fingerprints.

    Callers must thread the entity join keys through, as the planner does.
    Without them, a view whose entity name differs from its join-key column
    compiles a name-based entity column list that never matches the
    column-based applied hash, so no-change never short-circuits.

    Args:
        spec: The normalized spec dict to check.
        applied: The currently-deployed object, or None for new objects.
        target_database: Connection database for the compile step; falls back to
            the spec's own field when empty.
        target_schema: Connection schema for the compile step; falls back to the
            spec's own field when empty.
        entity_join_keys: Optional entity-name to join-key-columns map so the
            desired-side hash matches the applied side and the planner.

    Returns:
        Whether the spec is already up to date, plus any results.
    """
    if applied is None:
        return False, []

    name = spec.get("name", "")

    kind = spec.get("kind", "")
    if applied.from_specification and kind in _FV_RECREATE_KINDS:
        db_for_compile = target_database or spec.get("database", "") or ""
        sch_for_compile = target_schema or spec.get("schema", "") or spec.get("schema_", "") or ""
        try:
            current_hash = compute_local_spec_hash(
                spec, db_for_compile, sch_for_compile, entity_join_keys=entity_join_keys
            )
        except Exception:  # noqa: BLE001 — defensive fallback (mirrors planner)
            current_hash = structural_fingerprint_hash(spec)
    elif kind == "FeatureGroup":
        # FG identity hash mirrors the planner's ``fg_content_hash``;
        # there is no FV-style operational/structural split (see
        # ``decl/DESIGN.md`` §"Feature Group hash strategy").
        current_hash = fg_content_hash(spec)
    else:
        current_hash = structural_fingerprint_hash(spec)
    if current_hash == applied.content_hash:
        return True, [
            _result("WARNING", "NO_CHANGE", f"{name}: spec is identical to deployed version; skipping.", name)
        ]

    return False, []


def _check_column_evolution(
    spec: dict[str, Any],
    applied: AppliedObject | None,
) -> list[ValidationResult]:
    """Column-evolution invariants for feature views and sources."""
    if applied is None:
        return []

    results: list[ValidationResult] = []
    name = spec.get("name", "")
    kind = spec.get("kind", "")

    if "FeatureView" in kind:
        _check_fv_column_evolution(spec, applied, name, results)
    elif kind in ("StreamingSource", "BatchSource"):
        _check_source_column_evolution(spec, applied, name, results)

    return results


def _check_fv_column_evolution(
    spec: dict[str, Any],
    applied: AppliedObject,
    name: str,
    results: list[ValidationResult],
) -> None:
    """Column evolution checks for feature-view output columns.

    The features list sits at a different depth depending on where the applied
    payload came from: a specification DESCRIBE nests it one level inside
    ``spec``, while the legacy fingerprint shape keeps it at the top.  Without
    the depth-aware lookup the nested case sees zero deployed features, so every
    current feature looks new and each re-plan emits a spurious added-column
    warning per output column.

    Args:
        spec: The local spec dict being validated.
        applied: The deployed applied object, from either shape.
        name: The feature view's name, used in result messages.
        results: Mutable list accumulating any results.
    """
    if applied.from_specification:
        applied_features = applied.spec_payload.get("spec", {}).get("features", [])
    else:
        applied_features = applied.spec_payload.get("features", [])

    prev_features: dict[str, dict[str, Any]] = {
        f.get("output_column", {}).get("name"): f.get("output_column", {})
        for f in applied_features
        if f.get("output_column", {}).get("name")
    }
    curr_features: dict[str, dict[str, Any]] = {
        f.get("output_column", {}).get("name"): f.get("output_column", {})
        for f in spec.get("features", [])
        if f.get("output_column", {}).get("name")
    }

    prev_names = set(prev_features.keys())
    curr_names = set(curr_features.keys())

    # New columns: warning only (OFTs always recreate, no in-place update)
    added = curr_names - prev_names
    for col in sorted(added):
        results.append(
            _result(
                "WARNING",
                "COLUMN_ADDED",
                f"{name}.{col}: new output column (will be included in recreated OFT).",
                name,
            )
        )

    # Removed columns: warning for downstream consumers
    removed = prev_names - curr_names
    for col in sorted(removed):
        results.append(
            _result(
                "WARNING",
                "COLUMN_REMOVED",
                f"{name}.{col}: output column removed; downstream consumers may break.",
                name,
            )
        )

    # Type changes: warning
    for col in sorted(prev_names & curr_names):
        old_type = prev_features[col].get("type")
        new_type = curr_features[col].get("type")
        if old_type and new_type and old_type != new_type:
            results.append(
                _result(
                    "WARNING",
                    "COLUMN_TYPE_CHANGED",
                    f"{name}.{col}: output column type changed from '{old_type}' to '{new_type}'.",
                    name,
                )
            )


def _check_source_column_evolution(
    spec: dict[str, Any],
    applied: AppliedObject,
    name: str,
    results: list[ValidationResult],
) -> None:
    """Column evolution checks for source columns."""
    prev_cols: dict[str, dict[str, Any]] = {
        c.get("name"): c for c in applied.spec_payload.get("columns", []) if c.get("name")
    }
    curr_cols: dict[str, dict[str, Any]] = {c.get("name"): c for c in spec.get("columns", []) if c.get("name")}

    prev_names = set(prev_cols.keys())
    curr_names = set(curr_cols.keys())

    added = curr_names - prev_names
    for col in sorted(added):
        results.append(
            _result(
                "WARNING",
                "COLUMN_ADDED",
                f"{name}.{col}: new source column (will be included in recreated object).",
                name,
            )
        )

    removed = prev_names - curr_names
    for col in sorted(removed):
        results.append(
            _result(
                "WARNING",
                "COLUMN_REMOVED",
                f"{name}.{col}: source column removed; downstream feature views may break.",
                name,
            )
        )

    for col in sorted(prev_names & curr_names):
        old_type = prev_cols[col].get("type")
        new_type = curr_cols[col].get("type")
        if old_type and new_type and old_type != new_type:
            results.append(
                _result(
                    "WARNING",
                    "COLUMN_TYPE_CHANGED",
                    f"{name}.{col}: source column type changed from '{old_type}' to '{new_type}'.",
                    name,
                )
            )


def _check_dependencies(
    spec: dict[str, Any],
    batch_names: set[str],
    applied_state: AppliedState,
) -> list[ValidationResult]:
    """Dependency-resolution invariants."""
    results: list[ValidationResult] = []
    name = spec.get("name", "")
    kind = spec.get("kind", "")

    # Collect entity **names** and **join-key columns** from applied state.
    # ``validate_specs`` injects batch entities into ``applied_state`` as
    # ephemeral ``AppliedObject`` entries, so this covers both applied and
    # batch entities.  A FeatureView may reference an entity by authored
    # **name** or by a join-key **column**; both are valid.
    applied_entity_join_keys: set[str] = set()
    applied_entity_names: set[str] = set()
    applied_source_names: set[str] = set()
    applied_fv_names: set[str] = set()
    for obj in applied_state.objects.values():
        payload = obj.spec_payload
        obj_kind = payload.get("kind", "")
        if obj_kind == "Entity":
            entity_name = payload.get("name", "")
            if entity_name:
                # Mixed provenance: deployed entities are stored form, merged
                # batch entities are authoring SQL syntax. Index both so either
                # matches below.
                applied_entity_names.add(_to_stored_form(entity_name, authored=True))
                applied_entity_names.add(_to_stored_form(entity_name, authored=False))
            for jk in payload.get("join_keys", []):
                jk_name = jk.get("name", "") if isinstance(jk, dict) else str(jk)
                if jk_name:
                    applied_entity_join_keys.add(_to_stored_form(jk_name, authored=True))
                    applied_entity_join_keys.add(_to_stored_form(jk_name, authored=False))
        elif obj_kind in ("StreamingSource", "BatchSource"):
            src_name = payload.get("name", "")
            if src_name:
                applied_source_names.add(src_name)
        elif "FeatureView" in obj_kind:
            fv_name = payload.get("name", "")
            if fv_name:
                applied_fv_names.add(fv_name)

    if "FeatureView" in kind:
        # Entity reference validation.  A FeatureView's ``entities`` entries
        # are dicts (``{"name": ...}``) or plain strings, and may name an
        # entity by its authored **name** or by a join-key **column** (the two
        # coincide in the common case).  Accept either — callers merge batch
        # entities into ``applied_state`` so both the current batch and the
        # deployed state are covered.
        for eref in spec.get("entities", []):
            entity_name = eref.get("name", "") if isinstance(eref, dict) else str(eref)
            # Resolve the authoring reference (SQL syntax) to stored form before
            # the membership gate: a quoted authoring (``"userId"``) must match
            # the bare deployed join-key.  This is a membership gate, not
            # equality: the applied sets index both normalisations, so
            # over-accepting can only miss a ``MISSING_ENTITY``, never emit a
            # destructive op (the idempotency hash and
            # ``ENTITY_JOIN_KEY_IMMUTABLE`` remain the drift authority).
            eref_resolved = _to_stored_form(entity_name, authored=True) if entity_name else entity_name
            if (
                entity_name
                and eref_resolved not in applied_entity_join_keys
                and eref_resolved not in applied_entity_names
            ):
                results.append(
                    _result(
                        "ERROR",
                        "MISSING_ENTITY",
                        f"{name}: references entity '{entity_name}' but no entity with that "
                        "name or join key exists in the current batch or applied state.",
                        name,
                    )
                )

        # Source reference validation
        all_source_names = applied_source_names | batch_names
        for src_ref in spec.get("sources", []):
            src_name = src_ref.get("name", "")
            if src_name and src_name not in all_source_names:
                results.append(
                    _result(
                        "ERROR",
                        "MISSING_SOURCE",
                        f"{name}: references source '{src_name}' but no source with that name "
                        "exists in the current batch or applied state.",
                        name,
                    )
                )

    elif kind == "FeatureGroup":
        all_fv_names = applied_fv_names | batch_names
        for fv_ref in spec.get("feature_views", []):
            fv_name = fv_ref.get("name", "")
            if fv_name and fv_name not in all_fv_names:
                results.append(
                    _result(
                        "ERROR",
                        "MISSING_FEATURE_VIEW",
                        f"{name}: references feature view '{fv_name}' which does not exist "
                        "in the current batch or applied state.",
                        name,
                    )
                )

    return results


def _check_feature_group_sources(
    spec: dict[str, Any],
    batch_fv_specs: dict[tuple[str, str], dict[str, Any]],
    applied_state: AppliedState,
) -> list[ValidationResult]:
    """Cross-FV invariants for a single FeatureGroup spec.

    Mirrors the imperative register preflight so planning surfaces shape errors
    at compile time:

    * ``MISSING_FEATURE_VIEW`` — a referenced source resolves in neither the
      local batch nor the applied state.
    * ``FG_DUPLICATE_SOURCE`` — the same name and version appears twice.
      Pydantic catches this at model construction, but raw-dict callers need it
      too.
    * ``FG_SOURCE_NOT_ONLINE_POSTGRES`` — a referenced source that we can prove
      is not online and Postgres-backed.

    Online state resolves from the local payload first, then the applied one.
    When neither carries enough signal the check soft-passes and the imperative
    API raises the precondition at apply time, which avoids false negatives for
    the common authoring shape: the declarative layer does not expose the store
    type as an authoring field today.

    Args:
        spec: The FG spec dict to validate.
        batch_fv_specs: Map of uppercased name and version to local FV spec.
        applied_state: Applied-state snapshot.

    Returns:
        Validation results, possibly empty.
    """
    results: list[ValidationResult] = []
    if spec.get("kind") != "FeatureGroup":
        return results

    fg_name = spec.get("name", "")

    # Index applied FVs by uppercased ``(name, version)`` for
    # case-insensitive lookup (Snowflake identifiers are case-insensitive;
    # YAML round-trip can change the case of the source-FV ref).  Keying by
    # name alone would let a later version of one name clobber the earlier
    # one, so an FG pinning a specific version could be validated against
    # the wrong version's payload.
    applied_fvs_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for obj in applied_state.objects.values():
        if "FeatureView" in obj.kind:
            applied_fvs_by_id[((obj.name or "").upper(), (obj.version or "").upper())] = obj.spec_payload

    seen: set[tuple[str, str]] = set()
    for fv_ref in spec.get("feature_views", []) or []:
        if not isinstance(fv_ref, dict):
            continue
        fv_name = fv_ref.get("name", "")
        fv_version = fv_ref.get("version", "")
        if not fv_name:
            continue

        key = (fv_name.upper(), fv_version.upper())
        if key in seen:
            results.append(
                _result(
                    "ERROR",
                    "FG_DUPLICATE_SOURCE",
                    f"{fg_name}: duplicate source ({fv_name}, {fv_version}) in feature_views.",
                    fg_name,
                )
            )
            continue
        seen.add(key)

        # Resolution: prefer batch payload, then applied payload, else
        # soft-pass with a MISSING_FEATURE_VIEW unless it's in either view.
        # Both sides key by ``(name, version)`` so the pinned version is
        # resolved exactly, not shadowed by another version of the name.
        fv_id = (fv_name.upper(), fv_version.upper())
        local_payload = batch_fv_specs.get(fv_id)
        applied_payload = applied_fvs_by_id.get(fv_id)
        if local_payload is None and applied_payload is None:
            results.append(
                _result(
                    "ERROR",
                    "MISSING_FEATURE_VIEW",
                    f"{fg_name}: references feature view ({fv_name}, {fv_version}) "
                    "which does not exist in the current batch or applied state.",
                    fg_name,
                )
            )
            continue

        # Try local first, then applied — the first payload that carries
        # signal decides (the ``break`` below is unconditional).
        for payload in (local_payload, applied_payload):
            if payload is None:
                continue
            # The applied OFT SPECIFICATION payload carries no top-level
            # ``online`` flag; its signal is a non-empty ``online_store_type``.
            online = bool(payload.get("online", False)) or bool(payload.get("online_store_type"))
            online_config = payload.get("online_config")
            store_type = ""
            if isinstance(online_config, dict):
                store_type = (online_config.get("store_type") or "").upper()
            if not store_type:
                # ``online_store_type`` is the applied-state shape from
                # ``DESCRIBE ONLINE FEATURE TABLE`` recovery.
                ost = payload.get("online_store_type")
                if isinstance(ost, str):
                    store_type = ost.upper()

            if not online or (store_type and store_type != "POSTGRES"):
                # Visible violation — only emit when the payload actually
                # carries enough signal to prove it.  An unspecified
                # ``store_type`` in the local YAML soft-passes (the
                # authoring shape doesn't expose store_type today; the
                # imperative API is the source of truth at apply time).
                if not online:
                    results.append(
                        _result(
                            "ERROR",
                            "FG_SOURCE_NOT_ONLINE_POSTGRES",
                            f"{fg_name}: source FeatureView '{fv_name}' is not online "
                            "(register_feature_group requires online: true with "
                            "store_type: POSTGRES).",
                            fg_name,
                        )
                    )
                elif store_type and store_type != "POSTGRES":
                    results.append(
                        _result(
                            "ERROR",
                            "FG_SOURCE_NOT_ONLINE_POSTGRES",
                            f"{fg_name}: source FeatureView '{fv_name}' has store_type "
                            f"'{store_type}'; FeatureGroup requires Postgres-backed "
                            "online sources.",
                            fg_name,
                        )
                    )
            break  # first payload with signal decides this ref

    return results


def _ordered_join_key_names(payload: Any, *, authored: bool) -> list[str]:
    """Ordered, stored-form join-key column names from an Entity payload.

    Accepts the Entity shape where ``join_keys`` is a list of
    ``{"name": ..., "type": ...}`` dicts (or bare strings), preserving order
    (the OFT primary-key order). Each name goes through :func:`_to_stored_form`
    (not ``.upper()``) so a quoted authoring (``"userId"``) compares equal to
    the deployed bare ``userId`` instead of folding to ``USERID`` and raising a
    false ``ENTITY_JOIN_KEY_IMMUTABLE``.

    Args:
        payload: The Entity spec dict (authoring or applied ``spec_payload``).
        authored: ``True`` for the authoring half (SQL syntax, resolved first),
            ``False`` for the applied half (already stored, unquoted only).

    Returns:
        The ordered, stored-form join-key column names.
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("join_keys")
    if not isinstance(raw, list):
        return []
    cols: list[str] = []
    for jk in raw:
        if isinstance(jk, dict) and jk.get("name"):
            cols.append(str(_to_stored_form(str(jk["name"]), authored=authored)))
        elif isinstance(jk, str) and jk:
            cols.append(str(_to_stored_form(jk, authored=authored)))
    return cols


def _check_entity_join_keys_immutable(
    spec: dict[str, Any],
    applied: AppliedObject | None,
) -> list[ValidationResult]:
    """Reject join-key edits on an already-deployed entity at plan time.

    Join keys are identity for a registered entity, and updating one only
    rewrites its comment, so an edited YAML can never change the deployed key
    set.  Left unchecked, a rename, addition, removal or reorder flips a
    dependent feature view's entity column list and loops it on a destructive
    recreate that never converges, because the executor rebuilds the view from
    the live entity and its keys never moved.

    Order matters, so this compares the ordered names case-insensitively rather
    than the sorted set the structural fingerprint uses.  Callers must run it
    before the idempotency skip, since a reorder hashes identically under that
    sorted fingerprint.

    Args:
        spec: The local Entity spec dict being validated.
        applied: The deployed applied object, or None for new entities.

    Returns:
        Validation results, possibly empty.
    """
    if applied is None:
        return []
    if spec.get("kind", "") != "Entity":
        return []

    # The local Entity spec is authoring (SQL syntax); the applied payload is
    # Snowflake's stored form.  Resolve each half on its own provenance so a
    # quoted exported ``"userId"`` matches the deployed bare ``userId``.
    spec_jks = _ordered_join_key_names(spec, authored=True)
    applied_jks = _ordered_join_key_names(applied.spec_payload, authored=False)
    if spec_jks == applied_jks:
        return []

    name = spec.get("name", "")
    return [
        _result(
            "ERROR",
            "ENTITY_JOIN_KEY_IMMUTABLE",
            f"{name}: entity join keys cannot be changed after creation. "
            f"Deployed: {applied_jks}, requested: {spec_jks}. "
            "To change join keys, drop all dependent feature views, "
            "delete the entity, and re-apply.",
            name,
        )
    ]


def _check_destructive(
    spec: dict[str, Any],
    applied: AppliedObject | None,
) -> list[ValidationResult]:
    """Detect changes that require re-materialization.

    The deployed features list sits at a different depth depending on where the
    applied payload came from, exactly as in the column-evolution check.
    Without the depth-aware lookup the nested case sees zero deployed features
    and never raises the error that gates a plain apply against destructive
    aggregation edits.

    Args:
        spec: The local spec dict being validated.
        applied: The deployed applied object, or None for new objects.

    Returns:
        Validation results, possibly empty.
    """
    if applied is None:
        return []

    results: list[ValidationResult] = []
    name = spec.get("name", "")
    kind = spec.get("kind", "")

    if "FeatureView" not in kind:
        return results

    if applied.from_specification:
        applied_features = applied.spec_payload.get("spec", {}).get("features", [])
    else:
        applied_features = applied.spec_payload.get("features", [])

    prev_features: dict[str, dict[str, Any]] = {
        f.get("output_column", {}).get("name"): f for f in applied_features if f.get("output_column", {}).get("name")
    }
    for feat in spec.get("features", []):
        col_name = feat.get("output_column", {}).get("name")
        if not col_name or col_name not in prev_features:
            continue
        prev_feat = prev_features[col_name]
        # Compare functions (expression change).  Canonicalize the stddev
        # spelling so the applied SPECIFICATION token ``"stddev"`` and the
        # local-compile token ``"std"`` are not flagged as a function change.
        old_fn = _canonical_function_spelling(prev_feat.get("function"))
        new_fn = _canonical_function_spelling(feat.get("function"))
        if old_fn != new_fn:
            results.append(
                _result(
                    "ERROR",
                    "DESTRUCTIVE_CHANGE",
                    f"{name}.{col_name}: expression/function changed from '{old_fn}' to "
                    f"'{new_fn}'. This requires re-materialization. Use allow_recreate=True.",
                    name,
                )
            )

    return results


def _check_database_schema_mismatch(
    specs: list[dict[str, Any]],
    target_database: str,
    target_schema: str,
) -> list[ValidationResult]:
    """Warn when spec database or schema fields differ from the connection target.

    A spec that names a database or schema other than the connection target is
    ignored — the target always wins — so this surfaces the discrepancy for the
    user.  It only fires when both the spec field and the target are non-empty
    and they differ.

    Args:
        specs: Normalised spec dicts to check.
        target_database: The connection's target database.
        target_schema: The connection's target schema.

    Returns:
        Warning-severity validation results.
    """
    results: list[ValidationResult] = []
    target_db_upper = (target_database or "").upper()
    target_sch_upper = (target_schema or "").upper()

    for spec in specs:
        name = spec.get("name", "")

        # Database mismatch
        if target_db_upper:
            spec_db = (spec.get("database") or "").upper()
            if spec_db and spec_db != target_db_upper:
                results.append(
                    _result(
                        "WARNING",
                        "DB_MISMATCH",
                        f"{name}: spec has database='{spec.get('database')}' but target is "
                        f"'{target_database}'. The connection database ('{target_database}') "
                        "will be used.",
                        name,
                    )
                )

        # Schema mismatch — spec may use 'schema' or 'schema_' (Pydantic alias)
        if target_sch_upper:
            spec_sch_raw = spec.get("schema") or spec.get("schema_") or ""
            spec_sch = spec_sch_raw.upper()
            if spec_sch and spec_sch != target_sch_upper:
                results.append(
                    _result(
                        "WARNING",
                        "SCHEMA_MISMATCH",
                        f"{name}: spec has schema='{spec_sch_raw}' but target is "
                        f"'{target_schema}'. The connection schema ('{target_schema}') "
                        "will be used.",
                        name,
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _check_duplicate_spec_keys(
    specs: list[SpecBase],
    spec_sources: list[str],
    target_database: str,
    target_schema: str,
) -> list[ValidationResult]:
    """Reject specs that share one :func:`spec_key`.

    ``spec_sources`` is index-aligned with ``specs``. A short or empty
    list names the missing file ``<unknown>`` instead of raising.
    Distinct versions or schemas are not duplicates.

    Args:
        specs: Parsed specs, grouped by identity.
        spec_sources: Originating path of each spec. May be empty.
        target_database: Fallback database for :func:`spec_key`.
        target_schema: Fallback schema for :func:`spec_key`.

    Returns:
        One ``ERROR`` ``DUPLICATE_SPEC_KEY`` per key claimed more than once.
    """
    results: list[ValidationResult] = []

    # spec_key -> {"name": display name, "files": [source, ...]},
    # first-seen order preserved so the error message is deterministic.
    by_key: dict[str, dict[str, Any]] = {}
    for idx, spec_obj in enumerate(specs):
        data = model_to_dict(spec_obj)
        key = spec_key(data, database=target_database, schema=target_schema)
        kind = data.get("kind", "")
        if kind in _FV_DUPLICATE_ALIASES:
            # Collapse the FV subkind prefix so a batch and a streaming FV
            # sharing one name+version resolve to the same key.  count=1 only
            # rewrites the leading ``kind:`` token, leaving ``:VERSION`` intact.
            key = key.replace(f"{kind}:", "FeatureView:", 1)
        source = spec_sources[idx] if idx < len(spec_sources) else "<unknown>"
        entry = by_key.setdefault(key, {"name": data.get("name", ""), "files": []})
        entry["files"].append(source or "<unknown>")

    for key, entry in by_key.items():
        files: list[str] = entry["files"]
        if len(files) < 2:
            continue
        name = entry["name"]
        joined = ", ".join(files)
        results.append(
            _result(
                "ERROR",
                "DUPLICATE_SPEC_KEY",
                f"{name}: {len(files)} spec files define the same object "
                f"({key}) — {joined}. Object identity comes from the spec body "
                "(name / version / database / schema), not the filename or "
                "feature-view subkind (batch / streaming / realtime). Remove "
                "or rename one.",
                name,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Source compatibility checker
# ---------------------------------------------------------------------------


_SOURCE_KINDS: frozenset[str] = frozenset({"StreamingSource", "BatchSource"})

# Inner ``spec`` keys that define batch feature view semantics (source, entities,
# tiling).  When only keys outside this set differ from the deployed
# specification, the planner may emit a non-destructive ``UPDATE_FV``.
_BATCH_FV_STRUCTURAL_INNER_KEYS: frozenset[str] = frozenset(
    {
        "ordered_entity_column_names",
        "sources",
        "features",
        "timestamp_field",
        "feature_granularity_sec",
        "feature_aggregation_method",
        "udf",
        # Advanced BFV structural fields.  Each value here is included in
        # ``batch_feature_view_structural_equivalent``'s structural
        # projection so an authoring-side edit short-circuits the
        # operational-drift fast-path and falls through to ``RECREATE_FV``.
        # The exporter recovers each of these from the deployed DT text /
        # SPECIFICATION JSON so the inner-spec round-trip stays clean.
        "cluster_by",
        "refresh_mode",
        "initialize",
        "storage_config",
        "aggregation_secondary_keys",
        # ``append_only`` toggles the imperative snapshot-accumulation path
        # (companion ``$SNAPSHOTS`` table + CRON refresh Task); flipping it is
        # structural, so it must fall through to ``RECREATE_FV``.
        "append_only",
    }
)


def batch_feature_view_structural_equivalent(
    local_authoring: dict[str, Any],
    applied_payload: dict[str, Any],
    database: str,
    schema: str,
    *,
    entity_join_keys: Mapping[str, list[str]] | None = None,
) -> bool:
    """Return True if local and deployed batch FVs differ only operationally.

    Compares the structural subset of the inner spec — sources, tiling and
    entities — between the compiled local spec and the deployed one.  The planner
    uses this to emit an in-place update instead of a recreate when refresh
    cadence, warehouse, description or online toggles change without any
    semantic edit.

    Args:
        local_authoring: Parsed authoring dict for the batch feature view.
        applied_payload: Full applied-state payload from the state fetch.
        database: Target database for the local compile.
        schema: Target schema for the local compile.
        entity_join_keys: Optional entity-name to join-key-columns map so the
            local entity column wire field matches the applied side's join-key
            columns.

    Returns:
        True when the structural projections match, False otherwise or on a
        compile error.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    try:
        local_compiled = compile_to_spec(local_authoring, database, schema, entity_join_keys=entity_join_keys)
    except Exception:
        return False
    _li = local_compiled.get("spec")
    local_inner = _li if isinstance(_li, dict) else {}
    _ri = applied_payload.get("spec")
    remote_inner = _ri if isinstance(_ri, dict) else {}

    def _project(inner: dict[str, Any], ref_inner: dict[str, Any] | None = None) -> dict[str, Any]:
        """Project the FV inner spec onto its semantic-stable subset.

        Mirrors the parity normalisation the full-spec hash applies: source
        bindings reduce to a sorted projection and auto-derived 1:1 features are
        dropped.  Without it, the deployed empty source list and auto-derived
        features always disagree with the local shape, and every batch edit
        falls through to a recreate.

        For the applied side, pass the local inner spec so a refresh mode the
        operator never authored can be dropped.  Snowflake always stamps one, so
        leaving it in creates a spurious structural diff.

        Identifiers use the same asymmetric contract as :func:`_full_spec_hash`
        (local side resolves to stored form, applied side unquotes only) so a
        quoted column cannot make the ``UPDATE_FV`` gate and the ``RECREATE_FV``
        hash disagree.

        Args:
            inner: The FV inner spec block to project.
            ref_inner: The local inner spec when projecting the applied side;
                omit when projecting the local side.

        Returns:
            A copy of the inner spec restricted to the structurally-stable keys,
            with sources and features normalised for comparison.
        """
        block: dict[str, Any] = {}
        for k in _BATCH_FV_STRUCTURAL_INNER_KEYS:
            if k not in inner:
                continue
            block[k] = copy.deepcopy(inner[k])
        _canonicalise_inner_spec_identifiers(block, authored=ref_inner is None)
        if "sources" in block:
            block["sources"] = _normalise_fv_sources_for_hash(block.get("sources"))
        if "features" in block and isinstance(block["features"], list):
            block["features"] = [f for f in block["features"] if not _is_auto_derived_feature(f)]
        if ref_inner is not None and "refresh_mode" not in ref_inner:
            block.pop("refresh_mode", None)
        return block

    return json.dumps(_project(local_inner), sort_keys=True, default=str) == json.dumps(
        _project(remote_inner, ref_inner=local_inner), sort_keys=True, default=str
    )


def _check_batch_fv_udf(name: str, spec: dict[str, Any]) -> list[ValidationResult]:
    """BatchFeatureView cannot carry a UDF transform (offline store cannot run it)."""
    results: list[ValidationResult] = []
    udf = spec.get("udf")
    if udf and isinstance(udf, dict) and len(udf) > 0:
        results.append(
            _result(
                "ERROR",
                "BATCH_FV_UDF_FORBIDDEN",
                f"{name}: BatchFeatureView does not support UDF transforms (offline store cannot "
                "execute them). Remove the ``udf:`` block.",
                name,
            )
        )
    return results


def _check_batch_fv_tiling(
    name: str,
    spec: dict[str, Any],
    features: list[Any],
) -> list[ValidationResult]:
    """A tiled BatchFeatureView (non-empty authored ``features``) requires the
    timestamp / granularity / aggregation-method / refresh fields.

    Args:
        name: The feature view name, used in error messages.
        spec: The BatchFeatureView spec dict.
        features: Authored ``features`` with auto-derived entries stripped.

    Returns:
        Validation results for any missing tiling fields.
    """
    results: list[ValidationResult] = []
    if features:
        if not spec.get("timestamp_col"):
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_TILING_TIMESTAMP",
                    f"{name}: tiled BatchFeatureView requires ``timestamp_col`` when ``features`` "
                    "declares aggregation windows.",
                    name,
                )
            )
        if spec.get("feature_granularity_sec") is None and spec.get("feature_granularity") is None:
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_TILING_GRANULARITY",
                    f"{name}: tiled BatchFeatureView requires ``feature_granularity`` (or "
                    "``feature_granularity_sec``) when ``features`` is non-empty.",
                    name,
                )
            )
        if not spec.get("feature_aggregation_method"):
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_TILING_AGG_METHOD",
                    f"{name}: tiled BatchFeatureView requires ``feature_aggregation_method`` (e.g. "
                    "``tiles``) when ``features`` is non-empty.",
                    name,
                )
            )
        if not spec.get("refresh_freq") and spec.get("target_lag_sec") is None and spec.get("target_lag") is None:
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_TILING_REFRESH",
                    f"{name}: tiled BatchFeatureView requires ``refresh_freq`` or ``target_lag`` / "
                    "``target_lag_sec`` so the offline Dynamic Table has a refresh cadence.",
                    name,
                )
            )
    return results


def _check_batch_fv_secondary_keys(
    name: str,
    spec: dict[str, Any],
    features: list[Any],
    batch_source_specs: dict[str, dict[str, Any]],
) -> list[ValidationResult]:
    """``aggregation_secondary_keys`` rules: max length 1,
    plus a soft nudge when the key is absent from the source ``columns``.

    Args:
        name: The feature view name, used in error messages.
        spec: The BatchFeatureView spec dict.
        features: Authored ``features`` with auto-derived entries stripped.
        batch_source_specs: Resolved BatchSource specs keyed by name.

    Returns:
        Validation results for the secondary-key constraints.
    """
    results: list[ValidationResult] = []
    # ``aggregation_secondary_keys`` constraint: max
    # length 1 (current Snowflake cap).  The field is NOT tiled-only — on
    # a tiled BFV it adds a secondary group-by to each aggregation, and on
    # a non-tiled (passthrough) BFV it is still a first-class identity
    # column: the imperative register path persists it,
    # ``_build_batch_feature_view_spec`` folds it into ``entity_columns``
    # / ``secondary_key_columns``, and the POSTGRES OFT widens its primary
    # key with it.  The removed ``BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES``
    # gate rejected a state the imperative side persists and the exporter
    # re-emits into ``snow feature init`` YAML.
    secondary_keys = spec.get("aggregation_secondary_keys")
    if isinstance(secondary_keys, list) and secondary_keys:
        if len(secondary_keys) > 1:
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_SECONDARY_KEYS_MAX_LENGTH",
                    f"{name}: ``aggregation_secondary_keys`` is capped at length 1; got "
                    f"{len(secondary_keys)} entries.",
                    name,
                )
            )
        # Soft nudge — a tiled BFV synthesizes a ``_SECONDARY_KEY_ARRAY``
        # aggregation over the secondary key, so recovery's resolution pool
        # (rebuilt from ``FV_SOURCE_REFS`` columns) must contain it.  The
        # executor now auto-includes the key in the stamped source columns and
        # in the recovery pool, so an omission no longer loops on
        # ``RECREATE_FV`` — but listing the secondary key on the source is
        # still the clearer authoring shape.  Emit a WARNING (never an ERROR:
        # already-deployed FVs whose YAML omits the key must keep planning).
        if features:
            available: set[str] = set()
            for src in spec.get("sources", []) or []:
                if not isinstance(src, dict):
                    continue
                for col in src.get("columns") or []:
                    if isinstance(col, dict) and col.get("name"):
                        available.add(_to_stored_form(col["name"], authored=True))
                resolved = batch_source_specs.get(str(src.get("name", "")))
                if isinstance(resolved, dict):
                    for col in resolved.get("columns") or []:
                        if isinstance(col, dict) and col.get("name"):
                            available.add(_to_stored_form(col["name"], authored=True))
            for sk in secondary_keys:
                if _to_stored_form(sk, authored=True) not in available:
                    results.append(
                        _result(
                            "WARNING",
                            "BATCH_FV_SECONDARY_KEY_ABSENT_FROM_SOURCE_COLUMNS",
                            f"{name}: ``aggregation_secondary_keys`` column '{sk}' is not listed in "
                            "the source ``columns``. It is auto-included in FV_SOURCE_REFS so "
                            "replans still converge, but adding it to the BatchSource ``columns`` "
                            "is the recommended authoring shape.",
                            name,
                        )
                    )
    return results


def _check_batch_fv_storage(name: str, spec: dict[str, Any]) -> list[ValidationResult]:
    """Iceberg ``storage_config`` must declare an ``external_volume`` (or omit the
    block to accept the native-storage / FeatureStore default).

    Args:
        name: The feature view name, used in error messages.
        spec: The BatchFeatureView spec dict.

    Returns:
        Validation results for the iceberg storage constraint.
    """
    results: list[ValidationResult] = []
    # Iceberg ``storage_config`` requires an ``external_volume`` at the
    # authoring level.  Runtime fallback to
    # ``FeatureStore.default_iceberg_external_volume`` is the imperative
    # library's concern; the spec validator only requires that the local
    # YAML either declare a volume or accept the FS default by leaving
    # the field out entirely (in which case the imperative side raises a
    # clearer error at register time).  Authoring ``format: iceberg`` +
    # missing ``external_volume`` is almost always a mistake, so we flag
    # it early.
    storage_config = spec.get("storage_config")
    if isinstance(storage_config, dict):
        fmt = storage_config.get("format")
        if isinstance(fmt, str) and fmt.lower() == "iceberg":
            vol = storage_config.get("external_volume")
            if not (isinstance(vol, str) and vol.strip()):
                results.append(
                    _result(
                        "ERROR",
                        "BATCH_FV_STORAGE_ICEBERG_NO_VOLUME",
                        f"{name}: ``storage_config.format: iceberg`` requires "
                        "``storage_config.external_volume`` (or remove the block to "
                        "default to native Snowflake storage).",
                        name,
                    )
                )
    return results


def _check_batch_fv_sources(
    name: str,
    spec: dict[str, Any],
    batch_source_specs: dict[str, dict[str, Any]],
) -> list[ValidationResult]:
    """Each ``sources[]`` entry must resolve to a BatchSource with a table/query
    and must not use a realtime ``source_type``.

    Args:
        name: The feature view name, used in error messages.
        spec: The BatchFeatureView spec dict.
        batch_source_specs: Resolved BatchSource specs keyed by name.

    Returns:
        Validation results for the source constraints.
    """
    results: list[ValidationResult] = []
    for src in spec.get("sources", []):
        if not isinstance(src, dict):
            continue
        st = (src.get("source_type") or "").strip()
        if st in ("Stream", "Streaming", "Request", "Features"):
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_INVALID_SOURCE_TYPE",
                    f"{name}: BatchFeatureView cannot use realtime source_type '{st}'. Use ``Batch`` "
                    "with a ``BatchSource`` table or view.",
                    name,
                )
            )
        src_name = src.get("name", "")
        has_inline_table = bool(src.get("table") or src.get("query"))
        if has_inline_table:
            continue
        if not src_name:
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_SOURCE_UNRESOLVED",
                    f"{name}: each ``sources[]`` entry must declare ``table``/``query`` or reference a "
                    "named ``BatchSource`` with a ``table``.",
                    name,
                )
            )
        resolved = batch_source_specs.get(src_name)
        if resolved is None:
            continue  # MISSING_SOURCE from dependency check
        if resolved.get("kind") != "BatchSource":
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_SOURCE_KIND",
                    f"{name}: source '{src_name}' must be a ``BatchSource`` for BatchFeatureView.",
                    name,
                )
            )
        if not (resolved.get("table") or resolved.get("query")):
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_SOURCE_NO_TABLE",
                    f"{name}: BatchSource '{src_name}' must declare ``table`` or ``query``.",
                    name,
                )
            )
    return results


def _check_batch_feature_view_constraints(
    spec: dict[str, Any],
    batch_source_specs: dict[str, dict[str, Any]],
) -> list[ValidationResult]:
    """Validate ``BatchFeatureView`` authoring rules (no UDF, tiling, secondary
    keys, iceberg storage, batch sources).  Each concern lives in its own
    ``_check_batch_fv_*`` helper below.

    Args:
        spec: The BatchFeatureView spec dict.
        batch_source_specs: Resolved BatchSource specs keyed by name.

    Returns:
        The combined validation results from every ``_check_batch_fv_*`` helper.
    """
    name = spec.get("name", "")
    # ``_is_auto_derived_feature`` recognises the 1:1 passthrough entries
    # snowml-core's ``DESCRIBE … TYPE = SPECIFICATION`` round-trip auto-
    # generates from the source columns for any non-tiled BatchFV that
    # omits an explicit ``features:`` block (already stripped before
    # hashing in :func:`_full_spec_hash` and :func:`_check_idempotency`).
    # The tiling-invariant check must apply the same stripping; otherwise
    # every freshly-exported batch FV (e.g. step 4 of docs/BUG_BASH.md)
    # trips BATCH_FV_TILING_* against a non-tiled FV and blocks every
    # downstream plan.
    features = [f for f in (spec.get("features") or []) if not _is_auto_derived_feature(f)]
    results: list[ValidationResult] = []
    results += _check_batch_fv_udf(name, spec)
    results += _check_batch_fv_tiling(name, spec, features)
    results += _check_batch_fv_secondary_keys(name, spec, features, batch_source_specs)
    results += _check_batch_fv_storage(name, spec)
    results += _check_batch_fv_sources(name, spec, batch_source_specs)
    return results


def _check_streaming_feature_view_constraints(
    spec: dict[str, Any],
) -> list[ValidationResult]:
    """Validate StreamingFeatureView authoring rules before apply.

    A tiled streaming view — one where any feature declares an aggregation
    window — materialises its tiles as a managed Dynamic Table whose target lag
    comes from ``refresh_freq``.  The imperative layer requires it, and without
    it the register flow quietly degrades to a plain view over the untiled source
    query, producing no tiles at all.  So a tiled streaming view must carry
    ``refresh_freq``, mirroring the batch check.

    Unlike the batch check, ``target_lag`` and ``target_lag_sec`` are not
    accepted substitutes: on a streaming view they name the online table's lag,
    which the runtime stamps to zero on the ingest path, and they never schedule
    the offline tile table.  A non-tiled streaming view materialises to a
    zero-lag view with no table to schedule, so it is unaffected.

    Two further rules catch what the imperative stream-config builder would
    otherwise reject halfway through an apply, so an unappliable spec fails at
    plan time instead.  A streaming view is by definition backed by a stream
    config, and that cannot be built without both a source entry carrying a name
    and a UDF with a name and an inlinable body (either an authoring file or an
    already-inlined definition).

    These run after the idempotency skip, so a hash-matched legacy object that
    pre-dates the metadata — a view-backed feature view recovered as a near-empty
    shell, say — stays unchanged and is never flagged.

    Args:
        spec: The StreamingFeatureView authoring dict.

    Returns:
        Validation results for a tiled view missing its refresh cadence, a
        missing named source, and a missing or unusable UDF.
    """
    results: list[ValidationResult] = []
    name = spec.get("name", "")
    raw_features = spec.get("features") or []
    has_windows = any(
        isinstance(f, dict) and (f.get("window") is not None or f.get("window_sec") is not None) for f in raw_features
    )
    if has_windows and not spec.get("refresh_freq"):
        results.append(
            _result(
                "ERROR",
                "STREAM_FV_TILING_REFRESH",
                f"{name}: tiled StreamingFeatureView requires ``refresh_freq`` so the offline tile "
                "Dynamic Table has a refresh cadence (it drives ``CREATE DYNAMIC TABLE … TARGET_LAG`` "
                "over the tiling query). ``target_lag`` / ``target_lag_sec`` name the Online Feature "
                "Table's ingest-path lag (runtime-stamped 0) and do not schedule the offline DT.",
                name,
            )
        )

    sources = spec.get("sources") or []
    has_named_source = any(isinstance(s, dict) and str(s.get("name") or "").strip() for s in sources)
    if not has_named_source:
        results.append(
            _result(
                "ERROR",
                "STREAM_FV_SOURCE_REQUIRED",
                f"{name}: StreamingFeatureView must declare a stream source in ``sources[]`` with a "
                "``name`` — a StreamingFeatureView is always backed by a StreamConfig, which cannot "
                "be built without one (the imperative register flow raises the same error at apply).",
                name,
            )
        )

    udf = spec.get("udf")
    udf_ok = (
        isinstance(udf, dict)
        and str(udf.get("name") or "").strip()
        and (udf.get("file") or udf.get("function_definition"))
    )
    if not udf_ok:
        results.append(
            _result(
                "ERROR",
                "STREAM_FV_UDF_REQUIRED",
                f"{name}: StreamingFeatureView must declare a ``udf`` with a ``name`` and an inlinable "
                "body (authoring ``file`` or inlined ``function_definition``) — the StreamConfig "
                "transformation function is required (the imperative register flow raises the same "
                "error at apply).",
                name,
            )
        )
    return results


def _check_source_compatibility(
    batch_specs: list[dict[str, Any]],
    applied_state: AppliedState,
) -> list[ValidationResult]:
    """Check that source changes are compatible with dependent feature views.

    Compares each source spec in the batch against its applied version and looks
    at whether any feature view in the batch touches the affected columns.
    Removing or retyping a column that a view depends on is an error, while
    adding one is only a warning.  Dropping a source that a view still references
    is also an error.

    Args:
        batch_specs: Normalised spec dicts in the current batch.
        applied_state: Current applied-state snapshot.

    Returns:
        Validation results.
    """
    results: list[ValidationResult] = []

    # Map source name -> new spec dict (from batch)
    batch_sources: dict[str, dict[str, Any]] = {}
    for spec in batch_specs:
        if spec.get("kind") in _SOURCE_KINDS:
            name = spec.get("name", "")
            if name:
                batch_sources[name] = spec

    # Map source name -> list of FV dicts that reference it (from batch)
    batch_fv_by_source: dict[str, list[dict[str, Any]]] = {}
    for spec in batch_specs:
        if "FeatureView" in spec.get("kind", ""):
            for src_ref in spec.get("sources", []):
                src_name = src_ref.get("name", "")
                if src_name:
                    batch_fv_by_source.setdefault(src_name, []).append(spec)

    # -------------------------------------------------------------------------
    # Check 1: column-level changes for sources present in both batch and applied
    # -------------------------------------------------------------------------
    for src_name, new_src in batch_sources.items():
        applied_src_obj: AppliedObject | None = None
        for obj in applied_state.objects.values():
            if obj.kind in _SOURCE_KINDS and obj.name == src_name:
                applied_src_obj = obj
                break

        if applied_src_obj is None:
            # Brand-new source -- no compatibility issue possible
            continue

        prev_cols: dict[str, dict[str, Any]] = {
            c.get("name"): c for c in applied_src_obj.spec_payload.get("columns", []) if c.get("name")
        }
        curr_cols: dict[str, dict[str, Any]] = {c.get("name"): c for c in new_src.get("columns", []) if c.get("name")}

        # Collect source columns actually used by dependent FVs
        used_columns: set[str] = set()
        for fv in batch_fv_by_source.get(src_name, []):
            for feat in fv.get("features", []):
                col = (feat.get("source_column") or {}).get("name", "")
                if col:
                    used_columns.add(col)

        # Columns removed -- error only when a batch FV depends on that column
        removed = set(prev_cols.keys()) - set(curr_cols.keys())
        for col in sorted(removed):
            if col in used_columns:
                results.append(
                    _result(
                        "ERROR",
                        "SOURCE_COLUMN_REMOVED",
                        f"{src_name}.{col}: source column removed but referenced by feature view(s). "
                        "Remove the column from dependent feature views first.",
                        src_name,
                    )
                )

        # Columns added -- always a backward-compatible warning
        added = set(curr_cols.keys()) - set(prev_cols.keys())
        for col in sorted(added):
            results.append(
                _result(
                    "WARNING",
                    "SOURCE_COLUMN_ADDED",
                    f"{src_name}.{col}: new source column (backward compatible).",
                    src_name,
                )
            )

        # Column type changes -- error only when a batch FV depends on that column
        for col in sorted(set(prev_cols.keys()) & set(curr_cols.keys())):
            old_type = prev_cols[col].get("type")
            new_type = curr_cols[col].get("type")
            if old_type and new_type and old_type != new_type and col in used_columns:
                results.append(
                    _result(
                        "ERROR",
                        "SOURCE_COLUMN_TYPE_CHANGED",
                        f"{src_name}.{col}: source column type changed from '{old_type}' to "
                        f"'{new_type}'; referenced by feature view(s). "
                        "Update dependent feature views first.",
                        src_name,
                    )
                )

    # -------------------------------------------------------------------------
    # Check 2: applied sources absent from the batch but still referenced by
    # FVs in the batch (implicit deletion)
    # -------------------------------------------------------------------------
    for obj in applied_state.objects.values():
        if obj.kind not in _SOURCE_KINDS:
            continue
        src_name = obj.name
        if src_name in batch_sources:
            continue  # Source is present in the batch -- not being deleted

        dependent_fvs = batch_fv_by_source.get(src_name, [])
        if dependent_fvs:
            fv_names = ", ".join(sorted(fv.get("name", "?") for fv in dependent_fvs))
            results.append(
                _result(
                    "ERROR",
                    "SOURCE_DELETED_WITH_DEPENDENTS",
                    f"{src_name}: source is being removed but still referenced by feature view(s): "
                    f"{fv_names}. Remove or update the dependent feature views first.",
                    src_name,
                )
            )

    return results


def validate_specs(
    batch: SpecBatch,
    applied_state: AppliedState,
    target_database: str = "",
    target_schema: str = "",
) -> list[ValidationResult]:
    """Validate a spec batch against all invariant rules.

    Pure — no side effects, no database access.

    Args:
        batch: The parsed spec batch to validate.
        applied_state: Current applied-state snapshot.
        target_database: Connection target database; when non-empty, a spec
            naming a different database produces a mismatch warning.
        target_schema: Connection target schema; when non-empty, a spec naming a
            different schema produces a mismatch warning.

    Returns:
        Combined list of errors and warnings.
    """
    results: list[ValidationResult] = []

    # Before the per-spec loop: an idempotent batch must still report duplicates.
    results.extend(
        _check_duplicate_spec_keys(
            batch.specs,
            batch.spec_sources,
            target_database,
            target_schema,
        )
    )

    # Build lookup data from the batch for cross-spec dependency checks.
    # We need source names from ALL specs in this batch.
    batch_source_names: set[str] = set()
    batch_fv_names: set[str] = set()
    batch_names: set[str] = set()

    for spec_obj in batch.specs:
        data = model_to_dict(spec_obj)
        kind = data.get("kind", "")
        name = data.get("name", "")
        if name:
            batch_names.add(name)
        if kind in ("StreamingSource", "BatchSource"):
            if name:
                batch_source_names.add(name)
        elif "FeatureView" in kind:
            if name:
                batch_fv_names.add(name)

    # Merge batch entity join keys into a temporary applied_state for dependency checks.
    # We synthesise ephemeral AppliedObject entries for batch entities so that
    # _check_dependencies can find them via applied_state.objects.
    merged_objects = dict(applied_state.objects)
    for spec_obj in batch.specs:
        data = model_to_dict(spec_obj)
        kind = data.get("kind", "")
        if kind == "Entity":
            # Forward connection context so unqualified spec keys collide
            # with the (always fully-qualified) applied-state keys — same
            # rationale as the planner's diff loop.
            key = spec_key(data, database=target_database, schema=target_schema)
            if key not in merged_objects:
                merged_objects[key] = AppliedObject(
                    key=key,
                    kind=kind,
                    name=data.get("name", ""),
                    spec_payload=data,
                )
    from snowflake.ml.feature_store.decl.types import AppliedState as _AS

    merged_applied = _AS(objects=merged_objects)
    all_source_batch_names = batch_source_names | batch_fv_names | batch_names

    batch_source_specs: dict[str, dict[str, Any]] = {}
    for spec_obj in batch.specs:
        d = model_to_dict(spec_obj)
        if d.get("kind") == "BatchSource" and d.get("name"):
            batch_source_specs[d["name"]] = d

    # FeatureView specs keyed by uppercased ``(name, version)`` — the tuple
    # contract ``_check_feature_group_sources`` expects so an FG source
    # reference resolves to the exact pinned version rather than being
    # shadowed by another version of the same name.
    batch_fv_specs: dict[tuple[str, str], dict[str, Any]] = {}
    for spec_obj in batch.specs:
        d = model_to_dict(spec_obj)
        if "FeatureView" in str(d.get("kind", "")) and d.get("name"):
            batch_fv_specs[(d["name"].upper(), str(d.get("version") or "").upper())] = d

    # Entity-name → join-key-columns map, mirroring ``planner.generate_plan``.
    # ``_check_idempotency`` compiles each local FV through
    # ``compute_local_spec_hash``; without this map an FV whose entity name
    # differs from its join-key column hashes on the authored name and never
    # matches the column-based applied hash, so ``NO_CHANGE`` never short-
    # circuits for those projects (contradicting the planner).
    from snowflake.ml.feature_store.decl.spec_compiler import build_entity_join_key_map

    all_batch_dicts = [model_to_dict(s) for s in batch.specs]
    entity_join_keys = build_entity_join_key_map(all_batch_dicts, applied_state)

    for spec_obj in batch.specs:
        data = model_to_dict(spec_obj)
        # See ``planner.generate_plan`` for the rationale: connection
        # context is a fallback so bare YAMLs match fully-qualified
        # applied-state keys instead of silently producing phantom
        # diffs.  ``target_database`` / ``target_schema`` flow in via
        # the public ``validate_specs`` signature.
        key = spec_key(data, database=target_database, schema=target_schema)
        applied = applied_state.objects.get(key)

        # Authoring completeness is independent of applied state: an
        # incomplete aggregation must fail even on a no-op re-plan.
        if "FeatureView" in str(data.get("kind", "")):
            results.extend(_check_feature_aggregations(data))
            # Reject aggregation features whose authored output_column.type
            # cannot match the runtime-predetermined type (independent of
            # applied state; a wrong type would otherwise loop on RECREATE_FV).
            results.extend(_check_feature_output_types(data))
            # Reject ambiguous function_params (alias + internal key disagree),
            # which the planner's compile_to_spec fallbacks would otherwise hide.
            results.extend(_check_feature_param_conflicts(data))

        # Entity join keys are immutable after create.  Run this BEFORE the
        # idempotency skip: a composite-key reorder hashes identically under
        # the sorted Entity structural fingerprint, so it would otherwise be
        # skipped as up-to-date while a dependent FV loops on RECREATE_FV.
        results.extend(_check_entity_join_keys_immutable(data, applied))

        # 1. Idempotency — check first; if up-to-date, skip remaining checks.
        is_up_to_date, idem_results = _check_idempotency(
            data,
            applied,
            target_database=target_database,
            target_schema=target_schema,
            entity_join_keys=entity_join_keys,
        )
        results.extend(idem_results)
        if is_up_to_date:
            continue

        # 2. Version checks
        results.extend(_check_versions(data, applied))

        # 3. Column evolution
        results.extend(_check_column_evolution(data, applied))

        # 4. Dependency validation
        results.extend(_check_dependencies(data, all_source_batch_names, merged_applied))

        if data.get("kind") == "BatchFeatureView":
            results.extend(_check_batch_feature_view_constraints(data, batch_source_specs))
        elif data.get("kind") == "StreamingFeatureView":
            results.extend(_check_streaming_feature_view_constraints(data))
        elif data.get("kind") == "FeatureGroup":
            # Enforce the register_feature_group source preconditions at plan
            # time: unresolved / duplicate / not-online-Postgres sources.
            # Resolve against the raw applied_state (the helper reads only
            # FeatureView-kind applied objects, so the ephemeral-Entity
            # merged_applied is unnecessary here).
            results.extend(_check_feature_group_sources(data, batch_fv_specs, applied_state))

        # 5. Destructive changes
        results.extend(_check_destructive(data, applied))

    # 6. Database/schema mismatch warnings (batch-level, after per-spec loop)
    if target_database or target_schema:
        all_dicts = [model_to_dict(s) for s in batch.specs]
        results.extend(_check_database_schema_mismatch(all_dicts, target_database, target_schema))

    # 7. Source compatibility checks (batch-level)
    all_dicts = [model_to_dict(s) for s in batch.specs]
    results.extend(_check_source_compatibility(all_dicts, applied_state))

    return results
