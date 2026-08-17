"""Invariant engine for the declarative feature store library.

Pure functions only — no database connections, no file I/O.
Entry point: ``validate_specs(batch, applied_state) -> list[ValidationResult]``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Literal, Optional

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


def spec_key(data: dict[str, Any], *, database: str = "", schema: str = "") -> str:
    """Build a unique key: ``kind:database.schema:NAME`` (uppercased name).

    The two concrete source kinds (``StreamingSource``, ``BatchSource``)
    collapse to the canonical ``Datasource`` kind in
    the key so a freshly-loaded YAML matches the
    :func:`state._datasource_objects_from_specs` AppliedObject (which
    uses the generic :data:`ObjectKind.DATASOURCE` for display
    consistency in ``snow feature list``).

    The ``database`` / ``schema`` kwargs are *fallback only* — they are
    consulted only when the spec dict itself omits the corresponding
    field.  This keeps the multi-store contract intact (a spec that
    explicitly names ``OTHER_DB.OTHER_SCHEMA`` always wins over the
    connection context) while letting the planner qualify keys for
    bare YAMLs (single-file ``apply`` / ``plan`` invocations or
    declarative entities written without ``database:``/``schema:``)
    against the active ``snow`` connection so the lookup key collides
    with the applied-state key (which is *always* fully qualified by
    :func:`state._build_spec_key`).

    Args:
        data: Spec dict carrying ``kind``, ``name``, ``database``, and
            either ``schema`` or ``schema_``.
        database: Connection-context database used as a fallback when
            ``data`` does not carry a ``database`` field.  Defaults to
            ``""`` for backwards compatibility with callers that don't
            yet thread connection context through (e.g. ad-hoc spec
            inspection in tests).
        schema: Connection-context schema used as a fallback when
            ``data`` does not carry a ``schema`` / ``schema_`` field.
            Defaults to ``""``.

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
    return f"{kind}:{qualifier}:{name}"


def _spec_hash(data: dict[str, Any]) -> str:
    """SHA-256 over the full spec dict (excluding ``_hash`` / ``_content_hash``)."""
    clean = {k: v for k, v in data.items() if k not in _INTERNAL_KEYS}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _content_hash(data: dict[str, Any]) -> str:
    """SHA-256 excluding the ``version`` field (for dev-mode idempotency)."""
    d = {k: v for k, v in data.items() if k not in _INTERNAL_KEYS}
    d = copy.deepcopy(d)
    d.pop("version", None)
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sf_type_for_fp(type_str: str) -> str:
    """Normalise a type string for structural fingerprinting.

    Converts FSBaseType names (e.g. ``"StringType"``) to their Snowflake SQL
    equivalents (e.g. ``"VARCHAR"``), then uppercases the result.  SQL type
    strings that come directly from ``DESCRIBE ONLINE FEATURE TABLE`` are
    already in SQL format and are just uppercased.

    Args:
        type_str: A raw type string from either an authoring spec or DESCRIBE.

    Returns:
        Normalised uppercase SQL type string.
    """
    return _FINGERPRINT_TYPE_MAP.get(type_str, type_str).upper()


def _structural_fingerprint(data: dict[str, Any], include_version: bool = True) -> dict[str, Any]:
    """Build a structural fingerprint from a spec or DESCRIBE-derived dict.

    Extracts ``name``, ``version`` (optional), and sorted output column
    ``name``/``type`` pairs.  Both the authoring-spec path (via
    ``planner.generate_plan``) and the DESCRIBE-reconstruction path (via
    ``state.fetch_applied_state``) call this function, so they always produce
    identical fingerprints for the same deployed feature view.

    For specs: output columns come from ``features[].output_column`` or from
    ``udf.output_columns`` when a UDF is present.

    For DESCRIBE-derived dicts: ``features[].output_column`` is pre-populated
    with the non-primary-key column rows from ``DESCRIBE ONLINE FEATURE TABLE``.

    For ``Entity`` specs the fingerprint additionally folds in
    ``description`` and the sorted ``join_keys`` schema so a non-destructive
    ``UPDATE_ENTITY`` op fires for any of the three editable surfaces of an
    entity tag (``ALLOWED_VALUES`` → join keys, ``COMMENT`` → description,
    name).  Without this widen a ``description``-only edit hashes
    identically and the planner reports ``NO_CHANGE`` (BUG_BASH §11 cascade
    — see ``plans/step11_update_entity_bug_*.plan.md``).

    Args:
        data: Spec dict or DESCRIBE-derived dict.
        include_version: When ``False``, the version field is omitted from the
            fingerprint.  Used for dev-mode idempotency where auto-generated
            timestamp versions should be ignored.

    Returns:
        Dict with keys ``name``, ``version``, ``columns``, and (for
        ``Entity`` kinds only) ``description`` and ``join_keys``.
    """
    name = (data.get("name") or "").upper()
    version = (data.get("version") or "").upper() if include_version else ""

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
                    "name": (jk.get("name") or "").upper(),
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
                        "query": str(src_query),
                    }
                )
            else:
                fp_sources.append(
                    {
                        "name": src_name,
                        "table": src_table,
                        "query": str(src_query),
                    }
                )
        if fp_sources:
            fingerprint["sources"] = sorted(fp_sources, key=lambda x: (x["name"], x["table"], x["query"]))
    return fingerprint


def structural_fingerprint_hash(data: dict[str, Any], include_version: bool = True) -> str:
    """SHA-256 of the structural fingerprint (name + version + output columns).

    This is the canonical hash used for idempotency checks throughout the
    pipeline.  It replaces ``_spec_hash`` so that applied state reconstructed
    from ``DESCRIBE ONLINE FEATURE TABLE`` produces the same hash as the
    authoring spec, enabling ``NO_CHANGE`` detection without a specification
    column in SHOW results.

    Args:
        data: Spec dict or DESCRIBE-derived dict.
        include_version: Passed through to :func:`_structural_fingerprint`.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    fp = _structural_fingerprint(data, include_version=include_version)
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
    ``type`` key), canonicalises ``kind`` to ``"Datasource"``, and
    derives ``source_type`` from the original ``kind`` alias when the
    payload omits it.  After this normalisation a local YAML's
    ``StreamingSource`` payload hashes identically to the runtime
    applied row produced by :func:`state._build_stream_source_object`.

    Args:
        payload: The source spec / applied-state dict.  Non-dict input
            yields ``{}`` defensively so callers can pass through raw
            row dicts without a pre-check.
        strip_type: When ``True``, also remove the ``type`` field.  Set
            on the local payload only because the runtime row produced
            by ``list_stream_sources()`` does not carry the key — the
            applied side keeps any ``type`` value so a hypothetical
            runtime drift surfaces as ``"recreate"``.

    Returns:
        A new dict with the volatile keys removed, ``kind``
        canonicalised to ``"Datasource"``, and ``source_type``
        derived from the original ``kind`` alias when otherwise absent.
        An explicit ``source_type`` value on the input always wins so
        an authored mismatch still surfaces as a structural drift.
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
    """SHA-256 over the canonical-JSON serialisation of a stripped source payload.

    This helper is intentionally distinct from
    :func:`structural_fingerprint_hash` because the latter projects
    every input down to ``{name, version, columns}`` plus FV / Entity
    extensions — which means two source payloads that differ only in
    ``columns``, ``source_type``, ``table``, or ``query`` would
    fingerprint identically.  For a source diff we want to capture the
    full structural shape, so we hash the stripped dict directly.

    Args:
        stripped: A source payload that has had its volatile keys
            removed via :func:`_strip_volatile_source_keys` and its
            ``kind`` canonicalised to ``"Datasource"``.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(stripped, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve_source_description(payload: dict[str, Any]) -> str:
    """Pick the description text from a source payload.

    Looks at ``description`` first, then ``desc``, falling back to
    ``""``.  ``None`` and empty values collapse to ``""`` so the
    desc compare in :func:`compute_source_diff_kind` is symmetric
    across "missing key", "key present with ``None``", and "key
    present with empty string".

    Args:
        payload: Source spec / applied-state dict.

    Returns:
        Whitespace-trimmed, case-preserved description text.
    """
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("description") or payload.get("desc") or ""
    return str(raw).strip()


def _canonicalize_columns(columns: Any) -> list[dict[str, str]]:
    """Project a columns list to a stable, sorted, type-stripped form.

    Used by :func:`compute_source_diff_kind` (and any future caller
    that needs to compare two ``columns:`` lists across the
    authoring / applied diff surface) to wash out cosmetic drift that
    must NOT trigger a destructive ``RECREATE_SOURCE``:

    * Sort entries by uppercased ``name`` so YAML order vs. runtime
      CREATE order does not bump the hash.
    * Strip surrounding whitespace from ``type`` strings so a
      verbose DESCRIBE row formatter (``"  StringType  "``) hashes
      identically to the local YAML's tight ``"StringType"``.
    * Uppercase the ``name`` so a YAML lowercased identifier hashes
      identically to a runtime row that always uppercases.

    Pins the Bug 1 hash-equivalence contract from the
    ``metadata-roundtrip-limitations`` plan (Phase B5).  See the
    test class :class:`tests.test_invariants.
    TestSourceDiffColumnsCanonicalization` for the regression pins.

    Args:
        columns: A ``columns`` list value; non-list inputs return
            ``[]`` defensively (the caller should pass through any
            shape without a pre-check).

    Returns:
        A new, sorted list of ``{"name": <UPPER>, "type": <stripped>}``
        dicts.  Entries lacking a non-empty ``name`` are dropped.
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


def compute_source_diff_kind(
    local_payload: dict[str, Any],
    applied_payload: dict[str, Any],
) -> Literal["no_change", "update_desc_only", "recreate"]:
    """Classify a Datasource diff as no_change | update_desc_only | recreate.

    Pins the §2 truth table from
    ``plans/stream_source_contract.md``.  Used by the planner to pick
    the right ``OpKind`` for an authored source spec that already has
    an applied counterpart:

    * ``"recreate"`` — structural fingerprint differs (column
      addition / removal / type change, ``source_type`` / ``table`` /
      ``query`` / producer-``type`` swap).  Routes to
      ``OpKind.RECREATE_SOURCE`` (destructive; ``--allow-recreate``).
    * ``"update_desc_only"`` — structural fingerprint matches but the
      description text differs.  Routes to ``OpKind.UPDATE_SOURCE``
      (non-destructive).
    * ``"no_change"`` — structural fingerprint AND description match.
      Routes to ``OpKind.NO_CHANGE``.

    Implementation:

    1. Build two structurally-stable stripped payloads.  Description
       fields (``description`` / ``desc`` / ``comment`` / ``details``)
       are dropped on BOTH sides.  The producer-protocol ``type`` is
       dropped on the LOCAL side only because the runtime row produced
       by ``FeatureStore.list_stream_sources()`` never stamps it; the
       applied stripped payload keeps any ``type`` it was given so a
       synthetic / future runtime drift still surfaces as a recreate.
    2. Canonicalise both ``kind`` values to ``"Datasource"`` (mirrors
       :func:`spec_key`'s ``_SOURCE_KIND_ALIASES`` collapse).
    3. Canonicalize ``columns`` symmetrically via
       :func:`_canonicalize_columns` (sort by name, strip type
       whitespace).  When the applied side has no recovered columns
       (empty list / missing key) — the typical legacy DT-text
       recovery shape for a table-backed BatchSource — drop ``columns``
       from BOTH sides so a Bug 1 clean-apply replan does not force
       a destructive ``RECREATE_SOURCE``.  When both sides have
       columns, the canonicalised lists go back into the stripped
       payload before hashing.
    4. Hash each stripped payload via :func:`_source_structural_hash`.
       A mismatch short-circuits the desc compare and returns
       ``"recreate"`` — desc drift is "free" inside a recreate.
    5. When the structural hashes match, compare descriptions
       case-sensitively and whitespace-trimmed.  Equal descriptions →
       ``"no_change"``; unequal → ``"update_desc_only"``.

    Args:
        local_payload: The authored YAML payload for the source
            (``kind: StreamingSource`` / ``BatchSource``, plus
            ``columns`` / ``type`` / ``description`` / ...).
        applied_payload: The applied-state ``AppliedObject.spec_payload``
            for the same source (``kind: Datasource`` once
            canonicalised; carries ``description`` / ``columns`` /
            ``source_type`` etc. as set by
            ``state._build_stream_source_object``).

    Returns:
        One of ``"no_change"``, ``"update_desc_only"``, ``"recreate"``.
    """
    stripped_local = _strip_volatile_source_keys(local_payload, strip_type=True)
    stripped_applied = _strip_volatile_source_keys(applied_payload, strip_type=False)

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
# ``refresh_mode`` is intentionally excluded: Snowflake resolves it to
# INCREMENTAL or FULL, not AUTO, so there is no symmetric default to strip.
# Applied-side refresh_mode is handled asymmetrically in
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

    ``initialize`` is stamped by snowml-core at create time and recovered by
    the applied-state DT-text parsers, but the local-compile side only emits
    it when the operator authored it.  Strip the key when its value matches
    the documented snowml-core default so the two sides hash identically.

    ``refresh_mode`` is intentionally not stripped here: Snowflake resolves it
    to INCREMENTAL or FULL (never AUTO), so applied-side normalisation is
    asymmetric via :func:`_normalize_applied_bfv_for_hash`.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV spec_payload (mutated in
            place).  Non-dict / non-BatchFV inputs are no-ops.
    """
    if not isinstance(inner, dict):
        return
    for key, default in _BFV_OPERATIONAL_DEFAULTS.items():
        if inner.get(key) == default:
            inner.pop(key, None)


def _normalize_applied_bfv_for_hash(
    applied_payload: dict[str, Any],
    local_compiled: dict[str, Any],
) -> dict[str, Any]:
    """Strip Snowflake-resolved fields from applied payload when not authored locally.

    ``refresh_mode`` is stamped by Snowflake's DT engine (INCREMENTAL / FULL)
    even when the operator never specified it. The local compiled spec omits the
    key in that case. Strip it from the applied payload before hashing so the two
    hashes converge on an unspecified-refresh_mode round-trip.

    Must only be called on applied payloads, not local specs.

    Args:
        applied_payload: Full ``AppliedObject.spec_payload``.
        local_compiled: Output of ``compile_to_spec`` for the local authoring dict.

    Returns:
        A normalised deep copy of ``applied_payload``.
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

    snowml-core defaults ``cluster_by`` to the ordered entity column list
    for non-tiled BFVs and ``[<entities>..., "TILE_START"]`` for tiled
    BFVs (those with explicit ``features`` aggregations).  Authoring
    YAML that does not set ``cluster_by`` produces a local-compile spec
    without the key, while the applied side recovers the deployed
    default from the DT text.  Strip the key when the value equals the
    expected default so the two sides hash identically.  An operator who
    authors a non-default cluster list keeps the key in the hash so an
    edit triggers ``RECREATE_FV``.

    The check tolerates both tiled detections (an explicit
    ``feature_aggregation_method`` on the local-compile inner spec, or
    a non-auto-derived ``features`` list on the applied side) so the
    helper works symmetrically on both halves of the diff.

    Args:
        inner: The ``spec`` sub-dict of a BatchFV spec_payload (mutated in
            place).
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
    if cluster_uc == entities_uc or cluster_uc == entities_uc + ["TILE_START"]:
        inner.pop("cluster_by", None)


def _normalise_feature_column_types(features: Any) -> None:
    """Widen narrow integer / float types in a BatchFV ``features`` list.

    Walks each feature entry and rewrites ``source_column.type`` and
    ``output_column.type`` from the authoring narrow types
    (``IntegerType`` / ``FloatType``) to the deployed widened forms
    (``LongType`` / ``DoubleType``).  Snowflake stores Integer columns
    as 64-bit (LongType) and Float columns as 64-bit (DoubleType), so
    the deployed SPECIFICATION JSON reports the widened type even when
    the operator authored the narrower form.  Normalising both sides
    to the widened form before hashing keeps a clean round-trip.

    Args:
        features: The ``features`` list of a BatchFV inner spec (mutated
            in place).  Non-list / non-dict entries are skipped.
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
    """Remove runtime-default keys from a ``columns``/``output_columns`` list.

    Walks the list in place; entries that are not dicts are left
    untouched.  A key is stripped only when its value matches the
    documented runtime default — non-default values (e.g. an explicit
    ``length: 100`` set in the local YAML) survive so they continue to
    contribute to the hash.

    Args:
        columns: The list value of a ``columns`` / ``output_columns``
            spec key (or any non-list, in which case this is a no-op).
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
    """True when a feature entry is a 1:1 ``source_column == output_column`` pass-through.

    Snowflake's ``FROM SPECIFICATION`` deserialiser auto-derives this
    shape from the underlying source columns at CREATE time for any
    non-tiled BatchFV / StreamingFV that omits an explicit ``features:``
    block — so the deployed SPECIFICATION JSON returns N auto-derived
    feature entries even though the authoring YAML had none.  Stripping
    these before :func:`_full_spec_hash` keeps local-compile and
    applied-side hashes in sync on a clean round-trip; explicit
    aggregations (whose feature dict carries any key beyond
    ``source_column`` / ``output_column``, e.g. ``function``,
    ``window_sec``, ``feature_aggregation_method``) are NOT 1:1 and
    survive the normalisation so the planner still sees real semantic
    edits.

    Args:
        feature: A feature dict from ``spec.features[]``.

    Returns:
        ``True`` when both ``output_column.name`` and
        ``source_column.name`` are present and equal AND the entry
        carries no other keys (any extra key — ``function``,
        ``window_sec``, ``feature_aggregation_method``,
        ``feature_granularity[_sec]``, ``window``, etc. — means the
        feature carries semantic info that must influence the hash).
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
    """Decl-side canonicalizer for the operational-drift comparison surface.

    Wraps snowml-core's :func:`feature_store._canonicalize_operational_fields`
    (introduced by Phase A4) with a lazy import so the decl/ package
    does not pull the imperative ``feature_store`` module into its
    initial import graph.  Bug 2 closure — Phase B5 of the
    ``metadata-roundtrip-limitations`` plan.

    Two operational fields are canonicalized symmetrically across the
    local-compile and applied-state halves of the operational-drift
    diff so a clean replan does not emit a phantom ``UPDATE_FV``:

    * ``online_store_type`` — mixed-case / whitespace-padded enum
      values from the deployed SPECIFICATION (``"POSTGRES"``,
      ``"  POSTGRES  "``) collapse to the local-compile form
      (``"postgres"``).
    * ``target_lag_sec`` — ``"5 minutes"`` vs. ``"300 SECONDS"``
      both reduce to ``int(seconds)`` so the diff is stable
      regardless of which textual form the OFT happened to return.

    The ``"DOWNSTREAM"`` sentinel (CRON-task target_lag) is silently
    skipped: the planner recovers the cron expression via the
    companion Task, and ``interval_to_seconds`` would crash on the
    sentinel string.

    The helper is pure: the input ``payload`` is not mutated.  When
    the snowml-core helper is unavailable for any reason (missing
    module, import error during a partial install, etc.), a defensive
    in-package fallback canonicalizes ``online_store_type`` only —
    ``target_lag_sec`` falls through unchanged, which is the
    conservative choice (a missed canonicalization shows up as a
    benign ``UPDATE_FV`` rather than a silent miss).

    Args:
        payload: A dict carrying any subset of ``online_store_type`` /
            ``target_lag`` / ``target_lag_sec``.  Other keys pass
            through untouched.

    Returns:
        A new dict with the canonicalized values; the input is left
        unchanged so callers can chain through other normalisers.
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

    The deployed ``DESCRIBE … TYPE = SPECIFICATION`` JSON either drops
    ``sources`` entirely (BatchFV — empty list) or preserves a subset
    of fields (streaming) that does NOT line up with the local-compile
    shape (``columns``, ``source_type`` semantics, and ``name`` may
    differ from the local authoring name).  The stable identity per
    source is the **binding** — preferring, in order:

    1. ``table`` (recovered for table-backed BatchFVs by
       :func:`state._inject_batch_fv_source_from_dt_text` from the
       offline DT's FROM clause).
    2. ``query`` (the whitespace-normalized SQL body, recovered for
       query-backed BatchFVs by the same path with the Phase-4
       classifier). The query body itself IS the binding identity for
       a query-backed source — author-side and DT-recovery-side names
       differ (the recovery path synthesizes ``<FV>__SOURCE`` while
       the author writes their own ``BatchSource.name``), so the name
       cannot be the identity here. Both sides normalize the SQL via
       :func:`compiler.normalize_sql_whitespace`, so the strings
       converge.
    3. ``name`` (streaming sources, fixture-shaped tests, etc.) when
       neither ``table`` nor ``query`` is recorded.

    Projecting to ``[{"binding": <value>}]`` keeps the FV's source
    binding identity in the hash without false-positive bumps from
    name / column drift on the deployed side.

    The result is sorted so ordering differences across DT-text
    injection and authoring don't matter.

    Args:
        sources: ``spec.sources`` value (any type).

    Returns:
        A sorted list of single-key ``{"binding": <value>}`` dicts.
        Empty list when no source has a ``table``, ``query``, or
        ``name``.
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
            binding = "QUERY:" + query.strip()
        elif isinstance(name, str) and name.strip():
            binding = name.strip().upper()
        if not binding:
            continue
        out.append({"binding": binding})
    out.sort(key=lambda s: s["binding"])
    return out


def _full_spec_hash(spec: dict[str, Any]) -> str:
    """SHA-256 over the full spec JSON, with stable key ordering.

    Used by the planner for full-spec diffs when an ``AppliedObject`` was
    populated from ``DESCRIBE ONLINE FEATURE TABLE <name> TYPE =
    SPECIFICATION`` (i.e. ``from_specification=True``).  Volatile metadata
    fields (see :data:`_VOLATILE_METADATA_KEYS`) are removed before
    hashing so that a tool-version bump does not falsely flag a deployed
    spec as changed.  Spec-level runtime-stamped keys (see
    :data:`_RUNTIME_STAMPED_SPEC_KEYS`) and per-column runtime defaults
    (see :data:`_RUNTIME_STAMPED_COLUMN_DEFAULTS`) are likewise stripped
    so a deploy-time ``target_lag_sec: 0`` or ``length: 16777216``
    stamping does not bump the hash on the next ``snow feature plan``.

    **BatchFV parity normalisation.**  For FeatureView kinds in
    :data:`_FV_HASH_NORMALISE_KINDS` the inner ``spec.sources`` is
    projected to a sorted ``[{table}]`` shape (see
    :func:`_normalise_fv_sources_for_hash`) and ``spec.features`` is
    stripped of 1:1 ``source_column == output_column`` pass-throughs
    (see :func:`_is_auto_derived_feature`).  This is the parity fix for
    snowml-core's lossy ``DESCRIBE … TYPE = SPECIFICATION`` round-trip:
    the deployed BatchFV spec always has ``sources = []`` and N
    auto-derived feature entries, so the comparison must focus on the
    stable structural binding (source ``table``, entities, explicit
    aggregations) rather than the spec-payload boilerplate.  See
    docs/BATCH_FV_BUG_BASH.md §7/§8.

    Args:
        spec: A compiled-spec dict (e.g. the output of
            :func:`spec_compiler.compile_to_spec` or the JSON returned by
            ``DESCRIBE ... TYPE = SPECIFICATION``).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    cleaned = json.loads(json.dumps(spec, default=str))  # deep copy via JSON round-trip
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
            inner["sources"] = _normalise_fv_sources_for_hash(inner.get("sources"))
            features = inner.get("features")
            if isinstance(features, list):
                inner["features"] = [f for f in features if not _is_auto_derived_feature(f)]
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
            _normalise_feature_column_types(inner.get("features"))
    canonical = json.dumps(cleaned, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_local_spec_hash(model_dict: dict[str, Any], database: str, schema: str) -> str:
    """Compile a local FV spec dict and return its full-spec hash.

    The local model dict (produced by :func:`model_to_dict`) is run
    through :func:`spec_compiler.compile_to_spec` so the result has the
    same shape as the JSON returned by ``DESCRIBE ... TYPE =
    SPECIFICATION``.  The hash is then computed via
    :func:`_full_spec_hash` so a deployed spec and a local spec produce
    identical digests when their semantics match.

    This helper is only meaningful for FeatureView kinds.  Entity and
    Source kinds should keep using :func:`structural_fingerprint_hash`.

    Args:
        model_dict: Spec dict from :func:`model_to_dict`.
        database: Snowflake database name (used as ``metadata.database``).
        schema: Snowflake schema name (used as ``metadata.schema``).

    Returns:
        64-character lowercase hex SHA-256 digest of the compiled spec.
    """
    # Imported locally to avoid widening the import surface of this module
    # at package-load time.
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    compiled = compile_to_spec(model_dict, database, schema)
    return _full_spec_hash(compiled)


def fg_content_hash(spec: dict[str, Any]) -> str:
    """SHA-256 over the FeatureGroup identity + sources tuple.

    The basis is intentionally narrow:

    * ``name`` (canonical identifier on the imperative side).
    * ``version`` (FGs pin a specific version; a version edit is a recreate).
    * ``desc`` (free-form description; round-trips through
      ``FeatureGroupMetadata.desc``).
    * ``auto_prefix`` (deployed-side flag controlling output column naming).
    * Sorted tuple of ``(fv_name, fv_version, slice_columns or [], alias or
      None)`` — order-independent so a YAML reorder does NOT trigger a
      destructive recreate.

    Notes:
    * ``output_columns`` is **derived** from the source FVs at register
      time and lives in ``FeatureGroupMetadata`` only as a convenience for
      ``list_feature_groups``; it is NOT in the basis.
    * Unrelated client-side fields (e.g. an exporter scratch field) are
      ignored — the hash function reads only the named keys above.
    * ``alias = ""`` is preserved in the basis tuple as the literal empty
      string (semantically: "no prefix"), distinct from ``alias = None``.

    Args:
        spec: An FG spec dict from either authoring (``model_to_dict`` of
            a :class:`spec_models.FeatureGroup`) or applied state
            (reconstructed from ``FeatureGroupMetadata``).

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
    """Serialize a SpecBase model to a canonical dict for hashing/comparison.

    Renames ``schema_`` to ``schema`` so the key is consistent with raw dict
    representations produced by ``fetch_applied_state``.

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


# ---------------------------------------------------------------------------
# Individual invariant checkers
# ---------------------------------------------------------------------------


def _check_versions(
    spec: dict[str, Any],
    applied: Optional[AppliedObject],
    dev_mode: bool,
) -> list[ValidationResult]:
    """Version management invariants.

    Only ``FeatureView`` and ``FeatureGroup`` kinds are required to carry an
    explicit version string. ``Entity`` and ``Source`` specs are schema-level
    objects without semantic versioning.

    ``VERSION_CONFLICT`` fires only when the local version is *strictly*
    less than the deployed version (i.e. an attempt to "go backwards").
    Equal versions are not a conflict — they're the canonical re-plan
    case and are short-circuited upstream by ``_check_idempotency``
    when the content matches.  When the content has actually changed at
    the same version, the planner emits a destructive ``RECREATE_FV``
    op (gated behind ``--allow-recreate``), which provides natural
    friction without forcing an explicit version bump.  See
    ``plans/planner_revalidate_identical_spec.plan.md`` for the
    BUG_BASH §9 cascade where the prior ``<=`` semantics caused
    spurious ``VERSION_CONFLICT`` errors when the idempotency hash
    diverged on a re-plan of an unchanged FV.

    Args:
        spec: The normalized spec dict to validate.
        applied: The currently-deployed object, or ``None`` for new objects.
        dev_mode: When True, bypass version-conflict checks.

    Returns:
        List of ``ValidationResult`` objects (may be empty).
    """
    results: list[ValidationResult] = []
    name = spec.get("name", "")
    kind = spec.get("kind", "")
    version = spec.get("version")

    # Version is only mandatory for versioned kinds.
    if kind not in _VERSIONED_KINDS:
        return results

    if not version:
        if not dev_mode:
            results.append(
                _result(
                    "ERROR",
                    "MISSING_VERSION",
                    f"{name} has no version. Add a version or use dev_mode for automatic "
                    "timestamp-based versioning.",
                    name,
                )
            )
        # In dev_mode a version will be auto-generated later; no error.
        return results

    if applied is None:
        # New object — no conflict possible.
        return results

    # Object already exists: only flag a *strictly lower* local version.
    # Equal versions on identical content are NO_CHANGE (handled by
    # ``_check_idempotency``); equal versions on changed content surface
    # as a destructive ``RECREATE_FV`` plan op rather than a validator
    # error.
    applied_version = applied.version or ""
    if applied_version and version < applied_version:
        results.append(
            _result(
                "ERROR",
                "VERSION_CONFLICT",
                f"{name}: version '{version}' is lower than the currently deployed "
                f"'{applied_version}'. Bump the version or use overwrite to force.",
                name,
            )
        )
    return results


def _check_idempotency(
    spec: dict[str, Any],
    applied: Optional[AppliedObject],
    dev_mode: bool,
    target_database: str = "",
    target_schema: str = "",
) -> tuple[bool, list[ValidationResult]]:
    """Content-hash idempotency check.

    For FeatureView kinds whose ``applied`` object came from
    ``DESCRIBE ... TYPE = SPECIFICATION`` (``applied.from_specification ==
    True``), the desired-side hash is computed via
    :func:`compute_local_spec_hash` (compile + :func:`_full_spec_hash`) so it
    matches the strategy used by :func:`planner.generate_plan` and the
    state-side ``content_hash`` set by
    :func:`state.fetch_applied_state`.  Otherwise the legacy
    structural-fingerprint comparison is used.

    Args:
        spec: The normalized spec dict to check.
        applied: The currently-deployed object, or ``None`` for new objects.
        dev_mode: When True, compare structural fingerprints excluding version
            so that auto-generated timestamp versions are ignored.
        target_database: Connection target database used when compiling the
            local spec for the full-spec hash path.  Falls back to
            ``spec["database"]`` if empty.
        target_schema: Connection target schema used when compiling the local
            spec for the full-spec hash path.  Falls back to
            ``spec["schema"]`` / ``spec["schema_"]`` if empty.

    Returns:
        Tuple of ``(is_up_to_date, results)``.
    """
    if applied is None:
        return False, []

    name = spec.get("name", "")

    if dev_mode:
        # Compare structural fingerprints without version so auto-generated
        # timestamps do not cause spurious re-deploys.
        current_hash = structural_fingerprint_hash(spec, include_version=False)
        applied_hash = structural_fingerprint_hash(applied.spec_payload, include_version=False)
        if current_hash == applied_hash:
            return True, [_result("WARNING", "NO_CHANGE", f"{name}: content unchanged (dev-mode).", name)]
    else:
        kind = spec.get("kind", "")
        if applied.from_specification and kind in _FV_RECREATE_KINDS:
            db_for_compile = target_database or spec.get("database", "") or ""
            sch_for_compile = target_schema or spec.get("schema", "") or spec.get("schema_", "") or ""
            try:
                current_hash = compute_local_spec_hash(spec, db_for_compile, sch_for_compile)
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
    applied: Optional[AppliedObject],
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

    The ``features`` list lives at different nesting levels depending on
    where ``applied.spec_payload`` came from:

    - ``from_specification=True`` — the payload is the full
      ``DESCRIBE ... TYPE = SPECIFICATION`` JSON, so ``features`` is at
      ``spec_payload['spec']['features']`` (one level deeper).
    - ``from_specification=False`` — the payload is the legacy
      structural-fingerprint shape with ``features`` at the top level.

    Without this kind-aware lookup the ``from_specification=True`` path
    silently sees zero deployed features, so every current feature
    looks "added" and the validator emits a spurious ``COLUMN_ADDED``
    warning per output column on every re-plan against a SPECIFICATION-
    backed AppliedObject.  See ``plans/planner_revalidate_identical_spec.plan.md``.

    Args:
        spec: The local spec dict being validated.
        applied: The deployed applied object (any ``from_specification``).
        name: The feature view's name (used in result messages).
        results: Mutable list accumulating any ``ValidationResult`` rows.
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

    # Collect all entity join-keys from applied state
    applied_entity_join_keys: set[str] = set()
    applied_source_names: set[str] = set()
    applied_fv_names: set[str] = set()
    for obj in applied_state.objects.values():
        payload = obj.spec_payload
        obj_kind = payload.get("kind", "")
        if obj_kind == "Entity":
            for jk in payload.get("join_keys", []):
                jk_name = jk.get("name", "")
                if jk_name:
                    applied_entity_join_keys.add(jk_name)
        elif obj_kind in ("StreamingSource", "BatchSource"):
            src_name = payload.get("name", "")
            if src_name:
                applied_source_names.add(src_name)
        elif "FeatureView" in obj_kind:
            fv_name = payload.get("name", "")
            if fv_name:
                applied_fv_names.add(fv_name)

    if "FeatureView" in kind:
        # Entity column validation
        entity_cols = spec.get("entities", [])
        # Collect entity join-keys from the batch as well (by scanning applied state entities)
        # batch_names contains object names — but we need to check via applied state for join keys
        # We pass batch entity join keys via applied_state for simplicity; callers should merge.
        for ecol in entity_cols:
            if ecol not in applied_entity_join_keys:
                results.append(
                    _result(
                        "ERROR",
                        "MISSING_ENTITY",
                        f"{name}: references entity column '{ecol}' but no entity with that "
                        "join key exists in the current batch or applied state.",
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
    batch_fv_specs: dict[str, dict[str, Any]],
    applied_state: AppliedState,
) -> list[ValidationResult]:
    """Cross-FV invariants for a single FeatureGroup spec.

    Mirrors the imperative ``register_feature_group`` preflight so
    ``snow feature plan`` surfaces shape errors at compile time:

    * ``MISSING_FEATURE_VIEW`` — a referenced source ``(name, version)``
      pair that resolves in neither the local batch nor the applied state.
    * ``FG_DUPLICATE_SOURCE`` — the same ``(name, version)`` appears more
      than once in ``feature_views`` (Pydantic catches this at model
      construction time, but the validator surfaces it for raw-dict /
      compiled-spec callers).
    * ``FG_SOURCE_NOT_ONLINE_POSTGRES`` — a referenced source FV that we
      can prove is **not** ``online: true`` with ``store_type: POSTGRES``.
      Three resolution rules apply, in order:

      1. If the source FV is in ``batch_fv_specs`` and the local payload
         declares ``online_config.store_type``, validate against it.
      2. Else, if the FV is in ``applied_state``, validate against the
         applied payload's ``online`` + ``online_config.store_type`` /
         ``online_store_type`` field (whichever is populated).
      3. Else (source FV unknown to both views, or local payload omits
         ``online_config`` entirely), soft-pass: the imperative API will
         raise the precondition at apply time.  This avoids false
         negatives for the common authoring shape that doesn't carry
         ``store_type`` at all (the declarative layer doesn't expose
         ``store_type:`` as an authoring field today — TODO promote it
         per Phase 2 plan note).

    Args:
        spec: The FG spec dict to validate.
        batch_fv_specs: Map of ``fv_name`` to local FV spec dict (e.g.
            built by the caller from ``batch.specs``).
        applied_state: Applied-state snapshot.

    Returns:
        List of ``ValidationResult`` objects (may be empty).
    """
    results: list[ValidationResult] = []
    if spec.get("kind") != "FeatureGroup":
        return results

    fg_name = spec.get("name", "")

    # Index applied FVs by uppercased name for case-insensitive lookup
    # (Snowflake identifiers are case-insensitive; YAML round-trip can
    # change the case of the source-FV ref).
    applied_fvs_by_name: dict[str, dict[str, Any]] = {}
    for obj in applied_state.objects.values():
        if "FeatureView" in obj.kind:
            applied_fvs_by_name[(obj.name or "").upper()] = obj.spec_payload

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
        local_payload = batch_fv_specs.get(fv_name)
        applied_payload = applied_fvs_by_name.get(fv_name.upper())
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

        # Try local first, then applied — whichever payload can prove the
        # online + Postgres invariant gets used.
        for payload in (local_payload, applied_payload):
            if payload is None:
                continue
            online = bool(payload.get("online", False))
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
                break  # one violation per ref is enough

    return results


def _check_destructive(
    spec: dict[str, Any],
    applied: Optional[AppliedObject],
) -> list[ValidationResult]:
    """Detect changes that require re-materialization.

    The deployed features list lives at different nesting levels
    depending on where ``applied.spec_payload`` came from — see
    :func:`_check_fv_column_evolution` for the same nesting contract.
    Without this kind-aware lookup, ``_check_destructive`` silently
    sees zero deployed features for ``from_specification=True``
    AppliedObjects and never fires the ``DESTRUCTIVE_CHANGE`` error
    that gates plain ``snow feature apply`` against destructive
    aggregation/function edits.

    Args:
        spec: The local spec dict being validated.
        applied: The deployed applied object, or ``None`` for new objects.

    Returns:
        List of ``ValidationResult`` objects (may be empty).
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
        # Compare functions (expression change)
        old_fn = prev_feat.get("function")
        new_fn = feat.get("function")
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


def _check_state_sync(
    spec: dict[str, Any],
    applied: Optional[AppliedObject],
) -> list[ValidationResult]:
    """State-synchronization invariants (concurrent modification / state drift)."""
    if applied is None:
        return []

    results: list[ValidationResult] = []
    name = spec.get("name", "")
    spec_version = spec.get("version") or ""
    applied_version = applied.version or ""

    # State drift: applied has a strictly newer version than what the plan expects.
    # This indicates a concurrent modification between fetch_applied_state and now.
    if applied_version and spec_version and applied_version > spec_version:
        results.append(
            _result(
                "ERROR",
                "STATE_DRIFT",
                f"{name}: state drift detected. Expected version '{spec_version}', found "
                f"'{applied_version}'. Re-run plan to refresh state.",
                name,
            )
        )

    return results


def _check_database_schema_mismatch(
    specs: list[dict[str, Any]],
    target_database: str,
    target_schema: str,
) -> list[ValidationResult]:
    """Warn when spec database/schema fields differ from the connection target.

    When a spec explicitly sets ``database`` or ``schema`` / ``schema_`` to a
    value that differs from the connection target, the connection target will
    always be used — the spec field is ignored.  This check surfaces that
    discrepancy so users are aware.

    Only checks are emitted when both the spec field and the target are
    non-empty (and differ).

    Args:
        specs: List of normalised spec dicts to check.
        target_database: The connection's target database (uppercased for
            comparison).
        target_schema: The connection's target schema (uppercased for
            comparison).

    Returns:
        List of ``ValidationResult`` objects with severity ``WARNING``.
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
    }
)


def batch_feature_view_structural_equivalent(
    local_authoring: dict[str, Any],
    applied_payload: dict[str, Any],
    database: str,
    schema: str,
) -> bool:
    """Return True if local and deployed batch FVs differ only operationally.

    Compares the structural subset of ``spec`` (sources, tiling, entities)
    between ``compile_to_spec(local_authoring)`` and *applied_payload* from
    ``DESCRIBE … TYPE = SPECIFICATION``.  Used by the planner to emit
    ``UPDATE_FV`` instead of ``RECREATE_FV`` when refresh cadence, warehouse,
    description, or online toggles change without semantic FV edits.

    Args:
        local_authoring: Parsed authoring dict for ``BatchFeatureView``.
        applied_payload: Full ``AppliedObject.spec_payload`` from state fetch.
        database: Target database for local compile.
        schema: Target schema for local compile.

    Returns:
        ``True`` when structural projections match; ``False`` otherwise or on
        compile errors.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    try:
        local_compiled = compile_to_spec(local_authoring, database, schema)
    except Exception:
        return False
    _li = local_compiled.get("spec")
    local_inner = _li if isinstance(_li, dict) else {}
    _ri = applied_payload.get("spec")
    remote_inner = _ri if isinstance(_ri, dict) else {}

    def _project(inner: dict[str, Any], ref_inner: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Project the FV inner spec onto its semantic-stable subset.

        Mirrors the BatchFV parity normalisation in
        :func:`_full_spec_hash` (see that docstring for the lossy
        round-trip background): drop ``sources[].name`` / ``columns`` /
        ``source_type`` in favour of a sorted ``[{table}]`` projection,
        and strip auto-derived 1:1 features.  Without this projection
        the deployed BatchFV ``sources = []`` and auto-derived
        ``features`` always disagree with the local-compile shape and
        every BatchFV edit falls through to ``RECREATE_FV``.

        When ``ref_inner`` is provided (the local/authored side), strip
        ``refresh_mode`` from the projection when the operator did not
        author it.  Snowflake always stamps INCREMENTAL or FULL on the DT
        even when the operator never specified a value; without this strip
        a Snowflake-resolved ``refresh_mode`` on the applied side creates a
        spurious structural diff.

        Args:
            inner: The FV inner spec block (``spec`` payload) to project.
            ref_inner: When projecting the applied side, pass the local
                inner spec so asymmetric normalisation can fire.  Omit
                (``None``) when projecting the local side.

        Returns:
            A copy of *inner* restricted to the structurally-stable
            keys with sources / features normalised for cross-side
            comparison.
        """
        block: dict[str, Any] = {}
        for k in _BATCH_FV_STRUCTURAL_INNER_KEYS:
            if k not in inner:
                continue
            block[k] = copy.deepcopy(inner[k])
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


def _check_batch_feature_view_constraints(
    spec: dict[str, Any],
    batch_source_specs: dict[str, dict[str, Any]],
) -> list[ValidationResult]:
    """Validate ``BatchFeatureView`` authoring rules (no UDF, batch sources, tiling)."""
    results: list[ValidationResult] = []
    name = spec.get("name", "")
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

    # ``_is_auto_derived_feature`` recognises the 1:1 passthrough entries
    # snowml-core's ``DESCRIBE … TYPE = SPECIFICATION`` round-trip auto-
    # generates from the source columns for any non-tiled BatchFV that
    # omits an explicit ``features:`` block (already stripped before
    # hashing in :func:`_full_spec_hash` and :func:`_check_idempotency`).
    # The tiling-invariant check must apply the same stripping; otherwise
    # every freshly-exported batch FV (e.g. step 4 of docs/BUG_BASH.md)
    # trips BATCH_FV_TILING_* against a non-tiled FV and blocks every
    # downstream plan.
    raw_features = spec.get("features") or []
    features = [f for f in raw_features if not _is_auto_derived_feature(f)]
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

    # ``aggregation_secondary_keys`` constraints (private preview):
    #
    #   1. Tiled-only — the field is only meaningful when ``features``
    #      declares aggregation windows.  On a non-tiled BFV the value
    #      cannot be honoured and snowml-core would raise late at
    #      register time; we surface the error here instead.
    #   2. Max length 1 — current Snowflake preview cap.
    #
    # ``features`` is already stripped of 1:1 auto-derived passthroughs
    # above (``features = [...]``), so the tiled check below reads from
    # the same projection as the other ``BATCH_FV_TILING_*`` checks.
    secondary_keys = spec.get("aggregation_secondary_keys")
    if isinstance(secondary_keys, list) and secondary_keys:
        if not features:
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES",
                    f"{name}: ``aggregation_secondary_keys`` is only valid on a tiled "
                    "BatchFeatureView (one with explicit ``features`` aggregation windows).",
                    name,
                )
            )
        if len(secondary_keys) > 1:
            results.append(
                _result(
                    "ERROR",
                    "BATCH_FV_SECONDARY_KEYS_MAX_LENGTH",
                    f"{name}: ``aggregation_secondary_keys`` is capped at length 1 in the "
                    "current private preview; got "
                    f"{len(secondary_keys)} entries.",
                    name,
                )
            )

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


def _check_source_compatibility(
    batch_specs: list[dict[str, Any]],
    applied_state: AppliedState,
) -> list[ValidationResult]:
    """Check that source spec changes are compatible with dependent feature views.

    For each source spec in the batch, compares it against the applied version and
    checks whether any feature views in the batch reference affected columns.

    Rules:
    - Column removed and used by a batch FV: ERROR (SOURCE_COLUMN_REMOVED)
    - Column type changed and used by a batch FV: ERROR (SOURCE_COLUMN_TYPE_CHANGED)
    - Column added: WARNING (SOURCE_COLUMN_ADDED)
    - Source absent from batch but referenced by a batch FV: ERROR (SOURCE_DELETED_WITH_DEPENDENTS)

    Args:
        batch_specs: List of normalised spec dicts in the current batch.
        applied_state: Current applied-state snapshot.

    Returns:
        List of ``ValidationResult`` objects.
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
        applied_src_obj: Optional[AppliedObject] = None
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
    dev_mode: bool = False,
    target_database: str = "",
    target_schema: str = "",
) -> list[ValidationResult]:
    """Validate a spec batch against all invariant rules.

    This is a pure function — no side effects, no database access.

    Args:
        batch: The parsed spec batch to validate.
        applied_state: Current applied-state snapshot (from ``fetch_applied_state``).
        dev_mode: When True, skip version checks and use content-hash idempotency.
        target_database: Connection target database; when non-empty, specs with a
            differing ``database`` field produce a ``DB_MISMATCH`` warning.
        target_schema: Connection target schema; when non-empty, specs with a
            differing ``schema`` / ``schema_`` field produce a ``SCHEMA_MISMATCH``
            warning.

    Returns:
        Combined list of ``ValidationResult`` objects (ERRORRs and WARNINGGs).
    """
    results: list[ValidationResult] = []

    # Build lookup data from the batch for cross-spec dependency checks.
    # We need entity join-keys and source names from ALL specs in this batch.
    batch_entity_join_keys: set[str] = set()
    batch_source_names: set[str] = set()
    batch_fv_names: set[str] = set()
    batch_names: set[str] = set()

    for spec_obj in batch.specs:
        data = model_to_dict(spec_obj)
        kind = data.get("kind", "")
        name = data.get("name", "")
        if name:
            batch_names.add(name)
        if kind == "Entity":
            for jk in data.get("join_keys", []):
                jk_name = jk.get("name", "")
                if jk_name:
                    batch_entity_join_keys.add(jk_name)
        elif kind in ("StreamingSource", "BatchSource"):
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

    for spec_obj in batch.specs:
        data = model_to_dict(spec_obj)
        # See ``planner.generate_plan`` for the rationale: connection
        # context is a fallback so bare YAMLs match fully-qualified
        # applied-state keys instead of silently producing phantom
        # diffs.  ``target_database`` / ``target_schema`` flow in via
        # the public ``validate_specs`` signature.
        key = spec_key(data, database=target_database, schema=target_schema)
        applied = applied_state.objects.get(key)

        # 1. Idempotency — check first; if up-to-date, skip remaining checks.
        is_up_to_date, idem_results = _check_idempotency(
            data,
            applied,
            dev_mode,
            target_database=target_database,
            target_schema=target_schema,
        )
        results.extend(idem_results)
        if is_up_to_date:
            continue

        # 2. Version checks
        results.extend(_check_versions(data, applied, dev_mode))

        # 3. Column evolution
        results.extend(_check_column_evolution(data, applied))

        # 4. Dependency validation
        results.extend(_check_dependencies(data, all_source_batch_names, merged_applied))

        if data.get("kind") == "BatchFeatureView":
            results.extend(_check_batch_feature_view_constraints(data, batch_source_specs))

        # 5. Destructive changes
        results.extend(_check_destructive(data, applied))

        # 6. State sync / drift detection
        results.extend(_check_state_sync(data, applied))

    # 7. Database/schema mismatch warnings (batch-level, after per-spec loop)
    if target_database or target_schema:
        all_dicts = [model_to_dict(s) for s in batch.specs]
        results.extend(_check_database_schema_mismatch(all_dicts, target_database, target_schema))

    # 8. Source compatibility checks (batch-level)
    all_dicts = [model_to_dict(s) for s in batch.specs]
    results.extend(_check_source_compatibility(all_dicts, applied_state))

    return results
