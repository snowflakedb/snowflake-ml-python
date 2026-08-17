"""Plan generator: diffs desired specs against applied state.

Produces a topologically-sorted list of ``PlanOp`` objects.
No side effects, no connection access.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.dependencies import topological_sort
from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    _normalize_applied_bfv_for_hash,
    batch_feature_view_structural_equivalent,
    compute_local_spec_hash,
    compute_source_diff_kind,
    fg_content_hash,
    model_to_dict,
    spec_key,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedState,
    ObjectKind,
    Plan,
    PlanOp,
    PlanOptions,
    SpecBatch,
)

# Map spec kind to the appropriate OpKind for CREATE operations.
_CREATE_OP: dict[str, OpKind] = {
    "Entity": OpKind.CREATE_ENTITY,
    "StreamingSource": OpKind.CREATE_SOURCE,
    "BatchSource": OpKind.CREATE_SOURCE,
    "StreamingFeatureView": OpKind.CREATE_FV,
    "RealtimeFeatureView": OpKind.CREATE_FV,
    "BatchFeatureView": OpKind.CREATE_FV,
    "FeatureGroup": OpKind.CREATE_FG,
}

# Map spec kind to the appropriate OpKind for UPDATE operations.
_UPDATE_OP: dict[str, OpKind] = {
    "Entity": OpKind.UPDATE_ENTITY,
    "StreamingSource": OpKind.UPDATE_SOURCE,
    "BatchSource": OpKind.UPDATE_SOURCE,
    "StreamingFeatureView": OpKind.UPDATE_FV,
    "RealtimeFeatureView": OpKind.UPDATE_FV,
    "BatchFeatureView": OpKind.UPDATE_FV,
    "FeatureGroup": OpKind.CREATE_FG,
}

# Map spec kind to the appropriate OpKind for RECREATE operations.  Entities
# are NOT in this table — Snowflake's ``ALTER TAG`` supports both
# ``SET ALLOWED_VALUES`` and ``SET COMMENT``, so an entity hash mismatch
# becomes a non-destructive ``UPDATE_ENTITY`` rather than a drop+create.
# Falling through to ``RECREATE_FV`` (the default before this change) was
# misleading because it caused an Entity diff to surface as a feature-view
# operation in plan output.
_RECREATE_OP: dict[str, OpKind] = {
    "StreamingFeatureView": OpKind.RECREATE_FV,
    "RealtimeFeatureView": OpKind.RECREATE_FV,
    "BatchFeatureView": OpKind.RECREATE_FV,
    # Stream-source path lands the same destructive recreate semantics
    # used for FVs: drop the runtime stream-source via
    # ``FeatureStore.delete_stream_source`` and re-register from the
    # local YAML.  Gated by ``--allow-recreate`` at execute time.
    # BatchSource recreate is informational at the executor boundary
    # because the imperative API has no batch-source DDL, but the
    # planner still emits ``RECREATE_SOURCE`` (destructive) so the
    # --allow-recreate gate captures the operator intent.
    "StreamingSource": OpKind.RECREATE_SOURCE,
    "BatchSource": OpKind.RECREATE_SOURCE,
}

# Feature view kinds eligible for DROP_FV.
_FV_KINDS: frozenset[str] = frozenset({"StreamingFeatureView", "RealtimeFeatureView", "BatchFeatureView"})

# Feature group kinds eligible for DROP_FG.  Singleton today; declared as a
# frozenset so the orphan-drop pass below can use the same membership-test
# style as ``_FV_KINDS`` and a future variant ``FeatureGroup`` family
# extension does not require structural changes here.
_FG_KINDS: frozenset[str] = frozenset({"FeatureGroup"})

# Source kinds that are "virtual" objects in the declarative model — their
# state is fully derivable from the FVs that reference them (the deployed
# DT text on the offline side / StreamConfig on the realtime side). The
# planner uses this set to collapse ``CREATE_SOURCE`` to ``NO_CHANGE``
# when at least one referencing local FV already has an applied
# counterpart. See ``_sources_with_deployed_fv`` and
# ``declarative_feature_store/ARCHITECTURE.md`` §"Datasources as virtual
# objects".
_SOURCE_KIND_ALIASES: frozenset[str] = frozenset({"StreamingSource", "BatchSource"})

# Reason string used for the virtual-source NO_CHANGE branch. Kept as a
# module-level constant so unit tests and plan-file readers can pin the
# exact wording.
_VIRTUAL_SOURCE_NO_CHANGE_REASON = "Virtual source: referenced by deployed feature view(s); no DDL required."


def _sources_with_deployed_fv(
    sorted_dicts: list[dict[str, Any]],
    applied_state: AppliedState,
    database: str,
    schema: str,
) -> frozenset[str]:
    """Return source names referenced by ≥1 local FV with an applied counterpart.

    Sources (``StreamingSource`` / ``BatchSource``) are virtual: there is
    no Snowflake DDL that owns them.  Their lifecycle is bound to the
    FVs that consume them — when a referencing FV is already deployed,
    its source has already been materialised inside the FV's offline
    Dynamic Table / StreamConfig, so re-emitting ``CREATE_SOURCE``
    every plan is operator-noise (and historically a known R1/R3
    failure in ``verify_roundtrip.sh`` because the recovered applied
    ``Datasource`` carries a derived name that cannot collide with the
    authored source name — see ARCHITECTURE.md §"Plan Generation").

    The returned set lets the planner collapse the ``applied is None``
    branch for those sources to ``NO_CHANGE``.

    Args:
        sorted_dicts: Topologically-sorted spec dicts for this batch
            (mirrors :func:`generate_plan` body).
        applied_state: Current applied-state snapshot.
        database: Connection-context database (forwarded to
            :func:`spec_key` so the FV key matches the applied side
            when a YAML omits its ``database:`` field).
        schema: Connection-context schema (same fallback semantics).

    Returns:
        Frozenset of upper-cased source names that are referenced by at
        least one local FV with a matching applied counterpart in
        ``applied_state.objects``.
    """
    referenced: set[str] = set()
    for data in sorted_dicts:
        kind = data.get("kind", "")
        if "FeatureView" not in kind:
            continue
        fv_key = spec_key(data, database=database, schema=schema)
        if fv_key not in applied_state.objects:
            continue
        for src_ref in data.get("sources") or []:
            if isinstance(src_ref, dict):
                src_name = src_ref.get("name")
            else:
                src_name = getattr(src_ref, "name", None)
            if isinstance(src_name, str) and src_name:
                referenced.add(src_name.upper())
    return frozenset(referenced)


# ---------------------------------------------------------------------------
# Operational-drift surface (B6 + B7)
# ---------------------------------------------------------------------------
#
# Operational fields are FV-level knobs that the imperative
# ``FeatureStore.update_feature_view`` API can alter in place (no
# destructive recreate required).  Each FV kind defines its own
# operational subset — fields outside the subset are structural and
# any change must flow through the regular hash-mismatch ``RECREATE_FV``
# path.
#
# Pinned by `plans/metadata-roundtrip-limitations_fe945c22.plan.md`
# §B7.  The RealtimeFV subset is intentionally restricted because A5
# rejects ``refresh_freq`` and ``warehouse`` for RTFVs at the
# imperative layer (no Dynamic Table, no refresh Task).  See
# :meth:`snowflake.ml.feature_store.feature_store.FeatureStore.update_feature_view`.
#
# Coordination notes for downstream B-state and B-invariants streams:
#
# * B-state is wiring the deployed ``desc`` column from
#   ``list_feature_views()`` into the applied ``spec_payload`` so the
#   drift helper has a stable comparand.  The helper probes the
#   top-level applied key first (``spec_payload.get("desc")``) and
#   falls through to the inner spec for robustness.
# * B-invariants owns ``decl/invariants.py``; it will keep
#   ``description`` / ``desc`` out of the structural hash so a desc
#   edit lands as ``UPDATE_FV`` (not ``RECREATE_FV``) via this helper.
_FV_OPERATIONAL_FIELDS_BY_KIND: dict[str, frozenset[str]] = {
    "BatchFeatureView": frozenset({"desc", "warehouse", "refresh_freq", "online_config"}),
    "StreamingFeatureView": frozenset({"desc", "warehouse", "refresh_freq", "online_config"}),
    # RealtimeFV is OFT-only.  ``fs.update_feature_view`` rejects
    # ``refresh_freq`` / ``warehouse`` on this kind (A5), so the planner
    # must mirror the kind boundary: only ``desc`` and ``online_config``
    # are operational here; everything else (including warehouse and
    # refresh_freq) is structural for RTFV.
    "RealtimeFeatureView": frozenset({"desc", "online_config"}),
}


def _resolve_local_desc(local_authoring: dict[str, Any]) -> str:
    """Return the authoring-side description string, normalised."""
    raw = local_authoring.get("description")
    if not isinstance(raw, str) or not raw.strip():
        raw = local_authoring.get("desc")
    return str(raw).strip() if isinstance(raw, str) else ""


def _resolve_applied_desc(applied_payload: dict[str, Any]) -> str:
    """Return the applied-side description, probing the documented locations.

    B-state surfaces the deployed ``desc`` column from
    ``list_feature_views()`` onto ``applied.spec_payload`` at top
    level (mirroring the row key name).  Older applied-state builders
    placed it on the inner ``spec`` block; both shapes are accepted
    so the planner round-trips cleanly across the B-state migration.

    Args:
        applied_payload: The :attr:`AppliedObject.spec_payload` dict
            recovered for the FV.  May carry ``desc`` / ``description``
            at the top level (B-state row shape) or under a nested
            ``spec`` block (legacy DESCRIBE-derived shape).

    Returns:
        The stripped applied-side description string, or ``""`` when
        neither location carries a non-empty value.
    """
    for key in ("desc", "description"):
        raw = applied_payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    inner = applied_payload.get("spec") if isinstance(applied_payload.get("spec"), dict) else None
    if isinstance(inner, dict):
        for key in ("desc", "description"):
            raw = inner.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return ""


def _desc_drifted(local_authoring: dict[str, Any], applied_payload: dict[str, Any]) -> bool:
    """Detect FV description drift between authoring and applied payloads.

    Closes LIMITATIONS L3.  The local YAML uses ``description``; the
    imperative side surfaces it as ``desc`` (matching
    ``list_feature_views().desc``).  Both keys are accepted on either
    side for forward-compat with B-state's eventual enrichment of the
    applied ``spec_payload``.

    Args:
        local_authoring: Authoring-side spec dict.
        applied_payload: Applied-side ``spec_payload``.

    Returns:
        ``True`` when the normalised descriptions differ; ``False``
        otherwise (both empty / both equal).
    """
    return _resolve_local_desc(local_authoring) != _resolve_applied_desc(applied_payload)


def _refresh_freq_drifted(
    local_authoring: dict[str, Any],
    applied_payload: dict[str, Any],
    database: str,
    schema: str,
) -> bool:
    """Detect Dynamic-Table refresh-frequency drift in normalised seconds.

    Closes LIMITATIONS L4.  Local authoring uses ``refresh_freq``
    (string, e.g. ``"5 minutes"``); the applied wire form uses
    ``spec.target_lag_sec`` (int).  Compare on seconds to absorb
    cosmetic surface differences (``"5 minutes"`` vs ``"300 seconds"``).
    The authoring key matches the imperative
    ``FeatureView(refresh_freq=...)`` constructor kwarg one-to-one.

    Pinned contract for the BUG_BASH §5 fixture: when the YAML carries
    BOTH ``target_lag`` and ``refresh_freq``, the compiled
    ``spec.target_lag_sec`` is derived from ``refresh_freq``
    exclusively post-decoupling — ``target_lag`` is OFT staleness only
    and flows through ``OnlineConfig``, not the DT refresh.
    Comparing the compiled value (derived from ``refresh_freq``) is
    therefore the canonical refresh-cadence diff signal.

    The defensive ``target_lag`` / ``target_lag_sec`` authoring-dict
    fallback only fires when ``compile_to_spec`` raises — without it a
    malformed local spec would silently collapse every real edit into
    ``NO_CHANGE``.  Without the gating, a YAML that sets ``target_lag_sec``
    directly (e.g. ``test_planner_no_change_when_only_target_lag_changes_pending_oft_recovery``)
    would surface as drift even though ``target_lag`` no longer
    contributes to the DT cadence.

    Args:
        local_authoring: Authoring-side spec dict.
        applied_payload: Applied-side ``spec_payload``.
        database: Snowflake database used for ``compile_to_spec``.
        schema: Snowflake schema used for ``compile_to_spec``.

    Returns:
        ``True`` when the normalised refresh frequencies differ.
    """
    from snowflake.ml.feature_store.decl.compiler import parse_duration_to_seconds
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    rem_spec = applied_payload.get("spec") if isinstance(applied_payload.get("spec"), dict) else {}
    rem_target_lag_sec = rem_spec.get("target_lag_sec") if isinstance(rem_spec, dict) else None
    if rem_target_lag_sec is None:
        # Fall back to the top-level row-shaped key the imperative
        # ``list_feature_views()`` returns; B-state surfaces it here
        # for offline-only BFVs that lack a SPECIFICATION payload.
        raw_top = applied_payload.get("refresh_freq")
        if isinstance(raw_top, str) and raw_top.strip():
            try:
                _parsed_top = parse_duration_to_seconds(raw_top)
                rem_target_lag_sec = int(_parsed_top) if _parsed_top is not None else None
            except Exception:  # noqa: BLE001
                rem_target_lag_sec = None

    # Primary path: derive local seconds from ``refresh_freq`` (the
    # canonical DT cadence source) and compare against the applied
    # wire-form value.  Mirrors the BUG_BASH §7 fixture pin.
    local_schedule = local_authoring.get("refresh_freq")
    if local_schedule:
        try:
            _parsed_local = parse_duration_to_seconds(local_schedule)
            local_schedule_sec = int(_parsed_local) if _parsed_local is not None else None
            if (
                local_schedule_sec is not None
                and rem_target_lag_sec is not None
                and rem_target_lag_sec != local_schedule_sec
            ):
                return True
        except Exception:  # noqa: BLE001 — defensive: malformed authoring strings fall through
            pass

    # Compile-derived comparison: pick up authoring shapes where
    # ``compile_to_spec`` produces ``spec.target_lag_sec`` (e.g. via
    # ``refresh_freq``).  We only fall through to the authoring-dict
    # ``target_lag_sec`` / ``target_lag`` fallback when compile fails —
    # in normal operation the compiled value is the canonical source.
    try:
        loc = compile_to_spec(local_authoring, database, schema)
    except Exception:  # noqa: BLE001 — authoring-dict fallback below
        local_lag_sec = local_authoring.get("target_lag_sec")
        if local_lag_sec is None and local_authoring.get("target_lag") is not None:
            try:
                _parsed_lag = parse_duration_to_seconds(local_authoring["target_lag"])
                local_lag_sec = int(_parsed_lag) if _parsed_lag is not None else None
            except Exception:  # noqa: BLE001
                local_lag_sec = None
        if local_lag_sec is not None and rem_target_lag_sec is not None and local_lag_sec != rem_target_lag_sec:
            return True
        return False

    loc_spec = loc.get("spec") if isinstance(loc.get("spec"), dict) else {}
    loc_target_lag_sec = loc_spec.get("target_lag_sec") if isinstance(loc_spec, dict) else None
    if loc_target_lag_sec is not None and rem_target_lag_sec is not None and loc_target_lag_sec != rem_target_lag_sec:
        return True
    return False


def _online_config_drifted(
    local_authoring: dict[str, Any],
    applied_payload: dict[str, Any],
    database: str,
    schema: str,
) -> bool:
    """Detect online-store routing drift (online enable + store type).

    Comparison is grounded in ``compile_to_spec`` so the canonical
    ``online_store_type`` form (lowercase string or ``None``) is
    consistent with the applied side.  Falls back to an
    authoring-level ``online`` boolean comparison when compilation
    raises so the drift signal is never lost.

    Args:
        local_authoring: The local authoring dict (the same shape
            consumed by :func:`compile_to_spec`).
        applied_payload: The :attr:`AppliedObject.spec_payload` dict
            recovered for the FV.
        database: Snowflake database to scope the compile under
            (forwarded to :func:`compile_to_spec`).
        schema: Snowflake schema to scope the compile under
            (forwarded to :func:`compile_to_spec`).

    Returns:
        ``True`` when the local and applied sides disagree on online
        enablement or the canonical online-store routing string;
        ``False`` otherwise.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    try:
        loc = compile_to_spec(local_authoring, database, schema)
    except Exception:  # noqa: BLE001 — fall back to authoring-dict comparison
        local_online = bool(local_authoring.get("online", False))
        applied_online = (applied_payload.get("online_store_type") or "") != ""
        return local_online != applied_online
    return loc.get("online_store_type") != applied_payload.get("online_store_type")


def _split_operational_vs_structural(
    local_authoring: dict[str, Any],
    applied_payload: dict[str, Any],
    *,
    operational_fields: frozenset[str],
    database: str,
    schema: str,
) -> bool:
    """Kind-agnostic operational-drift detector for FeatureView updates.

    Called from the planner's hash-matches branch — the structural
    fingerprint has already collided, so any divergence in the fields
    listed by ``operational_fields`` is the operator's intent to
    alter an in-place knob via ``FeatureStore.update_feature_view(...)``.

    Each member of ``operational_fields`` maps to a focused per-field
    helper (``_desc_drifted``, ``_warehouse_drifted``,
    ``_refresh_freq_drifted``, ``_online_config_drifted``); a field
    absent from the set is skipped, so the helper is kind-agnostic.

    Args:
        local_authoring: Authoring-side spec dict (raw, pre-compile).
        applied_payload: Applied-side ``spec_payload`` from the deployed FV.
        operational_fields: Set of operational field names to check.
            See :data:`_FV_OPERATIONAL_FIELDS_BY_KIND`.
        database: Snowflake database for ``compile_to_spec`` calls.
        schema: Snowflake schema for ``compile_to_spec`` calls.

    Returns:
        ``True`` when any of the listed operational fields differ;
        ``False`` otherwise.
    """
    if "desc" in operational_fields and _desc_drifted(local_authoring, applied_payload):
        return True
    if "warehouse" in operational_fields and _warehouse_drifted(local_authoring, applied_payload):
        return True
    if "refresh_freq" in operational_fields and _refresh_freq_drifted(
        local_authoring, applied_payload, database, schema
    ):
        return True
    if "online_config" in operational_fields and _online_config_drifted(
        local_authoring, applied_payload, database, schema
    ):
        return True
    return False


def _batch_fv_operational_drift(
    local_authoring: dict[str, Any],
    applied_payload: dict[str, Any],
    database: str,
    schema: str,
) -> bool:
    """BatchFV-specific operational-drift detector.

    Closes LIMITATIONS L3 (description) and L4 (refresh frequency).
    Thin wrapper over :func:`_split_operational_vs_structural` with
    the BatchFV operational subset
    (:data:`_FV_OPERATIONAL_FIELDS_BY_KIND["BatchFeatureView"]`).
    The wrapper is preserved as a stable API for callers that pin the
    BatchFV semantics directly (e.g. ``tests/test_planner_batch_feature_view.py``).

    Args:
        local_authoring: Authoring-side spec dict (raw, pre-compile).
        applied_payload: Applied-side ``spec_payload`` from the deployed FV.
        database: Snowflake database for ``compile_to_spec`` calls.
        schema: Snowflake schema for ``compile_to_spec`` calls.

    Returns:
        ``True`` when any BatchFV operational knob (description,
        refresh frequency, online routing, warehouse) differs between
        the authoring view and the applied payload; ``False`` otherwise.
    """
    return _split_operational_vs_structural(
        local_authoring,
        applied_payload,
        operational_fields=_FV_OPERATIONAL_FIELDS_BY_KIND["BatchFeatureView"],
        database=database,
        schema=schema,
    )


def _warehouse_drifted(local_authoring: dict[str, Any], applied_payload: dict[str, Any]) -> bool:
    """Detect a meaningful ``warehouse`` drift on the authoring side.

    The deployed ``DESCRIBE … TYPE = SPECIFICATION`` payload does not carry
    the refresh warehouse — it lives only on the offline Dynamic Table.
    State recovery may inject the recovered value into
    ``spec_payload.spec.warehouse`` (DT-text parsing), and the planner
    compares the authored value against whatever the applied side carries:

    * Local omits warehouse → no drift, regardless of applied (the
      imperative ``update_feature_view`` cannot clear a deployed
      warehouse, so a missing local value is a no-op by contract).
    * Local sets warehouse + applied has no recovered warehouse → drift
      (operator is explicitly setting it for the first time).
    * Local sets warehouse + applied recovered a different value → drift.
    * Local sets warehouse + applied recovered the same value → no drift.

    Comparison is case-insensitive because Snowflake identifiers are
    case-insensitive unless quoted; the authoring YAML and the recovered
    DT-text identifier almost always differ in case otherwise.

    Args:
        local_authoring: The authoring-side spec dict (raw, pre-compile).
        applied_payload: The applied-side spec payload from the deployed
            FV (post-``fetch_applied_state`` normalisation).

    Returns:
        ``True`` when the operator has authored a ``warehouse`` value that
        differs from the deployed value (or the deployed value is
        unknown / missing); ``False`` otherwise.
    """
    raw_local = local_authoring.get("warehouse")
    if raw_local is None or (isinstance(raw_local, str) and not raw_local.strip()):
        return False
    local_norm = str(raw_local).strip().upper()
    rem_inner = applied_payload.get("spec") if isinstance(applied_payload.get("spec"), dict) else {}
    raw_rem = rem_inner.get("warehouse") if isinstance(rem_inner, dict) else None
    if raw_rem is None:
        raw_rem = applied_payload.get("warehouse")
    if raw_rem is None or (isinstance(raw_rem, str) and not raw_rem.strip()):
        # Authored a value, deployed-side recovery has none — drift.
        return True
    return str(raw_rem).strip().upper() != local_norm


def generate_plan(
    batch: SpecBatch,
    applied_state: AppliedState,
    options: PlanOptions,
    database: str = "",
    schema: str = "",
) -> Plan:
    """Generate a dependency-ordered execution plan.

    Steps:
    1. Topologically sort the batch specs.
    2. For each spec, compare against ``applied_state``:
       - Not in applied state → ``CREATE`` op.
       - Content hash matches → ``NO_CHANGE`` (skip).
       - Hash differs, non-destructive → ``UPDATE`` op.
       - Hash differs, destructive, ``allow_recreate`` → ``RECREATE`` op.
       - Hash differs, destructive, no flag → warning, no op.
    3. If ``full_directory_mode``, scan applied_state for objects not in the
       batch and emit ``DROP_FV`` / ``DROP_ENTITY`` ops for each orphan.
    4. Return ``Plan`` with ordered ops and any warnings.

    For FeatureView kinds, when ``applied.from_specification`` is True the
    diff uses the **full spec** (compiled via
    :func:`spec_compiler.compile_to_spec` and hashed via
    :func:`invariants._full_spec_hash`).  This catches changes that the
    structural fingerprint misses — UDF source code, source-column schema,
    aggregation windows, and target_lag.  When ``from_specification`` is
    False the planner falls back to the structural fingerprint so existing
    behaviour is preserved.

    Args:
        batch: The validated spec batch.
        applied_state: Current applied-state snapshot.
        options: Plan generation flags.
        database: Snowflake database name used when compiling local specs
            for the full-spec diff path.  Falls back to the spec's own
            ``database`` field when empty.
        schema: Snowflake schema name used when compiling local specs for
            the full-spec diff path.  Falls back to the spec's own
            ``schema`` / ``schema_`` field when empty.

    Returns:
        Topologically-sorted execution plan.
    """
    ops: list[PlanOp] = []
    warnings: list[str] = []

    # Convert models to dicts and sort topologically.
    if batch.specs:
        spec_dicts = [model_to_dict(s) for s in batch.specs]
        sorted_dicts = topological_sort(spec_dicts)
    else:
        sorted_dicts = []

    # Pre-compute the set of source names whose referencing local FVs are
    # already deployed.  Used to collapse ``CREATE_SOURCE`` to
    # ``NO_CHANGE`` for virtual sources whose state is implicit in the
    # deployed FV — see ``_sources_with_deployed_fv``.
    sources_with_deployed_fv = _sources_with_deployed_fv(sorted_dicts, applied_state, database, schema)

    for data in sorted_dicts:
        kind = data.get("kind", "")
        name = data.get("name", "")
        # Thread connection context as a fallback so bare YAMLs (specs
        # lacking ``database:`` / ``schema:``) qualify against the active
        # ``snow`` connection — otherwise the lookup key (e.g. ``Entity::USER``)
        # never collides with the fully-qualified applied-state key
        # (``Entity:JKEW_DB.JKEW_SCHEMA:USER``) and the planner emits
        # phantom CREATE+DROP pairs on a clean round-trip.  See
        # :func:`invariants.spec_key` for the fallback semantics
        # (dict fields always win when present).
        key = spec_key(data, database=database, schema=schema)
        applied = applied_state.objects.get(key)

        db_for_compile = database or data.get("database", "") or ""
        sch_for_compile = schema or data.get("schema", "") or data.get("schema_", "") or ""

        # ----- Source kinds: four-way decision.  Pinned by
        # ``plans/stream_source_contract.md`` §7b.  Placed BEFORE the
        # FV / FG / Entity branches so a source spec never falls
        # through to the FV diff logic.
        #
        # 1. ``applied is None`` + a deployed FV references the source →
        #    NO_CHANGE (virtual override; the source has already been
        #    materialised inside the FV's offline DT / StreamConfig).
        # 2. ``applied is None`` + no deployed FV → CREATE_SOURCE
        #    (genuinely new object).
        # 3. ``applied is not None`` → defer to
        #    :func:`compute_source_diff_kind` and route to
        #    NO_CHANGE / UPDATE_SOURCE / RECREATE_SOURCE.  Runtime row
        #    authority short-circuits the virtual override — once a
        #    ``Datasource`` AppliedObject exists in state.objects we
        #    trust the diff helper over the FV-derivation heuristic.
        if kind in _SOURCE_KIND_ALIASES:
            if applied is None:
                if name.upper() in sources_with_deployed_fv:
                    ops.append(
                        PlanOp(
                            kind=OpKind.NO_CHANGE,
                            name=name,
                            depends_on=[],
                            destructive=False,
                            reason=_VIRTUAL_SOURCE_NO_CHANGE_REASON,
                            payload=data,
                        )
                    )
                else:
                    ops.append(
                        PlanOp(
                            kind=_CREATE_OP[kind],
                            name=name,
                            depends_on=[],
                            destructive=False,
                            reason="New object: not found in applied state.",
                            payload=data,
                        )
                    )
                continue
            # Normalise the local payload to the same canonical shape
            # the applied side carries (uppercase ``name`` /
            # ``database`` / ``schema``) before diffing.  Mirrors the
            # case-folding both :func:`state._build_stream_source_object`
            # and :func:`state._datasource_objects_from_specs` apply at
            # build time; without it a clean round-trip against a
            # bare YAML (no ``database:`` / ``schema:`` keys, name
            # authored in lower case) would falsely surface as
            # ``"recreate"`` purely because the structural fingerprint
            # is case-sensitive.  Mirrors the connection-context
            # fallback semantics in :func:`invariants.spec_key`.
            local_for_diff = dict(data)
            local_for_diff["name"] = (data.get("name") or "").upper()
            eff_db = (data.get("database") or database or "").upper()
            eff_sch = (data.get("schema") or data.get("schema_") or schema or "").upper()
            if eff_db:
                local_for_diff["database"] = eff_db
            if eff_sch:
                local_for_diff["schema"] = eff_sch
            source_diff = compute_source_diff_kind(local_for_diff, applied.spec_payload)
            if source_diff == "no_change":
                ops.append(
                    PlanOp(
                        kind=OpKind.NO_CHANGE,
                        name=name,
                        depends_on=[],
                        destructive=False,
                        reason="Already deployed: no source-level changes detected.",
                        payload=data,
                    )
                )
            elif source_diff == "update_desc_only":
                ops.append(
                    PlanOp(
                        kind=OpKind.UPDATE_SOURCE,
                        name=name,
                        depends_on=[],
                        destructive=False,
                        reason="Source description changed; applying via "
                        "FeatureStore.update_stream_source() (StreamingSource only; "
                        "BatchSource is informational).",
                        payload=data,
                    )
                )
            else:  # "recreate"
                ops.append(
                    PlanOp(
                        kind=_RECREATE_OP[kind],
                        name=name,
                        depends_on=[],
                        destructive=True,
                        reason="Source schema or binding changed; dropping and "
                        "re-registering. Requires --allow-recreate.",
                        payload=data,
                    )
                )
            continue

        if applied is None:
            # New object — CREATE.
            op_kind = _CREATE_OP.get(kind, OpKind.CREATE_FV)
            ops.append(
                PlanOp(
                    kind=op_kind,
                    name=name,
                    depends_on=[],
                    destructive=False,
                    reason="New object: not found in applied state.",
                    payload=data,
                )
            )
            continue

        # Object exists — pick the diff strategy.
        # FeatureView kinds with a deployed full spec get a full-spec diff;
        # FeatureGroup uses its own (much simpler) content hash; everything
        # else stays on the structural fingerprint.
        used_full_spec = False
        if applied.from_specification and kind in _RECREATE_OP:
            try:
                current_hash = compute_local_spec_hash(data, db_for_compile, sch_for_compile)
                used_full_spec = True
            except Exception as exc:  # noqa: BLE001 — defensive fallback
                warnings.append(
                    f"{name}: full-spec diff unavailable ({exc}); " "falling back to structural fingerprint."
                )
                current_hash = structural_fingerprint_hash(data)
        elif kind == "FeatureGroup":
            # FG hash basis is a deliberately narrow tuple over (name,
            # version, desc, auto_prefix, sources[]).  No operational /
            # structural split — the imperative API has no
            # ``update_feature_group``, so every drift is a destructive
            # recreate by construction.  See ``decl/invariants.fg_content_hash``.
            current_hash = fg_content_hash(data)
        else:
            current_hash = structural_fingerprint_hash(data)

        # For BatchFeatureView, re-compute the applied hash against a
        # normalised copy of the applied payload so Snowflake-resolved
        # refresh_mode (INCREMENTAL / FULL) does not produce a spurious hash
        # mismatch when the operator never authored the field.
        applied_compare_hash = applied.content_hash
        if used_full_spec and kind == "BatchFeatureView" and isinstance(applied.spec_payload, dict):
            from snowflake.ml.feature_store.decl.spec_compiler import (  # noqa: PLC0415
                compile_to_spec,
            )

            try:
                local_compiled = compile_to_spec(data, db_for_compile, sch_for_compile)
                applied_normalized = _normalize_applied_bfv_for_hash(applied.spec_payload, local_compiled)
                applied_compare_hash = _full_spec_hash(applied_normalized)
            except Exception:  # noqa: BLE001 — defensive fallback
                pass

        if current_hash == applied_compare_hash:
            # Kind-agnostic operational-drift fast path (B6 + B7).  Structural
            # fingerprint matches, but the operator may have edited an
            # in-place knob — description (L3), refresh cadence (L4),
            # online routing, or warehouse (BatchFV / StreamingFV only).
            # Each kind brings its own operational subset; RealtimeFV is
            # restricted to ``{desc, online_config}`` because A5 rejects
            # ``refresh_freq`` / ``warehouse`` for that kind at the
            # imperative layer.
            if (
                used_full_spec
                and applied.from_specification
                and isinstance(applied.spec_payload, dict)
                and kind in _FV_OPERATIONAL_FIELDS_BY_KIND
                and _split_operational_vs_structural(
                    data,
                    applied.spec_payload,
                    operational_fields=_FV_OPERATIONAL_FIELDS_BY_KIND[kind],
                    database=db_for_compile,
                    schema=sch_for_compile,
                )
            ):
                ops.append(
                    PlanOp(
                        kind=OpKind.UPDATE_FV,
                        name=name,
                        depends_on=[],
                        destructive=False,
                        reason=(
                            f"{kind}: full-spec hash unchanged but operational fields "
                            "(description, refresh cadence, warehouse, or online store "
                            "routing) differ; applying via FeatureStore.update_feature_view()."
                        ),
                        payload=data,
                    )
                )
                continue
            # FV-level backfill.overwrite=True against an existing FV is the
            # operator opt-in for forced re-materialisation.  ``backfill`` is
            # stripped from the structural hash (see invariants._full_spec_hash),
            # so without this branch the plan would be NO_CHANGE and a plain
            # ``snow feature apply`` would never call register_feature_view —
            # the imperative ``overwrite=True`` semantics would be unreachable.
            # We emit a destructive CREATE_FV (re-register with overwrite=True
            # at the executor boundary) so the existing --allow-recreate gate
            # in execute_plan refuses plain apply for this destructive intent.
            backfill_block = data.get("backfill") if isinstance(data.get("backfill"), dict) else None
            backfill_overwrite = bool(backfill_block.get("overwrite")) if backfill_block else False
            if backfill_overwrite and "FeatureView" in kind:
                ops.append(
                    PlanOp(
                        kind=OpKind.CREATE_FV,
                        name=name,
                        depends_on=[],
                        destructive=True,
                        reason="FV-level backfill.overwrite=True forces re-materialisation; "
                        "re-registering via FeatureStore.register_feature_view(overwrite=True). "
                        "Requires --allow-recreate.",
                        payload=data,
                    )
                )
                continue
            # NO_CHANGE — include in plan for display; excluded from SQL by api.py.
            no_change_reason = (
                "Already deployed: no full-spec changes detected."
                if used_full_spec
                else "Already deployed: no structural changes detected."
            )
            ops.append(
                PlanOp(
                    kind=OpKind.NO_CHANGE,
                    name=name,
                    depends_on=[],
                    destructive=False,
                    reason=no_change_reason,
                    payload=data,
                )
            )
            continue

        # Hash differs.  For feature-view kinds, OFTs cannot be updated
        # in place so we emit a destructive RECREATE op.  For Entity
        # kinds, ``ALTER TAG ... SET ALLOWED_VALUES / SET COMMENT`` is
        # supported by Snowflake, so we emit a non-destructive
        # ``UPDATE_ENTITY`` op (handled by ``imperative_executor``)
        # instead of falling through to ``RECREATE_FV``.
        if kind == "Entity":
            ops.append(
                PlanOp(
                    kind=OpKind.UPDATE_ENTITY,
                    name=name,
                    depends_on=[],
                    destructive=False,
                    reason="Entity definition changed (join keys or description); "
                    "applying via FeatureStore.update_entity().",
                    payload=data,
                )
            )
            continue

        if (
            kind == "BatchFeatureView"
            and used_full_spec
            and applied.from_specification
            and isinstance(applied.spec_payload, dict)
            and batch_feature_view_structural_equivalent(data, applied.spec_payload, db_for_compile, sch_for_compile)
        ):
            ops.append(
                PlanOp(
                    kind=OpKind.UPDATE_FV,
                    name=name,
                    depends_on=[],
                    destructive=False,
                    reason="Batch feature view: structural spec unchanged; applying operational "
                    "updates via FeatureStore.update_feature_view() (refresh cadence, warehouse, "
                    "description, or online configuration).",
                    payload=data,
                )
            )
            continue

        if kind == "FeatureGroup":
            # FG hash mismatch → destructive CREATE_FG.  Mirrors the FV-level
            # ``backfill.overwrite=True`` pattern that reuses ``CREATE_FV``
            # with ``destructive=True`` rather than minting a new
            # ``RECREATE_FG`` enum.  The imperative side is a delete +
            # register pair; the executor walks both at apply time.  Gated
            # by ``--allow-recreate`` via the existing ``execute_plan`` gate.
            ops.append(
                PlanOp(
                    kind=OpKind.CREATE_FG,
                    name=name,
                    depends_on=[],
                    destructive=True,
                    reason="FeatureGroup content hash changed (desc, auto_prefix, or "
                    "feature_views[]); dropping and re-registering "
                    "(no imperative update_feature_group). Requires --allow-recreate.",
                    payload=data,
                )
            )
            continue

        recreate_kind = _RECREATE_OP.get(kind, OpKind.RECREATE_FV)
        recreate_reason = (
            "Full-spec change detected (UDF source, source columns, "
            "aggregation, or target lag); dropping and recreating "
            "(OFTs cannot be updated in place)."
            if used_full_spec
            else "Structural change detected; dropping and recreating " "(OFTs cannot be updated in place)."
        )
        ops.append(
            PlanOp(
                kind=recreate_kind,
                name=name,
                depends_on=[],
                destructive=True,
                reason=recreate_reason,
                payload=data,
            )
        )
        if used_full_spec:
            warnings.append(f"{name}: change detected via full-spec diff " "(DESCRIBE ... TYPE = SPECIFICATION).")

    # --- Deletion detection pass (only in full_directory_mode) ---
    if options.full_directory_mode:
        # Build set of spec keys from the batch.  Connection context is
        # forwarded to ``spec_key`` here too so the batch-key set
        # uses the same qualifier as the diff-lookup pass above —
        # otherwise an unqualified spec would round-trip cleanly in the
        # diff loop yet still get marked as a missing batch entry,
        # producing a phantom DROP for an object that's still present
        # in the user's source tree.
        batch_keys: set[str] = set()
        for data in sorted_dicts:
            batch_keys.add(spec_key(data, database=database, schema=schema))

        # Scan applied_state for objects not in the batch → DROP ops
        for key, applied_obj in applied_state.objects.items():
            if key in batch_keys:
                continue  # already handled above
            # This object exists in Snowflake but not in local files → DROP
            kind = applied_obj.kind
            name = applied_obj.name
            version = applied_obj.version or "V1"

            if kind in _FV_KINDS:
                op_kind = OpKind.DROP_FV
            elif kind in _FG_KINDS:
                op_kind = OpKind.DROP_FG
            elif kind == ObjectKind.ENTITY:
                op_kind = OpKind.DROP_ENTITY
            elif kind == ObjectKind.DATASOURCE:
                # Wave 3A §7c: applied ``Datasource`` orphan → DROP_SOURCE.
                # Routes the runtime stream-source through
                # ``FeatureStore.delete_stream_source`` at executor time.
                # The payload's ``kind`` is recovered from the applied
                # ``source_type`` so the executor can pick the
                # StreamingSource vs. BatchSource branch (BatchSource is
                # informational because there is no runtime DDL for
                # batch sources).
                op_kind = OpKind.DROP_SOURCE
            else:
                continue

            if op_kind is OpKind.DROP_SOURCE:
                source_type = (
                    applied_obj.spec_payload.get("source_type") if isinstance(applied_obj.spec_payload, dict) else None
                )
                payload_kind = "StreamingSource" if source_type == "Stream" else "BatchSource"
                drop_payload: dict[str, Any] = {"name": name, "version": None, "kind": payload_kind}
            else:
                drop_payload = {"name": name, "version": version, "kind": kind}

            ops.append(
                PlanOp(
                    kind=op_kind,
                    name=name,
                    depends_on=[],
                    destructive=True,
                    reason="Not present in local spec files: will be dropped.",
                    payload=drop_payload,
                )
            )

    return Plan(ops=ops, warnings=warnings)
