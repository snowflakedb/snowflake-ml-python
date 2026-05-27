"""Realtime feature view (RTFV) registration helpers.

Extracted from ``feature_store.py`` to keep that module manageable. Mirrors
the layout of ``streaming_registration.py`` and ``feature_group``:

- ``register_realtime_feature_view`` / ``delete_realtime_feature_view`` —
  RTFV lifecycle entry points called from ``FeatureStore``.
- ``create_realtime_online_feature_table`` — builds the spec + issues
  ``CREATE ONLINE FEATURE TABLE FROM SPECIFICATION``.
- ``append_realtime_listing_row(s)`` — produces ``list_feature_views`` rows.
- ``compose_rtfv_from_metadata`` — rehydrates an RTFV ``FeatureView`` from
  the persisted ``RealtimeConfigMetadata``.
- Small validation/build helpers used by the above.

RTFVs are OFT-only: no DT/View, no preamble/postamble, no async backfill.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_view as fv_mod, online_service
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_group import (
    build_source_refs,
    hydrate_source_refs,
    reject_name_collision,
    unwrap_fv,
    validate_sources_online_postgres,
)
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.metadata_manager import (
    FvSourceRef,
    RealtimeConfigMetadata,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.spec.builder import FeatureViewSpecBuilder, SourceInput
from snowflake.ml.feature_store.spec.enums import FeatureViewKind
from snowflake.ml.feature_store.spec.models import FeatureViewSpec
from snowflake.ml.feature_store.stream_source import _schema_from_dict, _schema_to_dict
from snowflake.snowpark import Row
from snowflake.snowpark.types import DataType, StructType

if TYPE_CHECKING:
    from snowflake.ml.feature_store.feature_store import FeatureStore

logger = logging.getLogger(__name__)


_RTFV_OFT_TARGET_LAG = "0 seconds"


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------


def _build_realtime_feature_view_spec(
    *,
    feature_view: FeatureView,
    feature_view_name: SqlIdentifier,
    version: str,
    target_lag: str,
    database: str,
    schema: str,
) -> FeatureViewSpec:
    """Build a ``FeatureViewSpec`` for an RTFV (no offline configs)."""
    realtime_config = feature_view.realtime_config
    if realtime_config is None:
        raise ValueError(
            f"realtime feature view: cannot build spec for FeatureView "
            f"{feature_view.name!r} because it has no realtime_config."
        )

    fn_source = realtime_config.get_function_source()
    fn_name = realtime_config.get_function_name()
    udf_output_cols: list[tuple[str, DataType]] = [(f.name, f.datatype) for f in realtime_config.output_schema.fields]

    # RequestSource at index 0, upstream FV/Slices after. The builder's
    # ``_validate_realtime`` enforces the contract; we just plumb the list.
    sources: list[SourceInput] = list(realtime_config.sources)

    builder = (
        FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database=database,
            schema=schema,
            name=feature_view.name.resolved(),
            version=version,
        )
        .set_sources(sources)
        .set_udf(
            name=fn_name,
            engine="pandas",
            output_columns=udf_output_cols,
            function_definition=fn_source,
        )
        .set_properties(
            target_lag=target_lag,
        )
    )

    return builder.build()


def _resolve_realtime_upstream_fvs(
    feature_view: FeatureView,
) -> list[Union[FeatureView, FeatureViewSlice]]:
    """Upstream FV/Slice list for an RTFV; empty for non-RTFVs."""
    if feature_view.realtime_config is None:
        return []
    return list(feature_view.realtime_config.feature_view_sources)


def _resolve_realtime_unwrapped_upstream_fvs(
    feature_view: FeatureView,
) -> list[FeatureView]:
    """Bare upstream FVs (slices unwrapped) in source order."""
    return [unwrap_fv(s) for s in _resolve_realtime_upstream_fvs(feature_view)]


# ---------------------------------------------------------------------------
# Validation helpers (called from registration)
# ---------------------------------------------------------------------------


def validate_sources_online_postgres_for_rtfv(realtime_config: RealtimeConfig) -> None:
    """Upstream FVs must be online + Postgres-backed. RequestSource is skipped."""
    upstream: list[Union[FeatureView, FeatureViewSlice]] = list(realtime_config.feature_view_sources)
    if not upstream:
        return
    validate_sources_online_postgres(upstream, consumer_label="RealtimeFeatureView")


def _validate_upstream_registered(realtime_config: RealtimeConfig) -> None:
    """Reject DRAFT upstream FVs; they have no OFT to read from."""
    unregistered: list[str] = []
    for src in realtime_config.feature_view_sources:
        fv = unwrap_fv(src)
        if fv.status == FeatureViewStatus.DRAFT or fv.version is None:
            unregistered.append(f"{fv.name.resolved()}@{fv.version}")
    if unregistered:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError(
                "realtime feature view: upstream FeatureView(s) are not registered: "
                f"{sorted(set(unregistered))}. Call fs.register_feature_view(...) on each "
                "upstream source first."
            ),
        )


def _flatten_join_keys(entities: list[Entity]) -> list[str]:
    """Flatten entity join keys into an ordered, deduplicated list of resolved names."""
    seen: set[str] = set()
    ordered: list[str] = []
    for e in entities:
        for jk in e.join_keys:
            resolved = jk.resolved()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)
    return ordered


def _resolved_entity_names(entities: list[Entity]) -> list[str]:
    """Return entity names resolved to canonical form, in declaration order and de-duplicated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for e in entities:
        name = e.name.resolved() if isinstance(e.name, SqlIdentifier) else SqlIdentifier(e.name).resolved()
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def validate_rtfv_entity_contract(
    declared_entities: list[Entity],
    realtime_config: RealtimeConfig,
) -> None:
    """Every upstream FV's join keys must be a subset of the declared entities,
    and all upstreams that share a join key must agree on its datatype.

    The RTFV's declared entities form the **superset** key tuple (and the OFT
    primary key). Each upstream FV may be keyed by any subset of that tuple;
    at read time, the per-row lookup against each upstream OFT projects the
    request key down to the columns that upstream actually uses. The read
    path also rebuilds the OFT-level schema (PK columns + feature columns)
    from upstream ``output_schema`` declarations, so any two upstreams that
    declare the same join key must agree on its Snowpark datatype --
    otherwise the synthesized schema would silently depend on source order.

    Args:
        declared_entities: Entities declared on the RTFV ``FeatureView``.
        realtime_config: The :class:`RealtimeConfig` whose upstream FVs are
            validated.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if any upstream FV has a join
            key not present in the RTFV's declared entity keys, or if two
            upstream FVs disagree on the datatype of a shared join key.
    """
    declared_keys = _flatten_join_keys(declared_entities)
    declared_set: set[str] = set(declared_keys)

    offenders: list[str] = []
    # Track the (datatype, declaring source) seen for each shared join key so
    # we can pinpoint mismatches with both sources named.
    seen_types: dict[str, tuple[Any, str]] = {}
    type_conflicts: list[str] = []

    for src in realtime_config.feature_view_sources:
        fv = unwrap_fv(src)
        upstream_keys = _flatten_join_keys(list(fv.entities))
        extra = [k for k in upstream_keys if k not in declared_set]
        if extra:
            offenders.append(
                f"{fv.name.resolved()}@{fv.version} keyed by {upstream_keys} " f"(missing from declared: {extra})"
            )

        upstream_label = f"{fv.name.resolved()}@{fv.version}"
        # Canonicalize so quoted-lowercase upstream identifiers match
        # ``upstream_keys`` (which are already ``.resolved()``).
        field_by_name = {SqlIdentifier(f.name).resolved(): f for f in fv.output_schema.fields}
        for jk in upstream_keys:
            field = field_by_name.get(jk)
            if field is None:
                # The upstream declares the entity but the column isn't in its
                # output_schema; nothing to type-check here.
                continue
            datatype = field.datatype
            prev = seen_types.get(jk)
            if prev is None:
                seen_types[jk] = (datatype, upstream_label)
            elif prev[0] != datatype:
                type_conflicts.append(
                    f"join key {jk!r}: source {prev[1]} declares {prev[0]}; "
                    f"source {upstream_label} declares {datatype}"
                )

    if offenders:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "realtime feature view: every upstream FeatureView's join keys must be a subset "
                f"of the realtime feature view's declared entity keys {declared_keys}. "
                f"Offending upstream(s): {sorted(set(offenders))}."
            ),
        )

    if type_conflicts:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "realtime feature view: upstream feature views disagree on the datatype of a "
                "shared join key. All upstreams that include the column must declare the same "
                f"Snowpark datatype. Conflicts: {sorted(set(type_conflicts))}."
            ),
        )

    # Server-side request_df prepends entity columns to RequestSource columns without
    # dedupe; an overlap produces a duplicate-label DataFrame that crashes the UDF
    # with an opaque 500. Reject at register time so the failure is a clear ValueError.
    # Only relevant when a RequestSource was provided.
    request_source = realtime_config.request_source
    if request_source is None:
        return
    canonical_declared = {SqlIdentifier(k).resolved() for k in declared_keys}
    overlap_display: list[str] = []
    for field in request_source.schema.fields:
        if SqlIdentifier(field.name).resolved() in canonical_declared:
            overlap_display.append(field.name)
    if overlap_display:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "realtime feature view: RequestSource.schema declares columns "
                f"{sorted(set(overlap_display))} that overlap with the realtime feature view's "
                f"declared entity join keys {sorted(canonical_declared)}. Entity join keys are "
                "supplied at read time via ``keys=[[...]]`` and prepended server-side to the "
                "request payload; declaring them in RequestSource.schema produces a duplicate "
                "column in the compute_fn's input DataFrame. Remove these columns from "
                "RequestSource.schema."
            ),
        )


def resolve_realtime_join_key_fields(feature_view: FeatureView) -> list[Any]:
    """Resolve ``(name, datatype, nullable=False)`` for every RTFV join key.

    The RTFV ``output_schema`` (= ``realtime_config.output_schema``) contains
    feature columns only -- no PK -- so the Postgres online read path cannot
    derive ``join_key_field_types`` from it. Rebuild the StructField list from
    upstream ``output_schema`` declarations: walk every upstream source FV,
    find each join key the RTFV declares, and pick the first matching field.
    The register-time entity contract (see :func:`validate_rtfv_entity_contract`)
    guarantees that any two upstreams that declare the same key agree on the
    datatype, so first-match is deterministic.

    Args:
        feature_view: The RealtimeFeatureView whose OFT-level join-key fields
            need to be resolved.

    Returns:
        ``[StructField(name, datatype, nullable=False)]`` in the order returned
        by :attr:`FeatureView.ordered_entity_columns`.

    Raises:
        SnowflakeMLException: ``[RuntimeError]`` with
            :data:`error_codes.INTERNAL_PYTHON_ERROR` if the persisted RTFV
            references join keys not declared on any upstream FV, or if two
            upstreams disagree on the datatype of a shared join key. Both
            cases indicate registry corruption -- the user-facing rejection
            happens at register time.
    """
    from snowflake.snowpark.types import StructField

    realtime_config = feature_view.realtime_config
    if realtime_config is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_PYTHON_ERROR,
            original_exception=RuntimeError(
                f"feature view {feature_view.name}/{feature_view.version} is not a "
                "realtime feature view; join-key resolution from upstream sources is "
                "only defined for RealtimeFeatureViews."
            ),
        )

    ordered_keys = feature_view.ordered_entity_columns
    type_by_name: dict[str, Any] = {}
    source_by_name: dict[str, str] = {}
    for src in realtime_config.feature_view_sources:
        underlying = unwrap_fv(src)
        label = f"{underlying.name.resolved()}@{underlying.version}"
        # ``ordered_keys`` are canonical; canonicalize field names too.
        field_by_name = {SqlIdentifier(f.name).resolved(): f for f in underlying.output_schema.fields}
        for key in ordered_keys:
            field = field_by_name.get(key)
            if field is None:
                continue
            previous = type_by_name.get(key)
            if previous is None:
                type_by_name[key] = field.datatype
                source_by_name[key] = label
            elif previous != field.datatype:
                # Registry-corruption path: register-time validator should have
                # rejected this. Defense-in-depth so we don't synthesize a
                # schema that depends on source order.
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_PYTHON_ERROR,
                    original_exception=RuntimeError(
                        f"realtime feature view: join key {key!r} has inconsistent "
                        f"datatypes across upstream sources ({source_by_name[key]} "
                        f"declares {previous}; {label} declares {field.datatype}). "
                        "Registry corruption -- the register-time validator should "
                        "have rejected this."
                    ),
                )

    missing = [k for k in ordered_keys if k not in type_by_name]
    if missing:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_PYTHON_ERROR,
            original_exception=RuntimeError(
                f"realtime feature view: join keys {missing} declared on the feature view "
                "but not found on any upstream source. Registry corruption."
            ),
        )

    return [StructField(k, type_by_name[k], nullable=False) for k in ordered_keys]


def build_rtfv_source_refs(realtime_config: RealtimeConfig) -> list[FvSourceRef]:
    """Translate upstream sources into ``FvSourceRef`` for persistence.

    Thin wrapper around :func:`feature_group.build_source_refs` that
    targets the upstream FVs only (the leading :class:`RequestSource` is
    persisted separately via ``RealtimeConfigMetadata.request_schema_json``).

    Args:
        realtime_config: The :class:`RealtimeConfig` to translate.

    Returns:
        Persistable ``FvSourceRef`` list, one per upstream item, in source
        order.
    """
    return build_source_refs(list(realtime_config.feature_view_sources))


def reject_rtfv_name_collision(
    feature_store: FeatureStore,
    rtfv_name: str,
    rtfv_version: str,
) -> None:
    """Reject if RTFV ``(name, version)`` collides with an existing FV or OFT.

    Thin wrapper around :func:`feature_group.reject_name_collision`; the
    RTFV-specific bits are the consumer label and the OFT name shape
    (``<name>$<version>$ONLINE`` via :meth:`FeatureView._get_online_table_name`).

    Args:
        feature_store: Calling :class:`FeatureStore`.
        rtfv_name: Candidate RTFV name.
        rtfv_version: Candidate RTFV version.
    """
    reject_name_collision(
        feature_store,
        rtfv_name,
        rtfv_version,
        consumer_label="realtime feature view",
        oft_name=FeatureView._get_online_table_name(rtfv_name, rtfv_version),
    )


def request_schema_to_json(schema_struct: StructType) -> str:
    """Serialize a ``StructType`` to JSON via ``_schema_to_dict``."""
    return json.dumps(_schema_to_dict(schema_struct))


def request_schema_from_json(payload: str) -> StructType:
    """Inverse of ``request_schema_to_json``."""
    return _schema_from_dict(json.loads(payload))


def canonicalize_request_context(
    *,
    request_context: Any,
    required: dict[str, str],
    keys: list[list[Any]],
    error_prefix: str,
    pandas_mod: Any,
) -> list[dict[str, Any]]:
    """Validate ``request_context`` shape and return the per-row payload.

    Shared by the single-RTFV read path and the FG-with-RTFV read path,
    so the contract (case-insensitive matching, missing-raises,
    extras-warn-and-drop, length match) cannot drift between them.
    Caller has already short-circuited the no-``RequestSource`` case.

    Args:
        request_context: Caller's DataFrame.
        required: Canonical-to-display mapping the read needs.
        keys: Entity rows; used for length match.
        error_prefix: Per-call leading text on every error/warning, e.g.
            ``"realtime feature view X/v1"`` or ``"feature group X/v1"``.
        pandas_mod: Imported ``pandas`` module.

    Returns:
        Per-row payload as ``list[dict[canonical_name, value]]`` in
        ``keys`` order.

    Raises:
        SnowflakeMLException: ``[ValueError]`` for non-DataFrame input,
            missing required columns, or row count != ``len(keys)``.
    """
    if not isinstance(request_context, pandas_mod.DataFrame):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"{error_prefix}: `request_context` must be a "
                f"pandas.DataFrame; got {type(request_context).__name__}."
            ),
        )

    canonical_provided: dict[str, str] = {}
    for provided_col in request_context.columns:
        try:
            canonical = SqlIdentifier(str(provided_col)).resolved()
        except (ValueError, AttributeError):
            canonical = str(provided_col)
        canonical_provided[canonical] = str(provided_col)

    missing_keys = [k for k in required.keys() if k not in canonical_provided]
    if missing_keys:
        missing_display = [required[k] for k in missing_keys]
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"{error_prefix}: `request_context` is missing required "
                f"columns {missing_display}. RequestSource declares "
                f"{list(required.values())}; provided columns: "
                f"{sorted(canonical_provided.values())}."
            ),
        )

    extra_keys = sorted(set(canonical_provided.keys()) - set(required.keys()))
    if extra_keys:
        extras_display = [canonical_provided[k] for k in extra_keys]
        warnings.warn(
            f"{error_prefix}: `request_context` contains extra columns "
            f"{extras_display} not declared on the RequestSource; these "
            "will be dropped before the request is sent.",
            stacklevel=3,
        )

    column_rename = {canonical_provided[k]: k for k in required.keys()}
    ordered_provided = [canonical_provided[k] for k in required.keys()]
    filtered = request_context[ordered_provided].rename(columns=column_rename)

    if len(filtered) != len(keys):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"{error_prefix}: `request_context` has {len(filtered)} "
                f"row(s) but `keys` has {len(keys)} row(s); each entity "
                "tuple needs exactly one matching request_context row in "
                "the same order."
            ),
        )

    return filtered.to_dict(orient="records")  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Lifecycle: register / OFT create / delete
# ---------------------------------------------------------------------------


def register_realtime_feature_view(
    *,
    feature_store: FeatureStore,
    feature_view: FeatureView,
    version: FeatureViewVersion,
    overwrite: bool,
) -> FeatureView:
    """Register an RTFV as a Postgres-backed OFT + metadata row.

    No DT/View, no offline materialization. ``RealtimeConfig`` is persisted
    under ``(FEATURE_VIEW, name, version, REALTIME_CONFIG)`` so the
    ``compute_fn`` and ``RequestSource`` schema can be rehydrated on read.

    Args:
        feature_store: Calling ``FeatureStore``.
        feature_view: The RTFV ``FeatureView`` to register.
        version: Validated ``FeatureViewVersion`` for the RTFV.
        overwrite: Whether to replace an existing RTFV/OFT with the same
            ``(name, version)``.

    Returns:
        The reconstructed RTFV via ``feature_store.get_feature_view``.

    Raises:
        SnowflakeMLException: If validation fails or OFT / metadata writes
            fail. Non-Snowflake failures are wrapped after best-effort
            rollback of the OFT and metadata row.

    # noqa: DAR401
    """
    # Imported lazily to avoid a circular import at module load.
    from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

    realtime_config = feature_view.realtime_config
    assert realtime_config is not None  # guaranteed by feature_view.is_realtime_feature_view

    # DRAFT upstreams first — most actionable error.
    _validate_upstream_registered(realtime_config)
    validate_sources_online_postgres_for_rtfv(realtime_config)
    validate_rtfv_entity_contract(feature_view.entities, realtime_config)

    canonical_name = feature_view.name.resolved()
    version_str = str(version)
    if not overwrite:
        reject_rtfv_name_collision(feature_store, canonical_name, version_str)

    online_service.assert_online_service_running_with_query_endpoint(
        feature_store._session,
        feature_store._config.database,
        feature_store._config.schema,
        statement_params=feature_store._telemetry_stmp,
    )

    physical_name = FeatureView._get_physical_name(feature_view.name, version)
    online_table_name = FeatureView._get_online_table_name(physical_name)
    fully_qualified_online_name = feature_store._get_fully_qualified_name(online_table_name)

    entity_names = _resolved_entity_names(feature_view.entities)

    # Track only what THIS call created so the failure rollback never tears
    # down a concurrent caller's resources (mirrors register_feature_group /
    # register_feature_view).
    created_resources: list[tuple[_FeatureStoreObjTypes, str]] = []
    metadata_saved = False
    try:
        create_realtime_online_feature_table(
            feature_store=feature_store,
            feature_view=feature_view,
            feature_view_name=physical_name,
            version=version_str,
            overwrite=overwrite,
        )
        created_resources.append((_FeatureStoreObjTypes.ONLINE_FEATURE_TABLE, fully_qualified_online_name))

        metadata = RealtimeConfigMetadata(
            name=canonical_name,
            version=version_str,
            desc=feature_view.desc,
            compute_fn_name=realtime_config.get_function_name(),
            compute_fn_source=realtime_config.get_function_source(),
            sources=build_rtfv_source_refs(realtime_config),
            request_schema_json=(
                request_schema_to_json(realtime_config.request_source.schema)
                if realtime_config.request_source is not None
                else None
            ),
            output_schema_json=request_schema_to_json(realtime_config.output_schema),
            output_columns=[f.name for f in realtime_config.output_schema.fields],
            entity_names=entity_names,
        )
        feature_store._metadata_manager.save_realtime_config(metadata)
        metadata_saved = True

        # Per-entity tags so list-by-entity finds the OFT without a metadata
        # round trip (matches DT/View per-entity tagging).
        for e in feature_view.entities:
            join_keys_csv = ",".join(key.resolved() for key in e.join_keys)
            fv_mod.execute_oft_set_tag(
                feature_store._session,
                fully_qualified_oft_name=fully_qualified_online_name,
                fully_qualified_tag_name=feature_store._get_fully_qualified_name(
                    feature_store._get_entity_name(e.name)
                ),
                tag_value_json=join_keys_csv,
                statement_params=feature_store._telemetry_stmp,
            )
    except Exception as e:
        feature_store._rollback_created_resources(created_resources)
        if metadata_saved:
            try:
                feature_store._metadata_manager.delete_realtime_config(canonical_name, version_str)
            except Exception as cleanup_err:
                logger.warning(
                    f"Best-effort rollback failed to delete RTFV metadata for "
                    f"{feature_view.name}/{version}: {cleanup_err}"
                )
        if isinstance(e, snowml_exceptions.SnowflakeMLException):
            raise
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"Failed to register realtime feature view {feature_view.name}/{version}: {e}"
            ),
        ) from e

    logger.info(f"Registered realtime FeatureView {feature_view.name}/{version} successfully.")
    return feature_store.get_feature_view(feature_view.name, version_str)


def create_realtime_online_feature_table(
    *,
    feature_store: FeatureStore,
    feature_view: FeatureView,
    feature_view_name: SqlIdentifier,
    version: str,
    overwrite: bool,
) -> str:
    """Create the Postgres-backed OFT for an RTFV and tag it."""
    # Late import to avoid feature_store <-> realtime_registration cycle.
    from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

    online_table_name = FeatureView._get_online_table_name(feature_view_name)
    fully_qualified_online_name = feature_store._get_fully_qualified_name(online_table_name)

    # Primary key from declared entities (validated against derived upstream key).
    ordered_join_keys: list[str] = []
    seen_join_keys: set[str] = set()
    for entity in feature_view.entities:
        for join_key in entity.join_keys:
            resolved_key = join_key.resolved()
            if resolved_key not in seen_join_keys:
                seen_join_keys.add(resolved_key)
                ordered_join_keys.append(resolved_key)
    primary_key_clause = fv_mod.build_oft_primary_key_clause(ordered_join_keys)

    spec = _build_realtime_feature_view_spec(
        feature_view=feature_view,
        feature_view_name=feature_view_name,
        version=version,
        target_lag=_RTFV_OFT_TARGET_LAG,
        database=feature_store._config.database.resolved(),
        schema=feature_store._config.schema.resolved(),
    )
    spec_json = spec.to_json()
    source_clause = f"FROM SPECIFICATION $${spec_json}$$"

    warehouse_clause = fv_mod.build_oft_warehouse_clause(feature_view.warehouse, feature_store._default_warehouse)

    try:
        query = fv_mod.build_oft_create_sql(
            fully_qualified_oft_name=fully_qualified_online_name,
            primary_key_clause=primary_key_clause,
            target_lag=_RTFV_OFT_TARGET_LAG,
            source_clause=source_clause,
            warehouse_clause=warehouse_clause,
            overwrite=overwrite,
        )
        feature_store._session.sql(query).collect(statement_params=feature_store._telemetry_stmp)
        feature_store._tag_oft(fully_qualified_online_name, _FeatureStoreObjTypes.REALTIME_FEATURE_VIEW)
    except Exception as e:
        logger.error(f"Failed to create realtime online feature table for {feature_view.name}: {e}")
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"Create realtime online feature table {fully_qualified_online_name} failed: {e}"
            ),
        ) from e

    return str(online_table_name)


def delete_realtime_feature_view(
    *,
    feature_store: FeatureStore,
    feature_view: FeatureView,
) -> None:
    """Drop the OFT + metadata for an RTFV.

    Assumes the caller has already gated on ``status != DRAFT`` and
    ``version is not None``.

    Args:
        feature_store: Calling ``FeatureStore``.
        feature_view: The RTFV ``FeatureView`` to drop.

    Raises:
        SnowflakeMLException: If ``DROP ONLINE FEATURE TABLE`` fails.
    """
    fully_qualified_online_name = feature_view.fully_qualified_online_table_name()
    try:
        feature_store._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {fully_qualified_online_name}").collect(
            statement_params=feature_store._telemetry_stmp
        )
    except Exception as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"Failed to delete online feature table {fully_qualified_online_name}: {e}"
            ),
        ) from e
    feature_store._metadata_manager.delete_realtime_config(feature_view.name.resolved(), str(feature_view.version))
    logger.info(f"Deleted realtime FeatureView {feature_view.name}/{feature_view.version}.")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def append_realtime_listing_row(
    *,
    feature_store: FeatureStore,
    rtfv_metadata: RealtimeConfigMetadata,
    oft_show_row: Optional[Row],
    output_values: list[list[Any]],
    fv_kind_realtime: str,
    default_storage_config_json: str,
) -> None:
    """Append one ``list_feature_views`` row for an RTFV.

    Reads entirely from persisted state: ``RealtimeConfigMetadata`` carries
    the declared entity names captured at register time, and the optional
    ``SHOW ONLINE FEATURE TABLES`` row supplies backend metadata
    (``created_on``, ``owner``). Missing OFT row → ``owner=None,
    created_on=None`` (matches FG listings).

    Args:
        feature_store: Calling ``FeatureStore``.
        rtfv_metadata: Persisted RTFV configuration row.
        oft_show_row: Optional ``SHOW ONLINE FEATURE TABLES`` row for the
            backing OFT. ``None`` means the OFT is missing.
        output_values: Mutated by appending the new row in place.
        fv_kind_realtime: Value emitted in the ``kind`` column (the FS
            module-level constant for the realtime kind).
        default_storage_config_json: Value emitted in the ``storage_config``
            column (the FS module-level default).
    """
    online_config = fv_mod.OnlineConfig(
        enable=True,
        target_lag=fv_mod._NON_BATCH_OFT_TARGET_LAG,
        store_type=OnlineStoreType.POSTGRES,
    )

    created_on = oft_show_row["created_on"] if oft_show_row is not None else None
    owner = oft_show_row["owner"] if oft_show_row is not None else None

    values: list[Any] = [
        rtfv_metadata.name,
        rtfv_metadata.version,
        feature_store._config.database.identifier(),
        feature_store._config.schema.identifier(),
        created_on,
        owner,
        rtfv_metadata.desc,
        list(rtfv_metadata.entity_names or []),
        None,  # refresh_freq
        None,  # refresh_mode
        None,  # scheduling_state
        None,  # warehouse
        None,  # cluster_by
        online_config.to_json(),
        default_storage_config_json,
        None,  # stream_config
        fv_kind_realtime,
    ]

    output_values.append(values)


def append_realtime_listing_rows(
    *,
    feature_store: FeatureStore,
    feature_view_name_prefix: Optional[SqlIdentifier],
    output_values: list[list[Any]],
    fv_kind_realtime: str,
    default_storage_config_json: str,
) -> None:
    """Append RTFV rows to a ``list_feature_views`` output buffer.

    Args:
        feature_store: Calling ``FeatureStore``.
        feature_view_name_prefix: Optional ``SqlIdentifier`` to filter RTFV
            names by leading prefix. ``None`` returns every RTFV.
        output_values: Mutated by appending one row per matching RTFV.
        fv_kind_realtime: Value emitted in the ``kind`` column.
        default_storage_config_json: Value emitted in the ``storage_config``
            column.
    """
    all_rtfv_meta = feature_store._metadata_manager.list_realtime_config_metadata()
    if not all_rtfv_meta:
        return

    if feature_view_name_prefix is not None:
        prefix = feature_view_name_prefix.resolved()
        all_rtfv_meta = [m for m in all_rtfv_meta if m.name.startswith(prefix)]

    # Single SHOW ONLINE FEATURE TABLES sweep keyed by resolved physical name;
    # avoids one catalog round-trip per RTFV row.
    oft_row_by_phys: dict[str, Row] = {
        SqlIdentifier(r["name"], case_sensitive=True).resolved(): r
        for r in feature_store._find_object("ONLINE FEATURE TABLES", None)
    }

    for meta in all_rtfv_meta:
        oft_phys = FeatureView._get_online_table_name(meta.name, meta.version).resolved()
        append_realtime_listing_row(
            feature_store=feature_store,
            rtfv_metadata=meta,
            oft_show_row=oft_row_by_phys.get(oft_phys),
            output_values=output_values,
            fv_kind_realtime=fv_kind_realtime,
            default_storage_config_json=default_storage_config_json,
        )


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def compose_rtfv_from_metadata(
    feature_store: FeatureStore,
    metadata: RealtimeConfigMetadata,
) -> FeatureView:
    """Rehydrate an RTFV ``FeatureView`` from persisted metadata.

    Re-fetches each upstream FV, reapplies slice/alias projection, rebuilds
    the ``RequestSource`` and ``RealtimeConfig`` (which re-runs ``compute_fn``
    validation including the round-trip exec), and builds the FV via
    ``FeatureView._construct_feature_view`` with ``is_realtime=True``.

    Args:
        feature_store: Calling ``FeatureStore`` (used to fetch upstream FVs).
        metadata: Persisted ``RealtimeConfigMetadata`` previously written by
            ``register_feature_view``.

    Returns:
        The reconstructed RTFV ``FeatureView`` with ``version`` populated
        from ``metadata.version`` and ``status=ACTIVE``.
    """
    upstream = hydrate_source_refs(feature_store, metadata.sources)

    request_source: Optional[RequestSource]
    if metadata.request_schema_json is not None:
        request_schema = request_schema_from_json(metadata.request_schema_json)
        request_source = RequestSource(schema=request_schema)
    else:
        request_source = None
    output_schema = request_schema_from_json(metadata.output_schema_json)

    from snowflake.ml.feature_store.realtime_config import (
        _rehydrate_realtime_compute_fn,
    )

    compute_fn = _rehydrate_realtime_compute_fn(metadata.compute_fn_source, metadata.compute_fn_name)
    sources_list: list[Any] = [request_source, *upstream] if request_source is not None else list(upstream)
    realtime_config = RealtimeConfig(
        compute_fn=compute_fn,
        sources=sources_list,
        output_schema=output_schema,
    )

    declared_entities: list[Entity] = []
    seen_entity_names: set[str] = set()
    for src in upstream:
        underlying = unwrap_fv(src)
        for e in underlying.entities:
            if e.name not in seen_entity_names:
                seen_entity_names.add(e.name)
                declared_entities.append(e)

    fv = FeatureView._construct_feature_view(
        name=SqlIdentifier(metadata.name, case_sensitive=True),
        entities=declared_entities,
        feature_df=None,
        timestamp_col=None,
        desc=metadata.desc,
        version=FeatureViewVersion(metadata.version),
        status=FeatureViewStatus.ACTIVE,
        feature_descs={},
        refresh_freq=None,
        database=feature_store._config.database.identifier(),
        schema=feature_store._config.schema.identifier(),
        warehouse=None,
        refresh_mode=None,
        refresh_mode_reason=None,
        initialize="ON_CREATE",
        owner=None,
        infer_schema_df=None,
        session=feature_store._session,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        is_realtime=True,
        realtime_config=realtime_config,
    )
    feature_store._hydrate_postgres_online_service(fv)
    return fv
