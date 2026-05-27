"""RTFV support in ``generate_training_set`` / ``generate_dataset``.

For each realtime feature view in the user's request, this module joins
its upstream feature views from their offline tables, evaluates
``compute_fn`` in the warehouse via ``map_in_pandas``, and merges the
result back onto the user-visible frame by a synthetic per-spine-row id
(so duplicate entity-key rows on the spine are handled correctly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, cast

import cloudpickle

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import (
    feature_group as fg_mod,
    realtime_config,
    realtime_dataset_udf,
)
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
)
from snowflake.snowpark import DataFrame

# Embed by value so the warehouse worker doesn't need snowflake-ml-python.
cloudpickle.register_pickle_by_value(realtime_dataset_udf)

if TYPE_CHECKING:
    from snowflake.ml.feature_store.feature_store import FeatureStore


# Synthetic per-row id used to merge per-RTFV results back onto the spine
# without entity-key joins (which would fan out for repeated entity keys).
# Dropped from the final result.
_RTFV_SPINE_ROW_ID_COL = "_RTFV_SPINE_ROW_ID"

# Internal column-namespace prefixes used inside a per-RTFV private join.
# Upper-cased so they survive Snowflake's identifier case-folding when
# unquoted; the wrapper strips them so ``compute_fn`` sees authored names.
_RTFV_REQ_PREFIX_FMT = "_RTFV_{idx}_REQ_"
_RTFV_SRC_PREFIX_FMT = "_RTFV_{idx}_SRC_{src_idx}_"

FeatureRef = Union[FeatureView, FeatureViewSlice]


def partition_features(
    features: list[FeatureRef],
) -> tuple[list[FeatureRef], list[FeatureRef]]:
    """Split ``features`` into batch/streaming refs and realtime refs.

    Order is preserved within each list (first-seen). Refs are kept
    distinct -- the same ``FeatureView`` passed twice with different
    aliases or slices yields two entries, mirroring the existing
    ``_join_features`` behavior.

    Args:
        features: The user's feature list. Each item is a
            :class:`FeatureView` or a :class:`FeatureViewSlice`.

    Returns:
        Tuple ``(direct_refs, rtfv_refs)``.
    """
    direct_refs: list[FeatureRef] = []
    rtfv_refs: list[FeatureRef] = []
    for f in features:
        if fg_mod.unwrap_fv(f).is_realtime_feature_view:
            rtfv_refs.append(f)
        else:
            direct_refs.append(f)
    return direct_refs, rtfv_refs


def validate_rtfvs_request_context_contract(rtfvs: list[FeatureView]) -> None:
    """Validate that two RTFVs sharing a RequestSource column agree on its datatype.

    Extracted from the FG-side helper at ``feature_group.py:988-...`` so
    both FG registration and dataset-time validation can call the same
    logic on a plain RTFV list.

    Args:
        rtfvs: The realtime feature views to validate.

    Raises:
        SnowflakeMLException: ``[INVALID_ARGUMENT]`` if two RTFVs declare
            the same RequestSource column with conflicting Snowpark
            datatypes.
    """
    type_by_canonical: dict[str, Any] = {}
    source_by_canonical: dict[str, str] = {}
    display_by_canonical: dict[str, str] = {}
    conflicts: list[str] = []

    for fv in rtfvs:
        rtc = fv.realtime_config
        if rtc is None or rtc.request_source is None:
            continue
        label = f"{fv.name.resolved()}@{fv.version}"
        for field in rtc.request_source.schema.fields:
            canonical = SqlIdentifier(field.name).resolved()
            previous = type_by_canonical.get(canonical)
            if previous is None:
                type_by_canonical[canonical] = field.datatype
                source_by_canonical[canonical] = label
                display_by_canonical[canonical] = field.name
            elif previous != field.datatype:
                conflicts.append(
                    f"request column {display_by_canonical[canonical]!r}: "
                    f"source {source_by_canonical[canonical]} declares {previous}; "
                    f"source {label} declares {field.datatype}"
                )

    if conflicts:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "realtime feature view sources disagree on the datatype of a shared "
                "RequestSource column. The shared request_context payload is keyed by canonical "
                "column name, so all sources that include the column must declare the same "
                f"Snowpark datatype. Conflicts: {sorted(set(conflicts))}."
            ),
        )


def _canonicalize(col: str) -> str:
    """Resolve a column name through ``SqlIdentifier`` for case-insensitive matching."""
    try:
        return SqlIdentifier(str(col)).resolved()
    except (ValueError, AttributeError):
        return str(col)


def _alias_suffix(canonical: str) -> str:
    """Strip surrounding double-quotes from a canonical identifier so it is safe to embed in another quoted alias."""
    return canonical.strip('"')


def validate_rtfv_dataset_inputs(
    features: list[FeatureRef],
    spine_columns: list[str],
) -> None:
    """Validate dataset-generation preconditions for every RTFV in ``features``.

    Checks (after unwrapping any ``FeatureViewSlice``):

    1. Spine carries every column declared in the RTFV's
       ``RequestSource.schema`` (canonical comparison).
    2. Spine carries every join key declared on the RTFV's entities
       (which are a register-time-enforced superset of the upstream FV
       join keys).
    3. Two RTFVs sharing a RequestSource column name agree on datatype
       (delegates to :func:`validate_rtfvs_request_context_contract`).
    4. No upstream FV is in DRAFT state.

    The slice-name-vs-output-schema check is enforced earlier by
    ``FeatureView.slice`` so an invalid slice never reaches this
    validator.

    Args:
        features: The user's feature list.
        spine_columns: Column names on the input spine DataFrame, before
            the synthetic row id is attached.

    Raises:
        SnowflakeMLException: ``[INVALID_ARGUMENT]`` on any failed check,
            with a domain-language message naming the realtime feature
            view and the offending column / state.
    """
    spine_canonical = {_canonicalize(c) for c in spine_columns}
    rtfvs: list[FeatureView] = []

    for ref in features:
        rtfv = fg_mod.unwrap_fv(ref)
        if not rtfv.is_realtime_feature_view:
            continue
        rtfvs.append(rtfv)
        rtc = rtfv.realtime_config
        if rtc is None:
            continue
        rtfv_label = f"{rtfv.name.resolved()}@{rtfv.version}"

        request_source = rtc.request_source
        missing_request_cols = (
            [field.name for field in request_source.schema.fields if _canonicalize(field.name) not in spine_canonical]
            if request_source is not None
            else []
        )
        if missing_request_cols:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"realtime feature view {rtfv_label}: spine_df is missing the request-context "
                    f"column(s) {missing_request_cols}. Realtime feature views require the spine "
                    "DataFrame to carry every request-time column declared on the feature view's "
                    "request source."
                ),
            )

        # The RTFV's entities are a register-time-enforced superset of upstream
        # join keys, so checking the RTFV's own entities is transitive.
        missing_join_keys: list[str] = []
        for entity in rtfv.entities:
            for join_key in entity.join_keys:
                if _canonicalize(str(join_key)) not in spine_canonical:
                    missing_join_keys.append(str(join_key))
        if missing_join_keys:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"realtime feature view {rtfv_label}: spine_df is missing the join key(s) "
                    f"{missing_join_keys} declared on the feature view's entities."
                ),
            )

        for upstream in rtc.feature_view_sources:
            upstream_fv = fg_mod.unwrap_fv(upstream)
            if upstream_fv.status == FeatureViewStatus.DRAFT:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"realtime feature view {rtfv_label}: upstream feature view "
                        f"{upstream_fv.name.resolved()} is a draft and cannot be used to "
                        "generate a training set. Register the upstream feature view first."
                    ),
                )

        # Slice names ⊆ output_schema is enforced earlier by FeatureView.slice.

    if rtfvs:
        validate_rtfvs_request_context_contract(rtfvs)


def attach_synthetic_row_id(spine_df: DataFrame) -> DataFrame:
    """Add ``_RTFV_SPINE_ROW_ID`` and cache the spine.

    The id is generated via Snowflake ``seq8()`` and the result is
    materialized via ``cache_result()`` so the values are stable across
    every downstream reference. Dataset generation joins on this id in
    Stage 3, never on entity keys, so spines with duplicate entity-key
    rows (but distinct label / request-context values) are handled
    correctly.

    Args:
        spine_df: The user-supplied spine.

    Returns:
        A new Snowpark DataFrame with the same columns plus
        ``_RTFV_SPINE_ROW_ID``, materialized.
    """
    from snowflake.snowpark import functions as F

    return cast(DataFrame, spine_df.with_column(_RTFV_SPINE_ROW_ID_COL, F.call_builtin("seq8")).cache_result())


def _build_one_rtfv(
    fs: FeatureStore,
    *,
    rtfv: FeatureView,
    idx: int,
    augmented_spine: DataFrame,
    spine_timestamp_col: Optional[str],
) -> DataFrame:
    """Build one RTFV's private join + ``map_in_pandas`` apply.

    Stages (matching the implementation plan):

    1. Project ``augmented_spine`` to the row id, the entity join keys,
       and the canonical RequestSource columns. Namespace the request
       columns with ``_rtfv_<idx>_req_``.
    2. For each upstream feature view source: project to its join keys
       plus the slice/full feature names, ASOF/LEFT-join into the
       projected spine using the existing PIT/ASOF builders on ``fs``,
       then namespace the upstream feature columns with
       ``_rtfv_<idx>_src_<j>_``.
    3. Run ``map_in_pandas`` with output schema
       ``(_RTFV_SPINE_ROW_ID, *realtime_config.output_schema)``. The
       wrapper strips the namespace prefixes, splits the per-batch frame
       back into the per-source positional pandas DataFrames
       ``compute_fn`` expects (RequestSource first), invokes the
       rehydrated ``compute_fn``, and yields a frame containing the row
       id + the compute_fn outputs.

    Args:
        fs: The owning :class:`FeatureStore`. Used to reach the existing
            ASOF/CTE join builders.
        rtfv: The realtime feature view to evaluate.
        idx: Position of this RTFV ref in the original features list.
            Used to derive unique internal namespaces.
        augmented_spine: Spine carrying ``_RTFV_SPINE_ROW_ID``.
        spine_timestamp_col: Spine timestamp column (canonical), or
            ``None``. Drives ASOF semantics on upstream joins.

    Returns:
        A Snowpark DataFrame with schema
        ``(_RTFV_SPINE_ROW_ID, *realtime_config.output_schema)``. Slice
        projection and prefixing happen in :func:`apply_rtfvs`.

    Raises:
        SnowflakeMLException: ``[INVALID_ARGUMENT]`` if ``rtfv`` does not
            have a ``realtime_config`` (callers should partition first).
    """
    from snowflake.snowpark.dataframe import map_in_pandas
    from snowflake.snowpark.types import StructField, StructType

    rtc = rtfv.realtime_config
    if rtc is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"feature view {rtfv.name.resolved()} is not a realtime feature view; "
                "internal _build_one_rtfv was called with a non-realtime feature view."
            ),
        )

    session = fs._session
    rtfv_label = f"{rtfv.name.resolved()}@{rtfv.version}"

    rtfv_entity_keys = sorted({k.identifier() for e in rtfv.entities for k in e.join_keys})
    request_source = rtc.request_source
    declared_request_field_names = [f.name for f in request_source.schema.fields] if request_source is not None else []
    canonical_req_names = [SqlIdentifier(n).resolved() for n in declared_request_field_names]
    req_prefix = _RTFV_REQ_PREFIX_FMT.format(idx=idx)

    # Always-quoted aliases keep mixed-case / quoted identifiers safe in SQL.
    # ``_alias_suffix`` strips the surrounding quotes so the canonical name
    # can be embedded inside another quoted alias without nested quotes.
    request_aliased_cols = [f"{req_prefix}{_alias_suffix(c)}" for c in canonical_req_names]
    request_alias_clauses = [
        f'{canonical} AS "{aliased}"' for canonical, aliased in zip(canonical_req_names, request_aliased_cols)
    ]

    spine_proj_cols: list[str] = [_RTFV_SPINE_ROW_ID_COL]
    spine_proj_cols.extend(rtfv_entity_keys)
    if spine_timestamp_col is not None:
        spine_proj_cols.append(spine_timestamp_col)

    spine_select_clauses: list[str] = list(spine_proj_cols) + request_alias_clauses

    spine_query = augmented_spine.queries["queries"][-1]
    current_query = f"SELECT {', '.join(spine_select_clauses)} FROM ({spine_query})"
    current_columns: list[str] = list(spine_proj_cols) + request_aliased_cols

    upstream_aliased_cols: list[list[str]] = []
    for src_idx, upstream_ref in enumerate(rtc.feature_view_sources):
        upstream_fv = fg_mod.unwrap_fv(upstream_ref)
        if isinstance(upstream_ref, FeatureViewSlice):
            upstream_feature_names = [c.identifier() for c in upstream_ref.names]
        else:
            upstream_feature_names = [c.identifier() for c in upstream_fv.feature_names]
        upstream_join_keys = sorted({k.identifier() for e in upstream_fv.entities for k in e.join_keys})
        upstream_table_name = upstream_fv.fully_qualified_name()
        src_prefix = _RTFV_SRC_PREFIX_FMT.format(idx=idx, src_idx=src_idx)

        upstream_ts_col: Optional[str] = (
            upstream_fv.timestamp_col.identifier() if upstream_fv.timestamp_col is not None else None
        )
        use_asof = spine_timestamp_col is not None and upstream_ts_col is not None and fs._is_asof_join_enabled()

        upstream_select_cols = list(upstream_join_keys)
        if use_asof:
            upstream_select_cols.append(upstream_ts_col)  # type: ignore[arg-type]
        upstream_select_cols.extend(upstream_feature_names)
        upstream_subquery = f"SELECT {', '.join(upstream_select_cols)} FROM {upstream_table_name}"

        # Alias upstream cols to per-source namespaces so two upstreams sharing
        # an authored feature name don't collide on the joined frame.
        new_aliased_cols = [f"{src_prefix}{_alias_suffix(c)}" for c in upstream_feature_names]
        carry_forward_cols = ", ".join(f"L.{c}" for c in current_columns)
        new_namespaced_feature_clauses = [
            f'R.{c} AS "{aliased}"' for c, aliased in zip(upstream_feature_names, new_aliased_cols)
        ]
        join_keys_str = " AND ".join(f"L.{k} = R.{k}" for k in upstream_join_keys)

        if use_asof:
            join_clause = (
                f"ASOF JOIN ({upstream_subquery}) R "
                f"MATCH_CONDITION (L.{spine_timestamp_col} >= R.{upstream_ts_col}) "
                f"ON {join_keys_str}"
            )
        else:
            join_clause = f"LEFT JOIN ({upstream_subquery}) R ON {join_keys_str}"

        current_query = (
            f"SELECT {carry_forward_cols}, {', '.join(new_namespaced_feature_clauses)} "
            f"FROM ({current_query}) L "
            f"{join_clause}"
        )
        current_columns = current_columns + new_aliased_cols
        upstream_aliased_cols.append(new_aliased_cols)

    joined_df = session.sql(current_query)

    declared_upstream_feature_groups: list[list[str]] = []
    for upstream_ref in rtc.feature_view_sources:
        if isinstance(upstream_ref, FeatureViewSlice):
            declared_upstream_feature_groups.append([c.identifier() for c in upstream_ref.names])
        else:
            declared_upstream_feature_groups.append(
                [c.identifier() for c in fg_mod.unwrap_fv(upstream_ref).feature_names]
            )

    # Wrapper module is registered for cloudpickle by-value at the top of
    # this file so the warehouse worker doesn't need snowflake-ml-python.
    compute_fn_source = rtc.get_function_source()
    compute_fn_name = rtc.get_function_name()

    _wrapper = realtime_dataset_udf.build_wrapper(
        compute_fn_source=compute_fn_source,
        compute_fn_name=compute_fn_name,
        runtime_namespace=realtime_config._RTFV_RUNTIME_NAMESPACE,
        request_aliased_cols=request_aliased_cols,
        upstream_aliased_cols=upstream_aliased_cols,
        declared_request_field_names=declared_request_field_names,
        declared_upstream_feature_groups=declared_upstream_feature_groups,
        rtfv_label=rtfv_label,
    )

    output_schema_fields = [
        StructField(_RTFV_SPINE_ROW_ID_COL, joined_df.schema[_RTFV_SPINE_ROW_ID_COL].datatype),
    ]
    output_schema_fields.extend(rtc.output_schema.fields)
    output_schema = StructType(output_schema_fields)

    return cast(DataFrame, map_in_pandas(joined_df, _wrapper, output_schema, packages=["pandas", "numpy"]))


def _expected_output_cols_for_ref(
    fs: FeatureStore,
    ref: FeatureRef,
    auto_prefix: bool,
) -> list[str]:
    """Predict the user-facing column names a feature ref will contribute.

    Mirrors the prefixing logic in :meth:`FeatureStore._build_cte_query`
    (``identifier.concat_names([prefix, col])``) so the final reorder
    step can locate each ref's columns by name without inspecting the
    intermediate Snowpark schemas.

    Args:
        fs: The owning :class:`FeatureStore`.
        ref: A :class:`FeatureView` or :class:`FeatureViewSlice`.
        auto_prefix: Caller's ``auto_prefix`` flag.

    Returns:
        Column names in declaration order, with the prefix applied if
        either ``auto_prefix`` or ``with_name(...)`` is in effect.
    """
    fv = fg_mod.unwrap_fv(ref)
    if isinstance(ref, FeatureViewSlice):
        names = [c.identifier() for c in ref.names]
    elif fv.is_realtime_feature_view:
        rtc = fv.realtime_config
        assert rtc is not None
        names = [SqlIdentifier(f.name).resolved() for f in rtc.output_schema.fields]
    else:
        names = [c.identifier() for c in fv.feature_names]
    prefix = fs._get_feature_prefix(ref, auto_prefix)
    if prefix:
        return [identifier.concat_names([prefix, n]) for n in names]
    return names


def apply_rtfvs(
    fs: FeatureStore,
    user_visible_df: DataFrame,
    *,
    rtfv_refs_in_order: list[FeatureRef],
    original_features: list[FeatureRef],
    augmented_spine: DataFrame,
    spine_timestamp_col: Optional[str],
    auto_prefix: bool,
) -> DataFrame:
    """Drive Stage 2 + Stage 3: per-RTFV apply, combine, project, prefix, reorder.

    For each RTFV ref, calls :func:`_build_one_rtfv` to produce a private
    DataFrame keyed by ``_RTFV_SPINE_ROW_ID``, LEFT-JOINs that onto
    ``user_visible_df`` by the row id, projects to the slice's ``names``
    if the ref is a slice, and applies the user's ``with_name`` /
    ``auto_prefix`` choice via ``_get_feature_prefix``. After all RTFVs
    are merged, reorders the user-facing columns to match
    ``original_features`` order and drops ``_RTFV_SPINE_ROW_ID``.

    Args:
        fs: The owning :class:`FeatureStore`.
        user_visible_df: Stage 1 result -- the user's direct refs joined
            into the augmented spine via the existing ``_join_features``.
        rtfv_refs_in_order: RTFV refs in their original first-seen order.
        original_features: The full features list (direct + RTFV refs in
            input order). Drives final column reordering.
        augmented_spine: Spine carrying ``_RTFV_SPINE_ROW_ID``.
        spine_timestamp_col: Spine timestamp column, or ``None``.
        auto_prefix: Caller's ``auto_prefix`` flag. Applied to RTFV
            outputs in Stage 3 (it was already applied to direct refs by
            ``_join_features`` in Stage 1).

    Returns:
        The user-facing training-set DataFrame.

    Raises:
        SnowflakeMLException: ``[INTERNAL_PYTHON_ERROR]`` if the final
            column reorder would reference a column that did not show up
            on the merged frame -- typically a ``compute_fn`` that
            returned fewer columns than its declared ``output_schema``.
    """
    private_dfs: list[tuple[FeatureRef, DataFrame, list[str]]] = []
    for ref in rtfv_refs_in_order:
        # First-seen index; identity match (not equality) so two slices of
        # the same FV don't collapse to one namespace.
        idx = next(i for i, f in enumerate(original_features) if f is ref)
        rtfv = fg_mod.unwrap_fv(ref)
        private = _build_one_rtfv(
            fs,
            rtfv=rtfv,
            idx=idx,
            augmented_spine=augmented_spine,
            spine_timestamp_col=spine_timestamp_col,
        )

        rtc = rtfv.realtime_config
        assert rtc is not None
        full_output_names = [SqlIdentifier(f.name).resolved() for f in rtc.output_schema.fields]
        if isinstance(ref, FeatureViewSlice):
            kept_names = [c.identifier() for c in ref.names]
        else:
            kept_names = full_output_names

        prefix = fs._get_feature_prefix(ref, auto_prefix)
        select_clauses: list[str] = [_RTFV_SPINE_ROW_ID_COL]
        output_names: list[str] = []
        if prefix:
            for c in kept_names:
                aliased = identifier.concat_names([prefix, c])
                select_clauses.append(f"{c} AS {aliased}")
                output_names.append(aliased)
        else:
            for c in kept_names:
                select_clauses.append(c)
                output_names.append(c)

        private_query = private.queries["queries"][-1]
        projected = fs._session.sql(f"SELECT {', '.join(select_clauses)} FROM ({private_query})")
        private_dfs.append((ref, projected, output_names))

    # Merge per-RTFV results onto the user-visible frame by the synthetic
    # row id (entity keys would fan out for repeated spine rows).
    merged = user_visible_df
    for _ref, projected, _names in private_dfs:
        merged = merged.join(projected, on=_RTFV_SPINE_ROW_ID_COL, how="left")

    # Reorder to match the original features list; unclaimed columns
    # (labels, spine extras, the synthetic id) stay in their existing position.
    ref_cols_by_position: list[list[str]] = []
    claimed: set[str] = set()
    for ref in original_features:
        if fg_mod.unwrap_fv(ref).is_realtime_feature_view:
            for r, _df, names in private_dfs:
                if r is ref:
                    ref_cols_by_position.append(names)
                    claimed.update(names)
                    break
            else:  # pragma: no cover - guaranteed by earlier partition
                ref_cols_by_position.append([])
        else:
            names = _expected_output_cols_for_ref(fs, ref, auto_prefix)
            ref_cols_by_position.append(names)
            claimed.update(names)

    merged_columns: list[str] = list(merged.columns)
    passthrough_cols = [c for c in merged_columns if c not in claimed and c != _RTFV_SPINE_ROW_ID_COL]

    final_order = passthrough_cols + [c for group in ref_cols_by_position for c in group]

    # Surface column-emission bugs loudly: a missing column means either our
    # ``_expected_output_cols_for_ref`` prediction drifted from the actual
    # join output, or ``compute_fn`` returned fewer columns than its declared
    # ``output_schema``. Either is worth a clear error rather than a silent
    # NULL column the caller has to chase.
    available = set(merged.columns)
    missing = [c for c in final_order if c not in available]
    if missing:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_PYTHON_ERROR,
            original_exception=RuntimeError(
                f"realtime feature view dataset generation: expected output column(s) "
                f"{missing} not produced. compute_fn output may not match the declared "
                f"output schema. Available columns: {sorted(available)}."
            ),
        )
    return cast(DataFrame, merged.select(final_order))
