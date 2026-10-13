"""Unit tests for BatchFeatureView validation invariants."""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.invariants import (
    _AGG_REQUIRED_OUTPUT_TYPE,
    validate_specs,
)
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    Feature,
    FeatureView,
    FSColumn,
)
from snowflake.ml.feature_store.decl.types import AppliedState, SpecBatch
from snowflake.ml.test_utils import pytest_driver


def _entity(name: str = "USER") -> Entity:
    return Entity(
        kind="Entity",
        name=name,
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )


def _batch_source(name: str = "EVENTS", table: str = "RAW_EVENTS") -> BatchSource:
    return BatchSource(
        kind="BatchSource",
        name=name,
        table=table,
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )


def _batch_fv(
    *,
    features: list[Feature] | None = None,
    sources: list[Any] | None = None,
    timestamp_col: str | None = None,
    feature_granularity_sec: int | None = None,
    feature_aggregation_method: str | None = None,
    refresh_freq: str | None = None,
) -> FeatureView:
    src = sources or [{"name": "EVENTS", "source_type": "Batch"}]
    fv = FeatureView(
        kind="BatchFeatureView",
        name="BFV_TEST",
        version="V1",
        database="DB1",
        schema_="SC1",
        online=False,
        entities=["USER_ID"],
        sources=src,
    )
    if features is not None:
        fv = fv.model_copy(update={"features": features})
    if timestamp_col is not None:
        fv = fv.model_copy(update={"timestamp_col": timestamp_col})
    if feature_granularity_sec is not None:
        fv = fv.model_copy(update={"feature_granularity_sec": feature_granularity_sec})
    if feature_aggregation_method is not None:
        fv = fv.model_copy(update={"feature_aggregation_method": feature_aggregation_method})
    if refresh_freq is not None:
        fv = fv.model_copy(update={"refresh_freq": refresh_freq})
    return fv


def _batch(specs: list[Any]) -> SpecBatch:
    return SpecBatch(specs=specs)


def test_batch_fv_rejects_udf() -> None:
    from snowflake.ml.feature_store.decl.spec_models import UDF

    fv = FeatureView(
        kind="BatchFeatureView",
        name="BFV_TEST",
        version="V1",
        database="DB1",
        schema_="SC1",
        online=False,
        entities=["USER_ID"],
        sources=[{"name": "EVENTS", "source_type": "Batch"}],
        udf=UDF(name="fn", engine="pandas", function_definition="def fn(df): return df"),
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    codes = {r.code for r in results}
    assert "BATCH_FV_UDF_FORBIDDEN" in codes


def test_batch_fv_rejects_stream_source_type() -> None:
    fv = _batch_fv(sources=[{"name": "S", "source_type": "Stream"}])
    results = validate_specs(_batch([_entity(), fv]), AppliedState(objects={}))
    assert any(r.code == "BATCH_FV_INVALID_SOURCE_TYPE" for r in results)


def test_batch_fv_requires_batch_source_table() -> None:
    fv = _batch_fv(sources=[{"name": "MISSING", "source_type": "Batch"}])
    results = validate_specs(_batch([_entity(), fv]), AppliedState(objects={}))
    assert any(r.code in ("MISSING_SOURCE", "BATCH_FV_SOURCE_NO_TABLE") for r in results)


def test_batch_fv_tiled_requires_timestamp_and_granularity() -> None:
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="CNT", type="LongType"),
                output_column=FSColumn(name="SUM_CNT", type="LongType"),
                function="sum",
                window_sec=3600,
            )
        ],
        refresh_freq="1 minute",
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    codes = {r.code for r in results}
    assert "BATCH_FV_TILING_TIMESTAMP" in codes
    assert "BATCH_FV_TILING_GRANULARITY" in codes
    assert "BATCH_FV_TILING_AGG_METHOD" in codes


def test_batch_fv_tiled_requires_refresh() -> None:
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="CNT", type="LongType"),
                output_column=FSColumn(name="SUM_CNT", type="LongType"),
                function="sum",
                window_sec=3600,
            )
        ],
        timestamp_col="TS",
        feature_granularity_sec=300,
        feature_aggregation_method="tiles",
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    assert any(r.code == "BATCH_FV_TILING_REFRESH" for r in results)


def test_batch_fv_tiled_valid_passes_core_rules() -> None:
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="CNT", type="LongType"),
                output_column=FSColumn(name="SUM_CNT", type="LongType"),
                function="sum",
                window_sec=3600,
            )
        ],
        timestamp_col="TS",
        feature_granularity_sec=300,
        feature_aggregation_method="tiles",
        refresh_freq="1 minute",
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    batch_errors = [r for r in results if r.code and r.code.startswith("BATCH_FV_")]
    assert not batch_errors


def test_batch_fv_passthrough_auto_derived_features_skip_tiling_checks() -> None:
    """Auto-derived 1:1 passthrough features must not trigger tiling-invariant errors.

    Reproduces the ``docs/BUG_BASH.md`` step-6 cascade against
    ``JKEW_DB.JKEW_SCHEMA``: after ``snow feature init`` exports the
    deployed batch FV ``MY_BATCH_FV_BATCH_DECL`` (originally registered
    with no explicit ``features:`` block), snowml-core's
    ``DESCRIBE ... TYPE = SPECIFICATION`` round-trip reconstructs the
    spec with N auto-derived passthrough feature entries
    (``output_column.name == source_column.name``, no other keys).
    ``_check_batch_feature_view_constraints`` then fires
    ``BATCH_FV_TILING_TIMESTAMP`` / ``BATCH_FV_TILING_GRANULARITY`` /
    ``BATCH_FV_TILING_AGG_METHOD`` against a non-tiled FV, blocking
    every downstream plan.  ``_is_auto_derived_feature`` already
    recognises these entries elsewhere (``_full_spec_hash`` and
    ``_check_idempotency``); the batch-FV constraint check must apply
    the same stripping before testing the "features non-empty" branch.
    """
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="EVENT_TS", type="TimestampType"),
                output_column=FSColumn(name="EVENT_TS", type="TimestampType"),
            ),
            Feature(
                source_column=FSColumn(name="METRIC_VAL", type="DoubleType"),
                output_column=FSColumn(name="METRIC_VAL", type="DoubleType"),
            ),
        ],
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    tiling_errors = [r for r in results if r.code and r.code.startswith("BATCH_FV_TILING_")]
    assert tiling_errors == [], (
        "auto-derived passthrough features must be stripped before tiling-invariant checks; "
        f"got {[r.code for r in tiling_errors]}"
    )


def test_batch_fv_mixed_features_still_validate_tiling_when_aggregations_present() -> None:
    """Regression guard: explicit aggregations must still trigger tiling checks.

    Pairs with ``test_batch_fv_passthrough_auto_derived_features_skip_tiling_checks``:
    the stripping helper must keep real aggregation features in the
    constraint check's view so an FV with even one ``function`` +
    ``window_sec`` feature still surfaces ``BATCH_FV_TILING_*`` when
    timestamp/granularity/agg_method are missing.  Without this
    asymmetry, the fix above would silently disable the entire
    tiling-invariant family.
    """
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="EVENT_TS", type="TimestampType"),
                output_column=FSColumn(name="EVENT_TS", type="TimestampType"),
            ),
            Feature(
                source_column=FSColumn(name="CNT", type="LongType"),
                output_column=FSColumn(name="SUM_CNT", type="LongType"),
                function="sum",
                window_sec=3600,
            ),
        ],
        refresh_freq="1 minute",
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    codes = {r.code for r in results}
    assert "BATCH_FV_TILING_TIMESTAMP" in codes
    assert "BATCH_FV_TILING_GRANULARITY" in codes
    assert "BATCH_FV_TILING_AGG_METHOD" in codes


def test_batch_fv_continuous_does_not_inherit_streaming_granularity_default() -> None:
    """Negative-case regression for the streaming CONTINUOUS-granularity
    default.

    The compiler's compile-time default for
    ``feature_aggregation_method: continuous`` MUST scope strictly to
    ``StreamingFeatureView``; ``BatchFeatureView`` continues to fail
    fast through the existing ``BATCH_FV_TILING_GRANULARITY``
    invariant.  Without this guard, a BFV authored as
    ``continuous`` without ``feature_granularity_sec`` would silently
    receive the streaming default and slip past the validator into
    snowml-core's ``FeatureView.__init__``, which then raises a much
    less actionable ``feature_aggregation_method is only supported
    for streaming feature views.``  Pinning the invariant codes here
    keeps the imperative side decoupled from the streaming default.
    """
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="CNT", type="LongType"),
                output_column=FSColumn(name="SUM_CNT", type="LongType"),
                function="sum",
                window_sec=3600,
            )
        ],
        timestamp_col="TS",
        feature_aggregation_method="continuous",
        refresh_freq="1 minute",
    )
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    codes = {r.code for r in results}
    assert "BATCH_FV_TILING_GRANULARITY" in codes, (
        "BFV + continuous without explicit feature_granularity must still "
        f"emit BATCH_FV_TILING_GRANULARITY; got codes={sorted(codes)!r}."
    )


def _acd_batch_fv(*, output_type: str, secondary_keys: list[str] | None = None) -> FeatureView:
    """A tiled offline BatchFV with a single ``approx_count_distinct`` feature.

    All the other tiling invariants (timestamp / granularity / agg-method /
    refresh) are satisfied so the only variable under test is the authored
    ``output_column.type`` of the acd feature.

    Args:
        output_type: The authored ``output_column.type`` for the acd feature.
        secondary_keys: When provided, sets ``aggregation_secondary_keys`` so
            the FV is treated as a secondary-key FV (output-type check skipped).

    Returns:
        The constructed BatchFeatureView spec.
    """
    fv = _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="ITEM", type="StringType"),
                output_column=FSColumn(name="ITEM_NDV_1H", type=output_type),
                function="approx_count_distinct",
                window_sec=3600,
                function_params={"hll_lg_k": 8},
            )
        ],
        timestamp_col="EVENT_TS",
        feature_granularity_sec=3600,
        feature_aggregation_method="tiles",
        refresh_freq="1 hour",
    )
    if secondary_keys is not None:
        fv = fv.model_copy(update={"aggregation_secondary_keys": secondary_keys})
    return fv


def test_acd_string_output_type_rejected() -> None:
    """Bug A: a non-SK ``approx_count_distinct`` feature authored with a
    ``StringType`` output must be rejected at plan time.

    Snowflake materialises the acd tile column as ``NUMBER(38,0)`` (recovered
    as ``LongType``), so a ``StringType`` author never hash-matches the
    applied side and loops on a destructive ``RECREATE_FV``.  We reject the
    incorrect on-disk type with an actionable ``FEATURE_OUTPUT_TYPE_INVALID``
    error rather than silently coercing it.
    """
    fv = _acd_batch_fv(output_type="StringType")
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    offenders = [r for r in results if r.code == "FEATURE_OUTPUT_TYPE_INVALID"]
    assert offenders, (
        "approx_count_distinct with a StringType output_column must emit "
        f"FEATURE_OUTPUT_TYPE_INVALID; got codes={sorted({r.code for r in results})!r}."
    )
    assert all(r.severity == "ERROR" for r in offenders)


def test_acd_long_output_type_accepted() -> None:
    """The predetermined ``LongType`` output is the correct spelling and
    must not raise ``FEATURE_OUTPUT_TYPE_INVALID``."""
    fv = _acd_batch_fv(output_type="LongType")
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    assert not any(r.code == "FEATURE_OUTPUT_TYPE_INVALID" for r in results), (
        "a LongType-authored approx_count_distinct feature is correct and must "
        "not trigger the output-type validation."
    )


def test_acd_secondary_key_string_output_skipped() -> None:
    """Secondary-key acd FVs legitimately author ``StringType``: the server
    derives an ``ArrayType`` (element ``LongType``) which
    ``_strip_secondary_key_derivations`` normalises away.  The output-type
    check must skip FVs carrying ``aggregation_secondary_keys`` so those
    (correctly round-tripping) FVs are not rejected.
    """
    fv = _acd_batch_fv(output_type="StringType", secondary_keys=["OFFER_KEY"])
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    assert not any(r.code == "FEATURE_OUTPUT_TYPE_INVALID" for r in results), (
        "secondary-key acd FVs author StringType and rely on the SK-derivation "
        "normalisation; the output-type check must skip them."
    )


def _acd_fv_with_params(function_params: dict[str, Any]) -> FeatureView:
    """A tiled acd BatchFV whose single feature carries ``function_params``."""
    return _batch_fv(
        features=[
            Feature(
                source_column=FSColumn(name="ITEM", type="StringType"),
                output_column=FSColumn(name="ITEM_NDV_1H", type="LongType"),
                function="approx_count_distinct",
                window_sec=3600,
                function_params=function_params,
            )
        ],
        timestamp_col="EVENT_TS",
        feature_granularity_sec=3600,
        feature_aggregation_method="tiles",
        refresh_freq="1 hour",
    )


def test_conflicting_function_params_rejected() -> None:
    """A feature that sets both ``precision`` and ``hll_lg_k`` to differing
    values is ambiguous and must surface as ``FEATURE_PARAMS_CONFLICT`` rather
    than be swallowed by the planner's ``compile_to_spec`` fallbacks."""
    fv = _acd_fv_with_params({"precision": 8, "hll_lg_k": 12})
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    offenders = [r for r in results if r.code == "FEATURE_PARAMS_CONFLICT"]
    assert offenders and all(r.severity == "ERROR" for r in offenders), (
        "conflicting precision/hll_lg_k must emit an ERROR FEATURE_PARAMS_CONFLICT; "
        f"got codes={sorted({r.code for r in results})!r}."
    )


def test_redundant_equal_function_params_accepted() -> None:
    """An alias that merely restates the internal key (equal values) is
    redundant, not conflicting, and must not be rejected."""
    fv = _acd_fv_with_params({"precision": 8, "hll_lg_k": 8})
    results = validate_specs(_batch([_entity(), _batch_source(), fv]), AppliedState(objects={}))
    assert not any(
        r.code == "FEATURE_PARAMS_CONFLICT" for r in results
    ), "equal alias/internal values are redundant, not a conflict."


def test_decl_mirror_matches_core_predetermined_output_table() -> None:
    """The decl mirror must stay complete as core adds predetermined-output
    functions (``invariants.py`` cannot import core, so this is the guard)."""
    from snowflake.ml.feature_store.spec.builder import _AGG_PREDETERMINED_OUTPUT

    assert _AGG_REQUIRED_OUTPUT_TYPE == {agg.value: col.type for agg, col in _AGG_PREDETERMINED_OUTPUT.items()}


if __name__ == "__main__":
    pytest_driver.main()
