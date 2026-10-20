"""Idempotency coverage for online-only BatchFV refresh-``warehouse`` drift.

Root cause B: an online-only ``BatchFeatureView`` (``online: true`` /
``offline: false``) that authors an explicit refresh ``warehouse`` re-plans as a
spurious non-destructive ``UPDATE_FV`` on every clean apply-then-plan.  The
deployed ``DESCRIBE ONLINE FEATURE TABLE … TYPE = SPECIFICATION`` payload never
carries the DT-only refresh warehouse, and — unlike the offline-only recovery
path (``imperative_executor._serialize_batch_fv_spec``, which injects the
list-row warehouse) — the online/OFT recovery path
(``state._inject_batch_fv_fields_from_list_row``) never re-injects it.  So the
applied ``spec.warehouse`` is absent and ``planner._warehouse_drifted`` reports
a permanent authored-vs-absent drift.

Fixtures are live QA6 captures of a faithful online-only BatchFV mirror
(``golden_specs/ONLINE_WAREHOUSE_BFV.json`` — the recovered applied spec_payload,
warehouse absent; ``fixtures/ONLINE_WAREHOUSE_BFV_ROW.json`` — the
``list_feature_views()`` row, which DOES carry ``warehouse: JKEW_WH``).  The fix
injects the list-row warehouse into ``spec.warehouse`` during recovery so the
planner's drift comparison finds a match on a clean round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snowflake.ml.feature_store.decl.planner import _warehouse_drifted
from snowflake.ml.feature_store.decl.state import _inject_batch_fv_fields_from_list_row

_APPLIED_GOLDEN = Path(__file__).parent / "golden_specs" / "ONLINE_WAREHOUSE_BFV.json"
_ROW_GOLDEN = Path(__file__).parent / "fixtures" / "ONLINE_WAREHOUSE_BFV_ROW.json"


def _load_applied() -> dict[str, Any]:
    """Recovered applied spec_payload for the online mirror BFV (no warehouse)."""
    data: dict[str, Any] = json.loads(_APPLIED_GOLDEN.read_text())
    return data


def _load_row() -> dict[str, Any]:
    """The ``list_feature_views()`` row for the mirror BFV (warehouse=JKEW_WH)."""
    data: dict[str, Any] = json.loads(_ROW_GOLDEN.read_text())
    return data


def _authoring() -> dict[str, Any]:
    """Minimal authoring dict matching the deployed mirror's warehouse."""
    return {
        "kind": "BatchFeatureView",
        "name": "MIRROR_WAREHOUSE_BFV",
        "version": "V1",
        "online": True,
        "offline": False,
        "warehouse": "JKEW_WH",
    }


def test_captured_applied_has_no_warehouse() -> None:
    """Ground-truth pin: the OFT-recovered applied spec omits the warehouse.

    This is the whole reason for the drift — the deployed SPECIFICATION carries
    no DT refresh warehouse, so recovery must re-inject it from the list row.
    """
    applied = _load_applied()
    inner = applied.get("spec") if isinstance(applied.get("spec"), dict) else {}
    assert isinstance(inner, dict)
    assert not (inner.get("warehouse") or applied.get("warehouse")), (
        "fixture drift: the online mirror applied spec unexpectedly carries a "
        "warehouse — recapture golden_specs/ONLINE_WAREHOUSE_BFV.json"
    )
    row = _load_row()
    assert str(row.get("warehouse") or "").strip(), (
        "fixture drift: the list_feature_views row must carry the deployed "
        "warehouse — recapture golden_specs/ONLINE_WAREHOUSE_BFV_ROW.json"
    )


def test_inject_warehouse_from_list_row() -> None:
    """``_inject_batch_fv_fields_from_list_row`` must inject ``spec.warehouse``
    from the list-FV row (mirrors the cluster_by / refresh_mode injections and
    the offline-path ``_serialize_batch_fv_spec`` warehouse injection).
    """
    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["UID_RAW"], "sources": [], "features": []},
    }
    row = {"warehouse": "WH_AIML", "cluster_by": "", "refresh_mode": ""}
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
    assert spec_payload["spec"].get("warehouse") == "WH_AIML", (
        "online recovery must inject the deployed refresh warehouse from the "
        f"list row; got spec keys={sorted(spec_payload['spec'])}"
    )


def test_online_warehouse_roundtrip_is_no_change() -> None:
    """After recovery injects the list-row warehouse, a clean online-only BFV
    round-trip must report NO warehouse drift (was a spurious ``UPDATE_FV``).
    """
    applied = _load_applied()
    row = _load_row()
    _inject_batch_fv_fields_from_list_row(applied, row, fv_obj=None)
    authored = _authoring()
    assert _warehouse_drifted(authored, applied) is False, (
        "online-only BatchFV with a recovered matching warehouse must not drift; "
        "the online recovery path failed to inject spec.warehouse (root cause B)."
    )


def test_genuine_warehouse_change_still_drifts() -> None:
    """Guard: a genuinely re-authored warehouse must still surface as drift
    (the fix must not over-normalize into swallowing real edits).
    """
    applied = _load_applied()
    row = _load_row()  # deployed warehouse == JKEW_WH
    _inject_batch_fv_fields_from_list_row(applied, row, fv_obj=None)
    authored = _authoring()
    authored["warehouse"] = "WH_SOMETHING_ELSE"
    assert (
        _warehouse_drifted(authored, applied) is True
    ), "a real authored-vs-deployed warehouse change must still drift -> UPDATE_FV"


def test_warehouse_drift_is_case_insensitive() -> None:
    """Guard: Snowflake identifiers are case-insensitive; a pure case delta
    between the authored and recovered warehouse must not drift.
    """
    applied = _load_applied()
    row = _load_row()
    _inject_batch_fv_fields_from_list_row(applied, row, fv_obj=None)
    authored = _authoring()
    authored["warehouse"] = "jkew_wh"  # deployed is JKEW_WH
    assert (
        _warehouse_drifted(authored, applied) is False
    ), "case-only warehouse differences must not be treated as drift"


def test_warehouse_injection_is_additive_noop_when_present() -> None:
    """Guard: the offline path already populates ``spec.warehouse`` — the online
    injection must not overwrite an existing value.
    """
    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {
            "ordered_entity_column_names": ["UID_RAW"],
            "sources": [],
            "features": [],
            "warehouse": "WH_ALREADY_SET",
        },
    }
    row = {"warehouse": "WH_FROM_ROW", "cluster_by": "", "refresh_mode": ""}
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
    assert spec_payload["spec"]["warehouse"] == "WH_ALREADY_SET", (
        "injection must be additive and never clobber a warehouse the offline " "path already recovered"
    )


def test_empty_list_row_warehouse_is_noop() -> None:
    """Guard: an empty/absent list-row warehouse must not inject a key (so a
    local-omitted warehouse stays a genuine no-op, not a spurious empty value).
    """
    for cell in ("", "   ", None):
        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "spec": {"ordered_entity_column_names": ["UID_RAW"], "sources": [], "features": []},
        }
        row = {"warehouse": cell, "cluster_by": "", "refresh_mode": ""}
        _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
        assert (
            "warehouse" not in spec_payload["spec"]
        ), f"empty list-row warehouse ({cell!r}) must not inject a spec.warehouse key"


def test_local_omits_warehouse_never_drifts() -> None:
    """Guard: when the operator omits ``warehouse``, the imperative
    ``update_feature_view`` cannot clear a deployed warehouse, so a missing
    local value is a no-op by contract — never drift, regardless of applied.
    """
    applied = _load_applied()
    row = _load_row()
    _inject_batch_fv_fields_from_list_row(applied, row, fv_obj=None)
    authored = _authoring()
    authored.pop("warehouse", None)
    assert _warehouse_drifted(authored, applied) is False, "a local-omitted warehouse must never be reported as drift"


def test_injection_is_safe_noop_for_specless_payload() -> None:
    """Guard: streaming/realtime recovery shapes (no BatchFV inner ``spec``)
    are untouched — the helper returns early and never fabricates a warehouse.
    """
    spec_payload: dict[str, Any] = {"kind": "StreamingFeatureView"}
    row = {"warehouse": "WH_FROM_ROW"}
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
    assert (
        "spec" not in spec_payload and "warehouse" not in spec_payload
    ), "a spec-less (non-BatchFV) payload must be left untouched"


# ---------------------------------------------------------------------------
# Streaming-FV refresh-warehouse recovery (second half of root cause B)
# ---------------------------------------------------------------------------
#
# A tiled ``StreamingFeatureView`` schedules an offline tile Dynamic Table and
# ``warehouse`` is in its operational field set
# (``planner._FV_OPERATIONAL_FIELDS_BY_KIND["StreamingFeatureView"]``), and the
# model accepts an authored ``warehouse:``.  But ``state.fetch_applied_state``
# only re-injects the deployed refresh warehouse for ``BatchFeatureView`` — the
# ``StreamingFeatureView`` branch recovers ``refresh_freq`` only — so a streaming
# FV that authors ``warehouse:`` re-plans as a spurious ``UPDATE_FV`` on every
# clean apply-then-plan (the streaming half of the online-warehouse gap).  The
# fix shares one ``_inject_fv_warehouse_from_list_row`` helper across both kinds
# and wires the streaming recovery branch to call it.


def _streaming_applied(*, warehouse: str | None = None) -> dict[str, Any]:
    """A recovered streaming spec_payload (tiled), warehouse absent by default.

    Args:
        warehouse: When given, pre-populates ``spec.warehouse`` (models the
            already-injected shape); ``None`` models the raw DESCRIBE recovery
            that omits the DT-only refresh warehouse.

    Returns:
        The applied ``spec_payload`` dict.
    """
    inner: dict[str, Any] = {
        "ordered_entity_column_names": ["USER_ID"],
        "sources": [],
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "FloatType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window_sec": 3600,
            }
        ],
        "refresh_freq": "5 minutes",
    }
    if warehouse is not None:
        inner["warehouse"] = warehouse
    return {
        "kind": "StreamingFeatureView",
        "metadata": {"database": "DB", "schema": "SCH", "name": "GUEST_X_FV", "version": "V1"},
        "online_store_type": "postgres",
        "spec": inner,
    }


def _streaming_authoring(*, warehouse: str = "JKEW_WH") -> dict[str, Any]:
    """Minimal streaming authoring dict matching the deployed mirror's warehouse."""
    return {
        "kind": "StreamingFeatureView",
        "name": "GUEST_X_FV",
        "version": "V1",
        "database": "DB",
        "schema": "SCH",
        "entities": ["USER_ID"],
        "sources": [],
        "features": [],
        "warehouse": warehouse,
    }


def test_streaming_warehouse_injected_from_list_row() -> None:
    """The shared helper must inject ``spec.warehouse`` for a streaming payload.

    Mirrors the BatchFV injection so a tiled streaming FV's deployed refresh
    warehouse is recovered onto the applied spec.
    """
    from snowflake.ml.feature_store.decl.state import _inject_fv_warehouse_from_list_row

    spec_payload = _streaming_applied()
    row = {"warehouse": "WH_AIML"}
    _inject_fv_warehouse_from_list_row(spec_payload, row)
    assert spec_payload["spec"].get("warehouse") == "WH_AIML", (
        "streaming recovery must inject the deployed refresh warehouse from the "
        f"list row; got spec keys={sorted(spec_payload['spec'])}"
    )


def test_streaming_warehouse_roundtrip_is_no_change() -> None:
    """After recovery injects the list-row warehouse, a clean streaming-FV
    round-trip must report NO warehouse drift (was a spurious ``UPDATE_FV``).
    """
    from snowflake.ml.feature_store.decl.state import _inject_fv_warehouse_from_list_row

    applied = _streaming_applied()
    _inject_fv_warehouse_from_list_row(applied, {"warehouse": "JKEW_WH"})
    assert _warehouse_drifted(_streaming_authoring(), applied) is False, (
        "streaming FV with a recovered matching warehouse must not drift; the "
        "streaming recovery path failed to inject spec.warehouse (root cause B)."
    )


def test_streaming_genuine_warehouse_change_still_drifts() -> None:
    """A genuinely re-authored streaming warehouse must still surface as drift."""
    from snowflake.ml.feature_store.decl.state import _inject_fv_warehouse_from_list_row

    applied = _streaming_applied()
    _inject_fv_warehouse_from_list_row(applied, {"warehouse": "JKEW_WH"})
    authored = _streaming_authoring(warehouse="WH_SOMETHING_ELSE")
    assert (
        _warehouse_drifted(authored, applied) is True
    ), "a real authored-vs-deployed streaming warehouse change must still drift -> UPDATE_FV"


def test_streaming_warehouse_case_insensitive() -> None:
    """A pure case delta on a streaming warehouse must not drift."""
    from snowflake.ml.feature_store.decl.state import _inject_fv_warehouse_from_list_row

    applied = _streaming_applied()
    _inject_fv_warehouse_from_list_row(applied, {"warehouse": "JKEW_WH"})
    authored = _streaming_authoring(warehouse="jkew_wh")
    assert _warehouse_drifted(authored, applied) is False


def test_streaming_warehouse_injection_additive_when_present() -> None:
    """The injection must never clobber an already-recovered warehouse."""
    from snowflake.ml.feature_store.decl.state import _inject_fv_warehouse_from_list_row

    applied = _streaming_applied(warehouse="WH_ALREADY_SET")
    _inject_fv_warehouse_from_list_row(applied, {"warehouse": "WH_FROM_ROW"})
    assert applied["spec"]["warehouse"] == "WH_ALREADY_SET"


def test_streaming_empty_list_row_warehouse_is_noop() -> None:
    """An empty/absent list-row warehouse must not inject a key on a streaming FV."""
    from snowflake.ml.feature_store.decl.state import _inject_fv_warehouse_from_list_row

    for cell in ("", "   ", None):
        applied = _streaming_applied()
        _inject_fv_warehouse_from_list_row(applied, {"warehouse": cell})
        assert (
            "warehouse" not in applied["spec"]
        ), f"empty list-row warehouse ({cell!r}) must not inject a spec.warehouse key"


def test_realtime_warehouse_never_operational() -> None:
    """Guard: warehouse is not an operational field for RealtimeFeatureView.

    Realtime FVs are OFT-only (no offline Dynamic Table), so warehouse recovery
    is deliberately excluded — ``update_feature_view`` rejects it.  Even an
    authored-vs-absent warehouse must never surface as drift for this kind.
    """
    from snowflake.ml.feature_store.decl.planner import _FV_OPERATIONAL_FIELDS_BY_KIND

    assert "warehouse" not in _FV_OPERATIONAL_FIELDS_BY_KIND["RealtimeFeatureView"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
