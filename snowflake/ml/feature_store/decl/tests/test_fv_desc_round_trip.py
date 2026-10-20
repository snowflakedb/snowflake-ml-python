"""Symmetric ``desc`` round-trip for feature views (Option A, Phase 1).

Closes the ``desc`` half of ``plans/bug_operational_field_drift_streaming_bfvs.md``.

The applied side recovers a deployed FV's description (from
``list_feature_views().desc``) onto the top level of ``spec_payload``, and the
planner's ``_desc_drifted`` compares it against the local authoring
``description``.  Two asymmetries produced a persistent ``UPDATE_FV`` for any
FV deployed with a non-empty description:

1. ``state`` injected the top-level ``desc`` only for ``BatchFeatureView`` —
   the ``StreamingFeatureView`` recovery path skipped it, so a described
   streaming FV compared ``local="" != applied="docs"`` on a BatchFV but the
   streaming path could not even surface the applied value.
2. ``exporter._build_full_fidelity_fv`` never emitted ``description`` into the
   exported YAML for any FV kind, so ``snow feature init`` dropped the
   deployed description; the re-plan then compared
   ``local="" != applied="docs"`` -> spurious ``UPDATE_FV`` every round.

The fix makes both halves symmetric: the exporter re-emits ``description``
from the recovered ``desc``, and state injects the top-level ``desc`` for
streaming/realtime FVs too.  Empty descriptions stay a no-op on both sides.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.planner import _desc_drifted
from snowflake.ml.feature_store.decl.state import fetch_applied_state


class TestExporterEmitsFvDescription:
    """``_build_full_fidelity_fv`` re-emits the recovered ``desc`` as ``description``."""

    def _payload(self, kind: str, desc: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": kind,
            "metadata": {"name": "FV_DESC", "version": "V1", "database": "DB", "schema": "SCH"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
        }
        if desc is not None:
            payload["desc"] = desc
        return payload

    def test_streaming_fv_description_emitted(self) -> None:
        doc = _build_full_fidelity_fv(
            self._payload("StreamingFeatureView", "user click stats v2"),
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        assert doc.get("description") == "user click stats v2"

    def test_batch_fv_description_emitted(self) -> None:
        doc = _build_full_fidelity_fv(
            self._payload("BatchFeatureView", "adv bfv docs"),
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        assert doc.get("description") == "adv bfv docs"

    def test_realtime_fv_description_emitted(self) -> None:
        doc = _build_full_fidelity_fv(
            self._payload("RealtimeFeatureView", "rt fv docs"),
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        assert doc.get("description") == "rt fv docs"

    def test_empty_desc_not_emitted(self) -> None:
        doc = _build_full_fidelity_fv(
            self._payload("StreamingFeatureView", ""),
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        assert "description" not in doc

    def test_absent_desc_not_emitted(self) -> None:
        doc = _build_full_fidelity_fv(
            self._payload("StreamingFeatureView", None),
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        assert "description" not in doc


class TestStreamingFvDescInjection:
    """State recovery surfaces a streaming FV's deployed desc at top level."""

    _SHOW_ROW: dict[str, Any] = {
        "name": "CLICK_FV$V1$ONLINE",
        "database_name": "DB",
        "schema_name": "SCH",
        "created_on": "2024-01-01 00:00:00",
    }
    _SPEC: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "metadata": {"name": "CLICK_FV", "version": "V1", "database": "DB", "schema": "SCH"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [],
        },
    }

    def test_streaming_desc_injected_at_top_level(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"CLICK_FV$V1$ONLINE": self._SPEC},
            feature_view_rows=[
                {
                    "name": "CLICK_FV",
                    "version": "V1",
                    "desc": "streaming docs",
                    "online_enabled": False,
                }
            ],
            default_database="DB",
            default_schema="SCH",
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "StreamingFeatureView"]
        assert len(fv_objs) == 1
        assert fv_objs[0].spec_payload.get("desc") == "streaming docs"

    def test_streaming_empty_desc_not_injected(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"CLICK_FV$V1$ONLINE": self._SPEC},
            feature_view_rows=[{"name": "CLICK_FV", "version": "V1", "desc": "", "online_enabled": False}],
            default_database="DB",
            default_schema="SCH",
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "StreamingFeatureView"]
        assert len(fv_objs) == 1
        assert "desc" not in fv_objs[0].spec_payload


class TestRealtimeFvDescInjection:
    """State recovery surfaces a realtime FV's deployed desc at top level.

    Pins the 6d2 hoist: the ``desc`` injection runs for every kind before the
    kind dispatch, so a RealtimeFeatureView (whose SPECIFICATION payload never
    carries ``desc``) recovers it from the list-FV row just like Batch and
    Streaming do.  Without the hoist a described RTFV re-planned as a spurious
    ``UPDATE_FV`` every round.
    """

    _SHOW_ROW: dict[str, Any] = {
        "name": "RT_FV$V1$ONLINE",
        "database_name": "DB",
        "schema_name": "SCH",
        "created_on": "2024-01-01 00:00:00",
    }
    _SPEC: dict[str, Any] = {
        "kind": "RealtimeFeatureView",
        "metadata": {"name": "RT_FV", "version": "V1", "database": "DB", "schema": "SCH"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [],
        },
    }

    def test_realtime_desc_injected_at_top_level(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"RT_FV$V1$ONLINE": self._SPEC},
            feature_view_rows=[
                {
                    "name": "RT_FV",
                    "version": "V1",
                    "desc": "realtime docs",
                    "online_enabled": True,
                }
            ],
            default_database="DB",
            default_schema="SCH",
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "RealtimeFeatureView"]
        assert len(fv_objs) == 1
        assert fv_objs[0].spec_payload.get("desc") == "realtime docs"

    def test_realtime_empty_desc_not_injected(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"RT_FV$V1$ONLINE": self._SPEC},
            feature_view_rows=[{"name": "RT_FV", "version": "V1", "desc": "", "online_enabled": True}],
            default_database="DB",
            default_schema="SCH",
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "RealtimeFeatureView"]
        assert len(fv_objs) == 1
        assert "desc" not in fv_objs[0].spec_payload


class TestDescRoundTripNoDrift:
    """End-to-end: exported description compiles to a matching applied desc."""

    def test_no_desc_drift_after_export(self) -> None:
        applied_payload: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "metadata": {"name": "FV_DESC", "version": "V1", "database": "DB", "schema": "SCH"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
            "desc": "round trip docs",
        }
        exported = _build_full_fidelity_fv(
            applied_payload,
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        # The exported authoring dict must resolve to the same description
        # the applied side carries -> ``_desc_drifted`` is False -> NO_CHANGE.
        assert _desc_drifted(exported, applied_payload) is False

    def test_no_desc_drift_after_export_realtime(self) -> None:
        applied_payload: dict[str, Any] = {
            "kind": "RealtimeFeatureView",
            "metadata": {"name": "FV_DESC", "version": "V1", "database": "DB", "schema": "SCH"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
            "desc": "round trip docs",
        }
        exported = _build_full_fidelity_fv(
            applied_payload,
            fallback_name="FV_DESC",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        assert _desc_drifted(exported, applied_payload) is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
