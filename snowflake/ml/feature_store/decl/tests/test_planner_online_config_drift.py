"""Regression tests for spurious ``UPDATE_FV`` from online-routing drift.

Closes ``plans/bug_operational_field_drift_streaming_bfvs.md`` (streaming half).

Live diagnosis against ``JKEW_DB.JKEW_SCHEMA`` showed that the only two
streaming FVs deployed with ``online_enabled=False``
(``USER_CLICK_STATS_DECL`` / ``USER_CLICK_STATS_CONTINUOUS_DECL``) emit a
persistent ``UPDATE_FV`` on every clean re-plan.  Their
``DESCRIBE … TYPE = SPECIFICATION`` payload omits ``online_store_type``
(applied side ``None``), but the declarative model forces every
``StreamingFeatureView`` online, so ``compile_to_spec`` always derives a
concrete store type (``postgres`` for the account's online store).
``planner._online_config_drifted`` therefore compared ``"postgres" != None``
and flagged drift — an ``UPDATE_FV`` the operator cannot express (the model
rejects ``online: false`` on streaming FVs) and that ``update_feature_view``
cannot apply (error 1300 / 2110 on streaming kinds).

The fix: for the always-online kinds (Streaming / Realtime), an *absent*
applied ``online_store_type`` is not an actionable operational drift — the
store type is derived, not authored, so a derived-local-vs-absent-applied
difference must round-trip as ``NO_CHANGE``.  A genuinely divergent applied
store type (e.g. ``snowflake`` vs ``postgres``) is still drift, and the
authorable ``BatchFeatureView`` online-enable toggle is unchanged.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.planner import _online_config_drifted

_DB = "DB"
_SCH = "SCH"


def _streaming_local(name: str = "USER_CLICK_STATS_DECL") -> dict[str, Any]:
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "version": "V1",
        "database": _DB,
        "schema": _SCH,
        "entities": ["USER_ID"],
        "sources": [],
        "features": [],
    }


def _batch_local(*, online: bool) -> dict[str, Any]:
    return {
        "kind": "BatchFeatureView",
        "name": "MY_BFV",
        "version": "V1",
        "database": _DB,
        "schema": _SCH,
        "online": online,
        "entities": ["USER_ID"],
        "sources": [],
        "features": [],
    }


def _applied(online_store_type: Any = "__absent__") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "metadata": {"database": _DB, "schema": _SCH, "name": "F", "version": "V1"},
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    if online_store_type != "__absent__":
        payload["online_store_type"] = online_store_type
    return payload


class TestStreamingOnlineConfigDrift:
    """Streaming FVs: an absent applied ``online_store_type`` is NOT drift."""

    def test_absent_applied_online_store_type_is_no_drift(self) -> None:
        # Live repro shape: local compiles to ``postgres`` (forced online),
        # applied payload carries no ``online_store_type`` at all.
        assert _online_config_drifted(_streaming_local(), _applied(), _DB, _SCH) is False

    def test_none_applied_online_store_type_is_no_drift(self) -> None:
        assert _online_config_drifted(_streaming_local(), _applied(None), _DB, _SCH) is False

    def test_empty_applied_online_store_type_is_no_drift(self) -> None:
        assert _online_config_drifted(_streaming_local(), _applied(""), _DB, _SCH) is False

    def test_matching_applied_online_store_type_is_no_drift(self) -> None:
        assert _online_config_drifted(_streaming_local(), _applied("postgres"), _DB, _SCH) is False

    def test_divergent_applied_online_store_type_is_drift(self) -> None:
        # A genuine store-type change (postgres -> snowflake) is still a
        # real operational drift and must remain detectable.
        assert _online_config_drifted(_streaming_local(), _applied("snowflake"), _DB, _SCH) is True


class TestBatchOnlineConfigDriftUnchanged:
    """BatchFV online-enable is authorable, so its semantics are preserved."""

    def test_batch_online_enable_still_drifts_against_absent_applied(self) -> None:
        # Operator authored ``online: true`` (local compiles to postgres);
        # deployed BFV has no online store -> a real enable -> drift.
        assert _online_config_drifted(_batch_local(online=True), _applied(), _DB, _SCH) is True

    def test_batch_offline_matches_absent_applied(self) -> None:
        # ``online: false`` compiles to None; absent applied is also None.
        assert _online_config_drifted(_batch_local(online=False), _applied(), _DB, _SCH) is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
