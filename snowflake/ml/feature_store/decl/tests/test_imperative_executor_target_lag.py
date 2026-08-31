"""Tests for ``_resolve_online_target_lag`` resolution precedence.

Pins the two-rung fallback chain that decides what string the imperative
executor passes to ``OnlineConfig(target_lag=...)`` when applying an
``online: true`` feature view:

1. ``payload["target_lag"]`` (raw authoring shape)
2. ``payload["target_lag_sec"]`` (post-``normalize_durations`` integer)
3. ``None`` (caller defaults — ``_BATCH_OFT_TARGET_LAG`` ("10 seconds")
   for online ``BatchFeatureView``, ``"0 seconds"`` for streaming /
   realtime / FG which Snowflake requires for those kinds).

After the ``feature_granularity`` / ``refresh_freq`` / ``target_lag``
decoupling, ``refresh_freq`` is strictly the offline Dynamic Table
refresh cadence and is **never** read by this helper.  The OFT
TARGET_LAG default for a fresh online BFV that omits ``target_lag``
comes from the caller (``_BATCH_OFT_TARGET_LAG``), not from the DT
cadence.

The behavioural section pins the BFV path end-to-end: a
``_build_feature_view`` invocation with ``online: true`` and
``refresh_freq: "5 minutes"`` produces ``OnlineConfig(target_lag=
"10 seconds", ...)`` (the imperative ``_BATCH_OFT_TARGET_LAG`` default)
when no ``target_lag`` is authored, and the explicit value when it is.
The streaming-FV regression at the end confirms the existing
``"0 seconds"`` default still fires for streaming payloads.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from snowflake.ml.feature_store.decl.imperative_executor import (
    _build_feature_view,
    _execute_update_feature_view,
    _resolve_online_target_lag,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# _resolve_online_target_lag — unit-level precedence tests.
# ---------------------------------------------------------------------------


class TestResolveOnlineTargetLag:
    """Pin the four-rung resolution precedence and the ``None`` fallback."""

    def test_target_lag_string_wins_over_target_lag_sec_and_refresh_freq(self) -> None:
        payload = {
            "target_lag": "1 hour",
            "target_lag_sec": 60,
            "refresh_freq": "5 minutes",
        }
        assert _resolve_online_target_lag(payload) == "1 hour"

    def test_target_lag_int_returns_seconds_format(self) -> None:
        # ``target_lag: 90`` (int) → ``"90 seconds"`` regardless of the
        # presence of ``target_lag_sec`` / ``refresh_freq``.
        assert _resolve_online_target_lag({"target_lag": 90}) == "90 seconds"
        assert (
            _resolve_online_target_lag({"target_lag": 90, "target_lag_sec": 60, "refresh_freq": "5 minutes"})
            == "90 seconds"
        )

    def test_target_lag_sec_int_wins_over_refresh_freq(self) -> None:
        payload = {"target_lag_sec": 120, "refresh_freq": "5 minutes"}
        assert _resolve_online_target_lag(payload) == "120 seconds"

    def test_target_lag_sec_string_digits_wins_over_refresh_freq(self) -> None:
        payload = {"target_lag_sec": "120", "refresh_freq": "5 minutes"}
        assert _resolve_online_target_lag(payload) == "120 seconds"

    def test_target_lag_sec_zero_falls_through(self) -> None:
        # Zero is rejected by the existing ``> 0`` guard so the helper
        # falls through.  After the decoupling, ``refresh_freq`` is
        # never consulted as an OFT-side fallback.
        assert _resolve_online_target_lag({"target_lag_sec": 0, "refresh_freq": "5 minutes"}) is None
        assert _resolve_online_target_lag({"target_lag_sec": 0}) is None

    def test_target_lag_empty_string_falls_through(self) -> None:
        # Empty / whitespace-only ``target_lag`` does not short-circuit.
        # ``refresh_freq`` (DT-only after decoupling) does not satisfy
        # the OFT side either.
        assert _resolve_online_target_lag({"target_lag": "", "refresh_freq": "5 minutes"}) is None
        assert _resolve_online_target_lag({"target_lag": "   ", "refresh_freq": "5 minutes"}) is None

    def test_refresh_freq_is_ignored(self) -> None:
        # After the decoupling, ``refresh_freq`` is strictly the
        # offline DT refresh cadence and is never read by
        # ``_resolve_online_target_lag``.  A BFV authoring only
        # ``refresh_freq`` (and no ``target_lag``) returns ``None``
        # so the caller can apply the imperative
        # ``_BATCH_OFT_TARGET_LAG`` ("10 seconds") default.
        assert _resolve_online_target_lag({"refresh_freq": "5 minutes"}) is None
        assert _resolve_online_target_lag({"refresh_freq": "1 hour"}) is None

    def test_refresh_freq_ignored_when_empty_or_non_string(self) -> None:
        # Defence-in-depth: every refresh_freq shape is ignored
        # post-decoupling (the field never enters the OFT path).
        assert _resolve_online_target_lag({"refresh_freq": ""}) is None
        assert _resolve_online_target_lag({"refresh_freq": "   "}) is None
        assert _resolve_online_target_lag({"refresh_freq": 300}) is None
        assert _resolve_online_target_lag({"refresh_freq": None}) is None

    def test_all_keys_absent_returns_none(self) -> None:
        # Pure streaming / realtime / FG payloads land here so the
        # caller's default fires (``"0 seconds"`` for those kinds).
        assert _resolve_online_target_lag({}) is None
        assert _resolve_online_target_lag({"online": True}) is None


# ---------------------------------------------------------------------------
# _build_feature_view — behavioural tests.
# ---------------------------------------------------------------------------


def _capture_online_config_kwargs() -> tuple[type, dict[str, Any]]:
    """Return a fake ``OnlineConfig`` class and a kwargs-capture dict.

    The fake records every call's kwargs so the test can assert on
    ``target_lag`` and ``store_type`` without the real
    ``OnlineConfig`` constructor's coercion getting in the way.

    Returns:
        Tuple of (FakeOnlineConfig, captured) where ``captured`` is a
        dict the fake mutates in place with the kwargs of the last
        call.  When the executor never calls ``OnlineConfig`` the dict
        stays empty.
    """
    captured: dict[str, Any] = {}

    class _FakeOnlineConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    return _FakeOnlineConfig, captured


def _minimal_batch_fv_payload(**overrides: Any) -> dict[str, Any]:
    """Minimal ``_build_feature_view`` payload mirroring the FG bug-bash YAMLs."""
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "USER_AMOUNTS_FG_DECL",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_FG_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_FG_DECL",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "5 minutes",
    }
    base.update(overrides)
    return base


def test_online_bfv_without_target_lag_uses_imperative_default() -> None:
    """Decoupled-contract: ``online: true`` BFV with only ``refresh_freq``.

    After the ``feature_granularity`` / ``refresh_freq`` /
    ``target_lag`` decoupling, ``refresh_freq`` is strictly the
    offline DT refresh cadence and is never read on the OFT path.
    Online BFVs that omit ``target_lag`` get the imperative-API
    default (``_BATCH_OFT_TARGET_LAG = "10 seconds"``) instead of the
    pre-fix ``refresh_freq`` mirror.  Snowflake accepts ``"10
    seconds"`` for batch OFTs and would have rejected ``"0 seconds"``
    here (``Invalid TARGET_LAG value '0 seconds' specified for online
    feature table. … '0 seconds' is only valid for
    StreamingFeatureView, RealtimeFeatureView, and FeatureGroup``), so
    the executor's fallback to ``_BATCH_OFT_TARGET_LAG`` keeps the
    apply path safe.
    """
    payload = _minimal_batch_fv_payload()
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()

    fake_online_config, captured = _capture_online_config_kwargs()

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", MagicMock(),), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        fake_online_config,
    ), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_CONNECTION_DEFAULT", fs=fs)

    assert captured.get("target_lag") == "10 seconds", (
        "online BFV without authored target_lag must default to "
        "_BATCH_OFT_TARGET_LAG (the pre-fix refresh_freq mirror is "
        "removed); see _resolve_online_target_lag docstring"
    )
    assert captured.get("enable") is True


def test_online_bfv_explicit_target_lag_wins_over_refresh_freq() -> None:
    """Explicit ``target_lag:`` must still beat the ``refresh_freq`` fallback."""
    payload = _minimal_batch_fv_payload(target_lag="1 hour")
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()

    fake_online_config, captured = _capture_online_config_kwargs()

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", MagicMock(),), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        fake_online_config,
    ), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_CONNECTION_DEFAULT", fs=fs)

    assert captured.get("target_lag") == "1 hour"


def test_online_bfv_explicit_target_lag_sec_wins_over_refresh_freq() -> None:
    """Normalised ``target_lag_sec:`` beats the ``refresh_freq`` fallback."""
    payload = _minimal_batch_fv_payload(target_lag_sec=120)
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()

    fake_online_config, captured = _capture_online_config_kwargs()

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", MagicMock(),), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        fake_online_config,
    ), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_CONNECTION_DEFAULT", fs=fs)

    assert captured.get("target_lag") == "120 seconds"


def test_online_streaming_fv_without_target_lag_still_uses_zero_seconds_default() -> None:
    """Regression: streaming FVs (no ``refresh_freq``) keep the ``"0 seconds"`` default.

    Snowflake accepts ``TARGET_LAG = '0 seconds'`` for
    StreamingFeatureView / RealtimeFeatureView / FeatureGroup, so the
    helper must return ``None`` for streaming payloads (they never
    carry ``refresh_freq``) and the caller's ``or "0 seconds"``
    default still fires.  Asserts the helper output directly to keep
    the regression simple — the integrated streaming-FV apply path is
    covered by the existing decl test suite.
    """
    streaming_payload = {
        "kind": "StreamingFeatureView",
        "online": True,
        "entities": ["USER_ID"],
        # Intentionally no target_lag / target_lag_sec / refresh_freq.
    }
    assert _resolve_online_target_lag(streaming_payload) is None


# ---------------------------------------------------------------------------
# _execute_update_feature_view — kind-aware target_lag fallback.
# ---------------------------------------------------------------------------
#
# Pins the UPDATE_FV path's parity with the CREATE_FV kind-aware default
# (BFV → ``_BATCH_OFT_TARGET_LAG`` "10 seconds", everything else →
# ``"0 seconds"``).  Pre-fix the UPDATE path unconditionally fell back
# to ``_BATCH_OFT_TARGET_LAG`` even for ``StreamingFeatureView`` /
# ``RealtimeFeatureView`` payloads, which Snowflake then rejected with
# ``Invalid TARGET_LAG value '10 seconds' specified for online feature
# table … StreamingFeatureView, RealtimeFeatureView, and FeatureGroup
# only support TARGET_LAG = '0 seconds'``.  The spec validator
# ``_reject_target_lag_on_stream_or_realtime`` forbids authoring
# ``target_lag`` on these kinds entirely, so the default must come
# from the executor.


def _make_update_fv_op(payload: dict[str, Any]) -> Any:
    """Construct a minimal stand-in for a ``PlanOp`` with ``UPDATE_FV`` payload.

    ``_execute_update_feature_view`` only reaches for ``op.payload`` and
    ``op.name``; the broader ``PlanOp`` surface is irrelevant here.

    Args:
        payload: The UPDATE_FV payload dict the stand-in op should expose
            via ``op.payload``; its ``"name"`` key (if present) is mirrored
            onto ``op.name`` for the executor's logging path.

    Returns:
        A ``MagicMock`` with ``payload`` and ``name`` attributes populated
        — sufficient for ``_execute_update_feature_view`` which only
        accesses those two attributes.
    """
    op = MagicMock()
    op.payload = payload
    op.name = payload.get("name", "")
    return op


def test_update_streaming_fv_without_target_lag_defaults_to_zero_seconds() -> None:
    """UPDATE_FV on a StreamingFeatureView must default ``target_lag`` to ``"0 seconds"``.

    Live regression: applying ``USER_CLICK_BACKFILL_DECL`` (a
    StreamingFeatureView) failed with ``Invalid TARGET_LAG value
    '10 seconds'`` because the UPDATE path's fallback used
    ``_BATCH_OFT_TARGET_LAG``.  The fix mirrors the kind-aware default
    already used by the CREATE path.
    """
    payload = {
        "kind": "StreamingFeatureView",
        "name": "USER_CLICK_BACKFILL_DECL",
        "version": "V1",
        "online": True,
        "refresh_freq": "5 minutes",
        # Intentionally no target_lag / target_lag_sec — the spec
        # validator forbids authoring this field on streaming kinds.
    }
    fs = MagicMock()
    fake_online_config, captured = _capture_online_config_kwargs()

    with patch("snowflake.ml.feature_store.feature_view.OnlineConfig", fake_online_config,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _execute_update_feature_view(fs, _make_update_fv_op(payload), "WH_CONNECTION_DEFAULT")

    assert captured.get("target_lag") == "0 seconds", (
        "UPDATE_FV on a StreamingFeatureView must default to '0 seconds' "
        "(Snowflake rejects any non-zero TARGET_LAG on streaming OFTs)"
    )
    assert captured.get("enable") is True
    fs.update_feature_view.assert_called_once()


def test_update_realtime_fv_without_target_lag_defaults_to_zero_seconds() -> None:
    """UPDATE_FV on a RealtimeFeatureView must default ``target_lag`` to ``"0 seconds"``.

    Same constraint as streaming — Snowflake only accepts
    ``TARGET_LAG = '0 seconds'`` on RTFV OFTs.
    """
    payload = {
        "kind": "RealtimeFeatureView",
        "name": "USER_RTFV_DECL",
        "version": "V1",
        "online": True,
    }
    fs = MagicMock()
    fake_online_config, captured = _capture_online_config_kwargs()

    with patch("snowflake.ml.feature_store.feature_view.OnlineConfig", fake_online_config,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _execute_update_feature_view(fs, _make_update_fv_op(payload), "WH_CONNECTION_DEFAULT")

    assert captured.get("target_lag") == "0 seconds"


def test_update_batch_fv_without_target_lag_keeps_imperative_default() -> None:
    """UPDATE_FV on an online ``BatchFeatureView`` must still use ``_BATCH_OFT_TARGET_LAG``.

    Symmetric to the streaming/realtime regressions above — the
    kind-aware fallback must NOT regress the BFV default to
    ``"0 seconds"`` (Snowflake rejects that for batch OFTs).
    """
    payload = {
        "kind": "BatchFeatureView",
        "name": "USER_AMOUNTS_FG_DECL",
        "version": "V1",
        "online": True,
        "refresh_freq": "5 minutes",
    }
    fs = MagicMock()
    fake_online_config, captured = _capture_online_config_kwargs()

    with patch("snowflake.ml.feature_store.feature_view.OnlineConfig", fake_online_config,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _execute_update_feature_view(fs, _make_update_fv_op(payload), "WH_CONNECTION_DEFAULT")

    assert captured.get("target_lag") == "10 seconds", (
        "UPDATE_FV on an online BatchFeatureView must keep the imperative "
        "_BATCH_OFT_TARGET_LAG ('10 seconds') default; '0 seconds' is "
        "rejected by Snowflake for batch OFTs"
    )


if __name__ == "__main__":
    pytest_driver.main()
