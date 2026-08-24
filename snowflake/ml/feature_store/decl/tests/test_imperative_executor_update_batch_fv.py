"""Tests for UPDATE_FV execution on BatchFeatureView."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.imperative_executor import _execute_op
from snowflake.ml.feature_store.decl.types import PlanOp, PlanOptions


def test_update_fv_calls_update_feature_view_for_batch() -> None:
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="BFV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload={
            "kind": "BatchFeatureView",
            "name": "BFV",
            "version": "V1",
            "refresh_freq": "10 minutes",
            "warehouse": "WH1",
            "description": "new desc",
            "online": True,
            "target_lag": 30,
        },
    )
    session = MagicMock()
    opts = PlanOptions()
    _execute_op(fs, session, op, "DB", "SCH", "WH0", opts)
    fs.update_feature_view.assert_called_once()
    args, kwargs = fs.update_feature_view.call_args
    assert args[0] == "BFV"
    assert args[1] == "V1"
    assert kwargs["refresh_freq"] == "10 minutes"
    assert kwargs["warehouse"] == "WH1"
    assert kwargs["desc"] == "new desc"
    assert kwargs["online_config"].enable is True


def test_update_fv_rejects_unsupported_kind() -> None:
    """UPDATE_FV is gated to the three FV kinds — any other ``kind``
    raises a defensive ``ValueError`` even if the planner emits one.

    Post-B7, ``StreamingFeatureView`` and ``RealtimeFeatureView`` are
    valid UPDATE_FV targets (see ``_UPDATE_FV_SUPPORTED_KINDS``); the
    rejection contract still holds for non-FV payloads (a malformed
    op pointed at an entity / source / group is the only path that
    can land here).
    """
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="NOT_FV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload={"kind": "Entity", "name": "NOT_FV", "version": "V1"},
    )
    session = MagicMock()
    opts = PlanOptions()
    with pytest.raises(ValueError, match="UPDATE_FV is only supported"):
        _execute_op(fs, session, op, "DB", "SCH", "WH", opts)


# ---------------------------------------------------------------------------
# W-C: OnlineConfig.target_lag must come from ``target_lag_sec`` when the
# compiler has normalised the YAML ``target_lag:`` string to integer seconds.
# Previously the executor only read ``target_lag`` from the payload, so the
# normalised key was silently dropped and ``OnlineConfig.target_lag`` fell
# back to ``"0 seconds"`` — the BUG_BASH §6 latent bug.
# ---------------------------------------------------------------------------


def test_update_fv_propagates_target_lag_sec_to_online_config() -> None:
    """payload[target_lag_sec]=60 (no target_lag) → OnlineConfig.target_lag='60 seconds'."""
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="BFV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload={
            "kind": "BatchFeatureView",
            "name": "BFV",
            "version": "V1",
            "refresh_freq": "1 minute",
            "online": True,
            "target_lag_sec": 60,
        },
    )
    session = MagicMock()
    opts = PlanOptions()
    _execute_op(fs, session, op, "DB", "SCH", "WH0", opts)

    fs.update_feature_view.assert_called_once()
    _, kwargs = fs.update_feature_view.call_args
    online_config = kwargs.get("online_config")
    assert online_config is not None
    assert online_config.enable is True
    assert (
        online_config.target_lag == "60 seconds"
    ), f"expected target_lag='60 seconds' from target_lag_sec=60; got {online_config.target_lag!r}"


def test_update_fv_propagates_target_lag_sec_long_value_to_online_config() -> None:
    """Larger target_lag_sec (3600) → OnlineConfig.target_lag='3600 seconds'."""
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="BFV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload={
            "kind": "BatchFeatureView",
            "name": "BFV",
            "version": "V1",
            "refresh_freq": "1 hour",
            "online": True,
            "target_lag_sec": 3600,
        },
    )
    session = MagicMock()
    opts = PlanOptions()
    _execute_op(fs, session, op, "DB", "SCH", "WH0", opts)

    fs.update_feature_view.assert_called_once()
    _, kwargs = fs.update_feature_view.call_args
    assert kwargs["online_config"].target_lag == "3600 seconds"


def test_update_fv_target_lag_string_still_wins_when_set() -> None:
    """When the payload still carries a literal ``target_lag`` string, it wins."""
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="BFV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload={
            "kind": "BatchFeatureView",
            "name": "BFV",
            "version": "V1",
            "refresh_freq": "1 minute",
            "online": True,
            "target_lag": "90 seconds",
            "target_lag_sec": 60,
        },
    )
    session = MagicMock()
    opts = PlanOptions()
    _execute_op(fs, session, op, "DB", "SCH", "WH0", opts)

    _, kwargs = fs.update_feature_view.call_args
    assert kwargs["online_config"].target_lag == "90 seconds"


# ---------------------------------------------------------------------------
# CREATE_FV path sibling: ``_build_feature_view`` must observe the same
# contract.  These tests stub the snowml-core ``FeatureView`` /
# ``OnlineConfig`` constructors so we can inspect the kwargs without
# invoking any registration code.
# ---------------------------------------------------------------------------


def _batch_create_payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "kind": "BatchFeatureView",
        "name": "BFV_CREATE",
        "version": "V1",
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "DB.SCH.RAW_T",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "online": True,
        "target_lag_sec": 60,
        "refresh_freq": "1 minute",
    }
    base.update(overrides)
    return base


def _drive_build_feature_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Run ``_build_feature_view`` with FeatureView/OnlineConfig stubbed; return captured kwargs."""
    from unittest.mock import patch

    from snowflake.ml.feature_store.decl.imperative_executor import _build_feature_view

    fv_calls: list[dict[str, Any]] = []
    online_calls: list[dict[str, Any]] = []

    def _fake_fv(**kwargs: Any) -> Any:
        fv_calls.append(kwargs)
        return type("_FakeFV", (), {**kwargs})()

    def _fake_online(**kwargs: Any) -> Any:
        online_calls.append(kwargs)
        return type("_FakeOnline", (), {**kwargs})()

    fs = MagicMock()
    fs.get_entity.return_value = MagicMock(name="USER", join_keys=["USER_ID"])
    fs._session = MagicMock()
    session = MagicMock()
    session.sql.return_value.collect.return_value = []

    with (
        patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=_fake_fv),
        patch("snowflake.ml.feature_store.feature_view.OnlineConfig", side_effect=_fake_online),
    ):
        _build_feature_view(payload, session, "DB", "SCH", "WH", fs=fs)

    assert fv_calls, "FeatureView was not constructed"
    return {"fv_kwargs": fv_calls[0], "online_kwargs": online_calls[0] if online_calls else None}


def test_build_feature_view_propagates_target_lag_sec_to_online_config() -> None:
    """CREATE path: payload[target_lag_sec]=60 → OnlineConfig(target_lag='60 seconds')."""
    captured = _drive_build_feature_view(_batch_create_payload(target_lag_sec=60))
    online_kwargs = captured["online_kwargs"]
    assert online_kwargs is not None, "OnlineConfig was not constructed when online=True"
    assert online_kwargs.get("enable") is True
    assert online_kwargs.get("target_lag") == "60 seconds", (
        f"expected OnlineConfig.target_lag='60 seconds' from target_lag_sec=60; "
        f"got {online_kwargs.get('target_lag')!r}"
    )


def test_build_feature_view_propagates_target_lag_sec_hour_value_to_online_config() -> None:
    """CREATE path: payload[target_lag_sec]=3600 → OnlineConfig(target_lag='3600 seconds')."""
    captured = _drive_build_feature_view(_batch_create_payload(target_lag_sec=3600, refresh_freq="1 hour"))
    online_kwargs = captured["online_kwargs"]
    assert online_kwargs is not None
    assert online_kwargs.get("target_lag") == "3600 seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
