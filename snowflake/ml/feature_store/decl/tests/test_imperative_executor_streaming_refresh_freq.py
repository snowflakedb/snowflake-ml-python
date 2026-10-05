"""Executor tests: tiled StreamingFeatureView forwards ``refresh_freq``.

A *tiled* ``StreamingFeatureView`` materialises its aggregate as an
offline Dynamic Table whose refresh cadence is ``refresh_freq``.  Both
the CREATE path (``_build_feature_view`` → imperative ``FeatureView``
constructor) and the operational UPDATE path (``_execute_update_feature_view``
→ ``FeatureStore.update_feature_view``) must forward the authored cadence
for a tiled streaming FV — and must NOT forward it for a *non-tiled*
streaming FV (which compiles to a zero-lag VIEW and rejects the field at
the validator).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.imperative_executor import (
    _build_feature_view,
    _execute_op,
)
from snowflake.ml.feature_store.decl.types import PlanOp, PlanOptions
from snowflake.ml.test_utils import pytest_driver


def _tiled_streaming_update_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "SFV",
        "version": "V1",
        "refresh_freq": "10 minutes",
        "feature_granularity_sec": 3600,
        "feature_aggregation_method": "tiles",
        "features": [
            {
                "source_column": {"name": "VALUE", "type": "FloatType"},
                "output_column": {"name": "VALUE_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window_sec": 3600,
            }
        ],
        "warehouse": "WH1",
        "description": "d",
    }
    base.update(overrides)
    return base


def test_update_fv_forwards_refresh_freq_for_tiled_streaming() -> None:
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="SFV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload=_tiled_streaming_update_payload(),
    )
    _execute_op(fs, MagicMock(), op, "DB", "SCH", "WH0", PlanOptions())
    fs.update_feature_view.assert_called_once()
    _, kwargs = fs.update_feature_view.call_args
    assert kwargs.get("refresh_freq") == "10 minutes", (
        "tiled streaming UPDATE_FV must forward refresh_freq to " f"update_feature_view; got kwargs={sorted(kwargs)}"
    )


def test_update_fv_omits_refresh_freq_for_non_tiled_streaming() -> None:
    fs = MagicMock()
    # A non-tiled streaming FV compiles to a zero-lag VIEW
    # (FeatureViewStatus.STATIC), which rejects both ``refresh_freq`` and
    # ``warehouse`` at ``FeatureStore.update_feature_view`` (error 2110).
    # The valid operational drift here is the ``description`` edit, so the
    # executor must still call update_feature_view — but with ``desc``
    # only, never the DT-only kwargs, even though the hand-built payload
    # smuggles them in.
    payload = {
        "kind": "StreamingFeatureView",
        "name": "SFV",
        "version": "V1",
        "refresh_freq": "10 minutes",
        "warehouse": "WH1",
        "description": "updated desc",
    }
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="SFV",
        depends_on=[],
        destructive=False,
        reason="test",
        payload=payload,
    )
    _execute_op(fs, MagicMock(), op, "DB", "SCH", "WH0", PlanOptions())
    fs.update_feature_view.assert_called_once()
    _, kwargs = fs.update_feature_view.call_args
    assert "refresh_freq" not in kwargs, (
        "non-tiled streaming UPDATE_FV must NOT forward refresh_freq "
        "(the validator rejects it); defence-in-depth against a "
        f"hand-built payload. got kwargs={sorted(kwargs)}"
    )
    assert "warehouse" not in kwargs, (
        "non-tiled streaming UPDATE_FV must NOT forward warehouse — the FV "
        "materialises as a STATIC VIEW which rejects it with error 2110. "
        f"got kwargs={sorted(kwargs)}"
    )


def _drive_build_feature_view(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run ``_build_feature_view`` for a streaming payload; return FV kwargs."""
    fv_calls: list[dict[str, Any]] = []

    def _fake_fv(**kwargs: Any) -> Any:
        fv_calls.append(kwargs)
        return MagicMock()

    fs = MagicMock()
    fs.get_entity.return_value = MagicMock(name="USER", join_keys=["USER_ID"])
    session = MagicMock()

    with (
        patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=_fake_fv),
        patch("snowflake.ml.feature_store.feature_view.OnlineConfig", side_effect=lambda **k: MagicMock()),
        patch("snowflake.ml.feature_store.feature_view.OnlineStoreType", MagicMock()),
        patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_stream_config",
            return_value=MagicMock(),
        ),
        patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_features",
            return_value=[MagicMock()],
        ),
    ):
        _build_feature_view(payload, session, "DB", "SCH", "WH", fs=fs)
    return fv_calls


def _tiled_streaming_create_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "SFV_CREATE",
        "version": "V1",
        "online": True,
        "entities": ["USER_ID"],
        "timestamp_col": "EVENT_TS",
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Stream",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "feature_granularity_sec": 3600,
        "feature_aggregation_method": "tiles",
        "features": [
            {
                "source_column": {"name": "VALUE", "type": "FloatType"},
                "output_column": {"name": "VALUE_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window_sec": 3600,
            }
        ],
        "refresh_freq": "5 minutes",
    }
    base.update(overrides)
    return base


def test_build_feature_view_forwards_refresh_freq_for_tiled_streaming() -> None:
    fv_calls = _drive_build_feature_view(_tiled_streaming_create_payload())
    assert fv_calls, "FeatureView constructor was not reached"
    assert fv_calls[0].get("refresh_freq") == "5 minutes", (
        "tiled streaming CREATE_FV must forward refresh_freq to the "
        f"imperative FeatureView constructor; got kwargs={sorted(fv_calls[0])}"
    )


def test_build_feature_view_omits_refresh_freq_for_non_tiled_streaming() -> None:
    # A non-tiled streaming FV compiles to a zero-lag VIEW, which has no
    # Dynamic Table to schedule.  The spec validator rejects
    # ``refresh_freq`` on this kind; ``_build_feature_view`` still drops
    # it as defence-in-depth against a hand-built payload that bypassed
    # the validator.  Leave the key in so this test fails if the tiled
    # check in ``_payload_forwards_refresh_freq`` is removed.
    payload = _tiled_streaming_create_payload()
    payload.pop("feature_granularity_sec")
    payload.pop("feature_aggregation_method")
    payload.pop("features")
    fv_calls = _drive_build_feature_view(payload)
    assert fv_calls, "FeatureView constructor was not reached"
    assert "refresh_freq" not in fv_calls[0], (
        "non-tiled streaming CREATE_FV must NOT forward refresh_freq "
        "(the validator rejects it); defence-in-depth against a "
        f"hand-built payload. got kwargs={sorted(fv_calls[0])}"
    )


if __name__ == "__main__":
    pytest_driver.main()
