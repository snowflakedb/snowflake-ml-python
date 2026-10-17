"""Unit tests for append-only BatchFeatureView authoring validation.

Append-only maps 1:1 to the imperative
``snowflake.ml.feature_store.FeatureView(append_only=True)`` snapshot-accumulation
feature.  The imperative core
(``feature_view_append_only_validation.validate_snapshot_config_for_register``)
requires ``refresh_mode='FULL'``, a CRON ``refresh_freq`` (with time zone), a
``timestamp_col``, a batch kind, no tiled aggregation, and rejects
``overwrite=True``.  These tests pin the declarative authoring-time mirror of
that contract on :class:`FeatureView` so operators get an actionable load-time
error (message contains ``"is not valid on"`` so ``loader._dict_to_spec``
re-raises it) rather than a late imperative crash at apply.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.spec_models import (
    FeatureView,
    _is_cron_refresh_freq,
)
from snowflake.ml.test_utils import pytest_driver


def _append_only_fv(**overrides: Any) -> FeatureView:
    """Construct an append-only BatchFeatureView, applying field overrides.

    Args:
        **overrides: Field values that replace the valid-baseline defaults.
            Passing a value of ``None`` drops the field entirely so the
            "missing required companion" branches can be exercised.

    Returns:
        The constructed FeatureView with validators run at construction.
    """
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "APPEND_BFV",
        "version": "V1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [{"name": "EVENTS", "source_type": "Batch"}],
        "timestamp_col": "EVENT_TS",
        "refresh_mode": "FULL",
        "refresh_freq": "0 0 * * * UTC",
        "append_only": True,
    }
    base.update(overrides)
    # A value of ``None`` means "omit the field" so the required-companion
    # branches can be exercised.
    base = {k: v for k, v in base.items() if v is not None}
    return FeatureView(**base)


# --------------------------------------------------------------------------
# CRON detector helper
# --------------------------------------------------------------------------


def test_is_cron_refresh_freq_recognizes_cron() -> None:
    assert _is_cron_refresh_freq("0 0 * * * UTC") is True
    assert _is_cron_refresh_freq("*/5 * * * * America/Los_Angeles") is True


def test_is_cron_refresh_freq_rejects_durations_and_sentinels() -> None:
    assert _is_cron_refresh_freq("1 minute") is False
    assert _is_cron_refresh_freq("5 minutes") is False
    assert _is_cron_refresh_freq("1h") is False
    assert _is_cron_refresh_freq("DOWNSTREAM") is False
    assert _is_cron_refresh_freq("downstream") is False
    assert _is_cron_refresh_freq(None) is False
    # ``_is_cron_refresh_freq`` now delegates to the core imperative classifier
    # (``feature_view_refresh_freq._is_cron_refresh_freq``) for exact parity.
    # Core treats an empty string as a (degenerate) CRON expression because
    # ``pytimeparse.timeparse("")`` returns ``None``; parity is the contract, so
    # the decl surface matches core rather than special-casing empty to False.
    assert _is_cron_refresh_freq("") is True


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------


def test_append_only_offline_valid() -> None:
    fv = _append_only_fv()
    assert fv.append_only is True
    assert fv.refresh_mode == "FULL"


def test_append_only_online_valid() -> None:
    fv = _append_only_fv(online=True, target_lag="1 minute")
    assert fv.append_only is True
    assert fv.online is True


def test_append_only_allows_passthrough_features() -> None:
    # Applied-state recovery (DESCRIBE SPECIFICATION / _serialize_batch_fv_spec)
    # enumerates a deployed append-only BFV's passthrough columns as 1:1
    # ``features`` (no ``window`` / ``function``).  Those are NOT tiling — the
    # imperative contract rejects ``is_tiled`` (windowed/rollup), not the mere
    # presence of a features list — so re-validating the recovered spec on the
    # plan path must NOT raise.
    fv = _append_only_fv(
        features=[
            {
                "source_column": {"name": "METRIC_VAL", "type": "FloatType"},
                "output_column": {"name": "METRIC_VAL", "type": "FloatType"},
            }
        ],
    )
    assert fv.append_only is True
    assert len(fv.features) == 1


def test_non_append_only_bfv_unaffected() -> None:
    fv = FeatureView(
        kind="BatchFeatureView",
        name="PLAIN_BFV",
        version="V1",
        online=False,
        entities=["USER_ID"],
        sources=[{"name": "EVENTS", "source_type": "Batch"}],
    )
    assert fv.append_only in (None, False)


# --------------------------------------------------------------------------
# Rejection paths — every message must contain "is not valid on"
# --------------------------------------------------------------------------


def _assert_append_only_rejected(**overrides: Any) -> str:
    with pytest.raises(ValidationError) as excinfo:
        _append_only_fv(**overrides)
    msg = str(excinfo.value)
    assert "is not valid on" in msg, msg
    return msg


def test_append_only_rejects_missing_refresh_mode() -> None:
    msg = _assert_append_only_rejected(refresh_mode=None)
    assert "refresh_mode" in msg


def test_append_only_rejects_incremental_refresh_mode() -> None:
    msg = _assert_append_only_rejected(refresh_mode="INCREMENTAL")
    assert "FULL" in msg


def test_append_only_rejects_duration_refresh_freq() -> None:
    msg = _assert_append_only_rejected(refresh_freq="1 minute")
    assert "CRON" in msg or "cron" in msg


def test_append_only_rejects_missing_refresh_freq() -> None:
    _assert_append_only_rejected(refresh_freq=None)


def test_append_only_rejects_missing_timestamp_col() -> None:
    msg = _assert_append_only_rejected(timestamp_col=None)
    assert "timestamp_col" in msg


def test_append_only_rejects_tiled_features() -> None:
    _assert_append_only_rejected(
        feature_granularity="1h",
        feature_aggregation_method="tiles",
        features=[
            {
                "source_column": {"name": "AMOUNT", "type": "FloatType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window": "1h",
            }
        ],
    )


def test_append_only_rejects_non_batch_kind() -> None:
    with pytest.raises(ValidationError) as excinfo:
        FeatureView(
            kind="StreamingFeatureView",
            name="APPEND_SFV",
            version="V1",
            online=True,
            entities=["USER_ID"],
            sources=[{"name": "EVENTS", "source_type": "Stream"}],
            timestamp_col="EVENT_TS",
            refresh_mode="FULL",
            refresh_freq="0 0 * * * UTC",
            feature_granularity="1h",
            features=[
                {  # type: ignore[list-item]
                    "source_column": {"name": "AMOUNT", "type": "FloatType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                    "function": "sum",
                    "window": "1h",
                }
            ],
            append_only=True,
        )
    assert "is not valid on" in str(excinfo.value)


def test_append_only_rejects_backfill_overwrite() -> None:
    _assert_append_only_rejected(backfill={"overwrite": True})


if __name__ == "__main__":
    pytest_driver.main()
