"""Parity coverage for the declarative CRON ``refresh_freq`` classifier.

``spec_models._is_cron_refresh_freq`` shadows the core imperative classifier
``feature_view_refresh_freq._is_cron_refresh_freq`` — a validator whose whole
justification is "surface an actionable load-time error instead of a late
imperative crash at apply" must agree with the thing it shadows or it produces
both false accepts and false rejects.

The original decl helper was built on ``interval_utils.interval_to_seconds``,
whose grammar is strictly narrower than ``pytimeparse`` (no ``weeks``, no
fractions, plus a ``"lifetime"`` sentinel that the two parsers disagree on), so
it diverged from core for ``"2 weeks"`` / ``"1.5h"`` (false accepts) and
``"lifetime"`` (false reject).  These tests pin exact classifier parity and the
downstream append-only validator behaviour, plus the compiler decoupling that
keeps the narrower ``interval_utils`` duration parsers from crashing on a
``pytimeparse`` duration they cannot parse.
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store import feature_view_refresh_freq
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    FeatureView,
    _is_cron_refresh_freq,
)
from snowflake.ml.test_utils import pytest_driver


@pytest.mark.parametrize(
    "freq",
    [
        "300",
        "2 weeks",
        "1.5h",
        "lifetime",
        "5 minutes",
        "1h",
        "DOWNSTREAM",
        "0 0 * * * UTC",
        "*/2 * * * * UTC",
    ],
)
def test_cron_classification_matches_core(freq: str) -> None:
    """decl ``_is_cron_refresh_freq`` must classify identically to core."""
    assert _is_cron_refresh_freq(freq) == feature_view_refresh_freq._is_cron_refresh_freq(freq)


def _append_only_authoring(**overrides: Any) -> dict[str, Any]:
    """Build a valid append-only BatchFeatureView authoring dict."""
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_AO",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "timestamp_col": "EVENT_TS",
        "refresh_mode": "FULL",
        "refresh_freq": "0 0 * * * UTC",
        "append_only": True,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("freq", ["1.5h", "2 weeks"])
def test_append_only_rejects_core_duration_refresh_freq(freq: str) -> None:
    """A duration ``refresh_freq`` (per core) must be rejected on append-only.

    Regression: the ``interval_utils``-based decl helper misclassified these as
    CRON and false-accepted them, letting apply reach the imperative preflight
    which then rejects them ("requires refresh_freq to be a cron expression").

    Args:
        freq: A duration ``refresh_freq`` that ``pytimeparse`` accepts.
    """
    with pytest.raises(Exception, match="is not valid on"):
        FeatureView.model_validate(_append_only_authoring(refresh_freq=freq))


def test_append_only_accepts_cron_refresh_freq() -> None:
    """A genuine CRON ``refresh_freq`` still validates on append-only."""
    fv = FeatureView.model_validate(_append_only_authoring(refresh_freq="0 0 * * * UTC"))
    assert fv.append_only is True


def test_compile_does_not_crash_on_non_interval_duration_refresh_freq() -> None:
    """A ``pytimeparse`` duration ``interval_utils`` cannot parse must not crash compile.

    A non-append-only BatchFeatureView with ``refresh_freq="2 weeks"`` reaches
    ``compile_to_spec``.  Because ``interval_utils`` has no ``weeks`` unit, the
    compiler must treat it as non-interval and skip ``target_lag_sec`` rather
    than crash ``parse_duration_to_seconds`` with ``Invalid interval format``.
    """
    authoring: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_DUR",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "2 weeks",
    }
    compiled = compile_to_spec(authoring, "DB1", "SC1")
    assert "target_lag_sec" not in compiled.get("spec", {})


if __name__ == "__main__":
    pytest_driver.main()
