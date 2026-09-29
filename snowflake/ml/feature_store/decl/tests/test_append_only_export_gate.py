"""Exporter gate for ``append_only`` on a recovered BatchFeatureView.

``spec_models.FeatureView._validate_append_only`` refuses ``append_only: true``
unless the spec also carries ``refresh_mode: FULL``, a CRON ``refresh_freq``, a
``timestamp_col``, a batch kind, and no tiled aggregation.  The rejection
message contains ``"is not valid on"``, so ``loader._dict_to_spec`` re-raises
and ``load_specs`` skips the whole file — which makes the next ``snow feature
plan`` see a deployed object with no local spec and propose a destructive op.

The exporter therefore must never emit ``append_only: true`` alongside a cadence
the loader would reject.  These tests pin ``exporter._build_full_fidelity_fv``:

* a full companion contract (CRON ``refresh_freq`` + ``refresh_mode: FULL`` +
  ``timestamp_col`` + non-tiled) emits the flag with no warning,
* any missing companion drops the flag and warns instead of writing an
  unloadable spec,
* the ``target_lag_sec`` -> ``"<n> seconds"`` cadence fallback is suppressed for
  an append-only BFV (a CRON cadence stamps no ``target_lag_sec``, so whatever
  sits there is the OFT staleness / ``"0 seconds"`` default, never the DT
  cadence),
* the same fallback still fires for a plain (non-append-only) BFV,
* every emitted doc loads through ``loader._dict_to_spec``, and
* the Python renderer mirrors the gated YAML doc (shared ``fv_doc``).
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl import python_codegen
from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.loader import _dict_to_spec
from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView
from snowflake.ml.test_utils import pytest_driver

_CRON = "0 0 * * * UTC"


def _append_only_full_spec(**inner_overrides: Any) -> dict[str, Any]:
    """Build a recovered append-only BatchFV SPECIFICATION-shaped document.

    Args:
        **inner_overrides: Inner ``spec`` keys to add or replace on top of the
            valid-baseline append-only payload.  Pass a value of ``None`` to
            drop a baseline key entirely (simulating a companion field that
            state recovery could not populate).

    Returns:
        A ``{kind, metadata, spec}`` dict shaped like a parsed
        ``DESCRIBE ... TYPE = SPECIFICATION`` payload for an append-only BFV.
    """
    inner: dict[str, Any] = {
        "ordered_entity_column_names": ["USER_ID"],
        "timestamp_field": "EVENT_TS",
        "refresh_mode": "FULL",
        "refresh_freq": _CRON,
        "append_only": True,
        "sources": [],
        "features": [],
    }
    for key, value in inner_overrides.items():
        if value is None:
            inner.pop(key, None)
        else:
            inner[key] = value
    return {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_AO", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": inner,
    }


def _build(full_spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Run ``_build_full_fidelity_fv`` and return ``(fv_doc, warnings)``."""
    warnings: list[str] = []
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_AO",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
        warnings=warnings,
    )
    return doc, warnings


def test_cron_contract_emits_flag_without_warning() -> None:
    """A full companion contract keeps ``append_only: true`` and warns nothing."""
    doc, warnings = _build(_append_only_full_spec())
    assert doc.get("append_only") is True
    assert doc.get("refresh_freq") == _CRON
    assert warnings == []
    assert isinstance(_dict_to_spec(doc), BatchFeatureView)


def test_missing_cadence_drops_flag_and_suppresses_seconds_fallback() -> None:
    """No ``refresh_freq`` + ``target_lag_sec`` present: no flag, no synthesised cadence."""
    doc, warnings = _build(_append_only_full_spec(refresh_freq=None, target_lag_sec=0))
    assert "append_only" not in doc
    # The ``"<n> seconds"`` fallback must be suppressed for append-only BFVs.
    assert "refresh_freq" not in doc
    assert len(warnings) == 1
    assert "BFV_AO" in warnings[0]
    assert isinstance(_dict_to_spec(doc), BatchFeatureView)


def test_duration_cadence_drops_flag_but_keeps_cadence() -> None:
    """A duration ``refresh_freq`` is not CRON: drop the flag, keep the cadence."""
    doc, warnings = _build(_append_only_full_spec(refresh_freq="5 minutes"))
    assert "append_only" not in doc
    assert doc.get("refresh_freq") == "5 minutes"
    assert len(warnings) == 1
    assert isinstance(_dict_to_spec(doc), BatchFeatureView)


def test_missing_refresh_mode_full_drops_flag() -> None:
    """Without ``refresh_mode: FULL`` the flag is dropped and warned."""
    doc, warnings = _build(_append_only_full_spec(refresh_mode=None))
    assert "append_only" not in doc
    assert len(warnings) == 1
    assert isinstance(_dict_to_spec(doc), BatchFeatureView)


def test_missing_timestamp_col_drops_flag() -> None:
    """Without ``timestamp_col`` the flag is dropped and warned."""
    doc, warnings = _build(_append_only_full_spec(timestamp_field=None))
    assert "append_only" not in doc
    assert len(warnings) == 1
    assert isinstance(_dict_to_spec(doc), BatchFeatureView)


def test_tiled_append_only_drops_flag() -> None:
    """A windowed/aggregated feature makes the BFV tiled: drop the flag."""
    doc, warnings = _build(
        _append_only_full_spec(
            features=[{"source_column": {"name": "AMOUNT", "type": "FloatType"}, "function": "sum"}],
        )
    )
    assert "append_only" not in doc
    assert len(warnings) == 1
    assert isinstance(_dict_to_spec(doc), BatchFeatureView)


def test_plain_bfv_still_gets_seconds_fallback() -> None:
    """A non-append-only BFV keeps the ``target_lag_sec`` -> ``"<n> seconds"`` fallback."""
    full_spec = _append_only_full_spec(append_only=None, refresh_freq=None, target_lag_sec=600)
    doc, warnings = _build(full_spec)
    assert "append_only" not in doc
    assert doc.get("refresh_freq") == "600 seconds"
    assert warnings == []


def test_python_renderer_mirrors_gated_doc() -> None:
    """The Python form emits ``append_only=True`` only when the YAML doc does."""
    honored_doc, _ = _build(_append_only_full_spec())
    gated_doc, _ = _build(_append_only_full_spec(refresh_freq="5 minutes"))

    honored_src = python_codegen.spec_dict_to_python_source("BatchFeatureView", "bfv_ao", honored_doc)
    gated_src = python_codegen.spec_dict_to_python_source("BatchFeatureView", "bfv_ao", gated_doc)

    assert "append_only=True" in honored_src
    assert "append_only=True" not in gated_src


if __name__ == "__main__":
    pytest_driver.main()
