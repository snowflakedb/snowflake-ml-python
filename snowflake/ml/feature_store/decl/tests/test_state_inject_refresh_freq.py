# Copyright (c) 2024 Snowflake Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 3 RED: pin the refresh_freq injection helper contract.

After the ``refresh_freq -> refresh_freq`` rename, the applied-side
decoding (Mechanism #3 of the Phase E Canonicalization Invariant) writes
the deployed Dynamic Table refresh cadence onto the recovered
``AppliedObject.spec_payload`` under ``spec.refresh_freq`` (the renamed
authoring key) — not the legacy ``spec.refresh_freq``.

The helper is also kind-aware: streaming and realtime FVs do not carry
a cadence knob in the declarative surface (the spec validator rejects
``refresh_freq`` on those kinds), so the helper must be a no-op for those
kinds — otherwise a runtime-stamped ``target_lag_sec=0`` would round-trip
back into the YAML as ``refresh_freq: "0 seconds"`` and fail to load.

These tests fail today because:
* The helper is named ``_inject_fv_refresh_freq_from_list_row`` and does
  not exist (today the in-line write happens inside
  ``_build_offline_fv_object``).
* The current in-line write targets ``spec.refresh_freq``.

Phase 3 GREEN extracts the logic into the helper and renames the target
field to ``spec.refresh_freq``, also adding the streaming / realtime
skip.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.state import (
    _inject_fv_refresh_freq_from_list_row,
    fetch_applied_state,
)

_DB = "JKEW_DB"
_SCHEMA = "JKEW_SCHEMA"


def _bfv_oft_specification() -> dict[str, Any]:
    """Return a minimal online-BFV spec_payload from DESCRIBE … TYPE = SPECIFICATION.

    Returns:
        A spec_payload dict with ``kind == "BatchFeatureView"``.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_AMOUNTS_FG_DECL",
            "version": "V1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT", "type": "DoubleType"},
                }
            ],
            "target_lag_sec": 300,
        },
    }


def _streaming_oft_specification() -> dict[str, Any]:
    """Return a minimal streaming-FV spec_payload from DESCRIBE … TYPE = SPECIFICATION.

    Mirrors the runtime-stamped shape (``spec.target_lag_sec = 0``) the
    deployed payload always carries for streaming kinds.

    Returns:
        A spec_payload dict with ``kind == "StreamingFeatureView"``.
    """
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_CLICK_BACKFILL_DECL",
            "version": "V1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [
                {
                    "source_column": {"name": "EVENT", "type": "StringType"},
                    "output_column": {"name": "EVENT", "type": "StringType"},
                }
            ],
            "target_lag_sec": 0,
        },
    }


def _realtime_oft_specification() -> dict[str, Any]:
    """Return a minimal realtime-FV spec_payload.

    Returns:
        A spec_payload dict with ``kind == "RealtimeFeatureView"``.
    """
    return {
        "kind": "RealtimeFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_REALTIME_FV",
            "version": "V1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [],
            "target_lag_sec": 0,
        },
    }


def _list_fv_row(refresh_freq: str, *, kind: str = "BATCH") -> dict[str, Any]:
    """Return a ``feature_view_rows`` entry the OFT loop matches against.

    Args:
        refresh_freq: DT cadence string the row should expose.
        kind: FV kind tag carried by the row (BATCH / STREAMING / REALTIME).

    Returns:
        A list-FV row dict shaped to match the
        ``decl.imperative_executor.fetch_feature_view_rows`` contract.
    """
    return {
        "name": "USER_AMOUNTS_FG_DECL",
        "version": "V1",
        "database_name": _DB,
        "schema_name": _SCHEMA,
        "kind": kind,
        "entities": ["USER_ID"],
        "online_enabled": True,
        "target_lag": "0 seconds",
        "refresh_freq": refresh_freq,
        "warehouse": "",
        "cluster_by": "",
        "refresh_mode": "",
        "desc": "",
        "physical_dt_name": "USER_AMOUNTS_FG_DECL$V1",
    }


# ===========================================================================
# Helper-level tests — exercise the injection contract in isolation.
# ===========================================================================


class TestInjectRefreshFreqWritesSpecRefreshFreq:
    """``_inject_fv_refresh_freq_from_list_row`` writes
    ``spec.refresh_freq`` (the renamed authoring key) on a BFV
    spec_payload when the matching row carries a non-empty
    ``refresh_freq`` cell.

    The legacy contract wrote ``spec.refresh_freq``; after the rename
    this must target the new field.  ``spec.target_lag_sec`` continues
    to be set (the wire-form key the structural hash already
    understands), so the planner's ``_refresh_freq_drifted`` path can
    compare local vs. applied symmetrically.
    """

    def test_writes_refresh_freq_when_inner_lacks_one(self) -> None:
        spec_payload = _bfv_oft_specification()
        row = _list_fv_row("5 minutes")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert spec_payload["spec"]["refresh_freq"] == "5 minutes", (
            "Helper must write spec.refresh_freq from row.refresh_freq; "
            f"got refresh_freq={spec_payload['spec'].get('refresh_freq')!r}."
        )

    def test_does_not_write_legacy_batch_schedule(self) -> None:
        spec_payload = _bfv_oft_specification()
        row = _list_fv_row("5 minutes")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert "batch_schedule" not in spec_payload["spec"], (
            "Helper must not write the legacy spec.batch_schedule key; "
            "the rename is hard. Got "
            f"batch_schedule={spec_payload['spec'].get('batch_schedule')!r}."
        )

    def test_preserves_existing_refresh_freq(self) -> None:
        """An already-set ``spec.refresh_freq`` must not be clobbered.

        Mirrors the additive-write contract — a pre-enriched payload
        (e.g. one produced via ``_extract_spec_from_oft`` that already
        carries the field) keeps its existing value.
        """
        spec_payload = _bfv_oft_specification()
        spec_payload["spec"]["refresh_freq"] = "1 minute"
        row = _list_fv_row("5 minutes")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert spec_payload["spec"]["refresh_freq"] == "1 minute"

    def test_no_op_when_row_refresh_freq_is_empty(self) -> None:
        spec_payload = _bfv_oft_specification()
        row = _list_fv_row("")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert "refresh_freq" not in spec_payload["spec"]

    def test_no_op_when_inner_spec_missing(self) -> None:
        """Defensive guard for the legacy embedded-specification flatten path."""
        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "X",
            "version": "V1",
        }
        row = _list_fv_row("5 minutes")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert "refresh_freq" not in spec_payload
        assert "spec" not in spec_payload

    def test_no_op_when_row_has_no_refresh_freq_key(self) -> None:
        """Tolerate partial / legacy row shapes that omit the cell entirely."""
        spec_payload = _bfv_oft_specification()
        row = {"name": "USER_AMOUNTS_FG_DECL", "version": "V1"}

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert "refresh_freq" not in spec_payload["spec"]

    def test_skips_streaming_kind(self) -> None:
        """Streaming FV spec_payloads must NOT receive a refresh_freq.

        The declarative spec validator now rejects ``refresh_freq`` on
        streaming kinds; the recovery helper must therefore skip the
        injection so a runtime-stamped value cannot leak into a kind
        that the next ``snow feature plan`` would reject on load.
        """
        spec_payload = _streaming_oft_specification()
        row = _list_fv_row("5 minutes", kind="STREAMING")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert "refresh_freq" not in spec_payload["spec"], (
            "Streaming FV spec_payload must not receive refresh_freq — "
            "the spec validator rejects the field on this kind, so an "
            "exported value would fail to load. Got "
            f"refresh_freq={spec_payload['spec'].get('refresh_freq')!r}."
        )

    def test_skips_realtime_kind(self) -> None:
        spec_payload = _realtime_oft_specification()
        row = _list_fv_row("5 minutes", kind="REALTIME")

        _inject_fv_refresh_freq_from_list_row(spec_payload, row)

        assert "refresh_freq" not in spec_payload["spec"]


# ===========================================================================
# End-to-end fetch_applied_state integration tests
# ===========================================================================


class TestFetchAppliedStateInjectsRefreshFreqForBatchFv:
    """``fetch_applied_state`` plumbs ``refresh_freq`` onto an online-BFV
    spec_payload via the new helper.

    Catches a future regression where the helper is removed or the call
    site short-circuits; the recovered ``AppliedObject.spec_payload``
    must carry the deployed DT cadence under the renamed key.
    """

    def test_online_batch_fv_spec_payload_gets_refresh_freq_from_row(self) -> None:
        spec = _bfv_oft_specification()
        show_row = {
            "name": "USER_AMOUNTS_FG_DECL$V1$ONLINE",
            "database_name": _DB,
            "schema_name": _SCHEMA,
            "created_on": "2024-01-01 00:00:00",
        }
        list_fv_row = _list_fv_row("5 minutes")

        state = fetch_applied_state(
            [show_row],
            None,
            specification_map={show_row["name"]: copy.deepcopy(spec)},
            feature_view_rows=[list_fv_row],
            default_database=_DB,
            default_schema=_SCHEMA,
        )

        key = f"BatchFeatureView:{_DB}.{_SCHEMA}:USER_AMOUNTS_FG_DECL"
        applied = state.objects.get(key)
        assert applied is not None, (
            f"expected AppliedObject under key {key!r}; got " f"keys={sorted(state.objects.keys())!r}"
        )
        assert applied.spec_payload["spec"]["refresh_freq"] == "5 minutes", (
            "fetch_applied_state did not plumb refresh_freq onto the "
            "online-BFV spec_payload. Recovered "
            f"spec={applied.spec_payload.get('spec')!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
